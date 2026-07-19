"""Regression tests for round-4 pipeline hardening.

Each test targets one fix applied in this round to ``pipeline/transcriber.py``
(and one against the atomic header write in ``pipeline/log.py``):

- fix 1: budget-deferred summary-only pages re-enter the wait/retry loop
  (``_finish_deferred_summaries``) instead of being silently dropped: a pure
  summary-only resume waits for the daily reset and completes; a mixed resume
  finishes its deferred summaries after the fresh pages; a cancelled wait leaves
  the remaining pages for a later run; a page that can never fit gives up after
  two stalled resets.
- fix 2: pass 1b re-appends every reusable prior summary to the (truncated)
  summary log up front, before any slow regeneration, and pass 2 does not
  double-append a reused summary.
- fix 3: a blank page's zero-cost placeholder summary skips the budget gate, so
  it is not deferred even when the budget is exhausted.
- fix 4: a header-write failure leaves the prior log content intact (atomic
  temp-file + os.replace).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import pipeline.log as log_module
import pipeline.transcriber as transcriber
from config import app as app_config
from pipeline.transcriber import ItemTranscriber


def _bare_transcriber() -> ItemTranscriber:
    """Construct an ItemTranscriber without running __init__.

    Only the attributes each test touches are set explicitly.
    """
    return ItemTranscriber.__new__(ItemTranscriber)


class _FakeTracker:
    """Minimal token tracker: ``try_reserve`` fails while ``blocked``.

    Stands in for DailyTokenTracker so a test can flip the budget open/closed
    deterministically (e.g. a fake ``wait_for_token_reset`` unblocking it).
    """

    def __init__(self, blocked: bool = False) -> None:
        self.blocked = blocked
        self.reserve_calls = 0

    def try_reserve(self, **_kw: Any) -> int | None:
        self.reserve_calls += 1
        return None if self.blocked else 1

    def release(self, _amount: int, **_kw: Any) -> None:
        pass

    def record_page_usage(self) -> None:
        pass


def _make_summary_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.generate_summary.return_value = {
        "page": 1,
        "page_information": {"page_types": ["other"]},
        "bullet_points": ["point"],
        "references": [],
    }
    return mgr


def _entry(idx: int, text: str = "content") -> dict[str, Any]:
    return {
        "original_input_order_index": idx,
        "image": f"p{idx}",
        "transcription": text,
    }


# ---------------------------------------------------------------------------
# fix 1: _finish_deferred_summaries (direct)
# ---------------------------------------------------------------------------
class TestFinishDeferredSummaries:
    def test_completes_after_simulated_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        obj._budget_exhausted.set()  # budget starts exhausted
        obj._summary_stamp = {}
        obj._prior_summary_by_idx = {}
        obj.summary_log_path = tmp_path / "s.log"
        obj.summary_manager = _make_summary_manager()
        tracker = _FakeTracker(blocked=True)
        obj._token_tracker = tracker  # type: ignore[assignment]

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        wait_calls: list[dict[str, Any]] = []

        def fake_wait(**kw: Any) -> bool:
            wait_calls.append(kw)
            tracker.blocked = False  # the reset frees the budget
            return True

        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        deferred = [_entry(0), _entry(1)]
        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        obj._finish_deferred_summaries(deferred, t_results, s_results)

        assert len(wait_calls) == 1
        assert sorted(r["original_input_order_index"] for r in t_results) == [0, 1]
        assert len(s_results) == 2

    def test_cancelled_wait_leaves_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        obj._budget_exhausted.set()
        obj._summary_stamp = {}
        obj._prior_summary_by_idx = {}
        obj.summary_log_path = tmp_path / "s.log"
        obj.summary_manager = _make_summary_manager()
        obj._token_tracker = _FakeTracker(blocked=True)  # type: ignore[assignment]

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        wait_calls: list[dict[str, Any]] = []

        def fake_wait(**kw: Any) -> bool:
            wait_calls.append(kw)
            return False  # user cancels

        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        deferred = [_entry(0), _entry(1)]
        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        obj._finish_deferred_summaries(deferred, t_results, s_results)

        # Nothing processed: the pages are left for a later resume run.
        assert len(wait_calls) == 1
        assert t_results == []
        assert s_results == []

    def test_gives_up_after_two_stalled_resets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        obj._summary_stamp = {}
        obj._prior_summary_by_idx = {}
        obj.summary_log_path = tmp_path / "s.log"
        obj.summary_manager = _make_summary_manager()
        # The page can never fit: try_reserve always fails.
        obj._token_tracker = _FakeTracker(blocked=True)  # type: ignore[assignment]

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        wait_calls: list[dict[str, Any]] = []

        def fake_wait(**kw: Any) -> bool:
            wait_calls.append(kw)  # reset happens but budget stays too small
            return True

        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        deferred = [_entry(0)]
        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        obj._finish_deferred_summaries(deferred, t_results, s_results)

        # Two consecutive no-progress resets, then give up.
        assert len(wait_calls) == 2
        assert t_results == []


# ---------------------------------------------------------------------------
# fix 1: integration through _transcribe_and_summarize
# ---------------------------------------------------------------------------
class _Source:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


def _wire_resume_transcriber(
    obj: ItemTranscriber, tmp_path: Path, prior_trans: list[dict[str, Any]]
) -> None:
    obj.total_items_to_transcribe = 0  # set inside the method
    obj.log_path = tmp_path / "t.log"
    obj.summary_log_path = tmp_path / "s.log"
    obj._budget_exhausted = threading.Event()
    obj._summary_stamp = {}
    obj._transcription_stamp = {}
    obj.summary_manager = _make_summary_manager()
    obj.name = "doc"
    obj.input_path = tmp_path / "doc.pdf"
    obj.input_type = "pdf"
    obj.summary_model = "gpt-5-mini"
    obj._prior_transcription_results = prior_trans
    obj._prior_summary_results = []
    # Skip the real summary-log header init (config-coupled); irrelevant here.
    obj._initialize_log_or_raise = lambda *a, **k: None  # type: ignore[method-assign]


class TestTranscribeAndSummarizeDeferredFinish:
    def test_pure_summary_only_resume_finishes_deferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        _wire_resume_transcriber(obj, tmp_path, [_entry(0), _entry(1)])
        obj.completed_page_indices = {0, 1}  # every page already transcribed
        tracker = _FakeTracker(blocked=True)  # both summaries defer in reload
        obj._token_tracker = tracker  # type: ignore[assignment]

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        wait_calls: list[dict[str, Any]] = []

        def fake_wait(**kw: Any) -> bool:
            wait_calls.append(kw)
            tracker.blocked = False
            return True

        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        t_results, s_results = obj._transcribe_and_summarize(_Source(2))  # type: ignore[arg-type]

        # The early-return (no pending pages) branch still waited and finished.
        assert len(wait_calls) >= 1
        assert sorted(r["original_input_order_index"] for r in t_results) == [0, 1]
        assert len(s_results) == 2

    def test_mixed_resume_finishes_deferred_after_fresh_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        _wire_resume_transcriber(obj, tmp_path, [_entry(0)])
        obj.completed_page_indices = {0}  # page 0 completed, summary deferred
        tracker = _FakeTracker(blocked=True)
        obj._token_tracker = tracker  # type: ignore[assignment]

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        def fake_process(
            idx: int,
            source: Any,
            t: list[dict[str, Any]],
            s: list[dict[str, Any]],
            total: int,
            count_ref: list[int],
            already_complete: int = 0,
        ) -> dict[str, Any]:
            t.append({"original_input_order_index": idx})
            count_ref[0] += 1
            return {"original_input_order_index": idx}

        obj._process_single_page = (  # type: ignore[method-assign]
            fake_process  # type: ignore[assignment]
        )

        wait_calls: list[dict[str, Any]] = []

        def fake_wait(**kw: Any) -> bool:
            wait_calls.append(kw)
            tracker.blocked = False  # reset frees the summary budget
            return True

        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        t_results, s_results = obj._transcribe_and_summarize(_Source(3))  # type: ignore[arg-type]

        indices = sorted(r["original_input_order_index"] for r in t_results)
        # Fresh pages 1,2 plus the deferred completed page 0 all present.
        assert indices == [0, 1, 2]
        assert len(wait_calls) >= 1
        assert any(r.get("original_input_order_index") == 0 for r in s_results)

    def test_fresh_wait_cancel_skips_deferred_finish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        _wire_resume_transcriber(obj, tmp_path, [_entry(0)])
        obj.completed_page_indices = {0}  # page 0's summary defers in reload
        tracker = _FakeTracker(blocked=True)
        obj._token_tracker = tracker  # type: ignore[assignment]

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        def fake_process(
            idx: int,
            source: Any,
            t: list[dict[str, Any]],
            s: list[dict[str, Any]],
            total: int,
            count_ref: list[int],
            already_complete: int = 0,
        ) -> dict[str, Any] | None:
            if idx >= 2:
                obj._budget_exhausted.set()
                return None
            t.append({"original_input_order_index": idx})
            count_ref[0] += 1
            return {"original_input_order_index": idx}

        obj._process_single_page = (  # type: ignore[method-assign]
            fake_process  # type: ignore[assignment]
        )

        wait_calls: list[dict[str, Any]] = []

        def fake_wait(**kw: Any) -> bool:
            wait_calls.append(kw)
            return False  # user cancels the fresh-page wait

        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        t_results, _s_results = obj._transcribe_and_summarize(_Source(3))  # type: ignore[arg-type]

        indices = sorted(r["original_input_order_index"] for r in t_results)
        # Only the one fresh page that fit is present; the deferred completed
        # page 0 was NOT force-finished after a Ctrl+C, and page 2 stays pending.
        assert indices == [1]
        assert 0 not in indices
        assert 2 not in indices
        # Exactly one wait (the fresh-page one); no second deferred-summary wait.
        assert len(wait_calls) == 1


# ---------------------------------------------------------------------------
# fix 2: pass 1b persists reusable summaries up front; no double-append
# ---------------------------------------------------------------------------
class TestReloadPassOneB:
    def test_reusable_summary_persisted_before_regeneration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj.completed_page_indices = {0, 1}
        obj.total_items_to_transcribe = 2
        obj.log_path = tmp_path / "t.log"
        obj.summary_log_path = tmp_path / "s.log"
        obj._budget_exhausted = threading.Event()
        obj._summary_stamp = {}
        obj._token_tracker = _FakeTracker(blocked=False)  # type: ignore[assignment]
        obj.summary_manager = _make_summary_manager()
        obj._prior_transcription_results = [_entry(0), _entry(1)]
        # Page 0 has a reusable error-free prior summary; page 1 has none.
        obj._prior_summary_results = [
            {
                "original_input_order_index": 0,
                "page_information": {"page_types": ["other"]},
                "bullet_points": ["reused"],
                "references": [],
            }
        ]

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        events: list[tuple[str, Any, Any]] = []

        def fake_append(path: Path, entry: dict[str, Any]) -> None:
            events.append(("append", path, entry.get("original_input_order_index")))

        monkeypatch.setattr(transcriber, "append_to_log", fake_append)

        def fake_generate(text: str, page: Any) -> dict[str, Any]:
            events.append(("generate", None, None))
            return {
                "page": page,
                "page_information": {"page_types": ["other"]},
                "bullet_points": ["new"],
                "references": [],
            }

        obj.summary_manager.generate_summary.side_effect = fake_generate

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        deferred = obj._reload_completed_pages(t_results, s_results)

        assert deferred == []

        # The reusable summary for page 0 was appended to the summary log BEFORE
        # any (slow) regeneration call happened.
        first_generate = next(i for i, e in enumerate(events) if e[0] == "generate")
        reusable_append = next(
            i
            for i, e in enumerate(events)
            if e[0] == "append" and e[1] == obj.summary_log_path and e[2] == 0
        )
        assert reusable_append < first_generate

        # No double-append: page 0's reused summary hits the summary log once.
        page0_summary_appends = [
            e
            for e in events
            if e[0] == "append" and e[1] == obj.summary_log_path and e[2] == 0
        ]
        assert len(page0_summary_appends) == 1

        # Both pages admitted; page 1 regenerated exactly once.
        assert sorted(r["original_input_order_index"] for r in t_results) == [0, 1]
        assert len(s_results) == 2
        assert obj.summary_manager.generate_summary.call_count == 1


# ---------------------------------------------------------------------------
# fix 3: blank page's placeholder summary skips the budget gate
# ---------------------------------------------------------------------------
class TestBlankPageFreePath:
    def test_blank_page_summary_not_deferred_when_budget_exhausted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj.completed_page_indices = {0}
        obj.total_items_to_transcribe = 1
        obj.log_path = tmp_path / "t.log"
        obj.summary_log_path = tmp_path / "s.log"
        obj._budget_exhausted = threading.Event()
        obj._budget_exhausted.set()  # budget exhausted
        obj._summary_stamp = {}
        # A reservation would fail if attempted; the free path must not attempt.
        tracker = _FakeTracker(blocked=True)
        obj._token_tracker = tracker  # type: ignore[assignment]
        obj.summary_manager = _make_summary_manager()
        obj._prior_transcription_results = [
            _entry(0, text="[p0: no transcribable text]")
        ]
        obj._prior_summary_results = []

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        deferred = obj._reload_completed_pages(t_results, s_results)

        # Blank page completed without a reservation, despite the exhausted
        # budget, and no LLM summary call was made.
        assert deferred == []
        assert tracker.reserve_calls == 0
        assert obj.summary_manager.generate_summary.call_count == 0
        assert len(t_results) == 1
        assert len(s_results) == 1
        assert s_results[0]["page_information"]["page_types"] == ["blank"]


# ---------------------------------------------------------------------------
# fix 4: header-write failure preserves the prior log content
# ---------------------------------------------------------------------------
class TestInitializeLogAtomicWrite:
    def test_header_write_failure_preserves_prior_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = tmp_path / "trans.log"
        prior = '{"_format_version": 1, "completed": "page"}\n'
        log_path.write_text(prior, encoding="utf-8")

        def boom(_src: Any, _dst: Any) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr("pipeline.log.os.replace", boom)

        ok = log_module.initialize_log_file(
            log_path,
            "item",
            str(tmp_path / "in.pdf"),
            "PDF",
            3,
            "gpt-5-mini",
        )

        assert ok is False
        # The prior completed-page entries survive the failed re-init.
        assert log_path.read_text(encoding="utf-8") == prior
        # No stray temp remnant is left behind.
        assert list(tmp_path.glob("*.tmp")) == []
