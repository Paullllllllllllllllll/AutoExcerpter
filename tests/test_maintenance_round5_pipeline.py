"""Round 5: pipeline maintenance regressions (log durability + item verdict).

One focused regression per verified fix:

1. ``_process_single_page`` appends the PAID transcription entry to the working
   log BEFORE the summary call. A crash during the (paid) summary call used to
   discard an already-paid transcription; it now lands in the supported
   summary-only resume state.
2. That crash state resumes: the resume checker reports a non-COMPLETE item
   with the page marked completed, and the reload path regenerates only the
   missing summary (no second transcription call).
3. The critical-error handler no longer appends a SECOND entry for an index
   whose transcription entry is already logged (a duplicate would sit in
   ``completed_page_indices`` and trip ``log_has_failures`` at once, and
   ``_eligible_prior_entries`` would replay the page forever). The failure
   signal moves to the summary entry instead.
4. ``_close_log_handle`` closes under the per-handle lock, and a write against
   a handle finalized between acquisition and use falls back to the one-shot
   append instead of dropping the entry.
5. ``_eligible_prior_entries`` collapses duplicate indices, preferring the
   error-free entry.
6. A failed hot-path working-log append is counted and turns the item verdict
   in ``_run_processing`` to False.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import pipeline.log as log_mod
import pipeline.transcriber as transcriber_mod
from config import app as app_config
from config.constants import LOG_FORMAT_VERSION
from llm.token_tracker import DailyTokenTracker
from pipeline.paths import create_safe_directory_name, create_safe_log_filename
from pipeline.resume import ProcessingState, ResumeChecker
from pipeline.transcriber import ItemTranscriber


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _release_cached_log_handles() -> Generator[None]:
    """Close every cached log handle after each test.

    ``pipeline.log`` caches one open handle per path for the process lifetime;
    a leaked descriptor blocks Windows tmp_path cleanup and leaks state between
    tests.
    """
    yield
    with log_mod._LOG_HANDLES_GUARD:
        paths = list(log_mod._LOG_HANDLES)
    for path in paths:
        log_mod.finalize_log_file(path)
    with log_mod._LOG_HANDLES_GUARD:
        for path in paths:
            log_mod._FINALIZED_LOGS.discard(path)


def _entries(path: Path) -> list[dict[str, Any]]:
    """Return every parsed JSONL object in *path* (header included)."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _page_entries(path: Path, index: int) -> list[dict[str, Any]]:
    """Return the per-page entries in *path* carrying *index*."""
    return [e for e in _entries(path) if e.get("original_input_order_index") == index]


def _bare(tmp_path: Path) -> ItemTranscriber:
    """Build an ItemTranscriber with only the page-path attributes wired up."""
    obj = ItemTranscriber.__new__(ItemTranscriber)
    obj.name = "doc"
    obj._budget_exhausted = threading.Event()
    obj._count_lock = threading.Lock()
    obj._log_append_failures = 0
    obj._token_tracker = DailyTokenTracker(
        daily_limit=1, enabled=False, state_file=tmp_path / "budget.json"
    )
    obj._transcription_stamp = {}
    obj._summary_stamp = {}
    obj.transcription_times = []
    obj.transcription_provider = "openai"
    obj.start_time_processing = time.time()
    obj.log_path = tmp_path / "doc_transcription.jsonl"
    obj.summary_log_path = tmp_path / "doc_summary.jsonl"
    obj.summary_manager = None
    obj.transcribe_manager = MagicMock()
    obj.transcribe_manager.transcribe_payload.return_value = {
        "image": "page_0001.jpg",
        "transcription": "Hello world",
        "processing_time": 0.01,
        "provider": "openai",
    }
    return obj


class _Source:
    """Minimal payload source: one renderable page, no I/O."""

    def image_name(self, idx: int) -> str:
        return f"page_{idx:04d}.jpg"

    def build_payload(self, idx: int) -> Any:
        return SimpleNamespace(source_file="doc.pdf", provenance={}, page_index=idx)


def _summary_payload() -> dict[str, Any]:
    return {
        "page": 1,
        "page_information": {
            "page_number_integer": 1,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": ["a point"],
        "references": [],
    }


# ---------------------------------------------------------------------------
# Fix 1: the paid transcription is on disk before the summary call
# ---------------------------------------------------------------------------
class TestTranscriptionLoggedBeforeSummary:
    """The summary call must never be the thing that risks a paid page."""

    def test_entry_is_on_disk_when_generate_summary_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare(tmp_path)
        observed: dict[str, Any] = {}

        def _generate_summary(text: str, page_num: int) -> dict[str, Any]:
            observed["logged_before_call"] = _page_entries(obj.log_path, 0)
            return _summary_payload()

        manager = MagicMock()
        manager.generate_summary.side_effect = _generate_summary
        obj.summary_manager = manager
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        obj._process_single_page(0, _Source(), t_results, s_results, 1, [0])  # type: ignore[arg-type]

        # The paid transcription was already persisted when the summary call ran.
        assert len(observed["logged_before_call"]) == 1
        assert "error" not in observed["logged_before_call"][0]
        # And exactly one entry per log survives the page, with no duplicates.
        assert len(_page_entries(obj.log_path, 0)) == 1
        assert len(_page_entries(obj.summary_log_path, 0)) == 1
        assert len(t_results) == 1
        assert len(s_results) == 1
        assert obj._log_append_failures == 0


# ---------------------------------------------------------------------------
# Fix 2: a crash mid-summary resumes as a summary-only repair
# ---------------------------------------------------------------------------
def _write_crash_state(tmp_path: Path, item_name: str = "doc") -> tuple[Path, Path]:
    """Write the on-disk state left by a crash during the summary call.

    Transcription log: header + one error-free page entry. Summary log: header
    only (the page's summary never made it).
    """
    out_dir = tmp_path / "out"
    working = out_dir / create_safe_directory_name(item_name, "_working_files")
    working.mkdir(parents=True)
    t_log = working / create_safe_log_filename(item_name, "transcription")
    s_log = working / create_safe_log_filename(item_name, "summary")

    def _header(log_type: str) -> dict[str, Any]:
        return {
            "_format_version": LOG_FORMAT_VERSION,
            "log_type": log_type,
            "input_item_name": item_name,
            "total_images": 1,
            "model_name": "test-model",
        }

    page = {
        "image": "page_0000.jpg",
        "transcription": "Hello world",
        "processing_time": 0.01,
        "provider": "openai",
        "original_input_order_index": 0,
    }
    t_log.write_text(
        json.dumps(_header("transcription"), ensure_ascii=False)
        + "\n"
        + json.dumps(page, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    s_log.write_text(
        json.dumps(_header("summary"), ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return t_log, s_log


class TestCrashMidSummaryResumes:
    """The logged transcription is reused; only the summary is re-bought."""

    def test_resume_checker_reports_the_page_completed(self, tmp_path: Path) -> None:
        _write_crash_state(tmp_path)

        result = ResumeChecker(resume_mode="skip", summarize=True)._check_item(
            "doc", tmp_path / "out"
        )

        assert result.state is not ProcessingState.COMPLETE
        assert result.completed_page_indices == {0}

    def test_reload_regenerates_only_the_missing_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t_log, s_log = _write_crash_state(tmp_path)
        obj = _bare(tmp_path)
        obj.log_path = t_log
        obj.summary_log_path = s_log
        obj.completed_page_indices = {0}
        obj.total_items_to_transcribe = 1
        obj._prior_transcription_results = [
            e for e in _entries(t_log) if "original_input_order_index" in e
        ]
        obj._prior_summary_results = []
        manager = MagicMock()
        manager.generate_summary.return_value = _summary_payload()
        obj.summary_manager = manager
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        deferred = obj._reload_completed_pages(t_results, s_results)

        assert deferred == []
        # The summary was regenerated from the logged transcription ...
        assert manager.generate_summary.call_count == 1
        # ... and no page was re-transcribed (the expensive half is reused).
        obj.transcribe_manager.transcribe_payload.assert_not_called()  # type: ignore[attr-defined]
        assert len(t_results) == 1
        assert len(s_results) == 1
        assert len(_page_entries(s_log, 0)) == 1


# ---------------------------------------------------------------------------
# Fix 3: no second log entry for an index already logged
# ---------------------------------------------------------------------------
class TestCriticalErrorAfterTranscriptionLogged:
    """A crash after the transcription append must not duplicate the index."""

    def test_single_error_free_transcription_entry_and_errored_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare(tmp_path)
        obj.summary_manager = MagicMock()
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        real_summarize = obj._summarize_transcription

        def flaky(
            transcription_result: dict[str, Any], idx: int, img_name: str
        ) -> dict[str, Any] | None:
            if "error" not in transcription_result:
                raise RuntimeError("summary API exploded")
            # The placeholder path (error_result) still runs for real.
            return real_summarize(transcription_result, idx, img_name)

        obj._summarize_transcription = flaky  # type: ignore[method-assign, assignment]

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        obj._process_single_page(0, _Source(), t_results, s_results, 1, [0])  # type: ignore[arg-type]

        logged = _page_entries(obj.log_path, 0)
        assert len(logged) == 1
        assert "error" not in logged[0]
        assert logged[0]["transcription"].strip() == "Hello world"

        summaries = _page_entries(obj.summary_log_path, 0)
        assert len(summaries) == 1
        # AE-2: the failure signal lives on the summary entry, so the item is
        # not classified COMPLETE and the summary is regenerated on resume.
        assert "summary API exploded" in summaries[0]["error"]

        # In memory the page mirrors its on-disk state: transcribed, not failed.
        assert len(t_results) == 1
        assert "error" not in t_results[0]

    def test_crash_before_the_append_still_logs_the_error_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: the pre-existing behavior is untouched when nothing is logged."""
        obj = _bare(tmp_path)
        obj.transcribe_manager.transcribe_payload.side_effect = (  # type: ignore[attr-defined]
            RuntimeError("boom")
        )
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)

        t_results: list[dict[str, Any]] = []
        obj._process_single_page(0, _Source(), t_results, [], 1, [0])  # type: ignore[arg-type]

        logged = _page_entries(obj.log_path, 0)
        assert len(logged) == 1
        assert "boom" in logged[0]["error"]
        assert len(t_results) == 1
        assert "error" in t_results[0]


# ---------------------------------------------------------------------------
# Fix 4: finalize-vs-append race never silently loses an entry
# ---------------------------------------------------------------------------
def _drop(path: Path) -> None:
    log_mod.finalize_log_file(path)
    with log_mod._LOG_HANDLES_GUARD:
        log_mod._FINALIZED_LOGS.discard(path)


class TestFinalizeAppendRace:
    """A page entry must reach disk whichever side of a finalize it lands on."""

    def test_write_against_a_closed_cached_handle_falls_back(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "race.jsonl"
        assert log_mod.append_to_log(path, {"page": 0}) is True
        # Simulate a finalize landing between handle acquisition and the write:
        # the cached handle is closed but still cached, so the write raises
        # ValueError ("I/O operation on closed file").
        handle, _lock = log_mod._LOG_HANDLES[path]
        handle.close()

        assert log_mod.append_to_log(path, {"page": 1}) is True

        assert [e["page"] for e in _entries(path)] == [0, 1]
        _drop(path)

    def test_append_after_finalize_uses_the_one_shot_path(self, tmp_path: Path) -> None:
        path = tmp_path / "finalized.jsonl"
        assert log_mod.append_to_log(path, {"page": 0}) is True
        log_mod.finalize_log_file(path)

        assert log_mod.append_to_log(path, {"page": 1}) is True

        assert [e["page"] for e in _entries(path)] == [0, 1]
        # The one-shot path must not re-cache (and leak) a handle.
        assert path not in log_mod._LOG_HANDLES
        _drop(path)

    def test_close_waits_for_the_per_handle_lock(self, tmp_path: Path) -> None:
        path = tmp_path / "locked.jsonl"
        assert log_mod.append_to_log(path, {"page": 0}) is True
        handle, lock = log_mod._LOG_HANDLES[path]

        lock.acquire()
        finalizer = threading.Thread(target=log_mod.finalize_log_file, args=(path,))
        finalizer.start()
        try:
            # The close cannot proceed while a "write" holds the per-handle lock.
            finalizer.join(0.3)
            assert finalizer.is_alive()
            assert handle.closed is False
        finally:
            lock.release()
        finalizer.join(5)
        assert finalizer.is_alive() is False
        assert handle.closed is True
        _drop(path)

    def test_concurrent_append_and_finalize_lose_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "concurrent.jsonl"
        rounds = 40
        for i in range(rounds):
            appender = threading.Thread(
                target=log_mod.append_to_log, args=(path, {"page": i})
            )
            finalizer = threading.Thread(target=log_mod.finalize_log_file, args=(path,))
            appender.start()
            finalizer.start()
            appender.join()
            finalizer.join()
            # Keep the log live so the next round races again.
            with log_mod._LOG_HANDLES_GUARD:
                log_mod._FINALIZED_LOGS.discard(path)

        assert sorted(e["page"] for e in _entries(path)) == list(range(rounds))
        _drop(path)


# ---------------------------------------------------------------------------
# Fix 5: duplicate indices collapse to the error-free entry
# ---------------------------------------------------------------------------
class TestEligiblePriorEntriesDedup:
    """A duplicated index must be replayed once, and as the good entry."""

    @staticmethod
    def _obj(prior: list[dict[str, Any]]) -> ItemTranscriber:
        obj = ItemTranscriber.__new__(ItemTranscriber)
        obj.completed_page_indices = {0}
        obj.total_items_to_transcribe = 2
        obj._prior_transcription_results = prior
        return obj

    def test_error_entry_first_is_dropped(self) -> None:
        bad = {"original_input_order_index": 0, "error": "api", "transcription": "x"}
        good = {"original_input_order_index": 0, "transcription": "real text"}

        eligible = self._obj([bad, good])._eligible_prior_entries()

        assert eligible == [good]

    def test_error_entry_last_is_dropped(self) -> None:
        good = {"original_input_order_index": 0, "transcription": "real text"}
        bad = {"original_input_order_index": 0, "error": "api", "transcription": "x"}

        eligible = self._obj([good, bad])._eligible_prior_entries()

        assert eligible == [good]

    def test_distinct_indices_are_all_kept_in_index_order(self) -> None:
        one = {"original_input_order_index": 1, "transcription": "b"}
        zero = {"original_input_order_index": 0, "transcription": "a"}
        obj = self._obj([one, zero])
        obj.completed_page_indices = {0, 1}

        assert obj._eligible_prior_entries() == [zero, one]


# ---------------------------------------------------------------------------
# Fix 6: a failed hot-path append is counted and fails the item
# ---------------------------------------------------------------------------
class _FakeSource:
    def __init__(self, pages: int) -> None:
        self.pages = pages
        self.closed = False

    def __len__(self) -> int:
        return self.pages

    def file_provenance(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        self.closed = True


class TestLogAppendFailureAccounting:
    """An entry that never reached disk must not be reported as complete."""

    def test_failed_append_increments_the_counter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare(tmp_path)
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber_mod, "append_to_log", lambda *a, **k: False, raising=True
        )

        obj._process_single_page(0, _Source(), [], [], 1, [0])  # type: ignore[arg-type]

        assert obj._log_append_failures == 1

    def test_failed_append_turns_the_item_verdict_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        obj = _bare(tmp_path)
        obj.input_path = tmp_path / "doc.pdf"
        obj.output_txt_path = out_dir / "doc.txt"
        obj.completed_page_indices = set()
        obj.written_outputs = []
        obj._output_metadata_notes = []
        obj._budget_wait_cancelled = False
        obj.transcription_model = "test-model"
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            obj,
            "_transcribe_and_summarize",
            lambda source: ([{"original_input_order_index": 0}], []),
        )

        # Control: nothing failed -> the item is complete.
        assert obj._run_processing(_FakeSource(1), "PDF") is True  # type: ignore[arg-type]

        obj.written_outputs = []
        obj._log_append_failures = 2

        assert obj._run_processing(_FakeSource(1), "PDF") is False  # type: ignore[arg-type]
        assert obj.last_run_report is not None
        assert obj.last_run_report["pages_failed"] == 0
