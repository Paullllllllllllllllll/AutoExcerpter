"""Tests for the page-level token-budget gate in pipeline/transcriber.py.

Covers the per-page admission gate in ``_process_single_page`` and the
drain/wait/resume loop in ``_transcribe_and_summarize``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

import pipeline.transcriber as transcriber
from config import app as app_config
from llm.token_tracker import DailyTokenTracker
from pipeline.transcriber import ItemTranscriber


def _bare_transcriber() -> ItemTranscriber:
    """Construct an ItemTranscriber without running __init__ (which wires up
    managers and paths). Only the attributes touched by the methods under test
    are set by each test."""
    return ItemTranscriber.__new__(ItemTranscriber)


class TestProcessSinglePageGate:
    def test_page_deferred_when_budget_cannot_fit(self, tmp_path: Path) -> None:
        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        # Seed estimate (1000) far exceeds the daily limit (10): the page can
        # never be reserved, so it must defer.
        obj._token_tracker = DailyTokenTracker(
            daily_limit=10,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=1000,
        )

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        count = [0]

        # None source is never dereferenced: the budget gate defers first.
        result = obj._process_single_page(
            0,
            None,  # type: ignore[arg-type]
            t_results,
            s_results,
            5,
            count,
        )

        # Deferred: no API call, no result, no counter movement, and the
        # exhausted flag is set so siblings stop admitting too.
        assert result is None
        assert t_results == []
        assert s_results == []
        assert count == [0]
        assert obj._budget_exhausted.is_set()

    def test_summary_bucket_gates_fresh_page(self, tmp_path: Path) -> None:
        """A pool-less transcription bucket must not bypass the summary key's
        pool cap: the fresh-page gate reserves against BOTH buckets when the
        summary stamp resolves to a different one."""
        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        # Combined budget is wide open; only the summary key's pool cap (10)
        # cannot fit the page estimate (1000).
        obj._token_tracker = DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=1000,
            pool_caps={"openai": {"large": 10}},
        )
        # Pool-less transcription endpoint (custom/local), pooled summary key.
        obj._transcription_stamp = {
            "provider": "custom",
            "key_env": "LOCAL_API_KEY",
            "model": "local-vision",
        }
        obj._summary_stamp = {
            "provider": "openai",
            "key_env": "OPENAI_API_KEY",
            "model": "gpt-5",
        }

        result = obj._process_single_page(
            0,
            None,  # type: ignore[arg-type]
            [],
            [],
            5,
            [0],
        )

        # Deferred by the summary bucket; the transcription-side reservation
        # was rolled back so no headroom leaks.
        assert result is None
        assert obj._budget_exhausted.is_set()
        assert obj._token_tracker._tokens_reserved == 0
        assert obj._token_tracker._bucket_reservations == {}


class TestTranscribeAndSummarizeResumeLoop:
    def test_resumes_pending_pages_after_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n = 6
        obj = _bare_transcriber()
        obj.completed_page_indices = set()
        obj._budget_exhausted = threading.Event()
        obj._token_tracker = DailyTokenTracker(
            daily_limit=10**9, enabled=True, state_file=tmp_path / "s.json"
        )

        class _Source:
            def __len__(self) -> int:
                return n

        # Skip the summary-log setup branch.
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        state = {"reset": False}

        # First pass defers pages >= 2 (budget exhausted); after a simulated
        # reset, the remaining pages all process.
        def fake_process(
            idx, source, t_results, s_results, total, count_ref, already_complete=0
        ):
            if not state["reset"] and idx >= 2:
                obj._budget_exhausted.set()
                return None
            t_results.append({"original_input_order_index": idx})
            count_ref[0] += 1
            return {"original_input_order_index": idx}

        def fake_wait() -> bool:
            state["reset"] = True
            return True

        obj._process_single_page = fake_process  # type: ignore[method-assign]
        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        t_results, _s_results = obj._transcribe_and_summarize(_Source())  # type: ignore[arg-type]

        indices = sorted(r["original_input_order_index"] for r in t_results)
        # Every page processed exactly once across the two passes.
        assert indices == list(range(n))
        assert len(indices) == len(set(indices))
        assert state["reset"] is True

    def test_stops_when_wait_cancelled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n = 6
        obj = _bare_transcriber()
        obj.completed_page_indices = set()
        obj._budget_exhausted = threading.Event()
        obj._token_tracker = DailyTokenTracker(
            daily_limit=10**9, enabled=True, state_file=tmp_path / "s.json"
        )

        class _Source:
            def __len__(self) -> int:
                return n

        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        def fake_process(
            idx, source, t_results, s_results, total, count_ref, already_complete=0
        ):
            if idx >= 2:
                obj._budget_exhausted.set()
                return None
            t_results.append({"original_input_order_index": idx})
            count_ref[0] += 1
            return {"original_input_order_index": idx}

        obj._process_single_page = fake_process  # type: ignore[method-assign]
        # User cancels the wait: the run stops with only the processed pages.
        monkeypatch.setattr(transcriber, "wait_for_token_reset", lambda: False)

        t_results, _s_results = obj._transcribe_and_summarize(_Source())  # type: ignore[arg-type]

        indices = sorted(r["original_input_order_index"] for r in t_results)
        # Only the pages that fit before exhaustion are present; the rest are
        # left for a later run (resume).
        assert indices == [0, 1]
        # The deferred pages appended nothing, so process_item's shortfall check
        # (total - len(results)) sees the gap: 2 present, 4 deferred out of n.
        assert len(t_results) == 2
        assert n - len(t_results) == 4
