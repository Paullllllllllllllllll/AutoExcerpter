"""Tests for the page-level token-budget gate in pipeline/transcriber.py.

Covers the per-page admission gate in ``_process_single_page`` and the
drain/wait/resume loop in ``_transcribe_and_summarize``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import pipeline.transcriber as transcriber
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

        t_results: list[dict] = []
        s_results: list[dict] = []
        count = [0]

        result = obj._process_single_page(0, None, t_results, s_results, 5, count)

        # Deferred: no API call, no result, no counter movement, and the
        # exhausted flag is set so siblings stop admitting too.
        assert result is None
        assert t_results == []
        assert s_results == []
        assert count == [0]
        assert obj._budget_exhausted.is_set()


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
        monkeypatch.setattr(transcriber.config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        state = {"reset": False}

        # First pass defers pages >= 2 (budget exhausted); after a simulated
        # reset, the remaining pages all process.
        def fake_process(idx, source, t_results, s_results, total, count_ref):
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

        t_results, _s_results = obj._transcribe_and_summarize(_Source())

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

        monkeypatch.setattr(transcriber.config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        def fake_process(idx, source, t_results, s_results, total, count_ref):
            if idx >= 2:
                obj._budget_exhausted.set()
                return None
            t_results.append({"original_input_order_index": idx})
            count_ref[0] += 1
            return {"original_input_order_index": idx}

        obj._process_single_page = fake_process  # type: ignore[method-assign]
        # User cancels the wait: the run stops with only the processed pages.
        monkeypatch.setattr(transcriber, "wait_for_token_reset", lambda: False)

        t_results, _s_results = obj._transcribe_and_summarize(_Source())

        indices = sorted(r["original_input_order_index"] for r in t_results)
        # Only the pages that fit before exhaustion are present; the rest are
        # left for a later run (resume).
        assert indices == [0, 1]
