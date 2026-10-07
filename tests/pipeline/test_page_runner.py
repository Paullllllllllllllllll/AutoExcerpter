"""One page through ``PageRunner.run``: its working-log entries and failures.

The transcription entry is on disk before the summary call, each index is
logged once per log, a failed append is counted, and a failed page reports
its error type.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import autoexcerpter.pipeline.log as log_mod
import autoexcerpter.pipeline.pages as pages_mod
from tests.pipeline.helpers import (
    clean_page_runner,
    content_summary,
    make_progress,
    make_runner,
    page_entries,
    summarizer,
    transcriber,
)


@pytest.fixture(autouse=True)
def _release_cached_log_handles() -> Generator[None]:
    """Close every cached log handle after each test.

    A leaked descriptor blocks Windows tmp_path cleanup.
    """
    yield
    log_mod.release_logs()


class _Source:
    """Minimal payload source: one renderable page, no I/O."""

    def image_name(self, idx: int) -> str:
        return f"page_{idx:04d}.jpg"

    def build_payload(self, idx: int) -> Any:
        return SimpleNamespace(source_file="doc.pdf", provenance={}, page_index=idx)


class TestTranscriptionLoggedBeforeSummary:
    """The transcription entry is persisted before the summary call runs."""

    def test_entry_is_on_disk_when_generate_summary_runs(self, tmp_path: Path) -> None:
        observed: dict[str, Any] = {}
        manager = summarizer()
        runner = clean_page_runner(tmp_path, summary_manager=manager)
        log_path = runner.paths.transcription_log

        def _generate_summary(text: str, page_num: int) -> dict[str, Any]:
            observed["logged_before_call"] = page_entries(log_path, 0)
            return content_summary()

        manager.generate_summary.side_effect = _generate_summary

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        asyncio.run(runner.run(0, _Source(), t_results, s_results, make_progress()))

        # The transcription was already persisted when the summary call ran.
        assert len(observed["logged_before_call"]) == 1
        assert "error" not in observed["logged_before_call"][0]
        # Exactly one entry per log survives the page, with no duplicates.
        assert len(page_entries(log_path, 0)) == 1
        assert len(page_entries(runner.paths.summary_log, 0)) == 1
        assert len(t_results) == 1
        assert len(s_results) == 1
        assert runner.log_append_failures == 0


class TestCriticalErrorAfterTranscriptionLogged:
    """A crash after the transcription append does not duplicate the index."""

    def test_single_error_free_transcription_entry_and_errored_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = clean_page_runner(tmp_path, summary_manager=summarizer())
        real_summarize = runner.summarize

        async def flaky(
            transcription_result: dict[str, Any], idx: int, img_name: str
        ) -> dict[str, Any] | None:
            if "error" not in transcription_result:
                raise RuntimeError("summary API exploded")
            # The placeholder path (error_result) still runs for real.
            return await real_summarize(transcription_result, idx, img_name)

        monkeypatch.setattr(runner, "summarize", flaky)

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        asyncio.run(runner.run(0, _Source(), t_results, s_results, make_progress()))

        logged = page_entries(runner.paths.transcription_log, 0)
        assert len(logged) == 1
        assert "error" not in logged[0]
        assert logged[0]["transcription"].strip() == "Hello world"

        summaries = page_entries(runner.paths.summary_log, 0)
        assert len(summaries) == 1
        # The failure lives on the summary entry, so the item is not
        # classified COMPLETE and the summary is regenerated on resume.
        assert "summary API exploded" in summaries[0]["error"]

        # In memory the page mirrors its on-disk state: transcribed, not failed.
        assert len(t_results) == 1
        assert "error" not in t_results[0]

    def test_crash_before_the_append_logs_the_error_entry(self, tmp_path: Path) -> None:
        """With nothing logged yet, a crash logs one error entry for the page."""
        runner = clean_page_runner(tmp_path)
        runner.transcribe_manager.transcribe_payload.side_effect = (  # type: ignore[attr-defined]
            RuntimeError("boom")
        )

        t_results: list[dict[str, Any]] = []
        asyncio.run(runner.run(0, _Source(), t_results, [], make_progress()))

        logged = page_entries(runner.paths.transcription_log, 0)
        assert len(logged) == 1
        assert "boom" in logged[0]["error"]
        assert len(t_results) == 1
        assert "error" in t_results[0]


class TestPageFailureReporting:
    def test_failed_append_increments_the_counter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = clean_page_runner(tmp_path)
        monkeypatch.setattr(
            pages_mod, "append_to_log", lambda *a, **k: False, raising=True
        )

        asyncio.run(runner.run(0, _Source(), [], [], make_progress()))

        assert runner.log_append_failures == 1

    def test_failed_page_reports_error_type(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mgr = transcriber(
            return_value={
                "image": "page_0001.jpg",
                "transcription": "[transcription error]",
                "processing_time": 0.01,
                "error": "API error after retries",
                "error_type": "api_failure",
                "schema_retries": {"gpt-5": 2},
                "provider": "openai",
            }
        )
        runner = make_runner(tmp_path, transcribe_manager=mgr)

        class _Src:
            def image_name(self, idx: int) -> str:
                return "page_0001.jpg"

            def build_payload(self, idx: int) -> Any:
                return SimpleNamespace(source_file="f", provenance={}, page_index=None)

        monkeypatch.setattr(pages_mod, "append_to_log", lambda *a, **k: None)

        with caplog.at_level(logging.DEBUG, logger="autoexcerpter.pipeline.pages"):
            asyncio.run(runner.run(0, _Src(), [], [], make_progress(started_at=0.0)))

        text = caplog.text
        assert "FAILED (api_failure" in text
        assert "2 schema retries" in text
        assert "0 retries" not in text
