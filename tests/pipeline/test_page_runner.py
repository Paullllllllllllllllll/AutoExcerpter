"""One page through ``PageRunner.run``: its units in the store and failures.

The transcription unit is committed before the summary call, each unit is
written once, a failed write leaves the page pending and is counted, and a
failed page reports its error type. The attempts of a unit's calls are
stored and attributed to that unit.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.llm.summary import SummaryManager
from autoexcerpter.llm.transcription import TranscriptionManager
from autoexcerpter.output.store import DATABASE_NAME, KIND_SUMMARY
from autoexcerpter.pipeline.units import DocumentHandle
from tests.llm import helpers as llm
from tests.pipeline.helpers import (
    clean_page_runner,
    content_summary,
    make_progress,
    make_runner,
    statuses,
    store_pages,
    stored_document,
    stored_units,
    summarizer,
    transcriber,
)


class _Source:
    """Minimal payload source: one renderable page, no I/O."""

    def image_name(self, idx: int) -> str:
        return f"page_{idx + 1:04d}.jpg"

    def build_payload(self, idx: int) -> Any:
        return SimpleNamespace(
            source_file="doc.pdf",
            provenance={},
            page_index=idx,
            image_name=self.image_name(idx),
            number=idx + 1,
            base64="AAAA",
            mime_type="image/jpeg",
        )


def _run_page(
    tmp_path: Path,
    summarize: bool,
    runner_for: Any,
    pages: int = 1,
) -> list[str]:
    """Plan *pages* pages, run page 0 through the runner; return the statuses."""

    async def main() -> list[str]:
        async with stored_document(tmp_path, summarize=summarize) as document:
            await store_pages(document, pages, {})
            runner = runner_for(document)
            return [await runner.run(0, _Source(), make_progress())]

    return asyncio.run(main())


def test_the_transcription_is_stored_when_the_summary_call_runs(
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}
    manager = summarizer()

    def _generate_summary(text: str, page_num: int) -> dict[str, Any]:
        observed["before_call"] = statuses(tmp_path)
        return content_summary()

    manager.generate_summary.side_effect = _generate_summary

    result = _run_page(tmp_path, True, lambda d: clean_page_runner(d, manager))

    assert result == ["ok"]
    assert observed["before_call"] == ["done"]
    (transcription,) = stored_units(tmp_path)
    assert transcription["content"] == "Hello world\n"
    assert transcription["attempts"] == 1
    (summary,) = stored_units(tmp_path, KIND_SUMMARY)
    assert (summary["status"], summary["outcome"]) == ("done", "ok")
    (row,) = read_rows(tmp_path / DATABASE_NAME, "summaries")
    assert row["bullet_points"] == '["a point"]'


def test_a_crash_in_the_summary_keeps_the_transcription_and_fails_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def runner_for(document: DocumentHandle) -> Any:
        runner = clean_page_runner(document, summarizer())

        async def exploding(*_args: Any) -> dict[str, Any] | None:
            raise RuntimeError("summary API exploded")

        monkeypatch.setattr(runner, "summarize", exploding)
        return runner

    assert _run_page(tmp_path, True, runner_for) == ["ok"]

    (transcription,) = stored_units(tmp_path)
    assert (transcription["status"], transcription["content"]) == (
        "done",
        "Hello world\n",
    )
    (summary,) = stored_units(tmp_path, KIND_SUMMARY)
    assert summary["status"] == "failed"
    assert "summary API exploded" in summary["error"]
    assert summary["error_type"] == "RuntimeError"


def test_a_crash_before_the_transcription_stores_a_failed_page(
    tmp_path: Path,
) -> None:
    def runner_for(document: DocumentHandle) -> Any:
        runner: Any = clean_page_runner(document)
        runner.transcribe_manager.transcribe_payload.side_effect = RuntimeError("boom")
        return runner

    assert _run_page(tmp_path, False, runner_for) == ["failed"]

    (unit,) = stored_units(tmp_path)
    assert unit["status"] == "failed"
    assert "boom" in unit["error"]
    assert unit["content"].startswith("[CRITICAL ERROR]")


def test_a_failed_unit_write_leaves_the_page_pending_and_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runners: list[Any] = []

    async def locked(_self: DocumentHandle, _result: Any) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(DocumentHandle, "write_transcription", locked)

    def runner_for(document: DocumentHandle) -> Any:
        runner = clean_page_runner(document)
        runners.append(runner)
        return runner

    assert _run_page(tmp_path, False, runner_for) == ["deferred"]

    assert len(runners[0].write_errors) == 1
    assert statuses(tmp_path) == ["pending"]


def test_a_failed_page_reports_its_error_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
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

    with caplog.at_level(logging.DEBUG, logger="autoexcerpter.pipeline.pages"):
        result = _run_page(
            tmp_path,
            False,
            lambda d: make_runner(tmp_path, transcribe_manager=mgr, document=d),
        )

    assert result == ["failed"]
    text = caplog.text
    assert "FAILED (api_failure" in text
    assert "2 schema retries" in text
    assert "0 retries" not in text
    (unit,) = stored_units(tmp_path)
    assert (unit["error"], unit["error_type"]) == (
        "API error after retries",
        "api_failure",
    )


@pytest.mark.usefixtures("mock_image_processing")
def test_each_units_attempts_are_stored_for_that_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_sleep = asyncio.sleep

    async def no_wait(_delay: float, result: Any = None) -> Any:
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    llm.install(
        monkeypatch,
        [
            llm.StatusError(500),
            llm.TRANSCRIPTION_ANSWER,
            llm.SUMMARY_ANSWER,
        ],
    )
    env = llm.env()

    def runner_for(document: DocumentHandle) -> Any:
        return make_runner(
            tmp_path,
            transcribe_manager=TranscriptionManager(llm.phase(), env),
            summary_manager=SummaryManager(llm.phase(), env),
            document=document,
        )

    assert _run_page(tmp_path, True, runner_for) == ["ok"]

    responses = read_rows(tmp_path / DATABASE_NAME, "responses")
    assert [(row["kind"], row["idx"], row["role"]) for row in responses] == [
        ("transcription", 0, "transcription"),
        ("transcription", 0, "transcription"),
        ("summary", 0, "summary"),
    ]
    assert [row["error"] is None for row in responses] == [False, True, True]
    (transcription,) = stored_units(tmp_path)
    (summary,) = stored_units(tmp_path, KIND_SUMMARY)
    assert transcription["attempts"] == 2
    assert transcription["response_id"] == responses[1]["response_id"]
    assert summary["response_id"] == responses[2]["response_id"]
    assert transcription["usage"] is not None
