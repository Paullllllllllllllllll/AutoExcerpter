"""The item verdict, the written outputs and the per-item run report.

A page-level failure (a transcription or summary API error after retries) or
a unit that could not be stored fails the item, so the run counts it failed
and exits non-zero. ``ItemProcessor.written_outputs`` lists every file
written, which the ``--json`` run summary reports. Pending pages withhold the
final outputs, and the DOCX and Markdown writers fail independently but share
one set of stored citations.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import autoexcerpter.pipeline.outputs as outputs_module
from autoexcerpter.common.images import PagePayload
from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.units import DocumentHandle
from tests.pipeline.helpers import process, statuses, summarizer, transcriber


def _ok_transcribe(payload: PagePayload) -> dict[str, Any]:
    """Successful stand-in for TranscriptionManager.transcribe_payload."""
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Transcribed text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _failed_transcribe(payload: PagePayload) -> dict[str, Any]:
    """API-failure stand-in mirroring TranscriptionManager's error shape."""
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": "[transcription error: API error after retries]",
        "processing_time": 0.01,
        "error": "API error after retries",
        "error_type": "api_failure",
        "provider": "openai",
    }


def _ok_summary(transcription: str, page_num: int) -> dict[str, Any]:
    """Successful stand-in for SummaryManager.generate_summary."""
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": ["A point."],
        "references": None,
        "processing_time": 0.01,
        "provider": "openai",
    }


def _failed_summary(transcription: str, page_num: int) -> dict[str, Any]:
    """API-failure stand-in mirroring SummaryManager's placeholder shape."""
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["other"],
        },
        "bullet_points": ["[Error generating summary: API error]"],
        "references": None,
        "error": "Summary API error after retries",
        "error_type": "api_failure",
        "provider": "openai",
    }


@pytest.fixture
def pipeline_env(mock_image_processing: dict[str, Any]) -> MagicMock:
    """Return a mock transcription manager for offline runs (no API calls)."""
    return transcriber(_ok_transcribe)


@pytest.fixture
def summary_env(monkeypatch: pytest.MonkeyPatch, pipeline_env: MagicMock) -> MagicMock:
    """Return a mock summary manager and mock the summary-file writers
    (no docx/md is actually rendered)."""
    mock_summary_manager = summarizer(_ok_summary)
    monkeypatch.setattr(outputs_module, "enrich_if_enabled", lambda cm: None)
    monkeypatch.setattr(outputs_module, "create_docx_summary", MagicMock())
    monkeypatch.setattr(outputs_module, "create_markdown_summary", MagicMock())
    return mock_summary_manager


def _process(
    pdf_path: Path,
    output_dir: Path,
    manager: MagicMock,
    summary_manager: MagicMock | None = None,
    prepare: Callable[[ItemProcessor], None] | None = None,
    **options: Any,
) -> tuple[ItemProcessor, bool]:
    return process(
        pdf_path,
        output_dir,
        prepare=prepare,
        transcribe_manager=manager,
        summary_manager=summary_manager,
        summarize=summary_manager is not None,
        **options,
    )


class TestItemFailurePropagation:
    """process must return False when any page fails."""

    def test_all_pages_succeed_returns_true(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        _, success = _process(make_pdf("OK.pdf", 2), tmp_path / "out", pipeline_env)
        assert success is True
        assert statuses(tmp_path / "out") == ["done", "done"]

    def test_all_pages_fail_transcription_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        """An item whose pages all fail is not complete."""
        pipeline_env.transcribe_payload.side_effect = _failed_transcribe
        _, success = _process(
            make_pdf("AllFail.pdf", 3), tmp_path / "out", pipeline_env
        )
        assert success is False
        assert statuses(tmp_path / "out") == ["failed"] * 3

    def test_single_failed_page_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        calls: list[int] = []

        def flaky(payload: PagePayload) -> dict[str, Any]:
            calls.append(payload.number)
            if payload.number == 2:
                return _failed_transcribe(payload)
            return _ok_transcribe(payload)

        pipeline_env.transcribe_payload.side_effect = flaky
        _, success = _process(
            make_pdf("OneFail.pdf", 3), tmp_path / "out", pipeline_env
        )
        assert success is False
        assert sorted(calls) == [1, 2, 3]

    def test_failed_summary_page_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        summary_env: MagicMock,
    ) -> None:
        """A summary that fails after retries fails the item."""

        def flaky_summary(transcription: str, page_num: int) -> dict[str, Any]:
            if page_num == 2:
                return _failed_summary(transcription, page_num)
            return _ok_summary(transcription, page_num)

        summary_env.generate_summary.side_effect = flaky_summary
        out = tmp_path / "out"
        _, success = _process(
            make_pdf("SummaryFail.pdf", 3), out, pipeline_env, summary_env
        )
        assert success is False
        assert statuses(out, "summary") == ["done", "failed", "done"]

    def test_successful_summaries_return_true(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        summary_env: MagicMock,
    ) -> None:
        _, success = _process(
            make_pdf("SummaryOK.pdf", 2), tmp_path / "out", pipeline_env, summary_env
        )
        assert success is True

    def test_summary_render_failure_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        summary_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed summary-file render means the item is not complete."""
        monkeypatch.setattr(
            outputs_module,
            "create_docx_summary",
            MagicMock(side_effect=RuntimeError("docx exploded")),
        )
        _, success = _process(
            make_pdf("RenderFail.pdf", 1), tmp_path / "out", pipeline_env, summary_env
        )
        assert success is False

    def test_failed_txt_write_returns_false_and_withholds_output(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed final .txt write (the writer returns False) fails the item
        and leaves the missing file out of written_outputs."""
        monkeypatch.setattr(
            outputs_module,
            "write_transcription_to_text",
            lambda *a, **k: False,
        )
        processor, success = _process(
            make_pdf("TxtFail.pdf", 1), tmp_path / "out", pipeline_env
        )
        assert success is False
        assert processor.written_outputs == []

    def test_a_unit_that_could_not_be_stored_fails_the_item(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real = DocumentHandle.write_transcription

        async def failing_second(self: DocumentHandle, result: Any) -> None:
            if result.idx == 1:
                raise sqlite3.OperationalError("disk I/O error")
            await real(self, result)

        monkeypatch.setattr(DocumentHandle, "write_transcription", failing_second)
        processor, success = _process(
            make_pdf("Unstored.pdf", 2), tmp_path / "out", pipeline_env
        )

        assert success is False
        assert processor.written_outputs == []
        assert processor.last_run_report is not None
        assert processor.last_run_report["pages_failed"] == 0
        assert processor.last_run_report["pages_deferred"] == 1


class TestJsonOutputs:
    """written_outputs must flow from the processor to the JSON summary."""

    def test_written_outputs_contains_txt(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        processor, success = _process(
            make_pdf("Outs.pdf", 1), tmp_path / "out", pipeline_env
        )
        assert success is True
        txt = processor.paths.transcription
        assert processor.written_outputs == [txt.resolve()]
        assert txt.exists()

    def test_written_outputs_include_summary_files(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        summary_env: MagicMock,
    ) -> None:
        processor, success = _process(
            make_pdf("Summ.pdf", 1), tmp_path / "out", pipeline_env, summary_env
        )
        assert success is True
        assert processor.written_outputs == [
            processor.paths.transcription.resolve(),
            processor.paths.summary_docx.resolve(),
            processor.paths.summary_md.resolve(),
        ]


class TestDeferralWithholding:
    """A run that leaves pages pending must NOT emit a truncated .txt that
    self-reports as complete, must fail the item (exit 1, items_failed), and
    must keep the stored pages so a later run finishes the item."""

    def test_partial_run_withholds_txt_and_fails(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """process() is False, the final .txt is NOT written, nothing is
        registered in written_outputs, and the store holds exactly the one
        page that completed before the deferral."""

        def deferring(processor: ItemProcessor) -> None:
            real_run = processor.runner.run

            async def run(idx: int, source: Any, progress: PageProgress) -> str:
                if idx >= 1:
                    return "deferred"
                return await real_run(idx, source, progress)

            monkeypatch.setattr(processor.runner, "run", run)

        out = tmp_path / "out"
        processor, success = _process(
            make_pdf("Partial.pdf", 3), out, pipeline_env, prepare=deferring
        )

        assert success is False
        assert not processor.paths.transcription.exists()
        assert processor.written_outputs == []
        assert statuses(out) == ["done", "pending", "pending"]


class TestIndependentSummaryWriters:
    def test_docx_failure_does_not_skip_markdown(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        summary_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A DOCX writer exception must not prevent the Markdown writer from
        running; the item still fails because not all outputs were produced."""
        md_writer = MagicMock()
        monkeypatch.setattr(
            outputs_module,
            "create_docx_summary",
            MagicMock(side_effect=RuntimeError("docx exploded")),
        )
        monkeypatch.setattr(outputs_module, "create_markdown_summary", md_writer)

        processor, success = _process(
            make_pdf("Split.pdf", 1), tmp_path / "out", pipeline_env, summary_env
        )
        assert success is False
        assert md_writer.called
        suffixes = {p.suffix for p in processor.written_outputs}
        assert ".docx" not in suffixes
        assert ".md" in suffixes

    def test_both_writers_render_the_same_stored_citations(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The DOCX and Markdown writers receive the citations the store holds."""
        captured: dict[str, Any] = {}
        citation = "Smith, J. (2020). A Title. Journal, 1, 1-2."

        def spy(kind: str) -> Callable[..., None]:
            def writer(*_args: Any, citation_manager: Any = None, **_: Any) -> None:
                captured[kind] = citation_manager

            return writer

        def summarize(transcription: str, page_num: int) -> dict[str, Any]:
            summary = _ok_summary(transcription, page_num)
            summary["references"] = [{"citation": citation, "is_partial": False}]
            return summary

        monkeypatch.setattr(outputs_module, "create_docx_summary", spy("docx"))
        monkeypatch.setattr(outputs_module, "create_markdown_summary", spy("md"))

        _process(
            make_pdf("Cite.pdf", num_pages=1),
            tmp_path / "out",
            pipeline_env,
            summarizer(summarize),
            openalex_enabled=False,
        )

        assert captured["docx"] is captured["md"]
        pairs = captured["docx"].get_citations_with_pages()
        assert [(pair[0].raw_text, pair[1]) for pair in pairs] == [(citation, "p. 1")]


class TestLastRunReport:
    def test_report_populated_on_success(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        processor, success = _process(
            make_pdf("Report.pdf", 2), tmp_path / "out", pipeline_env
        )
        assert success is True

        report = processor.last_run_report
        assert report is not None
        assert report["pages_total"] == 2
        assert report["pages_attempted"] == 2
        assert report["pages_ok"] == 2
        assert report["pages_failed"] == 0
        assert report["pages_deferred"] == 0
        assert report["summary_failures"] == 0
        assert report["elapsed_s"] >= 0
        assert report["outputs"] == [str(p) for p in processor.written_outputs]

    def test_report_populated_on_failure(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        pipeline_env.transcribe_payload.side_effect = _failed_transcribe
        processor, success = _process(
            make_pdf("RepFail.pdf", 2), tmp_path / "out", pipeline_env
        )
        assert success is False

        report = processor.last_run_report
        assert report is not None
        assert report["pages_total"] == 2
        assert report["pages_failed"] == 2
        assert report["pages_ok"] == 0
