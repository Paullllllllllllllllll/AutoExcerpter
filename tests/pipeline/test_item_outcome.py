"""The item verdict, the written outputs and the per-item run report.

A page-level failure (a transcription or summary API error after retries) or a
failed working-log append fails the item, so the run counts it failed and exits
non-zero. ``ItemProcessor.written_outputs`` lists every file written, which the
``--json`` run summary reports. Deferred pages withhold the final outputs, and
the DOCX and Markdown writers fail independently but share one citation
manager.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import autoexcerpter.pipeline.outputs as outputs_module
from autoexcerpter.common.images import PagePayload
from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.resume import load_results
from autoexcerpter.rendering.citations import CitationManager
from tests.pipeline.helpers import (
    make_processor,
    summarizer,
    transcriber,
    write_working_logs,
)


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
    monkeypatch.setattr(
        outputs_module,
        "build_render_context",
        lambda *a, **k: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(outputs_module, "enrich_if_enabled", lambda cm: None)
    monkeypatch.setattr(outputs_module, "create_docx_summary", MagicMock())
    monkeypatch.setattr(outputs_module, "create_markdown_summary", MagicMock())
    return mock_summary_manager


def _make_processor(
    pdf_path: Path,
    output_dir: Path,
    manager: MagicMock,
    summary_manager: MagicMock | None = None,
) -> ItemProcessor:
    return make_processor(
        pdf_path,
        output_dir,
        transcribe_manager=manager,
        summary_manager=summary_manager,
        summarize=summary_manager is not None,
    )


def _process(processor: ItemProcessor) -> bool:
    return asyncio.run(processor.process())


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


class TestItemFailurePropagation:
    """process must return False when any page fails."""

    def test_all_pages_succeed_returns_true(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        processor = _make_processor(
            make_pdf("OK.pdf", 2), tmp_path / "out", pipeline_env
        )
        assert _process(processor) is True

    def test_all_pages_fail_transcription_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        """An item whose pages all fail is not complete."""
        pipeline_env.transcribe_payload.side_effect = _failed_transcribe
        processor = _make_processor(
            make_pdf("AllFail.pdf", 3), tmp_path / "out", pipeline_env
        )
        assert _process(processor) is False

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
        processor = _make_processor(
            make_pdf("OneFail.pdf", 3), tmp_path / "out", pipeline_env
        )
        assert _process(processor) is False
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
        processor = _make_processor(
            make_pdf("SummaryFail.pdf", 3), tmp_path / "out", pipeline_env, summary_env
        )
        assert _process(processor) is False

    def test_successful_summaries_return_true(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        summary_env: MagicMock,
    ) -> None:
        processor = _make_processor(
            make_pdf("SummaryOK.pdf", 2), tmp_path / "out", pipeline_env, summary_env
        )
        assert _process(processor) is True

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
        processor = _make_processor(
            make_pdf("RenderFail.pdf", 1), tmp_path / "out", pipeline_env, summary_env
        )
        assert _process(processor) is False

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
        processor = _make_processor(
            make_pdf("TxtFail.pdf", 1), tmp_path / "out", pipeline_env
        )
        assert _process(processor) is False
        assert processor.written_outputs == []

    def test_failed_append_turns_the_item_verdict_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A page entry that never reached the working log fails the item."""
        processor = make_processor(tmp_path / "doc.pdf", tmp_path / "out")

        async def _pages(source: Any) -> tuple[list[dict[str, Any]], list[Any]]:
            return [{"original_input_order_index": 0}], []

        monkeypatch.setattr(processor, "_transcribe_and_summarize", _pages)

        # With no failed append the item is complete.
        assert asyncio.run(processor._run(_FakeSource(1))) is True  # type: ignore[arg-type]

        processor.written_outputs = []
        processor.runner.log_append_failures = 2

        assert asyncio.run(processor._run(_FakeSource(1))) is False  # type: ignore[arg-type]
        assert processor.last_run_report is not None
        assert processor.last_run_report["pages_failed"] == 0


class TestJsonOutputs:
    """written_outputs must flow from the processor to the JSON summary."""

    def test_written_outputs_contains_txt(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        processor = _make_processor(
            make_pdf("Outs.pdf", 1), tmp_path / "out", pipeline_env
        )
        assert _process(processor) is True
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
        processor = _make_processor(
            make_pdf("Summ.pdf", 1), tmp_path / "out", pipeline_env, summary_env
        )
        assert _process(processor) is True
        assert processor.written_outputs == [
            processor.paths.transcription.resolve(),
            processor.paths.summary_docx.resolve(),
            processor.paths.summary_md.resolve(),
        ]


class TestDeferralWithholding:
    """A run that leaves pages deferred must NOT emit a truncated .txt that
    self-reports as complete, must fail the item (exit 1, items_failed), and
    must retain the completed pages in the working log so a later run resumes
    and finishes them."""

    def test_partial_run_withholds_txt_and_fails(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """process() is False, the final .txt is NOT written, nothing is
        registered in written_outputs, and the working log holds exactly the
        one page that completed before the deferral."""
        processor = _make_processor(
            make_pdf("Partial.pdf", 3), tmp_path / "out", pipeline_env
        )

        # Defer every page after index 0; page 0 transcribes for real via the
        # mocked manager.
        real_run = processor.runner.run

        async def deferring(
            idx: int,
            source: Any,
            t_results: list[dict[str, Any]],
            s_results: list[dict[str, Any]],
            progress: PageProgress,
        ) -> dict[str, Any] | None:
            if idx >= 1:
                return None
            return await real_run(idx, source, t_results, s_results, progress)

        monkeypatch.setattr(processor.runner, "run", deferring)

        assert _process(processor) is False
        # The truncated .txt must not exist and must not be advertised.
        assert not processor.paths.transcription.exists()
        assert processor.written_outputs == []
        # The working log retains exactly the completed page, so a later
        # resume can finish the rest.
        log_path = processor.paths.transcription_log
        logged = load_results(log_path) or []
        assert [entry["original_input_order_index"] for entry in logged] == [0]
        assert "error" not in logged[0]


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

        processor = _make_processor(
            make_pdf("Split.pdf", 1), tmp_path / "out", pipeline_env, summary_env
        )
        assert _process(processor) is False
        # The Markdown writer ran despite the DOCX failure.
        assert md_writer.called
        # The DOCX path is not advertised; the Markdown path is.
        suffixes = {p.suffix for p in processor.written_outputs}
        assert ".docx" not in suffixes
        assert ".md" in suffixes

    def test_one_citation_manager_shared_by_both_writers(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The DOCX and Markdown writers receive the same CitationManager."""
        captured: dict[str, Any] = {}

        def spy_docx(
            results: list[dict[str, Any]],
            path: Path,
            name: str,
            citation_manager: CitationManager | None = None,
            data: object = None,
        ) -> None:
            captured["docx"] = citation_manager

        def spy_md(
            results: list[dict[str, Any]],
            path: Path,
            name: str,
            citation_manager: CitationManager | None = None,
            data: object = None,
        ) -> None:
            captured["md"] = citation_manager

        monkeypatch.setattr(outputs_module, "create_docx_summary", spy_docx)
        monkeypatch.setattr(outputs_module, "create_markdown_summary", spy_md)

        processor = make_processor(
            make_pdf("Cite.pdf", num_pages=1),
            tmp_path / "out",
            transcribe_manager=pipeline_env,
            summary_manager=summarizer(_ok_summary),
            completed={0},
            summarize=True,
            openalex_enabled=False,
        )
        write_working_logs(
            processor,
            tx_entries=[
                {
                    "image": "page_0001.jpg",
                    "original_input_order_index": 0,
                    "transcription": "text",
                }
            ],
            summary_entries=[
                {
                    "original_input_order_index": 0,
                    "page": 1,
                    "page_information": {
                        "page_number_integer": 1,
                        "page_number_type": "arabic",
                        "page_types": ["content"],
                    },
                    "bullet_points": ["a point"],
                    "references": [
                        {
                            "citation": "Smith, J. (2020). A Title. Journal, 1, 1-2.",
                            "is_partial": False,
                        }
                    ],
                }
            ],
            total_images=1,
        )

        asyncio.run(processor.process())

        assert captured["docx"] is not None
        assert captured["docx"] is captured["md"]
        assert isinstance(captured["docx"], CitationManager)


class TestLastRunReport:
    def test_report_populated_on_success(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        processor = _make_processor(
            make_pdf("Report.pdf", 2), tmp_path / "out", pipeline_env
        )
        assert _process(processor) is True

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
        processor = _make_processor(
            make_pdf("RepFail.pdf", 2), tmp_path / "out", pipeline_env
        )
        assert _process(processor) is False

        report = processor.last_run_report
        assert report is not None
        assert report["pages_total"] == 2
        assert report["pages_failed"] == 2
        assert report["pages_ok"] == 0
