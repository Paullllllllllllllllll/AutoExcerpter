"""Tests for the streaming in-memory pipeline in pipeline/transcriber.py.

Covers ItemTranscriber on top of PdfPayloadSource with a mocked
TranscriptionManager and summarization disabled:
- end-to-end process_item: .txt output, no images/ directory, log header
  with file_provenance, page entries with image_provenance/source_file
- pre-render page-level resume: build_payload called only for pending pages
- render failure: preprocessing_failure entries without aborting the run
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import pipeline.transcriber as transcriber_module
from config import app as app_config
from imaging.payload import PagePayload, PdfPayloadSource
from pipeline.log import finalize_log_file
from pipeline.transcriber import ItemTranscriber


def _fake_transcribe(
    payload: PagePayload, max_schema_retries: int = 3
) -> dict[str, Any]:
    """Deterministic stand-in for TranscriptionManager.transcribe_payload."""
    return {
        "image": payload.image_name,
        "sequence_number": payload.sequence_number,
        "transcription": f"Transcribed text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


@pytest.fixture
def streaming_env(
    monkeypatch: pytest.MonkeyPatch, mock_config_loader: MagicMock
) -> MagicMock:
    """Patch ItemTranscriber dependencies for offline streaming tests.

    Replaces the real TranscriptionManager with a mock, routes config-loader
    lookups (pipeline and payload modules) to the shared mock configuration,
    and disables summarization.
    """
    mock_manager = MagicMock()
    mock_manager.transcribe_payload.side_effect = _fake_transcribe
    mock_manager_cls = MagicMock(return_value=mock_manager)

    monkeypatch.setattr(transcriber_module, "TranscriptionManager", mock_manager_cls)
    monkeypatch.setattr(
        transcriber_module, "get_config_loader", lambda: mock_config_loader
    )
    monkeypatch.setattr("imaging.payload.get_config_loader", lambda: mock_config_loader)
    monkeypatch.setattr(app_config, "SUMMARIZE", False)
    return mock_manager


def _make_transcriber(
    pdf_path: Path,
    output_dir: Path,
    completed_page_indices: set[int] | None = None,
) -> ItemTranscriber:
    """Construct an ItemTranscriber for a PDF input."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return ItemTranscriber(
        input_path=pdf_path,
        input_type="pdf",
        base_output_dir=output_dir,
        completed_page_indices=completed_page_indices,
    )


def _read_log_entries(log_path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL transcription log (header line + one object per line)."""
    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert isinstance(entries, list)
    return entries


def _seed_log(transcriber: ItemTranscriber, entries: list[dict[str, Any]]) -> None:
    """Write a versioned JSONL transcription log mimicking a crashed run."""
    from config.constants import LOG_FORMAT_VERSION

    header = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": transcriber.name,
        "input_item_path": str(transcriber.input_path),
        "input_type": "PDF",
        "total_images": len(entries),
        "model_name": transcriber.transcription_model,
    }
    lines = [json.dumps(header)] + [json.dumps(entry) for entry in entries]
    transcriber.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestProcessItemEndToEnd:
    """Full process_item run over a 3-page PDF with mocked API."""

    def test_end_to_end_outputs_and_provenance(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        """process_item writes the .txt, keeps no images dir, and logs provenance."""
        pdf_path = make_pdf("Book.pdf", num_pages=3)
        output_dir = tmp_path / "out"
        transcriber = _make_transcriber(pdf_path, output_dir)

        transcriber.process_item()

        assert transcriber.output_txt_path.exists()
        txt_content = transcriber.output_txt_path.read_text(encoding="utf-8")
        for page in ("page_0001.jpg", "page_0002.jpg", "page_0003.jpg"):
            assert f"Transcribed text of {page}." in txt_content

        assert not (transcriber.working_dir / "images").exists()
        assert streaming_env.transcribe_payload.call_count == 3

        entries = _read_log_entries(transcriber.log_path)
        header = entries[0]
        assert "file_provenance" in header
        assert header["file_provenance"]["source_file"] == str(pdf_path)
        assert header["file_provenance"]["source_sha256"]
        assert header["file_provenance"]["target_dpi"] == 300

        page_entries = sorted(
            entries[1:], key=lambda e: e["original_input_order_index"]
        )
        assert len(page_entries) == 3
        for index, entry in enumerate(page_entries):
            assert entry["source_file"] == str(pdf_path)
            assert entry["page_index"] == index
            provenance = entry["image_provenance"]
            assert provenance["sha256"]
            assert provenance["byte_size"] > 0
            # Direct render strategy: the render DPI is derived from the
            # active resize profile and never exceeds the target_dpi ceiling.
            assert 0 < provenance["effective_dpi"] <= 300


class TestPreRenderResume:
    """Page-level resume filters indices before any page is rendered."""

    def test_build_payload_called_only_for_pending_pages(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With pages 0 and 1 completed, only index 2 is rendered."""
        pdf_path = make_pdf("Resume.pdf", num_pages=3)
        output_dir = tmp_path / "out"
        transcriber = _make_transcriber(
            pdf_path, output_dir, completed_page_indices={0, 1}
        )
        _seed_log(
            transcriber,
            [
                {
                    "image": "page_0001.jpg",
                    "original_input_order_index": 0,
                    "transcription": "prior page one",
                },
                {
                    "image": "page_0002.jpg",
                    "original_input_order_index": 1,
                    "transcription": "prior page two",
                },
            ],
        )

        source = PdfPayloadSource(pdf_path)
        built_indices: list[int] = []
        original_build = source.build_payload

        def spy_build(index: int) -> PagePayload:
            built_indices.append(index)
            return original_build(index)

        monkeypatch.setattr(source, "build_payload", spy_build)

        try:
            results, _ = transcriber._transcribe_and_summarize(source)
        finally:
            source.close()
            finalize_log_file(transcriber.log_path)

        assert built_indices == [2]
        assert streaming_env.transcribe_payload.call_count == 1
        assert len(results) == 3
        assert [r["original_input_order_index"] for r in results] == [0, 1, 2]
        assert results[0]["transcription"] == "prior page one"
        assert (
            results[2]["transcription"].strip() == "Transcribed text of page_0003.jpg."
        )

    def test_all_pages_completed_skips_transcription(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        """When every page is completed, no payload is built at all."""
        pdf_path = make_pdf("Done.pdf", num_pages=2)
        output_dir = tmp_path / "out"
        transcriber = _make_transcriber(
            pdf_path, output_dir, completed_page_indices={0, 1}
        )
        _seed_log(
            transcriber,
            [
                {
                    "image": "page_0001.jpg",
                    "original_input_order_index": 0,
                    "transcription": "a",
                },
                {
                    "image": "page_0002.jpg",
                    "original_input_order_index": 1,
                    "transcription": "b",
                },
            ],
        )

        source = PdfPayloadSource(pdf_path)
        try:
            results, _ = transcriber._transcribe_and_summarize(source)
        finally:
            source.close()

        assert streaming_env.transcribe_payload.call_count == 0
        assert len(results) == 2


class TestProcessItemResume:
    """End-to-end resume through process_item keeps prior page results.

    Regression test: initialize_log_file truncates the log, so prior
    results must be snapshotted before reinitialization and re-appended
    to the fresh log.
    """

    def test_resumed_pages_survive_log_reinitialization(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        """Prior transcriptions appear in the .txt and in the new log."""
        pdf_path = make_pdf("ResumeE2E.pdf", num_pages=3)
        output_dir = tmp_path / "out"
        transcriber = _make_transcriber(
            pdf_path, output_dir, completed_page_indices={0, 1}
        )
        _seed_log(
            transcriber,
            [
                {
                    "image": "page_0001.jpg",
                    "original_input_order_index": 0,
                    "transcription": "prior page one text",
                },
                {
                    "image": "page_0002.jpg",
                    "original_input_order_index": 1,
                    "transcription": "prior page two text",
                },
            ],
        )

        transcriber.process_item()

        assert streaming_env.transcribe_payload.call_count == 1
        txt_content = transcriber.output_txt_path.read_text(encoding="utf-8")
        assert "prior page one text" in txt_content
        assert "prior page two text" in txt_content
        assert "Transcribed text of page_0003.jpg." in txt_content

        # The reinitialized log must contain the resumed entries again so a
        # later crash + resume still sees them as completed.
        entries = _read_log_entries(transcriber.log_path)
        logged_indices = {
            e["original_input_order_index"]
            for e in entries
            if "original_input_order_index" in e
        }
        assert logged_indices == {0, 1, 2}


class TestRenderFailure:
    """A failing build_payload yields a preprocessing_failure entry."""

    def test_render_failure_records_error_and_run_completes(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Page index 1 fails to render; the other pages still succeed."""
        pdf_path = make_pdf("Flaky.pdf", num_pages=3)
        output_dir = tmp_path / "out"
        transcriber = _make_transcriber(pdf_path, output_dir)

        source = PdfPayloadSource(pdf_path)
        original_build = source.build_payload

        def flaky_build(index: int) -> PagePayload:
            if index == 1:
                raise RuntimeError("render exploded")
            return original_build(index)

        monkeypatch.setattr(source, "build_payload", flaky_build)

        try:
            results, _ = transcriber._transcribe_and_summarize(source)
        finally:
            source.close()
            finalize_log_file(transcriber.log_path)

        assert len(results) == 3
        failed = results[1]
        assert failed["original_input_order_index"] == 1
        assert failed["error_type"] == "preprocessing_failure"
        assert failed["transcription"].startswith("[preprocessing error:")
        assert "render exploded" in failed["error"]
        assert failed["image"] == "page_0002.jpg"
        assert failed["sequence_number"] == 2

        succeeded = [results[0], results[2]]
        for entry in succeeded:
            assert "error" not in entry
            assert "image_provenance" in entry
        assert streaming_env.transcribe_payload.call_count == 2
