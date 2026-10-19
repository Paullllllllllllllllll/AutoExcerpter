"""The streaming in-memory pipeline of pipeline/item.py, into the store.

Covers ItemProcessor on top of PdfPayloadSource with a mocked
TranscriptionManager and summarization disabled:
- end-to-end process: the transcription file, no images or working folder,
  the image settings on the document, the provenance of every page unit
- pre-render resume: build_payload called only for the pages still to do
- render failure: a failed preprocessing_failure unit without aborting
- shrunk input: the lost pages' units go with their finished batch requests
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from autoexcerpter.common.images import PagePayload
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.common.store import Submission, UnitKey
from autoexcerpter.output.store import DATABASE_NAME, KIND_TRANSCRIPTION
from autoexcerpter.pipeline.item import ItemProcessor, PayloadSource
from autoexcerpter.pipeline.units import DocumentHandle
from tests.pipeline.helpers import (
    MODEL,
    process,
    store_pages,
    stored_units,
    transcriber,
)


def _fake_transcribe(payload: PagePayload) -> dict[str, Any]:
    """Deterministic stand-in for TranscriptionManager.transcribe_payload."""
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Transcribed text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


@pytest.fixture
def streaming_env(mock_image_processing: dict[str, Any]) -> MagicMock:
    """Return a mock transcription manager over the small image sections."""
    return transcriber(_fake_transcribe)


def _spy_on_pages(built: list[int], fail_at: int | None = None) -> Any:
    """Return a ``prepare`` hook recording, or failing, the rendered pages."""

    def prepare(processor: ItemProcessor) -> None:
        open_source = processor._open_source

        def opened() -> PayloadSource:
            source = open_source()
            build = source.build_payload

            def spy(index: int) -> PagePayload:
                built.append(index)
                if index == fail_at:
                    raise RuntimeError("render exploded")
                return build(index)

            source.build_payload = spy  # type: ignore[method-assign]
            return source

        processor._open_source = opened  # type: ignore[method-assign]

    return prepare


def _two_stored_pages(texts: dict[int, str]) -> Callable[[DocumentHandle], Any]:
    async def before(document: DocumentHandle) -> None:
        await store_pages(document, 3, texts)

    return before


class TestProcessItemEndToEnd:
    """Full process run over a 3-page PDF with mocked API."""

    def test_end_to_end_outputs_and_provenance(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        pdf_path = make_pdf("Book.pdf", num_pages=3)
        out = tmp_path / "out"

        processor, success = process(pdf_path, out, transcribe_manager=streaming_env)

        assert success
        txt_content = processor.paths.transcription.read_text(encoding="utf-8")
        for page in ("page_0001.jpg", "page_0002.jpg", "page_0003.jpg"):
            assert f"Transcribed text of {page}." in txt_content
        assert not any(path.is_dir() for path in out.iterdir())
        assert streaming_env.transcribe_payload.call_count == 3

        (document,) = read_rows(out / DATABASE_NAME, "documents")
        settings = json.loads(document["image_settings"])
        assert settings["model_type"] == "openai"
        assert settings["target_dpi"] == 300
        units = stored_units(out)
        assert [unit["source_index"] for unit in units] == [0, 1, 2]
        for unit in units:
            provenance = json.loads(unit["provenance"])
            assert provenance["source_file"] == str(pdf_path)
            image = provenance["image_provenance"]
            assert "sha256" not in image
            assert image["byte_size"] > 0
            # Direct render strategy: the render DPI is derived from the
            # active resize profile and never exceeds the target_dpi ceiling.
            assert 0 < image["effective_dpi"] <= 300


class TestPreRenderResume:
    """Page-level resume filters indices before any page is rendered."""

    def test_build_payload_called_only_for_pending_pages(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        """With pages 0 and 1 stored, only index 2 is rendered."""
        built: list[int] = []
        out = tmp_path / "out"

        processor, success = process(
            make_pdf("Resume.pdf", num_pages=3),
            out,
            before=_two_stored_pages({0: "prior page one", 1: "prior page two"}),
            prepare=_spy_on_pages(built),
            transcribe_manager=streaming_env,
        )

        assert success
        assert built == [2]
        assert streaming_env.transcribe_payload.call_count == 1
        contents = [unit["content"] for unit in stored_units(out)]
        assert contents[:2] == ["prior page one", "prior page two"]
        assert contents[2].strip() == "Transcribed text of page_0003.jpg."
        txt_content = processor.paths.transcription.read_text(encoding="utf-8")
        assert "prior page one" in txt_content
        assert "Transcribed text of page_0003.jpg." in txt_content

    def test_all_pages_stored_skips_transcription(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        """When every page is stored, no payload is built at all."""
        built: list[int] = []

        _, success = process(
            make_pdf("Done.pdf", num_pages=3),
            tmp_path / "out",
            before=_two_stored_pages({0: "a", 1: "b", 2: "c"}),
            prepare=_spy_on_pages(built),
            transcribe_manager=streaming_env,
        )

        assert success
        assert built == []
        assert streaming_env.transcribe_payload.call_count == 0


class TestRenderFailure:
    """A failing build_payload yields a failed preprocessing_failure unit."""

    def test_render_failure_records_error_and_run_completes(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        """Page index 1 fails to render; the other pages still succeed."""
        out = tmp_path / "out"

        _, success = process(
            make_pdf("Flaky.pdf", num_pages=3),
            out,
            prepare=_spy_on_pages([], fail_at=1),
            transcribe_manager=streaming_env,
        )

        assert not success
        units = stored_units(out)
        failed = units[1]
        assert (failed["status"], failed["error_type"]) == (
            "failed",
            "preprocessing_failure",
        )
        assert failed["content"].startswith("[preprocessing error:")
        assert "render exploded" in failed["error"]
        assert failed["label"] == "page_0002.jpg"
        for unit in (units[0], units[2]):
            assert unit["status"] == "done"
            assert "image_provenance" in json.loads(unit["provenance"])
        assert streaming_env.transcribe_payload.call_count == 2


class TestShrunkInput:
    """An input that lost pages drops their units and finished batch requests."""

    def test_a_finalized_submission_on_lost_pages_does_not_block_the_drop(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        streaming_env: MagicMock,
    ) -> None:
        """Four stored pages, three of them batched, then a two-page input."""
        out = tmp_path / "out"

        async def upload(submission: Submission) -> str:
            return "file-1"

        async def create(submission: Submission) -> str:
            return "batch-1"

        async def before(document: DocumentHandle) -> None:
            await store_pages(document, 4, {idx: f"page {idx}" for idx in range(4)})
            units = [
                UnitKey(document.doc_id, KIND_TRANSCRIPTION, idx) for idx in (1, 2, 3)
            ]
            submission = await document.store.submit(
                provider="openai",
                model=MODEL,
                units=units,
                upload=upload,
                create=create,
            )
            await document.store.mark_finalized(submission.submission_id)

        _, success = process(
            make_pdf("Short.pdf", num_pages=2),
            out,
            before=before,
            transcribe_manager=streaming_env,
        )

        assert success
        assert [unit["idx"] for unit in stored_units(out)] == [0, 1]
        requests = read_rows(out / DATABASE_NAME, "submission_requests")
        assert [row["idx"] for row in requests] == [1]
