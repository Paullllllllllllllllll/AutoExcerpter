"""Input-change detection through the provenance in the working-log header.

A PDF's byte size and an image folder's image count, total bytes and image
names identify the input. A changed input refuses page-level reuse and keeps
existing outputs from counting as complete, so two documents are never
spliced into one output; a header without these fields proves no change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from autoexcerpter.constants import LOG_FORMAT_VERSION
from autoexcerpter.imaging.payload import FolderPayloadSource
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.resume import (
    ProcessingState,
    ResumeChecker,
    input_changed_since_log,
)


def _write_working_log(
    output_dir: Path,
    item_name: str,
    entries: list[dict[str, Any]],
    file_provenance: dict[str, Any] | None,
) -> Path:
    """Create a versioned transcription working log with an optional provenance."""
    paths = ItemPaths.for_item(item_name, output_dir)
    paths.working_dir.mkdir(parents=True, exist_ok=True)
    log_path = paths.transcription_log

    header: dict[str, Any] = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": item_name,
        "input_type": "PDF",
        "total_images": len(entries),
        "model_name": "gpt-5-mini",
    }
    if file_provenance is not None:
        header["file_provenance"] = file_provenance

    lines = [json.dumps(header)] + [json.dumps(e) for e in entries]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


class TestFileProvenance:
    """The byte size of a PDF decides whether its logged pages are reused."""

    @staticmethod
    def _entries() -> list[dict[str, Any]]:
        return [
            {"original_input_order_index": 0, "transcription": "page zero"},
            {"original_input_order_index": 1, "transcription": "page one"},
        ]

    def _checker(self) -> ResumeChecker:
        return ResumeChecker(
            resume_mode="skip",
            summarize=True,
            output_docx=True,
            output_markdown=True,
        )

    def test_size_mismatch_refuses_page_reuse(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        input_file = tmp_path / "TestDoc.pdf"
        input_file.write_bytes(b"the real current input bytes")

        _write_working_log(
            output_dir,
            "TestDoc",
            self._entries(),
            file_provenance={
                "source_file": str(input_file),
                "size": input_file.stat().st_size + 100,  # a different file
            },
        )

        result = self._checker().should_skip("TestDoc", output_dir)
        # Page-level reuse refused: no completed indices, full reprocess.
        assert result.completed_page_indices is None
        assert result.state == ProcessingState.NONE

    def test_size_match_resumes_normally(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        input_file = tmp_path / "TestDoc.pdf"
        input_file.write_bytes(b"the real current input bytes")

        _write_working_log(
            output_dir,
            "TestDoc",
            self._entries(),
            file_provenance={
                "source_file": str(input_file),
                "size": input_file.stat().st_size,  # matches: same file
            },
        )

        result = self._checker().should_skip("TestDoc", output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0, 1}

    def test_header_without_provenance_is_not_an_input_change(
        self, tmp_path: Path
    ) -> None:
        """An absent provenance proves no input change, so pages stay listed.

        The item processor's image-settings check refuses such a log before reuse.
        """
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        _write_working_log(
            output_dir,
            "TestDoc",
            self._entries(),
            file_provenance=None,
        )

        result = self._checker().should_skip("TestDoc", output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0, 1}


class TestFolderProvenance:
    """An image folder's image count and total bytes identify it."""

    def _make_folder(self, tmp_path: Path, count: int) -> Path:
        folder = tmp_path / "imgs"
        folder.mkdir(exist_ok=True)
        for i in range(count):
            Image.new("RGB", (20, 20), color=(i * 10, 0, 0)).save(
                folder / f"page_{i:03d}.jpg", "JPEG"
            )
        return folder

    @pytest.mark.usefixtures("mock_image_processing")
    def test_provenance_records_identity_fields(self, tmp_path: Path) -> None:
        folder = self._make_folder(tmp_path, 3)
        source = FolderPayloadSource(folder, provider="openai", model_name="gpt-5")
        prov = source.file_provenance()
        assert prov["image_count"] == 3
        expected = sum(p.stat().st_size for p in folder.glob("*.jpg"))
        assert prov["total_image_bytes"] == expected

    def test_added_image_is_an_input_change(self, tmp_path: Path) -> None:
        folder = self._make_folder(tmp_path, 2)
        total = sum(p.stat().st_size for p in folder.glob("*.jpg"))
        header = {
            "file_provenance": {
                "source_file": str(folder),
                "image_count": 2,
                "total_image_bytes": total,
            }
        }
        assert input_changed_since_log(header) is False

        # An added page changes the image count, so the input changed.
        Image.new("RGB", (20, 20), color=(9, 9, 9)).save(
            folder / "page_099.jpg", "JPEG"
        )
        assert input_changed_since_log(header) is True

    def test_header_without_identity_fields_is_not_a_change(
        self, tmp_path: Path
    ) -> None:
        # A folder path without image_count proves no change.
        folder = self._make_folder(tmp_path, 1)
        header = {"file_provenance": {"source_file": str(folder)}}
        assert input_changed_since_log(header) is False


def _write_complete_outputs(
    output_dir: Path, item_name: str, provenance: dict[str, Any]
) -> None:
    """Create a complete-looking output set plus a versioned working log."""
    paths = ItemPaths.for_item(item_name, output_dir)
    for output in (paths.transcription, paths.summary_docx, paths.summary_md):
        output.write_text("x", encoding="utf-8")
    entry = {
        "original_input_order_index": 0,
        "image": "page_001.png",
        "transcription": "text",
    }
    _write_working_log(output_dir, item_name, [entry], provenance)


class TestInputChangedBlocksComplete:
    """A changed input keeps existing outputs from classifying as COMPLETE."""

    def test_changed_size_forces_reprocess(self, tmp_path: Path) -> None:
        source = tmp_path / "book.pdf"
        source.write_bytes(b"new content of a different length")
        out = tmp_path / "out"
        out.mkdir()
        _write_complete_outputs(out, "book", {"source_file": str(source), "size": 5})

        result = ResumeChecker(resume_mode="skip").should_skip("book", out)

        assert result.state != ProcessingState.COMPLETE
        assert result.completed_page_indices is None

    def test_unchanged_size_is_complete(self, tmp_path: Path) -> None:
        source = tmp_path / "book.pdf"
        source.write_bytes(b"12345")
        out = tmp_path / "out"
        out.mkdir()
        _write_complete_outputs(out, "book", {"source_file": str(source), "size": 5})

        result = ResumeChecker(resume_mode="skip").should_skip("book", out)

        assert result.state == ProcessingState.COMPLETE


class TestFolderRenameDetected:
    """A rename that keeps the image count and bytes is an input change."""

    @staticmethod
    def _provenance(folder: Path, logged_names: list[str]) -> dict[str, Any]:
        names_hash = hashlib.sha256(
            "\n".join(sorted(logged_names)).encode("utf-8")
        ).hexdigest()
        return {
            "source_file": str(folder),
            "image_count": 2,
            "total_image_bytes": 4,
            "image_names_sha256": names_hash,
        }

    def test_rename_flagged_as_changed(self, tmp_path: Path) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "b.jpg").write_bytes(b"bb")
        (folder / "z.jpg").write_bytes(b"aa")  # logged as a.jpg
        provenance = self._provenance(folder, ["a.jpg", "b.jpg"])

        assert input_changed_since_log({"file_provenance": provenance}) is True

    def test_unchanged_names_pass(self, tmp_path: Path) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(b"aa")
        (folder / "b.jpg").write_bytes(b"bb")
        provenance = self._provenance(folder, ["a.jpg", "b.jpg"])

        assert input_changed_since_log({"file_provenance": provenance}) is False
