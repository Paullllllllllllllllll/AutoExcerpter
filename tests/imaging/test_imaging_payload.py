"""Tests for imaging/payload.py - FolderPayloadSource and module helpers.

`PdfPayloadSource` coverage lives in test_imaging_pdf.py and
test_imaging_pdf_extended.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from autoexcerpter.common.images import sequence_number, sha256_of_file
from autoexcerpter.imaging.payload import FolderPayloadSource


@pytest.fixture
def patched_payload_config(mock_image_processing: dict[str, Any]) -> dict[str, Any]:
    """Use the small image sections for the payload sources."""
    return mock_image_processing


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    """Create a folder with three small images carrying numbered names."""
    folder = tmp_path / "scans"
    folder.mkdir()
    for name, color in [
        ("page_0001.jpg", (200, 200, 200)),
        ("page_0002.jpg", (100, 100, 100)),
        ("scan-7.png", (50, 50, 50)),
    ]:
        Image.new("RGB", (120, 90), color=color).save(folder / name)
    return folder


class TestExtractSequenceNumber:
    """Tests for the sequence_number helper."""

    def test_page_pattern(self) -> None:
        assert sequence_number(Path("page_0001.jpg")) == 1

    def test_page_pattern_long(self) -> None:
        assert sequence_number(Path("page_0012.jpg")) == 12

    def test_dash_separated_number(self) -> None:
        assert sequence_number(Path("scan-7.jpg")) == 7

    def test_no_number_returns_zero(self) -> None:
        assert sequence_number(Path("noNumber.jpg")) == 0

    def test_multiple_numbers_takes_last(self) -> None:
        assert sequence_number(Path("doc_2024_page_005.jpg")) == 5

    def test_number_only_stem(self) -> None:
        assert sequence_number(Path("0010.jpg")) == 10

    def test_mixed_text_and_numbers(self) -> None:
        assert sequence_number(Path("vol2_ch3_page_15.png")) == 15

    def test_full_path(self) -> None:
        assert sequence_number(Path("/deep/path/scan_page_0123.tiff")) == 123


class TestModuleHelpers:
    """Tests for sha256_of_file."""

    def test_sha256_of_file_matches_hashlib(self, tmp_path: Path) -> None:
        """Streaming file hash matches a direct hashlib digest."""
        target = tmp_path / "data.bin"
        target.write_bytes(b"some binary content" * 100)

        expected = hashlib.sha256(target.read_bytes()).hexdigest()
        assert sha256_of_file(target) == expected


class TestFolderPayloadSource:
    """Tests for FolderPayloadSource."""

    def test_len_and_sorted_listing(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """Source lists all images in sorted filename order."""
        source = FolderPayloadSource(image_folder)

        assert len(source) == 3
        assert source.image_name(0) == "page_0001.jpg"
        assert source.image_name(1) == "page_0002.jpg"
        assert source.image_name(2) == "scan-7.png"

    def test_empty_folder(
        self, tmp_path: Path, patched_payload_config: MagicMock
    ) -> None:
        """An empty folder yields a zero-length source."""
        empty = tmp_path / "empty"
        empty.mkdir()
        source = FolderPayloadSource(empty)

        assert len(source) == 0

    def test_build_payload_basic_fields(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """Payload carries the real filename and folder-derived metadata."""
        source = FolderPayloadSource(image_folder)
        payload = source.build_payload(0)

        assert payload.image_name == "page_0001.jpg"
        assert payload.index == 0
        assert payload.page_index is None
        assert payload.source_file == str(image_folder / "page_0001.jpg")

    def test_sequence_numbers_from_filenames(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """Sequence numbers are parsed from the image filenames."""
        source = FolderPayloadSource(image_folder)

        assert source.build_payload(0).number == 1
        assert source.build_payload(1).number == 2
        assert source.build_payload(2).number == 7

    def test_provenance_fields(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """Provenance includes source_sha256 of the file and no effective_dpi."""
        source = FolderPayloadSource(image_folder)
        payload = source.build_payload(0)

        decoded = base64.b64decode(payload.base64)
        source_bytes = (image_folder / "page_0001.jpg").read_bytes()
        provenance = payload.provenance

        assert "sha256" not in provenance
        assert provenance["byte_size"] == len(decoded)
        assert provenance["width"] > 0
        assert provenance["height"] > 0
        assert provenance["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert "effective_dpi" not in provenance

    def test_payload_is_valid_jpeg(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """The base64 payload decodes to a readable JPEG."""
        source = FolderPayloadSource(image_folder)
        payload = source.build_payload(2)

        with Image.open(io.BytesIO(base64.b64decode(payload.base64))) as img:
            assert img.format == "JPEG"

    def test_file_provenance_shape(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """file_provenance() has the base record but no file hash or DPI."""
        source = FolderPayloadSource(image_folder)
        provenance = source.file_provenance()

        assert provenance["source_file"] == str(image_folder)
        assert provenance["image_config_section"] == "api_image_processing"
        assert isinstance(provenance["image_config"], dict)
        assert provenance["pillow_version"]
        assert "source_sha256" not in provenance
        assert "target_dpi" not in provenance

    def test_context_manager(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """FolderPayloadSource works as a context manager."""
        source = FolderPayloadSource(image_folder)
        with source:
            assert len(source) == 3

    def test_null_provider_and_model_default_to_openai(self, tmp_path: Path) -> None:
        """Without provider and model the OpenAI image settings apply."""
        source = FolderPayloadSource(tmp_path, provider=None, model_name=None)
        assert source.model_type == "openai"

    def test_model_name_decides_without_provider(self, tmp_path: Path) -> None:
        """Without a provider the model name selects the image settings."""
        source = FolderPayloadSource(tmp_path, None, "gemini-2.5-flash")
        assert source.model_type == "google"
