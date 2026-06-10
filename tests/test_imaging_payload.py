"""Tests for imaging/payload.py - FolderPayloadSource and module helpers.

`PdfPayloadSource` coverage lives in test_imaging_pdf.py and
test_imaging_pdf_extended.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imaging.payload import (
    FolderPayloadSource,
    PagePayload,
    _encode_jpeg,
    _extract_sequence_number,
    _sha256_of_file,
)


@pytest.fixture
def patched_payload_config(
    mock_config_loader: MagicMock,
) -> Generator[MagicMock]:
    """Patch the payload module's config loader with the mock configuration."""
    with patch("imaging.payload.get_config_loader", return_value=mock_config_loader):
        yield mock_config_loader


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
    """Tests for the module-level _extract_sequence_number helper."""

    def test_page_pattern(self) -> None:
        assert _extract_sequence_number(Path("page_0001.jpg")) == 1

    def test_page_pattern_long(self) -> None:
        assert _extract_sequence_number(Path("page_0012.jpg")) == 12

    def test_dash_separated_number(self) -> None:
        assert _extract_sequence_number(Path("scan-7.jpg")) == 7

    def test_no_number_returns_zero(self) -> None:
        assert _extract_sequence_number(Path("noNumber.jpg")) == 0

    def test_multiple_numbers_takes_last(self) -> None:
        assert _extract_sequence_number(Path("doc_2024_page_005.jpg")) == 5

    def test_number_only_stem(self) -> None:
        assert _extract_sequence_number(Path("0010.jpg")) == 10

    def test_mixed_text_and_numbers(self) -> None:
        assert _extract_sequence_number(Path("vol2_ch3_page_15.png")) == 15

    def test_full_path(self) -> None:
        assert _extract_sequence_number(Path("/deep/path/scan_page_0123.tiff")) == 123


class TestModuleHelpers:
    """Tests for _sha256_of_file and _encode_jpeg."""

    def test_sha256_of_file_matches_hashlib(self, tmp_path: Path) -> None:
        """Streaming file hash matches a direct hashlib digest."""
        target = tmp_path / "data.bin"
        target.write_bytes(b"some binary content" * 100)

        expected = hashlib.sha256(target.read_bytes()).hexdigest()
        assert _sha256_of_file(target) == expected

    def test_encode_jpeg_returns_bytes_and_dimensions(self) -> None:
        """_encode_jpeg returns valid JPEG bytes plus width and height."""
        img = Image.new("RGB", (40, 30), color=(10, 20, 30))
        jpeg_bytes, width, height = _encode_jpeg(img, 90)

        assert jpeg_bytes[:2] == b"\xff\xd8"
        assert (width, height) == (40, 30)

    def test_encode_jpeg_converts_rgba(self) -> None:
        """RGBA images are converted before JPEG encoding."""
        img = Image.new("RGBA", (20, 20), color=(10, 20, 30, 128))
        jpeg_bytes, width, height = _encode_jpeg(img, 90)

        assert jpeg_bytes[:2] == b"\xff\xd8"
        assert (width, height) == (20, 20)


class TestPagePayload:
    """Tests for the PagePayload dataclass."""

    def test_frozen(self) -> None:
        """PagePayload instances are immutable."""
        payload = PagePayload(
            base64="abc",
            image_name="page_0001.jpg",
            sequence_number=1,
            original_input_order_index=0,
            provenance={},
            source_file="doc.pdf",
        )
        with pytest.raises(AttributeError):
            payload.base64 = "xyz"  # type: ignore[misc]

    def test_page_index_defaults_to_none(self) -> None:
        """page_index is optional and defaults to None."""
        payload = PagePayload(
            base64="abc",
            image_name="img.jpg",
            sequence_number=1,
            original_input_order_index=0,
            provenance={},
            source_file="img.jpg",
        )
        assert payload.page_index is None


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
        assert payload.original_input_order_index == 0
        assert payload.page_index is None
        assert payload.source_file == str(image_folder / "page_0001.jpg")

    def test_sequence_numbers_from_filenames(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """Sequence numbers are parsed from the image filenames."""
        source = FolderPayloadSource(image_folder)

        assert source.build_payload(0).sequence_number == 1
        assert source.build_payload(1).sequence_number == 2
        assert source.build_payload(2).sequence_number == 7

    def test_provenance_fields(
        self, image_folder: Path, patched_payload_config: MagicMock
    ) -> None:
        """Provenance includes source_sha256 of the file and no effective_dpi."""
        source = FolderPayloadSource(image_folder)
        payload = source.build_payload(0)

        decoded = base64.b64decode(payload.base64)
        source_bytes = (image_folder / "page_0001.jpg").read_bytes()
        provenance = payload.provenance

        assert provenance["sha256"] == hashlib.sha256(decoded).hexdigest()
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
