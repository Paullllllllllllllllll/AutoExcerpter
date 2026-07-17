"""Tests for the PDF side of the streaming imaging pipeline.

Covers basic `PdfPayloadSource` behavior (length, naming contract, payload
construction, lifecycle) plus `get_image_paths_from_folder` and the
provider-to-config-section mapping. Extended provenance and config-path
coverage lives in test_imaging_pdf_extended.py.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imaging.payload import PdfPayloadSource
from imaging.pdf import get_image_paths_from_folder


@pytest.fixture
def patched_payload_config(
    mock_config_loader: MagicMock,
) -> Generator[MagicMock]:
    """Patch the payload module's config loader with the mock configuration."""
    with patch("imaging.payload.get_config_loader", return_value=mock_config_loader):
        yield mock_config_loader


class TestPdfPayloadSourceBasics:
    """Basic contract tests for PdfPayloadSource."""

    def test_len_matches_page_count(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """__len__ returns the number of pages in the PDF."""
        pdf_path = make_pdf("three.pdf", num_pages=3)
        source = PdfPayloadSource(pdf_path)
        with source:
            assert len(source) == 3

    def test_image_name_naming_contract(self) -> None:
        """image_name follows the legacy page_NNNN.jpg naming contract."""
        assert PdfPayloadSource.image_name(0) == "page_0001.jpg"
        assert PdfPayloadSource.image_name(1) == "page_0002.jpg"
        assert PdfPayloadSource.image_name(41) == "page_0042.jpg"

    def test_build_payload_indices_and_names(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """build_payload returns absolute indices and 1-based sequence numbers."""
        pdf_path = make_pdf("three.pdf", num_pages=3)
        source = PdfPayloadSource(pdf_path)
        with source:
            payload = source.build_payload(1)

        assert payload.original_input_order_index == 1
        assert payload.sequence_number == 2
        assert payload.page_index == 1
        assert payload.image_name == "page_0002.jpg"
        assert payload.source_file == str(pdf_path)

    def test_build_payload_provenance_populated(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """Provenance records hash, dimensions, byte size, and effective DPI.

        The default (direct) strategy with an OpenAI box-fit profile derives a
        render DPI below the 300 ceiling for a 200x300 pt page (the box-fit
        downscales), so effective_dpi is the derived value, not target_dpi.
        """
        pdf_path = make_pdf("one.pdf", num_pages=1)
        source = PdfPayloadSource(pdf_path)
        with source:
            payload = source.build_payload(0)

        decoded = base64.b64decode(payload.base64)
        provenance = payload.provenance
        assert provenance["sha256"] == hashlib.sha256(decoded).hexdigest()
        assert provenance["byte_size"] == len(decoded)
        assert provenance["width"] > 0
        assert provenance["height"] > 0
        # 200 pt wide page, box width 768: 300 * 768 / (200 * 300 / 72) = 276.48.
        assert provenance["effective_dpi"] == pytest.approx(276.48, abs=0.5)
        assert 0 < provenance["effective_dpi"] <= 300

    def test_close_then_build_raises(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """build_payload after close() raises RuntimeError."""
        pdf_path = make_pdf("one.pdf", num_pages=1)
        source = PdfPayloadSource(pdf_path)
        source.close()

        with pytest.raises(RuntimeError, match="closed"):
            source.build_payload(0)

    def test_close_is_idempotent(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """Calling close() twice does not raise."""
        pdf_path = make_pdf("one.pdf", num_pages=1)
        source = PdfPayloadSource(pdf_path)
        source.close()
        source.close()
        assert len(source) == 0

    def test_context_manager_closes_source(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """Leaving the context manager closes the document."""
        pdf_path = make_pdf("one.pdf", num_pages=1)
        source = PdfPayloadSource(pdf_path)
        with source:
            assert len(source) == 1

        with pytest.raises(RuntimeError, match="closed"):
            source.build_payload(0)

    def test_corrupt_pdf_raises_in_constructor(
        self, tmp_path: Path, patched_payload_config: MagicMock
    ) -> None:
        """An invalid PDF file makes the constructor raise."""
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_bytes(b"this is definitely not a PDF")

        with pytest.raises((RuntimeError, ValueError)):
            PdfPayloadSource(bad_pdf)


class TestGetImagePathsFromFolder:
    """Tests for get_image_paths_from_folder function."""

    def test_finds_jpg_files(self, temp_dir: Path) -> None:
        """Finds JPG files in folder."""
        (temp_dir / "image1.jpg").touch()
        (temp_dir / "image2.jpg").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 2
        assert all(p.suffix == ".jpg" for p in paths)

    def test_finds_multiple_formats(self, temp_dir: Path) -> None:
        """Finds images of multiple formats."""
        (temp_dir / "image.jpg").touch()
        (temp_dir / "image.png").touch()
        (temp_dir / "image.tiff").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 3

    def test_ignores_non_image_files(self, temp_dir: Path) -> None:
        """Non-image files are ignored."""
        (temp_dir / "image.jpg").touch()
        (temp_dir / "document.txt").touch()
        (temp_dir / "data.json").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 1

    def test_returns_sorted_paths(self, temp_dir: Path) -> None:
        """Returns paths sorted by filename."""
        (temp_dir / "c_image.jpg").touch()
        (temp_dir / "a_image.jpg").touch()
        (temp_dir / "b_image.jpg").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert paths[0].name == "a_image.jpg"
        assert paths[1].name == "b_image.jpg"
        assert paths[2].name == "c_image.jpg"

    def test_natural_sort_of_unpadded_numbers(self, temp_dir: Path) -> None:
        """Unpadded numeric filenames sort numerically, not lexicographically."""
        for i in (1, 2, 10, 11):
            (temp_dir / f"page_{i}.jpg").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert [p.name for p in paths] == [
            "page_1.jpg",
            "page_2.jpg",
            "page_10.jpg",
            "page_11.jpg",
        ]

    def test_empty_folder(self, temp_dir: Path) -> None:
        """Empty folder returns empty list."""
        paths = get_image_paths_from_folder(temp_dir)

        assert paths == []

    def test_case_insensitive_extensions(self, temp_dir: Path) -> None:
        """Extension matching is case-insensitive."""
        (temp_dir / "image.JPG").touch()
        (temp_dir / "image.PNG").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 2


class TestPdfProcessorProviderDetection:
    """Tests for provider detection used by the payload sources."""

    def test_openai_provider_uses_correct_config(self) -> None:
        """OpenAI provider uses api_image_processing config."""
        from imaging._provider import detect_model_type, get_image_config_section_name

        model_type = detect_model_type("openai", "gpt-5")
        section = get_image_config_section_name(model_type)

        assert section == "api_image_processing"

    def test_google_provider_uses_correct_config(self) -> None:
        """Google provider uses google_image_processing config."""
        from imaging._provider import detect_model_type, get_image_config_section_name

        model_type = detect_model_type("google", "gemini-2.5-flash")
        section = get_image_config_section_name(model_type)

        assert section == "google_image_processing"

    def test_anthropic_provider_uses_correct_config(self) -> None:
        """Anthropic provider uses anthropic_image_processing config."""
        from imaging._provider import detect_model_type, get_image_config_section_name

        model_type = detect_model_type("anthropic", "claude-3-opus")
        section = get_image_config_section_name(model_type)

        assert section == "anthropic_image_processing"

    def test_openrouter_passthrough_detection(self) -> None:
        """OpenRouter correctly detects underlying model type."""
        from imaging._provider import detect_model_type, get_image_config_section_name

        model_type = detect_model_type("openrouter", "google/gemini-2.5-flash")
        section = get_image_config_section_name(model_type)

        assert section == "google_image_processing"
