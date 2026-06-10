"""Extended tests for `PdfPayloadSource` — provenance and config coverage.

Covers:
- payload sha256 stability across repeated builds of the same page
- building an arbitrary page without touching its neighbors
- grayscale vs color rendering paths driven by the config dict
- provider-specific config section resolution (Google, Anthropic)
- file_provenance() contents including the source PDF hash
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imaging.payload import PdfPayloadSource


def _loader_with(sections: dict[str, dict[str, Any]]) -> MagicMock:
    """Build a mock ConfigLoader returning the given image-processing sections."""
    loader = MagicMock()
    loader.get_image_processing_config.return_value = sections
    return loader


@pytest.fixture
def patched_payload_config(
    mock_config_loader: MagicMock,
) -> Generator[MagicMock]:
    """Patch the payload module's config loader with the mock configuration."""
    with patch("imaging.payload.get_config_loader", return_value=mock_config_loader):
        yield mock_config_loader


class TestPayloadDeterminism:
    """Hash stability and page independence."""

    def test_sha256_stable_across_two_builds(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """Building the same page twice yields identical bytes and hashes."""
        pdf_path = make_pdf("stable.pdf", num_pages=2)
        source = PdfPayloadSource(pdf_path)
        with source:
            first = source.build_payload(0)
            second = source.build_payload(0)

        assert first.base64 == second.base64
        assert first.provenance["sha256"] == second.provenance["sha256"]

    def test_single_page_build_is_independent(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """Building only page index 2 of 3 works without touching the others."""
        pdf_path = make_pdf("three.pdf", num_pages=3)
        source = PdfPayloadSource(pdf_path)
        with source:
            payload = source.build_payload(2)

        assert payload.original_input_order_index == 2
        assert payload.sequence_number == 3
        assert payload.image_name == "page_0003.jpg"
        assert payload.provenance["byte_size"] > 0

    def test_distinct_pages_have_distinct_hashes(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """Pages with different content produce different payload hashes."""
        pdf_path = make_pdf("distinct.pdf", num_pages=2)
        source = PdfPayloadSource(pdf_path)
        with source:
            first = source.build_payload(0)
            second = source.build_payload(1)

        assert first.provenance["sha256"] != second.provenance["sha256"]


class TestConfigPaths:
    """Grayscale/color rendering and provider-specific config sections."""

    def test_grayscale_enabled_produces_l_mode_jpeg(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """With grayscale_conversion enabled, the encoded JPEG is grayscale."""
        pdf_path = make_pdf("gray.pdf", num_pages=1)
        source = PdfPayloadSource(pdf_path)
        with source:
            payload = source.build_payload(0)

        with Image.open(io.BytesIO(base64.b64decode(payload.base64))) as img:
            assert img.mode == "L"

    def test_grayscale_disabled_produces_rgb_jpeg(
        self, make_pdf: Callable[..., Path]
    ) -> None:
        """With grayscale_conversion disabled, the encoded JPEG stays RGB."""
        loader = _loader_with(
            {
                "api_image_processing": {
                    "target_dpi": 150,
                    "jpeg_quality": 90,
                    "grayscale_conversion": False,
                    "handle_transparency": True,
                    "llm_detail": "high",
                    "low_max_side_px": 512,
                    "high_target_box": [768, 1536],
                }
            }
        )
        pdf_path = make_pdf("color.pdf", num_pages=1)

        with patch("imaging.payload.get_config_loader", return_value=loader):
            source = PdfPayloadSource(pdf_path)
        with source:
            payload = source.build_payload(0)

        with Image.open(io.BytesIO(base64.b64decode(payload.base64))) as img:
            assert img.mode == "RGB"
        assert payload.provenance["effective_dpi"] == 150

    def test_google_provider_reads_google_section(
        self, make_pdf: Callable[..., Path]
    ) -> None:
        """Google provider resolves google_image_processing config."""
        loader = _loader_with(
            {
                "google_image_processing": {
                    "target_dpi": 200,
                    "jpeg_quality": 90,
                    "grayscale_conversion": True,
                    "handle_transparency": True,
                    "media_resolution": "high",
                    "low_max_side_px": 512,
                    "high_target_box": [768, 768],
                }
            }
        )
        pdf_path = make_pdf("google.pdf", num_pages=1)

        with patch("imaging.payload.get_config_loader", return_value=loader):
            source = PdfPayloadSource(
                pdf_path, provider="google", model_name="gemini-2.5-flash"
            )
        with source:
            assert source.target_dpi == 200
            payload = source.build_payload(0)

        assert payload.provenance["effective_dpi"] == 200
        provenance = source.file_provenance()
        assert provenance["image_config_section"] == "google_image_processing"

    def test_anthropic_provider_reads_anthropic_section(
        self, make_pdf: Callable[..., Path]
    ) -> None:
        """Anthropic provider resolves anthropic_image_processing config."""
        loader = _loader_with(
            {
                "anthropic_image_processing": {
                    "target_dpi": 300,
                    "jpeg_quality": 95,
                    "grayscale_conversion": True,
                    "handle_transparency": True,
                    "resize_profile": "auto",
                    "low_max_side_px": 512,
                    "high_max_side_px": 1568,
                }
            }
        )
        pdf_path = make_pdf("anthropic.pdf", num_pages=1)

        with patch("imaging.payload.get_config_loader", return_value=loader):
            source = PdfPayloadSource(
                pdf_path, provider="anthropic", model_name="claude-3-opus"
            )
        with source:
            payload = source.build_payload(0)
            provenance = source.file_provenance()

        assert max(payload.provenance["width"], payload.provenance["height"]) <= 1568
        assert provenance["image_config_section"] == "anthropic_image_processing"

    def test_missing_section_falls_back_to_defaults(
        self, make_pdf: Callable[..., Path]
    ) -> None:
        """An absent config section yields default DPI and JPEG quality."""
        from config.constants import DEFAULT_JPEG_QUALITY, DEFAULT_TARGET_DPI

        loader = _loader_with({})
        pdf_path = make_pdf("defaults.pdf", num_pages=1)

        with patch("imaging.payload.get_config_loader", return_value=loader):
            source = PdfPayloadSource(pdf_path)
            try:
                assert source.target_dpi == DEFAULT_TARGET_DPI
                assert source.jpeg_quality == DEFAULT_JPEG_QUALITY
            finally:
                source.close()


class TestFileProvenance:
    """Tests for PdfPayloadSource.file_provenance()."""

    def test_file_provenance_keys_and_hash(
        self, make_pdf: Callable[..., Path], patched_payload_config: MagicMock
    ) -> None:
        """file_provenance includes the correct PDF hash and config record."""
        pdf_path = make_pdf("prov.pdf", num_pages=2)
        source = PdfPayloadSource(pdf_path)
        with source:
            provenance = source.file_provenance()

        expected_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        assert provenance["source_file"] == str(pdf_path)
        assert provenance["source_sha256"] == expected_sha
        assert provenance["target_dpi"] == 300
        assert provenance["image_config_section"] == "api_image_processing"
        assert isinstance(provenance["image_config"], dict)
        assert provenance["pymupdf_version"]
        assert provenance["pillow_version"]
