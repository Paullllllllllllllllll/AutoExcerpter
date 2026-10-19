"""Extended tests for `PdfPayloadSource` — provenance and config coverage.

Covers:
- payload byte stability across repeated builds of the same page
- building an arbitrary page without touching its neighbors
- grayscale vs color rendering paths driven by the config dict
- provider-specific config section resolution (Google, Anthropic)
- file_provenance() contents including the source PDF hash
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest
from PIL import Image

from autoexcerpter.common.images import DEFAULT_JPEG_QUALITY
from autoexcerpter.common.native import resized_size
from autoexcerpter.imaging.payload import PdfPayloadSource
from tests.conftest import image_config


@pytest.fixture
def patched_payload_config(
    mock_image_processing: dict[str, Any],
) -> dict[str, Any]:
    """Use the small image sections for the payload sources."""
    return mock_image_processing


class TestPayloadDeterminism:
    """Payload stability and page independence."""

    def test_payload_stable_across_two_builds(
        self, make_pdf: Callable[..., Path], patched_payload_config: dict[str, Any]
    ) -> None:
        """Building the same page twice yields identical bytes."""
        pdf_path = make_pdf("stable.pdf", num_pages=2)
        source = PdfPayloadSource(pdf_path)
        with source:
            first = source.build_payload(0)
            second = source.build_payload(0)

        assert first.base64 == second.base64
        assert "sha256" not in first.provenance

    def test_single_page_build_is_independent(
        self, make_pdf: Callable[..., Path], patched_payload_config: dict[str, Any]
    ) -> None:
        """Building only page index 2 of 3 works without touching the others."""
        pdf_path = make_pdf("three.pdf", num_pages=3)
        source = PdfPayloadSource(pdf_path)
        with source:
            payload = source.build_payload(2)

        assert payload.index == 2
        assert payload.number == 3
        assert payload.image_name == "page_0003.jpg"
        assert payload.provenance["byte_size"] > 0

    def test_distinct_pages_have_distinct_payloads(
        self, make_pdf: Callable[..., Path], patched_payload_config: dict[str, Any]
    ) -> None:
        """Pages with different content produce different payload bytes."""
        pdf_path = make_pdf("distinct.pdf", num_pages=2)
        source = PdfPayloadSource(pdf_path)
        with source:
            first = source.build_payload(0)
            second = source.build_payload(1)

        assert first.base64 != second.base64


class TestConfigPaths:
    """Grayscale/color rendering and provider-specific config sections."""

    def test_grayscale_enabled_produces_l_mode_jpeg(
        self, make_pdf: Callable[..., Path], patched_payload_config: dict[str, Any]
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
        sections = {
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
        pdf_path = make_pdf("color.pdf", num_pages=1)

        with image_config(sections):
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
        sections = {
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
        pdf_path = make_pdf("google.pdf", num_pages=1)

        with image_config(sections):
            source = PdfPayloadSource(
                pdf_path, provider="google", model_name="gemini-2.5-flash"
            )
        with source:
            assert source.target_dpi == 200
            payload = source.build_payload(0)

        # Direct strategy derives the render DPI from the google box-fit profile
        # ([768, 768]), which downscales the 200x300 pt page below target_dpi.
        assert 0 < payload.provenance["effective_dpi"] <= 200
        provenance = source.file_provenance()
        assert provenance["image_config_section"] == "google_image_processing"

    def test_anthropic_provider_reads_anthropic_section(
        self, make_pdf: Callable[..., Path]
    ) -> None:
        """Anthropic provider resolves anthropic_image_processing config."""
        sections = {
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
        pdf_path = make_pdf("anthropic.pdf", num_pages=1)

        with image_config(sections):
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
        pdf_path = make_pdf("defaults.pdf", num_pages=1)

        with image_config({}):
            source = PdfPayloadSource(pdf_path)
            try:
                assert source.target_dpi == 300
                assert source.jpeg_quality == DEFAULT_JPEG_QUALITY
            finally:
                source.close()


class TestFileProvenance:
    """Tests for PdfPayloadSource.file_provenance()."""

    def test_file_provenance_keys_and_hash(
        self, make_pdf: Callable[..., Path], patched_payload_config: dict[str, Any]
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


def _sized_pdf(path: Path, w_pt: float, h_pt: float) -> Path:
    """Write a single-page PDF with the given point dimensions."""
    doc = fitz.open()
    doc.new_page(width=w_pt, height=h_pt)
    doc.save(path)
    doc.close()
    return path


class TestRenderStrategy:
    """Target-derived render DPI ('direct') vs fixed target DPI ('supersample')."""

    def test_anthropic_a4_direct_derives_dpi(self, tmp_path: Path) -> None:
        """A4 renders within both Anthropic's edge and patch budget."""
        pdf_path = _sized_pdf(tmp_path / "a4.pdf", 595.0, 842.0)
        sections = {
            "anthropic_image_processing": {
                "target_dpi": 300,
                "jpeg_quality": 95,
                "grayscale_conversion": True,
                "handle_transparency": True,
                "resize_profile": "auto",
                "low_max_side_px": 512,
                "high_max_side_px": 2576,
            }
        }
        with image_config(sections):
            source = PdfPayloadSource(
                pdf_path, provider="anthropic", model_name="claude-opus-4-8"
            )
        with source:
            payload = source.build_payload(0)

        size = resized_size(2480, 3509, 2576, 4784)
        expected = 300.0 * min(size[0] / 2480, size[1] / 3509)
        assert payload.provenance["effective_dpi"] == pytest.approx(expected, abs=1.0)
        assert payload.provenance["effective_dpi"] < 300
        # After render + the (near-no-op) resize, the long edge sits at the cap.
        assert max(payload.provenance["width"], payload.provenance["height"]) <= 2576
        assert (
            math.ceil(payload.provenance["width"] / 28)
            * math.ceil(payload.provenance["height"] / 28)
            <= 4784
        )

    def test_original_within_caps_renders_at_target_dpi(self, tmp_path: Path) -> None:
        """'original' profile whose target_dpi render fits the caps stays at DPI."""
        pdf_path = _sized_pdf(tmp_path / "orig.pdf", 200.0, 300.0)
        sections = {
            "api_image_processing": {
                "target_dpi": 300,
                "jpeg_quality": 95,
                "grayscale_conversion": True,
                "handle_transparency": True,
                "llm_detail": "original",
                "resize_profile": "high",
                "low_max_side_px": 512,
                "high_target_box": [768, 1536],
                "original_max_side_px": 6000,
                "original_max_pixels": 10_240_000,
            }
        }
        with image_config(sections, image_size="original"):
            source = PdfPayloadSource(pdf_path, model_name="gpt-6-astra")
        with source:
            payload = source.build_payload(0)

        assert payload.provenance["effective_dpi"] == 300

    def test_supersample_renders_at_target_dpi(self, tmp_path: Path) -> None:
        """'supersample' always renders at target_dpi."""
        pdf_path = _sized_pdf(tmp_path / "ss.pdf", 200.0, 300.0)
        sections = {
            "render_strategy": "supersample",
            "api_image_processing": {
                "target_dpi": 300,
                "jpeg_quality": 95,
                "grayscale_conversion": True,
                "handle_transparency": True,
                "llm_detail": "high",
                "resize_profile": "high",
                "low_max_side_px": 512,
                "high_target_box": [768, 1536],
            },
        }
        with image_config(sections):
            source = PdfPayloadSource(pdf_path)
        assert source.render_strategy == "supersample"
        with source:
            payload = source.build_payload(0)

        assert payload.provenance["effective_dpi"] == 300

    def test_direct_and_supersample_share_final_dimensions(
        self, tmp_path: Path
    ) -> None:
        """Box-fit final dimensions match across strategies; bytes differ."""
        pdf_path = _sized_pdf(tmp_path / "cmp.pdf", 200.0, 300.0)
        base_section = {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "llm_detail": "high",
            "resize_profile": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
        }

        with image_config({"api_image_processing": dict(base_section)}):
            direct = PdfPayloadSource(pdf_path)
        with direct:
            direct_payload = direct.build_payload(0)
        with image_config(
            {
                "render_strategy": "supersample",
                "api_image_processing": dict(base_section),
            }
        ):
            supersample = PdfPayloadSource(pdf_path)
        with supersample:
            ss_payload = supersample.build_payload(0)

        # Same padded box, different render path => different bytes.
        assert direct_payload.provenance["width"] == ss_payload.provenance["width"]
        assert direct_payload.provenance["height"] == ss_payload.provenance["height"]
        assert direct_payload.provenance["effective_dpi"] < 300
        assert ss_payload.provenance["effective_dpi"] == 300
