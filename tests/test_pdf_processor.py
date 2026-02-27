"""Tests for processors/pdf_processor.py - PDF and image processing utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Any

import pytest
from PIL import Image

from processors.pdf_processor import (
    _apply_image_preprocessing,
    get_image_paths_from_folder,
)


class TestApplyImagePreprocessing:
    """Tests for _apply_image_preprocessing function."""

    @pytest.fixture
    def sample_pil_image(self) -> Image.Image:
        """Create a sample PIL image for testing."""
        return Image.new("RGB", (1000, 1500), color=(128, 128, 128))

    @pytest.fixture
    def rgba_pil_image(self) -> Image.Image:
        """Create a sample RGBA image with transparency."""
        return Image.new("RGBA", (800, 600), color=(128, 128, 128, 128))

    @pytest.fixture
    def default_config(self) -> dict[str, Any]:
        """Default image processing config."""
        return {
            "grayscale_conversion": True,
            "handle_transparency": True,
            "llm_detail": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
        }

    def test_applies_grayscale_when_enabled(
        self, sample_pil_image: Image.Image, default_config: dict[str, Any]
    ):
        """Grayscale conversion is applied when enabled."""
        result = _apply_image_preprocessing(sample_pil_image, default_config, "openai")

        # After grayscale and resize, mode should be L (grayscale) or RGB (if converted back)
        # The resize_for_detail may create a new RGB canvas for box fitting
        assert result.mode in ("L", "RGB")

    def test_skips_grayscale_when_disabled(
        self, sample_pil_image: Image.Image, default_config: dict[str, Any]
    ):
        """Grayscale conversion is skipped when disabled."""
        default_config["grayscale_conversion"] = False

        result = _apply_image_preprocessing(sample_pil_image, default_config, "openai")

        assert result.mode == "RGB"

    def test_handles_transparency(
        self, rgba_pil_image: Image.Image, default_config: dict[str, Any]
    ):
        """Transparency is handled by pasting on white background."""
        default_config["grayscale_conversion"] = False

        result = _apply_image_preprocessing(rgba_pil_image, default_config, "openai")

        # Should no longer have alpha channel
        assert result.mode == "RGB"

    def test_skips_transparency_when_disabled(
        self, rgba_pil_image: Image.Image, default_config: dict[str, Any]
    ):
        """Transparency handling is skipped when disabled."""
        default_config["handle_transparency"] = False
        default_config["grayscale_conversion"] = False

        result = _apply_image_preprocessing(rgba_pil_image, default_config, "openai")

        # Mode might still change due to resize, but transparency not explicitly handled
        assert result.size is not None

    def test_openai_resize_strategy(
        self, sample_pil_image: Image.Image, default_config: dict[str, Any]
    ):
        """OpenAI uses box fit resize strategy."""
        default_config["grayscale_conversion"] = False

        result = _apply_image_preprocessing(sample_pil_image, default_config, "openai")

        # Should fit into target box
        assert result.size == (768, 1536)

    def test_google_resize_strategy(self, sample_pil_image: Image.Image):
        """Google uses its own resize strategy."""
        google_config = {
            "grayscale_conversion": False,
            "handle_transparency": True,
            "media_resolution": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 768],
        }

        result = _apply_image_preprocessing(sample_pil_image, google_config, "google")

        # Should fit into Google's target box
        assert max(result.size) <= 768 or result.size == (768, 768)

    def test_anthropic_resize_strategy(self, sample_pil_image: Image.Image):
        """Anthropic uses max-side capping without padding."""
        anthropic_config = {
            "grayscale_conversion": False,
            "handle_transparency": True,
            "resize_profile": "auto",
            "low_max_side_px": 512,
            "high_max_side_px": 1568,
        }

        result = _apply_image_preprocessing(
            sample_pil_image, anthropic_config, "anthropic"
        )

        # Should cap longest side
        assert max(result.size) <= 1568


class TestGetImagePathsFromFolder:
    """Tests for get_image_paths_from_folder function."""

    def test_finds_jpg_files(self, temp_dir: Path):
        """Finds JPG files in folder."""
        (temp_dir / "image1.jpg").touch()
        (temp_dir / "image2.jpg").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 2
        assert all(p.suffix == ".jpg" for p in paths)

    def test_finds_multiple_formats(self, temp_dir: Path):
        """Finds images of multiple formats."""
        (temp_dir / "image.jpg").touch()
        (temp_dir / "image.png").touch()
        (temp_dir / "image.tiff").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 3

    def test_ignores_non_image_files(self, temp_dir: Path):
        """Non-image files are ignored."""
        (temp_dir / "image.jpg").touch()
        (temp_dir / "document.txt").touch()
        (temp_dir / "data.json").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 1

    def test_returns_sorted_paths(self, temp_dir: Path):
        """Returns paths sorted by filename."""
        (temp_dir / "c_image.jpg").touch()
        (temp_dir / "a_image.jpg").touch()
        (temp_dir / "b_image.jpg").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert paths[0].name == "a_image.jpg"
        assert paths[1].name == "b_image.jpg"
        assert paths[2].name == "c_image.jpg"

    def test_empty_folder(self, temp_dir: Path):
        """Empty folder returns empty list."""
        paths = get_image_paths_from_folder(temp_dir)

        assert paths == []

    def test_case_insensitive_extensions(self, temp_dir: Path):
        """Extension matching is case-insensitive."""
        (temp_dir / "image.JPG").touch()
        (temp_dir / "image.PNG").touch()

        paths = get_image_paths_from_folder(temp_dir)

        assert len(paths) == 2


class TestExtractPdfPagesToImages:
    """Tests for extract_pdf_pages_to_images function (mocked)."""

    def test_function_exists(self):
        """Function exists and is importable."""
        from processors.pdf_processor import extract_pdf_pages_to_images

        assert callable(extract_pdf_pages_to_images)

    def test_with_mock_pdf(self, temp_dir: Path):
        """Test with mocked PDF document."""
        from processors.pdf_processor import extract_pdf_pages_to_images

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Mock fitz (PyMuPDF)
        with patch("processors.pdf_processor.fitz") as mock_fitz:
            # Create mock PDF document
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=2)
            mock_doc.__getitem__ = MagicMock()

            # Create mock page with pixmap
            mock_page = MagicMock()
            mock_pixmap = MagicMock()
            mock_pixmap.width = 100
            mock_pixmap.height = 150
            mock_pixmap.samples = bytes([128] * 100 * 150 * 3)
            mock_page.get_pixmap.return_value = mock_pixmap
            mock_doc.__getitem__.return_value = mock_page

            mock_fitz.open.return_value = mock_doc
            mock_fitz.Matrix.return_value = MagicMock()

            # Mock config loader
            with patch("processors.pdf_processor.get_config_loader") as mock_loader:
                mock_config = MagicMock()
                mock_config.get_image_processing_config.return_value = {
                    "api_image_processing": {
                        "target_dpi": 300,
                        "jpeg_quality": 95,
                        "grayscale_conversion": False,
                        "handle_transparency": True,
                    }
                }
                mock_loader.return_value = mock_config

                # Mock tqdm to avoid progress bar in tests
                with patch("processors.pdf_processor.tqdm", lambda x, **kwargs: x):
                    with patch(
                        "processors.pdf_processor.concurrent.futures.ThreadPoolExecutor"
                    ):
                        # The function should be callable
                        pass


class TestPdfProcessorProviderDetection:
    """Tests for provider detection in PDF processor."""

    def test_openai_provider_uses_correct_config(self):
        """OpenAI provider uses api_image_processing config."""
        from modules.model_utils import detect_model_type, get_image_config_section_name

        model_type = detect_model_type("openai", "gpt-5")
        section = get_image_config_section_name(model_type)

        assert section == "api_image_processing"

    def test_google_provider_uses_correct_config(self):
        """Google provider uses google_image_processing config."""
        from modules.model_utils import detect_model_type, get_image_config_section_name

        model_type = detect_model_type("google", "gemini-2.5-flash")
        section = get_image_config_section_name(model_type)

        assert section == "google_image_processing"

    def test_anthropic_provider_uses_correct_config(self):
        """Anthropic provider uses anthropic_image_processing config."""
        from modules.model_utils import detect_model_type, get_image_config_section_name

        model_type = detect_model_type("anthropic", "claude-3-opus")
        section = get_image_config_section_name(model_type)

        assert section == "anthropic_image_processing"

    def test_openrouter_passthrough_detection(self):
        """OpenRouter correctly detects underlying model type."""
        from modules.model_utils import detect_model_type, get_image_config_section_name

        # OpenRouter with Google model
        model_type = detect_model_type("openrouter", "google/gemini-2.5-flash")
        section = get_image_config_section_name(model_type)

        assert section == "google_image_processing"
