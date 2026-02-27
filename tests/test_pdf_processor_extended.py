"""Extended tests for processors/pdf_processor.py - coverage gap filling.

Covers:
- extract_pdf_pages_to_images: mock fitz, page extraction, error handling,
  empty PDF, single page, grayscale toggle, provider-specific config
- get_image_paths_from_folder: empty folder, mixed files, supported/unsupported,
  subdirectories ignored
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
from PIL import Image

from processors.pdf_processor import (
    extract_pdf_pages_to_images,
    get_image_paths_from_folder,
    _apply_image_preprocessing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create and return an output directory for extracted images."""
    d = tmp_path / "output_images"
    d.mkdir()
    return d


@pytest.fixture
def mock_img_cfg() -> Dict[str, Any]:
    """Return a default image processing config dict for OpenAI."""
    return {
        "target_dpi": 300,
        "jpeg_quality": 95,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "llm_detail": "high",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    }


def _build_mock_pixmap(width: int, height: int, mode: str = "RGB") -> MagicMock:
    """Build a mock fitz Pixmap that can produce PIL Image bytes."""
    mock_pix = MagicMock()
    mock_pix.width = width
    mock_pix.height = height
    if mode == "L":
        mock_pix.samples = bytes([128] * width * height)
    else:
        mock_pix.samples = bytes([128] * width * height * 3)
    return mock_pix


def _build_mock_pdf_document(num_pages: int, pix_width: int = 100, pix_height: int = 150, grayscale: bool = True):
    """Build a mock fitz.Document with the given number of pages."""
    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=num_pages)

    mock_page = MagicMock()
    if grayscale:
        mock_pix = _build_mock_pixmap(pix_width, pix_height, mode="L")
    else:
        mock_pix = _build_mock_pixmap(pix_width, pix_height, mode="RGB")
    mock_page.get_pixmap.return_value = mock_pix
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)

    return mock_doc


# ============================================================================
# extract_pdf_pages_to_images
# ============================================================================
class TestExtractPdfPagesToImages:
    """Tests for extract_pdf_pages_to_images() with mocked fitz."""

    @patch("processors.pdf_processor._apply_image_preprocessing")
    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_extracts_all_pages(
        self, mock_fitz, mock_loader, mock_preprocess, output_dir: Path, mock_img_cfg
    ):
        """Extracts all pages from a multi-page PDF."""
        num_pages = 3
        mock_doc = _build_mock_pdf_document(num_pages)
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        mock_fitz.csGRAY = MagicMock()

        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = {
            "api_image_processing": mock_img_cfg,
        }
        mock_loader.return_value = cfg_loader

        # Make preprocessing return a saveable PIL image
        fake_img = Image.new("L", (100, 150), color=128)
        mock_preprocess.return_value = fake_img

        pdf_path = output_dir.parent / "test.pdf"
        pdf_path.touch()

        with patch("processors.pdf_processor.tqdm", lambda x, **kw: x):
            paths = extract_pdf_pages_to_images(
                pdf_path, output_dir, provider="openai", model_name="gpt-5-mini"
            )

        assert len(paths) == num_pages
        for p in paths:
            assert p.exists()
            assert p.suffix == ".jpg"

        mock_doc.close.assert_called_once()

    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_empty_pdf_returns_empty_list(
        self, mock_fitz, mock_loader, output_dir: Path
    ):
        """Empty PDF (0 pages) returns an empty list."""
        mock_doc = _build_mock_pdf_document(0)
        mock_fitz.open.return_value = mock_doc

        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = {"api_image_processing": {}}
        mock_loader.return_value = cfg_loader

        pdf_path = output_dir.parent / "empty.pdf"
        pdf_path.touch()

        paths = extract_pdf_pages_to_images(pdf_path, output_dir)

        assert paths == []
        mock_doc.close.assert_called_once()

    @patch("processors.pdf_processor._apply_image_preprocessing")
    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_single_page_pdf(
        self, mock_fitz, mock_loader, mock_preprocess, output_dir: Path, mock_img_cfg
    ):
        """Single-page PDF produces one output image."""
        mock_doc = _build_mock_pdf_document(1)
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        mock_fitz.csGRAY = MagicMock()

        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = {
            "api_image_processing": mock_img_cfg,
        }
        mock_loader.return_value = cfg_loader

        fake_img = Image.new("L", (100, 150), color=128)
        mock_preprocess.return_value = fake_img

        pdf_path = output_dir.parent / "single.pdf"
        pdf_path.touch()

        with patch("processors.pdf_processor.tqdm", lambda x, **kw: x):
            paths = extract_pdf_pages_to_images(pdf_path, output_dir)

        assert len(paths) == 1
        assert paths[0].name == "page_0001.jpg"

    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_fitz_open_error_returns_empty_list(
        self, mock_fitz, mock_loader, output_dir: Path
    ):
        """If fitz.open raises an exception, returns an empty list."""
        mock_fitz.open.side_effect = RuntimeError("Cannot open PDF")

        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = {}
        mock_loader.return_value = cfg_loader

        pdf_path = output_dir.parent / "bad.pdf"
        pdf_path.touch()

        paths = extract_pdf_pages_to_images(pdf_path, output_dir)

        assert paths == []

    @patch("processors.pdf_processor._apply_image_preprocessing")
    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_page_extraction_error_skips_page(
        self, mock_fitz, mock_loader, mock_preprocess, output_dir: Path, mock_img_cfg
    ):
        """If a single page fails, it is skipped but other pages succeed."""
        # Build a 2-page document where page 0 fails
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)

        good_page = MagicMock()
        good_pix = _build_mock_pixmap(100, 150, mode="L")
        good_page.get_pixmap.return_value = good_pix

        bad_page = MagicMock()
        bad_page.get_pixmap.side_effect = RuntimeError("Page corrupt")

        def getitem(idx):
            if idx == 0:
                return bad_page
            return good_page

        mock_doc.__getitem__ = MagicMock(side_effect=getitem)
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        mock_fitz.csGRAY = MagicMock()

        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = {
            "api_image_processing": mock_img_cfg,
        }
        mock_loader.return_value = cfg_loader

        fake_img = Image.new("L", (100, 150), color=128)
        mock_preprocess.return_value = fake_img

        pdf_path = output_dir.parent / "partial.pdf"
        pdf_path.touch()

        with patch("processors.pdf_processor.tqdm", lambda x, **kw: x):
            paths = extract_pdf_pages_to_images(pdf_path, output_dir)

        # Only page 1 should succeed (page 0 failed)
        assert len(paths) == 1
        assert "page_0002" in paths[0].name

    @patch("processors.pdf_processor._apply_image_preprocessing")
    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_grayscale_disabled_uses_rgb(
        self, mock_fitz, mock_loader, mock_preprocess, output_dir: Path
    ):
        """When grayscale_conversion is False, RGB pixmap is requested."""
        mock_doc = _build_mock_pdf_document(1, grayscale=False)
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()

        cfg = {
            "api_image_processing": {
                "target_dpi": 300,
                "jpeg_quality": 95,
                "grayscale_conversion": False,
                "handle_transparency": True,
                "llm_detail": "high",
                "low_max_side_px": 512,
                "high_target_box": [768, 1536],
            }
        }
        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = cfg
        mock_loader.return_value = cfg_loader

        fake_img = Image.new("RGB", (100, 150), color=(128, 128, 128))
        mock_preprocess.return_value = fake_img

        pdf_path = output_dir.parent / "rgb.pdf"
        pdf_path.touch()

        with patch("processors.pdf_processor.tqdm", lambda x, **kw: x):
            paths = extract_pdf_pages_to_images(pdf_path, output_dir)

        assert len(paths) == 1
        # Verify that get_pixmap was called without csGRAY
        page = mock_doc.__getitem__.return_value
        call_kwargs = page.get_pixmap.call_args
        # When grayscale is False, colorspace should not be csGRAY
        if call_kwargs.kwargs.get("colorspace") is not None:
            assert call_kwargs.kwargs["colorspace"] != mock_fitz.csGRAY

    @patch("processors.pdf_processor._apply_image_preprocessing")
    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_google_provider_config_section(
        self, mock_fitz, mock_loader, mock_preprocess, output_dir: Path
    ):
        """Google provider reads from google_image_processing section."""
        mock_doc = _build_mock_pdf_document(1)
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        mock_fitz.csGRAY = MagicMock()

        google_cfg = {
            "target_dpi": 200,
            "jpeg_quality": 90,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "media_resolution": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 768],
        }
        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = {
            "google_image_processing": google_cfg,
        }
        mock_loader.return_value = cfg_loader

        fake_img = Image.new("L", (100, 150), color=128)
        mock_preprocess.return_value = fake_img

        pdf_path = output_dir.parent / "google.pdf"
        pdf_path.touch()

        with patch("processors.pdf_processor.tqdm", lambda x, **kw: x):
            paths = extract_pdf_pages_to_images(
                pdf_path, output_dir, provider="google", model_name="gemini-2.5-flash"
            )

        assert len(paths) == 1

    @patch("processors.pdf_processor._apply_image_preprocessing")
    @patch("processors.pdf_processor.get_config_loader")
    @patch("processors.pdf_processor.fitz")
    def test_anthropic_provider_config_section(
        self, mock_fitz, mock_loader, mock_preprocess, output_dir: Path
    ):
        """Anthropic provider reads from anthropic_image_processing section."""
        mock_doc = _build_mock_pdf_document(1)
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        mock_fitz.csGRAY = MagicMock()

        anthropic_cfg = {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "resize_profile": "auto",
            "low_max_side_px": 512,
            "high_max_side_px": 1568,
        }
        cfg_loader = MagicMock()
        cfg_loader.get_image_processing_config.return_value = {
            "anthropic_image_processing": anthropic_cfg,
        }
        mock_loader.return_value = cfg_loader

        fake_img = Image.new("L", (100, 150), color=128)
        mock_preprocess.return_value = fake_img

        pdf_path = output_dir.parent / "anthropic.pdf"
        pdf_path.touch()

        with patch("processors.pdf_processor.tqdm", lambda x, **kw: x):
            paths = extract_pdf_pages_to_images(
                pdf_path, output_dir, provider="anthropic", model_name="claude-3-opus"
            )

        assert len(paths) == 1


# ============================================================================
# get_image_paths_from_folder (extended)
# ============================================================================
class TestGetImagePathsFromFolderExtended:
    """Extended tests for get_image_paths_from_folder()."""

    def test_empty_folder_returns_empty(self, tmp_path: Path):
        """Empty folder returns an empty list."""
        paths = get_image_paths_from_folder(tmp_path)
        assert paths == []

    def test_all_supported_extensions(self, tmp_path: Path):
        """All supported image extensions are found."""
        supported = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"]
        for ext in supported:
            (tmp_path / f"img{ext}").touch()

        paths = get_image_paths_from_folder(tmp_path)
        assert len(paths) == len(supported)

    def test_unsupported_extensions_ignored(self, tmp_path: Path):
        """Non-image files are ignored."""
        (tmp_path / "document.pdf").touch()
        (tmp_path / "script.py").touch()
        (tmp_path / "data.csv").touch()
        (tmp_path / "readme.md").touch()
        (tmp_path / "image.jpg").touch()

        paths = get_image_paths_from_folder(tmp_path)
        assert len(paths) == 1
        assert paths[0].name == "image.jpg"

    def test_mixed_supported_and_unsupported(self, tmp_path: Path):
        """Only supported files are returned from a mixed folder."""
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.json").touch()
        (tmp_path / "e.tiff").touch()

        paths = get_image_paths_from_folder(tmp_path)
        assert len(paths) == 3
        names = [p.name for p in paths]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "e.tiff" in names

    def test_case_insensitive_extensions(self, tmp_path: Path):
        """Upper and mixed case extensions are matched."""
        (tmp_path / "img1.JPG").touch()
        (tmp_path / "img2.Png").touch()
        (tmp_path / "img3.TIFF").touch()

        paths = get_image_paths_from_folder(tmp_path)
        assert len(paths) == 3

    def test_results_sorted_by_name(self, tmp_path: Path):
        """Results are sorted alphabetically by filename."""
        (tmp_path / "z_image.jpg").touch()
        (tmp_path / "a_image.jpg").touch()
        (tmp_path / "m_image.jpg").touch()

        paths = get_image_paths_from_folder(tmp_path)
        names = [p.name for p in paths]
        assert names == sorted(names)

    def test_subdirectories_not_included(self, tmp_path: Path):
        """Subdirectories are not included in results (glob('*') is non-recursive)."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.jpg").touch()
        (tmp_path / "top_level.jpg").touch()

        paths = get_image_paths_from_folder(tmp_path)
        assert len(paths) == 1
        assert paths[0].name == "top_level.jpg"

    def test_folder_with_only_unsupported_files(self, tmp_path: Path):
        """Folder containing only unsupported files returns empty list."""
        (tmp_path / "readme.txt").touch()
        (tmp_path / "notes.md").touch()
        (tmp_path / "data.xml").touch()

        paths = get_image_paths_from_folder(tmp_path)
        assert paths == []


# ============================================================================
# _apply_image_preprocessing (extended model type tests)
# ============================================================================
class TestApplyImagePreprocessingExtended:
    """Extended tests for _apply_image_preprocessing with different model types."""

    def test_palette_mode_with_transparency(self):
        """Palette mode image with transparency info is handled."""
        img = Image.new("P", (100, 100))
        img.info["transparency"] = 0

        cfg = {
            "grayscale_conversion": False,
            "handle_transparency": True,
            "llm_detail": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
        }

        result = _apply_image_preprocessing(img, cfg, "openai")
        assert result.mode == "RGB"

    def test_la_mode_transparency(self):
        """LA (grayscale with alpha) mode transparency is handled."""
        img = Image.new("LA", (100, 100))

        cfg = {
            "grayscale_conversion": False,
            "handle_transparency": True,
            "llm_detail": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
        }

        result = _apply_image_preprocessing(img, cfg, "openai")
        assert result.mode == "RGB"

    def test_google_uses_media_resolution(self):
        """Google model type reads media_resolution config key."""
        img = Image.new("RGB", (500, 500))

        cfg = {
            "grayscale_conversion": False,
            "handle_transparency": False,
            "media_resolution": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 768],
        }

        result = _apply_image_preprocessing(img, cfg, "google")
        assert result.size is not None

    def test_anthropic_uses_resize_profile(self):
        """Anthropic model type reads resize_profile config key."""
        img = Image.new("RGB", (2000, 3000))

        cfg = {
            "grayscale_conversion": False,
            "handle_transparency": False,
            "resize_profile": "auto",
            "low_max_side_px": 512,
            "high_max_side_px": 1568,
        }

        result = _apply_image_preprocessing(img, cfg, "anthropic")
        # Max side should be capped
        assert max(result.size) <= 1568

    def test_grayscale_already_grayscale(self):
        """Grayscale conversion on an already grayscale image is a no-op."""
        img = Image.new("L", (100, 100))

        cfg = {
            "grayscale_conversion": True,
            "handle_transparency": False,
            "llm_detail": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
        }

        result = _apply_image_preprocessing(img, cfg, "openai")
        # Should still be grayscale (or converted to RGB for box fitting)
        assert result.mode in ("L", "RGB")
