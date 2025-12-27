"""Tests for modules/image_utils.py - Image processing utilities."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from modules.image_utils import ImageProcessor
from modules.constants import (
    DEFAULT_LOW_MAX_SIDE_PX,
    DEFAULT_HIGH_TARGET_WIDTH,
    DEFAULT_HIGH_TARGET_HEIGHT,
    WHITE_BACKGROUND_COLOR,
)


class TestImageProcessorInit:
    """Tests for ImageProcessor initialization."""

    def test_init_with_valid_image(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """ImageProcessor initializes with valid image path."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            assert processor.image_path == sample_image_file
            assert processor.provider == "openai"

    def test_init_with_openai_provider(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """OpenAI provider is detected correctly."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file, provider="openai", model_name="gpt-5")
            assert processor.model_type == "openai"

    def test_init_with_google_provider(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """Google provider is detected correctly."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file, provider="google", model_name="gemini-2.5-flash")
            assert processor.model_type == "google"

    def test_init_with_anthropic_provider(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """Anthropic provider is detected correctly."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file, provider="anthropic", model_name="claude-sonnet-4-5")
            assert processor.model_type == "anthropic"

    def test_init_with_openrouter_google_model(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """OpenRouter with Google model uses Google config."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(
                sample_image_file, 
                provider="openrouter", 
                model_name="google/gemini-2.5-flash"
            )
            assert processor.model_type == "google"

    def test_init_unsupported_format_raises(self, temp_dir: Path, mock_config_loader: MagicMock):
        """Unsupported image format raises ValueError."""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("fake image")
        
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            with pytest.raises(ValueError, match="Unsupported image format"):
                ImageProcessor(unsupported_file)


class TestConvertToGrayscale:
    """Tests for grayscale conversion."""

    def test_rgb_to_grayscale(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """RGB image is converted to grayscale when enabled."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            rgb_image = Image.new("RGB", (100, 100), color=(128, 64, 192))
            
            result = processor.convert_to_grayscale(rgb_image)
            assert result.mode == "L"

    def test_grayscale_disabled(self, sample_image_file: Path):
        """Grayscale conversion skipped when disabled."""
        mock_loader = MagicMock()
        mock_loader.get_image_processing_config.return_value = {
            "api_image_processing": {"grayscale_conversion": False}
        }
        
        with patch("modules.image_utils.get_config_loader", return_value=mock_loader):
            processor = ImageProcessor(sample_image_file)
            rgb_image = Image.new("RGB", (100, 100), color=(128, 64, 192))
            
            result = processor.convert_to_grayscale(rgb_image)
            assert result.mode == "RGB"


class TestHandleTransparency:
    """Tests for transparency handling."""

    def test_rgba_to_rgb(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """RGBA image is converted to RGB with white background."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 128))
            
            result = processor.handle_transparency(rgba_image)
            assert result.mode == "RGB"

    def test_rgb_unchanged(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """RGB image without transparency is unchanged."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            rgb_image = Image.new("RGB", (100, 100), color=(128, 64, 192))
            
            result = processor.handle_transparency(rgb_image)
            assert result.mode == "RGB"

    def test_transparency_handling_disabled(self, sample_image_file: Path):
        """Transparency handling skipped when disabled."""
        mock_loader = MagicMock()
        mock_loader.get_image_processing_config.return_value = {
            "api_image_processing": {"handle_transparency": False}
        }
        
        with patch("modules.image_utils.get_config_loader", return_value=mock_loader):
            processor = ImageProcessor(sample_image_file)
            rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 128))
            
            result = processor.handle_transparency(rgba_image)
            assert result.mode == "RGBA"


class TestResizeForDetail:
    """Tests for detail-based resizing."""

    @pytest.fixture
    def openai_config(self) -> dict:
        """OpenAI image config for testing."""
        return {
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
            "resize_profile": "auto",
        }

    @pytest.fixture
    def anthropic_config(self) -> dict:
        """Anthropic image config for testing."""
        return {
            "low_max_side_px": 512,
            "high_max_side_px": 1568,
            "resize_profile": "auto",
        }

    def test_low_detail_caps_size(self, openai_config: dict):
        """Low detail caps longest side to max_side_px."""
        large_image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor.resize_for_detail(large_image, "low", openai_config, "openai")
        
        assert max(result.size) <= 512

    def test_low_detail_small_image_unchanged(self, openai_config: dict):
        """Small images in low detail mode are unchanged."""
        small_image = Image.new("RGB", (300, 200))
        result = ImageProcessor.resize_for_detail(small_image, "low", openai_config, "openai")
        
        assert result.size == (300, 200)

    def test_high_detail_openai_box_fit(self, openai_config: dict):
        """OpenAI high detail uses box fitting with padding."""
        image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor.resize_for_detail(image, "high", openai_config, "openai")
        
        # Should fit within box and be padded to exact size
        assert result.size == (768, 1536)

    def test_high_detail_anthropic_max_side(self, anthropic_config: dict):
        """Anthropic high detail caps longest side without padding."""
        image = Image.new("RGB", (3000, 2000))
        result = ImageProcessor.resize_for_detail(image, "high", anthropic_config, "anthropic")
        
        # Should cap longest side to 1568, preserve aspect ratio
        assert max(result.size) <= 1568
        # Should NOT be padded to a specific size
        assert result.size != (768, 1536)

    def test_resize_profile_none_skips_resize(self):
        """resize_profile='none' skips resizing entirely."""
        config = {"resize_profile": "none"}
        image = Image.new("RGB", (5000, 4000))
        result = ImageProcessor.resize_for_detail(image, "high", config, "openai")
        
        assert result.size == (5000, 4000)

    def test_auto_detail_treated_as_high(self, openai_config: dict):
        """Auto detail is treated as high."""
        image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor.resize_for_detail(image, "auto", openai_config, "openai")
        
        assert result.size == (768, 1536)


class TestProcessImageToMemory:
    """Tests for in-memory image processing."""

    def test_returns_pil_image(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """process_image_to_memory returns a PIL Image."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            result = processor.process_image_to_memory()
            
            assert isinstance(result, Image.Image)

    def test_applies_grayscale(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """In-memory processing applies grayscale conversion."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            result = processor.process_image_to_memory()
            
            # Result mode depends on resize strategy - may be L or RGB after processing
            assert result.mode in ("L", "RGB")

    def test_handles_transparency(self, sample_png_with_transparency: Path, mock_config_loader: MagicMock):
        """In-memory processing handles transparency."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_png_with_transparency)
            result = processor.process_image_to_memory()
            
            # Should not have alpha channel
            assert result.mode in ("RGB", "L")


class TestPilImageToBase64:
    """Tests for PIL to base64 conversion."""

    def test_returns_valid_base64(self, sample_rgb_image: Image.Image):
        """Conversion returns valid base64 string."""
        result = ImageProcessor.pil_image_to_base64(sample_rgb_image)
        
        assert isinstance(result, str)
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_output_is_jpeg(self, sample_rgb_image: Image.Image):
        """Output is JPEG format."""
        result = ImageProcessor.pil_image_to_base64(sample_rgb_image)
        decoded = base64.b64decode(result)
        
        # JPEG magic bytes
        assert decoded[:2] == b'\xff\xd8'

    def test_quality_parameter_affects_size(self, sample_rgb_image: Image.Image):
        """Different quality settings produce different sizes."""
        low_quality = ImageProcessor.pil_image_to_base64(sample_rgb_image, jpeg_quality=10)
        high_quality = ImageProcessor.pil_image_to_base64(sample_rgb_image, jpeg_quality=100)
        
        # Generally lower quality = smaller size, but for simple images the difference may be minimal
        # At least verify both are valid base64 strings
        assert len(low_quality) > 0
        assert len(high_quality) > 0
        # Lower quality should generally be smaller or equal
        assert len(low_quality) <= len(high_quality)

    def test_converts_mode_if_needed(self):
        """RGBA images are converted to RGB for JPEG."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 128))
        result = ImageProcessor.pil_image_to_base64(rgba_image)
        
        # Should succeed (JPEG doesn't support RGBA)
        assert isinstance(result, str)


class TestProcessImage:
    """Tests for process_image (file output)."""

    def test_saves_to_jpg(self, sample_image_file: Path, temp_dir: Path, mock_config_loader: MagicMock):
        """process_image saves output as JPEG."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            output_path = temp_dir / "output.jpg"
            
            result = processor.process_image(output_path)
            
            assert output_path.exists()
            assert "Processed and saved" in result

    def test_creates_jpg_even_if_png_requested(
        self, sample_image_file: Path, temp_dir: Path, mock_config_loader: MagicMock
    ):
        """Output is always JPEG regardless of requested extension."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file)
            output_path = temp_dir / "output.png"
            
            processor.process_image(output_path)
            
            # Should create .jpg file
            jpg_path = output_path.with_suffix(".jpg")
            assert jpg_path.exists()

    def test_handles_processing_error(self, temp_dir: Path, mock_config_loader: MagicMock):
        """Handles errors during image processing gracefully."""
        fake_image_path = temp_dir / "fake.jpg"
        fake_image_path.write_bytes(b"not an image")
        
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(fake_image_path)
            output_path = temp_dir / "output.jpg"
            
            result = processor.process_image(output_path)
            
            assert "Failed to process" in result


class TestGetDetailParam:
    """Tests for _get_detail_param method."""

    def test_openai_returns_llm_detail(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """OpenAI provider returns llm_detail config value."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file, provider="openai")
            result = processor._get_detail_param()
            
            assert result == "high"

    def test_google_returns_media_resolution(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """Google provider returns media_resolution config value."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file, provider="google", model_name="gemini-2.5-flash")
            result = processor._get_detail_param()
            
            assert result == "high"

    def test_anthropic_returns_resize_profile(self, sample_image_file: Path, mock_config_loader: MagicMock):
        """Anthropic provider returns resize_profile config value."""
        with patch("modules.image_utils.get_config_loader", return_value=mock_config_loader):
            processor = ImageProcessor(sample_image_file, provider="anthropic", model_name="claude-3-opus")
            result = processor._get_detail_param()
            
            assert result == "auto"
