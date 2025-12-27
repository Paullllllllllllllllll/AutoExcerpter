"""Image preprocessing utilities for OCR optimization.

This module provides image processing functionality optimized for multiple LLM Vision APIs
(OpenAI, Google Gemini, Anthropic Claude) and OCR quality. All processing can be done
in-memory to avoid disk I/O overhead.

Key Features:
1. **Provider-Specific Preprocessing**: Different resize strategies per provider:
   - OpenAI: Box fitting with padding (768×1536)
   - Google Gemini: Box fitting optimized for 768px tiles
   - Anthropic Claude: Max-side capping (1568px, no padding)

2. **In-Memory Processing**: Process images without intermediate disk writes using
   PIL and base64 encoding
   
3. **Configurable Preprocessing**:
   - Grayscale conversion for better OCR
   - Transparency handling (paste on white background)
   - Intelligent resizing based on detail level (low/high/auto)
   - JPEG compression with configurable quality

4. **Configuration-Driven**: All settings loaded from image_processing.yaml
   with provider-specific sections (api_image_processing, google_image_processing,
   anthropic_image_processing)

5. **OpenRouter Passthrough**: Automatically detects underlying model type when
   using models via OpenRouter (e.g., 'google/gemini-2.5-flash' uses Google config)

The ImageProcessor class handles individual image processing with full configuration
support, while static methods provide utilities for batch operations.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from modules.config_loader import get_config_loader
from modules.constants import (
    SUPPORTED_IMAGE_EXTENSIONS,
    DEFAULT_LOW_MAX_SIDE_PX,
    DEFAULT_HIGH_TARGET_WIDTH,
    DEFAULT_HIGH_TARGET_HEIGHT,
    DEFAULT_JPEG_QUALITY,
    WHITE_BACKGROUND_COLOR,
)
from modules.model_utils import detect_model_type, get_image_config_section_name, ModelType
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Default max side for Anthropic high-detail mode
DEFAULT_ANTHROPIC_HIGH_MAX_SIDE = 1568


# ============================================================================
# Image Processing Class
# ============================================================================
class ImageProcessor:
    def __init__(
        self,
        image_path: Path,
        provider: str = "openai",
        model_name: str = "",
    ) -> None:
        """
        Initialize the ImageProcessor with provider-specific configuration.

        Args:
            image_path: The path to the image file.
            provider: Provider name (openai, google, anthropic, openrouter)
            model_name: Model name for detecting underlying model type when using
                       OpenRouter (e.g., 'google/gemini-2.5-flash' uses Google config)

        Raises:
            ValueError: If the image format is not supported.
        """
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.error(f"Unsupported image format: {image_path.suffix}")
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        self.image_path = image_path
        self.provider = provider.lower()
        self.model_name = model_name.lower() if model_name else ""
        
        # Detect underlying model type (for OpenRouter passthrough)
        self.model_type: ModelType = detect_model_type(self.provider, self.model_name)

        config_loader = get_config_loader()
        # Full config dict (contains provider-specific sections)
        self.image_config = config_loader.get_image_processing_config()
        
        # Get provider-specific config section
        section_name = get_image_config_section_name(self.model_type)
        self.img_cfg = self.image_config.get(section_name, {})
        
        logger.debug(
            f"ImageProcessor initialized: provider={self.provider}, "
            f"model_type={self.model_type}, config_section={section_name}"
        )

    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert the image to grayscale if enabled."""
        if self.img_cfg.get('grayscale_conversion', True):
            return ImageOps.grayscale(image)
        return image

    @staticmethod
    def resize_for_detail(
        image: Image.Image,
        detail: str,
        img_cfg: dict[str, Any],
        model_type: ModelType = "openai",
    ) -> Image.Image:
        """
        Resize strategy based on desired LLM detail and model type.

        Provider-specific strategies:
        - OpenAI/Google: Box fitting with padding (high_target_box)
        - Anthropic: Max-side capping without padding (high_max_side_px)

        Args:
            image: The input image.
            detail: The desired level of detail ('low', 'high', 'auto', 'medium', 'ultra_high').
            img_cfg: The image configuration dictionary.
            model_type: The model type ('openai', 'google', 'anthropic').

        Returns:
            The resized image.
        """
        # Check if resizing is disabled
        resize_profile = (img_cfg.get('resize_profile', 'auto') or 'auto').lower()
        if resize_profile == 'none':
            return image
        
        # Normalize detail level
        detail_norm = (detail or 'high').lower()
        if detail_norm not in ('low', 'high', 'auto', 'medium', 'ultra_high'):
            detail_norm = 'high'
        
        # Low detail: cap longest side (same strategy for all providers)
        if detail_norm == 'low':
            return ImageProcessor._resize_low_detail(image, img_cfg)
        
        # High/auto/medium/ultra_high: provider-specific strategy
        if model_type == "anthropic":
            # Anthropic: cap longest side to high_max_side_px (no padding)
            return ImageProcessor._resize_anthropic_high(image, img_cfg)
        else:
            # OpenAI/Google: fit into box and pad with white
            return ImageProcessor._resize_box_fit(image, img_cfg)
    
    @staticmethod
    def _resize_low_detail(image: Image.Image, img_cfg: dict[str, Any]) -> Image.Image:
        """Downscale image to max side length for low detail (all providers)."""
        max_side = int(img_cfg.get('low_max_side_px', DEFAULT_LOW_MAX_SIDE_PX))
        w, h = image.size
        longest = max(w, h)
        
        if longest <= max_side:
            return image
        
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def _resize_box_fit(image: Image.Image, img_cfg: dict[str, Any]) -> Image.Image:
        """Fit and pad image into target box (OpenAI/Google strategy)."""
        box = img_cfg.get('high_target_box', [DEFAULT_HIGH_TARGET_WIDTH, DEFAULT_HIGH_TARGET_HEIGHT])
        try:
            target_width = int(box[0])
            target_height = int(box[1])
        except Exception:
            target_width, target_height = DEFAULT_HIGH_TARGET_WIDTH, DEFAULT_HIGH_TARGET_HEIGHT
        
        orig_width, orig_height = image.size
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = max(1, int(orig_width * scale))
        new_height = max(1, int(orig_height * scale))
        
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        final_img = Image.new("RGB", (target_width, target_height), WHITE_BACKGROUND_COLOR)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        final_img.paste(resized_img, (paste_x, paste_y))
        return final_img
    
    @staticmethod
    def _resize_anthropic_high(image: Image.Image, img_cfg: dict[str, Any]) -> Image.Image:
        """Cap longest side for Anthropic (no padding, preserves aspect ratio)."""
        max_side = int(img_cfg.get('high_max_side_px', DEFAULT_ANTHROPIC_HIGH_MAX_SIDE))
        w, h = image.size
        longest = max(w, h)
        
        if longest <= max_side:
            return image
        
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def handle_transparency(self, image: Image.Image) -> Image.Image:
        """Handle transparency by pasting the image onto a white background."""
        if self.img_cfg.get('handle_transparency', True):
            if image.mode in ('RGBA', 'LA') or (
                    image.mode == 'P' and 'transparency' in image.info):
                background = Image.new("RGB", image.size, WHITE_BACKGROUND_COLOR)
                background.paste(image, mask=image.split()[-1])
                return background
        return image

    def _get_detail_param(self) -> str:
        """Get the detail/resolution parameter based on model type."""
        if self.model_type == "google":
            return self.img_cfg.get('media_resolution', 'high') or 'high'
        elif self.model_type == "anthropic":
            return self.img_cfg.get('resize_profile', 'auto') or 'auto'
        else:
            return self.img_cfg.get('llm_detail', 'high') or 'high'

    def process_image(self, output_path: Path) -> str:
        """Process the image and save it to the given output path with compression."""
        try:
            with Image.open(self.image_path) as img:
                img = self.handle_transparency(img)
                img = self.convert_to_grayscale(img)
                
                # Choose resizing based on model type and appropriate config param
                detail = self._get_detail_param()
                img = ImageProcessor.resize_for_detail(
                    img, detail, self.img_cfg, self.model_type
                )

                # Force output to JPEG with configurable quality
                jpg_output_path = output_path.with_suffix('.jpg')
                jpeg_quality = int(self.img_cfg.get('jpeg_quality', DEFAULT_JPEG_QUALITY))
                img.save(
                    jpg_output_path,
                    format='JPEG',
                    quality=jpeg_quality
                )
                logger.debug(
                    f"Saved processed image {jpg_output_path.name} size={img.size} "
                    f"quality={jpeg_quality} detail={detail} model_type={self.model_type}"
                )
            return f"Processed and saved: {jpg_output_path.name}"
        except Exception as e:
            logger.error(f"Error processing image {self.image_path.name}: {e}")
            return f"Failed to process {self.image_path.name}: {e}"

    def process_image_to_memory(self) -> Image.Image:
        """Process the image and return the PIL Image object in-memory."""
        with Image.open(self.image_path) as img:
            img = img.copy()  # Create a copy to work with after closing the file
            img = self.handle_transparency(img)
            img = self.convert_to_grayscale(img)
            
            # Choose resizing based on model type and appropriate config param
            detail = self._get_detail_param()
            img = ImageProcessor.resize_for_detail(
                img, detail, self.img_cfg, self.model_type
            )
            
            # Convert to RGB if grayscale for JPEG encoding
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            logger.debug(
                f"Processed image {self.image_path.name} in-memory: size={img.size} "
                f"detail={detail} model_type={self.model_type}"
            )
            return img

    @staticmethod
    def pil_image_to_base64(img: Image.Image, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
        """Convert a PIL Image to base64-encoded JPEG string."""
        buffer = io.BytesIO()
        # Ensure image is in RGB mode for JPEG
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "ImageProcessor",
    "SUPPORTED_IMAGE_EXTENSIONS",
]
