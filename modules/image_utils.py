"""Image preprocessing utilities for OCR optimization.

This module provides image processing functionality optimized for OpenAI Vision API
and OCR quality. All processing can be done in-memory to avoid disk I/O overhead.

Key Features:
1. **In-Memory Processing**: Process images without intermediate disk writes using
   PIL and base64 encoding
   
2. **Configurable Preprocessing**:
   - Grayscale conversion for better OCR
   - Transparency handling (paste on white background)
   - Intelligent resizing based on detail level (low/high/auto)
   - JPEG compression with configurable quality

3. **Detail-Level Optimization**:
   - 'low': Downscale to max side length for faster processing
   - 'high': Fit into target box with padding for maximum quality
   - 'auto': Defaults to high quality

4. **Configuration-Driven**: All settings loaded from image_processing.yaml

The ImageProcessor class handles individual image processing with full configuration
support, while static methods provide utilities for batch operations.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageOps

from modules.config_loader import get_config_loader
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Public API
__all__ = [
    "ImageProcessor",
    "SUPPORTED_IMAGE_EXTENSIONS",
]

# ============================================================================
# Constants
# ============================================================================
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'}
DEFAULT_LOW_MAX_SIDE_PX = 512
DEFAULT_HIGH_TARGET_WIDTH = 768
DEFAULT_HIGH_TARGET_HEIGHT = 1536
DEFAULT_JPEG_QUALITY = 95
WHITE_BACKGROUND_COLOR = (255, 255, 255)


# ============================================================================
# Image Processing Class
# ============================================================================
class ImageProcessor:
    def __init__(self, image_path: Path) -> None:
        """
        Initialize the ImageProcessor with the given image path.

        Args:
            image_path (Path): The path to the image file.

        Raises:
            ValueError: If the image format is not supported.
        """
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.error(f"Unsupported image format: {image_path.suffix}")
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        self.image_path = image_path

        config_loader = get_config_loader()
        # Full config dict (contains 'image_processing' and 'ocr' sections)
        self.image_config = config_loader.get_image_processing_config()
        # OpenAI API preprocessing settings
        self.img_cfg = self.image_config.get('api_image_processing', {})

    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert the image to grayscale if enabled.

        Args:
            image (Image.Image): The input image.

        Returns:
            Image.Image: The grayscale image.
        """
        if self.img_cfg.get('grayscale_conversion', True):
            return ImageOps.grayscale(image)
        return image

    @staticmethod
    def resize_for_detail(image: Image.Image, detail: str, img_cfg: Dict[str, Any]) -> Image.Image:
        """
        Resize strategy based on desired LLM detail.

        - low: downscale longest side to low_max_side_px.
        - high: fit/pad into high_target_box [width, height].
        - auto: default to 'high' strategy.

        Args:
            image: The input image.
            detail: The desired level of detail.
            img_cfg: The image configuration.

        Returns:
            The resized image.
        """
        # Check if resizing is disabled
        resize_profile = (img_cfg.get('resize_profile', 'auto') or 'auto').lower()
        if resize_profile == 'none':
            return image
        
        # Normalize detail level
        detail_norm = (detail or 'high').lower()
        if detail_norm not in ('low', 'high', 'auto'):
            detail_norm = 'high'
        
        # Apply appropriate resize strategy
        if detail_norm == 'low':
            return ImageProcessor._resize_low_detail(image, img_cfg)
        else:
            return ImageProcessor._resize_high_detail(image, img_cfg)
    
    @staticmethod
    def _resize_low_detail(image: Image.Image, img_cfg: Dict[str, Any]) -> Image.Image:
        """Downscale image to max side length for low detail."""
        max_side = int(img_cfg.get('low_max_side_px', DEFAULT_LOW_MAX_SIDE_PX))
        w, h = image.size
        longest = max(w, h)
        
        if longest <= max_side:
            return image
        
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def _resize_high_detail(image: Image.Image, img_cfg: Dict[str, Any]) -> Image.Image:
        """Fit and pad image into target box for high detail."""
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

    def handle_transparency(self, image: Image.Image) -> Image.Image:
        """
        Handle transparency by pasting the image onto a white background.

        Args:
            image (Image.Image): The input image.

        Returns:
            Image.Image: The processed image.
        """
        if self.img_cfg.get('handle_transparency', True):
            if image.mode in ('RGBA', 'LA') or (
                    image.mode == 'P' and 'transparency' in image.info):
                background = Image.new("RGB", image.size, WHITE_BACKGROUND_COLOR)
                background.paste(image, mask=image.split()[-1])
                return background
        return image

    def process_image(self, output_path: Path) -> str:
        """
        Process the image and save it to the given output path with compression.

        Args:
            output_path (Path): The output path for the processed image.

        Returns:
            str: A message indicating the outcome.
        """
        try:
            with Image.open(self.image_path) as img:
                img = self.handle_transparency(img)
                img = self.convert_to_grayscale(img)
                # Choose resizing based on llm_detail and resize_profile
                detail = (self.img_cfg.get('llm_detail', 'high') or 'high')
                img = ImageProcessor.resize_for_detail(img, detail, self.img_cfg)

                # Force output to JPEG with configurable quality (regardless of extension)
                # Create a new path with .jpg extension
                jpg_output_path = output_path.with_suffix('.jpg')
                jpeg_quality = int(self.img_cfg.get('jpeg_quality', DEFAULT_JPEG_QUALITY))
                img.save(
                    jpg_output_path,
                    format='JPEG',
                    quality=jpeg_quality
                )
                logger.debug(
                    f"Saved processed image {jpg_output_path.name} size={img.size} quality={jpeg_quality} detail={detail}"
                )
            return f"Processed and saved: {jpg_output_path.name}"
        except Exception as e:
            logger.error(f"Error processing image {self.image_path.name}: {e}")
            return f"Failed to process {self.image_path.name}: {e}"

    def process_image_to_memory(self) -> Image.Image:
        """
        Process the image and return the PIL Image object in-memory.

        Returns:
            Image.Image: The processed PIL Image object.
        
        Raises:
            Exception: If image processing fails.
        """
        with Image.open(self.image_path) as img:
            img = img.copy()  # Create a copy to work with after closing the file
            img = self.handle_transparency(img)
            img = self.convert_to_grayscale(img)
            # Choose resizing based on llm_detail and resize_profile
            detail = (self.img_cfg.get('llm_detail', 'high') or 'high')
            img = ImageProcessor.resize_for_detail(img, detail, self.img_cfg)
            
            # Convert to RGB if grayscale for JPEG encoding
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            logger.debug(
                f"Processed image {self.image_path.name} in-memory: size={img.size} detail={detail}"
            )
            return img

    @staticmethod
    def pil_image_to_base64(img: Image.Image, jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> str:
        """
        Convert a PIL Image to base64-encoded JPEG string.

        Args:
            img (Image.Image): The PIL Image object.
            jpeg_quality (int): JPEG quality (1-100).

        Returns:
            str: Base64-encoded JPEG image string.
        """
        buffer = io.BytesIO()
        # Ensure image is in RGB mode for JPEG
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    # Note: Removed unused folder-level processing helpers (prepare_image_folder,
    # process_images_multiprocessing, _process_image_task, process_and_save_images)
