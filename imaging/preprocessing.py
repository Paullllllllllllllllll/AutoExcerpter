"""Image preprocessing utilities for OCR optimization.

This module provides the in-memory preprocessing core used by the streaming
transcription pipeline (``imaging.payload``). Processing is optimized for
multiple LLM Vision APIs (OpenAI, Google Gemini, Anthropic Claude) and OCR
quality; nothing is written to disk.

Key Features:
1. **Provider-Specific Preprocessing**: Different resize strategies per provider:
   - OpenAI: Box fitting with padding (768×1536)
   - Google Gemini: Box fitting optimized for 768px tiles
   - Anthropic Claude: Max-side capping (1568px, no padding)

2. **Configurable Preprocessing**:
   - Grayscale conversion for better OCR
   - Transparency handling (paste on white background)
   - Intelligent resizing based on detail level (low/high/auto)

3. **Configuration-Driven**: All settings come from provider-specific
   sections of image_processing.yaml (api_image_processing,
   google_image_processing, anthropic_image_processing), resolved upstream
   by the payload sources.
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageOps

from config.constants import (
    DEFAULT_HIGH_TARGET_HEIGHT,
    DEFAULT_HIGH_TARGET_WIDTH,
    DEFAULT_LOW_MAX_SIDE_PX,
    SUPPORTED_IMAGE_EXTENSIONS,
    WHITE_BACKGROUND_COLOR,
)
from config.loader import get_config_loader
from config.logger import setup_logger
from imaging._provider import ModelType

logger = setup_logger(__name__)

# Default max side for Anthropic high-detail mode
DEFAULT_ANTHROPIC_HIGH_MAX_SIDE = 1568

# Defaults for OpenAI full-resolution ("original") detail: cap the longest side
# and the total pixel budget so an unresized folio scan cannot exceed the API's
# input limits (mirrors ChronoMiner/ChronoTranscriber).
DEFAULT_ORIGINAL_MAX_SIDE_PX = 6000
DEFAULT_ORIGINAL_MAX_PIXELS = 10_240_000

# Resampling algorithm mapping
_RESAMPLING_ALGORITHMS = {
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
}


def _get_resampling_filter() -> Image.Resampling:
    """Return the configured resampling filter from image_processing.yaml."""
    try:
        config_loader = get_config_loader()
        img_config = config_loader.get_image_processing_config()
        algo = str(img_config.get("resampling_algorithm", "bilinear")).lower()
        return _RESAMPLING_ALGORITHMS.get(algo, Image.Resampling.BILINEAR)
    except Exception:
        return Image.Resampling.BILINEAR


# ============================================================================
# Image Processing Class
# ============================================================================
class ImageProcessor:
    """Static in-memory preprocessing core (transparency, grayscale, resize)."""

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
            detail: The desired level of detail ('low', 'high', 'auto', 'medium',
                'ultra_high', 'original').
            img_cfg: The image configuration dictionary.
            model_type: The model type ('openai', 'google', 'anthropic').

        Returns:
            The resized image.
        """
        # Normalize detail level
        detail_norm = (detail or "high").lower()
        if detail_norm not in (
            "low",
            "high",
            "auto",
            "medium",
            "ultra_high",
            "original",
        ):
            detail_norm = "high"

        # Original detail (GPT-5.6 family): cap to max side / max pixels, no
        # padding. Checked before the resize_profile early-return so an explicit
        # 'none' cannot leave the full-resolution payload unbounded.
        if detail_norm == "original":
            return ImageProcessor._resize_original(image, img_cfg)

        # Check if resizing is disabled
        resize_profile = (img_cfg.get("resize_profile", "auto") or "auto").lower()
        if resize_profile == "none":
            return image

        # Low detail: cap longest side (same strategy for all providers)
        if detail_norm == "low":
            return ImageProcessor._resize_low_detail(image, img_cfg)

        # High/auto/medium/ultra_high: provider-specific strategy
        if model_type == "anthropic":
            # Anthropic: cap longest side to high_max_side_px (no padding)
            return ImageProcessor._resize_anthropic_high(image, img_cfg)
        else:
            # OpenAI/Google: fit into box and pad with white
            return ImageProcessor._resize_box_fit(image, img_cfg)

    @staticmethod
    def _resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
        """Cap longest side, preserving aspect ratio (shared by low-detail and
        Anthropic)."""
        w, h = image.size
        longest = max(w, h)

        if longest <= max_side:
            return image

        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, _get_resampling_filter())

    @staticmethod
    def _resize_low_detail(image: Image.Image, img_cfg: dict[str, Any]) -> Image.Image:
        """Downscale image to max side length for low detail (all providers)."""
        max_side = int(img_cfg.get("low_max_side_px", DEFAULT_LOW_MAX_SIDE_PX))
        return ImageProcessor._resize_max_side(image, max_side)

    @staticmethod
    def _resize_box_fit(image: Image.Image, img_cfg: dict[str, Any]) -> Image.Image:
        """Fit and pad image into target box (OpenAI/Google strategy)."""
        box = img_cfg.get(
            "high_target_box", [DEFAULT_HIGH_TARGET_WIDTH, DEFAULT_HIGH_TARGET_HEIGHT]
        )
        try:
            target_width = int(box[0])
            target_height = int(box[1])
        except Exception:
            target_width, target_height = (
                DEFAULT_HIGH_TARGET_WIDTH,
                DEFAULT_HIGH_TARGET_HEIGHT,
            )

        orig_width, orig_height = image.size
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = max(1, int(orig_width * scale))
        new_height = max(1, int(orig_height * scale))

        resized_img = image.resize((new_width, new_height), _get_resampling_filter())
        if image.mode == "L":
            final_img = Image.new("L", (target_width, target_height), 255)
        else:
            final_img = Image.new(
                "RGB", (target_width, target_height), WHITE_BACKGROUND_COLOR
            )
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        final_img.paste(resized_img, (paste_x, paste_y))
        return final_img

    @staticmethod
    def _resize_anthropic_high(
        image: Image.Image, img_cfg: dict[str, Any]
    ) -> Image.Image:
        """Cap longest side for Anthropic (no padding, preserves aspect ratio)."""
        max_side = int(img_cfg.get("high_max_side_px", DEFAULT_ANTHROPIC_HIGH_MAX_SIDE))
        return ImageProcessor._resize_max_side(image, max_side)

    @staticmethod
    def _resize_original(image: Image.Image, img_cfg: dict[str, Any]) -> Image.Image:
        """Cap 'original' (full-resolution) detail to bound API input size.

        Applies two aspect-preserving caps with no padding and no upscaling:
        the longest side to ``original_max_side_px`` and the total pixel budget
        to ``original_max_pixels``. Mirrors ChronoMiner/ChronoTranscriber.
        """
        max_side = int(
            img_cfg.get("original_max_side_px", DEFAULT_ORIGINAL_MAX_SIDE_PX)
        )
        max_pixels = int(
            img_cfg.get("original_max_pixels", DEFAULT_ORIGINAL_MAX_PIXELS)
        )
        # Cap longest side first (reuses the shared max-side helper).
        image = ImageProcessor._resize_max_side(image, max_side)
        # Then cap the total pixel budget if still exceeded.
        w, h = image.size
        if w * h > max_pixels:
            scale = (max_pixels / float(w * h)) ** 0.5
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            image = image.resize(new_size, _get_resampling_filter())
        return image

    @staticmethod
    def _resolve_detail(img_cfg: dict[str, Any], model_type: ModelType) -> str:
        """Resolve the effective detail level for a model type from its config.

        Mirrors the per-provider key lookup used by ``preprocess_pil_image``:
        Google reads ``media_resolution``, Anthropic reads ``resize_profile``,
        everything else reads ``llm_detail``.
        """
        if model_type == "google":
            return str(img_cfg.get("media_resolution", "high") or "high")
        if model_type == "anthropic":
            return str(img_cfg.get("resize_profile", "auto") or "auto")
        return str(img_cfg.get("llm_detail", "high") or "high")

    @staticmethod
    def _max_side_scale(w: float, h: float, max_side: int) -> float:
        """Aspect-preserving downscale factor that caps the longest side."""
        longest = max(w, h)
        if longest <= 0 or longest <= max_side:
            return 1.0
        return max_side / float(longest)

    @staticmethod
    def _original_scale(w: float, h: float, img_cfg: dict[str, Any]) -> float:
        """Combined downscale factor applied by the 'original' resize path."""
        max_side = int(
            img_cfg.get("original_max_side_px", DEFAULT_ORIGINAL_MAX_SIDE_PX)
        )
        max_pixels = int(
            img_cfg.get("original_max_pixels", DEFAULT_ORIGINAL_MAX_PIXELS)
        )
        s1 = ImageProcessor._max_side_scale(w, h, max_side)
        w1, h1 = w * s1, h * s1
        s2 = (max_pixels / float(w1 * h1)) ** 0.5 if w1 * h1 > max_pixels else 1.0
        return float(s1 * s2)

    @staticmethod
    def content_scale_factor(
        src_size: tuple[float, float],
        img_cfg: dict[str, Any],
        model_type: ModelType = "openai",
    ) -> float:
        """Aspect-preserving factor the resize step would apply to *src_size*.

        Returns the scale the active resize profile would multiply a source
        image of ``src_size`` (px) by, ignoring box-fit padding (padding does
        not change the content scale). Used to derive a direct PDF render DPI so
        a page can be rasterized straight to (approximately) its final content
        size instead of rendered large and downscaled.

        The box-fit profiles can return a factor > 1 (they upscale small pages);
        callers that must not render-upscale should clamp with ``min(scale, 1)``.
        Returns 1.0 whenever the profile would leave the image at source size.
        """
        w, h = src_size
        if w <= 0 or h <= 0:
            return 1.0

        detail_norm = ImageProcessor._resolve_detail(img_cfg, model_type).lower()
        if detail_norm not in (
            "low",
            "high",
            "auto",
            "medium",
            "ultra_high",
            "original",
        ):
            detail_norm = "high"

        if detail_norm == "original":
            return ImageProcessor._original_scale(w, h, img_cfg)

        resize_profile = (img_cfg.get("resize_profile", "auto") or "auto").lower()
        if resize_profile == "none":
            return 1.0

        if detail_norm == "low":
            max_side = int(img_cfg.get("low_max_side_px", DEFAULT_LOW_MAX_SIDE_PX))
            return ImageProcessor._max_side_scale(w, h, max_side)

        if model_type == "anthropic":
            max_side = int(
                img_cfg.get("high_max_side_px", DEFAULT_ANTHROPIC_HIGH_MAX_SIDE)
            )
            return ImageProcessor._max_side_scale(w, h, max_side)

        # OpenAI/Google box-fit: scale to fit the target box (may upscale).
        box = img_cfg.get(
            "high_target_box", [DEFAULT_HIGH_TARGET_WIDTH, DEFAULT_HIGH_TARGET_HEIGHT]
        )
        try:
            target_width = int(box[0])
            target_height = int(box[1])
        except Exception:
            target_width, target_height = (
                DEFAULT_HIGH_TARGET_WIDTH,
                DEFAULT_HIGH_TARGET_HEIGHT,
            )
        return min(target_width / w, target_height / h)

    @staticmethod
    def preprocess_pil_image(
        pil_img: Image.Image,
        img_cfg: dict[str, Any],
        model_type: ModelType = "openai",
    ) -> Image.Image:
        """Apply preprocessing steps to an in-memory PIL image.

        Performs: transparency handling -> grayscale -> resize (provider-specific).
        This is the shared preprocessing core used by the PDF and image-folder
        payload sources.

        Args:
            pil_img: PIL Image to preprocess.
            img_cfg: Provider-specific image configuration dict.
            model_type: Model type for resize strategy.

        Returns:
            Preprocessed PIL Image.
        """
        # Handle transparency
        if img_cfg.get("handle_transparency", True) and (
            pil_img.mode in ("RGBA", "LA")
            or (pil_img.mode == "P" and "transparency" in pil_img.info)
        ):
            # Promote palette-with-transparency to RGBA so its alpha channel
            # is honored as the paste mask instead of the palette color.
            if pil_img.mode == "P":
                pil_img = pil_img.convert("RGBA")
            background = Image.new("RGB", pil_img.size, WHITE_BACKGROUND_COLOR)
            mask = pil_img.split()[-1] if pil_img.mode in ("RGBA", "LA") else None
            background.paste(pil_img, mask=mask)
            pil_img = background

        # Grayscale conversion
        if img_cfg.get("grayscale_conversion", True) and pil_img.mode != "L":
            pil_img = ImageOps.grayscale(pil_img)

        # Get detail parameter based on model type
        detail = ImageProcessor._resolve_detail(img_cfg, model_type)

        # Resize with provider-specific strategy
        pil_img = ImageProcessor.resize_for_detail(pil_img, detail, img_cfg, model_type)

        return pil_img


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "ImageProcessor",
    "SUPPORTED_IMAGE_EXTENSIONS",
]
