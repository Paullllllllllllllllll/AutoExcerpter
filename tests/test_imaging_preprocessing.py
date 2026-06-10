"""Tests for imaging/preprocessing.py - static image preprocessing core."""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from imaging.preprocessing import (
    DEFAULT_ANTHROPIC_HIGH_MAX_SIDE,
    ImageProcessor,
    _get_resampling_filter,
)


@pytest.fixture
def openai_config() -> dict[str, Any]:
    """OpenAI image config for testing."""
    return {
        "grayscale_conversion": True,
        "handle_transparency": True,
        "llm_detail": "high",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
        "resize_profile": "auto",
    }


@pytest.fixture
def anthropic_config() -> dict[str, Any]:
    """Anthropic image config for testing."""
    return {
        "grayscale_conversion": True,
        "handle_transparency": True,
        "low_max_side_px": 512,
        "high_max_side_px": 1568,
        "resize_profile": "auto",
    }


class TestResizeForDetail:
    """Tests for detail-based resizing."""

    def test_low_detail_caps_size(self, openai_config: dict[str, Any]) -> None:
        """Low detail caps longest side to max_side_px."""
        large_image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor.resize_for_detail(
            large_image, "low", openai_config, "openai"
        )

        assert max(result.size) <= 512

    def test_low_detail_small_image_unchanged(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Small images in low detail mode are unchanged."""
        small_image = Image.new("RGB", (300, 200))
        result = ImageProcessor.resize_for_detail(
            small_image, "low", openai_config, "openai"
        )

        assert result.size == (300, 200)

    def test_high_detail_openai_box_fit(self, openai_config: dict[str, Any]) -> None:
        """OpenAI high detail uses box fitting with padding."""
        image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor.resize_for_detail(
            image, "high", openai_config, "openai"
        )

        assert result.size == (768, 1536)

    def test_high_detail_anthropic_max_side(
        self, anthropic_config: dict[str, Any]
    ) -> None:
        """Anthropic high detail caps longest side without padding."""
        image = Image.new("RGB", (3000, 2000))
        result = ImageProcessor.resize_for_detail(
            image, "high", anthropic_config, "anthropic"
        )

        assert max(result.size) <= 1568
        assert result.size != (768, 1536)

    def test_resize_profile_none_skips_resize(self) -> None:
        """resize_profile='none' skips resizing entirely."""
        config = {"resize_profile": "none"}
        image = Image.new("RGB", (5000, 4000))
        result = ImageProcessor.resize_for_detail(image, "high", config, "openai")

        assert result.size == (5000, 4000)

    def test_auto_detail_treated_as_high(self, openai_config: dict[str, Any]) -> None:
        """Auto detail is treated as high."""
        image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor.resize_for_detail(
            image, "auto", openai_config, "openai"
        )

        assert result.size == (768, 1536)

    def test_unknown_detail_treated_as_high(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Unknown detail values fall back to high-detail behavior."""
        image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor.resize_for_detail(
            image, "nonsense", openai_config, "openai"
        )

        assert result.size == (768, 1536)


class TestResizeHelpers:
    """Tests for the static resize helper methods."""

    def test_resize_max_side_preserves_aspect_ratio(self) -> None:
        """_resize_max_side preserves the aspect ratio."""
        image = Image.new("RGB", (2000, 1000))
        result = ImageProcessor._resize_max_side(image, 500)

        assert result.size == (500, 250)

    def test_resize_max_side_no_upscale(self) -> None:
        """Images already within the cap are returned unchanged."""
        image = Image.new("RGB", (300, 200))
        result = ImageProcessor._resize_max_side(image, 500)

        assert result is image

    def test_resize_low_detail_uses_config_cap(self) -> None:
        """_resize_low_detail reads low_max_side_px from the config."""
        image = Image.new("RGB", (1000, 800))
        result = ImageProcessor._resize_low_detail(image, {"low_max_side_px": 250})

        assert max(result.size) <= 250

    def test_resize_box_fit_pads_grayscale_canvas(self) -> None:
        """Box fitting an L-mode image keeps the canvas grayscale."""
        image = Image.new("L", (2000, 1500), color=128)
        result = ImageProcessor._resize_box_fit(image, {"high_target_box": [768, 1536]})

        assert result.size == (768, 1536)
        assert result.mode == "L"

    def test_resize_box_fit_invalid_box_uses_defaults(self) -> None:
        """An invalid high_target_box falls back to default dimensions."""
        from config.constants import (
            DEFAULT_HIGH_TARGET_HEIGHT,
            DEFAULT_HIGH_TARGET_WIDTH,
        )

        image = Image.new("RGB", (2000, 1500))
        result = ImageProcessor._resize_box_fit(image, {"high_target_box": "not-a-box"})

        assert result.size == (DEFAULT_HIGH_TARGET_WIDTH, DEFAULT_HIGH_TARGET_HEIGHT)

    def test_resize_anthropic_high_default_cap(self) -> None:
        """Anthropic high resize defaults to the module-level max side."""
        image = Image.new("RGB", (4000, 3000))
        result = ImageProcessor._resize_anthropic_high(image, {})

        assert max(result.size) <= DEFAULT_ANTHROPIC_HIGH_MAX_SIDE

    def test_get_resampling_filter_returns_resampling(self) -> None:
        """_get_resampling_filter returns a valid PIL resampling member."""
        assert _get_resampling_filter() in (
            Image.Resampling.BILINEAR,
            Image.Resampling.LANCZOS,
        )


class TestPreprocessPilImage:
    """Tests for the shared preprocess_pil_image() core."""

    def test_grayscale_applied_when_enabled(
        self, openai_config: dict[str, Any]
    ) -> None:
        """RGB input is converted to grayscale when enabled."""
        image = Image.new("RGB", (1000, 1500), color=(128, 64, 192))
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.mode == "L"

    def test_grayscale_skipped_when_disabled(
        self, openai_config: dict[str, Any]
    ) -> None:
        """RGB input stays RGB when grayscale conversion is disabled."""
        openai_config["grayscale_conversion"] = False
        image = Image.new("RGB", (1000, 1500), color=(128, 64, 192))
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.mode == "RGB"

    def test_transparency_flattened_to_white(
        self, openai_config: dict[str, Any]
    ) -> None:
        """RGBA input is pasted onto a white background."""
        openai_config["grayscale_conversion"] = False
        image = Image.new("RGBA", (800, 600), color=(128, 64, 192, 128))
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.mode == "RGB"

    def test_palette_mode_with_transparency(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Palette mode image with transparency info is handled."""
        openai_config["grayscale_conversion"] = False
        image = Image.new("P", (100, 100))
        image.info["transparency"] = 0
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.mode == "RGB"

    def test_la_mode_transparency(self, openai_config: dict[str, Any]) -> None:
        """LA (grayscale with alpha) mode transparency is handled."""
        openai_config["grayscale_conversion"] = False
        image = Image.new("LA", (100, 100))
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.mode == "RGB"

    def test_transparency_kept_when_disabled(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Transparency handling is skipped when disabled."""
        openai_config["handle_transparency"] = False
        openai_config["grayscale_conversion"] = False
        openai_config["resize_profile"] = "none"
        image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 128))
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.mode == "RGBA"

    def test_openai_box_fit_applied(self, openai_config: dict[str, Any]) -> None:
        """OpenAI input is fitted into the configured target box."""
        image = Image.new("RGB", (1000, 1500))
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.size == (768, 1536)

    def test_google_uses_media_resolution(self) -> None:
        """Google model type reads the media_resolution config key."""
        cfg = {
            "grayscale_conversion": False,
            "handle_transparency": False,
            "media_resolution": "low",
            "low_max_side_px": 256,
            "high_target_box": [768, 768],
        }
        image = Image.new("RGB", (1000, 1000))
        result = ImageProcessor.preprocess_pil_image(image, cfg, "google")

        assert max(result.size) <= 256

    def test_anthropic_uses_resize_profile(
        self, anthropic_config: dict[str, Any]
    ) -> None:
        """Anthropic model type caps the longest side without padding."""
        anthropic_config["grayscale_conversion"] = False
        image = Image.new("RGB", (2000, 3000))
        result = ImageProcessor.preprocess_pil_image(
            image, anthropic_config, "anthropic"
        )

        assert max(result.size) <= 1568

    def test_grayscale_already_grayscale_noop(
        self, openai_config: dict[str, Any]
    ) -> None:
        """Grayscale conversion on an already grayscale image is a no-op."""
        openai_config["resize_profile"] = "none"
        image = Image.new("L", (100, 100), color=128)
        result = ImageProcessor.preprocess_pil_image(image, openai_config, "openai")

        assert result.mode == "L"
        assert result.size == (100, 100)
