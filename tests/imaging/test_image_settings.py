"""The recommended image sections and the run's image options."""

from __future__ import annotations

import pytest

from autoexcerpter.common.native import image_settings_fingerprint
from autoexcerpter.imaging.settings import IMAGE_PROCESSING, resolve_image_settings
from autoexcerpter.spec import ImageSpec, recommended_images

PROVIDERS = ["api", "anthropic", "google", "custom"]
FAMILIES = {"api": "openai", "anthropic": "anthropic", "google": "google"}
DETAIL_KEYS = {
    "api": "llm_detail",
    "anthropic": "resize_profile",
    "google": "media_resolution",
    "custom": "llm_detail",
}

# The fingerprint the characterization goldens record for the default OpenAI
# transcription model; it guards resume, so the defaults must not move it.
GPT_56_TERRA_FINGERPRINT = (
    "6ac97af9ff976c10c92aa0123591af828a27e84634d5cfde8504605757603452"
)


def test_section_layout() -> None:
    assert list(IMAGE_PROCESSING) == [
        "render_strategy",
        "max_pixels_per_page",
        "resampling_algorithm",
        *(f"{name}_image_processing" for name in PROVIDERS),
    ]
    common = [
        "target_dpi",
        "native_fallback_dpi",
        "payload_format",
        "max_image_bytes",
        "grayscale_conversion",
        "handle_transparency",
        "jpeg_quality",
    ]
    for name in PROVIDERS:
        detail = [] if name == "anthropic" else [DETAIL_KEYS[name]]
        assert list(IMAGE_PROCESSING[f"{name}_image_processing"]) == common + detail + [
            "resize_profile",
            "low_max_side_px",
            "high_target_box",
        ]
    sections = [IMAGE_PROCESSING[f"{p}_image_processing"] for p in PROVIDERS]
    assert [s["target_dpi"] for s in sections] == ["native", "native", 300, 150]
    assert [s["jpeg_quality"] for s in sections] == [95, 95, 95, 85]
    assert [s["max_image_bytes"] for s in sections] == [
        50_000_000,
        10_000_000,
        20_000_000,
        0,
    ]
    assert IMAGE_PROCESSING["max_pixels_per_page"] == 24_000_000
    custom = IMAGE_PROCESSING["custom_image_processing"]
    assert custom["resize_profile"] == "low"
    assert custom["low_max_side_px"] == 768
    assert custom["high_target_box"] == [512, 1024]


@pytest.mark.parametrize("name", PROVIDERS)
def test_sections_match_the_recommended_run_options(name: str) -> None:
    section = IMAGE_PROCESSING[f"{name}_image_processing"]
    recommended = recommended_images(FAMILIES.get(name, "custom"))
    assert section["target_dpi"] == recommended.dpi
    assert section["payload_format"] == recommended.format
    assert section["jpeg_quality"] == recommended.jpeg_quality
    assert section["grayscale_conversion"] == recommended.grayscale
    assert section[DETAIL_KEYS[name]] == recommended.detail


def test_default_fingerprint_is_unchanged() -> None:
    settings, _, section = resolve_image_settings("openai", "gpt-5.6-terra")
    assert section == "api_image_processing"
    assert settings["request_detail"] == "original"
    assert image_settings_fingerprint(settings) == GPT_56_TERRA_FINGERPRINT


def test_recommended_values_given_explicitly_keep_the_fingerprint() -> None:
    explicit = ImageSpec("native", "original", "jpeg", 95, True)
    settings, _, _ = resolve_image_settings("openai", "gpt-5.6-terra", explicit)
    assert image_settings_fingerprint(settings) == GPT_56_TERRA_FINGERPRINT


def test_run_options_override_the_section() -> None:
    images = ImageSpec(200, "low", "png", 80, False)

    google, _, _ = resolve_image_settings("google", "gemini-3-flash", images)
    assert google["target_dpi"] == 200
    assert google["payload_format"] == "png"
    assert google["jpeg_quality"] == 80
    assert google["grayscale_conversion"] is False
    assert google["media_resolution"] == "low"
    assert google["resolved_detail"] == "low"

    anthropic, _, _ = resolve_image_settings("anthropic", "claude-opus-5", images)
    assert anthropic["resize_profile"] == "low"

    custom, _, _ = resolve_image_settings("custom", "local-model", images)
    assert custom["llm_detail"] == "low"
    assert custom["resolved_detail"] == "low"

    openai, _, _ = resolve_image_settings("openai", "gpt-5.6-terra", images)
    assert openai["request_detail"] == "low"
    assert openai["llm_detail"] == "low"


def test_unset_options_keep_the_section_values() -> None:
    settings, _, _ = resolve_image_settings(
        "custom", "local-model", ImageSpec(jpeg_quality=70)
    )
    assert settings["jpeg_quality"] == 70
    assert settings["target_dpi"] == 150
    assert settings["llm_detail"] == "high"
    assert settings["render_strategy"] == "direct"
