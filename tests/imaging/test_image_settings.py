"""The recommended image sections and the run's image options."""

from __future__ import annotations

import pytest

from autoexcerpter.common.native import (
    image_settings_fingerprint,
    resolve_target_size,
)
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

# The fingerprints of the default OpenAI image settings for gpt-5.6-terra:
# preset version 2 (colour) and version 1 (grayscale). They guard resume, so
# only a new preset version may move the default; the version 1 record stays
# reachable through explicit options, which resume pins for items stored
# under it.
GPT_56_TERRA_FINGERPRINT = (
    "a944acc3d1e9601bec5bb0d57119a17377f6638cd72b417766a2ffb2706fc478"
)
GPT_56_TERRA_FINGERPRINT_V1 = (
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
    assert [s["jpeg_quality"] for s in sections] == [95, 95, 95, 95]
    assert [s["grayscale_conversion"] for s in sections] == [False] * 4
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
    explicit = ImageSpec("native", "original", "jpeg", 95, False)
    settings, _, _ = resolve_image_settings("openai", "gpt-5.6-terra", explicit)
    assert image_settings_fingerprint(settings) == GPT_56_TERRA_FINGERPRINT


def test_version_1_values_given_explicitly_give_the_version_1_fingerprint() -> None:
    explicit = ImageSpec(format="jpeg", jpeg_quality=95, grayscale=True)
    settings, _, _ = resolve_image_settings("openai", "gpt-5.6-terra", explicit)
    assert image_settings_fingerprint(settings) == GPT_56_TERRA_FINGERPRINT_V1


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


def test_luna_original_pages_are_limited_to_3000_px() -> None:
    settings, _, _ = resolve_image_settings("openai", "gpt-6-luna")
    assert settings["original_max_side_px"] == 3000
    size, reason = resolve_target_size(
        4000, 6000, "openai", "gpt-6-luna", "original", settings
    )
    assert (size, reason) == ((2000, 3000), "model_cap")


@pytest.mark.parametrize(
    "model,provider,detail",
    [
        ("gpt-6.1-sol", "openai", None),
        ("gpt-6-astra", "openai", None),
        ("gpt-5.6-luna", "openai", None),
        ("gpt-6-luna", "openai", "high"),
        ("openai/gpt-6-luna", "openrouter", None),
        ("claude-opus-5", "anthropic", None),
    ],
)
def test_other_models_and_details_keep_their_sizing(
    model: str, provider: str, detail: str | None
) -> None:
    settings, _, _ = resolve_image_settings(provider, model, ImageSpec(detail=detail))
    assert "original_max_side_px" not in settings


def test_the_setting_overrides_the_model_default() -> None:
    def side(model: str, value: int | None) -> int | None:
        settings = resolve_image_settings("openai", model, None, value)[0]
        return settings.get("original_max_side_px")

    assert side("gpt-6-luna", 2500) == 2500
    assert side("gpt-6.1-sol", 2000) == 2000
    assert side("gpt-6-luna", None) == 3000
    assert side("gpt-6.1-sol", None) is None


def test_unset_options_keep_the_section_values() -> None:
    settings, _, _ = resolve_image_settings(
        "custom", "local-model", ImageSpec(jpeg_quality=70)
    )
    assert settings["jpeg_quality"] == 70
    assert settings["target_dpi"] == 150
    assert settings["llm_detail"] == "high"
    assert settings["render_strategy"] == "direct"
