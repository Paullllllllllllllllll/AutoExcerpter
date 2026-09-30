"""Keep provider defaults in the shared image configuration layout."""

from pathlib import Path

import pytest
import yaml

from config.loader import ConfigLoader
from imaging.settings import resolve_image_settings

EXAMPLE = Path(__file__).parents[1] / "config/defaults/image_processing.example.yaml"
PROVIDERS = ["api", "anthropic", "google", "custom"]


def test_example_config_drift() -> None:
    config = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert list(config) == [
        "render_strategy",
        "max_pixels_per_page",
        "resampling_algorithm",
        *(f"{name}_image_processing" for name in PROVIDERS),
        "text_cleaning",
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
    loader = ConfigLoader()
    loader._image_processing = config
    loader._model = {"transcription_model": {"image_size": "original"}}
    for name in PROVIDERS:
        section = f"{name}_image_processing"
        detail = (
            ["media_resolution"]
            if name == "google"
            else []
            if name == "anthropic"
            else ["llm_detail"]
        )
        assert list(config[section]) == common + detail + [
            "resize_profile",
            "low_max_side_px",
            "high_target_box",
        ]
        provider = "openai" if name == "api" else name
        model = "claude-opus-5" if provider == "anthropic" else "gpt-6-astra"
        resolve_image_settings(loader, provider, model)
    assert [config[f"{p}_image_processing"]["target_dpi"] for p in PROVIDERS] == [
        "native",
        "native",
        300,
        150,
    ]
    assert [config[f"{p}_image_processing"]["jpeg_quality"] for p in PROVIDERS] == [
        95,
        95,
        95,
        85,
    ]
    assert [config[f"{p}_image_processing"]["max_image_bytes"] for p in PROVIDERS] == [
        50000000,
        10000000,
        20000000,
        0,
    ]
    assert config["max_pixels_per_page"] == 24000000
    custom = config["custom_image_processing"]
    assert custom["resize_profile"] == "low"
    assert custom["low_max_side_px"] == 768
    assert custom["high_target_box"] == [512, 1024]


def test_real_section_wins_and_top_level_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import config.loader as loader_module

    (tmp_path / EXAMPLE.name).write_text(
        EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "image_processing.yaml").write_text(
        "target_dpi: 72\n"
        "native_fallback_dpi: 72\n"
        "payload_format: png\n"
        "max_image_bytes: 1\n"
        "google_image_processing:\n"
        "  target_dpi: native\n"
        "  render_strategy: supersample\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader_module, "CONFIG_DIR", tmp_path)
    loader = ConfigLoader()
    loader._image_processing = loader._load_yaml_config("image_processing.yaml")
    google, _, _ = resolve_image_settings(loader, "google", "gemini-3-flash")
    assert google["target_dpi"] == "native"
    assert google["native_fallback_dpi"] == 300
    assert google["payload_format"] == "jpeg"
    assert google["max_image_bytes"] == 20000000
    assert google["render_strategy"] == "supersample"
    custom, _, _ = resolve_image_settings(loader, "custom", "local-model")
    assert custom["target_dpi"] == 150
    assert custom["render_strategy"] == "direct"
