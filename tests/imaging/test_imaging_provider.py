"""Model-type detection and image settings sections in imaging/settings.py."""

from __future__ import annotations

import pytest

from autoexcerpter.imaging.settings import (
    detect_model_type,
    get_image_config_section_name,
)


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        # A direct provider decides, in any case, with or without a model.
        ("openai", "gpt-5-mini", "openai"),
        ("OpenAI", "gpt-5", "openai"),
        ("google", "gemini-2.5-flash", "google"),
        ("GOOGLE", None, "google"),
        ("anthropic", "claude-sonnet-4-5", "anthropic"),
        ("AnThRoPiC", "claude", "anthropic"),
        ("ANTHROPIC", None, "anthropic"),
        ("custom", None, "custom"),
        ("custom", "org/model-name", "custom"),
        # OpenRouter and unknown providers are detected from the model name.
        ("openrouter", "google/gemini-2.5-flash", "google"),
        ("openrouter", "gemini-2.5-pro", "google"),
        ("openrouter", "GEMINI-2.5-flash", "google"),
        ("openrouter", "anthropic/claude-3-opus", "anthropic"),
        ("openrouter", "claude-sonnet-4-5", "anthropic"),
        ("openrouter", "CLAUDE-3-opus", "anthropic"),
        ("openrouter", "openai/gpt-5", "openai"),
        ("openrouter", "gpt-4o-mini", "openai"),
        ("openrouter", "GPT-5", "openai"),
        ("openrouter", "o3-mini", "openai"),
        ("unknown", "gemini-1.5-pro", "google"),
        ("unknown", "claude-3-haiku", "anthropic"),
        ("unknown", "claude-opus-4-5", "anthropic"),
        ("unknown", "gpt-4.1-nano", "openai"),
        ("unknown", "o1-mini", "openai"),
        ("unknown", "o4-mini", "openai"),
        # Anything else falls back to OpenAI.
        ("unknown", "unknown-model", "openai"),
        ("", "", "openai"),
    ],
)
def test_detect_model_type(provider: str, model: str | None, expected: str) -> None:
    assert detect_model_type(provider, model) == expected


@pytest.mark.parametrize(
    ("model_type", "section"),
    [
        ("openai", "api_image_processing"),
        ("google", "google_image_processing"),
        ("anthropic", "anthropic_image_processing"),
        ("custom", "custom_image_processing"),
        ("unknown", "api_image_processing"),
    ],
)
def test_get_image_config_section_name(model_type: str, section: str) -> None:
    assert get_image_config_section_name(model_type) == section  # type: ignore[arg-type]
