"""Tests for modules/model_utils.py - Model detection utilities."""

from __future__ import annotations

import pytest

from modules.model_utils import (
    detect_model_type,
    get_image_config_section_name,
    ModelType,
)


class TestDetectModelType:
    """Tests for detect_model_type function."""

    # ========================================================================
    # Direct Provider Tests
    # ========================================================================
    def test_openai_provider_direct(self):
        """Direct OpenAI provider returns 'openai' type."""
        assert detect_model_type("openai", "gpt-5-mini") == "openai"
        assert detect_model_type("openai", "gpt-4o") == "openai"
        assert detect_model_type("OpenAI", "gpt-5") == "openai"

    def test_google_provider_direct(self):
        """Direct Google provider returns 'google' type."""
        assert detect_model_type("google", "gemini-2.5-flash") == "google"
        assert detect_model_type("Google", "gemini-1.5-pro") == "google"
        assert detect_model_type("GOOGLE", None) == "google"

    def test_anthropic_provider_direct(self):
        """Direct Anthropic provider returns 'anthropic' type."""
        assert detect_model_type("anthropic", "claude-sonnet-4-5") == "anthropic"
        assert detect_model_type("Anthropic", "claude-3-opus") == "anthropic"
        assert detect_model_type("ANTHROPIC", None) == "anthropic"

    # ========================================================================
    # OpenRouter Passthrough Tests
    # ========================================================================
    def test_openrouter_with_google_model(self):
        """OpenRouter with Google model name returns 'google' type."""
        assert detect_model_type("openrouter", "google/gemini-2.5-flash") == "google"
        assert detect_model_type("openrouter", "gemini-2.5-pro") == "google"

    def test_openrouter_with_anthropic_model(self):
        """OpenRouter with Anthropic model name returns 'anthropic' type."""
        assert detect_model_type("openrouter", "anthropic/claude-3-opus") == "anthropic"
        assert detect_model_type("openrouter", "claude-sonnet-4-5") == "anthropic"

    def test_openrouter_with_openai_model(self):
        """OpenRouter with OpenAI model name returns 'openai' type."""
        assert detect_model_type("openrouter", "openai/gpt-5") == "openai"
        assert detect_model_type("openrouter", "gpt-4o-mini") == "openai"
        assert detect_model_type("openrouter", "o3-mini") == "openai"

    # ========================================================================
    # Model Name Detection Tests
    # ========================================================================
    def test_model_name_detection_gemini(self):
        """Gemini in model name detected as google."""
        assert detect_model_type("unknown", "gemini-2.5-flash") == "google"
        assert detect_model_type("unknown", "gemini-1.5-pro") == "google"

    def test_model_name_detection_claude(self):
        """Claude in model name detected as anthropic."""
        assert detect_model_type("unknown", "claude-3-haiku") == "anthropic"
        assert detect_model_type("unknown", "claude-opus-4-5") == "anthropic"

    def test_model_name_detection_gpt(self):
        """GPT in model name detected as openai."""
        assert detect_model_type("unknown", "gpt-5-mini") == "openai"
        assert detect_model_type("unknown", "gpt-4.1-nano") == "openai"

    def test_model_name_detection_o_series(self):
        """O-series models detected as openai."""
        assert detect_model_type("unknown", "o1-mini") == "openai"
        assert detect_model_type("unknown", "o3") == "openai"
        assert detect_model_type("unknown", "o4-mini") == "openai"

    # ========================================================================
    # Default/Fallback Tests
    # ========================================================================
    def test_unknown_provider_and_model_defaults_to_openai(self):
        """Unknown provider and model defaults to 'openai'."""
        assert detect_model_type("unknown", "unknown-model") == "openai"
        assert detect_model_type("custom", None) == "openai"
        assert detect_model_type("", "") == "openai"

    # ========================================================================
    # Case Sensitivity Tests
    # ========================================================================
    def test_case_insensitive_provider(self):
        """Provider detection is case-insensitive."""
        assert detect_model_type("OPENAI", "gpt-5") == "openai"
        assert detect_model_type("Google", "gemini") == "google"
        assert detect_model_type("AnThRoPiC", "claude") == "anthropic"

    def test_case_insensitive_model_name(self):
        """Model name detection is case-insensitive."""
        assert detect_model_type("openrouter", "GEMINI-2.5-flash") == "google"
        assert detect_model_type("openrouter", "CLAUDE-3-opus") == "anthropic"
        assert detect_model_type("openrouter", "GPT-5") == "openai"


class TestGetImageConfigSectionName:
    """Tests for get_image_config_section_name function."""

    def test_openai_section_name(self):
        """OpenAI model type returns api_image_processing section."""
        assert get_image_config_section_name("openai") == "api_image_processing"

    def test_google_section_name(self):
        """Google model type returns google_image_processing section."""
        assert get_image_config_section_name("google") == "google_image_processing"

    def test_anthropic_section_name(self):
        """Anthropic model type returns anthropic_image_processing section."""
        assert (
            get_image_config_section_name("anthropic") == "anthropic_image_processing"
        )

    def test_unknown_type_defaults_to_openai(self):
        """Unknown type defaults to api_image_processing section."""
        # Note: This tests the else branch - any non-matching type defaults to OpenAI
        # In practice, detect_model_type always returns a valid ModelType
        result = get_image_config_section_name("unknown")  # type: ignore
        assert result == "api_image_processing"
