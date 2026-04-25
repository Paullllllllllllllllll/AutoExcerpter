"""Tests for llm/capabilities.py - Unified model capability detection."""

import pytest
from llm.capabilities import (
    ProviderCapabilities,
    CapabilityError,
    detect_provider,
    detect_capabilities,
    ensure_image_support,
)


class TestDetectProvider:
    """Test provider detection from model names."""

    def test_detect_openai_gpt_models(self) -> None:
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("gpt-4.1") == "openai"
        assert detect_provider("gpt-5") == "openai"
        assert detect_provider("gpt-5.1") == "openai"
        assert detect_provider("gpt-5.2") == "openai"
        assert detect_provider("gpt-5.3") == "openai"
        assert detect_provider("gpt-5.3-chat-latest") == "openai"
        assert detect_provider("gpt-5.4") == "openai"
        assert detect_provider("gpt-5.4-pro") == "openai"
        assert detect_provider("gpt-5.4-mini") == "openai"
        assert detect_provider("gpt-5.4-nano") == "openai"

    def test_detect_openai_reasoning_models(self) -> None:
        assert detect_provider("o1") == "openai"
        assert detect_provider("o1-mini") == "openai"
        assert detect_provider("o3") == "openai"
        assert detect_provider("o3-mini") == "openai"
        assert detect_provider("o3-pro") == "openai"
        assert detect_provider("o4-mini") == "openai"

    def test_detect_anthropic_models(self) -> None:
        assert detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert detect_provider("claude-3.5-sonnet") == "anthropic"
        assert detect_provider("claude-opus-4") == "anthropic"
        assert detect_provider("claude-sonnet-4") == "anthropic"
        assert detect_provider("claude-haiku-4.5") == "anthropic"

    def test_detect_google_models(self) -> None:
        assert detect_provider("gemini-2.0-flash") == "google"
        assert detect_provider("gemini-1.5-pro") == "google"
        assert detect_provider("gemini-3-pro") == "google"

    def test_detect_openrouter_models(self) -> None:
        assert detect_provider("openrouter/anthropic/claude-3.5-sonnet") == "openrouter"
        assert detect_provider("anthropic/claude-3.5-sonnet") == "openrouter"
        assert detect_provider("google/gemini-pro") == "openrouter"

    def test_detect_unknown_provider(self) -> None:
        assert detect_provider("unknown-model") == "unknown"
        assert detect_provider("some-random-model") == "unknown"

    def test_detect_provider_case_insensitive(self) -> None:
        assert detect_provider("GPT-4o") == "openai"
        assert detect_provider("CLAUDE-3.5-sonnet") == "anthropic"
        assert detect_provider("Gemini-2.0-flash") == "google"


class TestDetectCapabilities:
    """Test capability detection for different models."""

    # --- OpenAI GPT-5 family ---

    def test_gpt5_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5")
        assert caps.family == "gpt-5"
        assert caps.provider_name == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_text_verbosity is True
        assert caps.supports_temperature is False
        assert caps.supports_vision is True
        assert caps.supports_structured_output is True

    def test_gpt5_mini_inherits_family(self) -> None:
        caps = detect_capabilities("gpt-5-mini")
        assert caps.family == "gpt-5"
        assert caps.supports_text_verbosity is True

    def test_gpt51_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5.1")
        assert caps.family == "gpt-5.1"
        assert caps.is_reasoning_model is True
        assert caps.supports_text_verbosity is True

    def test_gpt52_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5.2")
        assert caps.family == "gpt-5.2"
        assert caps.max_context_tokens == 400000

    def test_gpt53_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5.3")
        assert caps.family == "gpt-5.3"
        assert caps.provider_name == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_text_verbosity is True
        assert caps.supports_temperature is False
        assert caps.supports_vision is True
        assert caps.max_context_tokens == 400000
        assert caps.max_output_tokens == 128000

    def test_gpt53_chat_latest_inherits_family(self) -> None:
        caps = detect_capabilities("gpt-5.3-chat-latest")
        assert caps.family == "gpt-5.3"
        assert caps.supports_text_verbosity is True

    def test_gpt54_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5.4")
        assert caps.family == "gpt-5.4"
        assert caps.provider_name == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_text_verbosity is True
        assert caps.supports_temperature is False
        assert caps.supports_vision is True
        assert caps.max_context_tokens == 1050000
        assert caps.max_output_tokens == 128000

    def test_gpt54_mini_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5.4-mini")
        assert caps.family == "gpt-5.4-mini"
        assert caps.provider_name == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_text_verbosity is True
        assert caps.supports_vision is True
        assert caps.supports_structured_output is True
        assert caps.max_context_tokens == 400000
        assert caps.max_output_tokens == 128000

    def test_gpt54_mini_does_not_match_gpt54(self) -> None:
        """Ensure gpt-5.4-mini is matched by its own entry, not the gpt-5.4 prefix."""
        mini = detect_capabilities("gpt-5.4-mini")
        base = detect_capabilities("gpt-5.4")
        assert mini.family == "gpt-5.4-mini"
        assert base.family == "gpt-5.4"
        assert mini.max_context_tokens == 400000
        assert base.max_context_tokens == 1050000

    def test_gpt54_nano_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5.4-nano")
        assert caps.family == "gpt-5.4-nano"
        assert caps.provider_name == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_text_verbosity is True
        assert caps.supports_vision is True
        assert caps.supports_structured_output is True
        assert caps.max_context_tokens == 400000
        assert caps.max_output_tokens == 128000

    def test_gpt54_nano_does_not_match_gpt54(self) -> None:
        """Ensure gpt-5.4-nano is matched by its own entry, not the gpt-5.4 prefix."""
        nano = detect_capabilities("gpt-5.4-nano")
        base = detect_capabilities("gpt-5.4")
        assert nano.family == "gpt-5.4-nano"
        assert base.family == "gpt-5.4"

    def test_gpt54_pro_capabilities(self) -> None:
        caps = detect_capabilities("gpt-5.4-pro")
        assert caps.family == "gpt-5.4-pro"
        assert caps.provider_name == "openai"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_text_verbosity is True
        assert caps.supports_structured_output is True
        assert caps.supports_json_mode is True
        assert caps.max_context_tokens == 1050000
        assert caps.max_output_tokens == 128000

    def test_gpt54_pro_does_not_match_gpt54(self) -> None:
        """Ensure gpt-5.4-pro is matched by its own entry, not the gpt-5.4 prefix."""
        pro = detect_capabilities("gpt-5.4-pro")
        base = detect_capabilities("gpt-5.4")
        assert pro.family == "gpt-5.4-pro"
        assert base.family == "gpt-5.4"

    # --- OpenAI GPT-4.x standard models ---

    def test_gpt4o_capabilities(self) -> None:
        caps = detect_capabilities("gpt-4o")
        assert caps.family == "gpt-4o"
        assert caps.provider_name == "openai"
        assert caps.is_reasoning_model is False
        assert caps.supports_temperature is True
        assert caps.supports_vision is True
        assert caps.supports_structured_output is True
        assert caps.supports_text_verbosity is False

    def test_gpt41_capabilities(self) -> None:
        caps = detect_capabilities("gpt-4.1")
        assert caps.family == "gpt-4.1"
        assert caps.is_reasoning_model is False
        assert caps.supports_temperature is True
        assert caps.max_context_tokens == 1000000

    # --- OpenAI o-series ---

    def test_o4_capabilities(self) -> None:
        caps = detect_capabilities("o4")
        assert caps.family == "o4"
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True
        assert caps.supports_vision is True

    def test_o3_pro_capabilities(self) -> None:
        caps = detect_capabilities("o3-pro")
        assert caps.family == "o3-pro"
        assert caps.is_reasoning_model is True
        assert caps.supports_vision is True

    def test_o3_mini_no_vision(self) -> None:
        caps = detect_capabilities("o3-mini")
        assert caps.family == "o3-mini"
        assert caps.is_reasoning_model is True
        assert caps.supports_vision is False

    def test_o3_capabilities(self) -> None:
        caps = detect_capabilities("o3")
        assert caps.family == "o3"
        assert caps.is_reasoning_model is True
        assert caps.supports_structured_output is True

    def test_o3_dated_variant(self) -> None:
        caps = detect_capabilities("o3-2025-04-16")
        assert caps.family == "o3"
        assert caps.is_reasoning_model is True

    def test_o1_capabilities(self) -> None:
        caps = detect_capabilities("o1")
        assert caps.family == "o1"
        assert caps.is_reasoning_model is True
        assert caps.supports_structured_output is True
        assert caps.supports_vision is True

    def test_o1_mini_capabilities(self) -> None:
        caps = detect_capabilities("o1-mini")
        assert caps.family == "o1-mini"
        assert caps.is_reasoning_model is True
        assert caps.supports_vision is False
        assert caps.supports_structured_output is False

    def test_o1_pro_capabilities(self) -> None:
        caps = detect_capabilities("o1-pro")
        assert caps.family == "o1-pro"
        assert caps.is_reasoning_model is True
        assert caps.supports_structured_output is False

    def test_o4_mini(self) -> None:
        caps = detect_capabilities("o4-mini")
        assert caps.family == "o4-mini"
        assert caps.is_reasoning_model is True
        assert caps.supports_vision is True

    def test_o4_mini_dated(self) -> None:
        caps = detect_capabilities("o4-mini-2025-04-16")
        assert caps.family == "o4-mini"

    def test_o4_mini_deep_research(self) -> None:
        caps = detect_capabilities("o4-mini-deep-research")
        assert caps.family == "o4-mini-deep-research"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 200000

    def test_o3_deep_research(self) -> None:
        caps = detect_capabilities("o3-deep-research")
        assert caps.family == "o3-deep-research"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 200000

    def test_gpt53_codex(self) -> None:
        caps = detect_capabilities("gpt-5.3-codex")
        assert caps.family == "gpt-5.3-codex"
        assert caps.is_reasoning_model is True
        assert caps.supports_text_verbosity is True
        assert caps.max_context_tokens == 400000

    def test_gpt52_pro(self) -> None:
        caps = detect_capabilities("gpt-5.2-pro")
        assert caps.family == "gpt-5.2-pro"
        assert caps.is_reasoning_model is True
        assert caps.supports_text_verbosity is True
        assert caps.supports_structured_output is False

    def test_gpt41_mini(self) -> None:
        caps = detect_capabilities("gpt-4.1-mini")
        assert caps.family == "gpt-4.1-mini"
        assert caps.is_reasoning_model is False
        assert caps.supports_vision is True
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 16384

    def test_gpt41_nano(self) -> None:
        caps = detect_capabilities("gpt-4.1-nano")
        assert caps.family == "gpt-4.1-nano"
        assert caps.is_reasoning_model is False
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 16384

    # --- Anthropic Claude models ---

    def test_claude_opus_47(self) -> None:
        caps = detect_capabilities("claude-opus-4-7")
        assert caps.family == "claude-opus-4.7"
        assert caps.provider_name == "anthropic"
        assert caps.is_reasoning_model is True
        assert caps.supports_top_p is False
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 128000

    def test_claude_opus_46(self) -> None:
        caps = detect_capabilities("claude-opus-4-6")
        assert caps.family == "claude-opus-4.6"
        assert caps.provider_name == "anthropic"
        assert caps.is_reasoning_model is True
        assert caps.supports_top_p is False
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 128000

    def test_claude_sonnet_46(self) -> None:
        caps = detect_capabilities("claude-sonnet-4-6")
        assert caps.family == "claude-sonnet-4.6"
        assert caps.provider_name == "anthropic"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 128000

    def test_claude_opus_45(self) -> None:
        caps = detect_capabilities("claude-opus-4-5")
        assert caps.family == "claude-opus-4.5"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 200000
        assert caps.max_output_tokens == 64000

    def test_claude_sonnet_45(self) -> None:
        caps = detect_capabilities("claude-sonnet-4-5-20250929")
        assert caps.family == "claude-sonnet-4.5"
        assert caps.is_reasoning_model is True
        assert caps.supports_vision is True
        assert caps.max_context_tokens == 1000000
        assert caps.max_output_tokens == 64000

    def test_claude_haiku_45_capabilities(self) -> None:
        caps = detect_capabilities("claude-haiku-4-5")
        assert caps.family == "claude-haiku-4.5"
        assert caps.supports_structured_output is True
        assert caps.supports_json_mode is True
        assert caps.max_output_tokens == 64000

    def test_claude_sonnet_4_not_reasoning(self) -> None:
        caps = detect_capabilities("claude-sonnet-4")
        assert caps.family == "claude-sonnet-4"
        assert caps.is_reasoning_model is False

    def test_claude_opus_4_reasoning(self) -> None:
        caps = detect_capabilities("claude-opus-4")
        assert caps.family == "claude-opus-4"
        assert caps.is_reasoning_model is True

    def test_claude_35_sonnet(self) -> None:
        caps = detect_capabilities("claude-3-5-sonnet-20241022")
        assert caps.family == "claude-3.5-sonnet"
        assert caps.provider_name == "anthropic"
        assert caps.is_reasoning_model is False
        assert caps.supports_vision is True
        assert caps.max_context_tokens == 200000

    def test_claude_3_opus_output_tokens(self) -> None:
        caps = detect_capabilities("claude-3-opus")
        assert caps.max_output_tokens == 4096

    def test_claude_fallback(self) -> None:
        caps = detect_capabilities("claude-future-model")
        assert caps.family == "claude"
        assert caps.provider_name == "anthropic"

    # --- Google Gemini models ---

    def test_gemini_31_pro(self) -> None:
        caps = detect_capabilities("gemini-3.1-pro-preview")
        assert caps.family == "gemini-3.1-pro"
        assert caps.provider_name == "google"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 1048576
        assert caps.max_output_tokens == 65536

    def test_gemini_31_flash_lite(self) -> None:
        caps = detect_capabilities("gemini-3.1-flash-lite-preview")
        assert caps.family == "gemini-3.1-flash-lite"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 1048576
        assert caps.max_output_tokens == 65536

    def test_gemini_3_flash(self) -> None:
        caps = detect_capabilities("gemini-3-flash-preview")
        assert caps.family == "gemini-3-flash"
        assert caps.provider_name == "google"
        assert caps.is_reasoning_model is True
        assert caps.supports_media_resolution is True

    def test_gemini_25_pro(self) -> None:
        caps = detect_capabilities("gemini-2.5-pro")
        assert caps.family == "gemini-2.5-pro"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 2000000

    def test_gemini_25_flash(self) -> None:
        caps = detect_capabilities("gemini-2.5-flash")
        assert caps.family == "gemini-2.5-flash"
        assert caps.is_reasoning_model is True
        assert caps.max_output_tokens == 65536

    def test_gemini_20_not_reasoning(self) -> None:
        caps = detect_capabilities("gemini-2.0-flash")
        assert caps.family == "gemini-2.0"
        assert caps.is_reasoning_model is False

    def test_gemini_15_pro(self) -> None:
        caps = detect_capabilities("gemini-1.5-pro")
        assert caps.provider_name == "google"
        assert caps.supports_vision is True
        assert caps.max_context_tokens == 2000000

    # --- OpenRouter models ---

    def test_openrouter_deepseek_r1(self) -> None:
        caps = detect_capabilities("deepseek/deepseek-r1")
        assert caps.provider_name == "openrouter"
        assert caps.family == "openrouter-deepseek"
        assert caps.is_reasoning_model is True

    def test_openrouter_gpt5(self) -> None:
        caps = detect_capabilities("openai/gpt-5")
        assert caps.family == "openrouter-gpt5"
        assert caps.is_reasoning_model is True
        assert caps.supports_temperature is False

    def test_openrouter_claude(self) -> None:
        caps = detect_capabilities("anthropic/claude-sonnet-4-5")
        assert caps.family == "openrouter-claude"
        assert caps.is_reasoning_model is True

    def test_openrouter_gemini_thinking(self) -> None:
        caps = detect_capabilities("google/gemini-2.5-flash")
        assert caps.family == "openrouter-gemini"
        assert caps.is_reasoning_model is True

    def test_openrouter_gemini_non_thinking(self) -> None:
        caps = detect_capabilities("google/gemini-2.0-flash")
        assert caps.family == "openrouter-gemini"
        assert caps.is_reasoning_model is False

    def test_openrouter_llama_vision(self) -> None:
        caps = detect_capabilities("meta/llama-3.2-90b-vision")
        assert caps.family == "openrouter-llama"
        assert caps.supports_vision is True

    def test_openrouter_llama_no_vision(self) -> None:
        caps = detect_capabilities("meta/llama-3.1-405b")
        assert caps.family == "openrouter-llama"
        assert caps.supports_vision is False

    def test_openrouter_mistral_pixtral(self) -> None:
        caps = detect_capabilities("mistral/pixtral-large")
        assert caps.family == "openrouter-mistral"
        assert caps.supports_vision is True

    def test_openrouter_fallback(self) -> None:
        caps = detect_capabilities("some-provider/some-model")
        assert caps.family == "openrouter"
        assert caps.provider_name == "openrouter"

    # --- SUZ custom endpoint models (slash in name) ---

    def test_suz_dots_mocr_detected_as_openrouter(self) -> None:
        caps = detect_capabilities("rednote-hilab/dots.mocr")
        assert caps.provider_name == "openrouter"
        assert caps.supports_vision is True

    def test_suz_dots_ocr_detected_as_openrouter(self) -> None:
        caps = detect_capabilities("kristaller486/dots.ocr-1.5")
        assert caps.provider_name == "openrouter"

    def test_suz_nanonets_detected_as_openrouter(self) -> None:
        caps = detect_capabilities("nanonets/Nanonets-OCR2-3B")
        assert caps.provider_name == "openrouter"

    # --- Fallback ---

    def test_unknown_model_fallback(self) -> None:
        caps = detect_capabilities("unknown-model-xyz")
        assert caps.family == "unknown"
        assert caps.supports_vision is False
        assert caps.is_reasoning_model is False
        assert caps.supports_temperature is True


class TestProviderCapabilitiesDataclass:
    """Test the ProviderCapabilities dataclass."""

    def test_defaults(self) -> None:
        caps = ProviderCapabilities(provider_name="test", model_name="test-model")
        assert caps.family == "unknown"
        assert caps.supports_vision is False
        assert caps.supports_text_verbosity is False
        assert caps.supports_streaming is True
        assert caps.max_context_tokens == 128000

    def test_immutable(self) -> None:
        caps = ProviderCapabilities(provider_name="test", model_name="test")
        with pytest.raises(AttributeError):
            caps.model_name = "changed"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        caps = ProviderCapabilities(
            provider_name="anthropic",
            model_name="custom",
            family="custom-family",
            is_reasoning_model=True,
            supports_temperature=False,
            max_context_tokens=200000,
        )
        assert caps.provider_name == "anthropic"
        assert caps.family == "custom-family"
        assert caps.is_reasoning_model is True
        assert caps.max_context_tokens == 200000


class TestCapabilityError:
    """Test CapabilityError and ensure_image_support."""

    def test_ensure_image_support_raises(self) -> None:
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="text-only",
            supports_vision=False,
        )
        with pytest.raises(CapabilityError, match="does not support image inputs"):
            ensure_image_support("text-only", caps)

    def test_ensure_image_support_passes(self) -> None:
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="gpt-4o",
            supports_vision=True,
        )
        ensure_image_support("gpt-4o", caps)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
