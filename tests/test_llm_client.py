"""Tests for api/llm_client.py - Multi-provider LLM client."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from api.llm_client import (
    LLMConfig,
    get_chat_model,
    get_model_capabilities,
    get_provider_for_model,
    is_provider_available,
    get_available_providers,
    SUPPORTED_PROVIDERS,
    _infer_provider,
    _get_api_key,
)
from api.model_capabilities import detect_capabilities


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_basic_init(self):
        """Basic initialization with model name."""
        config = LLMConfig(model="gpt-5-mini")
        
        assert config.model == "gpt-5-mini"
        assert config.provider == "openai"  # Auto-inferred

    def test_explicit_provider(self):
        """Explicit provider is used."""
        config = LLMConfig(model="gpt-5-mini", provider="openai")
        
        assert config.provider == "openai"

    def test_provider_inferred_from_model(self):
        """Provider is inferred from model name."""
        openai_config = LLMConfig(model="gpt-4o")
        assert openai_config.provider == "openai"
        
        anthropic_config = LLMConfig(model="claude-sonnet-4-5")
        assert anthropic_config.provider == "anthropic"
        
        google_config = LLMConfig(model="gemini-2.5-flash")
        assert google_config.provider == "google"

    def test_default_values(self):
        """Default values are set correctly."""
        config = LLMConfig(model="gpt-5")
        
        assert config.timeout == 900
        assert config.max_retries == 5
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.service_tier is None
        assert config.extra_kwargs == {}

    def test_custom_values(self):
        """Custom values are set correctly."""
        config = LLMConfig(
            model="gpt-5",
            temperature=0.7,
            max_tokens=4096,
            timeout=600,
            service_tier="flex",
        )
        
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 600
        assert config.service_tier == "flex"


class TestInferProvider:
    """Tests for _infer_provider function."""

    def test_explicit_prefix_openai(self):
        """Explicit openai: prefix is detected."""
        assert _infer_provider("openai:gpt-5") == "openai"

    def test_explicit_prefix_anthropic(self):
        """Explicit anthropic: prefix is detected."""
        assert _infer_provider("anthropic:claude-3-opus") == "anthropic"

    def test_explicit_prefix_google(self):
        """Explicit google: prefix is detected."""
        assert _infer_provider("google:gemini-2.5-flash") == "google"

    def test_gpt_models(self):
        """GPT models inferred as OpenAI."""
        assert _infer_provider("gpt-5-mini") == "openai"
        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gpt-4.1-nano") == "openai"

    def test_o_series_models(self):
        """O-series models inferred as OpenAI."""
        assert _infer_provider("o1-mini") == "openai"
        assert _infer_provider("o3") == "openai"
        assert _infer_provider("o4-mini") == "openai"

    def test_claude_models(self):
        """Claude models inferred as Anthropic."""
        assert _infer_provider("claude-sonnet-4-5") == "anthropic"
        assert _infer_provider("claude-3-opus") == "anthropic"
        assert _infer_provider("claude-3-5-haiku") == "anthropic"

    def test_gemini_models(self):
        """Gemini models inferred as Google."""
        assert _infer_provider("gemini-2.5-flash") == "google"
        assert _infer_provider("gemini-1.5-pro") == "google"
        assert _infer_provider("gemini-3-pro") == "google"

    def test_unknown_defaults_to_openai(self):
        """Unknown models default to OpenAI."""
        assert _infer_provider("unknown-model") == "openai"
        assert _infer_provider("custom-model-v1") == "openai"


class TestGetModelCapabilities:
    """Tests for get_model_capabilities function."""

    def test_gpt5_capabilities(self):
        """GPT-5 models have reasoning and multimodal."""
        caps = get_model_capabilities("gpt-5-mini")
        
        assert caps.get("reasoning") is True
        assert caps.get("multimodal") is True
        # GPT-5 is a reasoning model; temperature is not supported
        assert caps.get("temperature") is False

    def test_o_series_no_temperature(self):
        """O-series models don't support temperature."""
        caps = get_model_capabilities("o3-mini")
        
        assert caps.get("reasoning") is True
        assert caps.get("temperature") is False

    def test_gpt4o_no_reasoning(self):
        """GPT-4o models don't support reasoning."""
        caps = get_model_capabilities("gpt-4o")
        
        assert caps.get("reasoning") is False
        assert caps.get("multimodal") is True

    def test_claude_extended_thinking(self):
        """Claude 4.5 opus/sonnet/haiku have extended thinking (all are reasoning models)."""
        opus_caps = get_model_capabilities("claude-opus-4-5")
        sonnet_caps = get_model_capabilities("claude-sonnet-4-5")
        haiku_caps = get_model_capabilities("claude-haiku-4-5")
        
        assert opus_caps.get("extended_thinking") is True
        assert sonnet_caps.get("extended_thinking") is True
        # Haiku 4.5 is also a reasoning Anthropic model
        assert haiku_caps.get("extended_thinking") is True

    def test_gemini_thinking(self):
        """Gemini 2.5+ models have thinking capability."""
        caps = get_model_capabilities("gemini-2.5-flash")
        
        assert caps.get("thinking") is True

    def test_gemini_2_no_thinking(self):
        """Gemini 2.0 models don't have thinking."""
        caps = get_model_capabilities("gemini-2.0-flash")
        
        assert caps.get("thinking") is False

    def test_unknown_model_defaults(self):
        """Unknown models get conservative defaults."""
        caps = get_model_capabilities("unknown-model")
        
        assert caps.get("reasoning") is False
        assert caps.get("multimodal") is False
        assert caps.get("temperature") is True

    def test_versioned_model_names(self):
        """Versioned model names are matched correctly."""
        # Claude with date suffix
        caps = get_model_capabilities("claude-sonnet-4-5-20250929")
        assert caps.get("multimodal") is True


class TestGetApiKey:
    """Tests for _get_api_key function."""

    def test_uses_provided_key(self, mock_api_keys):
        """Uses provided API key when given."""
        result = _get_api_key("openai", "custom-key")
        assert result == "custom-key"

    def test_uses_env_variable(self, mock_api_keys):
        """Uses environment variable when no key provided."""
        result = _get_api_key("openai", None)
        assert result == "test-openai-key"

    def test_raises_on_missing_key(self):
        """Raises EnvironmentError when key not found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="API key not found"):
                _get_api_key("openai", None)


class TestGetChatModel:
    """Tests for get_chat_model function."""

    def test_creates_openai_model(self, mock_api_keys):
        """Creates OpenAI model with correct class."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            
            config = LLMConfig(model="gpt-5-mini", provider="openai")
            model = get_chat_model(config)
            
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["model"] == "gpt-5-mini"
            assert call_kwargs["api_key"] == "test-openai-key"

    def test_creates_anthropic_model(self, mock_api_keys):
        """Creates Anthropic model with correct class."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_class:
            mock_class.return_value = MagicMock()
            
            config = LLMConfig(model="claude-3-opus", provider="anthropic")
            model = get_chat_model(config)
            
            mock_class.assert_called_once()

    def test_creates_google_model(self, mock_api_keys):
        """Creates Google model with correct class."""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_class:
            mock_class.return_value = MagicMock()
            
            config = LLMConfig(model="gemini-2.5-flash", provider="google")
            model = get_chat_model(config)
            
            mock_class.assert_called_once()

    def test_creates_openrouter_model(self, mock_api_keys):
        """Creates OpenRouter model with correct base URL."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            
            config = LLMConfig(model="anthropic/claude-3-opus", provider="openrouter")
            model = get_chat_model(config)
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_includes_service_tier_for_openai(self, mock_api_keys):
        """Includes service_tier parameter for OpenAI."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            
            config = LLMConfig(model="gpt-5-mini", provider="openai", service_tier="flex")
            get_chat_model(config)
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["service_tier"] == "flex"

    def test_unsupported_provider_raises(self, mock_api_keys):
        """Unsupported provider raises ValueError."""
        config = LLMConfig(model="test", provider="unsupported")  # type: ignore
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_chat_model(config)

    def test_removes_provider_prefix_from_model(self, mock_api_keys):
        """Provider prefix is removed from model name."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            
            config = LLMConfig(model="openai:gpt-5-mini")
            get_chat_model(config)
            
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["model"] == "gpt-5-mini"


class TestGetProviderForModel:
    """Tests for get_provider_for_model function."""

    def test_returns_provider(self):
        """Returns correct provider for model."""
        assert get_provider_for_model("gpt-5") == "openai"
        assert get_provider_for_model("claude-3-opus") == "anthropic"
        assert get_provider_for_model("gemini-2.5-flash") == "google"


class TestIsProviderAvailable:
    """Tests for is_provider_available function."""

    def test_returns_true_when_key_set(self, mock_api_keys):
        """Returns True when API key is available."""
        assert is_provider_available("openai") is True
        assert is_provider_available("anthropic") is True
        assert is_provider_available("google") is True

    def test_returns_false_when_key_not_set(self):
        """Returns False when API key is not available."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_provider_available("openai") is False


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""

    def test_returns_available_providers(self, mock_api_keys):
        """Returns list of providers with available keys."""
        providers = get_available_providers()
        
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "openrouter" in providers

    def test_returns_empty_when_no_keys(self):
        """Returns empty list when no API keys set."""
        with patch.dict(os.environ, {}, clear=True):
            providers = get_available_providers()
            assert providers == []


class TestSupportedProviders:
    """Tests for SUPPORTED_PROVIDERS constant."""

    def test_has_required_providers(self):
        """Contains all required providers."""
        assert "openai" in SUPPORTED_PROVIDERS
        assert "anthropic" in SUPPORTED_PROVIDERS
        assert "google" in SUPPORTED_PROVIDERS
        assert "openrouter" in SUPPORTED_PROVIDERS

    def test_provider_has_required_keys(self):
        """Each provider has required configuration keys."""
        for provider, config in SUPPORTED_PROVIDERS.items():
            assert "package" in config
            assert "class" in config
            assert "env_key" in config


class TestModelCapabilities:
    """Tests for detect_capabilities (replaces MODEL_CAPABILITIES dict)."""

    def test_has_openai_models(self):
        """detect_capabilities returns valid results for OpenAI models."""
        caps = detect_capabilities("gpt-5-mini")
        assert caps.provider_name == "openai"
        caps = detect_capabilities("gpt-5")
        assert caps.is_reasoning_model is True
        caps = detect_capabilities("o3-mini")
        assert caps.is_reasoning_model is True

    def test_has_anthropic_models(self):
        """detect_capabilities returns valid results for Anthropic models."""
        caps = detect_capabilities("claude-sonnet-4-5")
        assert caps.provider_name == "anthropic"
        caps = detect_capabilities("claude-3-opus")
        assert caps.provider_name == "anthropic"

    def test_has_google_models(self):
        """detect_capabilities returns valid results for Google models."""
        caps = detect_capabilities("gemini-2.5-flash")
        assert caps.provider_name == "google"
        caps = detect_capabilities("gemini-1.5-pro")
        assert caps.provider_name == "google"
