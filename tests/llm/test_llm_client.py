"""Tests for llm/client.py - Multi-provider LLM client."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import autoexcerpter.llm.client as client_module
from autoexcerpter.llm.client import (
    OPENROUTER_BASE_URL,
    LLMConfig,
    _get_api_key,
    build_chat_model,
    default_key_env,
    get_chat_model,
    resolve_provider,
)
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.settings import Timeouts


class TestBuildChatModel:
    """The phase factory: one client from a phase and the timeouts."""

    def test_builds_with_the_phase_key_and_timeouts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALT_KEY", "alt-secret")
        captured: dict[str, Any] = {}

        def fake_get_chat_model(config: LLMConfig) -> str:
            captured["config"] = config
            return "model"

        monkeypatch.setattr(client_module, "get_chat_model", fake_get_chat_model)
        phase = PhaseModel("gpt-5-mini", "openai", api_key_env="ALT_KEY")

        built: Any = build_chat_model(phase, Timeouts(request=600.0))
        assert built == "model"
        config = captured["config"]
        assert config.provider == "openai"
        assert config.timeout == 600
        assert isinstance(config.timeout, int)
        assert config.api_key_env == "ALT_KEY"

        build_chat_model(phase, Timeouts(), api_key_env="OTHER_KEY")
        assert captured["config"].api_key_env == "OTHER_KEY"

    def test_sdk_retries_are_off(self, mock_api_keys: None) -> None:
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(LLMConfig(model="gpt-5-mini", provider="openai"))
        assert mock_class.call_args[1]["max_retries"] == 0

    def test_fine_tuned_ids_keep_their_colons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        model_id = "ft:gpt-4o-mini-2024-07-18:acme:culinary:abc123"
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(LLMConfig(model=model_id, provider="openai"))
        assert mock_class.call_args[1]["model"] == model_id

    def test_resolve_provider_returns_the_phase_provider(self) -> None:
        assert resolve_provider(PhaseModel("claude-sonnet-4-5", "openrouter")) == (
            "openrouter"
        )
        assert resolve_provider(PhaseModel("org/model", "custom")) == "custom"
        assert default_key_env("google") == "GOOGLE_API_KEY"
        assert default_key_env("custom") is None


class TestGetApiKey:
    """Tests for _get_api_key function."""

    def test_uses_env_variable(self, mock_api_keys: None) -> None:
        """Uses the provider's default variable."""
        assert _get_api_key("openai") == "test-openai-key"

    def test_raises_on_missing_key(self) -> None:
        """Raises EnvironmentError when key not found."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(EnvironmentError, match="API key not found"),
        ):
            _get_api_key("openai")

    def test_named_key_variable_wins(self, mock_api_keys: None) -> None:
        """The settings may point a provider at another key variable."""
        with patch.dict(os.environ, {"ALT_OPENAI_KEY": "test-alt-key"}):
            assert _get_api_key("openai", "ALT_OPENAI_KEY") == "test-alt-key"

    def test_named_key_variable_must_be_set(self, mock_api_keys: None) -> None:
        """A named variable that is unset is an error naming the variable."""
        with pytest.raises(EnvironmentError, match="ALT_OPENAI_KEY"):
            _get_api_key("openai", "ALT_OPENAI_KEY")

    def test_custom_endpoint_uses_its_key_variable(self) -> None:
        """A custom endpoint reads the variable the named endpoint declares."""
        with patch.dict(
            os.environ, {"TRANS_KEY": "trans-val", "SUM_KEY": "sum-val"}, clear=True
        ):
            assert _get_api_key("custom", "TRANS_KEY") == "trans-val"
            assert _get_api_key("custom", "SUM_KEY") == "sum-val"

    def test_custom_endpoint_without_key_variable_raises(self) -> None:
        """A custom provider needs a named endpoint's key variable."""
        with pytest.raises(EnvironmentError, match="api_key_env"):
            _get_api_key("custom")


class TestGetChatModel:
    """Tests for get_chat_model function."""

    def test_creates_openai_model(self, mock_api_keys: None) -> None:
        """Creates OpenAI model with correct class."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()

            config = LLMConfig(model="gpt-5-mini", provider="openai")
            get_chat_model(config)

            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["model"] == "gpt-5-mini"
            assert call_kwargs["api_key"] == "test-openai-key"
            assert call_kwargs["use_responses_api"] is True
            assert "service_tier" not in call_kwargs

    def test_creates_anthropic_model(self, mock_api_keys: None) -> None:
        """Creates Anthropic model with correct class."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_class:
            mock_class.return_value = MagicMock()

            config = LLMConfig(model="claude-3-opus", provider="anthropic")
            get_chat_model(config)

            mock_class.assert_called_once()

    def test_creates_google_model(self, mock_api_keys: None) -> None:
        """Creates Google model with correct class."""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_class:
            mock_class.return_value = MagicMock()

            config = LLMConfig(model="gemini-2.5-flash", provider="google")
            get_chat_model(config)

            mock_class.assert_called_once()

    def test_creates_openrouter_model(self, mock_api_keys: None) -> None:
        """Creates OpenRouter model with correct base URL."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()

            config = LLMConfig(model="anthropic/claude-3-opus", provider="openrouter")
            get_chat_model(config)

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["base_url"] == OPENROUTER_BASE_URL
            assert call_kwargs["default_headers"]["X-Title"] == "AutoExcerpter"

    def test_custom_without_base_url_raises(self) -> None:
        """A custom provider needs the endpoint's base URL."""
        with (
            patch.dict(os.environ, {"LOCAL_KEY": "local"}),
            pytest.raises(ValueError, match="base_url"),
        ):
            get_chat_model(
                LLMConfig(model="m", provider="custom", api_key_env="LOCAL_KEY")
            )

    def test_unsupported_provider_raises(self, mock_api_keys: None) -> None:
        """Unsupported provider raises ValueError."""
        config = LLMConfig(model="test", provider="unsupported")

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_chat_model(config)
