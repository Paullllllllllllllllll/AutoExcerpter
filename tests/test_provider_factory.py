"""Tests for api/providers/factory.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from api.providers.factory import (
    ProviderType,
    detect_provider_from_model,
    get_api_key_for_provider,
    get_available_providers,
    get_provider,
)


class TestDetectProviderFromModel:
    def test_openrouter_prefix_format_detected(self):
        assert detect_provider_from_model("anthropic/claude-sonnet-4-5") == ProviderType.OPENROUTER

    def test_openai_detected(self):
        assert detect_provider_from_model("gpt-5-mini") == ProviderType.OPENAI
        assert detect_provider_from_model("o3-mini") == ProviderType.OPENAI

    def test_anthropic_detected(self):
        assert detect_provider_from_model("claude-sonnet-4-5") == ProviderType.ANTHROPIC

    def test_google_detected(self):
        assert detect_provider_from_model("gemini-2.5-flash") == ProviderType.GOOGLE

    def test_common_openrouter_models_detected(self):
        assert detect_provider_from_model("llama-3.3-70b") == ProviderType.OPENROUTER


class TestGetApiKeyForProvider:
    def test_uses_explicit_key(self):
        assert get_api_key_for_provider(ProviderType.OPENAI, api_key="k") == "k"

    def test_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        assert get_api_key_for_provider(ProviderType.OPENAI) == "test"

    def test_openrouter_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai")
        assert get_api_key_for_provider(ProviderType.OPENROUTER) == "openai"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key found"):
            get_api_key_for_provider(ProviderType.OPENROUTER)


class TestGetAvailableProviders:
    def test_returns_only_providers_with_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("GOOGLE_API_KEY", "y")

        providers = get_available_providers()
        assert ProviderType.OPENAI in providers
        assert ProviderType.GOOGLE in providers
        assert ProviderType.ANTHROPIC not in providers


class TestGetProvider:
    def test_get_provider_instantiates_provider_class(self, monkeypatch):
        @dataclass
        class DummyProvider:
            api_key: str
            model: str
            temperature: float = 0.0
            max_tokens: int = 4096
            timeout: float | None = None
            extra: dict[str, Any] | None = None

            def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 4096, timeout: float | None = None, **kwargs):
                self.api_key = api_key
                self.model = model
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.timeout = timeout
                self.extra = kwargs

        # Avoid reading real config defaults
        mock_cfg_loader = MagicMock()
        mock_cfg_loader.get_model_config.return_value = {
            "transcription_model": {"name": "gpt-5-mini", "provider": "openai"}
        }
        monkeypatch.setattr("modules.config_loader.get_config_loader", lambda: mock_cfg_loader)

        monkeypatch.setattr("api.providers.factory._import_provider_class", lambda _t: DummyProvider)
        monkeypatch.setattr("api.providers.factory.get_api_key_for_provider", lambda _t, _k=None: "api-key")

        provider = get_provider(provider=None, model="gpt-5-mini")
        assert isinstance(provider, DummyProvider)
        assert provider.api_key == "api-key"
        assert provider.model == "gpt-5-mini"
