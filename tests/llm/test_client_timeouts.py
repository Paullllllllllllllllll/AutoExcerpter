"""Timeout wiring of llm/client.py per provider.

The three ChatOpenAI-based creators receive an ``httpx.Timeout`` with
per-phase budgets, while the Anthropic and Google wrappers keep the scalar
they require.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx

from autoexcerpter.common.http_timeouts import DEFAULT_WRITE_TIMEOUT
from autoexcerpter.llm.client import LLMConfig, get_chat_model


class TestProviderTimeoutWiring:
    """The OpenAI-family creators get httpx.Timeout; the others a scalar."""

    def test_openai_receives_httpx_timeout(self, mock_api_keys: None) -> None:
        """ChatOpenAI is constructed with per-phase budgets."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(
                    model="gpt-5-mini",
                    provider="openai",
                    timeout=300,
                    connect_timeout=4.0,
                )
            )

        timeout = mock_class.call_args[1]["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 300
        assert timeout.connect == 4.0
        assert timeout.write == DEFAULT_WRITE_TIMEOUT

    def test_openrouter_receives_httpx_timeout(self, mock_api_keys: None) -> None:
        """The OpenRouter branch shares the ChatOpenAI treatment."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(
                    model="anthropic/claude-3-opus",
                    provider="openrouter",
                    timeout=120,
                )
            )

        timeout = mock_class.call_args[1]["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 120

    def test_custom_endpoint_receives_httpx_timeout(self) -> None:
        """A custom OpenAI-compatible endpoint is treated the same way."""
        with (
            patch.dict(os.environ, {"CUSTOM_KEY": "secret"}, clear=False),
            patch("langchain_openai.ChatOpenAI") as mock_class,
        ):
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(
                    model="local-model",
                    provider="custom",
                    api_key_env="CUSTOM_KEY",
                    base_url="https://example.invalid/v1",
                    timeout=45,
                )
            )

        kwargs = mock_class.call_args[1]
        assert isinstance(kwargs["timeout"], httpx.Timeout)
        assert kwargs["timeout"].read == 45
        assert kwargs["base_url"] == "https://example.invalid/v1"
        assert kwargs["api_key"] == "secret"

    def test_anthropic_keeps_scalar_timeout(self, mock_api_keys: None) -> None:
        """ChatAnthropic compares its timeout with ``> 0``; keep the float."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(model="claude-3-opus", provider="anthropic", timeout=300)
            )

        assert mock_class.call_args[1]["timeout"] == 300

    def test_google_keeps_scalar_timeout(self, mock_api_keys: None) -> None:
        """ChatGoogleGenerativeAI computes int(timeout * 1000)."""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(model="gemini-2.5-flash", provider="google", timeout=300)
            )

        assert mock_class.call_args[1]["timeout"] == 300
