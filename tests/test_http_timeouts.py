"""Tests for llm/http_timeouts.py and its wiring into llm/client.py.

Covers the per-phase httpx timeout builder -- the read phase carries the
configured request timeout while connect/write/pool keep their own tight
budgets, config overrides are validated, and the documented unhashability of
``httpx.Timeout`` is pinned -- plus the provider split: the three
ChatOpenAI-based creators receive an ``httpx.Timeout`` while the Anthropic and
Google wrappers keep the scalar they require.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from llm.client import LLMConfig, get_chat_model
from llm.http_timeouts import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    build_httpx_timeout,
)


def _with_api_requests_config(cfg: dict[str, Any]) -> Any:
    """Patch the config loader so ``api_requests`` resolves to *cfg*."""
    mock_loader = MagicMock()
    mock_loader.return_value.get_concurrency_config.return_value = {"api_requests": cfg}
    return patch("llm.http_timeouts.get_config_loader", mock_loader)


class TestBuildHttpxTimeout:
    """Tests for build_httpx_timeout."""

    def test_none_is_passed_through(self) -> None:
        """No configured timeout means the SDK defaults stay untouched."""
        with _with_api_requests_config({}):
            assert build_httpx_timeout(None) is None

    def test_defaults_apply_to_the_other_phases(self) -> None:
        """The scalar lands on read only; connect/write/pool use defaults."""
        with _with_api_requests_config({}):
            timeout = build_httpx_timeout(900)

        assert timeout is not None
        assert timeout.read == 900
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT == 10.0
        assert timeout.write == DEFAULT_WRITE_TIMEOUT == 30.0
        assert timeout.pool == DEFAULT_POOL_TIMEOUT == 30.0

    def test_config_overrides_are_honored(self) -> None:
        """Configured per-phase values replace the built-in defaults."""
        with _with_api_requests_config(
            {"connect_timeout": 3, "write_timeout": 7.5, "pool_timeout": 12}
        ):
            timeout = build_httpx_timeout(120)

        assert timeout is not None
        assert timeout.read == 120
        assert timeout.connect == 3.0
        assert timeout.write == 7.5
        assert timeout.pool == 12.0

    @pytest.mark.parametrize("bad", [0, -1, "nonsense", None, True, False, [5]])
    def test_invalid_overrides_fall_back_to_defaults(self, bad: Any) -> None:
        """Zero, negative, non-numeric and bool overrides are ignored."""
        with _with_api_requests_config(
            {"connect_timeout": bad, "write_timeout": bad, "pool_timeout": bad}
        ):
            timeout = build_httpx_timeout(900)

        assert timeout is not None
        assert timeout.read == 900
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
        assert timeout.write == DEFAULT_WRITE_TIMEOUT
        assert timeout.pool == DEFAULT_POOL_TIMEOUT

    def test_non_mapping_section_falls_back_to_defaults(self) -> None:
        """A hand-edited scalar under ``api_requests`` is ignored."""
        with _with_api_requests_config("oops"):  # type: ignore[arg-type]
            timeout = build_httpx_timeout(60)

        assert timeout is not None
        assert timeout.read == 60
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT

    def test_broken_config_loader_falls_back_to_defaults(self) -> None:
        """Any loader failure leaves the built-in per-phase budgets in place."""
        mock_loader = MagicMock()
        mock_loader.return_value.get_concurrency_config.side_effect = AttributeError(
            "x"
        )
        with patch("llm.http_timeouts.get_config_loader", mock_loader):
            timeout = build_httpx_timeout(60)

        assert timeout is not None
        assert timeout.read == 60
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
        assert timeout.write == DEFAULT_WRITE_TIMEOUT
        assert timeout.pool == DEFAULT_POOL_TIMEOUT

    def test_timeout_is_unhashable(self) -> None:
        """httpx.Timeout is unhashable, bypassing langchain's client cache.

        Each chat model therefore builds its own httpx client (accepted: the
        SDK wrapper closes its pool on GC and managers are per item). The
        bypass lives in ``langchain_openai/chat_models/_client_utils.py``; if
        this test ever starts failing, httpx made Timeout hashable and the
        client-sharing semantics changed.
        """
        with _with_api_requests_config({}):
            timeout = build_httpx_timeout(900)

        assert isinstance(timeout, httpx.Timeout)
        with pytest.raises(TypeError):
            hash(timeout)


class TestProviderTimeoutWiring:
    """The OpenAI-family creators get httpx.Timeout; the others a scalar."""

    def test_openai_receives_httpx_timeout(self, mock_api_keys) -> None:
        """ChatOpenAI is constructed with per-phase budgets."""
        with (
            _with_api_requests_config({}),
            patch("langchain_openai.ChatOpenAI") as mock_class,
        ):
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(model="gpt-5-mini", provider="openai", timeout=300)
            )

        timeout = mock_class.call_args[1]["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 300
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT

    def test_openrouter_receives_httpx_timeout(self, mock_api_keys) -> None:
        """The OpenRouter branch shares the ChatOpenAI treatment."""
        with (
            _with_api_requests_config({}),
            patch("langchain_openai.ChatOpenAI") as mock_class,
        ):
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
        model_cfg = {
            "transcription_model": {
                "provider": "custom",
                "custom_endpoint": {
                    "api_key_env_var": "CUSTOM_KEY",
                    "base_url": "https://example.invalid/v1",
                },
            }
        }
        with (
            _with_api_requests_config({}),
            patch("config.loader.get_config_loader") as mock_loader,
            patch.dict(os.environ, {"CUSTOM_KEY": "secret"}, clear=False),
            patch("langchain_openai.ChatOpenAI") as mock_class,
        ):
            mock_loader.return_value.get_model_config.return_value = model_cfg
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(model="local-model", provider="custom", timeout=45)
            )

        timeout = mock_class.call_args[1]["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 45

    def test_anthropic_keeps_scalar_timeout(self, mock_api_keys) -> None:
        """ChatAnthropic compares its timeout with ``> 0``; keep the float."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(model="claude-3-opus", provider="anthropic", timeout=300)
            )

        assert mock_class.call_args[1]["timeout"] == 300

    def test_google_keeps_scalar_timeout(self, mock_api_keys) -> None:
        """ChatGoogleGenerativeAI computes int(timeout * 1000)."""
        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(model="gemini-2.5-flash", provider="google", timeout=300)
            )

        assert mock_class.call_args[1]["timeout"] == 300

    def test_extra_kwargs_override_becomes_the_read_budget(self, mock_api_keys) -> None:
        """A user override injected via extra_kwargs stays the read phase."""
        with (
            _with_api_requests_config({}),
            patch("langchain_openai.ChatOpenAI") as mock_class,
        ):
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(
                    model="gpt-5-mini",
                    provider="openai",
                    timeout=900,
                    extra_kwargs={"timeout": 77},
                )
            )

        timeout = mock_class.call_args[1]["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 77

    def test_existing_httpx_timeout_passes_through(self, mock_api_keys) -> None:
        """An already per-phase timeout is not rewrapped."""
        supplied = httpx.Timeout(50.0, connect=1.0, write=2.0, pool=3.0)
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(
                    model="gpt-5-mini",
                    provider="openai",
                    extra_kwargs={"timeout": supplied},
                )
            )

        assert mock_class.call_args[1]["timeout"] is supplied

    def test_none_timeout_passes_through(self, mock_api_keys) -> None:
        """``timeout=None`` keeps the SDK defaults rather than a Timeout."""
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(
                LLMConfig(
                    model="gpt-5-mini",
                    provider="openai",
                    extra_kwargs={"timeout": None},
                )
            )

        assert mock_class.call_args[1]["timeout"] is None
