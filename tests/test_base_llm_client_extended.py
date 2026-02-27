"""Extended tests for api/base_llm_client.py — covers untested code paths.

This file complements test_base_llm_client.py by covering:
- __init__() initialization with various parameters
- _load_model_config() with config_loader
- _determine_service_tier() for openai and non-openai providers
- _wait_for_rate_limit(), _report_success(), _report_error()
- _extract_output_text() — dict-style, fallback output list, warning on empty
- _build_text_format() — valid schema, None, non-dict
- _get_structured_chat_model() — per-provider behaviour
- _apply_structured_output_kwargs() — OpenAI response_format, Google response_mime
- get_stats() — zero requests, after some requests
- _load_schema_retry_config() — success, missing, error
- _should_retry_for_schema_flag() — exceeded max_attempts, non-truthy flag
- _build_invoke_kwargs() — reasoning, text verbosity, provider-specific keys
"""

from __future__ import annotations

import statistics
from collections import deque
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.messages import AIMessage

from api.base_llm_client import LLMClientBase


# ============================================================================
# Helpers
# ============================================================================
def _make_client(**overrides) -> LLMClientBase:
    """Create a bare LLMClientBase bypassing __init__ for isolated unit tests."""
    client = LLMClientBase.__new__(LLMClientBase)
    defaults = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "timeout": 300,
        "rate_limiter": None,
        "max_retries": 5,
        "chat_model": MagicMock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "model_config": {},
        "service_tier": "auto",
        "schema_retry_config": {},
        "_output_schema": None,
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(client, attr, val)
    return client


# ============================================================================
# __init__
# ============================================================================
class TestInit:
    """Tests for LLMClientBase.__init__()."""

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=600)
    def test_init_defaults(self, mock_timeout, mock_get_chat):
        """Initialization stores resolved provider, model, and stats defaults."""
        mock_model = MagicMock()
        mock_get_chat.return_value = mock_model

        client = LLMClientBase(
            model_name="gpt-5-mini",
            provider="openai",
            api_key="test-key",
        )

        assert client.model_name == "gpt-5-mini"
        assert client.provider == "openai"
        assert client.chat_model is mock_model
        assert client.successful_requests == 0
        assert client.failed_requests == 0
        assert isinstance(client.processing_times, deque)
        assert client.service_tier == "auto"
        assert client._output_schema is None

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=600)
    def test_init_explicit_timeout(self, mock_timeout, mock_get_chat):
        """Explicit timeout overrides the config-loaded default."""
        mock_get_chat.return_value = MagicMock()
        client = LLMClientBase(
            model_name="gpt-5-mini",
            provider="openai",
            api_key="k",
            timeout=120,
        )
        assert client.timeout == 120

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=600)
    def test_init_none_timeout_uses_config(self, mock_timeout, mock_get_chat):
        """When timeout is None, get_api_timeout() is used."""
        mock_get_chat.return_value = MagicMock()
        client = LLMClientBase(
            model_name="gpt-5-mini",
            provider="openai",
            api_key="k",
        )
        assert client.timeout == 600

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=600)
    def test_init_with_rate_limiter(self, mock_timeout, mock_get_chat):
        """Rate limiter is stored correctly."""
        mock_get_chat.return_value = MagicMock()
        limiter = MagicMock()
        client = LLMClientBase(
            model_name="gpt-5-mini",
            provider="openai",
            api_key="k",
            rate_limiter=limiter,
        )
        assert client.rate_limiter is limiter

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=600)
    def test_init_custom_service_tier(self, mock_timeout, mock_get_chat):
        """Explicit service_tier overrides default 'auto'."""
        mock_get_chat.return_value = MagicMock()
        client = LLMClientBase(
            model_name="gpt-5-mini",
            provider="openai",
            api_key="k",
            service_tier="flex",
        )
        assert client.service_tier == "flex"

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=600)
    def test_init_provider_resolved_from_llm_config(self, mock_timeout, mock_get_chat):
        """When provider is None, it is resolved via LLMConfig."""
        mock_get_chat.return_value = MagicMock()
        client = LLMClientBase(
            model_name="claude-sonnet-4-5-20250929",
            provider=None,
            api_key="k",
        )
        # LLMConfig.__post_init__ infers "anthropic" from model name
        assert client.provider == "anthropic"


# ============================================================================
# _load_model_config
# ============================================================================
class TestLoadModelConfig:
    """Tests for _load_model_config()."""

    def test_loads_successfully(self):
        """Returns config dict when key exists."""
        client = _make_client()
        mock_loader = MagicMock()
        mock_loader.get_model_config.return_value = {
            "transcription_model": {"name": "gpt-5", "max_output_tokens": 4096}
        }

        with patch("api.base_llm_client.get_config_loader", return_value=mock_loader):
            result = client._load_model_config("transcription_model")

        assert result == {"name": "gpt-5", "max_output_tokens": 4096}

    def test_returns_empty_dict_for_missing_key(self):
        """Returns empty dict when config key is absent."""
        client = _make_client()
        mock_loader = MagicMock()
        mock_loader.get_model_config.return_value = {}

        with patch("api.base_llm_client.get_config_loader", return_value=mock_loader):
            result = client._load_model_config("nonexistent_model")

        assert result == {}

    def test_returns_empty_dict_on_exception(self):
        """Returns empty dict and logs warning on exception."""
        client = _make_client()

        with patch(
            "api.base_llm_client.get_config_loader",
            side_effect=RuntimeError("broken"),
        ):
            result = client._load_model_config("transcription_model")

        assert result == {}


# ============================================================================
# _determine_service_tier
# ============================================================================
class TestDetermineServiceTier:
    """Tests for _determine_service_tier()."""

    def test_non_openai_returns_auto(self):
        """Non-OpenAI providers always return 'auto'."""
        for provider in ("anthropic", "google", "openrouter"):
            client = _make_client(provider=provider)
            assert client._determine_service_tier("transcription") == "auto"

    @patch("api.base_llm_client.get_service_tier", return_value="flex")
    def test_openai_delegates_to_get_service_tier(self, mock_tier):
        """OpenAI provider delegates to get_service_tier()."""
        client = _make_client(provider="openai")
        result = client._determine_service_tier("summary")
        mock_tier.assert_called_once_with("summary")
        assert result == "flex"


# ============================================================================
# _wait_for_rate_limit, _report_success, _report_error
# ============================================================================
class TestRateLimiterDelegation:
    """Tests for rate limiter helper methods."""

    def test_wait_for_rate_limit_calls_limiter(self):
        """Calls wait_for_capacity when limiter is set."""
        limiter = MagicMock()
        client = _make_client(rate_limiter=limiter)
        client._wait_for_rate_limit()
        limiter.wait_for_capacity.assert_called_once()

    def test_wait_for_rate_limit_noop_without_limiter(self):
        """No-op when rate_limiter is None."""
        client = _make_client(rate_limiter=None)
        client._wait_for_rate_limit()  # Should not raise

    def test_report_success_increments_and_delegates(self):
        """Increments counter and calls rate_limiter.report_success."""
        limiter = MagicMock()
        client = _make_client(rate_limiter=limiter, successful_requests=2)
        client._report_success()
        assert client.successful_requests == 3
        limiter.report_success.assert_called_once()

    def test_report_success_without_limiter(self):
        """Increments counter even without limiter."""
        client = _make_client(rate_limiter=None, successful_requests=0)
        client._report_success()
        assert client.successful_requests == 1

    def test_report_error_delegates_to_limiter(self):
        """Calls rate_limiter.report_error with correct flag."""
        limiter = MagicMock()
        client = _make_client(rate_limiter=limiter)
        client._report_error(True)
        limiter.report_error.assert_called_once_with(True)

    def test_report_error_false_flag(self):
        """Passes False flag through to limiter."""
        limiter = MagicMock()
        client = _make_client(rate_limiter=limiter)
        client._report_error(False)
        limiter.report_error.assert_called_once_with(False)

    def test_report_error_noop_without_limiter(self):
        """No-op when rate_limiter is None."""
        client = _make_client(rate_limiter=None)
        client._report_error(True)  # Should not raise


# ============================================================================
# _extract_output_text — extended cases
# ============================================================================
class TestExtractOutputTextExtended:
    """Additional tests for _extract_output_text beyond existing coverage."""

    def test_dict_style_access(self):
        """Extracts from dict with 'output_text' key."""
        data = {"output_text": "  hello from dict  "}
        assert LLMClientBase._extract_output_text(data) == "hello from dict"

    def test_dict_empty_output_text(self):
        """Empty dict output_text returns empty string with warning."""
        data = {"output_text": "   "}
        result = LLMClientBase._extract_output_text(data)
        assert result == ""

    def test_fallback_output_list_parsing(self):
        """Extracts text from output list structure."""

        class FakeResponse:
            def model_dump(self):
                return {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "part1"},
                                {"type": "text", "text": " part2"},
                            ]
                        }
                    ]
                }

        result = LLMClientBase._extract_output_text(FakeResponse())
        assert result == "part1 part2"

    def test_fallback_output_list_empty_content(self):
        """Handles output list with empty content gracefully."""

        class FakeResponse:
            def model_dump(self):
                return {"output": [{"content": []}]}

        result = LLMClientBase._extract_output_text(FakeResponse())
        assert result == ""

    def test_fallback_no_output_key(self):
        """Returns empty string when output key is missing."""

        class FakeResponse:
            def model_dump(self):
                return {"other_key": "value"}

        result = LLMClientBase._extract_output_text(FakeResponse())
        assert result == ""

    def test_aimessage_list_with_string_blocks(self):
        """AIMessage with string blocks in content list."""
        msg = AIMessage(content=["hello", " world"])
        assert LLMClientBase._extract_output_text(msg) == "hello world"

    def test_aimessage_empty_content_warns(self):
        """AIMessage with empty content returns empty string."""
        msg = AIMessage(content="   ")
        result = LLMClientBase._extract_output_text(msg)
        assert result == ""

    def test_aimessage_mixed_blocks(self):
        """AIMessage with mixed block types — non-text blocks are skipped."""
        msg = AIMessage(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "http://..."},
                "raw_string",
            ]
        )
        assert LLMClientBase._extract_output_text(msg) == "helloraw_string"

    def test_fallback_to_dict_with_non_text_content_items(self):
        """Non-dict content items in output list are skipped."""

        class FakeResponse:
            def model_dump(self):
                return {
                    "output": [
                        {
                            "content": [
                                "not_a_dict",
                                {"type": "text", "text": "valid"},
                            ]
                        }
                    ]
                }

        result = LLMClientBase._extract_output_text(FakeResponse())
        assert result == "valid"

    def test_exception_during_extraction_returns_empty(self):
        """Returns empty string when extraction raises an exception."""

        class Broken:
            pass  # No output_text, not a dict, no to_dict/model_dump

        result = LLMClientBase._extract_output_text(Broken())
        assert result == ""


# ============================================================================
# _build_text_format
# ============================================================================
class TestBuildTextFormat:
    """Tests for _build_text_format()."""

    def test_valid_schema(self):
        """Returns properly structured format dict for a valid schema."""
        client = _make_client(
            _output_schema={
                "name": "my_schema",
                "strict": True,
                "schema": {"type": "object", "properties": {}},
            }
        )
        result = client._build_text_format()
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["name"] == "my_schema"
        assert result["strict"] is True
        assert result["schema"] == {"type": "object", "properties": {}}

    def test_schema_without_name_uses_default(self):
        """Uses default_name when schema has no 'name' key."""
        client = _make_client(_output_schema={"schema": {"type": "object"}})
        result = client._build_text_format(default_name="custom_default")
        assert result["name"] == "custom_default"

    def test_none_schema_returns_none(self):
        """Returns None when _output_schema is None."""
        client = _make_client(_output_schema=None)
        assert client._build_text_format() is None

    def test_non_dict_schema_returns_none(self):
        """Returns None when _output_schema is not a dict."""
        client = _make_client(_output_schema="not a dict")
        assert client._build_text_format() is None

    def test_empty_inner_schema_returns_none(self):
        """Returns None when inner 'schema' is empty dict."""
        client = _make_client(_output_schema={"schema": {}})
        assert client._build_text_format() is None

    def test_schema_uses_self_as_schema_obj_when_no_inner_key(self):
        """When 'schema' key is missing, uses the dict itself as schema."""
        client = _make_client(
            _output_schema={"type": "object", "properties": {"a": {}}}
        )
        result = client._build_text_format()
        assert result is not None
        assert result["schema"] == {"type": "object", "properties": {"a": {}}}

    def test_strict_defaults_to_true(self):
        """Strict defaults to True when not specified."""
        client = _make_client(
            _output_schema={"schema": {"type": "object"}, "properties": {"a": {}}}
        )
        # The outer schema dict has properties but no explicit "strict"
        result = client._build_text_format()
        # Since schema key => {"type":"object"} which is non-empty
        assert result is not None
        assert result["strict"] is True


# ============================================================================
# _get_structured_chat_model
# ============================================================================
class TestGetStructuredChatModel:
    """Tests for _get_structured_chat_model()."""

    def test_openai_returns_base_model(self):
        """OpenAI provider returns the base chat model."""
        base_model = MagicMock()
        client = _make_client(provider="openai", chat_model=base_model)
        assert client._get_structured_chat_model() is base_model

    def test_anthropic_returns_base_model(self):
        """Anthropic provider returns the base chat model."""
        base_model = MagicMock()
        client = _make_client(provider="anthropic", chat_model=base_model)
        assert client._get_structured_chat_model() is base_model

    def test_google_returns_base_model(self):
        """Google provider returns the base chat model."""
        base_model = MagicMock()
        client = _make_client(provider="google", chat_model=base_model)
        assert client._get_structured_chat_model() is base_model

    def test_openrouter_with_schema_uses_structured_output(self):
        """OpenRouter with schema calls with_structured_output."""
        base_model = MagicMock()
        structured = MagicMock()
        base_model.with_structured_output.return_value = structured

        client = _make_client(
            provider="openrouter",
            chat_model=base_model,
            _output_schema={"schema": {"type": "object", "properties": {}}},
        )
        result = client._get_structured_chat_model()
        base_model.with_structured_output.assert_called_once_with(
            {"type": "object", "properties": {}},
            include_raw=True,
        )
        assert result is structured

    def test_openrouter_without_schema_returns_base(self):
        """OpenRouter without schema returns base model."""
        base_model = MagicMock()
        client = _make_client(
            provider="openrouter",
            chat_model=base_model,
            _output_schema=None,
        )
        assert client._get_structured_chat_model() is base_model

    def test_openrouter_with_empty_inner_schema_returns_base(self):
        """OpenRouter with empty inner schema returns base model."""
        base_model = MagicMock()
        client = _make_client(
            provider="openrouter",
            chat_model=base_model,
            _output_schema={"schema": {}},
        )
        assert client._get_structured_chat_model() is base_model


# ============================================================================
# _apply_structured_output_kwargs
# ============================================================================
class TestApplyStructuredOutputKwargs:
    """Tests for _apply_structured_output_kwargs()."""

    def test_openai_adds_response_format(self):
        """OpenAI provider adds response_format to kwargs."""
        client = _make_client(
            provider="openai",
            _output_schema={"schema": {"type": "object"}},
        )
        kwargs = {}
        client._apply_structured_output_kwargs(kwargs)
        assert "response_format" in kwargs
        assert kwargs["response_format"]["type"] == "json_schema"

    def test_openai_with_text_key_adds_format_inside(self):
        """When 'text' key exists, format is nested inside it."""
        client = _make_client(
            provider="openai",
            _output_schema={"schema": {"type": "object"}},
        )
        kwargs = {"text": {"verbosity": "low"}}
        client._apply_structured_output_kwargs(kwargs)
        assert "format" in kwargs["text"]
        assert "response_format" not in kwargs

    def test_openai_no_schema_no_format(self):
        """OpenAI without schema adds nothing."""
        client = _make_client(provider="openai", _output_schema=None)
        kwargs = {}
        client._apply_structured_output_kwargs(kwargs)
        assert "response_format" not in kwargs

    def test_google_adds_response_mime_and_schema(self):
        """Google provider adds response_mime_type and response_schema."""
        client = _make_client(
            provider="google",
            _output_schema={"schema": {"type": "object", "properties": {}}},
        )
        kwargs = {}
        client._apply_structured_output_kwargs(kwargs)
        assert kwargs["response_mime_type"] == "application/json"
        assert kwargs["response_schema"] == {"type": "object", "properties": {}}

    def test_google_no_schema_no_additions(self):
        """Google without schema adds nothing."""
        client = _make_client(provider="google", _output_schema=None)
        kwargs = {}
        client._apply_structured_output_kwargs(kwargs)
        assert "response_mime_type" not in kwargs

    def test_google_does_not_overwrite_existing(self):
        """Google respects existing response_mime_type."""
        client = _make_client(
            provider="google",
            _output_schema={"schema": {"type": "object"}},
        )
        kwargs = {"response_mime_type": "text/plain"}
        client._apply_structured_output_kwargs(kwargs)
        assert kwargs["response_mime_type"] == "text/plain"

    def test_anthropic_adds_nothing(self):
        """Anthropic provider does not modify kwargs."""
        client = _make_client(
            provider="anthropic",
            _output_schema={"schema": {"type": "object"}},
        )
        kwargs = {}
        client._apply_structured_output_kwargs(kwargs)
        assert kwargs == {}

    def test_openrouter_adds_nothing(self):
        """OpenRouter provider does not modify kwargs (uses with_structured_output)."""
        client = _make_client(
            provider="openrouter",
            _output_schema={"schema": {"type": "object"}},
        )
        kwargs = {}
        client._apply_structured_output_kwargs(kwargs)
        assert kwargs == {}


# ============================================================================
# get_stats
# ============================================================================
class TestGetStats:
    """Tests for get_stats()."""

    def test_stats_no_requests(self):
        """Stats with zero requests."""
        client = _make_client()
        stats = client.get_stats()
        assert stats["provider"] == "openai"
        assert stats["model"] == "gpt-5-mini"
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["average_processing_time"] == 0
        assert stats["recent_success_rate"] == 0.0

    def test_stats_with_requests(self):
        """Stats reflect actual request counts and timings."""
        client = _make_client(successful_requests=8, failed_requests=2)
        client.processing_times.extend([1.0, 2.0, 3.0])

        stats = client.get_stats()
        assert stats["successful_requests"] == 8
        assert stats["failed_requests"] == 2
        assert stats["average_processing_time"] == 2.0
        assert stats["recent_success_rate"] == 80.0

    def test_stats_all_successful(self):
        """100% success rate."""
        client = _make_client(successful_requests=5, failed_requests=0)
        stats = client.get_stats()
        assert stats["recent_success_rate"] == 100.0

    def test_stats_all_failed(self):
        """0% success rate."""
        client = _make_client(successful_requests=0, failed_requests=3)
        stats = client.get_stats()
        assert stats["recent_success_rate"] == 0.0


# ============================================================================
# _load_schema_retry_config
# ============================================================================
class TestLoadSchemaRetryConfig:
    """Tests for _load_schema_retry_config()."""

    def test_loads_successfully(self):
        """Returns schema retry config for the given api_type."""
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {
            "retry": {
                "schema_retries": {
                    "transcription": {
                        "no_transcribable_text": {"enabled": True, "max_attempts": 2},
                    }
                }
            }
        }
        client = _make_client()
        with patch("api.base_llm_client.get_config_loader", return_value=mock_loader):
            result = client._load_schema_retry_config("transcription")

        assert "no_transcribable_text" in result
        assert result["no_transcribable_text"]["enabled"] is True

    def test_missing_api_type_returns_empty(self):
        """Returns empty dict when api_type is not in config."""
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {
            "retry": {"schema_retries": {}}
        }
        client = _make_client()
        with patch("api.base_llm_client.get_config_loader", return_value=mock_loader):
            result = client._load_schema_retry_config("transcription")

        assert result == {}

    def test_missing_retry_key_returns_empty(self):
        """Returns empty dict when 'retry' key is absent."""
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {}
        client = _make_client()
        with patch("api.base_llm_client.get_config_loader", return_value=mock_loader):
            result = client._load_schema_retry_config("transcription")

        assert result == {}

    def test_exception_returns_empty(self):
        """Returns empty dict on exception."""
        client = _make_client()
        with patch(
            "api.base_llm_client.get_config_loader",
            side_effect=RuntimeError("fail"),
        ):
            result = client._load_schema_retry_config("transcription")

        assert result == {}


# ============================================================================
# _should_retry_for_schema_flag — extended cases
# ============================================================================
class TestShouldRetryForSchemaFlagExtended:
    """Extended tests for _should_retry_for_schema_flag()."""

    def test_exceeded_max_attempts(self):
        """Returns False when current_attempt >= max_attempts."""
        client = _make_client(
            schema_retry_config={
                "flag": {"enabled": True, "max_attempts": 2},
            }
        )
        should, backoff, max_att = client._should_retry_for_schema_flag("flag", True, 2)
        assert should is False
        assert backoff == 0.0
        assert max_att == 2

    def test_flag_not_truthy(self):
        """Returns False when flag_value is falsy."""
        client = _make_client(
            schema_retry_config={
                "flag": {"enabled": True, "max_attempts": 3},
            }
        )
        should, backoff, max_att = client._should_retry_for_schema_flag(
            "flag", False, 0
        )
        assert should is False

    def test_flag_none_value(self):
        """Returns False when flag_value is None."""
        client = _make_client(
            schema_retry_config={
                "flag": {"enabled": True, "max_attempts": 3},
            }
        )
        should, backoff, max_att = client._should_retry_for_schema_flag("flag", None, 0)
        assert should is False

    def test_unknown_flag_name(self):
        """Returns False when flag name is not in config."""
        client = _make_client(schema_retry_config={})
        should, backoff, max_att = client._should_retry_for_schema_flag(
            "nonexistent", True, 0
        )
        assert should is False

    def test_enabled_first_attempt(self):
        """Returns True with correct backoff for first attempt."""
        client = _make_client(
            schema_retry_config={
                "flag": {
                    "enabled": True,
                    "max_attempts": 3,
                    "backoff_base": 2.0,
                    "backoff_multiplier": 1.5,
                },
            }
        )
        with patch("api.base_llm_client.random.uniform", return_value=1.0):
            should, backoff, max_att = client._should_retry_for_schema_flag(
                "flag", True, 0
            )
        assert should is True
        # backoff = 2.0 * (1.5 ** 0) * 1.0 = 2.0
        assert backoff == pytest.approx(2.0)
        assert max_att == 3


# ============================================================================
# _build_invoke_kwargs — extended cases
# ============================================================================
class TestBuildInvokeKwargsExtended:
    """Extended tests for _build_invoke_kwargs()."""

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_google_uses_max_output_tokens(self, _):
        """Google provider uses max_output_tokens key."""
        client = _make_client(
            provider="google",
            model_name="gemini-2.5-flash",
            model_config={"max_output_tokens": 2048},
            service_tier="auto",
        )
        kwargs = client._build_invoke_kwargs()
        assert kwargs["max_output_tokens"] == 2048
        assert "service_tier" not in kwargs  # Not OpenAI

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_openrouter_uses_max_tokens(self, _):
        """OpenRouter (else branch) uses max_tokens key."""
        client = _make_client(
            provider="openrouter",
            model_name="openrouter/model",
            model_config={"max_output_tokens": 1024},
            service_tier="auto",
        )
        kwargs = client._build_invoke_kwargs()
        assert kwargs["max_tokens"] == 1024

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_no_max_output_tokens_in_config(self, _):
        """No max_output_tokens in model_config produces no token key."""
        client = _make_client(
            provider="openai",
            model_config={},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert "max_output_tokens" not in kwargs
        assert "max_tokens" not in kwargs

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": False, "reasoning": False, "text_verbosity": False},
    )
    def test_max_tokens_capability_disabled(self, _):
        """When max_tokens capability is False, token params are skipped."""
        client = _make_client(
            provider="openai",
            model_config={"max_output_tokens": 4096},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert "max_output_tokens" not in kwargs

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": True, "text_verbosity": False},
    )
    def test_reasoning_config_non_dict_ignored(self, _):
        """Non-dict reasoning config is ignored."""
        client = _make_client(
            provider="openai",
            model_config={"reasoning": "not_a_dict"},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert "reasoning" not in kwargs

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": True, "text_verbosity": False},
    )
    def test_reasoning_dict_without_effort_ignored(self, _):
        """Reasoning dict without 'effort' key adds nothing."""
        client = _make_client(
            provider="openai",
            model_config={"reasoning": {"summary": "medium"}},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert "reasoning" not in kwargs

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": True},
    )
    def test_text_verbosity_config(self, _):
        """Text verbosity is added for supported models."""
        client = _make_client(
            provider="openai",
            model_config={"text": {"verbosity": "medium"}},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert kwargs["text"] == {"verbosity": "medium"}

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": True},
    )
    def test_text_non_dict_ignored(self, _):
        """Non-dict text config is ignored."""
        client = _make_client(
            provider="openai",
            model_config={"text": "not_a_dict"},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert "text" not in kwargs

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": True},
    )
    def test_text_dict_without_verbosity_ignored(self, _):
        """Text dict without 'verbosity' key adds nothing."""
        client = _make_client(
            provider="openai",
            model_config={"text": {"other_param": "value"}},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert "text" not in kwargs

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_non_openai_no_service_tier(self, _):
        """Non-OpenAI providers do not get service_tier."""
        client = _make_client(
            provider="anthropic",
            model_name="claude-sonnet-4-5",
            model_config={},
            service_tier="flex",
        )
        kwargs = client._build_invoke_kwargs()
        assert "service_tier" not in kwargs

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_openai_empty_service_tier_omitted(self, _):
        """OpenAI with empty service_tier string does not add it."""
        client = _make_client(
            provider="openai",
            model_config={},
            service_tier="",
        )
        kwargs = client._build_invoke_kwargs()
        assert "service_tier" not in kwargs
