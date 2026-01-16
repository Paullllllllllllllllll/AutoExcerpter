"""Tests for api/base_llm_client.py."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage

from api.base_llm_client import LLMClientBase


class TestExtractOutputText:
    def test_extract_output_text_from_ai_message_string(self):
        msg = AIMessage(content=" hello ")
        assert LLMClientBase._extract_output_text(msg) == "hello"

    def test_extract_output_text_from_ai_message_blocks(self):
        msg = AIMessage(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "output_text", "text": " world"},
            ]
        )
        assert LLMClientBase._extract_output_text(msg) == "hello world"

    def test_extract_output_text_from_output_text_attr(self):
        class Obj:
            output_text = "  hi  "

        assert LLMClientBase._extract_output_text(Obj()) == "hi"


class TestSchemaRetryDecision:
    def test_should_retry_for_schema_flag_disabled(self):
        client = LLMClientBase.__new__(LLMClientBase)
        client.schema_retry_config = {"flag": {"enabled": False, "max_attempts": 3}}

        should_retry, backoff, max_attempts = client._should_retry_for_schema_flag(
            "flag", True, 0
        )
        assert should_retry is False
        assert backoff == 0.0
        assert max_attempts == 0

    def test_should_retry_for_schema_flag_enabled_with_backoff(self):
        client = LLMClientBase.__new__(LLMClientBase)
        client.schema_retry_config = {
            "flag": {
                "enabled": True,
                "max_attempts": 3,
                "backoff_base": 1.0,
                "backoff_multiplier": 2.0,
            }
        }

        with patch("api.base_llm_client.random.uniform", return_value=1.0):
            should_retry, backoff, max_attempts = client._should_retry_for_schema_flag(
                "flag", True, 1
            )

        assert should_retry is True
        assert backoff == 2.0
        assert max_attempts == 3


class TestBuildInvokeKwargs:
    def test_build_invoke_kwargs_openai_capability_guarding(self):
        client = LLMClientBase.__new__(LLMClientBase)
        client.provider = "openai"
        client.model_name = "gpt-5-mini"
        client.model_config = {
            "max_output_tokens": 123,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low"},
        }
        client.service_tier = "flex"

        with patch(
            "api.base_llm_client.get_model_capabilities",
            return_value={"max_tokens": True, "reasoning": True, "text_verbosity": True},
        ):
            kwargs = client._build_invoke_kwargs()

        assert kwargs["max_output_tokens"] == 123
        assert kwargs["service_tier"] == "flex"
        assert kwargs["reasoning"] == {"effort": "low"}
        assert kwargs["text"] == {"verbosity": "low"}

    def test_build_invoke_kwargs_skips_unsupported_reasoning_and_text(self):
        client = LLMClientBase.__new__(LLMClientBase)
        client.provider = "openai"
        client.model_name = "gpt-4o"
        client.model_config = {
            "max_output_tokens": 123,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low"},
        }
        client.service_tier = "flex"

        with patch(
            "api.base_llm_client.get_model_capabilities",
            return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
        ):
            kwargs = client._build_invoke_kwargs()

        assert kwargs["max_output_tokens"] == 123
        assert kwargs["service_tier"] == "flex"
        assert "reasoning" not in kwargs
        assert "text" not in kwargs

    def test_build_invoke_kwargs_anthropic_uses_max_tokens_key(self):
        client = LLMClientBase.__new__(LLMClientBase)
        client.provider = "anthropic"
        client.model_name = "claude-sonnet-4-5"
        client.model_config = {"max_output_tokens": 456}
        client.service_tier = "auto"

        with patch(
            "api.base_llm_client.get_model_capabilities",
            return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
        ):
            kwargs = client._build_invoke_kwargs()

        assert kwargs["max_tokens"] == 456
        assert "service_tier" not in kwargs
