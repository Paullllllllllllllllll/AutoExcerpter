"""Tests for api/providers/ -- all four LLM provider implementations, base classes, and lazy imports.

Covers:
- api/providers/base.py          (TranscriptionResult, BaseProvider)
- api/providers/anthropic_provider.py  (AnthropicProvider, _transform_schema_for_anthropic, _load_max_retries)
- api/providers/openai_provider.py     (OpenAIProvider, _load_max_retries)
- api/providers/google_provider.py     (GoogleProvider, _load_max_retries)
- api/providers/openrouter_provider.py (OpenRouterProvider, _effort_to_ratio, _compute_openrouter_reasoning_max_tokens, _load_max_retries)
- api/providers/__init__.py            (lazy __getattr__)
"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.model_capabilities import ProviderCapabilities
from api.providers.base import (
    SUPPORTED_IMAGE_FORMATS,
    BaseProvider,
    TranscriptionResult,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_caps(**overrides) -> ProviderCapabilities:
    """Build a ProviderCapabilities with sensible defaults and optional overrides."""
    defaults = dict(
        provider_name="test",
        model_name="test-model",
        family="test",
        supports_vision=True,
        supports_image_detail=True,
        default_image_detail="high",
        supports_structured_output=True,
        supports_json_mode=True,
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_temperature=True,
        supports_top_p=True,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
    )
    defaults.update(overrides)
    return ProviderCapabilities(**defaults)


def _make_ai_message(content="Hello world", metadata=None):
    """Build a lightweight object that mimics a LangChain AIMessage."""
    msg = SimpleNamespace()
    msg.content = content
    msg.response_metadata = metadata if metadata is not None else {}
    return msg


# ============================================================================
# TranscriptionResult
# ============================================================================

class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass."""

    def test_basic_creation(self):
        result = TranscriptionResult(content="some text")
        assert result.content == "some text"
        assert result.raw_response == {}
        assert result.parsed_output is None
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0
        assert result.no_transcribable_text is False
        assert result.transcription_not_possible is False
        assert result.error is None

    def test_post_init_parses_json_content(self):
        payload = {"transcription": "hello", "no_transcribable_text": True}
        result = TranscriptionResult(content=json.dumps(payload))
        assert result.parsed_output == payload
        assert result.no_transcribable_text is True
        assert result.transcription_not_possible is False

    def test_post_init_parses_transcription_not_possible(self):
        payload = {"transcription_not_possible": True, "no_transcribable_text": False}
        result = TranscriptionResult(content=json.dumps(payload))
        assert result.transcription_not_possible is True
        assert result.no_transcribable_text is False

    def test_post_init_skips_when_parsed_output_already_set(self):
        existing = {"pre": "existing"}
        result = TranscriptionResult(
            content='{"no_transcribable_text": true}',
            parsed_output=existing,
        )
        # Should NOT overwrite the existing parsed_output
        assert result.parsed_output == existing
        assert result.no_transcribable_text is False

    def test_post_init_handles_non_json_content(self):
        result = TranscriptionResult(content="just plain text")
        assert result.parsed_output is None

    def test_post_init_handles_empty_content(self):
        result = TranscriptionResult(content="")
        assert result.parsed_output is None

    def test_post_init_handles_non_dict_json(self):
        result = TranscriptionResult(content="[1, 2, 3]")
        # Starts with "{" check fails, so not parsed
        assert result.parsed_output is None

    def test_post_init_handles_invalid_json_starting_with_brace(self):
        result = TranscriptionResult(content="{not valid json}")
        assert result.parsed_output is None

    def test_to_dict_minimal(self):
        result = TranscriptionResult(content="hello")
        d = result.to_dict()
        assert d["output_text"] == "hello"
        assert d["usage"]["input_tokens"] == 0
        assert d["usage"]["output_tokens"] == 0
        assert d["usage"]["total_tokens"] == 0
        assert "metadata" not in d
        assert "parsed" not in d
        assert "error" not in d

    def test_to_dict_with_all_fields(self):
        result = TranscriptionResult(
            content="text",
            raw_response={"model": "test"},
            parsed_output={"key": "val"},
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            error="some error",
        )
        d = result.to_dict()
        assert d["metadata"] == {"model": "test"}
        assert d["parsed"] == {"key": "val"}
        assert d["error"] == "some error"
        assert d["usage"]["total_tokens"] == 30

    def test_to_dict_omits_empty_raw_response(self):
        result = TranscriptionResult(content="text", raw_response={})
        d = result.to_dict()
        assert "metadata" not in d

    def test_to_dict_omits_none_parsed_output(self):
        result = TranscriptionResult(content="text")
        d = result.to_dict()
        assert "parsed" not in d

    def test_to_dict_omits_none_error(self):
        result = TranscriptionResult(content="text")
        d = result.to_dict()
        assert "error" not in d


# ============================================================================
# BaseProvider
# ============================================================================

class TestBaseProvider:
    """Tests for BaseProvider static methods and async context manager."""

    def test_encode_image_to_base64_jpeg(self, sample_image_file):
        data, mime = BaseProvider.encode_image_to_base64(sample_image_file)
        assert mime == "image/jpeg"
        # Verify it is valid base64
        decoded = base64.b64decode(data)
        assert len(decoded) > 0

    def test_encode_image_to_base64_png(self, temp_dir):
        from PIL import Image
        path = temp_dir / "test.png"
        Image.new("RGB", (10, 10)).save(path)
        data, mime = BaseProvider.encode_image_to_base64(path)
        assert mime == "image/png"
        assert len(data) > 0

    def test_encode_image_to_base64_unsupported_format(self, temp_dir):
        path = temp_dir / "test.xyz"
        path.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported image format"):
            BaseProvider.encode_image_to_base64(path)

    def test_create_data_url(self):
        url = BaseProvider.create_data_url("abc123", "image/png")
        assert url == "data:image/png;base64,abc123"

    def test_create_data_url_jpeg(self):
        url = BaseProvider.create_data_url("xyz", "image/jpeg")
        assert url == "data:image/jpeg;base64,xyz"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test __aenter__ returns self, __aexit__ calls close."""
        # Create a concrete subclass for testing
        class ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"

            def get_capabilities(self):
                return _make_caps()

            async def transcribe_image(self, *args, **kwargs):
                pass

            async def transcribe_image_from_base64(self, *args, **kwargs):
                pass

            async def close(self):
                self._closed = True

        provider = ConcreteProvider(api_key="k", model="m")
        provider._closed = False

        async with provider as p:
            assert p is provider
        assert provider._closed is True

    @pytest.mark.asyncio
    async def test_async_context_manager_aexit_returns_false(self):
        """__aexit__ should return False (don't suppress exceptions)."""
        class ConcreteProvider(BaseProvider):
            @property
            def provider_name(self):
                return "test"

            def get_capabilities(self):
                return _make_caps()

            async def transcribe_image(self, *args, **kwargs):
                pass

            async def transcribe_image_from_base64(self, *args, **kwargs):
                pass

            async def close(self):
                pass

        provider = ConcreteProvider(api_key="k", model="m")
        result = await provider.__aexit__(None, None, None)
        assert result is False


# ============================================================================
# SUPPORTED_IMAGE_FORMATS constant
# ============================================================================

class TestSupportedImageFormats:
    def test_jpeg_extensions(self):
        assert SUPPORTED_IMAGE_FORMATS[".jpg"] == "image/jpeg"
        assert SUPPORTED_IMAGE_FORMATS[".jpeg"] == "image/jpeg"

    def test_png_extension(self):
        assert SUPPORTED_IMAGE_FORMATS[".png"] == "image/png"

    def test_gif_extension(self):
        assert SUPPORTED_IMAGE_FORMATS[".gif"] == "image/gif"

    def test_webp_extension(self):
        assert SUPPORTED_IMAGE_FORMATS[".webp"] == "image/webp"

    def test_bmp_extension(self):
        assert SUPPORTED_IMAGE_FORMATS[".bmp"] == "image/bmp"


# ============================================================================
# Anthropic Provider -- standalone functions
# ============================================================================

class TestTransformSchemaForAnthropic:
    """Tests for _transform_schema_for_anthropic."""

    def test_simple_schema_unchanged(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["properties"]["text"]["type"] == "string"
        assert result["title"] == "TranscriptionSchema"
        assert result["description"] == "Schema for document transcription output"

    def test_union_type_resolved_to_first_non_null(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
            },
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["properties"]["name"]["type"] == "string"

    def test_all_null_union_falls_back_to_string(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "type": "object",
            "properties": {
                "nothing": {"type": ["null"]},
            },
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["properties"]["nothing"]["type"] == "string"

    def test_nested_array_items_transformed(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": ["integer", "null"],
                    },
                },
            },
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["properties"]["items"]["items"]["type"] == "integer"

    def test_anyof_items_transformed(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "anyOf": [
                {"type": ["string", "null"]},
                {"type": "integer"},
            ],
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["anyOf"][0]["type"] == "string"
        assert result["anyOf"][1]["type"] == "integer"

    def test_oneof_and_allof_transformed(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "oneOf": [{"type": ["boolean", "null"]}],
            "allOf": [{"type": ["number", "null"]}],
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["oneOf"][0]["type"] == "boolean"
        assert result["allOf"][0]["type"] == "number"

    def test_preserves_existing_title_and_description(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "type": "object",
            "title": "Custom",
            "description": "Custom desc",
        }
        result = _transform_schema_for_anthropic(schema)
        assert result["title"] == "Custom"
        assert result["description"] == "Custom desc"

    def test_does_not_mutate_original(self):
        from api.providers.anthropic_provider import _transform_schema_for_anthropic
        schema = {
            "type": "object",
            "properties": {
                "val": {"type": ["string", "null"]},
            },
        }
        _transform_schema_for_anthropic(schema)
        # Original should be unchanged
        assert schema["properties"]["val"]["type"] == ["string", "null"]


class TestAnthropicLoadMaxRetries:
    """Tests for anthropic_provider._load_max_retries."""

    def test_returns_configured_value(self):
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {
            "retry": {"max_attempts": 7},
        }
        with patch(
            "modules.config_loader.get_config_loader",
            return_value=mock_loader,
        ):
            from api.providers.anthropic_provider import _load_max_retries
            assert _load_max_retries() == 7

    def test_returns_minimum_of_1(self):
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {
            "retry": {"max_attempts": 0},
        }
        with patch(
            "modules.config_loader.get_config_loader",
            return_value=mock_loader,
        ):
            from api.providers.anthropic_provider import _load_max_retries
            assert _load_max_retries() == 1

    def test_returns_default_on_exception(self):
        with patch(
            "modules.config_loader.get_config_loader",
            side_effect=Exception("boom"),
        ):
            from api.providers.anthropic_provider import _load_max_retries
            assert _load_max_retries() == 5

    def test_handles_missing_retry_config(self):
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {}
        with patch(
            "modules.config_loader.get_config_loader",
            return_value=mock_loader,
        ):
            from api.providers.anthropic_provider import _load_max_retries
            assert _load_max_retries() == 5

    def test_handles_none_concurrency_config(self):
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = None
        with patch(
            "modules.config_loader.get_config_loader",
            return_value=mock_loader,
        ):
            from api.providers.anthropic_provider import _load_max_retries
            assert _load_max_retries() == 5


# ============================================================================
# Anthropic Provider -- class
# ============================================================================

class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_basic(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-sonnet-4-5",
            temperature=0.5,
            max_tokens=2048,
        )
        assert provider.provider_name == "anthropic"
        assert provider.model == "claude-sonnet-4-5"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 2048
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-5"
        assert call_kwargs.kwargs["max_tokens"] == 2048
        assert call_kwargs.kwargs["max_retries"] == 3

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_with_reasoning_config(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(
            api_key="key",
            model="claude-opus-4-5",
            reasoning_config={"effort": "high"},
        )
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs.kwargs["thinking"]["budget_tokens"] == 16384

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_reasoning_effort_low(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        AnthropicProvider(
            api_key="key",
            model="claude-opus-4-5",
            reasoning_config={"effort": "low"},
        )
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["thinking"]["budget_tokens"] == 1024

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_reasoning_effort_medium(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        AnthropicProvider(
            api_key="key",
            model="claude-opus-4-5",
            reasoning_config={"effort": "medium"},
        )
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["thinking"]["budget_tokens"] == 4096

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_reasoning_effort_unknown_defaults_to_medium(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        AnthropicProvider(
            api_key="key",
            model="claude-opus-4-5",
            reasoning_config={"effort": "unknown_level"},
        )
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["thinking"]["budget_tokens"] == 4096

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_reasoning_effort=False,
            supports_temperature=False,
            supports_top_p=False,
        ),
    )
    def test_init_skips_temp_and_top_p_when_not_supported(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        AnthropicProvider(api_key="key", model="test-model")
        call_kwargs = mock_chat.call_args
        assert "temperature" not in call_kwargs.kwargs
        assert "top_p" not in call_kwargs.kwargs

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_with_top_k(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        AnthropicProvider(api_key="key", model="test", top_k=40)
        call_kwargs = mock_chat.call_args
        assert call_kwargs.kwargs["top_k"] == 40

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(provider_name="anthropic"),
    )
    def test_provider_name(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="k", model="m")
        assert provider.provider_name == "anthropic"

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(provider_name="anthropic"),
    )
    def test_get_capabilities(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="k", model="m")
        caps = provider.get_capabilities()
        assert caps.provider_name == "anthropic"

    @pytest.mark.asyncio
    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(provider_name="anthropic", supports_vision=False),
    )
    async def test_transcribe_image_from_base64_no_vision(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="k", model="m")
        result = await provider.transcribe_image_from_base64(
            "abc",
            "image/png",
            system_prompt="test",
        )
        assert result.transcription_not_possible is True
        assert "does not support vision" in result.error

    @pytest.mark.asyncio
    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_vision=True,
            supports_structured_output=False,
        ),
    )
    async def test_transcribe_image_from_base64_plain_text(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        mock_llm = mock_chat.return_value
        mock_llm.ainvoke = AsyncMock(
            return_value=_make_ai_message("Transcribed text")
        )
        provider = AnthropicProvider(api_key="k", model="m")
        result = await provider.transcribe_image_from_base64(
            "base64data",
            "image/jpeg",
            system_prompt="Transcribe",
        )
        assert result.content == "Transcribed text"
        assert result.error is None

    @pytest.mark.asyncio
    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_vision=True,
            supports_structured_output=True,
        ),
    )
    async def test_transcribe_image_from_base64_with_schema(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        parsed = {"text": "hello", "no_transcribable_text": False}
        raw_msg = _make_ai_message("raw content", metadata={
            "usage": {"input_tokens": 100, "output_tokens": 50},
        })
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": parsed}
        )
        mock_llm = mock_chat.return_value
        mock_llm.with_structured_output.return_value = mock_structured

        provider = AnthropicProvider(api_key="k", model="m")
        result = await provider.transcribe_image_from_base64(
            "base64data",
            "image/jpeg",
            system_prompt="Transcribe",
            json_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )
        assert result.parsed_output == parsed
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    @pytest.mark.asyncio
    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="anthropic",
            supports_vision=True,
            supports_structured_output=True,
        ),
    )
    async def test_transcribe_image_from_base64_schema_unwrap(self, mock_detect, mock_chat, mock_retries):
        """When json_schema has a nested 'schema' key, it should be unwrapped."""
        from api.providers.anthropic_provider import AnthropicProvider
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(
            return_value=_make_ai_message("text")
        )
        mock_llm = mock_chat.return_value
        mock_llm.with_structured_output.return_value = mock_structured

        provider = AnthropicProvider(api_key="k", model="m")
        nested_schema = {"schema": {"type": "object", "properties": {}}}
        await provider.transcribe_image_from_base64(
            "data",
            "image/png",
            system_prompt="test",
            json_schema=nested_schema,
        )
        # with_structured_output should be called with the unwrapped schema
        call_args = mock_llm.with_structured_output.call_args
        actual_schema = call_args.args[0]
        assert "type" in actual_schema
        assert "schema" not in actual_schema

    @pytest.mark.asyncio
    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(provider_name="anthropic", supports_vision=True),
    )
    async def test_transcribe_image_delegates_to_from_base64(self, mock_detect, mock_chat, mock_retries, sample_image_file):
        from api.providers.anthropic_provider import AnthropicProvider
        mock_llm = mock_chat.return_value
        mock_llm.ainvoke = AsyncMock(return_value=_make_ai_message("result"))
        provider = AnthropicProvider(api_key="k", model="m")
        result = await provider.transcribe_image(
            sample_image_file,
            system_prompt="test",
        )
        assert result.content == "result"

    @pytest.mark.asyncio
    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch(
        "api.providers.anthropic_provider.detect_capabilities",
        return_value=_make_caps(provider_name="anthropic", supports_vision=True),
    )
    async def test_close_is_noop(self, mock_detect, mock_chat, mock_retries):
        from api.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(api_key="k", model="m")
        await provider.close()  # Should not raise


# ============================================================================
# Anthropic _invoke_llm response handling
# ============================================================================

class TestAnthropicInvokeLlm:
    """Tests for AnthropicProvider._invoke_llm response handling."""

    @pytest.fixture
    def provider(self):
        with (
            patch("api.providers.anthropic_provider._load_max_retries", return_value=3),
            patch("api.providers.anthropic_provider.ChatAnthropic"),
            patch(
                "api.providers.anthropic_provider.detect_capabilities",
                return_value=_make_caps(provider_name="anthropic"),
            ),
        ):
            from api.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(api_key="k", model="m")

    @pytest.mark.asyncio
    async def test_structured_dict_with_raw_and_parsed(self, provider):
        parsed = {"text": "hello", "no_transcribable_text": True}
        raw_msg = _make_ai_message("raw", metadata={
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": parsed}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == json.dumps(parsed)
        assert result.parsed_output == parsed
        assert result.no_transcribable_text is True
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.total_tokens == 15

    @pytest.mark.asyncio
    async def test_structured_dict_parsed_none_falls_back_to_raw_content(self, provider):
        raw_msg = _make_ai_message("fallback content")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": None}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "fallback content"

    @pytest.mark.asyncio
    async def test_structured_dict_parsed_none_raw_content_is_dict(self, provider):
        raw_msg = _make_ai_message({"key": "val"})
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": None}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == json.dumps({"key": "val"})
        assert result.parsed_output == {"key": "val"}

    @pytest.mark.asyncio
    async def test_structured_dict_parsed_none_no_raw(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": None, "parsed": None}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_ai_message_with_string_content(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=_make_ai_message("hello world")
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "hello world"

    @pytest.mark.asyncio
    async def test_ai_message_with_dict_content(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=_make_ai_message({"some": "data"})
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == json.dumps({"some": "data"})
        assert result.parsed_output == {"some": "data"}

    @pytest.mark.asyncio
    async def test_ai_message_with_non_string_non_dict_content(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=_make_ai_message(12345)
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "12345"

    @pytest.mark.asyncio
    async def test_plain_dict_response(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"key": "value", "number": 42}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.parsed_output == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_string_response(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="just a string")
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "just a string"

    @pytest.mark.asyncio
    async def test_exception_returns_error_result(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == ""
        assert result.error == "API error"
        assert result.transcription_not_possible is True

    @pytest.mark.asyncio
    async def test_token_tracking_called(self, provider):
        raw_msg = _make_ai_message("text", metadata={
            "usage": {"input_tokens": 100, "output_tokens": 200},
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": {"text": "hello"}}
        )
        mock_tracker = MagicMock()
        with patch(
            "modules.token_tracker.get_token_tracker",
            return_value=mock_tracker,
        ):
            result = await provider._invoke_llm(mock_llm, [])
        assert result.total_tokens == 300
        mock_tracker.add_tokens.assert_called_once_with(300)

    @pytest.mark.asyncio
    async def test_token_tracking_error_does_not_propagate(self, provider):
        raw_msg = _make_ai_message("text", metadata={
            "usage": {"input_tokens": 10, "output_tokens": 20},
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": {"text": "hello"}}
        )
        with patch(
            "modules.token_tracker.get_token_tracker",
            side_effect=Exception("tracker broke"),
        ):
            result = await provider._invoke_llm(mock_llm, [])
        # Should still return a valid result
        assert result.total_tokens == 30


# ============================================================================
# OpenAI Provider
# ============================================================================

class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openai",
            is_reasoning_model=False,
        ),
    )
    def test_init_standard_model(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(
            api_key="key",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )
        assert provider.provider_name == "openai"
        assert provider.model == "gpt-4o"
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["frequency_penalty"] == 0.1
        assert call_kwargs["presence_penalty"] == 0.2

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openai",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
        ),
    )
    def test_init_reasoning_model(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(
            api_key="key",
            model="o3",
            max_tokens=8192,
            reasoning_config={"effort": "high"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["max_completion_tokens"] == 8192
        assert call_kwargs["reasoning_effort"] == "high"
        assert "max_tokens" not in call_kwargs
        assert "temperature" not in call_kwargs

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openai",
            is_reasoning_model=True,
            supports_reasoning_effort=False,
        ),
    )
    def test_init_reasoning_model_no_effort_support(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        OpenAIProvider(
            api_key="key",
            model="o1-mini",
            reasoning_config={"effort": "high"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(provider_name="openai"),
    )
    def test_init_with_service_tier(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        OpenAIProvider(api_key="key", model="gpt-4o", service_tier="flex")
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["service_tier"] == "flex"

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(provider_name="openai"),
    )
    def test_init_without_service_tier(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        OpenAIProvider(api_key="key", model="gpt-4o")
        call_kwargs = mock_chat.call_args.kwargs
        assert "service_tier" not in call_kwargs


class TestOpenAIBuildDisabledParams:
    """Tests for OpenAIProvider._build_disabled_params."""

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openai",
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
        ),
    )
    def test_all_supported_returns_none(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(api_key="k", model="gpt-4o")
        assert provider._build_disabled_params() is None

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openai",
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
        ),
    )
    def test_none_supported_returns_all_disabled(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(api_key="k", model="o3")
        disabled = provider._build_disabled_params()
        assert disabled is not None
        assert "temperature" in disabled
        assert "top_p" in disabled
        assert "frequency_penalty" in disabled
        assert "presence_penalty" in disabled

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch(
        "api.providers.openai_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openai",
            supports_temperature=False,
            supports_top_p=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=False,
        ),
    )
    def test_partial_disabled(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider(api_key="k", model="test")
        disabled = provider._build_disabled_params()
        assert "temperature" in disabled
        assert "presence_penalty" in disabled
        assert "top_p" not in disabled
        assert "frequency_penalty" not in disabled


class TestOpenAITranscription:
    """Tests for OpenAIProvider transcription methods."""

    @pytest.fixture
    def provider(self):
        with (
            patch("api.providers.openai_provider._load_max_retries", return_value=3),
            patch("api.providers.openai_provider.ChatOpenAI") as mock_chat,
            patch(
                "api.providers.openai_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="openai",
                    supports_vision=True,
                    supports_image_detail=True,
                    default_image_detail="high",
                    supports_structured_output=True,
                ),
            ),
        ):
            mock_chat.return_value.ainvoke = AsyncMock(
                return_value=_make_ai_message("transcribed")
            )
            from api.providers.openai_provider import OpenAIProvider
            yield OpenAIProvider(api_key="k", model="gpt-4o")

    @pytest.mark.asyncio
    async def test_transcribe_image_from_base64_no_vision(self):
        with (
            patch("api.providers.openai_provider._load_max_retries", return_value=3),
            patch("api.providers.openai_provider.ChatOpenAI"),
            patch(
                "api.providers.openai_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="openai",
                    supports_vision=False,
                ),
            ),
        ):
            from api.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="k", model="m")
            result = await provider.transcribe_image_from_base64(
                "abc", "image/png", system_prompt="test",
            )
            assert result.transcription_not_possible is True

    @pytest.mark.asyncio
    async def test_transcribe_image_from_base64_normalizes_detail(self, provider):
        result = await provider.transcribe_image_from_base64(
            "abc",
            "image/png",
            system_prompt="test",
            image_detail="AUTO",
        )
        # "AUTO" is not in ("low", "high"), so should fall back to default
        assert result.content == "transcribed"

    @pytest.mark.asyncio
    async def test_transcribe_image_from_base64_accepts_low_detail(self, provider):
        result = await provider.transcribe_image_from_base64(
            "abc",
            "image/png",
            system_prompt="test",
            image_detail="low",
        )
        assert result.content == "transcribed"

    @pytest.mark.asyncio
    async def test_transcribe_image_delegates(self, provider, sample_image_file):
        result = await provider.transcribe_image(
            sample_image_file,
            system_prompt="test",
        )
        assert result.content == "transcribed"

    @pytest.mark.asyncio
    async def test_close_is_noop(self, provider):
        await provider.close()  # Should not raise


class TestOpenAIInvokeLlm:
    """Tests for OpenAIProvider._invoke_llm response shapes."""

    @pytest.fixture
    def provider(self):
        with (
            patch("api.providers.openai_provider._load_max_retries", return_value=3),
            patch("api.providers.openai_provider.ChatOpenAI"),
            patch(
                "api.providers.openai_provider.detect_capabilities",
                return_value=_make_caps(provider_name="openai"),
            ),
        ):
            from api.providers.openai_provider import OpenAIProvider
            yield OpenAIProvider(api_key="k", model="m")

    @pytest.mark.asyncio
    async def test_structured_response_with_model_dump(self, provider):
        """Test parsed_data with model_dump (Pydantic model)."""
        class FakeModel:
            def model_dump(self):
                return {"field": "value"}

            def model_dump_json(self):
                return '{"field": "value"}'

        raw_msg = _make_ai_message("raw", metadata={
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80,
            },
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": FakeModel()}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == '{"field": "value"}'
        assert result.parsed_output == {"field": "value"}
        assert result.total_tokens == 80

    @pytest.mark.asyncio
    async def test_structured_response_parsed_dict(self, provider):
        parsed = {"text": "hello"}
        raw_msg = _make_ai_message("raw")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": parsed}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == json.dumps(parsed)

    @pytest.mark.asyncio
    async def test_structured_response_parsed_string(self, provider):
        raw_msg = _make_ai_message("raw")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": "just text"}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "just text"

    @pytest.mark.asyncio
    async def test_exception_handling(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ValueError("bad request"))
        result = await provider._invoke_llm(mock_llm, [])
        assert result.error == "bad request"
        assert result.transcription_not_possible is True

    @pytest.mark.asyncio
    async def test_token_usage_extracted_from_openai_format(self, provider):
        raw_msg = _make_ai_message("text", metadata={
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            },
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=raw_msg)
        result = await provider._invoke_llm(mock_llm, [])
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.total_tokens == 300


class TestOpenAILoadMaxRetries:
    def test_returns_configured_value(self):
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {
            "retry": {"max_attempts": 10},
        }
        with patch(
            "modules.config_loader.get_config_loader",
            return_value=mock_loader,
        ):
            from api.providers.openai_provider import _load_max_retries
            assert _load_max_retries() == 10

    def test_returns_default_on_exception(self):
        with patch(
            "modules.config_loader.get_config_loader",
            side_effect=ImportError("no module"),
        ):
            from api.providers.openai_provider import _load_max_retries
            assert _load_max_retries() == 5


# ============================================================================
# Google Provider
# ============================================================================

class TestGoogleProvider:
    """Tests for GoogleProvider."""

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch(
        "api.providers.google_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="google",
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_basic(self, mock_detect, mock_chat, mock_retries):
        from api.providers.google_provider import GoogleProvider
        provider = GoogleProvider(
            api_key="key",
            model="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=2048,
        )
        assert provider.provider_name == "google"
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["google_api_key"] == "key"
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["max_output_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.3

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch(
        "api.providers.google_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="google",
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_with_reasoning_low(self, mock_detect, mock_chat, mock_retries):
        from api.providers.google_provider import GoogleProvider
        GoogleProvider(
            api_key="key",
            model="gemini-2.5-pro",
            reasoning_config={"effort": "low"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["thinking_level"] == "low"

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch(
        "api.providers.google_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="google",
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_with_reasoning_high(self, mock_detect, mock_chat, mock_retries):
        from api.providers.google_provider import GoogleProvider
        GoogleProvider(
            api_key="key",
            model="gemini-2.5-pro",
            reasoning_config={"effort": "high"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["thinking_level"] == "high"

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch(
        "api.providers.google_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="google",
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_with_reasoning_medium(self, mock_detect, mock_chat, mock_retries):
        from api.providers.google_provider import GoogleProvider
        GoogleProvider(
            api_key="key",
            model="gemini-2.5-pro",
            reasoning_config={"effort": "medium"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["thinking_level"] == "high"

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch(
        "api.providers.google_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="google",
            supports_temperature=True,
            supports_top_p=True,
        ),
    )
    def test_init_with_top_k(self, mock_detect, mock_chat, mock_retries):
        from api.providers.google_provider import GoogleProvider
        GoogleProvider(api_key="key", model="gemini", top_k=50)
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["top_k"] == 50

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch(
        "api.providers.google_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="google",
            supports_temperature=False,
            supports_top_p=False,
        ),
    )
    def test_init_skips_unsupported_params(self, mock_detect, mock_chat, mock_retries):
        from api.providers.google_provider import GoogleProvider
        GoogleProvider(api_key="key", model="test")
        call_kwargs = mock_chat.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch(
        "api.providers.google_provider.detect_capabilities",
        return_value=_make_caps(provider_name="google"),
    )
    def test_get_capabilities(self, mock_detect, mock_chat, mock_retries):
        from api.providers.google_provider import GoogleProvider
        provider = GoogleProvider(api_key="k", model="m")
        caps = provider.get_capabilities()
        assert caps.provider_name == "google"


class TestGoogleInvokeLlm:
    """Tests for GoogleProvider._invoke_llm, including list content handling."""

    @pytest.fixture
    def provider(self):
        with (
            patch("api.providers.google_provider._load_max_retries", return_value=3),
            patch("api.providers.google_provider.ChatGoogleGenerativeAI"),
            patch(
                "api.providers.google_provider.detect_capabilities",
                return_value=_make_caps(provider_name="google"),
            ),
        ):
            from api.providers.google_provider import GoogleProvider
            yield GoogleProvider(api_key="k", model="m")

    @pytest.mark.asyncio
    async def test_ai_message_with_list_content_dict_parts(self, provider):
        """Gemini can return content as a list of part dicts."""
        msg = _make_ai_message([
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World"},
        ])
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=msg)
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "Hello World"

    @pytest.mark.asyncio
    async def test_ai_message_with_list_content_string_parts(self, provider):
        msg = _make_ai_message(["part1", "part2"])
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=msg)
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "part1part2"

    @pytest.mark.asyncio
    async def test_structured_parsed_none_raw_list_content(self, provider):
        """Structured output with parsed=None and raw content as list."""
        raw_msg = _make_ai_message([
            {"type": "text", "text": "text1"},
            "text2",
        ])
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": None}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "text1text2"

    @pytest.mark.asyncio
    async def test_google_token_extraction(self, provider):
        msg = _make_ai_message("text", metadata={
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 50,
                "total_token_count": 150,
            },
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=msg)
        result = await provider._invoke_llm(mock_llm, [])
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    @pytest.mark.asyncio
    async def test_google_token_extraction_fallback_sum(self, provider):
        """When total_token_count is 0, should sum input + output."""
        msg = _make_ai_message("text", metadata={
            "usage_metadata": {
                "prompt_token_count": 80,
                "candidates_token_count": 40,
                "total_token_count": 0,
            },
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=msg)
        result = await provider._invoke_llm(mock_llm, [])
        assert result.total_tokens == 120

    @pytest.mark.asyncio
    async def test_exception_handling(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("Gemini error"))
        result = await provider._invoke_llm(mock_llm, [])
        assert result.error == "Gemini error"
        assert result.transcription_not_possible is True


class TestGoogleTranscription:
    """Tests for GoogleProvider transcription methods."""

    @pytest.mark.asyncio
    async def test_transcribe_no_vision(self):
        with (
            patch("api.providers.google_provider._load_max_retries", return_value=3),
            patch("api.providers.google_provider.ChatGoogleGenerativeAI"),
            patch(
                "api.providers.google_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="google",
                    supports_vision=False,
                ),
            ),
        ):
            from api.providers.google_provider import GoogleProvider
            provider = GoogleProvider(api_key="k", model="m")
            result = await provider.transcribe_image_from_base64(
                "abc", "image/png", system_prompt="test",
            )
            assert result.transcription_not_possible is True

    @pytest.mark.asyncio
    async def test_transcribe_delegates_to_base64(self, sample_image_file):
        with (
            patch("api.providers.google_provider._load_max_retries", return_value=3),
            patch("api.providers.google_provider.ChatGoogleGenerativeAI") as mock_chat,
            patch(
                "api.providers.google_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="google",
                    supports_vision=True,
                ),
            ),
        ):
            mock_chat.return_value.ainvoke = AsyncMock(
                return_value=_make_ai_message("google result")
            )
            from api.providers.google_provider import GoogleProvider
            provider = GoogleProvider(api_key="k", model="m")
            result = await provider.transcribe_image(
                sample_image_file, system_prompt="test",
            )
            assert result.content == "google result"

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        with (
            patch("api.providers.google_provider._load_max_retries", return_value=3),
            patch("api.providers.google_provider.ChatGoogleGenerativeAI"),
            patch(
                "api.providers.google_provider.detect_capabilities",
                return_value=_make_caps(provider_name="google"),
            ),
        ):
            from api.providers.google_provider import GoogleProvider
            provider = GoogleProvider(api_key="k", model="m")
            await provider.close()


class TestGoogleLoadMaxRetries:
    def test_returns_configured_value(self):
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {
            "retry": {"max_attempts": 4},
        }
        with patch(
            "modules.config_loader.get_config_loader",
            return_value=mock_loader,
        ):
            from api.providers.google_provider import _load_max_retries
            assert _load_max_retries() == 4

    def test_returns_default_on_exception(self):
        with patch(
            "modules.config_loader.get_config_loader",
            side_effect=Exception("err"),
        ):
            from api.providers.google_provider import _load_max_retries
            assert _load_max_retries() == 5


# ============================================================================
# OpenRouter Provider -- standalone functions
# ============================================================================

class TestEffortToRatio:
    """Tests for _effort_to_ratio."""

    def test_known_efforts(self):
        from api.providers.openrouter_provider import _effort_to_ratio
        assert _effort_to_ratio("xhigh") == 0.95
        assert _effort_to_ratio("high") == 0.80
        assert _effort_to_ratio("medium") == 0.50
        assert _effort_to_ratio("low") == 0.20
        assert _effort_to_ratio("minimal") == 0.10
        assert _effort_to_ratio("none") == 0.0

    def test_case_insensitive(self):
        from api.providers.openrouter_provider import _effort_to_ratio
        assert _effort_to_ratio("HIGH") == 0.80
        assert _effort_to_ratio("  Medium  ") == 0.50

    def test_unknown_defaults_to_medium(self):
        from api.providers.openrouter_provider import _effort_to_ratio
        assert _effort_to_ratio("unknown") == 0.50
        assert _effort_to_ratio("") == 0.50

    def test_none_defaults_to_medium(self):
        from api.providers.openrouter_provider import _effort_to_ratio
        assert _effort_to_ratio(None) == 0.50


class TestComputeOpenRouterReasoningMaxTokens:
    """Tests for _compute_openrouter_reasoning_max_tokens."""

    def test_medium_effort(self):
        from api.providers.openrouter_provider import _compute_openrouter_reasoning_max_tokens
        result = _compute_openrouter_reasoning_max_tokens(max_tokens=8192, effort="medium")
        # ratio 0.50 -> budget = 4096, upper = 7936, min(4096, 32000, 7936) = 4096, max(4096, 1024)
        assert result == 4096

    def test_high_effort(self):
        from api.providers.openrouter_provider import _compute_openrouter_reasoning_max_tokens
        result = _compute_openrouter_reasoning_max_tokens(max_tokens=8192, effort="high")
        # ratio 0.80 -> budget = 6553, upper = 7936, min(6553, 32000, 7936) = 6553
        assert result == 6553

    def test_low_effort(self):
        from api.providers.openrouter_provider import _compute_openrouter_reasoning_max_tokens
        result = _compute_openrouter_reasoning_max_tokens(max_tokens=8192, effort="low")
        # ratio 0.20 -> budget = 1638, upper = 7936, min(1638, 32000, 7936) = 1638
        assert result == 1638

    def test_none_effort_returns_zero(self):
        from api.providers.openrouter_provider import _compute_openrouter_reasoning_max_tokens
        result = _compute_openrouter_reasoning_max_tokens(max_tokens=8192, effort="none")
        assert result == 0

    def test_xhigh_effort_caps_at_32000(self):
        from api.providers.openrouter_provider import _compute_openrouter_reasoning_max_tokens
        result = _compute_openrouter_reasoning_max_tokens(max_tokens=100000, effort="xhigh")
        # ratio 0.95 -> budget = 95000, upper = 99744, min(95000, 32000, 99744) = 32000
        assert result == 32000

    def test_minimum_is_1024(self):
        from api.providers.openrouter_provider import _compute_openrouter_reasoning_max_tokens
        result = _compute_openrouter_reasoning_max_tokens(max_tokens=2000, effort="minimal")
        # ratio 0.10 -> budget = 200, upper = 1744, min(200, 32000, 1744) = 200, max(200, 1024) = 1024
        assert result == 1024


class TestOpenRouterLoadMaxRetries:
    def test_returns_configured_value(self):
        mock_loader = MagicMock()
        mock_loader.get_concurrency_config.return_value = {
            "retry": {"max_attempts": 2},
        }
        with patch(
            "modules.config_loader.get_config_loader",
            return_value=mock_loader,
        ):
            from api.providers.openrouter_provider import _load_max_retries
            assert _load_max_retries() == 2

    def test_returns_default_on_exception(self):
        with patch(
            "modules.config_loader.get_config_loader",
            side_effect=Exception("err"),
        ):
            from api.providers.openrouter_provider import _load_max_retries
            assert _load_max_retries() == 5


# ============================================================================
# OpenRouter Provider -- class
# ============================================================================

class TestOpenRouterProvider:
    """Tests for OpenRouterProvider."""

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=False,
        ),
    )
    def test_init_basic(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openrouter_provider import OpenRouterProvider
        provider = OpenRouterProvider(
            api_key="key",
            model="openai/gpt-4o",
            temperature=0.5,
        )
        assert provider.provider_name == "openrouter"
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["model"] == "openai/gpt-4o"

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=False,
        ),
    )
    def test_init_default_headers(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider(
            api_key="key",
            model="test/model",
            site_url="https://example.com",
            app_name="TestApp",
        )
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://example.com"
        assert call_kwargs["default_headers"]["X-Title"] == "TestApp"

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=False,
        ),
    )
    def test_init_default_app_name(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openrouter_provider import OpenRouterProvider
        provider = OpenRouterProvider(api_key="key", model="test/model")
        assert provider.app_name == "AutoExcerpter"

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=True,
        ),
    )
    def test_init_with_reasoning_openai_model(self, mock_detect, mock_chat, mock_retries):
        """For non-Anthropic/non-Gemini models, reasoning.effort is passed directly."""
        from api.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider(
            api_key="key",
            model="openai/o3",
            reasoning_config={"effort": "high"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        assert "reasoning" in extra_body
        assert extra_body["reasoning"]["effort"] == "high"

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=True,
        ),
    )
    def test_init_with_reasoning_anthropic_model(self, mock_detect, mock_chat, mock_retries):
        """For Anthropic models via OpenRouter, effort is mapped to max_tokens."""
        from api.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider(
            api_key="key",
            model="anthropic/claude-sonnet-4-5",
            reasoning_config={"effort": "medium"},
            max_tokens=8192,
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        reasoning = extra_body["reasoning"]
        assert "max_tokens" in reasoning
        # effort should have been popped for Anthropic models
        assert "effort" not in reasoning

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=True,
        ),
    )
    def test_init_with_reasoning_deepseek_model(self, mock_detect, mock_chat, mock_retries):
        """DeepSeek models use enabled flag instead of effort."""
        from api.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider(
            api_key="key",
            model="deepseek/deepseek-r1",
            reasoning_config={"effort": "high"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        reasoning = extra_body["reasoning"]
        assert reasoning.get("enabled") is True
        assert "effort" not in reasoning

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=True,
        ),
    )
    def test_init_with_reasoning_deepseek_effort_none(self, mock_detect, mock_chat, mock_retries):
        """DeepSeek model with effort=none should set enabled=False."""
        from api.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider(
            api_key="key",
            model="deepseek/deepseek-r1",
            reasoning_config={"effort": "none"},
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        reasoning = extra_body["reasoning"]
        assert reasoning.get("enabled") is False

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=True,
        ),
    )
    def test_init_with_reasoning_explicit_max_tokens(self, mock_detect, mock_chat, mock_retries):
        """Explicit max_tokens in reasoning_config should be used as-is."""
        from api.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider(
            api_key="key",
            model="anthropic/claude-opus-4-5",
            reasoning_config={"effort": "high", "max_tokens": 5000},
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        reasoning = extra_body["reasoning"]
        assert reasoning["max_tokens"] == 5000

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_reasoning_effort=True,
        ),
    )
    def test_init_with_reasoning_exclude_and_enabled(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider(
            api_key="key",
            model="openai/o3",
            reasoning_config={"effort": "high", "exclude": True, "enabled": False},
        )
        call_kwargs = mock_chat.call_args.kwargs
        extra_body = call_kwargs.get("extra_body", {})
        reasoning = extra_body["reasoning"]
        assert reasoning["exclude"] is True
        assert reasoning["enabled"] is False


class TestOpenRouterBuildDisabledParams:
    """Tests for OpenRouterProvider._build_disabled_params."""

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
        ),
    )
    def test_all_supported_returns_none(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openrouter_provider import OpenRouterProvider
        provider = OpenRouterProvider(api_key="k", model="m")
        assert provider._build_disabled_params() is None

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch(
        "api.providers.openrouter_provider.detect_capabilities",
        return_value=_make_caps(
            provider_name="openrouter",
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
        ),
    )
    def test_none_supported(self, mock_detect, mock_chat, mock_retries):
        from api.providers.openrouter_provider import OpenRouterProvider
        provider = OpenRouterProvider(api_key="k", model="m")
        disabled = provider._build_disabled_params()
        assert len(disabled) == 4


class TestOpenRouterTranscription:
    """Tests for OpenRouterProvider transcription methods."""

    @pytest.fixture
    def provider(self):
        with (
            patch("api.providers.openrouter_provider._load_max_retries", return_value=3),
            patch("api.providers.openrouter_provider.ChatOpenAI") as mock_chat,
            patch(
                "api.providers.openrouter_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="openrouter",
                    supports_vision=True,
                    supports_image_detail=True,
                    default_image_detail="high",
                    supports_structured_output=True,
                ),
            ),
        ):
            mock_chat.return_value.ainvoke = AsyncMock(
                return_value=_make_ai_message("openrouter result")
            )
            from api.providers.openrouter_provider import OpenRouterProvider
            yield OpenRouterProvider(api_key="k", model="m")

    @pytest.mark.asyncio
    async def test_transcribe_from_base64_no_vision(self):
        with (
            patch("api.providers.openrouter_provider._load_max_retries", return_value=3),
            patch("api.providers.openrouter_provider.ChatOpenAI"),
            patch(
                "api.providers.openrouter_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="openrouter",
                    supports_vision=False,
                ),
            ),
        ):
            from api.providers.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(api_key="k", model="m")
            result = await provider.transcribe_image_from_base64(
                "abc", "image/png", system_prompt="test",
            )
            assert result.transcription_not_possible is True

    @pytest.mark.asyncio
    async def test_transcribe_from_base64_detail_normalization(self, provider):
        result = await provider.transcribe_image_from_base64(
            "abc", "image/png", system_prompt="test", image_detail="HIGH",
        )
        assert result.content == "openrouter result"

    @pytest.mark.asyncio
    async def test_transcribe_from_base64_with_schema(self):
        with (
            patch("api.providers.openrouter_provider._load_max_retries", return_value=3),
            patch("api.providers.openrouter_provider.ChatOpenAI") as mock_chat,
            patch(
                "api.providers.openrouter_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="openrouter",
                    supports_vision=True,
                    supports_structured_output=True,
                ),
            ),
        ):
            parsed = {"text": "hello"}
            raw_msg = _make_ai_message("raw")
            mock_structured = MagicMock()
            mock_structured.ainvoke = AsyncMock(
                return_value={"raw": raw_msg, "parsed": parsed}
            )
            mock_llm = mock_chat.return_value
            mock_llm.with_structured_output.return_value = mock_structured

            from api.providers.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(api_key="k", model="m")
            result = await provider.transcribe_image_from_base64(
                "data",
                "image/png",
                system_prompt="test",
                json_schema={"type": "object"},
            )
            assert result.parsed_output == parsed
            mock_llm.with_structured_output.assert_called_once()
            # Check json_mode method is used
            call_kwargs = mock_llm.with_structured_output.call_args
            assert call_kwargs.kwargs["method"] == "json_mode"

    @pytest.mark.asyncio
    async def test_transcribe_from_base64_schema_fallback_on_error(self):
        """When with_structured_output raises, should fall back to standard output."""
        with (
            patch("api.providers.openrouter_provider._load_max_retries", return_value=3),
            patch("api.providers.openrouter_provider.ChatOpenAI") as mock_chat,
            patch(
                "api.providers.openrouter_provider.detect_capabilities",
                return_value=_make_caps(
                    provider_name="openrouter",
                    supports_vision=True,
                    supports_structured_output=True,
                ),
            ),
        ):
            mock_llm = mock_chat.return_value
            mock_llm.with_structured_output.side_effect = NotImplementedError("nope")
            mock_llm.ainvoke = AsyncMock(
                return_value=_make_ai_message("fallback result")
            )

            from api.providers.openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(api_key="k", model="m")
            result = await provider.transcribe_image_from_base64(
                "data",
                "image/png",
                system_prompt="test",
                json_schema={"type": "object"},
            )
            assert result.content == "fallback result"

    @pytest.mark.asyncio
    async def test_transcribe_image_delegates(self, provider, sample_image_file):
        result = await provider.transcribe_image(
            sample_image_file, system_prompt="test",
        )
        assert result.content == "openrouter result"

    @pytest.mark.asyncio
    async def test_close_is_noop(self, provider):
        await provider.close()


class TestOpenRouterInvokeLlm:
    """Tests for OpenRouterProvider._invoke_llm response shapes."""

    @pytest.fixture
    def provider(self):
        with (
            patch("api.providers.openrouter_provider._load_max_retries", return_value=3),
            patch("api.providers.openrouter_provider.ChatOpenAI"),
            patch(
                "api.providers.openrouter_provider.detect_capabilities",
                return_value=_make_caps(provider_name="openrouter"),
            ),
        ):
            from api.providers.openrouter_provider import OpenRouterProvider
            yield OpenRouterProvider(api_key="k", model="m")

    @pytest.mark.asyncio
    async def test_structured_with_model_dump(self, provider):
        class FakePydantic:
            def model_dump(self):
                return {"f": "v"}

            def model_dump_json(self):
                return '{"f": "v"}'

        raw_msg = _make_ai_message("raw")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": FakePydantic()}
        )
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == '{"f": "v"}'
        assert result.parsed_output == {"f": "v"}

    @pytest.mark.asyncio
    async def test_token_extraction_openai_format(self, provider):
        msg = _make_ai_message("text", metadata={
            "token_usage": {
                "prompt_tokens": 300,
                "completion_tokens": 150,
                "total_tokens": 450,
            },
        })
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=msg)
        result = await provider._invoke_llm(mock_llm, [])
        assert result.input_tokens == 300
        assert result.output_tokens == 150
        assert result.total_tokens == 450

    @pytest.mark.asyncio
    async def test_plain_string_response(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="plain string")
        result = await provider._invoke_llm(mock_llm, [])
        assert result.content == "plain string"

    @pytest.mark.asyncio
    async def test_plain_dict_response(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value={"a": 1})
        result = await provider._invoke_llm(mock_llm, [])
        assert result.parsed_output == {"a": 1}

    @pytest.mark.asyncio
    async def test_exception_handling(self, provider):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("timeout"))
        result = await provider._invoke_llm(mock_llm, [])
        assert result.error == "timeout"
        assert result.transcription_not_possible is True


# ============================================================================
# __init__.py lazy imports
# ============================================================================

class TestProvidersInit:
    """Tests for api/providers/__init__.py lazy __getattr__."""

    def test_import_base_provider(self):
        import api.providers
        cls = api.providers.BaseProvider
        assert cls is BaseProvider

    def test_import_transcription_result(self):
        import api.providers
        cls = api.providers.TranscriptionResult
        assert cls is TranscriptionResult

    def test_import_provider_capabilities(self):
        import api.providers
        cls = api.providers.ProviderCapabilities
        assert cls is ProviderCapabilities

    def test_import_provider_type(self):
        import api.providers
        pt = api.providers.ProviderType
        assert pt is not None

    def test_import_get_provider(self):
        import api.providers
        func = api.providers.get_provider
        assert callable(func)

    def test_import_get_available_providers(self):
        import api.providers
        func = api.providers.get_available_providers
        assert callable(func)

    def test_import_detect_provider_from_model(self):
        import api.providers
        func = api.providers.detect_provider_from_model
        assert callable(func)

    def test_import_get_api_key_for_provider(self):
        import api.providers
        func = api.providers.get_api_key_for_provider
        assert callable(func)

    @patch("api.providers.anthropic_provider._load_max_retries", return_value=3)
    @patch("api.providers.anthropic_provider.ChatAnthropic")
    @patch("api.providers.anthropic_provider.detect_capabilities", return_value=_make_caps())
    def test_import_anthropic_provider(self, mock_detect, mock_chat, mock_retries):
        import api.providers
        cls = api.providers.AnthropicProvider
        assert cls is not None

    @patch("api.providers.openai_provider._load_max_retries", return_value=3)
    @patch("api.providers.openai_provider.ChatOpenAI")
    @patch("api.providers.openai_provider.detect_capabilities", return_value=_make_caps())
    def test_import_openai_provider(self, mock_detect, mock_chat, mock_retries):
        import api.providers
        cls = api.providers.OpenAIProvider
        assert cls is not None

    @patch("api.providers.google_provider._load_max_retries", return_value=3)
    @patch("api.providers.google_provider.ChatGoogleGenerativeAI")
    @patch("api.providers.google_provider.detect_capabilities", return_value=_make_caps())
    def test_import_google_provider(self, mock_detect, mock_chat, mock_retries):
        import api.providers
        cls = api.providers.GoogleProvider
        assert cls is not None

    @patch("api.providers.openrouter_provider._load_max_retries", return_value=3)
    @patch("api.providers.openrouter_provider.ChatOpenAI")
    @patch("api.providers.openrouter_provider.detect_capabilities", return_value=_make_caps())
    def test_import_openrouter_provider(self, mock_detect, mock_chat, mock_retries):
        import api.providers
        cls = api.providers.OpenRouterProvider
        assert cls is not None

    def test_unknown_attribute_raises(self):
        import api.providers
        with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
            _ = api.providers.NonExistent
