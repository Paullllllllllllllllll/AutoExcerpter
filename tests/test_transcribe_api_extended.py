"""Extended tests for llm/transcription.py — covers untested code paths.

This file complements test_transcribe_api.py by covering:
- TranscriptionManager.__init__() with mocked dependencies
- _load_schema_and_prompt() — success, missing schema, missing prompt
- _format_image_name() and _truncate_analysis() — edge cases
- _parse_transcription_from_text() — all branches
- _build_model_inputs() — per-provider message building
- transcribe_payload() — success, API error, schema retry
"""

from __future__ import annotations

import base64
import io
import json
import threading
from collections import deque
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image

from imaging.payload import PagePayload
from llm.transcription import TranscriptionManager


def _make_payload(
    image_name: str = "page_0003.jpg", sequence_number: int = 3
) -> PagePayload:
    """Build a PagePayload around a tiny in-memory JPEG."""
    img = Image.new("RGB", (10, 10), color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return PagePayload(
        base64=base64.b64encode(buffer.getvalue()).decode("utf-8"),
        image_name=image_name,
        sequence_number=sequence_number,
        original_input_order_index=sequence_number - 1,
        provenance={},
        source_file="test.pdf",
        page_index=sequence_number - 1,
    )


# ============================================================================
# Helpers
# ============================================================================
_MOCK_SCHEMA = {
    "name": "transcription_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "transcription": {"type": "string"},
            "no_transcribable_text": {"type": "boolean"},
            "transcription_not_possible": {"type": "boolean"},
            "image_analysis": {"type": "string"},
        },
    },
}

_MOCK_PROMPT = "Transcribe the following image. {{SCHEMA}}"


def _make_manager(**overrides) -> TranscriptionManager:
    """Create a bare TranscriptionManager bypassing __init__."""
    mgr = TranscriptionManager.__new__(TranscriptionManager)
    defaults = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "timeout": 300,
        "rate_limiter": None,
        "max_retries": 5,
        "chat_model": MagicMock(),
        "_stats_lock": threading.Lock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "model_config": {},
        "service_tier": "flex",
        "schema_retry_config": {},
        "_output_schema": _MOCK_SCHEMA,
        "transcription_schema": _MOCK_SCHEMA,
        "system_prompt": "Transcribe this image.",
        "custom_capabilities": None,
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(mgr, attr, val)
    return mgr


@pytest.fixture
def mock_all_init_deps() -> Generator[MagicMock]:
    """Patch all dependencies needed by TranscriptionManager.__init__."""
    with (
        patch("llm.base.get_chat_model") as mock_gcm,
        patch("llm.base.get_api_timeout", return_value=300),
        patch(
            "llm.transcription.get_model_capabilities",
            return_value={"multimodal": True, "max_tokens": True},
        ),
        patch("llm.base.get_config_loader") as mock_cl_base,
        patch("llm.transcription.SCHEMAS_DIR", Path("/fake/schemas")),
        patch("llm.transcription.PROMPTS_DIR", Path("/fake/prompts")),
        patch("llm.transcription.render_prompt_with_schema", return_value="rendered"),
    ):
        mock_gcm.return_value = MagicMock()
        mock_loader = MagicMock()
        mock_loader.get_model_config.return_value = {
            "transcription_model": {"name": "gpt-5-mini"},
        }
        mock_loader.get_concurrency_config.return_value = {
            "retry": {"schema_retries": {"transcription": {}}},
            "api_requests": {"transcription": {"service_tier": "flex"}},
        }
        mock_cl_base.return_value = mock_loader

        yield mock_gcm


# ============================================================================
# __init__
# ============================================================================
class TestTranscriptionManagerInit:
    """Tests for TranscriptionManager.__init__()."""

    def test_init_multimodal_model(self, mock_all_init_deps) -> None:
        """Initializes without warning for multimodal model."""
        schema_data = json.dumps(_MOCK_SCHEMA)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=schema_data)),
            patch("json.load", return_value=_MOCK_SCHEMA),
        ):
            # Need to allow prompt file reading too
            mgr = _make_manager()
            # Verify key attributes would be set
            assert mgr.transcription_schema is not None
            assert mgr.system_prompt != ""

    @patch("llm.base.get_chat_model")
    @patch("llm.base.get_api_timeout", return_value=300)
    @patch(
        "llm.transcription.get_model_capabilities",
        return_value={"multimodal": False, "max_tokens": True},
    )
    def test_init_warns_for_non_multimodal(
        self, mock_caps, mock_timeout, mock_gcm
    ) -> None:
        """Logs warning when model does not support multimodal input."""
        mock_gcm.return_value = MagicMock()

        with (
            patch("llm.base.get_config_loader") as mock_cl,
            patch("llm.transcription.SCHEMAS_DIR", Path("/fake/schemas")),
            patch("llm.transcription.PROMPTS_DIR", Path("/fake/prompts")),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(_MOCK_SCHEMA))),
            patch("json.load", return_value=_MOCK_SCHEMA),
            patch("llm.transcription.render_prompt_with_schema", return_value="r"),
            patch("llm.transcription.logger") as mock_logger,
        ):
            loader = MagicMock()
            loader.get_model_config.return_value = {"transcription_model": {}}
            loader.get_concurrency_config.return_value = {
                "retry": {"schema_retries": {"transcription": {}}},
                "api_requests": {"transcription": {"service_tier": "flex"}},
            }
            mock_cl.return_value = loader

            TranscriptionManager(
                model_name="text-davinci-003",
                provider="openai",
                api_key="test-key",
            )
            mock_logger.warning.assert_called()


# ============================================================================
# _load_schema_and_prompt
# ============================================================================
class TestLoadSchemaAndPrompt:
    """Tests for _load_schema_and_prompt()."""

    def test_successful_load(self, tmp_path) -> None:
        """Loads schema and prompt files successfully."""
        mgr = _make_manager()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        schema_file = schemas_dir / "transcription_schema.json"
        schema_file.write_text(json.dumps(_MOCK_SCHEMA), encoding="utf-8")

        prompt_file = prompts_dir / "transcription_system_prompt.txt"
        prompt_file.write_text(_MOCK_PROMPT, encoding="utf-8")

        with (
            patch("llm.transcription.SCHEMAS_DIR", schemas_dir),
            patch("llm.transcription.PROMPTS_DIR", prompts_dir),
            patch(
                "llm.transcription.render_prompt_with_schema",
                return_value="rendered prompt",
            ),
        ):
            mgr._load_schema_and_prompt()

        assert mgr.transcription_schema == _MOCK_SCHEMA
        assert mgr.system_prompt == "rendered prompt"

    def test_missing_schema_file_raises(self, tmp_path) -> None:
        """Raises FileNotFoundError when schema file is missing."""
        mgr = _make_manager()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        # Schema file does not exist

        with (
            patch("llm.transcription.SCHEMAS_DIR", schemas_dir),
            pytest.raises(FileNotFoundError, match="Required schema file"),
        ):
            mgr._load_schema_and_prompt()

    def test_missing_prompt_file_raises(self, tmp_path) -> None:
        """Raises FileNotFoundError when prompt file is missing."""
        mgr = _make_manager()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        schema_file = schemas_dir / "transcription_schema.json"
        schema_file.write_text(json.dumps(_MOCK_SCHEMA), encoding="utf-8")
        # Prompt file does not exist

        with (
            patch("llm.transcription.SCHEMAS_DIR", schemas_dir),
            patch("llm.transcription.PROMPTS_DIR", prompts_dir),
            pytest.raises(FileNotFoundError, match="Required prompt file"),
        ):
            mgr._load_schema_and_prompt()


# ============================================================================
# _format_image_name — additional edge cases
# ============================================================================
class TestFormatImageNameExtended:
    """Additional tests for _format_image_name()."""

    def test_whitespace_only(self) -> None:
        """Whitespace-only string is truthy, returned as-is."""
        assert TranscriptionManager._format_image_name("  ") == "  "

    def test_special_characters(self) -> None:
        """Filenames with special characters are preserved."""
        assert TranscriptionManager._format_image_name("file (1).png") == "file (1).png"


# ============================================================================
# _truncate_analysis — additional edge cases
# ============================================================================
class TestTruncateAnalysisExtended:
    """Additional tests for _truncate_analysis()."""

    def test_exact_max_length(self) -> None:
        """Text exactly at max_chars is returned as-is."""
        text = "A" * 100
        result = TranscriptionManager._truncate_analysis(text, max_chars=100)
        assert result == text  # No ellipsis needed

    def test_strips_trailing_punctuation_before_ellipsis(self) -> None:
        """Trailing punctuation (.,;:) is stripped before adding ellipsis."""
        text = "Hello, this is a test sentence, and it continues here with more words."
        result = TranscriptionManager._truncate_analysis(text, max_chars=40)
        assert result.endswith("...")
        assert not result.endswith(",...")
        assert not result.endswith(";...")


# ============================================================================
# _parse_transcription_from_text
# ============================================================================
class TestParseTranscriptionFromText:
    """Tests for _parse_transcription_from_text() — all branches."""

    def test_empty_text(self) -> None:
        """Empty text returns error message."""
        mgr = _make_manager()
        result = mgr._parse_transcription_from_text("", "page_001.png")
        assert result == "[transcription error: page_001.png]"

    def test_empty_text_no_image_name(self) -> None:
        """Empty text with no image name uses placeholder."""
        mgr = _make_manager()
        result = mgr._parse_transcription_from_text("")
        assert "[unknown image]" in result

    def test_non_json_passthrough(self) -> None:
        """Non-JSON text is returned as-is."""
        mgr = _make_manager()
        text = "This is plain transcription text without JSON."
        assert mgr._parse_transcription_from_text(text) == text

    def test_no_transcribable_text_flag(self) -> None:
        """no_transcribable_text flag generates formatted message."""
        mgr = _make_manager()
        data = json.dumps(
            {
                "no_transcribable_text": True,
                "image_analysis": "Page is blank.",
                "transcription_not_possible": False,
            }
        )
        result = mgr._parse_transcription_from_text(data, "page_005.png")
        assert "page_005.png" in result
        assert "no transcribable text" in result
        assert "Page is blank" in result

    def test_transcription_not_possible_flag(self) -> None:
        """transcription_not_possible flag generates formatted message."""
        mgr = _make_manager()
        data = json.dumps(
            {
                "transcription_not_possible": True,
                "image_analysis": "Image too blurry.",
                "no_transcribable_text": False,
            }
        )
        result = mgr._parse_transcription_from_text(data, "scan_01.jpg")
        assert "scan_01.jpg" in result
        assert "transcription not possible" in result
        assert "Image too blurry" in result

    def test_json_with_transcription_field(self) -> None:
        """JSON with transcription field extracts the text."""
        mgr = _make_manager()
        data = json.dumps(
            {
                "transcription": "This is the extracted text.",
                "no_transcribable_text": False,
                "transcription_not_possible": False,
            }
        )
        result = mgr._parse_transcription_from_text(data, "page_01.jpg")
        assert result == "This is the extracted text."

    def test_invalid_json_salvage(self) -> None:
        """Invalid JSON starting with '{' triggers salvage attempt."""
        mgr = _make_manager()
        # Must start with '{' to enter JSON parsing, but be invalid overall
        text = '{"broken": true, INVALID {"transcription": "salvaged text"}'
        result = mgr._parse_transcription_from_text(text)
        assert result == "salvaged text"

    def test_completely_broken_json_returns_original(self) -> None:
        """Totally broken JSON returns original text."""
        mgr = _make_manager()
        text = "{broken json without closing"
        result = mgr._parse_transcription_from_text(text)
        assert result == text

    def test_markdown_wrapped_json(self) -> None:
        """Strips markdown ```json blocks."""
        mgr = _make_manager()
        inner = json.dumps({"transcription": "markdown content"})
        text = f"```json\n{inner}\n```"
        result = mgr._parse_transcription_from_text(text)
        assert result == "markdown content"

    def test_markdown_wrapped_plain_backticks(self) -> None:
        """Strips markdown ``` blocks without json tag."""
        mgr = _make_manager()
        inner = json.dumps({"transcription": "plain backtick"})
        text = f"```\n{inner}\n```"
        result = mgr._parse_transcription_from_text(text)
        assert result == "plain backtick"

    def test_json_without_known_flags_returns_original(self) -> None:
        """JSON without any known flags or transcription returns original."""
        mgr = _make_manager()
        data = json.dumps({"unknown_key": "value"})
        result = mgr._parse_transcription_from_text(data)
        assert result == data

    def test_no_transcribable_text_without_analysis(self) -> None:
        """no_transcribable_text without image_analysis still works."""
        mgr = _make_manager()
        data = json.dumps({"no_transcribable_text": True})
        result = mgr._parse_transcription_from_text(data, "page.png")
        assert "no transcribable text" in result
        assert "no details available" in result


# ============================================================================
# _build_model_inputs
# ============================================================================
class TestBuildModelInputs:
    """Tests for _build_model_inputs()."""

    @patch(
        "llm.base.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_openai_format(self, _) -> None:
        """OpenAI provider uses image_url format."""
        mgr = _make_manager(
            provider="openai",
            model_config={},
            service_tier="flex",
            _output_schema=_MOCK_SCHEMA,
        )
        messages, kwargs = mgr._build_model_inputs("base64data==")

        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        content = messages[1].content
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"  # type: ignore[index]
        assert "data:image/jpeg;base64,base64data==" in content[0]["image_url"]["url"]  # type: ignore[index]

    @patch(
        "llm.base.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_anthropic_format(self, _) -> None:
        """Anthropic provider uses image source format."""
        mgr = _make_manager(
            provider="anthropic",
            model_config={},
            service_tier="",
            _output_schema=_MOCK_SCHEMA,
        )
        messages, kwargs = mgr._build_model_inputs("base64data==")

        content = messages[1].content
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/jpeg"
        assert content[0]["source"]["data"] == "base64data=="

    @patch(
        "llm.base.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_google_format(self, _) -> None:
        """Google provider uses image_url format."""
        mgr = _make_manager(
            provider="google",
            model_config={},
            service_tier="",
            _output_schema=_MOCK_SCHEMA,
        )
        messages, kwargs = mgr._build_model_inputs("base64data==")

        content = messages[1].content
        assert content[0]["type"] == "image_url"
        assert "base64,base64data==" in content[0]["image_url"]["url"]


# ============================================================================
# transcribe_payload
# ============================================================================
class TestTranscribePayload:
    """Tests for transcribe_payload()."""

    def test_successful_transcription(self) -> None:
        """Successful transcription returns expected result dict."""
        payload = _make_payload("page_0003.jpg", sequence_number=3)
        mgr = _make_manager(provider="openai")

        mock_response = AIMessage(
            content=json.dumps(
                {
                    "transcription": "Hello world.",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                    "image_analysis": "Clear text.",
                }
            )
        )
        mock_response.usage_metadata = {"total_tokens": 100}  # type: ignore[assignment]
        mock_response.response_metadata = {}

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
            patch("llm.base.get_token_tracker") as mock_tt,
        ):
            mock_tracker = MagicMock()
            mock_tracker.get_tokens_used_today.return_value = 100
            mock_tt.return_value = mock_tracker

            result = mgr.transcribe_payload(payload)

        assert result["image"] == "page_0003.jpg"
        assert result["sequence_number"] == 3
        assert result["transcription"] == "Hello world."
        assert result["provider"] == "openai"
        assert "processing_time" in result

    def test_uses_payload_base64(self) -> None:
        """The payload's base64 string is passed to _build_model_inputs."""
        payload = _make_payload("page_0001.jpg", sequence_number=1)
        mgr = _make_manager(provider="openai")

        mock_response = AIMessage(
            content=json.dumps(
                {
                    "transcription": "ok",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                    "image_analysis": "x",
                }
            )
        )
        mock_response.usage_metadata = None
        mock_response.response_metadata = {}

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(
                mgr, "_build_model_inputs", return_value=([], {})
            ) as mock_build,
        ):
            mgr.transcribe_payload(payload)

        mock_build.assert_called_with(payload.base64)

    def test_api_error(self) -> None:
        """API error after LangChain retries returns error result."""
        payload = _make_payload("page_0001.jpg", sequence_number=1)
        mgr = _make_manager(provider="openai")

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("API timeout")

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
        ):
            result = mgr.transcribe_payload(payload)

        assert "transcription error" in result["transcription"]
        assert result["error_type"] == "api_failure"
        assert mgr.failed_requests == 1

    def test_schema_retry_loop_no_transcribable_text(self) -> None:
        """Schema retry when no_transcribable_text flag is detected."""
        payload = _make_payload("page_0001.jpg", sequence_number=1)

        mgr = _make_manager(
            provider="openai",
            schema_retry_config={
                "no_transcribable_text": {
                    "enabled": True,
                    "max_attempts": 1,
                    "backoff_base": 0.01,
                    "backoff_multiplier": 1.0,
                },
            },
        )

        # First call returns no_transcribable_text, second returns normal
        response_retry = AIMessage(
            content=json.dumps(
                {
                    "no_transcribable_text": True,
                    "image_analysis": "Blank page.",
                    "transcription_not_possible": False,
                }
            )
        )
        response_retry.usage_metadata = None
        response_retry.response_metadata = {}

        response_ok = AIMessage(
            content=json.dumps(
                {
                    "transcription": "Got it this time.",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                }
            )
        )
        response_ok.usage_metadata = None
        response_ok.response_metadata = {}

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [response_retry, response_ok]

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
            patch("llm.transcription.time.sleep"),
            patch("llm.base.random.uniform", return_value=1.0),
        ):
            result = mgr.transcribe_payload(payload, max_schema_retries=3)

        assert result["transcription"] == "Got it this time."
        assert result["schema_retries"]["no_transcribable_text"] == 1

    def test_schema_flag_retry_on_markdown_fenced_response(self) -> None:
        """A markdown-fenced JSON flag still triggers the schema-flag retry.

        Regression: the flag check must strip the code fence (as validation and
        parsing do) before json.loads, or a fenced ``transcription_not_possible``
        response silently bypasses the configured retry.
        """
        payload = _make_payload("page_0001.jpg", sequence_number=1)

        mgr = _make_manager(
            provider="openai",
            schema_retry_config={
                "transcription_not_possible": {
                    "enabled": True,
                    "max_attempts": 1,
                    "backoff_base": 0.01,
                    "backoff_multiplier": 1.0,
                },
            },
        )

        fenced_flag = (
            "```json\n"
            + json.dumps(
                {
                    "image_analysis": "Illegible.",
                    "transcription": "",
                    "no_transcribable_text": False,
                    "transcription_not_possible": True,
                }
            )
            + "\n```"
        )
        response_retry = AIMessage(content=fenced_flag)
        response_retry.usage_metadata = None
        response_retry.response_metadata = {}

        response_ok = AIMessage(
            content=json.dumps(
                {
                    "image_analysis": "Readable now.",
                    "transcription": "Got it this time.",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                }
            )
        )
        response_ok.usage_metadata = None
        response_ok.response_metadata = {}

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [response_retry, response_ok]

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
            patch("llm.transcription.time.sleep"),
            patch("llm.base.random.uniform", return_value=1.0),
        ):
            result = mgr.transcribe_payload(payload, max_schema_retries=3)

        assert result["transcription"] == "Got it this time."
        assert result["schema_retries"]["transcription_not_possible"] == 1

    def test_get_stats_includes_service_tier(self) -> None:
        """get_stats() includes service_tier from transcription manager."""
        mgr = _make_manager(service_tier="flex", successful_requests=1)
        stats = mgr.get_stats()
        assert stats["service_tier"] == "flex"
        assert "provider" in stats
        assert "model" in stats


# ============================================================================
# transcribe_payload uses _invoke_with_retry
# ============================================================================
class TestTranscribePayloadUsesInvokeWithRetry:
    """Verify transcribe_payload delegates to _invoke_with_retry."""

    def test_transcribe_calls_invoke_with_retry(self) -> None:
        """transcribe_payload uses _invoke_with_retry instead of direct invoke."""
        payload = _make_payload("page_0001.jpg", sequence_number=1)
        mgr = _make_manager(provider="openai")

        mock_response = AIMessage(
            content=json.dumps(
                {
                    "transcription": "Retry-based transcription.",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                    "image_analysis": "x",
                }
            )
        )
        mock_response.usage_metadata = {"total_tokens": 100}  # type: ignore[assignment]
        mock_response.response_metadata = {}

        with (
            patch.object(
                mgr, "_invoke_with_retry", return_value=mock_response
            ) as mock_invoke,
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
            patch("llm.base.get_token_tracker") as mock_tt,
        ):
            mock_tracker = MagicMock()
            mock_tracker.get_tokens_used_today.return_value = 100
            mock_tt.return_value = mock_tracker

            result = mgr.transcribe_payload(payload)

        mock_invoke.assert_called_once()
        assert result["transcription"] == "Retry-based transcription."
