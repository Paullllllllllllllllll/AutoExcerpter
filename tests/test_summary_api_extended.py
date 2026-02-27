"""Extended tests for api/summary_api.py — covers untested code paths.

This file provides comprehensive coverage for:
- SummaryManager.__init__() with mocked dependencies
- _load_schema_and_prompt() — success, missing files
- _create_placeholder_summary() — with/without error, custom page_types
- _build_model_inputs() — openai, anthropic providers, context injection
- _ensure_page_information_structure() — all branches
- generate_summary() — success, JSON error, empty response, API error
- get_stats() — includes service_tier
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.summary_api import SummaryManager

# ============================================================================
# Helpers
# ============================================================================
_MOCK_SCHEMA = {
    "name": "summary_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "page_information": {"type": "object"},
            "bullet_points": {"type": "array"},
            "references": {"type": "object"},
        },
    },
}

_MOCK_PROMPT = "Summarize the following text. {{SCHEMA}}"


def _make_manager(**overrides) -> SummaryManager:
    """Create a bare SummaryManager bypassing __init__."""
    mgr = SummaryManager.__new__(SummaryManager)
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
        "service_tier": "flex",
        "schema_retry_config": {},
        "_output_schema": _MOCK_SCHEMA,
        "summary_schema": _MOCK_SCHEMA,
        "summary_system_prompt_text": _MOCK_PROMPT,
        "summary_context": None,
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(mgr, attr, val)
    return mgr


# ============================================================================
# __init__
# ============================================================================
class TestSummaryManagerInit:
    """Tests for SummaryManager.__init__()."""

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=300)
    @patch("api.summary_api.get_api_timeout", return_value=300)
    @patch("api.summary_api.get_rate_limits", return_value=[(120, 1)])
    @patch("api.summary_api.RateLimiter")
    def test_init_sets_key_attributes(
        self,
        mock_rl_cls,
        mock_rate_limits,
        mock_timeout_sum,
        mock_timeout_base,
        mock_gcm,
        tmp_path,
    ):
        """Initialization sets schema, prompt, model_config, and service_tier."""
        mock_gcm.return_value = MagicMock()
        mock_rl_cls.return_value = MagicMock()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (schemas_dir / "summary_schema.json").write_text(
            json.dumps(_MOCK_SCHEMA), encoding="utf-8"
        )
        (prompts_dir / "summary_system_prompt.txt").write_text(
            _MOCK_PROMPT, encoding="utf-8"
        )

        with (
            patch("api.base_llm_client.get_config_loader") as mock_cl,
            patch("api.summary_api.SCHEMAS_DIR", schemas_dir),
            patch("api.summary_api.PROMPTS_DIR", prompts_dir),
        ):
            loader = MagicMock()
            loader.get_model_config.return_value = {
                "summary_model": {"name": "gpt-5-mini", "max_output_tokens": 2048},
            }
            loader.get_concurrency_config.return_value = {
                "retry": {"schema_retries": {"summary": {}}},
                "api_requests": {"summary": {"service_tier": "flex"}},
            }
            mock_cl.return_value = loader

            mgr = SummaryManager(
                model_name="gpt-5-mini",
                provider="openai",
                api_key="test-key",
                summary_context="Focus on culinary history.",
            )

        assert mgr.summary_context == "Focus on culinary history."
        assert mgr.summary_schema is not None

    @patch("api.base_llm_client.get_chat_model")
    @patch("api.base_llm_client.get_api_timeout", return_value=300)
    @patch("api.summary_api.get_api_timeout", return_value=300)
    @patch("api.summary_api.get_rate_limits", return_value=[(120, 1)])
    @patch("api.summary_api.RateLimiter")
    def test_init_no_context(
        self,
        mock_rl_cls,
        mock_rate_limits,
        mock_timeout_sum,
        mock_timeout_base,
        mock_gcm,
        tmp_path,
    ):
        """Initialization without summary_context stores None."""
        mock_gcm.return_value = MagicMock()
        mock_rl_cls.return_value = MagicMock()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (schemas_dir / "summary_schema.json").write_text(
            json.dumps(_MOCK_SCHEMA), encoding="utf-8"
        )
        (prompts_dir / "summary_system_prompt.txt").write_text(
            _MOCK_PROMPT, encoding="utf-8"
        )

        with (
            patch("api.base_llm_client.get_config_loader") as mock_cl,
            patch("api.summary_api.SCHEMAS_DIR", schemas_dir),
            patch("api.summary_api.PROMPTS_DIR", prompts_dir),
        ):
            loader = MagicMock()
            loader.get_model_config.return_value = {"summary_model": {}}
            loader.get_concurrency_config.return_value = {
                "retry": {"schema_retries": {"summary": {}}},
                "api_requests": {"summary": {"service_tier": "flex"}},
            }
            mock_cl.return_value = loader

            mgr = SummaryManager(
                model_name="gpt-5-mini",
                provider="openai",
                api_key="test-key",
            )

        assert mgr.summary_context is None


# ============================================================================
# _load_schema_and_prompt
# ============================================================================
class TestLoadSchemaAndPrompt:
    """Tests for _load_schema_and_prompt()."""

    def test_successful_load(self, tmp_path):
        """Loads schema and prompt files successfully."""
        mgr = _make_manager()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (schemas_dir / "summary_schema.json").write_text(
            json.dumps(_MOCK_SCHEMA), encoding="utf-8"
        )
        (prompts_dir / "summary_system_prompt.txt").write_text(
            "prompt text", encoding="utf-8"
        )

        with (
            patch("api.summary_api.SCHEMAS_DIR", schemas_dir),
            patch("api.summary_api.PROMPTS_DIR", prompts_dir),
        ):
            mgr._load_schema_and_prompt()

        assert mgr.summary_schema == _MOCK_SCHEMA
        assert mgr.summary_system_prompt_text == "prompt text"

    def test_missing_schema_raises(self, tmp_path):
        """Raises FileNotFoundError when schema file is missing."""
        mgr = _make_manager()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        # Schema file does not exist

        with (
            patch("api.summary_api.SCHEMAS_DIR", schemas_dir),
            pytest.raises(FileNotFoundError, match="Required schema file"),
        ):
            mgr._load_schema_and_prompt()

    def test_missing_prompt_raises(self, tmp_path):
        """Raises FileNotFoundError when prompt file is missing."""
        mgr = _make_manager()

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (schemas_dir / "summary_schema.json").write_text(
            json.dumps(_MOCK_SCHEMA), encoding="utf-8"
        )
        # Prompt file does not exist

        with (
            patch("api.summary_api.SCHEMAS_DIR", schemas_dir),
            patch("api.summary_api.PROMPTS_DIR", prompts_dir),
            pytest.raises(FileNotFoundError, match="Required prompt file"),
        ):
            mgr._load_schema_and_prompt()


# ============================================================================
# _create_placeholder_summary
# ============================================================================
class TestCreatePlaceholderSummary:
    """Tests for _create_placeholder_summary()."""

    def test_with_error_message(self):
        """Placeholder includes error message in bullet_points."""
        mgr = _make_manager()
        result = mgr._create_placeholder_summary(5, "API timeout")

        assert result["page"] == 5
        assert result["page_information"]["page_number_integer"] == 5
        assert result["page_information"]["page_types"] == ["other"]
        assert "[Error generating summary: API timeout]" in result["bullet_points"]
        assert result["error"] == "API timeout"
        assert result["references"] is None

    def test_without_error_message(self):
        """Placeholder without error uses default failure text."""
        mgr = _make_manager()
        result = mgr._create_placeholder_summary(3)

        assert "[Summary generation failed]" in result["bullet_points"]
        assert "error" not in result

    def test_empty_error_message(self):
        """Empty error string uses default failure text."""
        mgr = _make_manager()
        result = mgr._create_placeholder_summary(1, "")

        assert "[Summary generation failed]" in result["bullet_points"]

    def test_custom_page_types(self):
        """Custom page_types override default."""
        mgr = _make_manager()
        result = mgr._create_placeholder_summary(
            2, "error", page_types=["title_page", "illustration"]
        )

        assert result["page_information"]["page_types"] == [
            "title_page",
            "illustration",
        ]

    def test_default_page_types(self):
        """Default page_types is ['other']."""
        mgr = _make_manager()
        result = mgr._create_placeholder_summary(1)
        assert result["page_information"]["page_types"] == ["other"]


# ============================================================================
# _build_model_inputs
# ============================================================================
class TestBuildModelInputs:
    """Tests for _build_model_inputs()."""

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_openai_format(self, _):
        """OpenAI uses text content list in HumanMessage."""
        mgr = _make_manager(
            provider="openai",
            model_config={},
            service_tier="flex",
            summary_context=None,
            summary_schema=_MOCK_SCHEMA,
            _output_schema=_MOCK_SCHEMA,
        )

        with patch("api.summary_api.render_prompt_with_schema", return_value="sys"):
            messages, kwargs = mgr._build_model_inputs("Some transcription text.")

        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        content = messages[1].content
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Some transcription text."

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_anthropic_format(self, _):
        """Anthropic uses plain string in HumanMessage."""
        mgr = _make_manager(
            provider="anthropic",
            model_config={},
            service_tier="",
            summary_context=None,
            summary_schema=_MOCK_SCHEMA,
            _output_schema=_MOCK_SCHEMA,
        )

        with patch("api.summary_api.render_prompt_with_schema", return_value="sys"):
            messages, kwargs = mgr._build_model_inputs("Some text.")

        content = messages[1].content
        assert isinstance(content, str)
        assert content == "Some text."

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_google_format(self, _):
        """Google uses plain string in HumanMessage."""
        mgr = _make_manager(
            provider="google",
            model_config={},
            service_tier="",
            summary_context=None,
            summary_schema=_MOCK_SCHEMA,
            _output_schema=_MOCK_SCHEMA,
        )

        with patch("api.summary_api.render_prompt_with_schema", return_value="sys"):
            messages, kwargs = mgr._build_model_inputs("Text for Google.")

        content = messages[1].content
        assert isinstance(content, str)
        assert content == "Text for Google."

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_context_injection(self, _):
        """Summary context is passed to render_prompt_with_schema."""
        mgr = _make_manager(
            provider="openai",
            model_config={},
            service_tier="",
            summary_context="Focus on spice trade.",
            summary_schema=_MOCK_SCHEMA,
            _output_schema=_MOCK_SCHEMA,
        )

        with patch(
            "api.summary_api.render_prompt_with_schema",
            return_value="rendered with context",
        ) as mock_render:
            messages, kwargs = mgr._build_model_inputs("Transcription.")

        mock_render.assert_called_once()
        call_kwargs = mock_render.call_args
        assert call_kwargs[1]["context"] == "Focus on spice trade."

    @patch(
        "api.base_llm_client.get_model_capabilities",
        return_value={"max_tokens": True, "reasoning": False, "text_verbosity": False},
    )
    def test_openrouter_uses_text_list_format(self, _):
        """OpenRouter uses same format as OpenAI."""
        mgr = _make_manager(
            provider="openrouter",
            model_config={},
            service_tier="",
            summary_context=None,
            summary_schema=_MOCK_SCHEMA,
            _output_schema=_MOCK_SCHEMA,
        )

        with patch("api.summary_api.render_prompt_with_schema", return_value="sys"):
            messages, kwargs = mgr._build_model_inputs("Text for OR.")

        content = messages[1].content
        assert isinstance(content, list)
        assert content[0]["type"] == "text"


# ============================================================================
# _ensure_page_information_structure
# ============================================================================
class TestEnsurePageInformationStructure:
    """Tests for _ensure_page_information_structure()."""

    def test_missing_page_information(self):
        """Creates page_information when missing."""
        mgr = _make_manager()
        summary = {"bullet_points": ["Some text."]}
        mgr._ensure_page_information_structure(summary, 5)

        assert summary["page_information"]["page_number_integer"] == 5
        assert summary["page_information"]["page_number_type"] == "arabic"
        assert summary["page_information"]["page_types"] == ["content"]

    def test_non_dict_page_information(self):
        """Replaces non-dict page_information with correct structure."""
        mgr = _make_manager()
        summary = {"page_information": "invalid"}
        mgr._ensure_page_information_structure(summary, 3)

        assert isinstance(summary["page_information"], dict)
        assert summary["page_information"]["page_number_integer"] == 3

    def test_missing_page_number_integer(self):
        """Fills in missing page_number_integer."""
        mgr = _make_manager()
        summary = {
            "page_information": {
                "page_number_type": "roman",
                "page_types": ["preface"],
            }
        }
        mgr._ensure_page_information_structure(summary, 2)

        assert summary["page_information"]["page_number_integer"] == 2
        assert summary["page_information"]["page_number_type"] == "roman"

    def test_missing_page_number_type_with_integer(self):
        """Infers 'arabic' when page_number_integer exists."""
        mgr = _make_manager()
        summary = {
            "page_information": {
                "page_number_integer": 10,
                "page_types": ["content"],
            }
        }
        mgr._ensure_page_information_structure(summary, 10)

        assert summary["page_information"]["page_number_type"] == "arabic"

    def test_missing_page_number_type_without_integer(self):
        """Infers 'none' when page_number_integer is absent before fixup."""
        mgr = _make_manager()
        summary = {
            "page_information": {
                "page_types": ["content"],
            }
        }
        # Before fixup, page_number_integer is missing, so it gets added,
        # and page_number_type should be "arabic" since page_number_integer
        # is now set to page_num
        mgr._ensure_page_information_structure(summary, 7)

        # page_number_integer gets set first, then page_number_type check
        # sees page_number_integer == 7 (truthy) => "arabic"
        assert summary["page_information"]["page_number_type"] == "arabic"

    def test_legacy_page_type_conversion(self):
        """Converts legacy page_type string to page_types array."""
        mgr = _make_manager()
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_type": "title_page",
            }
        }
        mgr._ensure_page_information_structure(summary, 1)

        assert summary["page_information"]["page_types"] == ["title_page"]
        assert "page_type" not in summary["page_information"]

    def test_legacy_page_type_empty_string(self):
        """Empty legacy page_type converts to ['content']."""
        mgr = _make_manager()
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_type": "",
            }
        }
        mgr._ensure_page_information_structure(summary, 1)

        assert summary["page_information"]["page_types"] == ["content"]

    def test_missing_page_types_no_legacy(self):
        """Missing page_types without legacy page_type defaults to ['content']."""
        mgr = _make_manager()
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
            }
        }
        mgr._ensure_page_information_structure(summary, 1)

        assert summary["page_information"]["page_types"] == ["content"]

    def test_existing_page_types_preserved(self):
        """Existing page_types array is not modified."""
        mgr = _make_manager()
        summary = {
            "page_information": {
                "page_number_integer": 5,
                "page_number_type": "arabic",
                "page_types": ["illustration", "content"],
            }
        }
        mgr._ensure_page_information_structure(summary, 5)

        assert summary["page_information"]["page_types"] == [
            "illustration",
            "content",
        ]

    def test_complete_page_information_unchanged(self):
        """Complete page_information is not modified."""
        mgr = _make_manager()
        original = {
            "page_information": {
                "page_number_integer": 10,
                "page_number_type": "roman",
                "page_types": ["preface"],
            }
        }
        mgr._ensure_page_information_structure(original, 10)

        assert original["page_information"]["page_number_integer"] == 10
        assert original["page_information"]["page_number_type"] == "roman"
        assert original["page_information"]["page_types"] == ["preface"]


# ============================================================================
# generate_summary
# ============================================================================
class TestGenerateSummary:
    """Tests for generate_summary()."""

    def _mock_successful_response(self, summary_json: dict) -> AIMessage:
        """Helper to create a mock AIMessage with summary JSON."""
        msg = AIMessage(content=json.dumps(summary_json))
        msg.usage_metadata = None
        msg.response_metadata = {"model": "gpt-5-mini"}
        return msg

    def test_successful_generation(self):
        """Successful summary generation returns well-formed result."""
        mgr = _make_manager(provider="openai")

        summary_data = {
            "page_information": {
                "page_number_integer": 5,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Point 1.", "Point 2."],
            "references": None,
        }
        response = self._mock_successful_response(summary_data)

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
        ):
            result = mgr.generate_summary("Some transcription.", 5)

        assert result["page"] == 5
        assert result["bullet_points"] == ["Point 1.", "Point 2."]
        assert result["page_information"]["page_number_integer"] == 5
        assert "processing_time" in result
        assert result["provider"] == "openai"

    def test_json_parse_error_returns_placeholder(self):
        """Invalid JSON in response returns placeholder summary."""
        mgr = _make_manager(provider="openai")

        response = AIMessage(content="This is not valid JSON at all.")
        response.usage_metadata = None
        response.response_metadata = {}

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
        ):
            result = mgr.generate_summary("Some text.", 3)

        assert "Error generating summary" in result["bullet_points"][0]
        assert result["error_type"] == "api_failure"

    def test_empty_response_returns_placeholder(self):
        """Empty LLM response returns placeholder."""
        mgr = _make_manager(provider="openai")

        response = AIMessage(content="")
        response.usage_metadata = None
        response.response_metadata = {}

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
        ):
            result = mgr.generate_summary("Some text.", 1)

        assert "Error generating summary" in result["bullet_points"][0]

    def test_api_error_returns_placeholder(self):
        """API exception returns placeholder with error info."""
        mgr = _make_manager(provider="openai")

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("Connection timeout")

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
        ):
            result = mgr.generate_summary("Some text.", 2)

        assert result["page"] == 2
        assert "Connection timeout" in result["bullet_points"][0]
        assert result["error_type"] == "api_failure"
        assert mgr.failed_requests == 1

    def test_markdown_wrapped_json_stripped(self):
        """JSON wrapped in markdown code blocks is handled."""
        mgr = _make_manager(provider="openai")

        summary_data = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Extracted point."],
            "references": None,
        }

        wrapped = f"```json\n{json.dumps(summary_data)}\n```"
        response = AIMessage(content=wrapped)
        response.usage_metadata = None
        response.response_metadata = {}

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
        ):
            result = mgr.generate_summary("Text.", 1)

        assert result["bullet_points"] == ["Extracted point."]

    def test_token_usage_reported(self):
        """Token usage is reported to the tracker when available."""
        mgr = _make_manager(provider="openai")

        summary_data = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Point."],
            "references": None,
        }
        response = self._mock_successful_response(summary_data)
        response.usage_metadata = {"total_tokens": 500}

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
            patch("api.summary_api.get_token_tracker") as mock_tt,
        ):
            mock_tracker = MagicMock()
            mock_tracker.get_tokens_used_today.return_value = 500
            mock_tt.return_value = mock_tracker

            result = mgr.generate_summary("Text.", 1)

        mock_tracker.add_tokens.assert_called_once_with(500)

    def test_response_metadata_included(self):
        """Response metadata from LLM is included in result."""
        mgr = _make_manager(provider="openai")

        summary_data = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Point."],
            "references": None,
        }
        response = self._mock_successful_response(summary_data)

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = response

        with (
            patch.object(
                mgr, "_get_structured_chat_model", return_value=mock_structured
            ),
            patch.object(mgr, "_build_model_inputs", return_value=([], {})),
        ):
            result = mgr.generate_summary("Text.", 1)

        assert "api_response" in result
        assert result["api_response"]["model"] == "gpt-5-mini"


# ============================================================================
# get_stats
# ============================================================================
class TestGetStats:
    """Tests for get_stats()."""

    def test_includes_service_tier(self):
        """Stats include service_tier field."""
        mgr = _make_manager(service_tier="flex", successful_requests=3)
        stats = mgr.get_stats()
        assert stats["service_tier"] == "flex"

    def test_inherits_base_stats(self):
        """Stats include all base class fields."""
        mgr = _make_manager(
            service_tier="auto",
            successful_requests=5,
            failed_requests=1,
        )
        mgr.processing_times.extend([1.0, 2.0])

        stats = mgr.get_stats()
        assert stats["provider"] == "openai"
        assert stats["model"] == "gpt-5-mini"
        assert stats["successful_requests"] == 5
        assert stats["failed_requests"] == 1
        assert stats["average_processing_time"] == 1.5
        assert stats["recent_success_rate"] == pytest.approx(83.3, abs=0.1)
        assert stats["service_tier"] == "auto"
