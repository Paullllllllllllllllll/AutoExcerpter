"""Regression tests for the round-3 LLM/config hardening fixes.

One focused test class per fix:

1.  ``_classify_error`` — google.genai APIError (int ``code``, string ``status``)
    is classified retryable (429 -> rate_limit, 500 -> server_error), plus the
    quota/internal-error message fallbacks.
2.  ``TranscriptionManager.transcribe_payload`` — an empty HTTP-200 response is
    counted as a failure (not success), and the schema-exhausted fallback dict
    carries ``error``/``error_type`` keys.
3.  ``_extract_from_structured_output_wrapper`` — tool-call args and pydantic
    ``parsed`` objects are serialized rather than dropped.
4.  Config conversions — malformed values fall back to documented defaults
    instead of raising.
5.  ``DailyTokenTracker.flush`` — a degraded shared ledger persists own usage to
    the private state file at exit.
6.  ``strip_markdown_code_block`` — any fence label (not just ``json``) is
    stripped.
7.  ``_parse_transcription_from_text`` — fence-stripped text is returned on the
    non-JSON, salvage-failure, and plain-text paths.
8.  ``_report_token_usage`` — float token counts are accepted.
9.  ``detect_capabilities`` — ``gpt-4.5-preview`` resolves vision-capable.
10. Summary schema/prompt — comprehensiveness wording, valid JSON, intact
    placeholders.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import llm.base as base_mod
from config.loader import PROMPTS_DIR, SCHEMAS_DIR
from llm.base import LLMClientBase, _cfg_float, _cfg_int, _coerce_token_count
from llm.capabilities import detect_capabilities
from llm.prompts import render_prompt_with_schema, strip_markdown_code_block
from llm.token_tracker import DailyTokenTracker
from llm.transcription import TranscriptionManager


# ============================================================================
# Shared helpers
# ============================================================================
def _make_client(**overrides: Any) -> LLMClientBase:
    """Bare LLMClientBase bypassing __init__ for isolated unit tests."""
    client = LLMClientBase.__new__(LLMClientBase)
    defaults: dict[str, Any] = {
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
        "service_tier": "auto",
        "schema_retry_config": {},
        "_output_schema": None,
        "key_env": None,
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(client, attr, val)
    return client


def _make_tm(**overrides: Any) -> TranscriptionManager:
    """Bare TranscriptionManager bypassing __init__."""
    tm = TranscriptionManager.__new__(TranscriptionManager)
    defaults: dict[str, Any] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "custom_capabilities": None,
        "_stats_lock": threading.Lock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "rate_limiter": None,
        "model_config": {},
        "system_prompt": "sys",
        "_output_schema": None,
        "schema_retry_config": {},
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(tm, attr, val)
    return tm


def _payload(name: str = "page_001.png") -> Any:
    payload = MagicMock()
    payload.image_name = name
    payload.sequence_number = 1
    payload.base64 = "AAAA"
    return payload


# ============================================================================
# Fix 1: _classify_error handles google.genai APIError
# ============================================================================
class TestClassifyErrorGoogle:
    """Google APIError int ``code`` must surface; string ``status`` must not."""

    def test_google_429_is_retryable_rate_limit(self) -> None:
        from google.genai.errors import APIError

        exc = APIError(
            429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}
        )
        assert LLMClientBase._classify_error(exc) == (True, "rate_limit")

    def test_google_500_is_retryable_server_error(self) -> None:
        from google.genai.errors import APIError

        exc = APIError(500, {"error": {"message": "boom", "status": "INTERNAL"}})
        assert LLMClientBase._classify_error(exc) == (True, "server_error")

    def test_string_status_ignored_as_http_code(self) -> None:
        """A bare string ``status`` must not be read as an HTTP status code.

        Without a numeric code and with a benign message, the classifier falls
        through to a terminal result rather than crashing on the string.
        """
        exc = Exception("something benign")
        exc.status = "RESOURCE_EXHAUSTED"  # type: ignore[attr-defined]
        assert LLMClientBase._classify_error(exc) == (False, "other")

    def test_quota_message_fallback(self) -> None:
        assert LLMClientBase._classify_error(
            Exception("Quota exceeded for the day")
        ) == (True, "rate_limit")

    def test_internal_error_message_fallback(self) -> None:
        assert LLMClientBase._classify_error(
            Exception("Internal error encountered")
        ) == (True, "server_error")

    def test_plain_error_still_terminal(self) -> None:
        assert LLMClientBase._classify_error(ValueError("bad schema")) == (
            False,
            "other",
        )


# ============================================================================
# Fix 2: empty response and schema-exhausted fallback
# ============================================================================
class TestTranscriptionEmptyAndFallback:
    """Empty content is a failure; exhausted fallback carries error keys."""

    def _wire(self, tm: TranscriptionManager, raw_text: str) -> None:
        tm._build_model_inputs = MagicMock(return_value=([], {}))  # type: ignore[method-assign]
        tm._get_structured_chat_model = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
        tm._invoke_with_retry = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
        tm._report_token_usage = MagicMock()  # type: ignore[method-assign]
        tm._extract_output_text = MagicMock(return_value=raw_text)  # type: ignore[method-assign]

    def test_empty_response_counts_as_failure(self) -> None:
        tm = _make_tm()
        self._wire(tm, raw_text="")

        result = tm.transcribe_payload(_payload(), max_schema_retries=0)

        assert result["error_type"] == "api_failure"
        assert tm.successful_requests == 0
        assert tm.failed_requests == 1

    def test_schema_exhausted_fallback_has_error_keys(self) -> None:
        tm = _make_tm(
            schema_retry_config={
                "validation_failure": {
                    "enabled": True,
                    "max_attempts": 5,
                    "backoff_base": 0.0,
                    "backoff_multiplier": 1.0,
                }
            }
        )
        # Non-empty but never valid JSON -> validation retries exhaust.
        self._wire(tm, raw_text="not valid json")

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload(), max_schema_retries=1)

        assert result["error_type"] == "schema_validation_exhausted"
        assert "error" in result
        assert result["schema_retries"]["validation_failure"] >= 1


# ============================================================================
# Fix 3: structured-output wrapper extraction
# ============================================================================
class TestStructuredOutputWrapper:
    """tool_calls args and pydantic parsed objects are serialized, not dropped."""

    def test_empty_content_with_tool_calls(self) -> None:
        raw = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "article_page_summary",
                    "args": {"bullet_points": ["a", "b"]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        data = {"raw": raw, "parsed": None, "parsing_error": "boom"}
        result = LLMClientBase._extract_output_text(data)
        assert json.loads(result) == {"bullet_points": ["a", "b"]}

    def test_pydantic_parsed_is_model_dumped(self) -> None:
        from pydantic import BaseModel

        class Answer(BaseModel):
            value: int

        data = {"raw": AIMessage(content=""), "parsed": Answer(value=7)}
        result = LLMClientBase._extract_output_text(data)
        assert json.loads(result) == {"value": 7}

    def test_dict_parsed_still_preferred(self) -> None:
        data = {"raw": AIMessage(content="ignored"), "parsed": {"k": 1}}
        result = LLMClientBase._extract_output_text(data)
        assert json.loads(result) == {"k": 1}


# ============================================================================
# Fix 4: guarded config conversions
# ============================================================================
class TestConfigConversionGuards:
    """Malformed values fall back to documented defaults instead of raising."""

    def test_cfg_int_malformed(self) -> None:
        assert _cfg_int("fast", 8) == 8
        assert _cfg_int(None, 8) == 8
        assert _cfg_int("12", 8) == 12

    def test_cfg_float_malformed(self) -> None:
        assert _cfg_float("slow", 1.5) == 1.5
        assert _cfg_float(None, 1.5) == 1.5
        assert _cfg_float("2.5", 1.5) == 2.5

    def test_module_constants_are_numeric(self) -> None:
        assert isinstance(base_mod.DEFAULT_MAX_RETRIES, int)
        assert isinstance(base_mod.BACKOFF_CAP_S, float)
        assert isinstance(base_mod.JITTER_MIN, float)
        assert isinstance(base_mod.JITTER_MAX, float)

    def test_calculate_backoff_tolerates_malformed_retry_config(self) -> None:
        dummy = cast(LLMClientBase, object())
        bad = {"backoff_base": "x", "backoff_multipliers": None}
        with patch.object(base_mod, "_RETRY_CONFIG", bad):
            result = LLMClientBase._calculate_backoff(dummy, 3, "server_error")
        assert 0.0 <= result <= base_mod.BACKOFF_CAP_S

    def test_get_api_timeout_malformed_falls_back(self, monkeypatch) -> None:
        import config.accessors as accessors
        from config.constants import DEFAULT_OPENAI_TIMEOUT

        loader = MagicMock()
        loader.get_concurrency_config.return_value = {
            "api_requests": {"api_timeout": "slow"}
        }
        monkeypatch.setattr(accessors, "get_config_loader", lambda: loader)
        assert accessors.get_api_timeout() == DEFAULT_OPENAI_TIMEOUT

    def test_get_target_dpi_malformed_falls_back(self, monkeypatch) -> None:
        import config.accessors as accessors
        from config.constants import DEFAULT_TARGET_DPI

        loader = MagicMock()
        loader.get_image_processing_config.return_value = {
            "api_image_processing": {"target_dpi": "huge"}
        }
        monkeypatch.setattr(accessors, "get_config_loader", lambda: loader)
        assert accessors.get_target_dpi() == DEFAULT_TARGET_DPI


# ============================================================================
# Fix 5: degraded shared ledger persists own usage at flush
# ============================================================================
class TestFlushDegradedPersistence:
    """A ledger degraded through exit must persist own usage to the private
    file."""

    def _make_tracker(
        self, degraded: bool
    ) -> tuple[DailyTokenTracker, MagicMock, MagicMock]:
        tr = DailyTokenTracker.__new__(DailyTokenTracker)
        tr._lock = threading.Lock()
        tr._shared_enabled = True
        tr._ledger_sync_in_flight = False
        tr._ledger_degraded = degraded
        tr._pending_save = True
        sync = MagicMock()
        save = MagicMock()
        tr.sync_ledger_now = sync  # type: ignore[method-assign]
        tr._save_state = save  # type: ignore[method-assign]
        return tr, sync, save

    def test_degraded_ledger_writes_private_state(self) -> None:
        tr, sync, save = self._make_tracker(degraded=True)
        tr.flush()
        sync.assert_called_once()
        save.assert_called_once_with(force=True)

    def test_healthy_ledger_does_not_write_private_state(self) -> None:
        tr, sync, save = self._make_tracker(degraded=False)
        tr.flush()
        sync.assert_called_once()
        save.assert_not_called()


# ============================================================================
# Fix 6: strip any fence label
# ============================================================================
class TestStripFenceLabel:
    """A non-json info-string label must be dropped with the fence."""

    def test_markdown_label_stripped(self) -> None:
        assert strip_markdown_code_block("```markdown\nhello\n```") == "hello"

    def test_python_label_stripped(self) -> None:
        assert strip_markdown_code_block("```python\nprint(1)\n```") == "print(1)"

    def test_bare_fence_stripped(self) -> None:
        assert strip_markdown_code_block("```\nplain\n```") == "plain"

    def test_json_label_still_stripped(self) -> None:
        payload = '{"k": 1}'
        assert strip_markdown_code_block(f"```json\n{payload}\n```") == payload


# ============================================================================
# Fix 7: fence-stripped returns in transcription parsing
# ============================================================================
class TestTranscriptionFenceReturns:
    """Fence debris must never reach the returned transcription text."""

    def test_non_json_path_returns_stripped(self) -> None:
        tm = _make_tm()
        result = tm._parse_transcription_from_text("```\nplain text\n```", "p.png")
        assert "```" not in result
        assert result == "plain text"

    def test_salvage_failure_returns_stripped(self) -> None:
        tm = _make_tm()
        # Starts with "{" after stripping, but unparseable and unsalvageable.
        result = tm._parse_transcription_from_text("```json\n{bad json\n```", "p.png")
        assert "```" not in result
        assert result.startswith("{bad json")

    def test_plain_text_mode_strips_fence(self) -> None:
        caps = SimpleNamespace(use_plain_text_prompt=True, supports_vision=True)
        tm = _make_tm(custom_capabilities=caps)
        assert tm.is_plain_text_mode is True
        result = tm._parse_transcription_from_text("```\nhello world\n```", "p.png")
        assert result == "hello world"


# ============================================================================
# Fix 8: float token counts accepted
# ============================================================================
class TestFloatTokenCounts:
    """Float token counts are committed rather than silently dropped."""

    def test_coerce_helper(self) -> None:
        assert _coerce_token_count(123.0) == 123
        assert _coerce_token_count(122.4) == 122
        assert _coerce_token_count(50) == 50
        assert _coerce_token_count(True) is None
        assert _coerce_token_count("x") is None

    def test_float_total_tokens_reported(self) -> None:
        client = _make_client()
        response = MagicMock()
        response.usage_metadata = {"total_tokens": 123.0}

        with patch("llm.base.get_token_tracker") as mock_tt:
            tracker = MagicMock()
            tracker.get_tokens_used_today.return_value = 123
            mock_tt.return_value = tracker
            client._report_token_usage(response, "ctx")

        tracker.add_tokens.assert_called_once_with(
            123, provider="openai", key_env=None, model="gpt-5-mini"
        )

    def test_float_input_output_fallback(self) -> None:
        client = _make_client()
        response = MagicMock()
        response.usage_metadata = {"input_tokens": 100.0, "output_tokens": 50.0}

        with patch("llm.base.get_token_tracker") as mock_tt:
            tracker = MagicMock()
            tracker.get_tokens_used_today.return_value = 150
            mock_tt.return_value = tracker
            client._report_token_usage(response, "ctx")

        tracker.add_tokens.assert_called_once_with(
            150, provider="openai", key_env=None, model="gpt-5-mini"
        )


# ============================================================================
# Fix 9: gpt-4.5-preview resolves vision-capable
# ============================================================================
class TestGpt45Capabilities:
    """gpt-4.5-preview must not fall through to the vision-less gpt-4 entry."""

    def test_gpt_45_preview_supports_vision(self) -> None:
        caps = detect_capabilities("gpt-4.5-preview")
        assert caps.supports_vision is True
        assert caps.family == "gpt-4.5"

    def test_bare_gpt_4_still_vision_less(self) -> None:
        caps = detect_capabilities("gpt-4")
        assert caps.supports_vision is False


# ============================================================================
# Fix 10: summary comprehensiveness (schema + prompt)
# ============================================================================
class TestSummaryComprehensiveness:
    """Schema stays valid; wording demands comprehensive coverage; placeholders
    intact."""

    def _load_schema(self) -> dict[str, Any]:
        path = (SCHEMAS_DIR / "summary_schema.json").resolve()
        with open(path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))

    def test_schema_is_valid_json_and_comprehensive(self) -> None:
        schema = self._load_schema()
        desc = schema["schema"]["properties"]["bullet_points"]["description"]
        assert "2-5 concise" not in desc
        assert "3-10" in desc
        assert "LaTeX" in desc  # existing instruction preserved
        # Page-type applicability sentences preserved.
        assert "table_of_contents" in desc

    def test_system_prompt_not_brief_and_placeholders_intact(self) -> None:
        path = (PROMPTS_DIR / "summary_system_prompt.txt").resolve()
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "{{SCHEMA}}" in text
        assert "{{CONTEXT}}" in text
        assert "comprehensive" in text.lower()

    def test_plain_text_prompt_aligned(self) -> None:
        path = (PROMPTS_DIR / "summary_plain_text_prompt.txt").resolve()
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "{{SCHEMA}}" in text
        assert "comprehensive" in text.lower()

    def test_schema_still_renders_into_prompt(self) -> None:
        schema = self._load_schema()
        path = (PROMPTS_DIR / "summary_system_prompt.txt").resolve()
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        bare = schema.get("schema", schema)
        rendered = render_prompt_with_schema(raw, bare, context="X")
        assert "{{SCHEMA}}" not in rendered
        assert "{{CONTEXT}}" not in rendered
        assert "bullet_points" in rendered


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
