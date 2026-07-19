"""Regression tests for the round-4 LLM hardening fixes.

One focused test class per fix:

1.  ``TranscriptionManager.transcribe_payload`` — when schema-validation retries
    are exhausted, a persistently non-JSON response returns an error-marked dict
    (``schema_validation_exhausted``) so resume repairs it, while a genuinely
    recoverable JSON response (a string ``transcription``) is still salvaged as
    an unmarked success.
2.  Empty HTTP-200 responses are retried in-run (transcription and summary):
    one empty then a good response succeeds without failing the page; a
    persistently empty response fails with the same shape as before.
3.  ``_should_retry_for_schema_flag`` — string config values are coerced instead
    of raising.
4.  ``DailyTokenTracker._load_state`` — string/negative/float/bool
    ``tokens_used`` values are sanitized; a valid same-day count is preserved.
5.  ``_extract_tokens_from_exception`` — float token counts in an exception body
    are counted.
6.  ``_report_token_usage`` — a response whose only usable counts are additive
    cache tokens still commits them.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from llm.base import LLMClientBase
from llm.summary import SummaryManager
from llm.token_tracker import DailyTokenTracker
from llm.transcription import TranscriptionManager


# ============================================================================
# Shared builders
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


def _make_sm(**overrides: Any) -> SummaryManager:
    """Bare SummaryManager bypassing __init__."""
    sm = SummaryManager.__new__(SummaryManager)
    defaults: dict[str, Any] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "custom_capabilities": None,
        "transcription_was_plain_text": False,
        "_stats_lock": threading.Lock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "rate_limiter": None,
        "model_config": {},
        "summary_context": None,
        "_output_schema": None,
        "schema_retry_config": {},
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(sm, attr, val)
    return sm


def _payload(name: str = "page_001.png") -> Any:
    payload = MagicMock()
    payload.image_name = name
    payload.sequence_number = 1
    payload.base64 = "AAAA"
    return payload


def _wire_tm(tm: TranscriptionManager, extract: Any) -> None:
    """Stub the API boundary; *extract* is the return_value or side_effect."""
    tm._build_model_inputs = MagicMock(return_value=([], {}))  # type: ignore[method-assign]
    tm._get_structured_chat_model = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    tm._invoke_with_retry = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    tm._report_token_usage = MagicMock()  # type: ignore[method-assign]
    mock = MagicMock()
    if isinstance(extract, list):
        mock.side_effect = extract
    else:
        mock.return_value = extract
    tm._extract_output_text = mock  # type: ignore[method-assign]


def _wire_sm(sm: SummaryManager, extract: Any) -> None:
    sm._build_model_inputs = MagicMock(return_value=([], {}))  # type: ignore[method-assign]
    sm._get_structured_chat_model = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    sm._invoke_with_retry = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    sm._report_token_usage = MagicMock()  # type: ignore[method-assign]
    mock = MagicMock()
    if isinstance(extract, list):
        mock.side_effect = extract
    else:
        mock.return_value = extract
    sm._extract_output_text = mock  # type: ignore[method-assign]


_GOOD_TRANSCRIPTION = json.dumps(
    {
        "image_analysis": "clear",
        "transcription": "hello world",
        "no_transcribable_text": False,
        "transcription_not_possible": False,
    }
)

_GOOD_SUMMARY = json.dumps(
    {
        "page_information": {
            "page_number_integer": 1,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": ["A point."],
        "references": None,
    }
)


# ============================================================================
# Fix 1: validation-exhaustion marks the page (unless genuinely recoverable)
# ============================================================================
class TestValidationExhaustionMarking:
    """A persistently invalid response is marked; a recoverable one salvaged."""

    _CFG = {
        "validation_failure": {
            "enabled": True,
            "max_attempts": 1,
            "backoff_base": 0.0,
            "backoff_multiplier": 1.0,
        }
    }

    def test_persistent_non_json_returns_error_marked(self) -> None:
        tm = _make_tm(schema_retry_config=self._CFG)
        _wire_tm(tm, "not valid json at all")

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload(), max_schema_retries=3)

        assert result["error_type"] == "schema_validation_exhausted"
        assert "error" in result
        assert result["schema_retries"]["validation_failure"] >= 1

    def test_recoverable_json_after_exhaustion_is_salvaged(self) -> None:
        # Valid JSON dict with a string transcription but missing the other
        # required schema keys -> fails validation, yet is genuinely usable.
        tm = _make_tm(schema_retry_config=self._CFG)
        _wire_tm(tm, json.dumps({"transcription": "recovered text"}))

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload(), max_schema_retries=3)

        assert result["transcription"] == "recovered text"
        assert "error_type" not in result
        assert "error" not in result
        assert result["schema_retries"]["validation_failure"] >= 1

    def test_recoverable_flag_after_exhaustion_is_salvaged(self) -> None:
        # An explicit no_transcribable_text flag counts as recoverable even
        # though the object is missing other required keys.
        tm = _make_tm(schema_retry_config=self._CFG)
        _wire_tm(tm, json.dumps({"no_transcribable_text": True}))

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload("p.png"), max_schema_retries=3)

        assert "error_type" not in result
        assert "no transcribable text" in result["transcription"]

    def test_recoverability_helper(self) -> None:
        tm = _make_tm()
        assert tm._transcription_is_recoverable('{"transcription": "x"}') is True
        assert tm._transcription_is_recoverable('{"no_transcribable_text": true}')
        assert tm._transcription_is_recoverable('{"transcription_not_possible": true}')
        assert tm._transcription_is_recoverable("not json") is False
        assert tm._transcription_is_recoverable('{"other": 1}') is False
        assert tm._transcription_is_recoverable('{"transcription": 5}') is False


# ============================================================================
# Fix 2: empty HTTP-200 responses are retried in-run
# ============================================================================
class TestEmptyResponseInRunRetry:
    """One empty then good succeeds; persistent empty fails as before."""

    def test_transcription_empty_then_good_succeeds(self) -> None:
        tm = _make_tm()
        _wire_tm(tm, ["", _GOOD_TRANSCRIPTION])

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload())

        assert result["transcription"] == "hello world"
        assert "error_type" not in result
        assert tm.successful_requests == 1
        assert tm.failed_requests == 0

    def test_transcription_persistent_empty_fails_as_before(self) -> None:
        tm = _make_tm()
        _wire_tm(tm, "")

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload())

        assert result["error_type"] == "api_failure"
        assert tm.successful_requests == 0
        assert tm.failed_requests == 1

    def test_summary_empty_then_good_succeeds(self) -> None:
        sm = _make_sm()
        _wire_sm(sm, ["", _GOOD_SUMMARY])

        with patch("llm.summary.time.sleep"):
            result = sm.generate_summary("text", 1)

        assert result["bullet_points"] == ["A point."]
        assert "error_type" not in result
        assert sm.successful_requests == 1
        assert sm.failed_requests == 0

    def test_summary_persistent_empty_fails_as_before(self) -> None:
        sm = _make_sm()
        _wire_sm(sm, "")

        with patch("llm.summary.time.sleep"):
            result = sm.generate_summary("text", 1)

        assert result["error_type"] == "api_failure"
        assert "Error generating summary" in result["bullet_points"][0]
        assert sm.successful_requests == 0
        assert sm.failed_requests == 1


# ============================================================================
# Fix 3: string schema-retry config values are coerced, not raised
# ============================================================================
class TestSchemaFlagConfigGuards:
    """Malformed flag-config values must not raise inside the retry check."""

    def test_string_max_attempts_does_not_raise(self) -> None:
        client = _make_client(
            schema_retry_config={
                "validation_failure": {
                    "enabled": True,
                    "max_attempts": "three",
                    "backoff_base": "x",
                    "backoff_multiplier": "y",
                }
            }
        )
        # _cfg_int("three", 0) == 0 -> current_attempt 0 >= 0 -> no retry, no raise.
        should_retry, backoff, max_attempts = client._should_retry_for_schema_flag(
            "validation_failure", True, 0
        )
        assert should_retry is False
        assert max_attempts == 0

    def test_string_backoff_values_do_not_raise(self) -> None:
        client = _make_client(
            schema_retry_config={
                "validation_failure": {
                    "enabled": True,
                    "max_attempts": "5",  # numeric string -> 5 via _cfg_int
                    "backoff_base": "nope",  # -> default 2.0
                    "backoff_multiplier": "nope",  # -> default 1.5
                }
            }
        )
        with patch("llm.base.random.uniform", return_value=0.5):
            should_retry, backoff, max_attempts = client._should_retry_for_schema_flag(
                "validation_failure", True, 0
            )
        assert should_retry is True
        assert max_attempts == 5
        # 2.0 * 1.5**0 + 0.5 == 2.5
        assert backoff == 2.5


# ============================================================================
# Fix 4: saved-token sanitization in _load_state
# ============================================================================
class TestLoadStateSanitization:
    """A malformed ``tokens_used`` is sanitized; a valid count is preserved."""

    def _make_tracker(self, tmp_path: Any, tokens_used: Any) -> DailyTokenTracker:
        tr = DailyTokenTracker.__new__(DailyTokenTracker)
        tr.state_file = tmp_path / "token_state.json"
        tr._current_date = ""
        tr._tokens_used_today = 0
        tr._own_buckets = {}
        tr._get_current_date_str = lambda: "2026-07-19"  # type: ignore[method-assign]
        tr.state_file.write_text(
            json.dumps({"date": "2026-07-19", "tokens_used": tokens_used}),
            encoding="utf-8",
        )
        return tr

    def test_valid_int_preserved(self, tmp_path: Any) -> None:
        tr = self._make_tracker(tmp_path, 500)
        tr._load_state()
        assert tr._tokens_used_today == 500
        assert sum(tr._own_buckets.values()) == 500

    def test_float_coerced(self, tmp_path: Any) -> None:
        tr = self._make_tracker(tmp_path, 500.0)
        tr._load_state()
        assert tr._tokens_used_today == 500

    def test_negative_clamped_to_zero(self, tmp_path: Any) -> None:
        tr = self._make_tracker(tmp_path, -100)
        tr._load_state()
        assert tr._tokens_used_today == 0

    def test_bool_treated_as_zero(self, tmp_path: Any) -> None:
        tr = self._make_tracker(tmp_path, True)
        tr._load_state()
        assert tr._tokens_used_today == 0

    def test_string_treated_as_zero_without_crash(self, tmp_path: Any) -> None:
        tr = self._make_tracker(tmp_path, "500")
        tr._load_state()
        # Same-day branch is taken and does not silently reset via an exception:
        # the count is a clean 0 and the date is retained.
        assert tr._tokens_used_today == 0
        assert tr._current_date == "2026-07-19"


# ============================================================================
# Fix 5: float token counts recovered from an exception body
# ============================================================================
class TestExceptionTokenFloatCoercion:
    """Float usage counts in a failed request's body are committed."""

    def test_float_total_tokens_recovered(self) -> None:
        client = _make_client()
        exc = RuntimeError("boom")
        exc.body = {"usage": {"total_tokens": 123.0}}  # type: ignore[attr-defined]

        with patch("llm.base.get_token_tracker") as mock_tt:
            tracker = MagicMock()
            tracker.get_tokens_used_today.return_value = 123
            mock_tt.return_value = tracker
            client._extract_tokens_from_exception(exc, "ctx")

        tracker.add_tokens.assert_called_once_with(
            123, provider="openai", key_env=None, model="gpt-5-mini"
        )

    def test_float_input_output_recovered(self) -> None:
        client = _make_client()
        exc = RuntimeError("boom")
        exc.body = {  # type: ignore[attr-defined]
            "usage": {"input_tokens": 100.0, "output_tokens": 50.0}
        }

        with patch("llm.base.get_token_tracker") as mock_tt:
            tracker = MagicMock()
            tracker.get_tokens_used_today.return_value = 150
            mock_tt.return_value = tracker
            client._extract_tokens_from_exception(exc, "ctx")

        tracker.add_tokens.assert_called_once_with(
            150, provider="openai", key_env=None, model="gpt-5-mini"
        )


# ============================================================================
# Fix 6: cache-only usage is committed
# ============================================================================
class TestCacheOnlyTokenCommit:
    """A response with 0 total but additive cache tokens still commits them."""

    def test_cache_only_committed(self) -> None:
        client = _make_client()
        response = SimpleNamespace(
            usage_metadata={
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 500,
            }
        )

        with patch("llm.base.get_token_tracker") as mock_tt:
            tracker = MagicMock()
            tracker.get_tokens_used_today.return_value = 500
            mock_tt.return_value = tracker
            client._report_token_usage(response, "ctx")

        tracker.add_tokens.assert_called_once_with(
            500, provider="openai", key_env=None, model="gpt-5-mini"
        )

    def test_truly_empty_commits_nothing(self) -> None:
        client = _make_client()
        response = SimpleNamespace(
            usage_metadata={"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        )

        with patch("llm.base.get_token_tracker") as mock_tt:
            tracker = MagicMock()
            mock_tt.return_value = tracker
            client._report_token_usage(response, "ctx")

        tracker.add_tokens.assert_not_called()
