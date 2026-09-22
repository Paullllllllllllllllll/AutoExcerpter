"""Tests for the response status inspector and early stops on filtered pages.

Covers ``inspect_response`` for every response shape LangChain delivers, the
content-filter and refusal paths of ``TranscriptionManager.transcribe_payload``
and ``SummaryManager.generate_summary`` (one API call, no retries, token usage
still reported), the unchanged schema-retry path for ``max_output_tokens``,
and resume treatment of a content-filtered page.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_openai.chat_models.base import OpenAIRefusalError

from config import app as app_config
from llm.base import ResponseOutcome, get_response_metadata, inspect_response
from llm.summary import SummaryManager
from llm.transcription import TranscriptionManager
from pipeline.log import append_to_log, initialize_log_file
from pipeline.resume import _completed_pages_from_entries, load_completed_pages
from pipeline.transcriber import ItemTranscriber


# ============================================================================
# Builders
# ============================================================================
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


def _wire(manager: Any, response: Any) -> tuple[MagicMock, MagicMock]:
    """Stub the API boundary so every call returns *response*.

    Returns the ``_invoke_with_retry`` and ``_report_token_usage`` mocks.
    """
    invoke = MagicMock(return_value=response)
    report = MagicMock()
    manager._build_model_inputs = MagicMock(return_value=([], {}))
    manager._get_structured_chat_model = MagicMock(return_value=MagicMock())
    manager._invoke_with_retry = invoke
    manager._report_token_usage = report
    return invoke, report


def _filtered_message() -> AIMessage:
    """A Responses API message stopped by the content filter."""
    return AIMessage(
        content=[],
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
        },
    )


def _truncated_message(text: str) -> AIMessage:
    """A Responses API message cut off at max_output_tokens."""
    return AIMessage(
        content=[{"type": "text", "text": text}],
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    )


def _refusal_message(text: str = "I can't help with that.") -> AIMessage:
    """A Responses API message whose only content block is a refusal."""
    return AIMessage(content=[{"type": "refusal", "refusal": text}])


_GOOD_SUMMARY = json.dumps(
    {
        "page_information": {
            "page_number_integer": 12,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": ["A point."],
        "references": None,
    }
)


# ============================================================================
# inspect_response per shape
# ============================================================================
class TestInspectResponse:
    def test_responses_api_content_filter(self) -> None:
        outcome = inspect_response(_filtered_message())
        assert outcome.kind == "content_filter"
        assert outcome.reason is not None
        assert "content_filter" in outcome.reason
        assert outcome.stops_page

    def test_responses_api_max_output_tokens(self) -> None:
        outcome = inspect_response(_truncated_message("partial"))
        assert outcome.kind == "max_output_tokens"
        assert not outcome.stops_page

    def test_include_raw_dict_unwraps_raw_message(self) -> None:
        wrapped = {"raw": _filtered_message(), "parsed": None, "parsing_error": None}
        assert inspect_response(wrapped).kind == "content_filter"

    def test_include_raw_dict_with_refusal_parsing_error(self) -> None:
        wrapped = {
            "raw": AIMessage(content=""),
            "parsed": None,
            "parsing_error": OpenAIRefusalError("Refused for policy reasons."),
        }
        outcome = inspect_response(wrapped)
        assert outcome == ResponseOutcome("refusal", "Refused for policy reasons.")

    def test_additional_kwargs_refusal(self) -> None:
        message = AIMessage(content="", additional_kwargs={"refusal": "No can do."})
        outcome = inspect_response(message)
        assert outcome == ResponseOutcome("refusal", "No can do.")

    def test_refusal_content_block(self) -> None:
        outcome = inspect_response(_refusal_message("Not allowed."))
        assert outcome == ResponseOutcome("refusal", "Not allowed.")

    def test_chat_completions_finish_reason_content_filter(self) -> None:
        message = AIMessage(
            content="", response_metadata={"finish_reason": "content_filter"}
        )
        outcome = inspect_response(message)
        assert outcome.kind == "content_filter"
        assert outcome.reason == "finish_reason=content_filter"

    def test_chat_completions_finish_reason_length_is_ok(self) -> None:
        message = AIMessage(
            content="text", response_metadata={"finish_reason": "length"}
        )
        assert inspect_response(message).kind == "ok"

    def test_completed_response_is_ok(self) -> None:
        message = AIMessage(
            content=[{"type": "text", "text": "fine"}],
            response_metadata={"status": "completed"},
        )
        outcome = inspect_response(message)
        assert outcome == ResponseOutcome("ok")
        assert not outcome.stops_page

    def test_anthropic_stop_reason_refusal(self) -> None:
        message = AIMessage(content="", response_metadata={"stop_reason": "refusal"})
        assert inspect_response(message) == ResponseOutcome(
            "refusal", "stop_reason=refusal"
        )
        wrapped = {"raw": message, "parsed": None, "parsing_error": None}
        assert inspect_response(wrapped).kind == "refusal"

    @pytest.mark.parametrize("response", [MagicMock(), None, "text", {"x": 1}])
    def test_foreign_shapes_are_ok(self, response: Any) -> None:
        assert inspect_response(response).kind == "ok"

    def test_get_response_metadata_unwraps_include_raw(self) -> None:
        raw = AIMessage(content="", response_metadata={"model": "gpt-5-mini"})
        assert get_response_metadata({"raw": raw, "parsed": {}}) == {
            "model": "gpt-5-mini"
        }
        assert get_response_metadata(MagicMock()) == {}


# ============================================================================
# Transcription early stop
# ============================================================================
class TestTranscriptionEarlyStop:
    def test_content_filter_fails_page_after_one_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tm = _make_tm()
        invoke, report = _wire(tm, _filtered_message())

        with (
            patch("llm.transcription.time.sleep") as sleep,
            caplog.at_level(logging.WARNING),
        ):
            result = tm.transcribe_payload(_payload(), max_schema_retries=3)

        assert invoke.call_count == 1
        report.assert_called_once()
        sleep.assert_not_called()
        assert result["error_type"] == "content_filter"
        assert "content_filter" in result["error"]
        assert "schema_retries" not in result
        assert tm.failed_requests == 1
        assert any(
            "content_filter" in rec.getMessage() and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_refusal_fails_page_after_one_call(self) -> None:
        tm = _make_tm()
        invoke, report = _wire(tm, _refusal_message("Declined."))

        result = tm.transcribe_payload(_payload(), max_schema_retries=3)

        assert invoke.call_count == 1
        report.assert_called_once()
        assert result["error_type"] == "refusal"
        assert "Declined." in result["error"]

    def test_max_output_tokens_keeps_schema_retries(self) -> None:
        cfg = {
            "validation_failure": {
                "enabled": True,
                "max_attempts": 2,
                "backoff_base": 0.0,
                "backoff_multiplier": 1.0,
            }
        }
        tm = _make_tm(schema_retry_config=cfg)
        invoke, _ = _wire(tm, _truncated_message('{"transcription": "cut o'))

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload(), max_schema_retries=3)

        assert invoke.call_count == 3
        assert result["error_type"] == "schema_validation_exhausted"
        assert result["schema_retries"]["validation_failure"] == 2


# ============================================================================
# Summary early stop
# ============================================================================
class TestSummaryEarlyStop:
    def test_content_filter_fails_page_after_one_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sm = _make_sm()
        wrapped = {"raw": _filtered_message(), "parsed": None, "parsing_error": None}
        invoke, report = _wire(sm, wrapped)

        with (
            patch("llm.summary.time.sleep") as sleep,
            caplog.at_level(logging.WARNING),
        ):
            result = sm.generate_summary("text", page_num=4, max_schema_retries=3)

        assert invoke.call_count == 1
        report.assert_called_once()
        sleep.assert_not_called()
        assert result["error_type"] == "content_filter"
        assert "content_filter" in result["error"]
        assert result["page"] == 4
        assert result["page_information"]["page_number_integer"] is None
        assert result["page_information"]["page_number_type"] == "none"
        assert any("content_filter" in rec.getMessage() for rec in caplog.records)

    def test_refusal_parsing_error_fails_page_after_one_call(self) -> None:
        sm = _make_sm()
        invoke, report = _wire(
            sm,
            {
                "raw": AIMessage(content=""),
                "parsed": None,
                "parsing_error": OpenAIRefusalError("Policy refusal."),
            },
        )

        result = sm.generate_summary("text", page_num=2)

        assert invoke.call_count == 1
        report.assert_called_once()
        assert result["error_type"] == "refusal"
        assert "Policy refusal." in result["error"]

    def test_max_output_tokens_keeps_schema_retries(self) -> None:
        cfg = {
            "validation_failure": {
                "enabled": True,
                "max_attempts": 1,
                "backoff_base": 0.0,
                "backoff_multiplier": 1.0,
            }
        }
        sm = _make_sm(schema_retry_config=cfg)
        invoke, _ = _wire(sm, _truncated_message('{"bullet_points": ["cut'))

        with patch("llm.summary.time.sleep"):
            result = sm.generate_summary("text", page_num=1, max_schema_retries=3)

        assert invoke.call_count == 2
        assert result["error_type"] == "schema_validation"

    def test_api_response_stores_unwrapped_metadata(self) -> None:
        sm = _make_sm()
        raw = AIMessage(
            content=_GOOD_SUMMARY,
            response_metadata={"status": "completed", "model": "gpt-5-mini"},
        )
        _wire(sm, {"raw": raw, "parsed": json.loads(_GOOD_SUMMARY)})

        result = sm.generate_summary("text", page_num=1)

        assert "error" not in result
        assert result["api_response"] == {"status": "completed", "model": "gpt-5-mini"}


# ============================================================================
# Resume treats a content-filtered page as failed
# ============================================================================
_FILTERED_ENTRY = {
    "image": "page_0002.png",
    "sequence_number": 2,
    "transcription": "[transcription error: filtered]",
    "error": "Response stopped by the provider's content filter",
    "error_type": "content_filter",
    "original_input_order_index": 1,
}


class TestResumeTreatsFilteredPageAsFailed:
    def test_completed_pages_skip_content_filter_entry(self) -> None:
        entries: list[dict[str, Any]] = [
            {"_format_version": 1},
            {"transcription": "ok", "original_input_order_index": 0},
            _FILTERED_ENTRY,
        ]
        assert _completed_pages_from_entries(entries) == {0}

    def test_load_completed_pages_from_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "item_transcription_log.json"
        assert initialize_log_file(
            log_path, "item", str(tmp_path / "in.pdf"), "PDF", 3, "gpt-5-mini"
        )
        append_to_log(log_path, {"transcription": "a", "original_input_order_index": 0})
        append_to_log(log_path, _FILTERED_ENTRY)
        append_to_log(log_path, {"transcription": "c", "original_input_order_index": 2})

        completed = load_completed_pages(log_path)

        assert completed == {0, 2}

    def test_summary_of_filtered_page_is_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)
        obj = ItemTranscriber.__new__(ItemTranscriber)
        obj.summary_manager = MagicMock()

        result = obj._summarize_transcription(dict(_FILTERED_ENTRY), 1, "page_0002.png")

        assert result is not None
        obj.summary_manager.generate_summary.assert_not_called()
        assert result["bullet_points"][0].startswith("[Transcription failed:")
        assert result["page"] == 2
        assert result["page_information"]["page_number_integer"] is None
        assert result["page_information"]["page_number_type"] == "none"

    def test_filtered_summary_is_error_marked_for_regeneration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)
        sm = _make_sm()
        _wire(sm, _filtered_message())
        obj = ItemTranscriber.__new__(ItemTranscriber)
        obj.summary_manager = sm
        entry = {"transcription": "Some text.", "original_input_order_index": 0}

        result = obj._summarize_transcription(entry, 0, "page_0001.png")

        assert result is not None
        # An "error" key makes resume regenerate this summary on a later run.
        assert "content_filter" in result["error"]
        assert result["error_type"] == "content_filter"
