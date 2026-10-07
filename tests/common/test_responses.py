"""Tests for common/responses.py: how a model response ended."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_openai.chat_models.base import OpenAIRefusalError

from autoexcerpter.common.responses import (
    ResponseOutcome,
    get_response_metadata,
    inspect_response,
)


def _filtered_message() -> AIMessage:
    return AIMessage(
        content=[],
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
        },
    )


def _truncated_message(text: str = "partial") -> AIMessage:
    return AIMessage(
        content=[{"type": "text", "text": text}],
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    )


class TestInspectResponse:
    def test_responses_api_content_filter(self) -> None:
        outcome = inspect_response(_filtered_message())
        assert outcome.kind == "content_filter"
        assert outcome.reason == "incomplete_details.reason=content_filter"
        assert outcome.stops_page

    def test_responses_api_max_output_tokens(self) -> None:
        outcome = inspect_response(_truncated_message())
        assert outcome.kind == "max_output_tokens"
        assert not outcome.stops_page

    @pytest.mark.parametrize(
        ("metadata", "reason"),
        [
            ({"stop_reason": "max_tokens"}, "stop_reason=max_tokens"),
            ({"finish_reason": "MAX_TOKENS"}, "finish_reason=MAX_TOKENS"),
        ],
    )
    def test_other_provider_truncation(
        self, metadata: dict[str, Any], reason: str
    ) -> None:
        message = AIMessage(content="cut", response_metadata=metadata)
        assert inspect_response(message) == ResponseOutcome("max_output_tokens", reason)

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

    def test_other_parsing_error_is_ok(self) -> None:
        wrapped = {
            "raw": AIMessage(content="{"),
            "parsed": None,
            "parsing_error": ValueError("bad json"),
        }
        assert inspect_response(wrapped).kind == "ok"

    def test_additional_kwargs_refusal(self) -> None:
        message = AIMessage(content="", additional_kwargs={"refusal": "No can do."})
        assert inspect_response(message) == ResponseOutcome("refusal", "No can do.")

    def test_refusal_content_block(self) -> None:
        message = AIMessage(content=[{"type": "refusal", "refusal": "Not allowed."}])
        assert inspect_response(message) == ResponseOutcome("refusal", "Not allowed.")

    def test_empty_refusal_block_has_no_reason(self) -> None:
        message = AIMessage(content=[{"type": "refusal"}])
        assert inspect_response(message) == ResponseOutcome("refusal")

    def test_chat_completions_content_filter(self) -> None:
        message = AIMessage(
            content="", response_metadata={"finish_reason": "content_filter"}
        )
        outcome = inspect_response(message)
        assert outcome == ResponseOutcome(
            "content_filter", "finish_reason=content_filter"
        )

    def test_chat_completions_length_is_ok(self) -> None:
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


class TestResponseOutcome:
    def test_message_with_reason(self) -> None:
        outcome = ResponseOutcome("refusal", "Declined.")
        assert outcome.message == "Model refused the request: Declined."

    def test_message_without_reason(self) -> None:
        assert ResponseOutcome("content_filter").message == (
            "Response stopped by the provider's content filter"
        )


def test_get_response_metadata_unwraps_include_raw() -> None:
    raw = AIMessage(content="", response_metadata={"model": "gpt-5-mini"})
    assert get_response_metadata({"raw": raw, "parsed": {}}) == {"model": "gpt-5-mini"}
    assert get_response_metadata(MagicMock()) == {}
