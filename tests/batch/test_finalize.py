"""How finalization reads a stopped result and names a failed one."""

from __future__ import annotations

from typing import Any

import pytest

from autoexcerpter.batch.finalize import MISSING_TEXT, _Answer, _answer, placeholder
from autoexcerpter.common.batch import BatchResultItem
from autoexcerpter.common.batch_google import parse_response
from autoexcerpter.common.responses import ResponseOutcome


def _item(response: dict[str, Any], **fields: Any) -> BatchResultItem:
    return BatchResultItem(custom_id="c-00000", response=response, **fields)


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION"])
def test_a_filtered_answer_stops_the_page(reason: str) -> None:
    candidate = {"content": {"parts": [{"text": "a"}]}, "finishReason": reason}
    item = parse_response("c-00000", {"candidates": [candidate]})
    assert _answer("google", item) == _Answer(
        None,
        f"Response stopped by the provider's content filter: finish_reason={reason}",
        "content_filter",
    )


@pytest.mark.parametrize(
    ("outcome", "answer"),
    [
        (ResponseOutcome("ok"), _Answer("text")),
        (
            ResponseOutcome("truncated", "stop_reason=max_tokens"),
            _Answer(
                None,
                "Response truncated at the output-token limit: stop_reason=max_tokens",
                "truncated",
            ),
        ),
    ],
)
def test_a_finished_answer_is_kept_and_a_truncated_one_fails(
    outcome: ResponseOutcome, answer: _Answer
) -> None:
    item = _item({}, text="text")
    item.settle(outcome, "empty")
    assert _answer("anthropic", item) == answer


def test_the_placeholder_names_the_openai_status_and_code() -> None:
    item = _item(
        {"status_code": 400},
        error="bad image",
        error_code="invalid_request",
        origin="output",
    )
    assert placeholder("openai", item) == (
        "[transcription error; status 400; code invalid_request; bad image]"
    )
    missing = _item({}, origin="missing")
    assert placeholder("anthropic", missing) == MISSING_TEXT
    assert placeholder("google", _item({}, error="overloaded")) == (
        "[transcription error: overloaded]"
    )
