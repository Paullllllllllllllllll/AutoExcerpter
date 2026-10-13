"""How finalization reads a stopped result and names a failed one."""

from __future__ import annotations

from typing import Any

import pytest

from autoexcerpter.batch.finalize import MISSING_TEXT, placeholder, stopped
from autoexcerpter.common.batch import BatchResultItem


def _item(response: dict[str, Any], **fields: Any) -> BatchResultItem:
    return BatchResultItem(custom_id="c-00000", response=response, **fields)


def _openai(body: dict[str, Any]) -> BatchResultItem:
    return _item({"status_code": 200, "body": body}, text="", origin="output")


@pytest.mark.parametrize(
    ("provider", "item", "kind"),
    [
        (
            "openai",
            _openai(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                }
            ),
            "content_filter",
        ),
        (
            "openai",
            _openai(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "refusal", "refusal": "No."}],
                        }
                    ],
                }
            ),
            "refusal",
        ),
        (
            "anthropic",
            _item({"message": {"stop_reason": "refusal", "content": []}}),
            "refusal",
        ),
        (
            "google",
            _item({"candidates": [{"finish_reason": "SAFETY"}]}),
            "content_filter",
        ),
    ],
)
def test_a_filtered_or_refused_answer_stops_the_page(
    provider: str, item: BatchResultItem, kind: str
) -> None:
    outcome = stopped(provider, item)
    assert outcome is not None
    assert outcome.kind == kind


@pytest.mark.parametrize(
    ("provider", "response"),
    [
        ("openai", {"status_code": 200, "body": {"status": "completed"}}),
        ("anthropic", {"message": {"stop_reason": "end_turn", "content": []}}),
        ("google", {"candidates": [{"finish_reason": "STOP"}]}),
    ],
)
def test_a_finished_answer_does_not_stop_the_page(
    provider: str, response: dict[str, Any]
) -> None:
    assert stopped(provider, _item(response)) is None


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
