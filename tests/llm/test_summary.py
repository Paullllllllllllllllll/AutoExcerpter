"""SummaryManager: requests per provider and format, retries, results."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from autoexcerpter.common.usage import Usage
from autoexcerpter.llm.caller import CallEnv
from autoexcerpter.llm.prompts import PROMPTS_DIR
from autoexcerpter.llm.summary import (
    SummaryManager,
    ensure_page_information,
    placeholder_summary,
)
from autoexcerpter.settings import RetrySettings
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import SUMMARY_ANSWER, StatusError, message

TEXT = "Bread prices rose after 1550."


def _manager(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[Any],
    *,
    retry: RetrySettings | None = None,
    env: CallEnv | None = None,
    context: str | None = None,
    transcription_format: str = "schema",
    **phase_fields: Any,
) -> tuple[SummaryManager, helpers.Script]:
    script = helpers.install(monkeypatch, answers)
    manager = SummaryManager(
        helpers.phase(**phase_fields),
        env or helpers.env(retry),
        context=context,
        transcription_format=transcription_format,
    )
    return manager, script


def _summarize(manager: SummaryManager, page: int = 3) -> dict[str, Any]:
    return asyncio.run(manager.generate_summary(TEXT, page))


# ============================================================================
# Requests
# ============================================================================
@pytest.mark.parametrize(
    ("provider", "model", "as_list"),
    [
        ("openai", "gpt-5.6-terra", True),
        ("openrouter", "anthropic/claude-sonnet-4.5", True),
        ("anthropic", "claude-sonnet-4-5", False),
        ("google", "gemini-2.5-flash-lite", False),
    ],
)
def test_user_message_shape_per_provider(
    monkeypatch: pytest.MonkeyPatch, provider: str, model: str, as_list: bool
) -> None:
    manager, script = _manager(
        monkeypatch, [SUMMARY_ANSWER], name=model, provider=provider
    )

    _summarize(manager)

    content = script.requests[0].messages[1].content
    if as_list:
        assert content == [{"type": "text", "text": TEXT}]
    else:
        assert content == TEXT


def test_context_enters_the_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, script = _manager(
        monkeypatch, [SUMMARY_ANSWER], context="Grain prices, Wages"
    )
    _summarize(manager)
    assert "Grain prices, Wages" in str(script.requests[0].messages[0].content)

    without, script = _manager(monkeypatch, [SUMMARY_ANSWER])
    _summarize(without)
    assert "{{CONTEXT}}" not in str(script.requests[0].messages[0].content)


@pytest.mark.parametrize(
    ("transcription_format", "prompt_file"),
    [
        ("schema", "summary_system_prompt.txt"),
        ("text", "summary_plain_text_prompt.txt"),
    ],
)
def test_the_prompt_follows_the_transcription_format(
    monkeypatch: pytest.MonkeyPatch, transcription_format: str, prompt_file: str
) -> None:
    manager, script = _manager(
        monkeypatch, [SUMMARY_ANSWER], transcription_format=transcription_format
    )

    _summarize(manager)

    first_line = (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
    system = str(script.requests[0].messages[0].content)
    assert system.splitlines()[0] == first_line.splitlines()[0]
    assert "page_information" in system


@pytest.mark.parametrize(("fmt", "enforced"), [("schema", True), ("json", False)])
def test_schema_is_enforced_only_in_the_schema_format(
    monkeypatch: pytest.MonkeyPatch, fmt: str, enforced: bool
) -> None:
    manager, script = _manager(monkeypatch, [SUMMARY_ANSWER], response_format=fmt)

    _summarize(manager)

    text = script.requests[0].kwargs.get("text", {})
    assert ("format" in text) is enforced


# ============================================================================
# Results and retries
# ============================================================================
def test_success_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    env = helpers.env()
    manager, _ = _manager(monkeypatch, [SUMMARY_ANSWER], env=env)

    result = _summarize(manager)

    assert result["page"] == 3
    assert result["bullet_points"] == ["Bread prices rose."]
    assert result["page_information"]["page_number_integer"] == 3
    assert result["provider"] == "openai"
    assert result["api_response"] == {}
    assert "schema_retries" not in result
    assert "error" not in result
    assert env.usage.get("summary") == Usage(100, 20, 120)


def test_response_metadata_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = message(
        json.dumps(SUMMARY_ANSWER), response_metadata={"model_name": "gpt-5.6-terra"}
    )
    manager, _ = _manager(monkeypatch, [answer])

    assert _summarize(manager)["api_response"] == {"model_name": "gpt-5.6-terra"}


def test_fenced_json_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = f"```json\n{json.dumps(SUMMARY_ANSWER)}\n```"
    manager, script = _manager(monkeypatch, [fenced], response_format="json")

    result = _summarize(manager)

    assert len(script.requests) == 1
    assert result["bullet_points"] == ["Bread prices rose."]


def test_invalid_json_is_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(monkeypatch, ["no json", SUMMARY_ANSWER])

    result = _summarize(manager)

    assert len(script.requests) == 2
    assert result["schema_retries"] == {"validation_failure": 1}
    assert "error" not in result


def test_exhausted_json_leaves_an_error_placeholder(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(monkeypatch, ["no json"], response_format="json")

    result = _summarize(manager)

    assert len(script.requests) == 4
    assert result["error_type"] == "schema_validation"
    assert result["error"] == "Invalid summary JSON: invalid JSON"
    assert result["schema_retries"] == {"validation_failure": 3}
    assert result["page_information"]["page_number_integer"] is None


def test_exhausted_text_answers_become_one_bullet(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(
        monkeypatch, ["```\nA plain summary.\n```"], response_format="text"
    )

    result = _summarize(manager)

    assert len(script.requests) == 4
    assert "format" not in script.requests[0].kwargs.get("text", {})
    assert result["bullet_points"] == ["A plain summary."]
    assert result["schema_retries"] == {"validation_failure": 3}
    assert result["page_information"] == {
        "page_number_integer": None,
        "page_number_type": "none",
        "page_types": ["content"],
        "is_two_page_spread": False,
        "page_number_integer_end": None,
    }
    assert "error" not in result


def test_empty_answers_are_retried_then_fail(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(monkeypatch, ["", SUMMARY_ANSWER])
    assert "error" not in _summarize(manager)
    assert len(script.requests) == 2

    manager, script = _manager(monkeypatch, [""])
    result = _summarize(manager)
    assert len(script.requests) == 3
    assert result["error_type"] == "api_failure"
    assert "empty content" in result["error"]


@pytest.mark.parametrize(
    "response",
    [
        message(response_metadata={"finish_reason": "content_filter"}),
        message(response_metadata={"stop_reason": "refusal"}),
    ],
)
def test_a_filtered_summary_stops_after_one_call(
    monkeypatch: pytest.MonkeyPatch, response: AIMessage
) -> None:
    manager, script = _manager(monkeypatch, [response, SUMMARY_ANSWER])

    result = _summarize(manager)

    assert len(script.requests) == 1
    assert result["error_type"] in ("content_filter", "refusal")
    assert result["bullet_points"][0].startswith("[Error generating summary:")
    assert result["provider"] == "openai"


def test_an_api_error_leaves_a_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = _manager(monkeypatch, [StatusError(403)])

    result = _summarize(manager)

    assert result["error_type"] == "api_failure"
    assert result["error"] == "status 403"


# ============================================================================
# Placeholders and page information
# ============================================================================
def test_placeholder_summary() -> None:
    with_error = placeholder_summary(4, "boom")
    assert with_error["bullet_points"] == ["[Error generating summary: boom]"]
    assert with_error["error"] == "boom"
    assert with_error["page_information"]["page_types"] == ["other"]

    plain = placeholder_summary(4, page_types=["blank"])
    assert plain["bullet_points"] == ["[Summary generation failed]"]
    assert "error" not in plain
    assert plain["page_information"]["page_types"] == ["blank"]


@pytest.mark.parametrize(
    ("info", "expected_type"),
    [({"page_number_integer": 7}, "arabic"), ({"page_number_integer": None}, "none")],
)
def test_ensure_page_information_fills_the_type(
    info: dict[str, Any], expected_type: str
) -> None:
    summary: dict[str, Any] = {"page_information": dict(info)}
    ensure_page_information(summary)
    filled = summary["page_information"]
    assert filled["page_number_type"] == expected_type
    assert filled["page_types"] == ["content"]
    assert filled["is_two_page_spread"] is False
    assert filled["page_number_integer_end"] is None

    replaced: dict[str, Any] = {"page_information": "garbage"}
    ensure_page_information(replaced)
    assert replaced["page_information"]["page_number_type"] == "none"
