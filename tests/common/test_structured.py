"""Tests for the vendored structured-call routine."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from autoexcerpter.common.structured import (
    PreparedRequest,
    StructuredRequest,
    default_invoke,
    extract_output_text,
    parse_json_object,
    prepare_request,
    sanitize_schema_for_anthropic,
    strip_code_fence,
    structured_call,
    validate_json,
)

GOLDEN = Path(__file__).resolve().parents[1] / "characterization" / "golden" / "formats"

TRANSCRIPTION_SPEC: dict[str, Any] = {
    "name": "markdown_transcription_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "image_analysis": {"type": "string"},
            "transcription": {"type": ["string", "null"]},
            "no_transcribable_text": {"type": "boolean"},
            "transcription_not_possible": {"type": "boolean"},
        },
        "required": [
            "image_analysis",
            "transcription",
            "no_transcribable_text",
            "transcription_not_possible",
        ],
        "additionalProperties": False,
    },
}

ANSWER = {
    "image_analysis": "Single column.",
    "transcription": "Bread prices rose.",
    "no_transcribable_text": False,
    "transcription_not_possible": False,
}

MESSAGES = [SystemMessage(content="Transcribe."), HumanMessage(content="page")]


class FakeModel:
    """Chat model stand-in that records each request and answers from a script.

    ``replies`` holds one entry per call: a string (message content), an
    ``AIMessage`` or an exception to raise.
    """

    def __init__(self, replies: list[Any], log: list[dict[str, Any]] | None = None):
        self.replies = replies
        self.log: list[dict[str, Any]] = [] if log is None else log
        self.bound: dict[str, Any] = {}

    def model_copy(self, *, update: Mapping[str, Any] | None = None) -> FakeModel:
        copied = FakeModel(self.replies, self.log)
        copied.bound = {**self.bound, **dict(update or {})}
        return copied

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        method: str = "function_calling",
    ) -> FakeStructured:
        return FakeStructured(self, schema, method, include_raw)

    def _answer(
        self, messages: list[Any], kwargs: dict[str, Any], structured: Any
    ) -> AIMessage:
        self.log.append(
            {
                "kwargs": kwargs,
                "bound": dict(self.bound),
                "structured": structured,
                "messages": messages,
            }
        )
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, AIMessage):
            return reply
        return AIMessage(content=reply)

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        return self._answer(messages, kwargs, None)


class FakeStructured:
    """Result of ``FakeModel.with_structured_output``; sync ``invoke`` only."""

    def __init__(
        self, model: FakeModel, schema: Any, method: str, include_raw: bool
    ) -> None:
        self.model = model
        self.schema = schema
        self.method = method
        self.include_raw = include_raw

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        structured = {
            "method": self.method,
            "include_raw": self.include_raw,
            "schema_title": self.schema.get("title"),
            "schema": self.schema,
        }
        raw = self.model._answer(messages, kwargs, structured)
        try:
            parsed = json.loads(str(raw.content))
        except json.JSONDecodeError:
            parsed = None
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _shape(call: Mapping[str, Any]) -> dict[str, Any]:
    """Project a recorded call (here or in a golden file) onto its shape."""
    kwargs = call["kwargs"]
    shape: dict[str, Any] = {
        "kwargs": sorted(kwargs),
        "bound": sorted(call["bound"]),
    }
    text = kwargs.get("text")
    if isinstance(text, Mapping) and "format" in text:
        fmt = text["format"]
        shape["text"] = sorted(text)
        shape["text_format"] = [fmt["type"], fmt["name"], fmt["strict"]]
    response_format = kwargs.get("response_format")
    if isinstance(response_format, Mapping):
        inner = response_format["json_schema"]
        shape["response_format"] = [
            response_format["type"],
            inner["name"],
            inner["strict"],
        ]
    if "response_mime_type" in kwargs:
        shape["mime"] = kwargs["response_mime_type"]
        shape["response_schema"] = sorted(kwargs["response_schema"]["properties"])
    structured = call["structured"]
    if structured:
        shape["structured"] = [
            structured["method"],
            structured["include_raw"],
            structured["schema_title"],
        ]
    return shape


CASES: dict[str, tuple[str, str, dict[str, Any], dict[str, Any]]] = {
    "openai": (
        "schema_openai_pdf",
        "openai",
        {
            "max_output_tokens": 128000,
            "reasoning": {"effort": "high"},
            "service_tier": "flex",
            "text": {"verbosity": "medium"},
        },
        {
            "kwargs": ["max_output_tokens", "reasoning", "service_tier", "text"],
            "bound": [],
            "text": ["format", "verbosity"],
            "text_format": ["json_schema", "markdown_transcription_schema", True],
        },
    ),
    "anthropic": (
        "schema_anthropic_images",
        "anthropic",
        {
            "max_tokens": 64000,
            "thinking": {"type": "enabled", "budget_tokens": 8192},
        },
        {
            "kwargs": ["max_tokens", "thinking"],
            "bound": ["max_tokens", "thinking"],
            "structured": ["json_schema", True, "markdown_transcription_schema"],
        },
    ),
    "google": (
        "schema_google_images",
        "google",
        {"max_output_tokens": 8192, "temperature": 1.0},
        {
            "kwargs": [
                "max_output_tokens",
                "response_mime_type",
                "response_schema",
                "temperature",
            ],
            "bound": [],
            "mime": "application/json",
            "response_schema": sorted(ANSWER),
        },
    ),
    "openrouter": (
        "schema_openrouter_pdf",
        "openrouter",
        {"temperature": 1.0},
        {
            "kwargs": ["temperature"],
            "bound": ["temperature"],
            "structured": ["function_calling", True, "markdown_transcription_schema"],
        },
    ),
    "custom": (
        "schema_custom_pdf",
        "custom",
        {"temperature": 1.0},
        {
            "kwargs": ["response_format", "temperature"],
            "bound": [],
            "response_format": [
                "json_schema",
                "markdown_transcription_schema",
                True,
            ],
        },
    ),
}


def _schema_call(provider: str, kwargs: dict[str, Any]) -> tuple[Any, list[Any]]:
    model = FakeModel([json.dumps(ANSWER)])
    request = StructuredRequest(
        provider=provider,
        response_format="schema",
        schema=TRANSCRIPTION_SPEC,
        invoke_kwargs=kwargs,
    )
    return run(structured_call(model, MESSAGES, request)), model.log


@pytest.mark.parametrize("name", sorted(CASES))
def test_schema_request_shape_per_provider(name: str) -> None:
    _golden, provider, kwargs, expected = CASES[name]
    result, log = _schema_call(provider, kwargs)

    assert len(log) == 1
    assert _shape(log[0]) == expected
    assert result.ok and result.enforced
    assert result.data == ANSWER


@pytest.mark.parametrize("name", sorted(CASES))
def test_schema_request_shape_matches_golden(name: str) -> None:
    golden, provider, kwargs, _expected = CASES[name]
    path = GOLDEN / golden / "requests.json"
    if not path.exists():
        pytest.skip(f"golden file missing: {path}")
    calls = json.loads(path.read_text(encoding="utf-8"))["calls"]
    recorded = next(c for c in calls if c["role"] == "transcription")

    _result, log = _schema_call(provider, kwargs)

    assert _shape(log[0]) == _shape(recorded)


def test_openai_schema_keeps_caller_kwargs_unchanged() -> None:
    kwargs = {"text": {"verbosity": "low"}}
    request = prepare_request(object(), "openai", "schema", TRANSCRIPTION_SPEC, kwargs)

    assert kwargs == {"text": {"verbosity": "low"}}
    assert request.kwargs["text"]["format"]["schema"] is not None


def test_google_thinking_suppresses_response_schema() -> None:
    model = FakeModel([json.dumps(ANSWER)])
    request = StructuredRequest(
        provider="google",
        response_format="schema",
        schema=TRANSCRIPTION_SPEC,
        invoke_kwargs={"thinking_config": {"thinking_budget": 8192}},
    )

    result = run(structured_call(model, MESSAGES, request))

    assert sorted(model.log[0]["kwargs"]) == ["thinking_config"]
    assert not result.enforced
    assert result.ok


def test_anthropic_schema_is_sanitized_without_mutating_the_spec() -> None:
    original = json.dumps(TRANSCRIPTION_SPEC, sort_keys=True)
    model = FakeModel([json.dumps(ANSWER)])
    request = StructuredRequest("anthropic", "schema", TRANSCRIPTION_SPEC)

    run(structured_call(model, MESSAGES, request))

    sent = model.log[0]["structured"]["schema"]
    assert sent["properties"]["transcription"]["type"] == "string"
    assert sent["title"] == "markdown_transcription_schema"
    assert json.dumps(TRANSCRIPTION_SPEC, sort_keys=True) == original


def test_existing_schema_title_is_kept() -> None:
    spec = {"name": "summary", "schema": {"title": "Page Summary", "type": "object"}}
    model = FakeModel(["{}"])
    request = StructuredRequest("openrouter", "schema", spec)

    run(structured_call(model, MESSAGES, request))

    assert model.log[0]["structured"]["schema_title"] == "Page Summary"


def test_sanitize_reduces_nested_union_types() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "items": {"type": ["array", "null"], "items": {"type": ["null"]}},
        },
    }

    result = sanitize_schema_for_anthropic(schema)

    assert result["properties"]["items"]["type"] == "array"
    assert result["properties"]["items"]["items"]["type"] == "string"
    assert schema["properties"]["items"]["type"] == ["array", "null"]


def test_schema_format_requires_a_schema() -> None:
    with pytest.raises(ValueError, match="needs a JSON schema"):
        StructuredRequest("openai", "schema", None)


def test_unknown_response_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown response format"):
        StructuredRequest("openai", "xml")  # type: ignore[arg-type]


def test_anthropic_schema_rejection_falls_back_to_prompted_json() -> None:
    error = RuntimeError("Tool schema contains too many conditional branches")
    model = FakeModel([error, f"```json\n{json.dumps(ANSWER)}\n```"])
    request = StructuredRequest(
        "anthropic", "schema", TRANSCRIPTION_SPEC, {"max_tokens": 10}
    )

    result = run(structured_call(model, MESSAGES, request))

    assert len(model.log) == 2
    assert model.log[1]["structured"] is None
    assert model.log[1]["kwargs"] == {"max_tokens": 10}
    assert result.fallback and not result.enforced
    assert result.ok and result.data == ANSWER


def test_other_errors_propagate() -> None:
    model = FakeModel([RuntimeError("boom")])
    request = StructuredRequest("anthropic", "schema", TRANSCRIPTION_SPEC)

    with pytest.raises(RuntimeError, match="boom"):
        run(structured_call(model, MESSAGES, request))


def test_schema_rejection_on_other_providers_propagates() -> None:
    error = RuntimeError("Tool schema contains too many conditional branches")
    model = FakeModel([error])
    request = StructuredRequest("openrouter", "schema", TRANSCRIPTION_SPEC)

    with pytest.raises(RuntimeError):
        run(structured_call(model, MESSAGES, request))


def test_json_format_sends_no_schema_and_parses_fenced_reply() -> None:
    model = FakeModel([f"```json\n{json.dumps(ANSWER)}\n```"])
    request = StructuredRequest(
        "openai", "json", TRANSCRIPTION_SPEC, {"max_output_tokens": 5}
    )

    result = run(structured_call(model, MESSAGES, request))

    assert model.log[0]["kwargs"] == {"max_output_tokens": 5}
    assert model.log[0]["structured"] is None
    assert result.ok and not result.enforced
    assert result.data == ANSWER
    assert result.validation_retries == 0


def test_json_invalid_then_valid_retries_within_budget() -> None:
    model = FakeModel(["not json at all", json.dumps(ANSWER)])
    delays: list[int] = []

    def delay(attempt: int) -> float:
        delays.append(attempt)
        return 0.0

    request = StructuredRequest(
        "anthropic",
        "json",
        TRANSCRIPTION_SPEC,
        validation_retries=2,
        validation_delay=delay,
    )

    result = run(structured_call(model, MESSAGES, request))

    assert len(model.log) == 2
    assert delays == [1]
    assert result.ok and result.data == ANSWER
    assert result.validation_retries == 1
    assert len(result.responses) == 2


def test_json_validation_budget_exhausted_reports_reason() -> None:
    partial = json.dumps({"transcription": "x"})
    model = FakeModel(["{broken", partial])
    request = StructuredRequest("openai", "json", TRANSCRIPTION_SPEC, None, None, 1)

    result = run(structured_call(model, MESSAGES, request))

    assert len(model.log) == 2
    assert not result.ok
    assert result.error == (
        "missing keys: image_analysis, no_transcribable_text, "
        "transcription_not_possible"
    )
    assert result.data == {"transcription": "x"}


def test_json_without_retries_returns_first_failure() -> None:
    model = FakeModel(["[1, 2]"])
    request = StructuredRequest("custom", "json", TRANSCRIPTION_SPEC)

    result = run(structured_call(model, MESSAGES, request))

    assert result.error == "expected JSON object, got list"
    assert len(model.log) == 1


def test_explicit_required_keys_override_the_schema() -> None:
    model = FakeModel(['{"transcription": "x"}'])
    request = StructuredRequest(
        "openai", "json", TRANSCRIPTION_SPEC, required=("transcription",)
    )

    result = run(structured_call(model, MESSAGES, request))

    assert result.ok


def test_schema_answers_are_validated_too() -> None:
    model = FakeModel(['{"transcription": "x"}', json.dumps(ANSWER)])
    request = StructuredRequest(
        "openai", "schema", TRANSCRIPTION_SPEC, validation_retries=1
    )

    result = run(structured_call(model, MESSAGES, request))

    assert len(model.log) == 2
    assert result.ok


def test_empty_answer_returns_without_validation_retry() -> None:
    model = FakeModel(["", json.dumps(ANSWER)])
    request = StructuredRequest(
        "openai", "json", TRANSCRIPTION_SPEC, validation_retries=3
    )

    result = run(structured_call(model, MESSAGES, request))

    assert len(model.log) == 1
    assert result.error == "empty response"


def test_stop_predicate_ends_the_call() -> None:
    refusal = AIMessage(content="no", additional_kwargs={"refusal": "no"})
    model = FakeModel([refusal, json.dumps(ANSWER)])
    request = StructuredRequest(
        "openai", "json", TRANSCRIPTION_SPEC, validation_retries=3
    )

    def stop(response: Any) -> bool:
        return bool(response.additional_kwargs.get("refusal"))

    result = run(structured_call(model, MESSAGES, request, stop=stop))

    assert len(model.log) == 1
    assert result.stopped and not result.ok


def test_text_format_returns_the_answer_unchanged() -> None:
    model = FakeModel(["```\n# Title\n```"])
    request = StructuredRequest("custom", "text", None, {"max_tokens": 5})

    result = run(structured_call(model, MESSAGES, request))

    assert model.log[0]["kwargs"] == {"max_tokens": 5}
    assert result.ok and result.data is None
    assert result.text == "```\n# Title\n```"


def test_text_format_ignores_a_schema() -> None:
    model = FakeModel(["plain"])
    request = StructuredRequest("openai", "text", TRANSCRIPTION_SPEC)

    result = run(structured_call(model, MESSAGES, request))

    assert "text" not in model.log[0]["kwargs"]
    assert result.text == "plain"


def test_invoke_hook_wraps_every_call() -> None:
    model = FakeModel(["bad", json.dumps(ANSWER)])
    seen: list[dict[str, Any]] = []

    async def invoke(prepared: PreparedRequest, messages: Any) -> Any:
        seen.append(prepared.kwargs)
        return await prepared.runnable.ainvoke(messages, **prepared.kwargs)

    request = StructuredRequest(
        "openai", "json", TRANSCRIPTION_SPEC, {"a": 1}, validation_retries=1
    )

    result = run(structured_call(model, MESSAGES, request, invoke=invoke))

    assert seen == [{"a": 1}, {"a": 1}]
    assert result.ok


def test_for_model_rebuilds_the_request_for_another_client() -> None:
    first = FakeModel([json.dumps(ANSWER)])
    second = FakeModel([json.dumps(ANSWER)])
    prepared = prepare_request(
        first, "anthropic", "schema", TRANSCRIPTION_SPEC, {"max_tokens": 10}
    )

    assert prepared.for_model(first) is prepared
    rebuilt = prepared.for_model(second)

    assert rebuilt.runnable is not prepared.runnable
    assert rebuilt.enforced and rebuilt.model is second
    assert rebuilt.kwargs == prepared.kwargs
    run(default_invoke(rebuilt, MESSAGES))
    assert len(second.log) == 1
    assert not first.log


def test_strip_code_fence_variants() -> None:
    assert strip_code_fence("```json\n{}\n```") == "{}"
    assert strip_code_fence("```markdown\n# A\n```") == "# A"
    assert strip_code_fence('```json{"a": 1}```') == '{"a": 1}'
    assert strip_code_fence("plain") == "plain"


def test_parse_json_object_handles_prompted_shapes() -> None:
    inner = '{"transcription": "```py\\nx\\n```"}'
    assert parse_json_object(f"```json\n{inner}\n```") == {
        "transcription": "```py\nx\n```"
    }
    thinking = '```\nthinking\n```\n```json\n{"a": 1}\n```'
    assert parse_json_object(thinking) == {"a": 1}
    assert parse_json_object('Here it is: {"a": 1} Done.') == {"a": 1}
    assert parse_json_object('{"a": 1} {"b": 2}') == {"b": 2}
    assert parse_json_object("no json") is None
    assert parse_json_object("") is None


def test_validate_json_reasons() -> None:
    assert validate_json("nope", ("a",)) == (None, "invalid JSON")
    assert validate_json('{"a": 1}', ("a",)) == ({"a": 1}, None)


def test_extract_output_text_from_tool_call_arguments() -> None:
    raw = AIMessage(
        content="",
        tool_calls=[{"name": "s", "args": {"a": 1}, "id": "c1", "type": "tool_call"}],
    )
    wrapper = {"raw": raw, "parsed": None, "parsing_error": ValueError("x")}

    assert json.loads(extract_output_text(wrapper)) == {"a": 1}


def test_extract_output_text_from_responses_shapes() -> None:
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, "b"])
    nested = {"output": [{"content": [{"type": "output_text", "text": "c"}]}]}

    assert extract_output_text(blocks) == "ab"
    assert extract_output_text({"output_text": " d "}) == "d"
    assert extract_output_text(nested) == "c"
    assert extract_output_text(object()) == ""
