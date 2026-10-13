"""Batch request bodies carry what the synchronous calls send."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.batch.bodies import ROLE_SUMMARY, ROLE_TRANSCRIPTION, build_plan
from autoexcerpter.common.batch import BatchRequest
from autoexcerpter.common.structured import (
    anthropic_output_schema,
    build_text_format,
    schema_body,
)
from autoexcerpter.llm.options import build_invoke_kwargs
from autoexcerpter.llm.summary import summary_prompt
from autoexcerpter.llm.transcription import transcription_prompt
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.pipeline.job import job_from_spec
from autoexcerpter.settings import Endpoint, Settings
from autoexcerpter.spec import resolve_spec

SETTINGS = Settings(
    endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
)
IMAGE = BatchRequest(
    custom_id="page-0", image_base64="AAAA", mime_type="image/jpeg", order_index=0
)
TEXT = BatchRequest(custom_id="summary-0", text="## Page text", order_index=0)


def _phase(**flags: Any) -> PhaseModel:
    spec = resolve_spec({"input": Path("x.pdf"), **flags}).spec
    return job_from_spec(spec, SETTINGS).transcription


def _kwargs(phase: PhaseModel) -> dict[str, Any]:
    return build_invoke_kwargs(
        phase.provider, phase.name, phase.options, phase.service_tier
    )


def test_an_openai_transcription_sends_the_synchronous_request() -> None:
    phase = _phase(model="gpt-5.6-terra")
    plan = build_plan(phase, ROLE_TRANSCRIPTION)
    body = plan.body(IMAGE)
    schema, prompt = transcription_prompt(False)
    kwargs = _kwargs(phase)

    assert body["model"] == "gpt-5.6-terra"
    assert body["input"][0] == {
        "role": "system",
        "content": [{"type": "input_text", "text": prompt}],
    }
    (image,) = body["input"][1]["content"]
    assert image["type"] == "input_image"
    assert image["image_url"] == "data:image/jpeg;base64,AAAA"
    assert image["detail"] == plan.detail
    assert body["max_output_tokens"] == kwargs["max_output_tokens"]
    assert body["reasoning"] == kwargs["reasoning"]
    assert body["text"] == {**kwargs["text"], "format": build_text_format(schema)}
    assert kwargs["service_tier"] == "flex"
    assert body["service_tier"] == "auto"


def test_an_openai_summary_sends_the_page_text_with_the_context() -> None:
    phase = _phase(model="gpt-5.6-terra", service_tier="priority")
    plan = build_plan(phase, ROLE_SUMMARY, context="Prices, Wages")
    body = plan.body(TEXT)
    schema, prompt = summary_prompt("Prices, Wages", "schema")

    assert body["input"][0]["content"][0]["text"] == prompt
    assert body["input"][1]["content"] == [
        {"type": "input_text", "text": "## Page text"}
    ]
    assert body["text"]["format"] == build_text_format(schema)
    assert body["service_tier"] == "priority"


def test_an_anthropic_request_caches_the_system_prompt_and_enforces_the_schema() -> (
    None
):
    phase = _phase(model="claude-sonnet-4-5")
    plan = build_plan(phase, ROLE_SUMMARY)
    body = plan.body(TEXT)
    schema, prompt = summary_prompt(None, "schema")
    kwargs = _kwargs(phase)

    assert body["system"] == [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": "## Page text"}]
    assert body["max_tokens"] == kwargs["max_tokens"]
    assert body["thinking"] == kwargs["thinking"]
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": anthropic_output_schema(schema),
    }


def test_an_anthropic_transcription_sends_the_image_alone() -> None:
    body = build_plan(_phase(model="claude-sonnet-4-5"), ROLE_TRANSCRIPTION).body(IMAGE)
    assert body["messages"][0]["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
        }
    ]


def test_a_google_request_enforces_the_schema_unless_thinking_is_set() -> None:
    plain = _phase(model="gemini-2.5-flash-lite")
    body = build_plan(plain, ROLE_SUMMARY).body(TEXT)
    schema, _prompt = summary_prompt(None, "schema")
    assert "thinking_config" not in _kwargs(plain)
    assert body["config"]["response_mime_type"] == "application/json"
    assert body["config"]["response_json_schema"] == schema_body(schema)
    assert body["contents"] == [{"role": "user", "parts": [{"text": "## Page text"}]}]

    thinking = _phase(model="gemini-3-pro-preview")
    config = build_plan(thinking, ROLE_SUMMARY).body(TEXT)["config"]
    assert config["thinking_config"] == _kwargs(thinking)["thinking_config"]
    assert "response_json_schema" not in config


def test_the_text_format_sends_the_plain_prompt_without_a_schema() -> None:
    phase = _phase(model="gpt-5.6-terra", response_format="text")
    plan = build_plan(phase, ROLE_TRANSCRIPTION)
    _schema, prompt = transcription_prompt(True)
    assert plan.system_prompt == prompt
    assert plan.schema is None
    assert "format" not in plan.body(IMAGE).get("text", {})


def test_a_text_summary_of_plain_transcriptions_asks_for_prompted_json() -> None:
    phase = _phase(model="gpt-5.6-terra", response_format="text")
    plan = build_plan(phase, ROLE_SUMMARY, transcription_format="text")
    _schema, prompt = summary_prompt(None, "text")
    assert plan.system_prompt == prompt
    assert plan.response_format == "text"
    assert plan.schema is None


@pytest.mark.parametrize(
    "flags",
    [
        {"model": "anthropic/claude-sonnet-4.5"},
        {"provider": "custom", "endpoint": "local", "model": "m"},
    ],
)
def test_a_phase_without_a_batch_api_is_refused(flags: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="no batch API"):
        build_plan(_phase(**flags), ROLE_TRANSCRIPTION)


def test_a_request_without_text_or_image_is_refused() -> None:
    plan = build_plan(_phase(model="gpt-5.6-terra"), ROLE_SUMMARY)
    with pytest.raises(ValueError, match="no text or image"):
        plan.body(BatchRequest(custom_id="empty", text="", order_index=0))
