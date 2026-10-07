"""TranscriptionManager: requests per provider and format, retries, results."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from autoexcerpter.common.images import PagePayload
from autoexcerpter.common.testing import FakeRequest
from autoexcerpter.common.usage import Usage
from autoexcerpter.llm.caller import CallEnv
from autoexcerpter.llm.prompts import PROMPTS_DIR
from autoexcerpter.llm.transcription import TranscriptionManager, page_text
from autoexcerpter.settings import RetrySettings
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import TRANSCRIPTION_ANSWER, StatusError, message

NAME = "page_0001.jpg"


def _payload(mime: str = "image/jpeg") -> PagePayload:
    return PagePayload(
        base64="QUJD",
        image_name=NAME,
        number=1,
        index=0,
        provenance={},
        source_file="doc.pdf",
        mime_type=mime,
    )


def _answer(**changes: Any) -> dict[str, Any]:
    return {**TRANSCRIPTION_ANSWER, **changes}


def _manager(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[Any],
    *,
    retry: RetrySettings | None = None,
    env: CallEnv | None = None,
    **phase_fields: Any,
) -> tuple[TranscriptionManager, helpers.Script]:
    script = helpers.install(monkeypatch, answers)
    manager = TranscriptionManager(
        helpers.phase(**phase_fields), env or helpers.env(retry)
    )
    return manager, script


def _transcribe(manager: TranscriptionManager, mime: str = "image/jpeg") -> Any:
    return asyncio.run(manager.transcribe_payload(_payload(mime)))


def _image_block(request: FakeRequest) -> dict[str, Any]:
    content = request.messages[1].content
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    return block


# ============================================================================
# Requests
# ============================================================================
def test_openai_request_carries_detail_and_the_enforced_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, script = _manager(
        monkeypatch, [_answer()], options={"image_size": "original"}
    )

    result = _transcribe(manager)

    assert result["transcription"] == "Bread prices rose."
    (request,) = script.requests
    block = _image_block(request)
    assert block["type"] == "image_url"
    assert block["image_url"]["detail"] == "original"
    assert block["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")
    assert request.kwargs["text"]["format"]["type"] == "json_schema"
    assert request.kwargs["service_tier"] == "flex"


def test_anthropic_and_google_image_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic, script = _manager(
        monkeypatch, [_answer()], name="claude-sonnet-4-5", provider="anthropic"
    )
    _transcribe(anthropic, "image/png")
    block = _image_block(script.requests[0])
    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }

    google, script = _manager(
        monkeypatch,
        [_answer()],
        name="gemini-2.5-flash-lite",
        provider="google",
        options={"image_size": "high"},
    )
    _transcribe(google)
    block = _image_block(script.requests[0])
    assert block == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }
    assert script.requests[0].kwargs["response_mime_type"] == "application/json"


@pytest.mark.parametrize(
    ("fmt", "prompt_file", "enforced"),
    [
        ("schema", "transcription_system_prompt.txt", True),
        ("json", "transcription_system_prompt.txt", False),
        ("text", "transcription_plain_text_prompt.txt", False),
    ],
)
def test_response_format_picks_prompt_and_enforcement(
    monkeypatch: pytest.MonkeyPatch, fmt: str, prompt_file: str, enforced: bool
) -> None:
    answer = "Plain page text." if fmt == "text" else _answer()
    manager, script = _manager(monkeypatch, [answer], response_format=fmt)

    result = _transcribe(manager)

    raw_prompt = (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
    system = str(script.requests[0].messages[0].content)
    assert system.startswith(raw_prompt.split("\n", 1)[0])
    assert ("{{SCHEMA}}" in system) is False
    assert ("format" in script.requests[0].kwargs.get("text", {})) is enforced
    expected = "Plain page text." if fmt == "text" else "Bread prices rose."
    assert result["transcription"] == expected
    assert "error" not in result


def test_a_model_without_vision_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _manager(monkeypatch, [_answer()], name="o3-mini")
    assert "may not support image input" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _manager(monkeypatch, [_answer()], name="o3-mini", supports_vision=True)
    assert "may not support image input" not in caplog.text


# ============================================================================
# Results and retries
# ============================================================================
def test_success_entry_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    env = helpers.env()
    manager, _ = _manager(monkeypatch, [_answer()], env=env)

    result = _transcribe(manager)

    assert result == {
        "image": NAME,
        "sequence_number": 1,
        "transcription": "Bread prices rose.",
        "processing_time": result["processing_time"],
        "provider": "openai",
    }
    assert env.usage.get("transcription") == Usage(100, 20, 120)


def test_invalid_json_is_retried_with_the_rule_backoff(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(
        monkeypatch, ["not json", _answer()], response_format="json"
    )

    result = _transcribe(manager)

    assert len(script.requests) == 2
    assert result["transcription"] == "Bread prices rose."
    assert result["schema_retries"] == {
        "validation_failure": 1,
        "no_transcribable_text": 0,
        "transcription_not_possible": 0,
    }
    assert 0.5 <= sleeps[0] <= 1.5


def test_exhausted_validation_marks_the_page_failed(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(monkeypatch, ["still not json"])

    result = _transcribe(manager)

    assert len(script.requests) == 4
    assert result["error_type"] == "schema_validation_exhausted"
    assert result["error"] == "schema validation retries exhausted: invalid JSON"
    assert result["schema_retries"]["validation_failure"] == 3
    assert result["transcription"] == "still not json"


def test_a_recoverable_answer_survives_exhausted_validation(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    partial = {"transcription": "Recovered page."}
    manager, _ = _manager(monkeypatch, [partial])

    result = _transcribe(manager)

    assert "error" not in result
    assert result["transcription"] == "Recovered page."
    assert result["schema_retries"]["validation_failure"] == 3


def test_transcription_not_possible_is_retried_then_kept(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    flagged = _answer(
        transcription=None,
        transcription_not_possible=True,
        image_analysis="Blurred scan.",
    )
    manager, script = _manager(monkeypatch, [flagged])

    result = _transcribe(manager)

    assert len(script.requests) == 4
    assert result["schema_retries"]["transcription_not_possible"] == 3
    assert result["transcription"] == (
        f"[{NAME}: transcription not possible — Blurred scan.]"
    )
    assert "error" not in result


def test_no_transcribable_text_is_not_retried_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blank = _answer(transcription=None, no_transcribable_text=True)
    manager, script = _manager(monkeypatch, [blank])

    result = _transcribe(manager)

    assert len(script.requests) == 1
    assert result["transcription"] == (
        f"[{NAME}: no transcribable text — Single column.]"
    )


@pytest.mark.parametrize(
    "sentinel", ["[no transcribable text]", "```\n[no transcribable text]\n```"]
)
def test_plain_text_sentinels_use_the_flag_rules(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps, sentinel: str
) -> None:
    retry = helpers.rules("transcription", no_transcribable_text=(True, 2, 0.0, 1.0))
    manager, script = _manager(
        monkeypatch,
        [sentinel, "Real transcribed text."],
        retry=retry,
        response_format="text",
    )

    result = _transcribe(manager)

    assert len(script.requests) == 2
    assert result["transcription"] == "Real transcribed text."
    assert result["schema_retries"]["no_transcribable_text"] == 1


def test_an_exhausted_plain_sentinel_is_a_status_message(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(
        monkeypatch, ["[transcription not possible]"], response_format="text"
    )

    result = _transcribe(manager)

    assert len(script.requests) == 4
    assert result["transcription"] == f"[{NAME}: transcription not possible]"
    assert "error" not in result


def test_an_empty_answer_is_retried_in_run(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    manager, script = _manager(monkeypatch, ["", _answer()])

    result = _transcribe(manager)

    assert len(script.requests) == 2
    assert "error" not in result
    assert sleeps == [0.5]


def test_persistent_empty_answers_fail_the_page(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env()
    manager, script = _manager(monkeypatch, [""], env=env)

    result = _transcribe(manager)

    assert len(script.requests) == 3
    assert result["error_type"] == "api_failure"
    assert "empty content" in result["error"]
    assert env.usage.get("transcription").total_tokens == 360


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (
            message(
                response_metadata={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                }
            ),
            "content_filter",
        ),
        (message(additional_kwargs={"refusal": "I cannot help."}), "refusal"),
    ],
)
def test_a_filtered_or_refused_page_stops_after_one_call(
    monkeypatch: pytest.MonkeyPatch, response: AIMessage, kind: str
) -> None:
    env = helpers.env()
    manager, script = _manager(monkeypatch, [response, _answer()], env=env)

    result = _transcribe(manager)

    assert len(script.requests) == 1
    assert result["error_type"] == kind
    assert result["transcription"].startswith("[transcription error: ")
    assert "schema_retries" not in result
    assert env.usage.get("transcription").total_tokens == 120


def test_a_truncated_answer_keeps_the_validation_retries(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    truncated = message(
        '{"transcription": "cut',
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    )
    manager, script = _manager(monkeypatch, [truncated, _answer()])

    result = _transcribe(manager)

    assert len(script.requests) == 2
    assert "error" not in result


def test_an_api_error_after_the_ladder_fails_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = _manager(monkeypatch, [StatusError(401)])

    result = _transcribe(manager)

    assert result["error_type"] == "api_failure"
    assert result["error"] == "status 401"
    assert result["transcription"] == "[transcription error: status 401]"


def test_openrouter_tool_call_answers_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, script = _manager(
        monkeypatch,
        [json.dumps(_answer())],
        name="anthropic/claude-sonnet-4.5",
        provider="openrouter",
    )

    result = _transcribe(manager)

    assert script.requests[0].structured is not None
    assert result["transcription"] == "Bread prices rose."


# ============================================================================
# Page text
# ============================================================================
@pytest.mark.parametrize(
    ("text", "data", "plain", "expected"),
    [
        ("Just text.", None, False, "Just text."),
        ('```json\n{"transcription": "Fenced."}\n```', None, False, "Fenced."),
        ('{"transcription": "A"} trailing', None, False, "A"),
        ("{broken", None, False, "{broken"),
        ('{"other": 1}', None, False, '{"other": 1}'),
        ('{"transcription": null}', None, False, f"[{NAME}: no transcribable text]"),
        ("ignored", {"transcription": "From data."}, False, "From data."),
        ("```\nPlain.\n```", None, True, "Plain."),
        ("[no transcribable text]", None, True, f"[{NAME}: no transcribable text]"),
    ],
)
def test_page_text(text: str, data: Any, plain: bool, expected: str) -> None:
    assert page_text(text, NAME, data, plain=plain) == expected


def test_page_text_flags_name_the_image_and_shorten_the_analysis() -> None:
    analysis = "word " * 40
    text = page_text(
        "", "", {"no_transcribable_text": True, "image_analysis": analysis}
    )
    assert text.startswith("[unknown_image: no transcribable text — word")
    assert text.endswith("...]")
    assert len(text) < 150
