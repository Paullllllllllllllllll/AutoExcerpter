"""ModelCaller: attempt contexts, the retry ladder, usage totals and waits."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import autoexcerpter.llm.caller as caller_module
from autoexcerpter.common.retry import Decision
from autoexcerpter.common.structured import StructuredRequest, StructuredResult
from autoexcerpter.common.testing import ScriptedChatModel
from autoexcerpter.common.usage import Usage, UsageTotals
from autoexcerpter.events import Waiting
from autoexcerpter.ext import Identity
from autoexcerpter.llm.caller import CallEnv, ModelCaller, collect_usage
from autoexcerpter.settings import RetrySettings
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import (
    TRANSCRIPTION_ANSWER,
    RecordingExt,
    StatusError,
)
from tests.recording_events import RecordingEvents

MESSAGES = [SystemMessage(content="Transcribe."), HumanMessage(content="page")]
ANSWER_USAGE = Usage(input_tokens=100, output_tokens=20, total_tokens=120)


def _caller(env: CallEnv | None = None, **phase_fields: Any) -> ModelCaller:
    return ModelCaller(
        "transcription", helpers.phase(**phase_fields), env or helpers.env()
    )


def _request(fmt: str = "json", retries: int = 0) -> StructuredRequest:
    return StructuredRequest(
        provider="openai",
        response_format=fmt,  # type: ignore[arg-type]
        schema={"name": "s", "schema": {"type": "object", "required": ["a"]}},
        validation_retries=retries,
    )


def _call(caller: ModelCaller, request: StructuredRequest) -> StructuredResult:
    return asyncio.run(caller.call(MESSAGES, request, label="Transcription for p1"))


def test_a_failed_then_retried_call_opens_one_attempt_per_try(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [StatusError(503), {"a": 1}])
    caller = _caller(helpers.env(clock=sleeps.clock))

    result = _call(caller, _request())

    assert result.ok and result.data == {"a": 1}
    assert len(recorder.attempts) == 2
    assert recorder.roles == ["transcription", "transcription"]
    assert all(attempt.released for attempt in recorder.attempts)
    assert [type(error) for error in recorder.errors] == [StatusError]
    assert recorder.commits == [ANSWER_USAGE]
    assert recorder.attempts[0].identity == Identity(
        "openai", "OPENAI_API_KEY", "transcription", model="gpt-5.6-terra"
    )
    assert sleeps[0] >= 1.0


def test_validation_retries_are_attempts_of_their_own(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, ["not json", {"a": 1}])
    env = helpers.env()

    result = _call(_caller(env), _request(retries=1))

    assert result.ok and result.validation_retries == 1
    assert len(recorder.attempts) == 2
    assert recorder.commits == [ANSWER_USAGE, ANSWER_USAGE]
    assert env.usage.get("transcription") == ANSWER_USAGE + ANSWER_USAGE


def test_usage_recovered_from_a_failed_attempt_reaches_the_totals(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    usage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    helpers.install(monkeypatch, [StatusError(500, usage), {"a": 1}])
    env = helpers.env(clock=sleeps.clock)

    _call(_caller(env), _request())

    assert env.usage.roles == ("transcription",)
    assert env.usage.get("transcription") == ANSWER_USAGE + Usage(7, 3, 10)


def test_collect_usage_counts_the_calls_of_its_block_only(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    helpers.install(monkeypatch, [{"a": 1}, {"a": 2}, {"a": 3}])
    env = helpers.env()
    caller = _caller(env)

    async def main() -> tuple[UsageTotals, UsageTotals]:
        with collect_usage() as first:
            await caller.call(MESSAGES, _request(), label="p1")
        await caller.call(MESSAGES, _request(), label="outside")

        async def page() -> UsageTotals:
            with collect_usage() as totals:
                await caller.call(MESSAGES, _request(), label="p2")
            return totals

        second = await asyncio.create_task(page())
        return first, second

    first, second = asyncio.run(main())

    assert first.get("transcription") == ANSWER_USAGE
    assert second.get("transcription") == ANSWER_USAGE
    assert env.usage.get("transcription") == ANSWER_USAGE + ANSWER_USAGE + ANSWER_USAGE


def test_an_enclosing_block_counts_the_calls_of_nested_blocks(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    helpers.install(monkeypatch, [{"a": 1}, {"a": 2}, {"a": 3}])
    caller = _caller()

    async def page(label: str) -> UsageTotals:
        with collect_usage() as totals:
            await caller.call(MESSAGES, _request(), label=label)
        return totals

    async def main() -> tuple[UsageTotals, UsageTotals, UsageTotals]:
        with collect_usage() as outer:
            await caller.call(MESSAGES, _request(), label="outer")
            inner = await page("p1")
            child = await asyncio.create_task(page("p2"))
        return outer, inner, child

    outer, inner, child = asyncio.run(main())

    assert outer.get("transcription") == ANSWER_USAGE + ANSWER_USAGE + ANSWER_USAGE
    assert inner.get("transcription") == ANSWER_USAGE
    assert child.get("transcription") == ANSWER_USAGE


def test_the_on_error_decision_drives_the_ladder(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt(decision=Decision.FAIL).install(monkeypatch)
    script = helpers.install(monkeypatch, [StatusError(503), {"a": 1}])

    with pytest.raises(StatusError):
        _call(_caller(helpers.env(clock=sleeps.clock)), _request())

    assert len(recorder.attempts) == 1
    assert len(script.requests) == 1
    assert sleeps == []


def test_a_switch_key_decision_retries_at_once(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt(decision=Decision.SWITCH_KEY).install(monkeypatch)
    events = RecordingEvents()
    helpers.install(monkeypatch, [StatusError(429), {"a": 1}])
    env = helpers.env(clock=sleeps.clock, events=events)

    result = _call(_caller(env), _request())

    assert result.ok
    assert len(recorder.attempts) == 2
    (waiting,) = events.of("waiting")
    assert waiting.seconds == 0.0
    assert "switch_key" in waiting.reason


def test_a_client_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [StatusError(400), {"a": 1}])

    with pytest.raises(StatusError):
        _call(_caller(), _request())

    assert len(recorder.attempts) == 1
    assert recorder.commits == []


def test_retries_stop_at_the_attempt_limit(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    retry = RetrySettings(max_attempts=3)
    recorder = RecordingExt().install(monkeypatch)
    events = RecordingEvents()
    helpers.install(monkeypatch, [StatusError(429)])
    env = helpers.env(retry, clock=sleeps.clock, events=events)

    with pytest.raises(StatusError):
        _call(_caller(env), _request())

    assert len(recorder.attempts) == 3
    assert len(recorder.errors) == 3
    assert len(events.of("waiting")) == 2


def test_cancellation_releases_the_open_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [{"a": 1}])
    started = asyncio.Event()

    async def hang(*_args: Any) -> Any:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(caller_module, "default_invoke", hang)
    caller = _caller()

    async def scenario() -> None:
        task = asyncio.create_task(caller.call(MESSAGES, _request(), label="p"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(recorder.attempts) == 1
    assert recorder.attempts[0].released
    assert recorder.errors == []


def test_retry_waits_are_reported_as_events(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    events = RecordingEvents()
    env = helpers.env(clock=sleeps.clock, events=events)
    helpers.install(monkeypatch, [StatusError(503), {"a": 1}])

    _call(_caller(env), _request())

    (waiting,) = events.of("waiting")
    assert isinstance(waiting, Waiting)
    assert waiting.seconds == sleeps[0]
    assert "server_error" in waiting.reason
    assert "Transcription for p1" in waiting.reason


def test_every_attempt_passes_the_provider_rate_limiter(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(clock=sleeps.clock)
    helpers.install(monkeypatch, [StatusError(503), {"a": 1}])

    _call(_caller(env), _request())

    limiter = env.limiters.get("openai")
    assert limiter.total_requests == 2
    assert "anthropic" not in env.limiters


def test_the_limiter_is_keyed_by_the_resolved_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers.install(monkeypatch, [{"a": 1}])
    env = helpers.env()
    caller = ModelCaller("summary", helpers.phase("gpt-5-mini", None), env)

    _call(caller, _request())

    assert caller.provider == "openai"
    assert env.limiters.get("openai").total_requests == 1


def test_one_client_per_caller_and_one_more_per_other_key_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = helpers.install(monkeypatch, [{"a": 1}])
    caller = _caller()

    same = caller.client_for(Identity("openai", "OPENAI_API_KEY", "transcription"))
    other = caller.client_for(Identity("openai", "SECOND_KEY", "transcription"))

    assert same is caller.model
    assert other is not caller.model
    assert caller.client_for(Identity("openai", "SECOND_KEY", "x")) is other
    assert [c.get("api_key_env") for c in script.constructions] == [
        "OPENAI_API_KEY",
        "SECOND_KEY",
    ]


def test_an_attempt_on_another_client_rebuilds_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = helpers.install(monkeypatch, [json.dumps(TRANSCRIPTION_ANSWER)])
    caller = _caller(name="claude-sonnet-4-5", provider="anthropic")
    second = ScriptedChatModel(
        responder=script.respond, recorder=script.record, tag="second"
    )

    def other_client(identity: Identity) -> Any:
        return second

    monkeypatch.setattr(caller, "client_for", other_client)
    request = StructuredRequest(
        provider="anthropic",
        response_format="schema",
        schema={"name": "s", "schema": {"type": "object"}},
    )

    result = _call(caller, request)

    assert result.enforced
    (sent,) = script.requests
    assert sent.tag == "second"
    assert sent.structured is not None and sent.structured.method == "json_schema"
