"""Chat attempts are recorded with the body that came over HTTP.

The chat models are the ones :func:`build_chat_model` builds; a
``MockTransport`` answers behind their httpx clients (``tests.llm.network``),
so no request leaves the machine. Each record's body must equal the
transport's body byte for byte, and the record of an attempt must be stored
before the next attempt is sent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import google.genai._api_client as genai_api_client
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from autoexcerpter.common.capture import capture_attempt, use_recorder
from autoexcerpter.common.structured import StructuredRequest, StructuredResult
from autoexcerpter.llm import PhaseModel
from autoexcerpter.llm.caller import CallEnv, ModelCaller
from autoexcerpter.llm.client import build_chat_model
from autoexcerpter.settings import Timeouts
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.network import MockNetwork, Recorder, bodies

IMAGE = "data:image/jpeg;base64," + "QUJD" * 600
MESSAGES = [
    SystemMessage(content="Transcribe."),
    HumanMessage(
        content=[
            {"type": "text", "text": "page"},
            {"type": "image_url", "image_url": {"url": IMAGE}},
        ]
    ),
]

_USAGE = (
    b'"usage": {"input_tokens": 1500, "output_tokens": 4, "total_tokens": 1504,'
    b' "input_tokens_details": {"cached_tokens": 0},'
    b' "output_tokens_details": {"reasoning_tokens": 0}}'
)


def _openai_body(content: bytes) -> bytes:
    return (
        b'{"id": "resp_1",  "object": "response", "created_at": 1700000000,'
        b' "status": "completed", "model": "gpt-5.6-luna", "output": [{"type":'
        b' "message", "id": "msg_1", "status": "completed", "role": "assistant",'
        b' "content": [' + content + b"]}], " + _USAGE + b"}\n"
    )


OPENAI_OK = _openai_body(
    b'{"type": "output_text", "text": "Gr\xc3\xbc\xc3\x9fe aus  Z\xc3\xbcrich",'
    b' "annotations": []}'
)
OPENAI_REFUSAL = _openai_body(b'{"type": "refusal", "refusal": "I cannot help."}')
OPENAI_MALFORMED = (
    b'{"id": "resp_2", "object": "response", "status": "completed",'
    b' "output": [{"type": "message", "content": null}]}'
)
SERVER_ERROR = (
    b'{"error": {"message": "The server had an error.", "type": "server_error",'
    b' "param": null, "code": null}}'
)
ANTHROPIC_OK = (
    b'{"id": "msg_1", "type": "message", "role": "assistant", "model":'
    b' "claude-sonnet-4-5", "content": [{"type": "text",'
    b' "text": "Gr\xc3\xbc\xc3\x9fe"}],'
    b' "stop_reason": "end_turn", "stop_sequence": null,'
    b' "usage": {"input_tokens": 1500, "output_tokens": 4}}'
)
GOOGLE_OK = (
    b'{\n  "candidates": [{"content": {"parts": [{"text": "Gr\xc3\xbc\xc3\x9fe"}],'
    b' "role": "model"}, "finishReason": "STOP", "index": 0}],\n'
    b'  "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 4,'
    b' "totalTokenCount": 1504},\n  "modelVersion": "gemini-3-flash"\n}\n'
)

TEXT = StructuredRequest(provider="openai", response_format="text")
OPENAI = helpers.phase(
    "gpt-5.6-luna", "openai", options={"reasoning": {"effort": "high"}}
)
ANTHROPIC = helpers.phase("claude-sonnet-4-5", "anthropic")
GOOGLE = helpers.phase("gemini-3-flash", "google")


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> MockNetwork:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    return MockNetwork(recorder).install(monkeypatch)


def _run(
    env: CallEnv,
    request: StructuredRequest = TEXT,
    phase: PhaseModel = OPENAI,
    recorder: Recorder | None = None,
    role: str = "transcription",
) -> StructuredResult:
    async def main() -> StructuredResult:
        caller = ModelCaller(role, phase, env)
        with use_recorder(recorder):
            return await caller.call(MESSAGES, request, label="p1")

    return asyncio.run(main())


class TestOpenAI:
    """The oracle cases on the Responses API."""

    def test_a_success_is_one_record_with_the_verbatim_body(
        self, network: MockNetwork, recorder: Recorder, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_OK)]

        result = _run(helpers.env(clock=sleeps.clock), recorder=recorder)

        assert result.text == "Grüße aus  Zürich"
        (record,) = recorder.records
        assert bodies(recorder.records) == [OPENAI_OK]
        assert record.raw_kind == "http"
        assert record.http_status == 200
        assert (record.provider, record.model, record.role) == (
            "openai",
            "gpt-5.6-luna",
            "transcription",
        )
        assert (record.attempt, record.label, record.error) == (1, "p1", None)
        assert record.usage is not None and record.usage["input_tokens"] == 1500
        assert record.latency is not None and record.latency >= 0
        assert network.violations == []

    def test_a_summary_attempt_carries_the_summary_role(
        self, network: MockNetwork, recorder: Recorder, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_OK)]

        _run(helpers.env(clock=sleeps.clock), recorder=recorder, role="summary")

        assert [record.role for record in recorder.records] == ["summary"]

    def test_the_request_parameters_carry_no_image(
        self, network: MockNetwork, recorder: Recorder, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_OK)]
        schema = {
            "name": "page",
            "strict": True,
            "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        }
        request = StructuredRequest("openai", "schema", schema=schema)

        _run(helpers.env(clock=sleeps.clock), request, recorder=recorder)

        params = recorder.records[0].request
        assert params["model"] == "gpt-5.6-luna"
        assert params["options"] == {"reasoning": {"effort": "high"}}
        assert params["response_format"] == "schema"
        assert params["schema"] == "page"
        assert params["invoke_kwargs"]["text"]["format"]["name"] == "page"
        assert "api_key_env" not in params
        assert "QUJD" not in json.dumps(params)

    def test_a_malformed_output_is_recorded_with_its_error(
        self, network: MockNetwork, recorder: Recorder, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_MALFORMED)]

        with pytest.raises(TypeError):
            _run(helpers.env(clock=sleeps.clock), recorder=recorder)

        assert bodies(recorder.records) == [OPENAI_MALFORMED]
        (record,) = recorder.records
        assert (record.attempt, record.http_status, record.error_type) == (
            1,
            200,
            "TypeError",
        )

    def test_a_refusal_is_one_record(
        self, network: MockNetwork, recorder: Recorder, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_REFUSAL)]

        _run(helpers.env(clock=sleeps.clock), recorder=recorder)

        assert bodies(recorder.records) == [OPENAI_REFUSAL]
        assert recorder.records[0].http_status == 200

    def test_a_500_then_a_200_is_two_records_stored_before_the_retry(
        self, network: MockNetwork, recorder: Recorder, sleeps: Sleeps
    ) -> None:
        network.answers = [(500, SERVER_ERROR), (200, OPENAI_OK)]

        result = _run(helpers.env(clock=sleeps.clock), recorder=recorder)

        assert result.ok
        assert bodies(recorder.records) == [SERVER_ERROR, OPENAI_OK]
        failed, answered = recorder.records
        assert (failed.http_status, failed.error_type) == (500, "InternalServerError")
        assert failed.raw_kind == "http"
        assert (answered.http_status, answered.attempt) == (200, 2)
        assert network.violations == []
        assert network.scoped == [True, True]

    def test_without_a_recorder_nothing_is_captured(
        self, network: MockNetwork, recorder: Recorder, sleeps: Sleeps
    ) -> None:
        network.answers = [(500, SERVER_ERROR), (200, OPENAI_OK)]

        result = _run(helpers.env(clock=sleeps.clock))

        assert result.text == "Grüße aus  Zürich"
        assert recorder.records == []
        assert network.scoped == [False, False]


def test_anthropic_sdk_clients_are_injected_copies_on_hooked_httpx_clients(
    network: MockNetwork, recorder: Recorder, sleeps: Sleeps
) -> None:
    """ChatAnthropic caches its SDK clients; the factory replaces both."""
    model: Any = build_chat_model(ANTHROPIC, Timeouts())
    assert "_client" in model.__dict__ and "_async_client" in model.__dict__
    network.answers = [(500, SERVER_ERROR), (200, ANTHROPIC_OK)]
    request = StructuredRequest(provider="anthropic", response_format="text")

    result = _run(helpers.env(clock=sleeps.clock), request, ANTHROPIC, recorder)

    assert result.text == "Grüße"
    assert bodies(recorder.records) == [SERVER_ERROR, ANTHROPIC_OK]
    assert [r.http_status for r in recorder.records] == [500, 200]
    assert network.violations == []


def test_google_client_args_hook_both_clients_and_keep_async_on_httpx(
    monkeypatch: pytest.MonkeyPatch,
    network: MockNetwork,
    recorder: Recorder,
    sleeps: Sleeps,
) -> None:
    """``client_args`` reach both httpx clients; the transport rules out aiohttp."""
    monkeypatch.setattr(genai_api_client, "has_aiohttp", True)
    model: Any = build_chat_model(GOOGLE, Timeouts())
    assert model.client._api_client._use_aiohttp() is False
    network.answers = [(200, GOOGLE_OK)]
    request = StructuredRequest(provider="google", response_format="text")

    result = _run(helpers.env(clock=sleeps.clock), request, GOOGLE, recorder)

    assert result.text == "Grüße"
    assert bodies(recorder.records) == [GOOGLE_OK]
    assert recorder.records[0].raw_kind == "http"


@pytest.mark.parametrize(
    ("phase", "body", "text"),
    [
        (OPENAI, OPENAI_OK, "Grüße aus  Zürich"),
        (ANTHROPIC, ANTHROPIC_OK, "Grüße"),
        (GOOGLE, GOOGLE_OK, "Grüße"),
    ],
    ids=["openai", "anthropic", "google"],
)
def test_sync_calls_are_captured_and_still_parsed(
    network: MockNetwork, phase: PhaseModel, body: bytes, text: str
) -> None:
    """The hook reads the body first; the SDK's own read still parses it."""
    model = build_chat_model(phase, Timeouts())
    network.answers = [(200, body)]

    with capture_attempt() as scope:
        answer = model.invoke("page")

    assert [captured.body for captured in scope.responses] == [body]
    assert answer.text == text


def test_a_client_without_hooks_records_the_sdk_dump(
    monkeypatch: pytest.MonkeyPatch, recorder: Recorder, sleeps: Sleeps
) -> None:
    """A response no hook saw is stored as the SDK object's JSON dump."""
    helpers.install(monkeypatch, ["Bread prices rose."])

    _run(helpers.env(clock=sleeps.clock), recorder=recorder)

    (record,) = recorder.records
    assert record.raw_kind == "sdk"
    assert record.http_status is None
    dump = json.loads(record.raw or "")
    assert (dump["type"], dump["content"]) == ("ai", "Bread prices rose.")
