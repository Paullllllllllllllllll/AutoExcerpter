"""Tests for common/testing.py: hermetic primitives and the scripted fake model."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from autoexcerpter.common.testing import (
    FakeRequest,
    LaunchEntry,
    ScriptedChatModel,
    WriteGuard,
    home_env,
    is_loopback_address,
    is_secret_env_var,
    launches_entry,
)

ENTRY = LaunchEntry(
    programs=("mytool",), scripts=("app/run.py",), modules=("mytool", "app.run")
)


class TestEnvironment:
    def test_secret_variables(self) -> None:
        assert is_secret_env_var("OPENAI_API_KEY")
        assert not is_secret_env_var("SERVICE_TOKEN")
        assert is_secret_env_var("service_token", ("SERVICE_",))

    def test_home_env_points_every_variable_at_home(self, tmp_path: Path) -> None:
        env = home_env(tmp_path)

        assert env["HOME"] == env["USERPROFILE"] == str(tmp_path)
        assert env["HOMEDRIVE"] + env["HOMEPATH"] == str(tmp_path)

    def test_loopback_classification(self) -> None:
        assert is_loopback_address(("127.0.0.1", 80))
        assert is_loopback_address(("::ffff:127.0.0.1", 80, 0, 0))
        assert is_loopback_address("/run/socket")
        assert not is_loopback_address(("192.0.2.1", 443))


class TestLaunchEntry:
    def test_entry_names_match(self) -> None:
        assert launches_entry(["mytool.exe", "--cli"], None, ENTRY)
        assert launches_entry(["python", "C:\\src\\app\\run.py"], None, ENTRY)
        assert launches_entry("python -m app.run --cli", None, ENTRY)
        assert launches_entry(["python", "-mmytool"], None, ENTRY)
        assert launches_entry(["python"], "mytool", ENTRY)

    def test_other_programs_pass(self) -> None:
        assert not launches_entry(["python", "-m", "pip", "list"], None, ENTRY)
        assert not launches_entry(["git", "status"], None, LaunchEntry())


class TestWriteGuard:
    def test_records_writes_outside_the_roots(self, tmp_path: Path) -> None:
        guard = WriteGuard()
        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside.txt"
        guard.start([allowed])

        guard.audit("open", (str(allowed / "a.txt"), "w", os.O_WRONLY))
        guard.audit("open", (str(outside), "r", os.O_RDONLY))
        guard.audit("open", (str(outside), "w", os.O_WRONLY))
        guard.audit("open", (str(outside), "a", os.O_APPEND))
        guard.audit("os.mkdir", (str(tmp_path / "__pycache__" / "x"), 0o777, None))

        assert guard.stop() == [os.path.normcase(str(outside))]
        guard.audit("open", (str(outside), "w", os.O_WRONLY))
        assert guard.violations == []

    def test_consume_takes_violations_under_a_path(self, tmp_path: Path) -> None:
        guard = WriteGuard()
        guard.start([tmp_path / "allowed"])
        guard.audit("open", (str(tmp_path / "a" / "x.txt"), "w", os.O_WRONLY))
        guard.audit("open", (str(tmp_path / "b.txt"), "w", os.O_WRONLY))

        taken = guard.consume(tmp_path / "a")

        assert taken == [os.path.normcase(str(tmp_path / "a" / "x.txt"))]
        assert guard.stop() == [os.path.normcase(str(tmp_path / "b.txt"))]


def _model(
    content: str, requests: list[FakeRequest], tag: Any = "instance"
) -> ScriptedChatModel:
    def respond(request: FakeRequest) -> AIMessage:
        return AIMessage(content=content, response_metadata={**request.notes})

    def record(request: FakeRequest) -> None:
        requests.append(request)
        request.notes["seen"] = len(requests)

    return ScriptedChatModel(responder=respond, recorder=record, tag=tag)


class TestScriptedChatModel:
    def test_invoke_records_then_responds(self) -> None:
        requests: list[FakeRequest] = []
        model = _model("hello", requests)

        reply = model.bind(temperature=0).invoke([HumanMessage("hi")], top_p=1)

        assert reply.content == "hello"
        assert reply.response_metadata == {"seen": 1}
        (request,) = requests
        assert request.kwargs == {"temperature": 0, "top_p": 1}
        assert request.tag == "instance"
        assert request.structured is None
        assert [m.content for m in request.messages] == ["hi"]

    def test_model_copy_keeps_tag_and_accumulates_fields(self) -> None:
        requests: list[FakeRequest] = []
        model = _model("x", requests)

        copied = model.model_copy(update={"max_tokens": 5})
        copied = copied.model_copy(update={"temperature": 0.5})
        copied.invoke("hi")

        assert model.bound == {}
        assert requests[0].bound == {"max_tokens": 5, "temperature": 0.5}
        assert requests[0].tag == "instance"

    def test_structured_function_calling_returns_a_tool_call(self) -> None:
        requests: list[FakeRequest] = []
        model = _model('{"a": 1}', requests)
        schema = {"title": "Answer", "type": "object"}

        result = model.with_structured_output(schema, include_raw=True).invoke("q")

        assert result["parsed"] == {"a": 1}
        assert result["parsing_error"] is None
        assert result["raw"].tool_calls[0]["name"] == "Answer"
        call = requests[0].structured
        assert call is not None
        assert (call.method, call.include_raw, call.title) == (
            "function_calling",
            True,
            "Answer",
        )

    def test_structured_json_schema_parses_the_content(self) -> None:
        model = _model('{"a": 1}', [])

        wrapper = model.with_structured_output({}, method="json_schema")

        assert wrapper.invoke("q") == {"a": 1}

    def test_unparsable_content_reports_the_error(self) -> None:
        model = _model("not json", [])

        result = model.with_structured_output({}, include_raw=True).invoke("q")

        assert result["parsed"] is None
        assert result["parsing_error"] is not None
        assert result["raw"].content == "not json"

    def test_async_paths(self) -> None:
        requests: list[FakeRequest] = []
        model = _model('{"a": 1}', requests)
        wrapper = model.with_structured_output({}, method="json_schema")

        async def run() -> tuple[Any, Any]:
            return await model.ainvoke("one"), await wrapper.ainvoke("two")

        reply, parsed = asyncio.run(run())

        assert reply.content == '{"a": 1}'
        assert parsed == {"a": 1}
        assert [r.structured is not None for r in requests] == [False, True]
