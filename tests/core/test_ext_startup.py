"""The extension seam as the front ends and the model caller use it.

- ``ext.startup`` receives the settings and the parsed arguments from the
  CLI and from the ui, dry runs included; a SettingsError it raises is a
  configuration error.
- ``ModelCaller`` builds its client through an attempt that is not entered,
  sets the model in the identity, and asks ``on_error`` about failures on
  entry.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

import autoexcerpter.run as core
from autoexcerpter import __main__ as entry
from autoexcerpter import ext
from autoexcerpter.cli.command import summary_fields
from autoexcerpter.common.retry import Decision
from autoexcerpter.common.structured import StructuredRequest
from autoexcerpter.common.testing import ScriptedPrompter
from autoexcerpter.llm import ModelCaller
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.settings import Settings, SettingsError
from autoexcerpter.spec import RunSpec
from autoexcerpter.ui import app
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import RecordingExt, StatusError
from tests.ui.test_app import _answers, _fake_run

MakePdf = Callable[..., Path]
MESSAGES = [SystemMessage(content="Transcribe."), HumanMessage(content="page")]
REQUEST = StructuredRequest(
    provider="openai",
    response_format="json",
    schema={"name": "s", "schema": {"type": "object", "required": ["a"]}},
)


class _Startup:
    """Stand-in for ``ext.startup`` that records its calls and may raise."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[RunSpec, Settings, argparse.Namespace]] = []

    def __call__(
        self, spec: RunSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        self.calls.append((spec, settings, args))
        if self.error is not None:
            raise self.error


def _install(monkeypatch: pytest.MonkeyPatch, startup: _Startup) -> _Startup:
    monkeypatch.setattr(ext, "startup", startup)
    return startup


async def _empty_run(*_args: Any, **_kwargs: Any) -> core.RunResult:
    return core.RunResult(items=())


def _no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(*_args: Any, **_kwargs: Any) -> core.RunResult:
        raise AssertionError("core.run must not be called")

    monkeypatch.setattr(core, "run", run)


def _json_lines(stdout: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


def _console() -> Console:
    return Console(file=io.StringIO(), width=100, color_system=None)


def _text(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def test_the_cli_passes_the_settings_and_the_parsed_arguments(
    make_pdf: MakePdf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = _install(monkeypatch, _Startup())
    monkeypatch.setattr(core, "run", _empty_run)
    pdf = make_pdf("doc.pdf")

    code = entry.main(["run", "--input", str(pdf), "--output", str(tmp_path / "o")])

    assert code == 0
    ((spec, settings, args),) = startup.calls
    assert isinstance(settings, Settings)
    assert isinstance(args, argparse.Namespace)
    assert args.command == "run"
    assert Path(args.input) == pdf
    assert isinstance(spec, RunSpec)


def test_a_settings_error_from_startup_exits_2_with_one_json_line(
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install(monkeypatch, _Startup(SettingsError("extension block is invalid")))
    _no_run(monkeypatch)

    code = entry.main(["run", "--input", str(make_pdf("doc.pdf")), "--json"])

    captured = capsys.readouterr()
    assert code == 2
    assert _json_lines(captured.out) == [{"dry_run": False, **summary_fields(total=1)}]
    assert "extension block is invalid" in captured.err


def test_a_startup_error_counts_the_planned_and_the_skipped_items(
    make_pdf: MakePdf,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install(monkeypatch, _Startup(SettingsError("extension block is invalid")))
    _no_run(monkeypatch)
    (tmp_path / "in").mkdir()
    make_pdf("in/Open.pdf")
    make_pdf("in/Done.pdf")
    out = tmp_path / "out"
    out.mkdir()
    ItemPaths.for_item("Done", out).transcription.write_text("x", encoding="utf-8")

    code = entry.main(
        [
            "run",
            "--input",
            str(tmp_path / "in"),
            "--output",
            str(out),
            "--no-summarize",
            "--all",
            "--json",
        ]
    )

    assert code == 2
    assert _json_lines(capsys.readouterr().out) == [
        {"dry_run": False, **summary_fields(total=1, skipped=1)}
    ]


def test_startup_is_called_once_for_a_dry_run(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    startup = _install(monkeypatch, _Startup())
    _no_run(monkeypatch)

    code = entry.main(["run", "--input", str(make_pdf("doc.pdf")), "--dry-run"])

    assert code == 0
    ((spec, _settings, _args),) = startup.calls
    assert spec.run.dry_run


def test_a_settings_error_from_startup_in_a_dry_run_exits_2(
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install(monkeypatch, _Startup(SettingsError("flags do not combine")))
    _no_run(monkeypatch)

    code = entry.main(
        ["run", "--input", str(make_pdf("doc.pdf")), "--dry-run", "--json"]
    )

    captured = capsys.readouterr()
    assert code == 2
    (line,) = _json_lines(captured.out)
    assert line["dry_run"] is True
    assert line["items_total"] == 1
    assert "flags do not combine" in captured.err


def test_ui_execute_threads_its_arguments_to_run_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []

    def run_app(settings_path: Any = None, **kwargs: Any) -> int:
        seen.append((settings_path, kwargs))
        return 0

    monkeypatch.setattr(app, "run_app", run_app)
    args = argparse.Namespace(command="ui", settings=tmp_path / "s.yaml")

    assert app.execute(args) == 0
    assert seen == [(tmp_path / "s.yaml", {"args": args})]


def test_the_ui_passes_the_settings_and_its_arguments(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    startup = _install(monkeypatch, _Startup())
    seen: list[core.RunPlan] = []
    monkeypatch.setattr(core, "run", _fake_run(seen, success=True))
    pdf = make_pdf("book.pdf", num_pages=2)
    args = argparse.Namespace(command="ui", settings=None)

    code = app.run_app(
        args=args,
        prompter=ScriptedPrompter(_answers(pdf, "run")),
        console=_console(),
        require_tty=False,
    )

    assert code == 0
    ((spec, settings, passed),) = startup.calls
    assert spec == seen[0].spec
    assert isinstance(settings, Settings)
    assert passed is args


def test_a_settings_error_from_startup_in_the_ui_exits_2(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _install(monkeypatch, _Startup(SettingsError("extension block is invalid")))
    _no_run(monkeypatch)
    console = _console()

    code = app.run_app(
        prompter=ScriptedPrompter(_answers(make_pdf("book.pdf"), "run")),
        console=console,
        require_tty=False,
    )

    assert code == 2
    assert "Error: extension block is invalid" in _text(console)


def test_attempt_build_builds_the_client_without_entering() -> None:
    built: list[ext.Identity] = []

    def build(identity: ext.Identity) -> str:
        built.append(identity)
        return f"client {len(built)}"

    attempt = ext.attempt(
        "summary", "openai", settings=Settings(), build_client=build, model="m"
    )

    assert attempt.build() == "client 1"
    assert attempt.client == "client 1"
    assert not attempt.released
    assert built == [ext.Identity("openai", "OPENAI_API_KEY", "summary", model="m")]


def test_the_caller_builds_its_client_through_an_attempt_with_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    script = helpers.install(monkeypatch, [{"a": 1}])

    caller = ModelCaller("summary", helpers.phase("gpt-5-mini"), helpers.env())

    (opened,) = recorder.opened
    assert recorder.attempts == []
    assert opened.client is caller.model
    assert not opened.released
    assert opened.identity.model == "gpt-5-mini"
    assert len(script.constructions) == 1

    asyncio.run(caller.call(MESSAGES, REQUEST, label="p1"))

    (entered,) = recorder.attempts
    assert entered.identity == ext.Identity(
        "openai", "OPENAI_API_KEY", "summary", model="gpt-5-mini"
    )
    assert len(script.constructions) == 1


def test_a_missing_key_variable_fails_at_construction() -> None:
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        ModelCaller("transcription", helpers.phase(), helpers.env())


def _failing_entry(
    caller: ModelCaller, error: Exception, times: int
) -> Callable[[ext.Identity], Any]:
    real = caller.client_for
    left = [times]

    def client_for(identity: ext.Identity) -> Any:
        if left[0] > 0:
            left[0] -= 1
            raise error
        return real(identity)

    return client_for


def test_on_error_decides_about_a_failure_on_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RecordingExt(decision=Decision.FAIL).install(monkeypatch)
    script = helpers.install(monkeypatch, [{"a": 1}])
    caller = ModelCaller("transcription", helpers.phase(), helpers.env())
    error = StatusError(503)
    monkeypatch.setattr(caller, "client_for", _failing_entry(caller, error, 1))

    with pytest.raises(StatusError):
        asyncio.run(caller.call(MESSAGES, REQUEST, label="p1"))

    assert recorder.errors == [error]
    (entered,) = recorder.attempts
    assert entered.released
    assert entered.errors == [error]
    assert script.requests == []


def test_a_retried_failure_on_entry_reaches_the_next_attempt(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    script = helpers.install(monkeypatch, [{"a": 1}])
    caller = ModelCaller(
        "transcription", helpers.phase(), helpers.env(clock=sleeps.clock)
    )
    error = StatusError(503)
    monkeypatch.setattr(caller, "client_for", _failing_entry(caller, error, 1))

    result = asyncio.run(caller.call(MESSAGES, REQUEST, label="p1"))

    assert result.ok and result.data == {"a": 1}
    assert recorder.errors == [error]
    assert len(recorder.attempts) == 2
    assert all(attempt.released for attempt in recorder.attempts)
    assert len(script.requests) == 1


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> Sleeps:
    """Record ``asyncio.sleep`` delays and advance a fake clock instead."""
    recorded = Sleeps()
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, result: object = None) -> object:
        recorded.append(delay)
        recorded.now += max(0.0, delay)
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded
