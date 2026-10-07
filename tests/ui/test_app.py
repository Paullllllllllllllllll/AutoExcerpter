"""The ui subcommand: terminal check, settings, quit, copy, dry run, run, Ctrl+C."""

from __future__ import annotations

import argparse
import dataclasses
import io
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import autoexcerpter.run as core
from autoexcerpter import ext
from autoexcerpter.common.testing import KEEP, ScriptedPrompter
from autoexcerpter.common.wizard import BACK
from autoexcerpter.events import ItemFinished, ItemStarted, PageDone, RunEvents
from autoexcerpter.settings import SettingsError
from autoexcerpter.spec import parse
from autoexcerpter.ui import app

INTERRUPT = object()


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Interrupting(ScriptedPrompter):
    """Raise KeyboardInterrupt where the script holds ``INTERRUPT``."""

    def _take(self, message: str) -> Any:
        answer = super()._take(message)
        if answer is INTERRUPT:
            raise KeyboardInterrupt
        return answer


def _console() -> Console:
    return Console(file=io.StringIO(), width=100, color_system=None)


def _text(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def _answers(pdf: Path, *review: Any, run_options: Any = KEEP) -> list[Any]:
    return [
        str(pdf),
        "beside",
        True,
        "md",
        KEEP,
        "auto",
        "none",
        "openai",
        "gpt-5.6-luna",
        KEEP,
        KEEP,
        "recommended",
        "same",
        "recommended",
        run_options,
        KEEP,
        *review,
    ]


class _Calls:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(args)


@pytest.fixture
def pdf(make_pdf: Callable[..., Path]) -> Path:
    return make_pdf("book.pdf", num_pages=2)


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    calls = _Calls()
    monkeypatch.setattr(ext, "startup", calls)
    return calls


def _no_run(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(*args: Any, **kwargs: Any) -> core.RunResult:
        raise AssertionError("core.run must not be called")

    monkeypatch.setattr(core, "run", run)


def _launch(
    answers: Sequence[Any],
    *,
    prompter: ScriptedPrompter | None = None,
    **kwargs: Any,
) -> tuple[int, Console, ScriptedPrompter]:
    console = _console()
    scripted = prompter if prompter is not None else ScriptedPrompter(answers)
    code = app.run_app(prompter=scripted, console=console, require_tty=False, **kwargs)
    return code, console, scripted


@pytest.mark.parametrize("stream", ["stdin", "stdout"])
def test_a_non_interactive_stream_is_refused_before_reading_settings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    stream: str,
) -> None:
    loads = _Calls()
    monkeypatch.setattr(app, "load", loads)
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr(sys, stream, io.StringIO())
    settings = tmp_path / "settings.yaml"
    settings.write_text("defaults: {}\n", encoding="utf-8")
    code = app.execute(argparse.Namespace(command="ui", settings=settings))
    assert code == 2
    assert loads.calls == []
    assert "autoexcerpter run" in capsys.readouterr().err


def test_a_settings_error_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    code, console, prompter = _launch([], settings_path=missing)
    assert code == 2
    assert f"Settings file not found: {missing}" in _text(console)
    assert prompter.lines == []


def test_an_invalid_defaults_block_exits_2(tmp_path: Path) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("defaults:\n  force: true\n", encoding="utf-8")
    code, console, _prompter = _launch([], settings_path=settings)
    assert code == 2
    assert "force" in _text(console)


def test_back_at_the_first_question_and_quit_exits_0(
    startup: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    code, _console, prompter = _launch([BACK, "quit"])
    assert code == 0
    assert "? Quit without running?" in prompter.lines
    assert startup.calls == []


def test_quit_on_the_review_exits_0(
    pdf: Path, startup: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    code, _console, _prompter = _launch(_answers(pdf, "quit"), environ={})
    assert code == 0
    assert startup.calls == []


def test_copy_prints_the_command_and_copies_it(
    pdf: Path, startup: _Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _no_run(monkeypatch)
    copied: list[tuple[list[str], bytes]] = []

    def clipboard(argv: Sequence[str], data: bytes) -> None:
        copied.append((list(argv), data))

    settings = tmp_path / "settings.yaml"
    settings.write_text("defaults:\n  concurrency: 4\n", encoding="utf-8")
    stdout = io.StringIO()
    code, console, _prompter = _launch(
        _answers(pdf, "copy"),
        settings_path=settings,
        stdout=stdout,
        clipboard=clipboard,
        environ={},
    )
    assert code == 0
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("autoexcerpter run --input ")
    assert str(pdf) in line
    assert "--concurrency 4" in line
    assert line.endswith(f"--settings {settings}")
    if sys.platform in ("win32", "darwin"):
        assert len(copied) == 1
        encoding = "utf-16" if sys.platform == "win32" else "utf-8"
        assert copied[0][1].decode(encoding) == line
        assert "Copied the command to the clipboard." in _text(console)
    assert startup.calls == []


def test_dry_run_calls_startup_and_shows_the_plan_without_writes(
    pdf: Path, startup: _Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _no_run(monkeypatch)
    before = sorted(path.name for path in tmp_path.iterdir())
    code, console, _prompter = _launch(
        _answers(pdf, "run", run_options=["openalex", "dry_run"]), environ={}
    )
    assert code == 0
    assert "Plan: 1 to process, 0 already complete" in _text(console)
    assert len(startup.calls) == 1
    assert startup.calls[0][0].run.dry_run
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_a_settings_error_from_startup_in_a_dry_run_exits_2(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def startup(*args: Any, **kwargs: Any) -> None:
        raise SettingsError("flags do not combine")

    monkeypatch.setattr(ext, "startup", startup)
    _no_run(monkeypatch)
    code, console, _prompter = _launch(
        _answers(pdf, "run", run_options=["openalex", "dry_run"]), environ={}
    )
    assert code == 2
    assert "Error: flags do not combine" in _text(console)
    assert "Plan: 1 to process" not in _text(console)


def test_a_key_missing_from_the_process_exits_2_before_any_work(
    pdf: Path, startup: _Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = sorted(path.name for path in tmp_path.iterdir())
    code, console, _prompter = _launch(
        _answers(pdf, "run"), environ={"OPENAI_API_KEY": "seen-by-the-review"}
    )
    assert code == 2
    text = _text(console)
    assert "Error: Cannot build the transcription model client" in text
    assert "OPENAI_API_KEY" in text
    assert "Run summary" not in text
    assert len(startup.calls) == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def _fake_run(seen: list[core.RunPlan], *, success: bool) -> Callable[..., Any]:
    async def run(run_plan: core.RunPlan, events: RunEvents) -> core.RunResult:
        seen.append(run_plan)
        planned = run_plan.to_process[0]
        events.item_started(ItemStarted(planned.name, "pdf", 1, 1))
        events.page_done(PageDone(planned.name, 0, "ok", 1, 2))
        events.page_done(PageDone(planned.name, 1, "ok", 2, 2))
        report = {"pages_ok": 2, "pages_failed": 0 if success else 1}
        events.item_finished(ItemFinished(planned.name, 1, 1, success, (), report))
        return core.RunResult(
            items=(core.ItemResult(planned.name, success, planned.paths, (), report),),
            seconds=1.5,
        )

    return run


@pytest.mark.parametrize(("success", "code"), [(True, 0), (False, 1)])
def test_run_shows_progress_and_the_summary(
    pdf: Path,
    startup: _Calls,
    monkeypatch: pytest.MonkeyPatch,
    success: bool,
    code: int,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen: list[core.RunPlan] = []
    monkeypatch.setattr(core, "run", _fake_run(seen, success=success))
    exit_code, console, _prompter = _launch(_answers(pdf, "run"))
    assert exit_code == code
    assert len(seen) == 1
    assert len(startup.calls) == 1
    assert startup.calls[0][0] == seen[0].spec
    assert seen[0].spec == parse(["--input", str(pdf)])
    output = _text(console)
    assert "[1/1] book (PDF)" in output
    assert "Run summary" in output
    assert ("1 complete, 0 incomplete" if success else "0 complete, 1") in output


def test_nothing_to_do_when_every_item_is_complete(
    pdf: Path, startup: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _no_run(monkeypatch)
    real = core.plan

    def plan(*args: Any, **kwargs: Any) -> core.RunPlan:
        result = real(*args, **kwargs)
        return dataclasses.replace(result, to_process=(), skipped=result.to_process)

    monkeypatch.setattr(core, "plan", plan)
    code, console, _prompter = _launch(_answers(pdf, "run"))
    assert code == 0
    assert "Nothing to do: 1 item already complete." in _text(console)
    assert startup.calls == []


def test_ctrl_c_in_the_wizard_exits_130(
    pdf: Path, startup: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_run(monkeypatch)
    prompter = _Interrupting([str(pdf), "beside", INTERRUPT])
    code, console, _prompter = _launch([], prompter=prompter)
    assert code == 130
    assert "Interrupted by the user." in _text(console)
    assert startup.calls == []


def test_ctrl_c_during_the_run_exits_130(
    pdf: Path, startup: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def run(*args: Any, **kwargs: Any) -> core.RunResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(core, "run", run)
    code, console, _prompter = _launch(_answers(pdf, "run"))
    assert code == 130
    assert "Interrupted by the user." in _text(console)
    assert len(startup.calls) == 1
