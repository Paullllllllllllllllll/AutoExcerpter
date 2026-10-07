"""A recorded guided session: a three-page PDF with summary, run to the end.

The scripted session goes through the real wizard and app, goes Back once,
gives the context as topics and runs from the review screen. The provider
classes and OpenAlex are the characterization fakes. The golden
``golden/session_pdf_summary.txt`` holds the prompter transcript followed by
the run screen and the summary table, with the temp path and the run time
replaced by placeholders. ``AE_UPDATE_GOLDEN=1`` rewrites it.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import io
import os
import re
import shlex
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Imported at module load: google.auth subclasses requests.Session at import,
# and the fake_openalex fixture replaces requests.Session.
import langchain_anthropic
import langchain_google_genai
import langchain_openai
import pytest
import requests
import yaml
from rich.console import Console

from autoexcerpter import ext
from autoexcerpter.cli.parser import build_parser
from autoexcerpter.common.testing import KEEP, ScriptedPrompter, WriteGuard
from autoexcerpter.common.wizard import BACK, command_display
from autoexcerpter.settings import Settings, load, resolve
from autoexcerpter.spec import OPTIONS, RunSpec
from autoexcerpter.ui import app
from tests.characterization.adapter import write_settings
from tests.characterization.fakes import DUMMY_KEYS, FakeLLM, FakeOpenAlex
from tests.characterization.golden import Normalizer
from tests.characterization.inputs import make_pdf

GOLDEN = Path(__file__).resolve().parent / "golden" / "session_pdf_summary.txt"
UPDATE_ENV = "AE_UPDATE_GOLDEN"
WIDTH = 100
TOPICS = "bread prices, real wages"
SETTINGS_DEFAULTS = {"model": "gpt-5.6-terra", "service_tier": "priority"}

PROVIDER_CLASSES = (
    (langchain_openai, "ChatOpenAI"),
    (langchain_anthropic, "ChatAnthropic"),
    (langchain_google_genai, "ChatGoogleGenerativeAI"),
)

_COMMAND_HEADER = "Equivalent command:"
_RUN_TIME_RE = re.compile(r"^Run time: \d+(?:\.\d+)?s$", re.MULTILINE)
_ASYNC_SLEEP = asyncio.sleep


def _no_sleep(_seconds: float) -> None:
    """Stand-in for ``time.sleep`` that returns at once."""


async def _yield_only(_delay: float, result: Any = None) -> Any:
    """Stand-in for ``asyncio.sleep`` that only yields to the event loop."""
    await _ASYNC_SLEEP(0)
    return result


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Install the fake provider classes, fake OpenAlex, keys and no-wait sleeps."""
    controller = FakeLLM()
    for module, name in PROVIDER_CLASSES:
        monkeypatch.setattr(module, name, controller.constructor(name))
    service = FakeOpenAlex()
    monkeypatch.setattr(requests, "Session", service.session)
    monkeypatch.setattr(requests, "get", service.get)
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(time, "sleep", _no_sleep)
    monkeypatch.setattr(asyncio, "sleep", _yield_only)
    return controller


def _settings_file(folder: Path) -> Path:
    """Write the characterization settings plus a defaults block."""
    path = write_settings(folder)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["defaults"] = dict(SETTINGS_DEFAULTS)
    path.write_text(yaml.safe_dump(data), encoding="utf-8", newline="\n")
    return path


def _answers(pdf: Path, output: Path) -> list[Any]:
    return [
        str(pdf),
        "folder",
        str(output),
        BACK,
        KEEP,
        KEEP,
        True,
        KEEP,
        KEEP,
        "text",
        TOPICS,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "recommended",
        "same",
        "recommended",
        KEEP,
        KEEP,
        "run",
    ]


def _command_block(lines: Sequence[str]) -> tuple[int, int]:
    """Return the slice of the last printed equivalent command."""
    start = len(lines) - 1 - lines[::-1].index(_COMMAND_HEADER) + 1
    end = start
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return start, end


def _split(line: str) -> list[str]:
    """Split a command line printed for the current platform into words."""
    if os.name != "nt":
        return shlex.split(line)
    words = shlex.split(line, posix=False)
    return [w[1:-1] if len(w) > 1 and w[0] == w[-1] == '"' else w for w in words]


def _spec_of(words: Sequence[str]) -> RunSpec:
    """Parse and resolve a printed command as ``autoexcerpter`` would."""
    assert list(words[:2]) == ["autoexcerpter", "run"]
    flags = OPTIONS.values(build_parser().parse_args(list(words[1:])))
    settings = load(flags.get("settings"), blocks=ext.settings_blocks())
    return resolve(settings, flags).spec


def _normalize(lines: list[str], words: Sequence[str], console: str, root: Path) -> str:
    """Replace the temp path and the run time; re-wrap the command.

    The command is wrapped again from its normalized words, quoted for
    Windows, so its line breaks do not depend on the temp path or the OS.
    """
    norm = Normalizer(root)
    start, end = _command_block(lines)
    shown = command_display(
        [norm.text(word) for word in words], width=WIDTH, windows=True
    )
    lines = [*lines[:start], *shown, *lines[end:]]
    text = norm.text("\n".join(lines) + "\n" + console)
    text = _RUN_TIME_RE.sub("Run time: <duration>", text)
    return "".join(line.rstrip() + "\n" for line in text.splitlines())


def test_recorded_session_runs_a_pdf_with_summary(
    tmp_path: Path,
    fakes: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    write_guard: WriteGuard,
) -> None:
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=3)
    output = tmp_path / "out"
    settings = _settings_file(tmp_path / "config")
    started: list[RunSpec] = []
    real_startup = ext.startup

    def startup(spec: RunSpec, *, settings: Settings, args: argparse.Namespace) -> None:
        started.append(spec)
        real_startup(spec, settings=settings, args=args)

    monkeypatch.setattr(ext, "startup", startup)
    monkeypatch.chdir(tmp_path)

    console = Console(
        file=io.StringIO(),
        width=WIDTH,
        color_system=None,
        force_terminal=False,
        legacy_windows=False,
        soft_wrap=True,
    )
    prompter = ScriptedPrompter(_answers(pdf, output))
    with prompter:
        code = app.run_app(
            settings, prompter=prompter, console=console, require_tty=False
        )
    screen = console.file
    assert isinstance(screen, io.StringIO)
    assert code == 0, prompter.transcript + screen.getvalue()

    lines = prompter.transcript.splitlines()
    start, end = _command_block(lines)
    words = _split(" ".join(line.strip() for line in lines[start:end]))
    assert lines[start:end] == command_display(words, width=WIDTH)
    assert len(started) == 1
    assert _spec_of(words) == started[0]

    actual = _normalize(lines, words, screen.getvalue(), tmp_path)
    if os.environ.get(UPDATE_ENV) == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(actual, encoding="utf-8", newline="\n")
        write_guard.consume(GOLDEN.parent)
        return
    if not GOLDEN.is_file():
        pytest.fail(f"No golden {GOLDEN.name}; run with {UPDATE_ENV}=1 to create.")
    expected = GOLDEN.read_bytes().decode("utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile="golden/session_pdf_summary.txt",
                tofile="session",
            )
        )
        pytest.fail(
            f"Session transcript changed (set {UPDATE_ENV}=1 to accept):\n{diff}"
        )
