"""Help invariants beyond the golden snapshots, and the ui dispatch."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

import autoexcerpter.ui
from autoexcerpter import __main__ as entry
from autoexcerpter.spec import IMAGE_DETAILS, OPTIONS, OUTPUT_FORMATS

WIDTHS = (80, 100)


def _help(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    with pytest.raises(SystemExit) as exc:
        entry.main([*argv, "--help"])
    assert exc.value.code == 0
    return capsys.readouterr().out


@pytest.fixture(params=WIDTHS, ids=[f"columns{w}" for w in WIDTHS])
def columns(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    width: int = request.param
    monkeypatch.setenv("COLUMNS", str(width))
    return width


def test_run_help_lists_every_option(
    columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, "run")
    tokens = set(re.findall(r"--[\w-]+", out))
    for option in OPTIONS:
        assert option.flag in tokens, option.flag


def test_run_help_never_splits_a_flag_or_a_value(
    columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, "run")
    for line in out.splitlines():
        assert len(line) <= columns, line
        assert not re.search(r"[\w-]-$", line), line
    words = set(re.findall(r"[\w./-]+", out))
    for value in (*OUTPUT_FORMATS, *IMAGE_DETAILS, "X_autoexcerpter/"):
        assert value in words, value


def test_top_level_help_matches_the_bare_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert entry.main([]) == 0
    bare = capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        entry.main(["--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out == bare


def _stand_in(monkeypatch: pytest.MonkeyPatch, calls: list[argparse.Namespace]) -> None:
    def execute(args: argparse.Namespace) -> int:
        calls.append(args)
        return 7

    module = types.ModuleType("autoexcerpter.ui.app")
    module.execute = execute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "autoexcerpter.ui.app", module)
    monkeypatch.setattr(autoexcerpter.ui, "app", module, raising=False)


def test_ui_dispatches_to_the_ui_app(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[argparse.Namespace] = []
    _stand_in(monkeypatch, calls)

    assert entry.main(["ui", "--settings", "other.yaml"]) == 7

    (args,) = calls
    assert args.command == "ui"
    assert args.settings == Path("other.yaml")


def test_ui_without_settings_leaves_the_option_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[argparse.Namespace] = []
    _stand_in(monkeypatch, calls)

    assert entry.main(["ui"]) == 7
    assert OPTIONS.values(calls[0]) == {}


def test_ui_rejects_run_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[argparse.Namespace] = []
    _stand_in(monkeypatch, calls)

    assert entry.main(["ui", "--input", "doc.pdf"]) == 2
    assert calls == []
    assert "unrecognized arguments" in capsys.readouterr().err


_PROBE = """\
import contextlib, io, json, sys
from autoexcerpter.__main__ import main
with contextlib.redirect_stdout(io.StringIO()):
    try:
        main(["run", "--help"])
    except SystemExit:
        pass
loaded = sorted(
    name for name in sys.modules
    if name.split(".")[0] in {"rich", "questionary"}
    or name.startswith("autoexcerpter.ui")
)
print(json.dumps(loaded))
"""


def test_run_imports_neither_the_ui_nor_its_libraries() -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=True,
        timeout=120,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == []
