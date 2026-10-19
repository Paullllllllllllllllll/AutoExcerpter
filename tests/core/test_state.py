"""The atomic JSON writer of the state directory."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.state as state_mod


def test_writes_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state_mod.write_json_atomic(path, {"a": 1, "b": "x"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": "x"}


def test_temp_name_is_per_process_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any) -> None:
        seen.append(str(src))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    state_mod.write_json_atomic(tmp_path / "state.json", {"a": 1})
    assert str(os.getpid()) in seen[0]
    assert seen[0].endswith(".tmp")


@pytest.mark.parametrize("error", [PermissionError, FileNotFoundError])
def test_a_failed_replace_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    path = tmp_path / "state.json"
    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src: Any, dst: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise error("transient")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    state_mod.write_json_atomic(path, {"a": 1})
    assert calls["n"] == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_no_temp_file_left_behind(tmp_path: Path) -> None:
    state_mod.write_json_atomic(tmp_path / "state.json", {"a": 1})
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
