"""Ctrl+Break ends a run like Ctrl+C: the lease is released, the run exits 130.

The test starts ``tests/cli/interrupt_child.py`` in its own process group,
waits until the child has stored the first page of a three-page PDF and
sends it Ctrl+Break. The oracle: exit code 130 with the JSON line, no lease
left on the document, the run's end recorded, and a rerun right away that
transcribes only the two missing pages. Windows only.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.managers as managers_module
from autoexcerpter import __main__ as entry
from autoexcerpter.common.sqlite import read
from autoexcerpter.output.store import DATABASE_NAME

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Ctrl+Break is a Windows console event"
)

MakePdf = Callable[..., Path]
REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD = "tests/cli/interrupt_child.py"
TIMEOUT = 120.0


def _rows(db: Path, sql: str) -> list[tuple[Any, ...]]:
    def query(connection: sqlite3.Connection) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in connection.execute(sql).fetchall()]

    return read(db, query)


def _settings(folder: Path) -> Path:
    """Write settings whose rate limits never wait and state stays in *folder*."""
    path = folder / "settings.yaml"
    path.write_text(
        "rate_limits: [[10000, 1]]\n"
        f"state_dir: {json.dumps(str(folder / 'state'))}\n"
        "openalex: {email: '', api_key_env: ''}\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


async def _answer(payload: Any) -> dict[str, Any]:
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Text of page {payload.number}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _interrupt_child(pdf: Path, out: Path, settings: Path, log: Path) -> str:
    """Run the child until its first page is stored, send Ctrl+Break and
    return its stdout; the process's exit code is checked by the caller."""
    assert sys.platform == "win32"
    with log.open("w", encoding="utf-8") as errors:
        process = subprocess.Popen(
            [sys.executable, CHILD, str(pdf), str(out), str(settings)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            encoding="utf-8",
            env=dict(os.environ),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            stdout = process.stdout
            assert stdout is not None
            first: list[str] = []
            reader = threading.Thread(
                target=lambda: first.append(stdout.readline().strip()), daemon=True
            )
            reader.start()
            reader.join(TIMEOUT)
            if first != ["READY"]:
                process.kill()
                process.communicate(timeout=TIMEOUT)
                err = log.read_text(encoding="utf-8")
                raise AssertionError(f"child did not get ready: {first!r}\n{err}")
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
            rest, _ = process.communicate(timeout=TIMEOUT)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=TIMEOUT)
    assert process.returncode == 130, log.read_text(encoding="utf-8")
    return rest


def test_ctrl_break_releases_the_lease_and_a_rerun_resumes(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pdf = make_pdf(num_pages=3)
    out = tmp_path / "out"
    settings = _settings(tmp_path)
    log = tmp_path / "child.err"

    stdout = _interrupt_child(pdf, out, settings, log)

    assert "Interrupted by the user." in log.read_text(encoding="utf-8")
    (line,) = [line for line in stdout.splitlines() if line.startswith("{")]
    assert json.loads(line)["items_complete"] == 0
    db = out / DATABASE_NAME
    assert _rows(db, "SELECT lease_run_id, lease_until FROM documents") == [
        (None, None)
    ]
    ((child_run, exit_code),) = _rows(db, "SELECT run_id, exit_code FROM runs")
    assert exit_code == 130

    manager = MagicMock()
    manager.transcribe_payload = AsyncMock(side_effect=_answer)
    monkeypatch.setattr(
        managers_module, "TranscriptionManager", MagicMock(return_value=manager)
    )
    argv = [
        "run",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--settings",
        str(settings),
        "--no-summarize",
        "--json",
    ]
    assert entry.main(argv) == 0

    payload = json.loads(capsys.readouterr().out)
    assert (payload["items_complete"], payload["items_blocked"]) == (1, 0)
    assert manager.transcribe_payload.await_count == 2
    units = _rows(db, "SELECT idx, status, run_id FROM units ORDER BY idx")
    assert [status for _idx, status, _run in units] == ["done"] * 3
    assert units[0][2] == child_run
    assert child_run not in {run for _idx, _status, run in units[1:]}
