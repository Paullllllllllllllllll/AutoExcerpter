"""Child process of the Ctrl+Break test.

Usage, from the repository root:

    python tests/cli/interrupt_child.py PDF OUT SETTINGS

Runs ``autoexcerpter run --input PDF --output OUT --settings SETTINGS
--no-summarize --concurrency 1 --json`` through the console entry point, with
a transcription manager that answers without a model. Once the first page is
stored, the second page's call prints READY and waits, so the parent can send
Ctrl+Break while the document is leased. The exit code is the command's.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import autoexcerpter.pipeline.managers as managers  # noqa: E402
from autoexcerpter.__main__ import main  # noqa: E402
from autoexcerpter.output.store import DATABASE_NAME  # noqa: E402


def _first_page_stored(db: Path) -> bool:
    try:
        connection = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        row = connection.execute(
            "SELECT status FROM units WHERE kind = 'transcription' AND idx = 0"
        ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        connection.close()
    return row is not None and row[0] == "done"


class _Manager:
    """Answer the first page; hold the second once the first is stored."""

    def __init__(self, db: Path) -> None:
        self.db = db

    async def transcribe_payload(self, payload: Any) -> dict[str, Any]:
        if payload.number > 1:
            while not _first_page_stored(self.db):
                await asyncio.sleep(0.05)
            print("READY", flush=True)  # noqa: T201 - the parent reads stdout
            await asyncio.sleep(300)
        return {
            "image": payload.image_name,
            "sequence_number": payload.number,
            "transcription": f"Text of page {payload.number}.",
            "processing_time": 0.01,
            "provider": "openai",
        }


def run(argv: list[str]) -> int:
    pdf, out, settings = argv
    manager = _Manager(Path(out) / DATABASE_NAME)
    managers.TranscriptionManager = lambda *args, **kwargs: manager  # type: ignore[assignment,attr-defined]
    return main(
        [
            "run",
            "--input",
            pdf,
            "--output",
            out,
            "--settings",
            settings,
            "--no-summarize",
            "--concurrency",
            "1",
            "--json",
        ]
    )


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
