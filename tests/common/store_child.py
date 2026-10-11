"""Child process and shared fixtures for the store tests.

Usage, from the repository root:

    python tests/common/store_child.py acquire DB NAME
        Open the store, begin a run, print READY and wait for a line on
        stdin; then try to lease document NAME and print ``ACQUIRED <gen>``
        or ``REFUSED``, wait for another line and exit.

    python tests/common/store_child.py convert DB SNAPSHOT
        Convert the versionless database DB with a converter that, inside
        the conversion transaction, spills pages to the database file, prints
        READY and sleeps until the parent kills the process.

The module also defines the versionless layouts (the documents and pages
tables of ChronoTranscriber's P2 database and the AutoExcerpter v1.0.x
tables) and a converter for the first, which the tests import.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autoexcerpter.common.sqlite import read  # noqa: E402
from autoexcerpter.common.store import (  # noqa: E402
    Converter,
    Store,
    convert_versionless,
    refresh_documents,
)

TOOL = "tool"
LEGACY_RUN = "legacy-run"
SPILL_ROWS = 400
SPILL_BYTES = 4096


class FakeClock:
    """An injectable clock in epoch seconds that tests advance by hand."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


P2_DDL = """
CREATE TABLE documents (
    name TEXT PRIMARY KEY, path TEXT, kind TEXT, hash TEXT, tool_version TEXT,
    model TEXT, image_settings_fingerprint TEXT, spec TEXT, pages_total INTEGER,
    pages_ok INTEGER, pages_failed INTEGER, pages_no_text INTEGER,
    pages_not_possible INTEGER, complete INTEGER NOT NULL, outputs TEXT,
    updated_at TEXT
);
CREATE TABLE pages (
    document TEXT NOT NULL, page_index INTEGER NOT NULL, page_number INTEGER,
    image TEXT, order_index INTEGER, status TEXT NOT NULL, error TEXT,
    model TEXT, usage TEXT, text TEXT, PRIMARY KEY (document, page_index)
);
"""

AE_V1_DDL = """
CREATE TABLE documents (
    name TEXT PRIMARY KEY, path TEXT NOT NULL, kind TEXT NOT NULL, hash TEXT,
    tool_version TEXT, transcription_model TEXT, summary_model TEXT,
    settings_fingerprint TEXT, spec TEXT, pages_total INTEGER, pages_ok INTEGER,
    pages_failed INTEGER, pages_deferred INTEGER, summary_failures INTEGER,
    complete INTEGER NOT NULL, outputs TEXT, updated_at TEXT
);
CREATE TABLE pages (
    document TEXT NOT NULL, page_index INTEGER NOT NULL, image TEXT,
    pdf_page_index INTEGER, status TEXT NOT NULL, error TEXT, error_type TEXT,
    model TEXT, usage TEXT, text TEXT, PRIMARY KEY (document, page_index)
);
CREATE TABLE summaries (
    document TEXT NOT NULL, page_index INTEGER NOT NULL, status TEXT NOT NULL,
    error TEXT, model TEXT, usage TEXT, page_information TEXT,
    bullet_points TEXT, page_references TEXT, PRIMARY KEY (document, page_index)
);
CREATE TABLE citations (
    document TEXT NOT NULL, position INTEGER NOT NULL, text TEXT NOT NULL,
    pages TEXT, partial INTEGER NOT NULL, doi TEXT, url TEXT, metadata TEXT,
    PRIMARY KEY (document, position)
);
"""


def build_p2(db: Path, documents: int = 3, pages: int = 5) -> None:
    """Write a populated versionless P2 database at *db*."""
    connection = sqlite3.connect(db)
    try:
        connection.executescript(P2_DDL)
        for number in range(documents):
            name = f"doc{number}"
            connection.execute(
                "INSERT INTO documents (name, path, kind, complete) "
                "VALUES (?, ?, 'pdf', 1)",
                (name, f"C:/in/{name}.pdf"),
            )
            for page in range(pages):
                status = "failed" if page == pages - 1 else "ok"
                connection.execute(
                    "INSERT INTO pages (document, page_index, page_number, image, "
                    "order_index, status, model, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'model-a', ?)",
                    (
                        name,
                        page,
                        page + 1,
                        f"page_{page + 1:04d}.png",
                        page,
                        status,
                        f"text {name} {page}",
                    ),
                )
        connection.commit()
    finally:
        connection.close()


def build_ae_v1(db: Path) -> None:
    """Write a populated versionless AutoExcerpter v1.0.x database at *db*."""
    connection = sqlite3.connect(db)
    try:
        connection.executescript(AE_V1_DDL)
        connection.execute(
            "INSERT INTO documents (name, path, kind, complete) "
            "VALUES ('paper', 'C:/in/paper.pdf', 'pdf', 1)"
        )
        connection.execute(
            "INSERT INTO pages (document, page_index, status, text) "
            "VALUES ('paper', 0, 'ok', 'text')"
        )
        connection.execute(
            "INSERT INTO summaries (document, page_index, status) "
            "VALUES ('paper', 0, 'ok')"
        )
        connection.execute(
            "INSERT INTO citations (document, position, text, partial) "
            "VALUES ('paper', 0, 'A. Author, Book (1900).', 0)"
        )
        connection.commit()
    finally:
        connection.close()


def _p2_matches(tables: Mapping[str, tuple[str, ...]]) -> bool:
    return set(tables) == {"documents", "pages"} and "page_number" in tables["pages"]


def _p2_rebuild(connection: sqlite3.Connection, legacy: Mapping[str, str]) -> None:
    connection.execute(
        "INSERT INTO runs (run_id, tool, command, started_at) "
        "VALUES (?, ?, 'convert', '1970-01-01T00:00:00.000+00:00')",
        (LEGACY_RUN, TOOL),
    )
    connection.execute(
        "INSERT INTO documents (name, tool, input_path, kind, first_run_id, "
        f"last_run_id) SELECT name, ?, path, kind, ?, ? FROM {legacy['documents']} "
        "ORDER BY name",
        (TOOL, LEGACY_RUN, LEGACY_RUN),
    )
    connection.execute(
        "INSERT INTO units (doc_id, kind, idx, status, source_index, label, "
        "content, tool, model, provider, run_id) "
        "SELECT d.doc_id, 'page', p.page_index, "
        "CASE p.status WHEN 'failed' THEN 'failed' ELSE 'done' END, "
        "p.page_number, p.image, p.text, ?, COALESCE(p.model, 'unknown'), "
        f"'unknown', ? FROM {legacy['pages']} AS p "
        "JOIN documents AS d ON d.name = p.document",
        (TOOL, LEGACY_RUN),
    )
    refresh_documents(connection)


def _count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _p2_verify(connection: sqlite3.Connection, snapshot: Path) -> None:
    old = read(
        snapshot,
        lambda c: (
            _count(c, "documents"),
            _count(c, "pages"),
            _count(c, "pages WHERE status = 'failed'"),
        ),
    )
    new = (
        _count(connection, "documents"),
        _count(connection, "units"),
        _count(connection, "units WHERE status = 'failed'"),
    )
    if old != new:
        raise ValueError(f"Row counts differ after conversion: {old} != {new}")


P2_CONVERTER = Converter("p2", _p2_matches, _p2_rebuild, _p2_verify)


def _spill_and_hang(connection: sqlite3.Connection, legacy: Mapping[str, str]) -> None:
    _p2_rebuild(connection, legacy)
    connection.execute("PRAGMA cache_size = 2")
    for number in range(SPILL_ROWS):
        connection.execute(
            "INSERT INTO responses (run_id, tool, provider, model, raw, raw_kind, "
            "created_at) VALUES (?, ?, 'p', 'm', randomblob(?), 'legacy', 'now')",
            (LEGACY_RUN, TOOL, SPILL_BYTES + number),
        )
    print("READY", flush=True)
    time.sleep(300)


HANGING_CONVERTER = Converter("p2", _p2_matches, _spill_and_hang, _p2_verify)


async def acquire(db: Path, name: str) -> None:
    async with Store(db, TOOL) as store:
        await store.begin_run("acquire")
        doc_id = await store.ensure_document(name)
        print("READY", flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        lease = await store.acquire(doc_id)
        print(
            "REFUSED" if lease is None else f"ACQUIRED {lease.generation}", flush=True
        )
        await asyncio.to_thread(sys.stdin.readline)


def main(argv: list[str]) -> None:
    mode, db = argv[0], Path(argv[1])
    if mode == "acquire":
        asyncio.run(acquire(db, argv[2]))
    elif mode == "convert":
        convert_versionless(db, [HANGING_CONVERTER], snapshot_path=Path(argv[2]))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
