"""AutoExcerpter's SQLite output: the tables and the rows of one item.

One ``autoexcerpter.sqlite`` per output folder, written through the vendored
:class:`~autoexcerpter.common.sqlite.SqliteWriter`:

- ``documents``: one row per item, keyed by its name: input path and content
  hash, tool version, models, settings fingerprint, the effective spec with
  the source of each value, page counts and the files written;
- ``pages``: one row per page: status, model, token usage and text;
- ``summaries``: one row per page summary: status, model, token usage, page
  information, bullet points and references;
- ``citations``: the document's consolidated citations in rendered order.

An item's rows are written in one transaction and upserted, so resumed and
repeated runs update them in place. Pages and summaries reused from a working
log keep the model and usage recorded by the run that produced them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from autoexcerpter.common.sqlite import SqliteWriter, TableSpec, read_rows, table_names
from autoexcerpter.common.usage import Usage, UsageTotals
from autoexcerpter.rendering.citations.model import Citation

__all__ = [
    "DATABASE_NAME",
    "TABLES",
    "DocumentRecord",
    "RunInfo",
    "complete_documents",
    "open_database",
    "settings_fingerprint",
    "tool_version",
    "write_document",
]

logger = logging.getLogger(__name__)

DATABASE_NAME = "autoexcerpter.sqlite"
PACKAGE = "autoexcerpter"

TABLES = (
    TableSpec(
        "documents",
        """
        CREATE TABLE IF NOT EXISTS documents (
            name TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            hash TEXT,
            tool_version TEXT,
            transcription_model TEXT,
            summary_model TEXT,
            settings_fingerprint TEXT,
            spec TEXT,
            pages_total INTEGER,
            pages_ok INTEGER,
            pages_failed INTEGER,
            pages_deferred INTEGER,
            summary_failures INTEGER,
            complete INTEGER NOT NULL,
            outputs TEXT,
            updated_at TEXT
        );
        """,
        ("name",),
    ),
    TableSpec(
        "pages",
        """
        CREATE TABLE IF NOT EXISTS pages (
            document TEXT NOT NULL,
            page_index INTEGER NOT NULL,
            image TEXT,
            pdf_page_index INTEGER,
            status TEXT NOT NULL,
            error TEXT,
            error_type TEXT,
            model TEXT,
            usage TEXT,
            text TEXT,
            PRIMARY KEY (document, page_index)
        );
        """,
        ("document", "page_index"),
    ),
    TableSpec(
        "summaries",
        """
        CREATE TABLE IF NOT EXISTS summaries (
            document TEXT NOT NULL,
            page_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            model TEXT,
            usage TEXT,
            page_information TEXT,
            bullet_points TEXT,
            page_references TEXT,
            PRIMARY KEY (document, page_index)
        );
        """,
        ("document", "page_index"),
    ),
    TableSpec(
        "citations",
        """
        CREATE TABLE IF NOT EXISTS citations (
            document TEXT NOT NULL,
            position INTEGER NOT NULL,
            text TEXT NOT NULL,
            pages TEXT,
            partial INTEGER NOT NULL,
            doi TEXT,
            url TEXT,
            metadata TEXT,
            PRIMARY KEY (document, position)
        );
        """,
        ("document", "position"),
    ),
)

PAGE_KEY = "original_input_order_index"
# Spec fields that do not shape the outputs' content.
_UNFINGERPRINTED_PREFIXES = ("input.", "run.")
_UNFINGERPRINTED = frozenset(
    {"output.directory", "output.formats", "output.keep_working_files"}
)


def tool_version() -> str:
    """Return the installed version of the tool, or ``unknown``."""
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return "unknown"


def settings_fingerprint(spec: Mapping[str, Mapping[str, Any]]) -> str:
    """Return a SHA-256 of the spec values that shape the outputs' content.

    *spec* maps dotted field paths to ``{"value": ..., "source": ...}``; the
    input selection, the output folder and formats and the run control are
    left out, and so are the sources.
    """
    values = {
        path: entry.get("value")
        for path, entry in spec.items()
        if path not in _UNFINGERPRINTED
        and not path.startswith(_UNFINGERPRINTED_PREFIXES)
    }
    canonical = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunInfo:
    """Values shared by the documents of one run.

    *spec* maps dotted field paths to ``{"value": ..., "source": ...}``.
    """

    spec: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    version: str = field(default_factory=tool_version)

    @property
    def fingerprint(self) -> str:
        """The settings fingerprint of :attr:`spec`."""
        return settings_fingerprint(self.spec)


@dataclass(frozen=True)
class DocumentRecord:
    """What one run produced for one item.

    *pages* and *summaries* are the working-log entries in page order;
    *summaries* is None when the run did not summarize and *citations* None
    when no summary was rendered. *usage* holds the usage per page index of
    the calls made in this run. Pages in *reused_pages* were transcribed by
    an earlier run with the ``model`` their entry records, else with
    *reused_model*, the model of that run's log header. *report* holds the
    page counts and the outputs written.
    """

    name: str
    path: Path
    kind: str
    hash: str | None
    transcription_model: str
    summary_model: str | None
    complete: bool
    report: Mapping[str, Any]
    pages: Sequence[Mapping[str, Any]]
    summaries: Sequence[Mapping[str, Any]] | None = None
    citations: Sequence[Citation] | None = None
    usage: Mapping[int, UsageTotals] = field(default_factory=dict)
    reused_pages: Collection[int] = frozenset()
    reused_model: str | None = None


def open_database(folder: Path) -> SqliteWriter:
    """Return an unopened writer for the database of *folder*."""
    return SqliteWriter(folder / DATABASE_NAME, TABLES)


def complete_documents(folder: Path) -> frozenset[str]:
    """Return the names of the complete ``documents`` rows in *folder*'s database.

    Empty when the folder has no database, which is then not created; an
    unreadable database logs a warning and counts as empty.
    """
    path = folder / DATABASE_NAME
    if not path.is_file():
        return frozenset()
    try:
        if "documents" not in table_names(path):
            return frozenset()
        rows = read_rows(path, "documents")
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Could not read the database %s: %s", path, exc)
        return frozenset()
    return frozenset(str(row["name"]) for row in rows if row.get("complete") == 1)


def _indices(entries: Sequence[Mapping[str, Any]]) -> list[int]:
    indices = (entry.get(PAGE_KEY) for entry in entries)
    return [index for index in indices if isinstance(index, int)]


def _usage(record: DocumentRecord, index: int, role: str) -> dict[str, int]:
    totals = record.usage.get(index)
    usage = totals.get(role) if totals is not None else Usage()
    return usage.as_dict()


def _status(entry: Mapping[str, Any]) -> str:
    return "failed" if "error" in entry else "ok"


def document_row(record: DocumentRecord, run: RunInfo) -> dict[str, Any]:
    """Return the ``documents`` row of *record*."""
    report = record.report
    return {
        "name": record.name,
        "path": str(record.path),
        "kind": record.kind,
        "hash": record.hash,
        "tool_version": run.version,
        "transcription_model": record.transcription_model,
        "summary_model": record.summary_model,
        "settings_fingerprint": run.fingerprint,
        "spec": dict(run.spec),
        "pages_total": report.get("pages_total"),
        "pages_ok": report.get("pages_ok"),
        "pages_failed": report.get("pages_failed"),
        "pages_deferred": report.get("pages_deferred"),
        "summary_failures": report.get("summary_failures"),
        "complete": int(record.complete),
        "outputs": [Path(path).name for path in report.get("outputs", [])],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def page_rows(record: DocumentRecord) -> list[dict[str, Any]]:
    """Return the ``pages`` rows of *record*."""
    rows: list[dict[str, Any]] = []
    for entry in record.pages:
        index = entry.get(PAGE_KEY)
        if not isinstance(index, int):
            continue
        row: dict[str, Any] = {
            "document": record.name,
            "page_index": index,
            "image": entry.get("image"),
            "pdf_page_index": entry.get("page_index"),
            "status": _status(entry),
            "error": entry.get("error"),
            "error_type": entry.get("error_type"),
            "text": entry.get("transcription"),
        }
        if index in record.reused_pages:
            logged = entry.get("model")
            model = logged if isinstance(logged, str) else record.reused_model
            if model is not None:
                row["model"] = model
        else:
            row["model"] = record.transcription_model
            row["usage"] = _usage(record, index, "transcription")
        rows.append(row)
    return rows


def summary_rows(record: DocumentRecord) -> list[dict[str, Any]]:
    """Return the ``summaries`` rows of *record*."""
    rows: list[dict[str, Any]] = []
    for entry in record.summaries or ():
        index = entry.get(PAGE_KEY)
        if not isinstance(index, int):
            continue
        row: dict[str, Any] = {
            "document": record.name,
            "page_index": index,
            "status": _status(entry),
            "error": entry.get("error"),
            "page_information": entry.get("page_information"),
            "bullet_points": entry.get("bullet_points"),
            "page_references": entry.get("references"),
        }
        if index in record.usage:
            row["model"] = record.summary_model
            row["usage"] = _usage(record, index, "summary")
        rows.append(row)
    return rows


def citation_rows(record: DocumentRecord) -> list[dict[str, Any]]:
    """Return the ``citations`` rows of *record* in rendered order."""
    return [
        {
            "document": record.name,
            "position": position,
            "text": citation.raw_text,
            "pages": citation.get_page_range_str(),
            "partial": int(citation.partial),
            "doi": citation.doi,
            "url": citation.url,
            "metadata": citation.metadata,
        }
        for position, citation in enumerate(record.citations or (), start=1)
    ]


async def write_document(
    writer: SqliteWriter, record: DocumentRecord, run: RunInfo
) -> None:
    """Write the rows of one item in one transaction.

    Pages and summaries of the item that this run no longer has are
    deleted; citations are replaced whenever a summary was rendered.
    """
    match = {"document": record.name}
    async with writer.document() as tx:
        tx.upsert("documents", document_row(record, run))
        tx.delete("pages", match, keep={"page_index": _indices(record.pages)})
        for row in page_rows(record):
            tx.upsert("pages", row)
        if record.summaries is not None:
            kept = _indices(record.summaries)
            tx.delete("summaries", match, keep={"page_index": kept})
            for row in summary_rows(record):
                tx.upsert("summaries", row)
        if record.citations is not None:
            tx.delete("citations", match)
            for row in citation_rows(record):
                tx.upsert("citations", row)
