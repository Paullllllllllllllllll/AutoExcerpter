"""Version 1.0 outputs built in a temp folder: working logs, files and database.

An :class:`Item` holds the pages of one PDF item. :func:`write_item` writes
what a version 1.0 run left for it: the two working logs in
``<name>_autoexcerpter/`` and the rendered files with a version 1.0 header
(the elapsed-time line included). :func:`write_database` writes the version
1.0 ``autoexcerpter.sqlite`` with the item's rows.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from autoexcerpter.migrate import offline_citations
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.rendering.citations import CitationManager, OpenAlexCache
from autoexcerpter.rendering.docx import docx_summary
from autoexcerpter.rendering.markdown import markdown_summary
from autoexcerpter.rendering.page_numbering import PageNumberProcessor
from autoexcerpter.rendering.summary import StoredCitations, prepare_summary_data
from autoexcerpter.rendering.transcription import transcription_text

MODEL = "gpt-5-mini"
SUMMARY_MODEL = "claude-sonnet-5-5"
CITATION = "Smith, J. (2020). A Title of the Cited Work. Journal of Tests, 1, 1-2."
OLD_TIME = datetime(2026, 1, 1, 10, 0, 0)
IMAGE_CONFIG = {
    "target_dpi": 300,
    "jpeg_quality": 95,
    "grayscale_conversion": True,
    "llm_detail": "high",
    "model_name": MODEL,
    "provider": "openai",
    "model_type": "openai",
}
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}

V1_SCHEMA = """
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


@dataclass
class Page:
    """One page: its text, a failed transcription or summary, its references.

    A page without a printed number gets a page-break marker in the
    transcription file.
    """

    text: str = ""
    error: str | None = None
    summary_error: str | None = None
    references: tuple[str, ...] = ()
    numbered: bool = True


@dataclass
class Item:
    """A version 1.0 PDF item in *root*; *total* pages, *pages* in the logs."""

    root: Path
    name: str
    pages: list[Page]
    total: int
    state_dir: Path
    summarized: bool = True
    retries: dict[int, int] = field(default_factory=dict)

    @property
    def paths(self) -> ItemPaths:
        return ItemPaths.for_item(self.name, self.root)

    @property
    def folder(self) -> Path:
        return self.paths.legacy_logs()[0].parent

    def image(self, idx: int) -> str:
        return f"{self.name}_page_{idx + 1:04d}.jpg"

    def text(self, idx: int) -> str:
        page = self.pages[idx]
        if page.error is not None:
            return f"[transcription error: {page.error}]"
        if page.numbered:
            return f"<page_number>{idx + 1}</page_number>\n{page.text}"
        return page.text


def _header(item: Item, log_type: str, model: str) -> dict[str, Any]:
    header: dict[str, Any] = {
        "_format_version": 2,
        "log_type": log_type,
        "input_item_name": item.name,
        "input_item_path": str(item.root / f"{item.name}.pdf"),
        "input_type": "PDF",
        "processing_start_time": OLD_TIME.isoformat(),
        "total_images": item.total,
        "model_name": model,
        "configuration": {"concurrent_requests": 16},
    }
    if log_type == "transcription":
        header["file_provenance"] = {
            "source_file": str(item.root / f"{item.name}.pdf"),
            "image_config": IMAGE_CONFIG,
            "source_sha256": "ab" * 32,
        }
    return header


def transcription_record(item: Item, idx: int) -> dict[str, Any]:
    """Return the transcription log record of page *idx*."""
    page = item.pages[idx]
    record: dict[str, Any] = {
        "image": item.image(idx),
        "sequence_number": idx + 1,
        "transcription": item.text(idx),
        "processing_time": 0.5,
        "provider": "openai",
        "original_input_order_index": idx,
        "source_file": str(item.root / f"{item.name}.pdf"),
        "image_provenance": {"dpi": 300, "page": idx},
        "page_index": idx,
    }
    if page.error is not None:
        record["error"] = page.error
        record["error_type"] = "api_error"
    return record


def summary_record(item: Item, idx: int) -> dict[str, Any]:
    """Return the summary log record of page *idx*, as version 1.0 built it."""
    page = item.pages[idx]
    record: dict[str, Any] = {
        "original_input_order_index": idx,
        "image_filename": item.image(idx),
        "page": idx + 1,
    }
    if page.error is not None:
        record |= {
            "page_information": {
                "page_number_integer": None,
                "is_two_page_spread": False,
                "page_number_integer_end": None,
                "page_number_type": "none",
                "page_types": ["other"],
            },
            "bullet_points": [f"[Transcription failed: {page.error}]"],
            "references": None,
        }
        return record
    record |= {
        "page_information": {
            "page_number_integer": idx + 1 if page.numbered else None,
            "page_number_type": "arabic" if page.numbered else "none",
            "page_types": ["content"],
        },
        "bullet_points": [f"A point of page {idx + 1}."],
        "references": [
            {"citation": reference, "is_partial": False}
            for reference in page.references
        ],
        "processing_time": 0.4,
        "provider": "anthropic",
    }
    if page.summary_error is not None:
        record["error"] = page.summary_error
        record["bullet_points"] = [f"[Summary failed: {page.summary_error}]"]
    return record


def _lines(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def write_logs(item: Item, *, tail: str = "") -> None:
    """Write the item's working logs; *tail* is appended to the summary log."""
    item.folder.mkdir(parents=True, exist_ok=True)
    transcription_path, summary_path = item.paths.legacy_logs()
    records = [_header(item, "transcription", MODEL)]
    for idx in range(len(item.pages)):
        records.extend(
            transcription_record(item, idx) for _ in range(item.retries.get(idx, 0) + 1)
        )
    transcription_path.write_text(_lines(records), encoding="utf-8", newline="\n")
    if item.summarized:
        summaries = [_header(item, "summary", SUMMARY_MODEL)]
        summaries.extend(summary_record(item, idx) for idx in range(len(item.pages)))
        summary_path.write_text(
            _lines(summaries) + tail, encoding="utf-8", newline="\n"
        )


def _page_entry(item: Item, idx: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "original_input_order_index": idx,
        "image": item.image(idx),
        "transcription": item.text(idx),
        "page_index": idx,
    }
    if item.pages[idx].error is not None:
        entry["error"] = item.pages[idx].error
    return entry


def _summary_entry(item: Item, idx: int) -> dict[str, Any]:
    record = summary_record(item, idx)
    entry = {
        key: record.get(key)
        for key in (
            "original_input_order_index",
            "image_filename",
            "page",
            "page_information",
            "bullet_points",
            "references",
        )
    }
    if item.pages[idx].summary_error is not None and item.pages[idx].error is None:
        entry["error"] = item.pages[idx].summary_error
    return entry


def transcription_file(item: Item, *, markers: bool, final_newline: bool) -> str:
    """Return the version 1.0 transcription file of the item."""
    entries = [_page_entry(item, idx) for idx in range(len(item.pages))]
    rendered = transcription_text(
        entries, item.name, "PDF", item.root / f"{item.name}.pdf", OLD_TIME
    )
    body = rendered.partition("\n\n")[2]
    if not markers:
        body = "".join(
            line
            for line in body.splitlines(keepends=True)
            if not line.startswith("<page_break ")
        )
    if not final_newline:
        body = body.removesuffix("\n")
    failed = sum(page.error is not None for page in item.pages)
    header = (
        f"# Transcription of: {item.name}\n"
        f"# Source Path: {item.root / f'{item.name}.pdf'}\n"
        "# Type: PDF\n"
        f"# Processed on: {OLD_TIME.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Total images processed: {len(item.pages)}\n"
        f"# Successfully transcribed: {len(item.pages) - failed}\n"
        f"# Failed items: {failed}\n"
        "# Total processing time: 0:01:05\n\n"
    )
    return header + body


def _summary_parts(item: Item) -> tuple[list[dict[str, Any]], Any, StoredCitations]:
    entries = [_summary_entry(item, idx) for idx in range(len(item.pages))]
    adjusted = PageNumberProcessor().adjust_and_sort_page_numbers(entries)
    data = prepare_summary_data(
        adjusted, CitationManager(openalex_enabled=False), position_label="pdf"
    )
    fresh = [_summary_entry(item, idx) for idx in range(len(item.pages))]
    citations = offline_citations(fresh, "pdf", OpenAlexCache(item.state_dir))
    return adjusted, data, StoredCitations(citations)


def summary_file(item: Item) -> str:
    """Return the version 1.0 Markdown summary of the item."""
    adjusted, data, citations = _summary_parts(item)
    return markdown_summary(adjusted, item.name, citations, data, generated=OLD_TIME)


def summary_docx(item: Item) -> bytes:
    """Return the version 1.0 Word summary of the item."""
    adjusted, data, citations = _summary_parts(item)
    return docx_summary(adjusted, item.name, citations, data, generated=OLD_TIME)


def write_files(
    item: Item,
    *,
    summary: bool = True,
    markers: bool = True,
    final_newline: bool = True,
) -> None:
    """Write the item's rendered files as version 1.0 left them."""
    paths = item.paths
    item.root.mkdir(parents=True, exist_ok=True)
    paths.transcription.write_text(
        transcription_file(item, markers=markers, final_newline=final_newline),
        encoding="utf-8",
        newline="\n",
    )
    if summary:
        paths.summary_md.write_text(summary_file(item), encoding="utf-8", newline="\n")
        paths.summary_docx.write_bytes(summary_docx(item))


def make_item(
    root: Path,
    name: str,
    pages: list[Page],
    *,
    state_dir: Path,
    total: int | None = None,
    summarized: bool = True,
) -> Item:
    """Return an item of *pages*; *total* defaults to their number."""
    return Item(
        root,
        name,
        pages,
        len(pages) if total is None else total,
        state_dir,
        summarized=summarized,
    )


def write_database(root: Path, items: list[Item], *, citations: bool = False) -> Path:
    """Write the version 1.0 database of *items* into *root*."""
    path = root / "autoexcerpter.sqlite"
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript(V1_SCHEMA)
        for item in items:
            _insert_item(connection, item, citations)
    connection.close()
    return path


def _insert_item(connection: sqlite3.Connection, item: Item, citations: bool) -> None:
    paths = item.paths
    failed = sum(page.error is not None for page in item.pages)
    outputs = [path.name for path in (paths.transcription, paths.summary_md)]
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, 'pdf', ?, '1.0.2', ?, ?, 'fp', ?, ?, ?, "
        "?, 0, 0, 1, ?, ?)",
        (
            item.name,
            str(item.root / f"{item.name}.pdf"),
            "ab" * 32,
            MODEL,
            SUMMARY_MODEL,
            json.dumps({"model.name": {"value": MODEL, "source": "flag"}}),
            item.total,
            len(item.pages) - failed,
            failed,
            json.dumps(outputs),
            OLD_TIME.isoformat(timespec="seconds"),
        ),
    )
    for idx, page in enumerate(item.pages):
        connection.execute(
            "INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.name,
                idx,
                item.image(idx),
                idx,
                "failed" if page.error is not None else "ok",
                page.error,
                "api_error" if page.error is not None else None,
                MODEL,
                json.dumps(USAGE),
                item.text(idx),
            ),
        )
        entry = _summary_entry(item, idx)
        connection.execute(
            "INSERT INTO summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.name,
                idx,
                "failed" if "error" in entry else "ok",
                entry.get("error"),
                SUMMARY_MODEL,
                json.dumps(USAGE),
                json.dumps(entry["page_information"]),
                json.dumps(entry["bullet_points"]),
                json.dumps(entry["references"]),
            ),
        )
    if citations:
        connection.execute(
            "INSERT INTO citations VALUES (?, 1, ?, 'p. 1', 0, NULL, NULL, NULL)",
            (item.name, CITATION),
        )
