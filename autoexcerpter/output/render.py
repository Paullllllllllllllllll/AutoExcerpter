"""The files of a document, rendered from the store.

Four formats exist: ``transcription-md`` and ``transcription-txt`` (the same
text under either extension), ``summary-md`` and ``summary-docx``. Each
renderer reads one document from a read-only connection: its done and
failed units in page order (pending units are left out), the ``summaries``
rows and the stored citations; OpenAlex is never asked here, since the item
worker enriches the citations before it stores them. The transcription
header and the summaries' metadata line show the document's ``updated_at``
in UTC, so a document renders to the same bytes on any machine until its
units change. The header notes a resume with another model when a run that wrote
one of the document's current units recorded it. Text files end with
exactly one newline; the DOCX archive is reproducible
(:func:`~autoexcerpter.rendering.docx.docx_summary`).

:func:`render_item` writes an item's files through the store's render
driver: each file is replaced atomically under the document's lease and
recorded in ``outputs``; a file changed outside the tool is kept unless
forced, and a file that cannot be replaced is marked stale while the units
stay stored. :func:`render_database` renders every document of a database
that way (the ``render`` subcommand) and :func:`preview_database` reports
what it would write without writing.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from autoexcerpter.common.sqlite import read
from autoexcerpter.common.store import (
    DocumentRow,
    Lease,
    LeaseLostError,
    OutputRow,
    Renderer,
    RenderReport,
    RenderTarget,
    Store,
    file_sha256,
)
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.output.store import (
    DATABASE_NAME,
    KIND_SUMMARY,
    MODEL_NOTE_PREFIX,
    open_store,
    read_citations,
    read_only_connection,
    read_pages,
    read_store,
)
from autoexcerpter.rendering.citations import CitationManager
from autoexcerpter.rendering.docx import docx_summary
from autoexcerpter.rendering.markdown import markdown_summary
from autoexcerpter.rendering.page_numbering import PageNumberProcessor
from autoexcerpter.rendering.summary import StoredCitations, prepare_summary_data
from autoexcerpter.rendering.transcription import transcription_text
from autoexcerpter.spec import RENDER_FORMATS

__all__ = [
    "RENDERERS",
    "RENDER_FORMATS",
    "SUMMARY_DOCX",
    "SUMMARY_FORMATS",
    "SUMMARY_MD",
    "TRANSCRIPTION_MD",
    "TRANSCRIPTION_TXT",
    "DocumentRender",
    "document_targets",
    "find_databases",
    "has_summaries",
    "item_targets",
    "preview_database",
    "render_database",
    "render_item",
    "render_summary_docx",
    "render_summary_md",
    "render_transcription",
    "rendered_units",
]

TRANSCRIPTION_MD, TRANSCRIPTION_TXT, SUMMARY_MD, SUMMARY_DOCX = RENDER_FORMATS
SUMMARY_FORMATS: Final = (SUMMARY_MD, SUMMARY_DOCX)

_COUNT = "SELECT count(*) FROM units WHERE doc_id = ? AND status != 'pending'"
_SUMMARIES = (
    "SELECT count(*) FROM units AS u JOIN summaries AS s "
    "ON s.doc_id = u.doc_id AND s.idx = u.idx "
    "WHERE u.doc_id = ? AND u.kind = ? AND u.status != 'pending'"
)
_STALE = (
    "INSERT INTO outputs (doc_id, path, format, sha256, rendered_at, run_id, stale) "
    "VALUES (?, ?, ?, NULL, NULL, ?, 1) "
    "ON CONFLICT (doc_id, path) DO UPDATE SET stale = 1"
)
_MODEL_NOTES = (
    "SELECT message FROM notes WHERE doc_id = ? AND substr(message, 1, ?) = ? "
    "AND run_id IN (SELECT run_id FROM units WHERE doc_id = ?) ORDER BY note_id"
)


@dataclass(frozen=True)
class _Document:
    name: str
    input_path: str
    kind: str | None
    processed: datetime

    @property
    def type_label(self) -> str:
        return "PDF" if self.kind == "pdf" else "Image Folder"

    @property
    def position_label(self) -> str:
        return "pdf" if self.kind == "pdf" else "image"


def _processed(updated_at: str | None) -> datetime:
    """Return *updated_at* in UTC; the epoch when it is missing.

    The store writes aware UTC times; a naive one is read as UTC.
    """
    if not updated_at:
        return datetime.fromtimestamp(0, tz=UTC)
    moment = datetime.fromisoformat(updated_at)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _document(connection: sqlite3.Connection, doc_id: int) -> _Document:
    row = connection.execute(
        "SELECT name, input_path, kind, updated_at FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"No document {doc_id}")
    name, input_path, kind, updated_at = row
    return _Document(str(name), str(input_path or ""), kind, _processed(updated_at))


def _model_notes(connection: sqlite3.Connection, doc_id: int) -> list[str]:
    rows = connection.execute(
        _MODEL_NOTES, (doc_id, len(MODEL_NOTE_PREFIX), MODEL_NOTE_PREFIX, doc_id)
    )
    return [str(message) for (message,) in rows]


def render_transcription(connection: sqlite3.Connection, doc_id: int) -> str:
    """Return the transcription file of document *doc_id*."""
    document = _document(connection, doc_id)
    transcriptions, _ = read_pages(connection, doc_id)
    return transcription_text(
        transcriptions,
        document.name,
        document.type_label,
        document.input_path,
        document.processed,
        metadata_notes=_model_notes(connection, doc_id) or None,
    )


def _summary_parts(
    connection: sqlite3.Connection, doc_id: int
) -> tuple[_Document, list[dict[str, Any]], Any, StoredCitations]:
    document = _document(connection, doc_id)
    _, summaries = read_pages(connection, doc_id)
    adjusted = PageNumberProcessor().adjust_and_sort_page_numbers(summaries)
    data = prepare_summary_data(
        adjusted,
        CitationManager(openalex_enabled=False),
        position_label=document.position_label,
    )
    return document, adjusted, data, StoredCitations(read_citations(connection, doc_id))


def render_summary_md(connection: sqlite3.Connection, doc_id: int) -> str:
    """Return the Markdown summary of document *doc_id*."""
    document, adjusted, data, citations = _summary_parts(connection, doc_id)
    return markdown_summary(
        adjusted, document.name, citations, data, generated=document.processed
    )


def render_summary_docx(connection: sqlite3.Connection, doc_id: int) -> bytes:
    """Return the DOCX summary of document *doc_id*."""
    document, adjusted, data, citations = _summary_parts(connection, doc_id)
    return docx_summary(
        adjusted, document.name, citations, data, generated=document.processed
    )


RENDERERS: Final[Mapping[str, Renderer]] = {
    TRANSCRIPTION_MD: render_transcription,
    TRANSCRIPTION_TXT: render_transcription,
    SUMMARY_MD: render_summary_md,
    SUMMARY_DOCX: render_summary_docx,
}


def _paths_by_format(paths: ItemPaths) -> dict[str, Path]:
    transcription = (
        TRANSCRIPTION_TXT if paths.transcription.suffix == ".txt" else TRANSCRIPTION_MD
    )
    return {
        transcription: paths.transcription,
        SUMMARY_DOCX: paths.summary_docx,
        SUMMARY_MD: paths.summary_md,
    }


def item_targets(paths: ItemPaths, formats: Sequence[str]) -> list[RenderTarget]:
    """Return the render targets of an item's *formats*, at its paths.

    The transcription format follows the extension of ``paths.transcription``.
    """
    by_format = _paths_by_format(paths)
    return [
        RenderTarget(path, name, RENDERERS[name])
        for name, path in by_format.items()
        if name in formats
    ]


def _count(store: Store, sql: str, *params: Any) -> int:
    def query(connection: sqlite3.Connection) -> int:
        (count,) = connection.execute(sql, params).fetchone()
        return int(count)

    return read(store.path, query)


def rendered_units(store: Store, doc_id: int) -> int:
    """Return the number of a document's units that a rendering shows."""
    return _count(store, _COUNT, doc_id)


def has_summaries(store: Store, doc_id: int) -> bool:
    """Return whether a document has a stored page summary to render."""
    return _count(store, _SUMMARIES, doc_id, KIND_SUMMARY) > 0


async def render_item(
    store: Store,
    lease: Lease,
    paths: ItemPaths,
    formats: Sequence[str],
    *,
    force: bool = False,
) -> RenderReport:
    """Render the leased document into its *formats* at *paths*.

    The summary formats are left out while the document has no stored
    summary. The files are rendered first, each on its own, then replaced
    in one transaction: a renderer that raises leaves its file as it is
    while the others are written, and a file that cannot be replaced is
    left as well. Both are marked stale in ``outputs`` and listed as stale
    with a warning, so that a later run renders them again.

    Raises:
        LeaseLostError: The lease on the document was lost.
        sqlite3.Error: The store could not be read or written.
    """
    chosen = [
        name
        for name in formats
        if name not in SUMMARY_FORMATS or has_summaries(store, lease.doc_id)
    ]
    rendered: list[RenderTarget] = []
    failed: list[tuple[Path, str, Exception]] = []

    def render_each(connection: sqlite3.Connection) -> None:
        for target in item_targets(paths, chosen):
            try:
                content = target.renderer(connection, lease.doc_id)
            except sqlite3.Error:
                raise
            except Exception as exc:  # noqa: BLE001 - one format fails alone
                failed.append((target.path, target.format, exc))
                continue
            rendered.append(RenderTarget(target.path, target.format, _fixed(content)))

    read(store.path, render_each)
    report = await store.render(lease, rendered, force=force)
    if failed:
        await _mark_stale(store, lease, [(path, fmt) for path, fmt, _ in failed])
    for path, _, exc in failed:
        report.stale.append(path)
        report.warnings.append(f"{path} could not be rendered: {exc}")
    return report


async def _mark_stale(
    store: Store, lease: Lease, targets: Sequence[tuple[Path, str]]
) -> None:
    """Record *targets* as stale under *lease*, keeping a registered hash.

    A target without a row is registered without a hash, so that an existing
    file there is not taken for the tool's own.
    """
    root = store.path.resolve().parent
    rows = []
    for path, fmt in targets:
        resolved = path.resolve()
        key = (
            resolved.relative_to(root).as_posix()
            if resolved.is_relative_to(root)
            else str(resolved)
        )
        rows.append((lease.doc_id, key, fmt, store.run_id))

    def mark(connection: sqlite3.Connection) -> None:
        connection.executemany(_STALE, rows)

    await store.write(lease, mark)


def _fixed(content: str | bytes) -> Renderer:
    """Return a renderer that returns *content*, rendered beforehand."""

    def renderer(connection: sqlite3.Connection, doc_id: int) -> str | bytes:
        return content

    return renderer


@dataclass
class DocumentRender:
    """What a render did, or would do, with one document's files.

    *skipped* says why the document was not rendered (its lease is held by
    another run, or it was lost); *kept* lists files changed outside the
    tool and left alone, *stale* files that could not be replaced.
    """

    database: Path
    name: str
    written: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    kept: list[Path] = field(default_factory=list)
    stale: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: str | None = None

    @property
    def failed(self) -> bool:
        """Whether a file could not be rendered."""
        return bool(self.stale) or self.skipped is not None


def find_databases(path: Path) -> list[Path]:
    """Return the tool's databases: *path* itself when it is a file, else below it."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(found for found in path.rglob(DATABASE_NAME) if found.is_file())


def _registered_path(root: Path, row: OutputRow) -> Path:
    path = Path(row.path)
    return path if path.is_absolute() else root / path


def document_targets(
    root: Path,
    name: str,
    registered: Sequence[OutputRow],
    formats: Sequence[str] | None = None,
    *,
    summaries: bool = True,
) -> list[RenderTarget]:
    """Return the render targets of document *name* of output root *root*.

    Without *formats* the document's registered formats are rendered, else
    the transcription in Markdown and, when the document has *summaries*,
    both summary formats. A format keeps its registered path; another takes
    the item's default path in *root*. Summary formats are left out without
    *summaries*.
    """
    paths = {
        row.format: _registered_path(root, row)
        for row in registered
        if row.format in RENDERERS
    }
    if formats:
        chosen = [fmt for fmt in RENDER_FORMATS if fmt in formats]
    elif paths:
        chosen = [fmt for fmt in RENDER_FORMATS if fmt in paths]
    else:
        chosen = [TRANSCRIPTION_MD, SUMMARY_MD, SUMMARY_DOCX]
    if not summaries:
        chosen = [fmt for fmt in chosen if fmt not in SUMMARY_FORMATS]
    defaults = {
        **_paths_by_format(ItemPaths.for_item(name, root, "md")),
        **_paths_by_format(ItemPaths.for_item(name, root, "txt")),
    }
    return [
        RenderTarget(paths.get(fmt) or defaults[fmt], fmt, RENDERERS[fmt])
        for fmt in chosen
    ]


def _preview_document(
    connection: sqlite3.Connection,
    database: Path,
    document: DocumentRow,
    targets: Sequence[RenderTarget],
    registered: Sequence[OutputRow],
    force: bool,
) -> DocumentRender:
    result = DocumentRender(database, document.name)
    recorded = {
        _registered_path(database.parent, row): row.sha256 for row in registered
    }
    for target in targets:
        rendered = target.renderer(connection, document.doc_id)
        data = rendered if isinstance(rendered, bytes) else rendered.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        try:
            current = file_sha256(target.path)
        except OSError as exc:
            result.stale.append(target.path)
            result.warnings.append(f"Could not read {target.path}: {exc}")
            continue
        if current == digest:
            result.unchanged.append(target.path)
        elif current is not None and current != recorded.get(target.path) and not force:
            result.kept.append(target.path)
        else:
            result.written.append(target.path)
    return result


def _renderable_documents(root: Path) -> tuple[Store, list[DocumentRow]] | None:
    reader = read_store(root)
    if reader is None:
        return None
    documents = [
        doc for doc in reader.documents() if rendered_units(reader, doc.doc_id)
    ]
    return reader, documents


def _targets(
    store: Store, document: DocumentRow, formats: Sequence[str] | None
) -> list[RenderTarget]:
    return document_targets(
        store.path.parent,
        document.name,
        store.outputs(document.doc_id),
        formats,
        summaries=has_summaries(store, document.doc_id),
    )


def preview_database(
    database: Path,
    *,
    formats: Sequence[str] | None = None,
    force: bool = False,
) -> list[DocumentRender]:
    """Return what :func:`render_database` would do; write nothing.

    A target that cannot be read is listed as stale with a warning.

    Raises:
        StoreError: The database is versionless, newer or unreadable.
    """
    root = database.parent
    found = _renderable_documents(root)
    if found is None:
        return []
    reader, documents = found
    planned = [
        (doc, _targets(reader, doc, formats), reader.outputs(doc.doc_id))
        for doc in documents
    ]
    with read_only_connection(root) as connection:
        return [
            _preview_document(connection, database, doc, targets, registered, force)
            for doc, targets, registered in planned
        ]


async def _render_document(
    store: Store,
    document: DocumentRow,
    formats: Sequence[str] | None,
    force: bool,
) -> DocumentRender:
    result = DocumentRender(store.path, document.name)
    targets = _targets(store, document, formats)
    lease = await store.acquire(document.doc_id)
    if lease is None:
        result.skipped = "another run holds its lease"
        await store.add_note(
            f"render skipped '{document.name}': another run holds its lease",
            doc_id=document.doc_id,
        )
        return result
    try:
        report = await store.render(lease, targets, force=force)
    except (LeaseLostError, OSError, sqlite3.Error) as exc:
        result.skipped = f"not rendered: {exc}"
        return result
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    result.written = report.written
    result.unchanged = report.unchanged
    result.kept = report.kept
    result.stale = report.stale
    result.warnings = report.warnings
    return result


async def render_database(
    database: Path,
    *,
    formats: Sequence[str] | None = None,
    force: bool = False,
) -> list[DocumentRender]:
    """Render every document of *database* from the store, without model calls.

    Each document is rendered under its lease; a document another run holds
    is skipped with a note. A file changed outside the tool is left alone
    unless *force*; a file that cannot be replaced stays stale for a later
    render. Documents without a done or failed unit are left out.

    Raises:
        StoreError: The database is versionless, newer or unreadable.
    """
    root = database.parent
    found = _renderable_documents(root)
    if found is None or not found[1]:
        return []
    documents = found[1]
    results: list[DocumentRender] = []
    async with open_store(root) as store:
        await store.begin_run(
            "render",
            spec={
                "formats": {"value": list(formats or ()), "source": "flag"},
                "force": {"value": force, "source": "flag"},
            },
        )
        try:
            for document in documents:
                results.append(await _render_document(store, document, formats, force))
        finally:
            await store.end_run(1 if any(result.failed for result in results) else 0)
    return results
