"""AutoExcerpter's store: its schema component, read views, writes and resume.

Every output folder holds one ``autoexcerpter.sqlite``. Its core tables come
from :mod:`autoexcerpter.common.store`; this module adds the tool's
component: the content tables ``summaries`` and ``citations``, a trigger and
the views below.

A document is one input item, named by its output stem (``doc`` for
``doc.pdf``, the folder name for an image folder), of kind ``pdf`` or
``images``. Each page has two units with the same ``idx``, the page's
position in the item (``original_input_order_index``):

- ``transcription``: ``label`` is the page image name and ``source_index``
  the 0-based PDF page index or the image file name; ``content`` holds the
  cleaned text, or the placeholder text of a failed page; ``outcome`` is
  ``ok``, ``blank`` or ``failed``; ``provenance`` holds the image provenance
  and the source file.
- ``summary``: same ``label`` and ``source_index``; ``content`` stays NULL,
  the summary itself is a ``summaries`` row. A change of the page's
  transcription unit (its status or content) marks a summary that is not
  pending stale, so resume summarizes the page again.

``summaries`` holds one row per summarized page (``page`` is its 1-based
position, ``page_information``, ``bullet_points`` and ``page_references``
are JSON); ``citations`` the document's consolidated citations in rendered
order. Units carry the model and provider of their role.

Read contract
=============

Outside code reads the store only through these views; the tables behind
them may change between versions, the views stay stable.

``page_text``
    One transcription row per page, ordered by document and
    ``order_index``: ``document``, ``kind`` (``transcription``),
    ``page_index`` (the 0-based PDF page, NULL for an image),
    ``image_name``, ``order_index``, ``text``, ``status`` (pending, done or
    failed), ``model``, ``usage`` (JSON token counts), ``outcome`` (ok,
    blank or failed), ``error``, ``provider`` and ``run_id``. These are the
    columns of ChronoTranscriber's ``page_text``.

``page_summaries``
    One summary row per page, ordered by document and ``order_index``:
    ``document``, ``order_index``, ``page``, ``image_name``, ``status``,
    ``stale`` (1 when the page's transcription changed since), ``outcome``,
    ``page_information``, ``bullet_points``, ``page_references`` (JSON),
    ``model``, ``usage``, ``error``, ``provider`` and ``run_id``.

``document_citations``
    The consolidated citations, ordered by document and ``position``:
    ``document``, ``position``, ``text``, ``pages``, ``partial``, ``doi``,
    ``url`` and ``metadata`` (JSON, from OpenAlex).

``document_status``
    One row per document: ``document``, ``kind`` (pdf or images),
    ``input_path``, ``status`` (pending, partial, complete or failed),
    ``units_total``, ``units_done``, ``units_failed``, ``units_pending``,
    ``last_run_id`` and ``updated_at``.

``page_attempts``
    One row per provider attempt in the order they were made: ``document``,
    ``kind``, ``order_index``, ``response_id``, ``run_id``, ``provider``,
    ``model``, ``attempt``, ``submission_id``, ``custom_id``, ``raw`` (the
    response body as received), ``raw_kind``, ``usage``, ``error``,
    ``http_status``, ``created_at`` and ``role`` (transcription or summary).

Open the database read-only and close it after the query, so that a running
item never waits on the reader::

    import sqlite3
    from pathlib import Path

    database = Path("out") / "autoexcerpter.sqlite"
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True
    )
    try:
        rows = connection.execute(
            "SELECT order_index, image_name, status, text FROM page_text "
            "WHERE document = ? ORDER BY order_index",
            ("doc",),
        ).fetchall()
    finally:
        connection.close()
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from autoexcerpter.common.native import MODEL_DERIVED_IMAGE_KEYS, changed_image_settings
from autoexcerpter.common.sqlite import read, read_only
from autoexcerpter.common.store import (
    Component,
    Lease,
    PlannedUnit,
    ResumeState,
    SchemaState,
    Store,
    UnitKey,
    UnitResult,
    WriterOptions,
    check_schema,
)
from autoexcerpter.constants import is_blank_transcription

__all__ = [
    "COMPONENT",
    "DATABASE_NAME",
    "KIND_SUMMARY",
    "KIND_TRANSCRIPTION",
    "TOOL",
    "UNIT_KINDS",
    "VIEWS",
    "CitationRow",
    "Classification",
    "InputMismatchError",
    "ItemState",
    "ResumeRequest",
    "SummaryRow",
    "begin_run",
    "classify",
    "database_path",
    "document_kind",
    "image_settings_record",
    "model_derived_keys",
    "open_store",
    "plan_pages",
    "read_only_connection",
    "read_store",
    "settings_fingerprint",
    "store_artifacts",
    "tool_version",
    "transcription_outcome",
    "write_citations",
    "write_summary",
]

TOOL: Final = "autoexcerpter"
DATABASE_NAME: Final = "autoexcerpter.sqlite"

KIND_TRANSCRIPTION: Final = "transcription"
KIND_SUMMARY: Final = "summary"
UNIT_KINDS: Final = (KIND_TRANSCRIPTION, KIND_SUMMARY)

VIEWS: Final = (
    "page_text",
    "page_summaries",
    "document_citations",
    "document_status",
    "page_attempts",
)

_V1 = """
CREATE TABLE summaries (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    idx INTEGER NOT NULL,
    page INTEGER,
    page_information TEXT,
    bullet_points TEXT,
    page_references TEXT,
    PRIMARY KEY (doc_id, idx)
);
CREATE TABLE citations (
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    position INTEGER NOT NULL CHECK (position >= 1),
    text TEXT NOT NULL,
    pages TEXT,
    partial INTEGER NOT NULL DEFAULT 0 CHECK (partial IN (0, 1)),
    doi TEXT,
    url TEXT,
    metadata TEXT,
    PRIMARY KEY (doc_id, position)
);
CREATE TRIGGER summaries_stale_on_transcription
AFTER UPDATE OF status, content ON units
WHEN NEW.kind = 'transcription'
AND (OLD.status IS NOT NEW.status OR OLD.content IS NOT NEW.content)
BEGIN
    UPDATE units SET stale = 1
    WHERE doc_id = NEW.doc_id AND kind = 'summary' AND idx = NEW.idx
    AND status != 'pending';
END;
CREATE VIEW page_text AS
SELECT
    d.name AS document,
    u.kind AS kind,
    CASE WHEN typeof(u.source_index) = 'integer'
        THEN u.source_index END AS page_index,
    u.label AS image_name,
    u.idx AS order_index,
    u.content AS text,
    u.status AS status,
    u.model AS model,
    u.usage AS usage,
    u.outcome AS outcome,
    u.error AS error,
    u.provider AS provider,
    u.run_id AS run_id
FROM units AS u
JOIN documents AS d ON d.doc_id = u.doc_id
WHERE u.kind = 'transcription'
ORDER BY d.name, u.idx;
CREATE VIEW page_summaries AS
SELECT
    d.name AS document,
    u.idx AS order_index,
    s.page AS page,
    u.label AS image_name,
    u.status AS status,
    u.stale AS stale,
    u.outcome AS outcome,
    s.page_information AS page_information,
    s.bullet_points AS bullet_points,
    s.page_references AS page_references,
    u.model AS model,
    u.usage AS usage,
    u.error AS error,
    u.provider AS provider,
    u.run_id AS run_id
FROM units AS u
JOIN documents AS d ON d.doc_id = u.doc_id
LEFT JOIN summaries AS s ON s.doc_id = u.doc_id AND s.idx = u.idx
WHERE u.kind = 'summary'
ORDER BY d.name, u.idx;
CREATE VIEW document_citations AS
SELECT
    d.name AS document,
    c.position AS position,
    c.text AS text,
    c.pages AS pages,
    c.partial AS partial,
    c.doi AS doi,
    c.url AS url,
    c.metadata AS metadata
FROM citations AS c
JOIN documents AS d ON d.doc_id = c.doc_id
ORDER BY d.name, c.position;
CREATE VIEW document_status AS
SELECT
    name AS document,
    kind,
    input_path,
    status,
    units_total,
    units_done,
    units_failed,
    units_pending,
    last_run_id,
    updated_at
FROM documents
ORDER BY name;
CREATE VIEW page_attempts AS
SELECT
    d.name AS document,
    r.kind AS kind,
    r.idx AS order_index,
    r.response_id AS response_id,
    r.run_id AS run_id,
    r.provider AS provider,
    r.model AS model,
    r.attempt AS attempt,
    r.submission_id AS submission_id,
    r.custom_id AS custom_id,
    r.raw AS raw,
    r.raw_kind AS raw_kind,
    r.usage AS usage,
    r.error AS error,
    r.http_status AS http_status,
    r.created_at AS created_at,
    r.role AS role
FROM responses AS r
LEFT JOIN documents AS d ON d.doc_id = r.doc_id
ORDER BY r.response_id;
"""

COMPONENT: Final = Component(TOOL, (_V1,))
"""The tool's schema component: version 1 adds the content tables and views."""

_COMPONENTS: Final = (COMPONENT,)

# Spec fields that do not shape the outputs' content.
_UNFINGERPRINTED_PREFIXES: Final = ("input.", "run.")
_UNFINGERPRINTED: Final = frozenset(
    {"output.directory", "output.formats", "output.keep_working_files"}
)


def tool_version() -> str:
    """Return the installed version of the tool, or ``unknown``."""
    try:
        return metadata.version(TOOL)
    except metadata.PackageNotFoundError:
        return "unknown"


def database_path(root: Path) -> Path:
    """Return the database file of output folder *root*."""
    return Path(root) / DATABASE_NAME


def open_store(
    root: Path,
    *,
    clock: Callable[[], float] = time.time,
    writer: WriterOptions | None = None,
) -> Store:
    """Return the store of output folder *root*, not yet opened.

    ``async with open_store(root) as store`` opens its writer, creating the
    database when missing.
    """
    return Store(
        database_path(root),
        TOOL,
        tool_version=tool_version(),
        components=_COMPONENTS,
        clock=clock,
        writer=writer,
    )


def _schema(root: Path) -> SchemaState:
    return check_schema(database_path(root), _COMPONENTS)


def read_store(root: Path) -> Store | None:
    """Return the store of *root* for reads only; None without a database.

    The store is not opened: its read methods use short read-only
    connections. A versionless, newer or unreadable database raises the
    :class:`~autoexcerpter.common.store.StoreError` that refuses it.
    """
    state = _schema(root)
    if not state.exists or state.empty:
        return None
    return Store(state.path, TOOL, components=_COMPONENTS)


@contextlib.contextmanager
def read_only_connection(root: Path) -> Iterator[sqlite3.Connection]:
    """Yield a ``mode=ro`` connection to *root*'s database; close it after.

    The schema is checked first, as for :func:`read_store`;
    ``FileNotFoundError`` without a database.
    """
    state = _schema(root)
    if not state.exists:
        raise FileNotFoundError(f"No database at {state.path}")
    with read_only(state.path) as connection:
        yield connection


def document_kind(item_kind: str) -> str:
    """Return the document kind of an item kind: ``pdf`` or ``images``."""
    return "pdf" if item_kind == "pdf" else "images"


def _plain(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return *value* as JSON would store it (tuples become lists)."""
    plain = json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))
    return plain if isinstance(plain, dict) else {}


def image_settings_record(
    settings: Mapping[str, Any], model_type: str | None = None
) -> dict[str, Any]:
    """Return the image-settings record a document's pages are produced with.

    *settings* is the resolved image config of the run's model, *model_type*
    its model family (kept from *settings* when None). The record holds the
    values the image-settings fingerprint covers, in their stored JSON form.
    """
    record = dict(settings)
    if model_type is not None:
        record["model_type"] = model_type
    return _plain(record)


def model_derived_keys(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> frozenset[str]:
    """Return the keys of an image config computed from the model.

    In the OpenAI section ``llm_detail`` is overwritten with the detail the
    model resolves, so it is model-derived when both configs are OpenAI's.
    """
    if recorded.get("model_type") == current.get("model_type") == "openai":
        return MODEL_DERIVED_IMAGE_KEYS | {"llm_detail"}
    return MODEL_DERIVED_IMAGE_KEYS


# =============================================================================
# Run stamps
# =============================================================================


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


async def begin_run(
    store: Store, command: str, spec: Mapping[str, Mapping[str, Any]]
) -> str:
    """Record a run with its command, effective spec and settings fingerprint."""
    return await store.begin_run(
        command, spec=spec, settings_fingerprint=settings_fingerprint(spec)
    )


async def store_artifacts(
    store: Store, texts: Mapping[str, str | None]
) -> dict[str, str]:
    """Store the prompt, schema and context texts of a run by kind.

    Empty and missing texts are skipped. Returns the SHA-256 of each stored
    text by kind.
    """
    return {
        kind: await store.store_artifact(kind, text)
        for kind, text in texts.items()
        if text
    }


# =============================================================================
# Units and content rows
# =============================================================================


async def plan_pages(
    store: Store,
    lease: Lease,
    pages: Sequence[PlannedUnit],
    *,
    transcription: tuple[str, str],
    summary: tuple[str, str] | None,
) -> int:
    """Create a document's transcription and summary units as ``pending``.

    *transcription* and *summary* are the (model, provider) of each role;
    *summary* None plans no summary units. Existing units are kept. Returns
    the number of units created.
    """
    model, provider = transcription
    created = await store.plan_units(
        lease, KIND_TRANSCRIPTION, pages, model=model, provider=provider
    )
    if summary is not None:
        model, provider = summary
        created += await store.plan_units(
            lease, KIND_SUMMARY, pages, model=model, provider=provider
        )
    return created


def transcription_outcome(text: str | None, error: str | None) -> str:
    """Return a transcription's outcome: ``failed``, ``blank`` or ``ok``."""
    if error is not None:
        return "failed"
    if is_blank_transcription(text):
        return "blank"
    return "ok"


@dataclass(frozen=True)
class SummaryRow:
    """A ``summaries`` row: the structured summary of one page."""

    idx: int
    page: int | None = None
    page_information: Mapping[str, Any] | None = None
    bullet_points: Sequence[str] | None = None
    references: Sequence[Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class CitationRow:
    """A ``citations`` row: one consolidated citation in rendered order."""

    text: str
    pages: str | None = None
    partial: bool = False
    doi: str | None = None
    url: str | None = None
    metadata: Mapping[str, Any] | None = None


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


async def write_summary(
    store: Store, lease: Lease, result: UnitResult, row: SummaryRow | None
) -> None:
    """Commit a page's ``summaries`` row, then its summary unit.

    The row commits first: a crash between the two leaves the unit pending,
    and the next run summarizes the page again. *row* None (a failed call
    without a placeholder) removes the page's row.
    """
    if result.kind != KIND_SUMMARY or (row is not None and row.idx != result.idx):
        raise ValueError("write_summary takes a summary unit and its own row.")
    doc_id, idx = lease.doc_id, result.idx

    def upsert(connection: sqlite3.Connection) -> None:
        if row is None:
            connection.execute(
                "DELETE FROM summaries WHERE doc_id = ? AND idx = ?", (doc_id, idx)
            )
            return
        connection.execute(
            "INSERT INTO summaries (doc_id, idx, page, page_information, "
            "bullet_points, page_references) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (doc_id, idx) DO UPDATE SET page = excluded.page, "
            "page_information = excluded.page_information, "
            "bullet_points = excluded.bullet_points, "
            "page_references = excluded.page_references",
            (
                doc_id,
                idx,
                row.page,
                _json(row.page_information),
                _json(row.bullet_points),
                _json(row.references),
            ),
        )

    await store.write(lease, upsert)
    await store.write_unit(lease, result)


async def write_citations(
    store: Store, lease: Lease, citations: Sequence[CitationRow]
) -> None:
    """Replace a document's citations with *citations*, numbered from 1."""
    doc_id = lease.doc_id
    rows = [
        (
            doc_id,
            position,
            citation.text,
            citation.pages,
            int(citation.partial),
            citation.doi,
            citation.url,
            _json(citation.metadata),
        )
        for position, citation in enumerate(citations, start=1)
    ]

    def replace(connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM citations WHERE doc_id = ?", (doc_id,))
        connection.executemany(
            "INSERT INTO citations (doc_id, position, text, pages, partial, doi, "
            "url, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    await store.write(lease, replace)


# =============================================================================
# Resume classification
# =============================================================================


class ItemState(StrEnum):
    """What the store holds for an item, as resume sees it."""

    NONE = "none"
    COMPLETE = "complete"
    PARTIAL = "partial"
    TRANSCRIPTION_ONLY = "transcription_only"
    LEGACY = "legacy"
    BLOCKED = "blocked"
    REFUSED = "refused"


class InputMismatchError(ValueError):
    """The stored document belongs to another input with the same name."""


@dataclass(frozen=True)
class Classification:
    """The resume state of one item.

    *units* are the units to process: transcription and summary units of a
    partial item, summary units of a ``TRANSCRIPTION_ONLY`` item (summary
    units not yet planned included). *render* asks a complete item to
    re-render its files without model calls; *missing* lists those files
    absent on disk. *reset* asks to reset every unit first, since the input
    changed since they were stored. *note* records a model change, which
    resumes.
    """

    state: ItemState
    name: str
    doc_id: int | None = None
    units: tuple[UnitKey, ...] = ()
    render: bool = False
    missing: tuple[Path, ...] = ()
    reset: bool = False
    reason: str = ""
    note: str | None = None

    def indices(self, kind: str) -> tuple[int, ...]:
        """Return the indices of the units of *kind* to process."""
        return tuple(unit.idx for unit in self.units if unit.kind == kind)


@dataclass(frozen=True)
class _Details:
    input_path: str | None
    input_sha256: str | None
    models: Mapping[str, tuple[str, ...]]
    units: Mapping[str, Mapping[int, tuple[str, int]]]
    outputs: Mapping[str, int]


def _nonempty(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _path_key(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _normalized(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _details(connection: sqlite3.Connection, doc_id: int) -> _Details:
    input_path, input_sha256 = connection.execute(
        "SELECT input_path, input_sha256 FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    models: dict[str, list[str]] = {}
    for kind, model in connection.execute(
        "SELECT DISTINCT kind, model FROM units WHERE doc_id = ? "
        "AND status != 'pending' ORDER BY kind, model",
        (doc_id,),
    ):
        models.setdefault(str(kind), []).append(str(model))
    units: dict[str, dict[int, tuple[str, int]]] = {}
    for kind, idx, status, stale in connection.execute(
        "SELECT kind, idx, status, stale FROM units WHERE doc_id = ?", (doc_id,)
    ):
        units.setdefault(str(kind), {})[int(idx)] = (str(status), int(stale))
    outputs = {
        str(path): int(stale)
        for path, stale in connection.execute(
            "SELECT path, stale FROM outputs WHERE doc_id = ?", (doc_id,)
        )
    }
    return _Details(
        input_path=input_path,
        input_sha256=input_sha256,
        models={kind: tuple(names) for kind, names in models.items()},
        units=units,
        outputs=outputs,
    )


def _model_note(
    name: str, details: _Details, models: Mapping[str, str | None], derived: bool
) -> str | None:
    changes = []
    for kind, model in models.items():
        earlier = [used for used in details.models.get(kind, ()) if used != model]
        if model is not None and earlier:
            changes.append(f"{kind} with {', '.join(earlier)}; this run uses {model}")
    if not changes and derived:
        model = models.get(KIND_TRANSCRIPTION)
        changes.append(
            f"{KIND_TRANSCRIPTION} with another model; this run uses "
            f"{model or 'another model'}"
        )
    if not changes:
        return None
    return f"'{name}' holds units made by another model: " + "; ".join(changes) + "."


def _check_input(name: str, details: _Details, input_path: Path | None) -> None:
    recorded = details.input_path
    if input_path is None or not recorded:
        return
    if _normalized(recorded) == _normalized(input_path):
        return
    if Path(recorded).exists():
        raise InputMismatchError(
            f"'{name}' in this output folder belongs to {recorded}, but this run "
            f"would write {input_path} there under the same name. Choose another "
            "--output for one of them."
        )


def _work(
    doc_id: int, details: _Details, summarize: bool
) -> tuple[tuple[UnitKey, ...], tuple[UnitKey, ...]]:
    """Return the transcription and summary units to process.

    A page transcribed again is summarized again; so is a page whose summary
    is missing, pending, failed or stale.
    """
    transcriptions = details.units.get(KIND_TRANSCRIPTION, {})
    redo = {idx for idx, (status, _) in transcriptions.items() if status != "done"}
    transcribe = tuple(UnitKey(doc_id, KIND_TRANSCRIPTION, idx) for idx in sorted(redo))
    if not summarize:
        return transcribe, ()
    summaries = details.units.get(KIND_SUMMARY, {})
    summarize_units = tuple(
        UnitKey(doc_id, KIND_SUMMARY, idx)
        for idx in sorted(transcriptions.keys() | summaries.keys())
        if idx in redo or summaries.get(idx, ("pending", 0)) != ("done", 0)
    )
    return transcribe, summarize_units


@dataclass(frozen=True)
class ResumeRequest:
    """What this run brings to the classification of one item.

    *outputs* are the item's rendered files for this run's formats; *legacy*
    its pre-store working files (the v1.0.x logs). *image_settings* is this
    run's image-settings record, *model* and *summary_model* the models of
    the two roles; *summarize* False leaves summary units out.
    """

    outputs: Sequence[Path] = ()
    legacy: Sequence[Path] = ()
    input_path: Path | None = None
    input_sha256: str | None = None
    image_settings: Mapping[str, Any] | None = None
    model: str | None = None
    summary_model: str | None = None
    summarize: bool = True


def _blocked(resume: ResumeState, holder: str | None, now: float | None) -> str | None:
    moment = time.time() if now is None else now
    if (
        resume.lease_run_id is not None
        and resume.lease_run_id != holder
        and resume.lease_until is not None
        and resume.lease_until > moment
    ):
        return f"leased by run {resume.lease_run_id}"
    if resume.blocked:
        return (
            f"{len(resume.blocked)} unit(s) await an open or unresolved batch "
            "submission"
        )
    return None


def _image_check(recorded: Any, request: ResumeRequest) -> tuple[list[str], bool]:
    """Return the changed user image settings and whether derived keys differ."""
    if not isinstance(recorded, Mapping) or request.image_settings is None:
        return [], False
    before = _plain(recorded)
    current = _plain(request.image_settings)
    changed = changed_image_settings(
        before, current, model_derived_keys(before, current)
    )
    return changed, not changed and before != current


def _complete(
    root: Path, name: str, doc_id: int, details: _Details, request: ResumeRequest
) -> Classification:
    outputs = request.outputs
    missing = tuple(output for output in outputs if not output.is_file())
    stale = any(details.outputs.get(_path_key(output, root)) == 1 for output in outputs)
    render = bool(missing) or stale
    reason = "all units stored"
    if render:
        reason += "; its files are rendered again"
    return Classification(
        ItemState.COMPLETE,
        name,
        doc_id,
        render=render,
        missing=missing,
        reason=reason,
    )


def classify(
    root: Path,
    name: str,
    request: ResumeRequest | None = None,
    *,
    holder: str | None = None,
    now: float | None = None,
) -> Classification:
    """Classify item *name* of output folder *root* from the store; never write.

    An item without stored units but with rendered or working files of
    *request* is ``LEGACY``. The request's input path is compared with the
    stored one before the run touches the document: another input that still
    exists raises :class:`InputMismatchError`. A changed input hash resets
    the document. A lease held by another run than *holder* that has not
    expired at *now* blocks the item, as an open or unresolved batch
    submission does.

    Pending and failed units are processed again; when summarizing, so are
    summary units that are missing or stale. Transcription work makes the
    item partial, summary work alone ``TRANSCRIPTION_ONLY``. A changed user
    image setting (the keys derived from the model left out) refuses an item
    with transcription work; a changed model alone resumes with a note.
    Raises the store's errors for a versionless, newer or unreadable
    database.
    """
    request = request if request is not None else ResumeRequest()
    path = database_path(root)
    state = _schema(root)
    store = None
    if state.exists and not state.empty:
        store = Store(path, TOOL, components=_COMPONENTS)
    resume = store.resume_state(name) if store is not None else None
    doc_id = resume.doc_id if resume is not None else None
    if resume is None or not resume.counts:
        files = (*request.outputs, *request.legacy)
        present = [file for file in files if _nonempty(file)]
        if present:
            return Classification(
                ItemState.LEGACY,
                name,
                doc_id,
                reason=(
                    f"{present[0].name} exists without units in {DATABASE_NAME}; "
                    "migrate imports it, --force processes it again"
                ),
            )
        return Classification(ItemState.NONE, name, doc_id, reason="no stored units")
    assert doc_id is not None

    details = read(path, lambda connection: _details(connection, doc_id))
    _check_input(name, details, request.input_path)
    blocked = _blocked(resume, holder, now)
    if blocked is not None:
        return Classification(ItemState.BLOCKED, name, doc_id, reason=blocked)
    if (
        request.input_sha256 is not None
        and details.input_sha256 is not None
        and request.input_sha256 != details.input_sha256
    ):
        return Classification(
            ItemState.NONE,
            name,
            doc_id,
            reset=True,
            reason="the input changed since its units were stored",
        )

    changed, derived = _image_check(resume.image_settings, request)
    models = {KIND_TRANSCRIPTION: request.model}
    if request.summarize:
        models[KIND_SUMMARY] = request.summary_model
    note = _model_note(name, details, models, derived)

    transcribe, summarize_units = _work(doc_id, details, request.summarize)
    if not transcribe and not summarize_units:
        complete = _complete(root, name, doc_id, details, request)
        return dataclasses.replace(complete, note=note)
    if transcribe and changed:
        return Classification(
            ItemState.REFUSED,
            name,
            doc_id,
            reason=(
                f"Image settings changed for '{name}': {', '.join(changed)}. "
                "Skipping it; use --force to replace the existing run."
            ),
        )
    total = len(details.units.get(KIND_TRANSCRIPTION, {}))
    if not transcribe:
        return Classification(
            ItemState.TRANSCRIPTION_ONLY,
            name,
            doc_id,
            units=summarize_units,
            reason=f"{len(summarize_units)} of {total} page summaries to make",
            note=note,
        )
    return Classification(
        ItemState.PARTIAL,
        name,
        doc_id,
        units=transcribe + summarize_units,
        reason=(
            f"{len(transcribe)} of {total} pages to transcribe, "
            f"{len(summarize_units)} summaries to make"
        ),
        note=note,
    )
