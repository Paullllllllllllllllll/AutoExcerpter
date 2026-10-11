"""AutoExcerpter's store component: views, content writes, stale summaries, stamps.

The view oracles query the documented views through a ``mode=ro`` connection
over a store seeded through the store API and compare columns and rows,
failed and pending units included.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.common.native import image_settings_fingerprint
from autoexcerpter.common.store import (
    Lease,
    PlannedUnit,
    ResponseRow,
    Store,
    UnitKey,
    UnitResult,
    VersionlessDatabaseError,
)
from autoexcerpter.output.store import (
    COMPONENT,
    DATABASE_NAME,
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    TOOL,
    VIEWS,
    CitationRow,
    SummaryRow,
    begin_run,
    database_path,
    document_kind,
    image_settings_record,
    open_store,
    plan_pages,
    read_only_connection,
    read_store,
    settings_fingerprint,
    store_artifacts,
    transcription_outcome,
    write_citations,
    write_summary,
)

MODEL = "gpt-5.6-luna"
SUMMARY_MODEL = "claude-sonnet-5-5"
PROVIDER = "openai"
SUMMARY_PROVIDER = "anthropic"
NAME = "doc"
USAGE = {"input_tokens": 1500, "output_tokens": 400, "total_tokens": 1900}

PAGE_TEXT_COLUMNS = (
    "document",
    "kind",
    "page_index",
    "image_name",
    "order_index",
    "text",
    "status",
    "model",
    "usage",
    "outcome",
    "error",
    "provider",
    "run_id",
)
PAGE_SUMMARIES_COLUMNS = (
    "document",
    "order_index",
    "page",
    "image_name",
    "status",
    "stale",
    "outcome",
    "page_information",
    "bullet_points",
    "page_references",
    "model",
    "usage",
    "error",
    "provider",
    "run_id",
)
INFO = {
    "is_two_page_spread": False,
    "page_number_integer": 11,
    "page_number_integer_end": None,
    "page_number_type": "roman",
    "page_types": ["preface"],
}


@dataclass(frozen=True)
class Page:
    """One page to seed: its transcription state and text."""

    idx: int
    status: str = "done"
    text: str | None = None
    label: str | None = None
    source_index: int | str | None = None
    model: str = MODEL


def page(idx: int, status: str = "done", **changes: Any) -> Page:
    """Return a PDF page with the text its status implies."""
    text = {"done": f"Page {idx + 1} text", "failed": "[transcription error: boom]"}
    values: dict[str, Any] = {
        "idx": idx,
        "status": status,
        "text": text.get(status),
        "label": f"page_{idx + 1:04d}.jpg",
        "source_index": idx,
    }
    values.update(changes)
    return Page(**values)


def transcription(page: Page) -> UnitResult:
    """Return the transcription unit of *page*."""
    error = "boom" if page.status == "failed" else None
    return UnitResult(
        kind=KIND_TRANSCRIPTION,
        idx=page.idx,
        status=page.status,
        model=page.model,
        provider=PROVIDER,
        content=page.text,
        outcome=transcription_outcome(page.text, error),
        label=page.label,
        source_index=page.source_index,
        usage=USAGE if page.status == "done" else None,
        error=error,
        error_type="api_error" if error else None,
    )


def summary(
    idx: int, *, model: str = SUMMARY_MODEL, error: str | None = None
) -> UnitResult:
    """Return the summary unit of page *idx*."""
    return UnitResult(
        kind=KIND_SUMMARY,
        idx=idx,
        status="failed" if error else "done",
        model=model,
        provider=SUMMARY_PROVIDER,
        outcome="failed" if error else "ok",
        error=error,
    )


def summary_row(idx: int) -> SummaryRow:
    """Return a summaries row for page *idx*."""
    return SummaryRow(
        idx,
        page=idx + 1,
        page_information=INFO,
        bullet_points=[f"Point of page {idx + 1}."],
        references=[{"citation": "Allen (2001).", "is_partial": True}],
    )


async def seed_document(
    store: Store,
    pages: Sequence[Page],
    *,
    name: str = NAME,
    kind: str = "pdf",
    summaries: Sequence[int] = (),
    summarize: bool = True,
    image_settings: dict[str, Any] | None = None,
    input_path: str | None = None,
    input_sha256: str | None = None,
) -> Lease:
    """Create document *name*, plan its pages, write the transcriptions not
    pending and the summaries of the pages in *summaries*."""
    doc_id = await store.ensure_document(
        name,
        input_path=input_path if input_path is not None else f"{name}.pdf",
        input_sha256=input_sha256,
        kind=kind,
    )
    lease = await store.acquire(doc_id)
    assert lease is not None
    planned = [
        PlannedUnit(p.idx, label=p.label, source_index=p.source_index) for p in pages
    ]
    await plan_pages(
        store,
        lease,
        planned,
        transcription=(MODEL, PROVIDER),
        summary=(SUMMARY_MODEL, SUMMARY_PROVIDER) if summarize else None,
    )
    if image_settings is not None:
        await store.set_image_settings(lease, image_settings)
    for item in pages:
        if item.status != "pending":
            await store.write_unit(lease, transcription(item))
    for idx in summaries:
        await write_summary(store, lease, summary(idx), summary_row(idx))
    return lease


def build_store(root: Path, pages: Sequence[Page], **options: Any) -> Path:
    """Seed one document in *root*'s store; return the database path."""

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("run")
            await seed_document(store, pages, **options)
            await store.end_run(0)

    asyncio.run(main())
    return database_path(root)


def query(database: Path, sql: str) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    """Run *sql* on a read-only connection; return the columns and rows."""
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(sql)
        columns = tuple(description[0] for description in cursor.description)
        return columns, cursor.fetchall()
    finally:
        connection.close()


def test_the_component_adds_tables_and_views_at_version_1(tmp_path: Path) -> None:
    database = build_store(tmp_path, [page(0)])

    _, meta = query(database, "SELECT component, version FROM meta ORDER BY 1")
    _, views = query(database, "SELECT name FROM sqlite_master WHERE type = 'view'")
    _, tables = query(database, "SELECT name FROM sqlite_master WHERE type = 'table'")

    assert database.name == DATABASE_NAME
    assert COMPONENT.name == TOOL == "autoexcerpter"
    assert meta == [("autoexcerpter", 1), ("core", 1)]
    assert sorted(name for (name,) in views) == sorted(VIEWS)
    assert {"summaries", "citations"} <= {name for (name,) in tables}


def test_page_text_returns_the_transcriptions_in_page_order(tmp_path: Path) -> None:
    pdf = [page(0), page(1, "failed"), page(2, text="[Empty page]"), page(3, "pending")]
    images = [
        Page(0, text="Ledger one", label="a.png", source_index="a.png"),
        Page(1, text="Ledger two", label="b.png", source_index="b.png"),
    ]

    async def main() -> str:
        async with open_store(tmp_path) as store:
            run_id = await store.begin_run("run")
            await seed_document(store, pdf, summaries=[0])
            await seed_document(store, images, name="ledger", kind="images")
            return run_id

    run_id = asyncio.run(main())
    columns, rows = query(database_path(tmp_path), "SELECT * FROM page_text")

    assert columns == PAGE_TEXT_COLUMNS
    usage = json.dumps(USAGE)
    assert rows == [
        (NAME, "transcription", 0, "page_0001.jpg", 0, "Page 1 text", "done",
         MODEL, usage, "ok", None, PROVIDER, run_id),
        (NAME, "transcription", 1, "page_0002.jpg", 1, "[transcription error: boom]",
         "failed", MODEL, None, "failed", "boom", PROVIDER, run_id),
        (NAME, "transcription", 2, "page_0003.jpg", 2, "[Empty page]", "done",
         MODEL, usage, "blank", None, PROVIDER, run_id),
        (NAME, "transcription", 3, "page_0004.jpg", 3, None, "pending",
         MODEL, None, None, None, PROVIDER, run_id),
        ("ledger", "transcription", None, "a.png", 0, "Ledger one", "done",
         MODEL, usage, "ok", None, PROVIDER, run_id),
        ("ledger", "transcription", None, "b.png", 1, "Ledger two", "done",
         MODEL, usage, "ok", None, PROVIDER, run_id),
    ]  # fmt: skip


def test_page_summaries_join_the_units_and_their_rows(tmp_path: Path) -> None:
    async def main() -> str:
        async with open_store(tmp_path) as store:
            run_id = await store.begin_run("run")
            lease = await seed_document(
                store, [page(0), page(1), page(2)], summaries=[0]
            )
            await write_summary(store, lease, summary(1, error="refused"), None)
            return run_id

    run_id = asyncio.run(main())
    columns, rows = query(database_path(tmp_path), "SELECT * FROM page_summaries")

    assert columns == PAGE_SUMMARIES_COLUMNS
    assert rows == [
        (NAME, 0, 1, "page_0001.jpg", "done", 0, "ok", json.dumps(INFO),
         json.dumps(["Point of page 1."]),
         json.dumps([{"citation": "Allen (2001).", "is_partial": True}]),
         SUMMARY_MODEL, None, None, SUMMARY_PROVIDER, run_id),
        (NAME, 1, None, "page_0002.jpg", "failed", 0, "failed", None, None, None,
         SUMMARY_MODEL, None, "refused", SUMMARY_PROVIDER, run_id),
        (NAME, 2, None, "page_0003.jpg", "pending", 0, None, None, None, None,
         SUMMARY_MODEL, None, None, SUMMARY_PROVIDER, run_id),
    ]  # fmt: skip


def test_a_summary_row_commits_before_its_unit(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("run")
            lease = await seed_document(store, [page(0)])
            calls: list[str] = []
            original = store.write_unit

            async def failing(lease: Lease, result: UnitResult, **_: object) -> None:
                calls.append(result.kind)
                raise sqlite3.OperationalError("disk I/O error")

            store.write_unit = failing  # type: ignore[method-assign]
            with pytest.raises(sqlite3.OperationalError):
                await write_summary(store, lease, summary(0), summary_row(0))
            store.write_unit = original  # type: ignore[method-assign]
            assert calls == [KIND_SUMMARY]
            with pytest.raises(ValueError, match="summary unit"):
                await write_summary(store, lease, transcription(page(0)), None)

    asyncio.run(main())
    _, rows = query(database_path(tmp_path), "SELECT page, status FROM page_summaries")
    assert rows == [(1, "pending")]


def test_a_changed_transcription_marks_its_summary_stale(tmp_path: Path) -> None:
    database = database_path(tmp_path)

    def stale() -> list[tuple[int, int]]:
        return query(database, "SELECT order_index, stale FROM page_summaries")[1]

    async def main() -> list[list[tuple[int, int]]]:
        seen = []
        async with open_store(tmp_path) as store:
            await store.begin_run("run")
            lease = await seed_document(
                store, [page(0), page(1, "failed"), page(2)], summaries=[0, 1]
            )
            seen.append(stale())
            await store.write_unit(lease, transcription(page(0)))
            await store.write_unit(lease, transcription(page(2, text="Retold")))
            seen.append(stale())
            await store.write_unit(lease, transcription(page(1)))
            seen.append(stale())
            await write_summary(store, lease, summary(1), summary_row(1))
            await store.reset_units(lease, kind=KIND_TRANSCRIPTION, idxs=[0])
            seen.append(stale())
        return seen

    assert asyncio.run(main()) == [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (1, 1), (2, 0)],
        [(0, 1), (1, 0), (2, 0)],
    ]


def test_citations_are_replaced_in_rendered_order(tmp_path: Path) -> None:
    first = [
        CitationRow("Allen (2001).", pages="1, 3", partial=True),
        CitationRow(
            "Montanari, M. (1994). The Culture of Food.",
            pages="3",
            doi="10.1000/x",
            url="https://doi.org/10.1000/x",
            metadata={"title": "The Culture of Food", "year": 1994},
        ),
    ]

    async def main() -> list[list[tuple[Any, ...]]]:
        seen = []
        async with open_store(tmp_path) as store:
            await store.begin_run("run")
            lease = await seed_document(store, [page(0)])
            await write_citations(store, lease, first)
            seen.append(
                query(database_path(tmp_path), "SELECT * FROM document_citations")[1]
            )
            await write_citations(store, lease, first[1:])
            seen.append(
                query(database_path(tmp_path), "SELECT * FROM document_citations")[1]
            )
        return seen

    before, after = asyncio.run(main())
    metadata = json.dumps({"title": "The Culture of Food", "year": 1994})
    montanari = (
        "Montanari, M. (1994). The Culture of Food.", "3", 0, "10.1000/x",
        "https://doi.org/10.1000/x", metadata,
    )  # fmt: skip
    assert before == [
        (NAME, 1, "Allen (2001).", "1, 3", 1, None, None, None),
        (NAME, 2, *montanari),
    ]
    assert after == [(NAME, 1, *montanari)]


def test_document_status_and_page_attempts_with_roles(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            await store.begin_run("run")
            lease = await seed_document(store, [page(0), page(1, "failed")])
            for kind, role in (
                (KIND_TRANSCRIPTION, "transcription"),
                (KIND_SUMMARY, "summary"),
            ):
                await store.append_response(
                    ResponseRow(
                        provider=PROVIDER,
                        model=MODEL,
                        unit=UnitKey(lease.doc_id, kind, 1),
                        role=role,
                        attempt=1,
                        raw='{"error": "invalid request"}',
                        raw_kind="http",
                        error="Error code: 400",
                        http_status=400,
                    )
                )

    asyncio.run(main())
    database = database_path(tmp_path)
    _, status = query(database, "SELECT * FROM document_status")
    _, attempts = query(
        database,
        "SELECT document, kind, order_index, role, raw, raw_kind, http_status "
        "FROM page_attempts",
    )

    assert [row[:8] for row in status] == [
        (NAME, "pdf", "doc.pdf", "partial", 4, 1, 1, 2)
    ]
    raw = '{"error": "invalid request"}'
    assert attempts == [
        (NAME, "transcription", 1, "transcription", raw, "http", 400),
        (NAME, "summary", 1, "summary", raw, "http", 400),
    ]


def test_read_helpers_open_short_read_only_connections(tmp_path: Path) -> None:
    assert read_store(tmp_path) is None
    with pytest.raises(FileNotFoundError), read_only_connection(tmp_path):
        pass

    build_store(tmp_path, [page(0)], summaries=[0])
    store = read_store(tmp_path)
    assert store is not None
    document = store.document(NAME)
    assert document is not None and document.status == "complete"
    with read_only_connection(tmp_path) as connection:
        count = connection.execute("SELECT count(*) FROM page_summaries").fetchone()[0]
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM summaries")
    assert count == 1


def test_read_helpers_refuse_a_versionless_database(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / DATABASE_NAME)
    connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
    connection.execute("CREATE TABLE pages (document TEXT, page_index INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(VersionlessDatabaseError):
        read_store(tmp_path)
    with pytest.raises(VersionlessDatabaseError), read_only_connection(tmp_path):
        pass


def test_plan_pages_without_summary_plans_transcriptions_only(tmp_path: Path) -> None:
    database = build_store(
        tmp_path, [page(0, "pending"), page(1, "pending")], summarize=False
    )

    _, units = query(database, "SELECT kind, idx, model, provider FROM units")

    assert units == [
        (KIND_TRANSCRIPTION, 0, MODEL, PROVIDER),
        (KIND_TRANSCRIPTION, 1, MODEL, PROVIDER),
    ]


def test_runs_carry_the_spec_fingerprint_and_artifacts(tmp_path: Path) -> None:
    spec = {
        "model.name": {"value": MODEL, "source": "settings"},
        "output.directory": {"value": "out", "source": "cli"},
    }
    moved = {**spec, "output.directory": {"value": "elsewhere", "source": "cli"}}

    async def main() -> tuple[str, dict[str, str]]:
        async with open_store(tmp_path) as store:
            run_id = await begin_run(store, "autoexcerpter run --input doc.pdf", spec)
            digests = await store_artifacts(
                store, {"transcription_prompt": "Transcribe.", "context": None}
            )
            await store.end_run(0)
            return run_id, digests

    run_id, digests = asyncio.run(main())
    store = read_store(tmp_path)
    assert store is not None
    run = store.run(run_id)

    assert run is not None
    assert run["tool"] == TOOL
    assert run["command"] == "autoexcerpter run --input doc.pdf"
    assert run["spec"] == spec
    assert run["settings_fingerprint"] == settings_fingerprint(spec)
    assert settings_fingerprint(moved) == settings_fingerprint(spec)
    assert run["exit_code"] == 0
    assert list(digests) == ["transcription_prompt"]
    _, artifacts = query(database_path(tmp_path), "SELECT kind, text FROM artifacts")
    assert artifacts == [("transcription_prompt", "Transcribe.")]


def test_image_settings_record_matches_the_provenance_fingerprint() -> None:
    settings = {
        "jpeg_quality": 95,
        "high_target_box": (768, 1536),
        "model_name": MODEL,
        "provider": PROVIDER,
    }

    record = image_settings_record(settings, "openai")

    assert record["high_target_box"] == [768, 1536]
    assert record["model_type"] == "openai"
    assert image_settings_fingerprint(record) == image_settings_fingerprint(
        {**settings, "model_type": "openai"}
    )
    assert image_settings_record({"model_type": "google"})["model_type"] == "google"


@pytest.mark.parametrize(
    ("text", "error", "outcome"),
    [
        ("Some text", None, "ok"),
        ("[Empty page]", None, "blank"),
        ("[No transcribable text]", None, "blank"),
        ("[transcription error: boom]", "boom", "failed"),
    ],
)
def test_transcription_outcome(text: str, error: str | None, outcome: str) -> None:
    assert transcription_outcome(text, error) == outcome


def test_document_kind_names_pdfs_and_image_folders() -> None:
    assert document_kind("pdf") == "pdf"
    assert document_kind("image_folder") == "images"
