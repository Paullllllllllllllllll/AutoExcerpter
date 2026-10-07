"""The resume-state matrix: on-disk starting states and their classification.

Each row writes an item's outputs, working logs and database rows in
``tmp_path`` through the real writers and pins what :class:`ResumeChecker`
makes of them. Rows marked ``cli`` also run through ``autoexcerpter run
--dry-run --json``, whose plan must report the same state.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter import __main__ as entry
from autoexcerpter.common.jsonl import file_identity, folder_identity
from autoexcerpter.pipeline.log import (
    LogHeader,
    append_to_log,
    finalize_log_file,
    initialize_log_file,
)
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.resume import ProcessingState, ResumeChecker, ResumeResult
from autoexcerpter.rendering.sqlite import DATABASE_NAME, open_database

MakePdf = Callable[..., Path]
Entry = dict[str, Any]

NAME = "doc"
PAGES = 2
MODEL = "gpt-5-mini"
ALL = ("transcription", "summary_docx", "summary_md")
SUMMARIES = ("summary_docx", "summary_md")

NONE = ProcessingState.NONE
PARTIAL = ProcessingState.PARTIAL
TRANSCRIPTION_ONLY = ProcessingState.TRANSCRIPTION_ONLY
COMPLETE = ProcessingState.COMPLETE

TRANSCRIPTION_EXISTS = "transcription exists, missing: "
DATABASE_REASON = (
    f"no complete row for '{NAME}' in {DATABASE_NAME}",
    "finishing it from the working logs",
)
NO_ROW_WARNING = (
    f"'{NAME}' has no complete row in",
    "no working logs to rebuild it from; --force rebuilds it.",
)
INPUT_CHANGED = "(input changed since log; reprocessing from scratch)"


def ok(*indices: int) -> tuple[Entry, ...]:
    """Return transcription entries of pages *indices*."""
    return tuple(
        {
            "original_input_order_index": index,
            "image": f"page_{index + 1:04d}.jpg",
            "transcription": f"Text of page {index + 1}.",
        }
        for index in indices
    )


def failed(*indices: int) -> tuple[Entry, ...]:
    """Return failed transcription or summary entries of pages *indices*."""
    return tuple(
        {
            "original_input_order_index": index,
            "image": f"page_{index + 1:04d}.jpg",
            "error": "refused",
        }
        for index in indices
    )


def summarized(*indices: int) -> tuple[Entry, ...]:
    """Return summary entries of pages *indices*."""
    return tuple(
        {"original_input_order_index": index, "bullet_points": ["A point."]}
        for index in indices
    )


@dataclass(frozen=True)
class Row:
    """One starting state and the checker result it must give.

    *outputs* and *empty* name ``ItemPaths`` fields written with content and
    empty. A log of None is absent and an empty log holds the header only.
    *change* is ``"size"`` (the PDF grew) or ``"images"`` (an image joined
    the input folder) after the logs were written. *database* None turns the
    database check off; otherwise the folder's database holds a ``complete``
    or ``incomplete`` row for the item, a row for an ``other`` item, or is
    ``absent``.
    """

    id: str
    state: ProcessingState
    pages: frozenset[int] | None = None
    reason: tuple[str, ...] = ()
    warning: tuple[str, ...] | None = None
    missing: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    empty: tuple[str, ...] = ()
    transcription_log: tuple[Entry, ...] | None = None
    summary_log: tuple[Entry, ...] | None = None
    total: int = PAGES
    change: str | None = None
    summarize: bool = True
    docx: bool = True
    markdown: bool = True
    retranscribe: bool = False
    database: str | None = None
    force: bool = False
    cli: bool = False


def pages(*indices: int) -> frozenset[int]:
    return frozenset(indices)


ROWS = (
    Row(
        "force_reprocesses_a_complete_item",
        NONE,
        reason=("overwrite mode",),
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        force=True,
        cli=True,
    ),
    Row(
        "none_without_outputs_or_logs",
        NONE,
        reason=("no output",),
        missing=ALL,
        cli=True,
    ),
    Row(
        "none_header_only_log",
        NONE,
        reason=("no output",),
        missing=ALL,
        transcription_log=(),
    ),
    Row(
        "none_only_error_entries",
        NONE,
        reason=("no output",),
        missing=ALL,
        transcription_log=failed(0, 1),
    ),
    Row(
        "none_empty_transcription_file",
        NONE,
        missing=("transcription",),
        outputs=SUMMARIES,
        empty=("transcription",),
    ),
    Row(
        "none_changed_input_drops_logged_pages",
        NONE,
        reason=("no output",),
        missing=ALL,
        transcription_log=ok(0),
        change="size",
    ),
    Row(
        "partial_log_shortfall",
        PARTIAL,
        pages(0, 1),
        reason=("partial log with 2 completed page(s)",),
        missing=ALL,
        transcription_log=ok(0, 1),
        total=3,
        cli=True,
    ),
    Row(
        "partial_log_shortfall_retranscribe",
        PARTIAL,
        reason=("partial log with 2 completed page(s)",),
        missing=ALL,
        transcription_log=ok(0, 1),
        total=3,
        retranscribe=True,
        cli=True,
    ),
    Row(
        "partial_log_with_error_entry",
        PARTIAL,
        pages(0, 2),
        reason=("partial log with 2 completed page(s)",),
        missing=ALL,
        transcription_log=ok(0) + failed(1) + ok(2),
        total=3,
    ),
    Row(
        "partial_summary_files_without_transcription",
        PARTIAL,
        pages(0),
        missing=("transcription",),
        outputs=SUMMARIES,
        transcription_log=ok(0),
    ),
    Row(
        "partial_without_summaries_log_shortfall",
        PARTIAL,
        pages(0),
        reason=("partial log with 1 completed page(s)",),
        outputs=("transcription",),
        transcription_log=ok(0),
        summarize=False,
    ),
    Row(
        "partial_without_summaries_database_absent",
        PARTIAL,
        pages(0, 1),
        reason=DATABASE_REASON,
        outputs=("transcription",),
        transcription_log=ok(0, 1),
        summarize=False,
        database="absent",
        cli=True,
    ),
    Row(
        "transcription_only_without_logs",
        TRANSCRIPTION_ONLY,
        reason=(TRANSCRIPTION_EXISTS,),
        missing=SUMMARIES,
        outputs=("transcription",),
        cli=True,
    ),
    Row(
        "transcription_only_with_logs",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=(TRANSCRIPTION_EXISTS,),
        missing=SUMMARIES,
        outputs=("transcription",),
        transcription_log=ok(0, 1),
        summary_log=summarized(0),
        cli=True,
    ),
    Row(
        "transcription_only_with_logs_retranscribe",
        TRANSCRIPTION_ONLY,
        reason=(TRANSCRIPTION_EXISTS,),
        missing=SUMMARIES,
        outputs=("transcription",),
        transcription_log=ok(0, 1),
        summary_log=summarized(0),
        retranscribe=True,
        cli=True,
    ),
    Row(
        "transcription_only_markdown_summary_missing",
        TRANSCRIPTION_ONLY,
        missing=("summary_md",),
        outputs=("transcription", "summary_docx"),
    ),
    Row(
        "transcription_only_docx_format_docx_missing",
        TRANSCRIPTION_ONLY,
        missing=("summary_docx",),
        outputs=("transcription", "summary_md"),
        markdown=False,
    ),
    Row(
        "transcription_only_log_shortfall",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=("(log shows 2/3 pages; resuming missing pages)",),
        outputs=ALL,
        transcription_log=ok(0, 1),
        total=3,
    ),
    Row(
        "transcription_only_log_shortfall_retranscribe",
        TRANSCRIPTION_ONLY,
        reason=("(log shows 2/3 pages; resuming missing pages)",),
        outputs=ALL,
        transcription_log=ok(0, 1),
        total=3,
        retranscribe=True,
    ),
    Row(
        "transcription_only_header_only_log",
        TRANSCRIPTION_ONLY,
        reason=("(log shows 0/2 pages; resuming missing pages)",),
        outputs=ALL,
        transcription_log=(),
    ),
    Row(
        "transcription_only_duplicate_entries_for_one_page",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=("(log shows 2/3 pages; resuming missing pages)",),
        outputs=ALL,
        transcription_log=ok(0, 0, 1),
        total=3,
    ),
    Row(
        "transcription_only_transcription_error_entry",
        TRANSCRIPTION_ONLY,
        pages(0),
        reason=(
            "(1 failed transcription page(s), 0 failed summary page(s); retrying them)",
        ),
        outputs=ALL,
        transcription_log=ok(0) + failed(1),
        summary_log=summarized(0, 1),
    ),
    Row(
        "transcription_only_summary_error_entry",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=(
            "(0 failed transcription page(s), 1 failed summary page(s); retrying them)",
        ),
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0) + failed(1),
    ),
    Row(
        "transcription_only_input_file_size_changed",
        TRANSCRIPTION_ONLY,
        reason=(INPUT_CHANGED,),
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        change="size",
    ),
    Row(
        "transcription_only_input_image_set_changed",
        TRANSCRIPTION_ONLY,
        reason=(INPUT_CHANGED,),
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        change="images",
    ),
    Row(
        "transcription_only_database_row_missing",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=DATABASE_REASON,
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        database="other",
        cli=True,
    ),
    Row(
        "transcription_only_database_row_incomplete",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=DATABASE_REASON,
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        database="incomplete",
    ),
    Row(
        "transcription_only_database_absent",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=DATABASE_REASON,
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        database="absent",
    ),
    Row(
        "transcription_only_database_row_missing_retranscribe",
        TRANSCRIPTION_ONLY,
        pages(0, 1),
        reason=DATABASE_REASON,
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        retranscribe=True,
        database="other",
        cli=True,
    ),
    Row(
        "complete_outputs_without_logs",
        COMPLETE,
        reason=("all outputs exist: ",),
        outputs=ALL,
    ),
    Row(
        "complete_with_full_logs",
        COMPLETE,
        reason=("all outputs exist: ",),
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        cli=True,
    ),
    Row(
        "complete_without_summary_log",
        COMPLETE,
        outputs=ALL,
        transcription_log=ok(0, 1),
    ),
    Row(
        "complete_without_summaries",
        COMPLETE,
        outputs=("transcription",),
        summarize=False,
    ),
    Row(
        "complete_without_summaries_ignores_summary_errors",
        COMPLETE,
        outputs=("transcription",),
        transcription_log=ok(0, 1),
        summary_log=summarized(0) + failed(1),
        summarize=False,
    ),
    Row(
        "complete_docx_format",
        COMPLETE,
        outputs=("transcription", "summary_docx"),
        markdown=False,
    ),
    Row(
        "complete_markdown_format",
        COMPLETE,
        outputs=("transcription", "summary_md"),
        docx=False,
    ),
    Row(
        "complete_database_row",
        COMPLETE,
        outputs=ALL,
        transcription_log=ok(0, 1),
        summary_log=summarized(0, 1),
        database="complete",
        cli=True,
    ),
    Row(
        "complete_database_row_missing_without_logs",
        COMPLETE,
        warning=NO_ROW_WARNING,
        outputs=ALL,
        database="other",
        cli=True,
    ),
    Row(
        "complete_database_absent_without_logs",
        COMPLETE,
        warning=NO_ROW_WARNING,
        outputs=ALL,
        database="absent",
    ),
)

CLI_ROWS = tuple(row for row in ROWS if row.cli)
RETRANSCRIBE_ROWS = tuple(
    row for row in ROWS if row.retranscribe and row.database is None
)


def _row_id(row: Row) -> str:
    return row.id


def _input(row: Row, tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, Entry, str]:
    """Write the item's input; return it, its provenance and its type label."""
    if row.change == "images":
        folder = tmp_path / NAME
        folder.mkdir()
        files = [folder / f"page_{index + 1:04d}.png" for index in range(PAGES)]
        for file in files:
            file.write_bytes(b"image bytes")
        return folder, folder_identity(folder, files), "Image Folder"
    pdf = make_pdf(f"{NAME}.pdf", num_pages=PAGES)
    return pdf, file_identity(pdf), "PDF"


def _write_log(path: Path, entries: tuple[Entry, ...], header: LogHeader) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert initialize_log_file(path, header)
    for record in entries:
        assert append_to_log(path, record)
    finalize_log_file(path)


def _write_documents(out: Path, row: Row, source: Path) -> None:
    document = {
        "name": "other" if row.database == "other" else NAME,
        "path": str(source),
        "kind": "pdf",
        "complete": row.database != "incomplete",
    }

    async def write() -> None:
        async with open_database(out) as writer:
            await writer.upsert("documents", [document])

    asyncio.run(write())


def _build(row: Row, tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, Path]:
    """Write the row's starting state; return the input and the output folder."""
    out = tmp_path / "out"
    out.mkdir()
    source, provenance, type_label = _input(row, tmp_path, make_pdf)
    paths = ItemPaths.for_item(NAME, out)
    for field in row.outputs:
        getattr(paths, field).write_text("content", encoding="utf-8")
    for field in row.empty:
        getattr(paths, field).write_text("", encoding="utf-8")
    for log_type, entries, path in (
        ("transcription", row.transcription_log, paths.transcription_log),
        ("summary", row.summary_log, paths.summary_log),
    ):
        if entries is not None:
            header = LogHeader(
                item_name=NAME,
                input_path=str(source),
                input_type=type_label,
                total_images=row.total,
                model_name=MODEL,
                file_provenance=provenance,
                log_type=log_type,
            )
            _write_log(path, entries, header)
    if row.database not in (None, "absent"):
        _write_documents(out, row, source)
    if row.change == "size":
        with source.open("ab") as handle:
            handle.write(b"\n% appended\n")
    elif row.change == "images":
        (source / "page_0003.png").write_bytes(b"image bytes")
    return source, out


def _checker(row: Row, *, retranscribe: bool | None = None) -> ResumeChecker:
    return ResumeChecker(
        resume_mode="overwrite" if row.force else "skip",
        summarize=row.summarize,
        output_docx=row.docx,
        output_markdown=row.markdown,
        retranscribe=row.retranscribe if retranscribe is None else retranscribe,
        check_database=row.database is not None,
    )


def _logged(entries: tuple[Entry, ...] | None) -> list[Entry] | None:
    return list(entries) if entries else None


def _assert_log_data(row: Row, result: ResumeResult) -> None:
    """Resumable states carry the parsed logs; the others carry none."""
    if result.state not in (PARTIAL, TRANSCRIPTION_ONLY):
        assert result.transcription_results is None
        assert result.summary_results is None
        assert result.log_header is None
        return
    assert result.transcription_results == _logged(row.transcription_log)
    summaries = _logged(row.summary_log) if row.summarize else None
    assert result.summary_results == summaries
    assert (result.log_header is not None) is (row.transcription_log is not None)


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_resume_state_matrix(row: Row, tmp_path: Path, make_pdf: MakePdf) -> None:
    _, out = _build(row, tmp_path, make_pdf)

    result = _checker(row).should_skip(NAME, out)

    paths = ItemPaths.for_item(NAME, out)
    assert result.state is row.state
    assert result.completed_page_indices == row.pages
    for fragment in row.reason:
        assert fragment in result.reason
    if row.warning is None:
        assert result.warning is None
    else:
        assert result.warning is not None
        for fragment in row.warning:
            assert fragment in result.warning
    assert result.missing_outputs == [getattr(paths, field) for field in row.missing]
    _assert_log_data(row, result)
    written = row.database not in (None, "absent")
    assert (out / DATABASE_NAME).exists() is written


@pytest.mark.parametrize("row", RETRANSCRIBE_ROWS, ids=_row_id)
def test_retranscribe_keeps_the_state_and_reuses_no_page(
    row: Row, tmp_path: Path, make_pdf: MakePdf
) -> None:
    _, out = _build(row, tmp_path, make_pdf)

    reused = _checker(row, retranscribe=False).should_skip(NAME, out)
    fresh = _checker(row, retranscribe=True).should_skip(NAME, out)

    assert fresh.state is reused.state
    assert reused.completed_page_indices
    assert fresh.completed_page_indices is None


def _argv(row: Row, source: Path, out: Path) -> list[str]:
    formats = [
        name
        for name, enabled in (
            ("summary-md", row.markdown),
            ("summary-docx", row.docx),
            ("sqlite", row.database is not None),
        )
        if enabled
    ]
    argv = ["run", "--input", str(source), "--output", str(out)]
    argv += ["--formats", ",".join(formats), "--dry-run", "--json"]
    if not row.summarize:
        argv.append("--no-summarize")
    if row.retranscribe:
        argv.append("--retranscribe")
    if row.force:
        argv.append("--force")
    return argv


def test_every_state_reaches_the_dry_run_plan() -> None:
    assert {row.state for row in CLI_ROWS} == set(ProcessingState)


@pytest.mark.parametrize("row", CLI_ROWS, ids=_row_id)
def test_the_dry_run_plan_reports_the_matrix_state(
    row: Row,
    tmp_path: Path,
    make_pdf: MakePdf,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, out = _build(row, tmp_path, make_pdf)

    assert entry.main(_argv(row, source, out)) == 0

    captured = capsys.readouterr()
    (payload,) = [json.loads(line) for line in captured.out.splitlines()]
    if row.state is COMPLETE:
        assert payload["to_process"] == []
        assert payload["skipped"] == [NAME]
    else:
        planned = {
            "name": NAME,
            "state": row.state.value,
            "completed_pages": len(row.pages or ()),
        }
        assert payload["to_process"] == [planned]
        assert payload["skipped"] == []
    for fragment in row.warning or ():
        assert fragment in captured.err
