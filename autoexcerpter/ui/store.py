"""The render and drop tasks of the guided UI.

:func:`render_flow` renders the files of the stored documents at an input
path again; :func:`drop_flow` deletes stored documents and their files.
Their runs report like ``autoexcerpter render`` and ``drop`` and return the
same exit codes.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rich.console import Console

from autoexcerpter.common.contract import ExitCode
from autoexcerpter.common.options import OptionTable
from autoexcerpter.common.selection import is_numeric_selection
from autoexcerpter.common.store import DocumentSelection, DropReport, StoreError
from autoexcerpter.common.wizard import (
    BACK,
    DONE,
    Choice,
    Line,
    Nav,
    Review,
    State,
    Step,
    StepContext,
)
from autoexcerpter.output.drop import (
    KEPT_FILE_HINT,
    DropInputError,
    drop_documents,
    drop_exit_code,
    find_database,
    select_stored,
)
from autoexcerpter.output.render import (
    DocumentRender,
    find_databases,
    preview_database,
    render_database,
)
from autoexcerpter.output.store import DATABASE_NAME, read_store
from autoexcerpter.spec import (
    DROP_OPTIONS,
    RENDER_FORMATS,
    RENDER_OPTIONS,
    DropSpec,
    RenderSpec,
    SpecError,
    resolve_drop,
    resolve_render,
)
from autoexcerpter.ui.session import (
    TASKS,
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    command_words,
    plural,
)

__all__ = [
    "DROP_PROGRAM",
    "RENDER_PROGRAM",
    "drop_flow",
    "render_flow",
]

logger = logging.getLogger(__name__)

RENDER_PROGRAM: Final = ("autoexcerpter", "render")
DROP_PROGRAM: Final = ("autoexcerpter", "drop")
LIST_LIMIT: Final = 20
"""The drop task lists at most this many stored documents."""

_KEEP = "keep"
_CHOOSE = "choose"


def _path_text(text: str) -> str | None:
    return "enter a path" if not text.strip() else None


def _input_step(message: str, validate: Callable[[str], str | None]) -> Step:
    def ask(ctx: StepContext) -> Nav:
        current = ctx.state.value("input")
        answer = ctx.prompter.path(
            message,
            default="" if current is None else str(absolute(current)),
            validate=validate,
        )
        if answer is BACK:
            return BACK
        ctx.state.set("input", absolute(answer))
        return DONE

    return Step("input", "Input", ask=ask, sets=("input",))


def _flag_step(
    name: str, title: str, message: str, off: Choice[bool], on: Choice[bool]
) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return ctx.select_option(name, message, [off, on])

    return Step(name, title, ask=ask, sets=(name,))


def _dry_run_step(write: str) -> Step:
    return _flag_step(
        "dry_run",
        "Dry run",
        "Dry run:",
        Choice(False, "No", write),
        Choice(True, "Yes", "show the plan, write nothing"),
    )


def _formats_step() -> Step:
    def ask(ctx: StepContext) -> Nav:
        state = ctx.state
        chosen: dict[str, str] = {}

        def mode() -> Nav:
            current = _KEEP if state.value("formats") is None else _CHOOSE
            answer = ctx.prompter.select(
                "Formats:",
                [
                    Choice(_KEEP, "As rendered before", "each document's formats"),
                    Choice(_CHOOSE, "Choose formats"),
                ],
                default=current,
            )
            if answer is BACK:
                return BACK
            chosen["mode"] = answer
            if answer == _KEEP:
                state.set("formats", None)
            return DONE

        def formats() -> Nav | None:
            if chosen.get("mode") != _CHOOSE:
                return None
            checked = state.value("formats") or RENDER_FORMATS[:1]
            while True:
                answer = ctx.prompter.checkbox(
                    "Formats to render:",
                    [Choice(fmt, fmt) for fmt in RENDER_FORMATS],
                    checked=tuple(checked),
                )
                if answer is BACK:
                    return BACK
                if answer:
                    state.set("formats", tuple(answer))
                    return DONE
                ctx.prompter.show("  choose at least one format")

        return StepContext.sequence([mode, formats])

    return Step("formats", "Formats", ask=ask, sets=("formats",))


def _command(
    program: tuple[str, ...], table: OptionTable
) -> Callable[[State, Session], list[str]]:
    def command(state: State, session: Session) -> list[str]:
        answers = {
            name: value
            for name, value in state.answers.items()
            if name != "settings" and value is not None
        }
        return command_words(program, table.to_argv(answers), session, table)

    return command


def _say(console: Console, text: str) -> None:
    console.print(text, markup=False, highlight=False, soft_wrap=True)


def _guarded(console: Console, body: Callable[[], int]) -> int:
    """Run *body* with the exit codes of the CLI for refusals and crashes."""
    try:
        return body()
    except StoreError as exc:
        _say(console, f"Error: {exc}")
        return exc.exit_code
    except KeyboardInterrupt:
        _say(console, "Interrupted by the user.")
        return ExitCode.INTERRUPTED
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        _say(console, f"Unexpected error: {exc}")
        return ExitCode.FAILURE


# =============================================================================
# render
# =============================================================================


def _render_refusal(path: Path) -> str | None:
    """Return why render cannot run on *path*, or None."""
    if not path.exists():
        return f"Input path does not exist: {path}"
    if path.is_file() and path.name != DATABASE_NAME:
        return f"{path} is not a {DATABASE_NAME} file"
    return None


@dataclass(frozen=True)
class _RenderPreview:
    databases: list[Path]
    results: list[DocumentRender]
    refused: str | None = None


def _preview_render(path: Path, spec: RenderSpec) -> _RenderPreview:
    databases = find_databases(path)
    results: list[DocumentRender] = []
    for database in databases:
        try:
            results.extend(
                preview_database(database, formats=spec.formats, force=spec.force)
            )
        except (StoreError, sqlite3.Error, OSError) as exc:
            return _RenderPreview(databases, results, refused=str(exc))
    return _RenderPreview(databases, results)


def _render_review(state: State, session: Session) -> Review:
    try:
        spec = resolve_render(state.answers)
    except SpecError as exc:
        return Review([Line("input", [str(state.value("input"))])], withheld=str(exc))
    path = absolute(spec.input)
    formats = ", ".join(spec.formats) if spec.formats else "as rendered before"
    force = (
        "replace files changed outside the tool"
        if spec.force
        else "keep files changed outside the tool"
    )
    lines = [
        Line("formats", [formats]),
        Line("force", [force]),
        Line("run", ["dry run" if spec.dry_run else "write the files"]),
    ]
    refusal = _render_refusal(path)
    if refusal is not None:
        return Review([Line("input", [str(path)]), *lines], withheld=refusal)
    preview = _preview_render(path, spec)
    found = (
        f"{plural(len(preview.databases), 'database')}, "
        f"{plural(len(preview.results), 'document')}"
    )
    lines.insert(0, Line("input", [str(path), found]))
    if preview.refused is not None:
        return Review(lines, withheld=preview.refused)
    written = sum(len(result.written) for result in preview.results)
    unchanged = sum(len(result.unchanged) for result in preview.results)
    kept = sum(len(result.kept) for result in preview.results)
    notes = [
        f"Would write {plural(written, 'file')}, {unchanged} unchanged, "
        f"{kept} left as edited."
    ]
    return Review(lines, notes=notes)


def _report_render(
    console: Console, results: list[DocumentRender], dry_run: bool
) -> None:
    verb = "would write" if dry_run else "wrote"
    for result in results:
        label = f"{result.database.parent.name}/{result.name}"
        if result.skipped is not None:
            _say(console, f"Error: {label}: {result.skipped}")
        for path in result.written:
            _say(console, f"{label}: {verb} {path.name}")
        for path in result.kept:
            _say(console, f"Warning: {label}: {path.name} was changed; left as it is")
    written = sum(len(result.written) for result in results)
    unchanged = sum(len(result.unchanged) for result in results)
    kept = sum(len(result.kept) for result in results)
    _say(
        console,
        f"Render of {plural(len(results), 'document')}: {verb} "
        f"{plural(written, 'file')}, {unchanged} unchanged, {kept} left as edited.",
    )


def _execute_render(spec: RenderSpec, console: Console) -> int:
    path = spec.input.expanduser().resolve()
    refusal = _render_refusal(path)
    if refusal is not None:
        _say(console, f"Error: {refusal}")
        return ExitCode.USAGE
    databases = find_databases(path)
    for database in databases:
        # A database the store refuses stops the render before any file.
        read_store(database.parent)
    results: list[DocumentRender] = []
    for database in databases:
        if spec.dry_run:
            results.extend(
                preview_database(database, formats=spec.formats, force=spec.force)
            )
        else:
            results.extend(
                asyncio.run(
                    render_database(database, formats=spec.formats, force=spec.force)
                )
            )
    _report_render(console, results, spec.dry_run)
    failed = any(result.failed for result in results)
    return ExitCode.FAILURE if failed else ExitCode.OK


def _run_render(request: RunRequest) -> int:
    try:
        spec = resolve_render(dict(request.answers))
    except SpecError as exc:
        _say(request.console, f"Error: {exc}")
        return ExitCode.USAGE
    return _guarded(request.console, lambda: _execute_render(spec, request.console))


def render_flow() -> TaskFlow:
    """Return the render task: input, formats, force and dry run."""

    def steps(session: Session) -> list[Step]:
        return [
            _input_step(f"Input (folder or {DATABASE_NAME}):", _path_text),
            _formats_step(),
            _flag_step(
                "force",
                "Force",
                "Files changed outside the tool:",
                Choice(False, "Keep them"),
                Choice(True, "Replace them"),
            ),
            _dry_run_step("render the files"),
        ]

    return TaskFlow(
        name="render",
        title=TASKS["render"],
        program=RENDER_PROGRAM,
        table=RENDER_OPTIONS,
        steps=steps,
        review=_render_review,
        command=_command(RENDER_PROGRAM, RENDER_OPTIONS),
        run=_run_render,
    )


# =============================================================================
# drop
# =============================================================================


def _store_path(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    try:
        find_database(absolute(text))
    except DropInputError as exc:
        return str(exc)
    return None


def _stored_names(database: Path) -> list[str]:
    """Return the names of the stored documents in name order, or none."""
    try:
        selection = select_stored(database, select_all=True)
    except (DropInputError, StoreError, sqlite3.Error):
        return []
    return [document.name for document in selection.documents]


def _selection_validator(
    database: Path, by_number: bool
) -> Callable[[str], str | None]:
    def validate(text: str) -> str | None:
        expr = text.strip()
        if not expr:
            return "enter a selection"
        if by_number != is_numeric_selection(expr):
            if by_number:
                return "enter numbers or ranges, such as 1,3-5"
            return "a name search needs letters; choose 'By number or range'"
        try:
            select_stored(database, select=expr)
        except (DropInputError, StoreError, sqlite3.Error) as exc:
            return str(exc)
        return None

    return validate


def _list_documents(ctx: StepContext, names: list[str]) -> None:
    for number, name in enumerate(names[:LIST_LIMIT], start=1):
        ctx.prompter.show(f"  {number:>4}  {name}")
    if len(names) > LIST_LIMIT:
        ctx.prompter.show(f"        ... and {len(names) - LIST_LIMIT} more")


def _documents_step() -> Step:
    def ask(ctx: StepContext) -> Nav:
        state = ctx.state
        chosen: dict[str, str] = {}

        def database() -> Path:
            return find_database(absolute(state.value("input")))

        def how() -> Nav:
            names = _stored_names(database())
            _list_documents(ctx, names)
            # A drop is destructive, so "all" is preselected only once chosen.
            select = state.value("select")
            current = "all" if state.value("all") else "numbers"
            if not state.value("all") and select and not is_numeric_selection(select):
                current = "name"
            answer = ctx.prompter.select(
                "Which documents?",
                [
                    Choice("all", f"All {plural(len(names), 'document')}"),
                    Choice("numbers", "By number or range", "such as 1,3-5"),
                    Choice("name", "By name search"),
                ],
                default=current,
            )
            if answer is BACK:
                return BACK
            chosen["how"] = answer
            if answer == "all":
                state.set("all", True)
            return DONE

        def expression() -> Nav | None:
            how = chosen.get("how")
            if how not in ("numbers", "name"):
                return None
            by_number = how == "numbers"
            current = state.value("select") or ""
            if current and is_numeric_selection(current) != by_number:
                current = ""
            answer = ctx.prompter.text(
                "Document numbers or ranges:" if by_number else "Name contains:",
                default=current,
                validate=_selection_validator(database(), by_number),
            )
            if answer is BACK:
                return BACK
            _pin_selection(ctx, database(), answer.strip())
            return DONE

        return StepContext.sequence([how, expression])

    return Step("documents", "Documents", ask=ask, sets=("select", "files", "all"))


def _pin_selection(ctx: StepContext, database: Path, expr: str) -> None:
    """Answer with the names of the documents *expr* picks, as ``--files``.

    The drop then deletes the documents the review showed, even when a run
    adds or removes documents in between and the numbers shift. A name with
    a comma cannot be spelled in ``--files``; then *expr* itself is kept.
    """
    try:
        names = [
            document.name for document in select_stored(database, select=expr).documents
        ]
    except (DropInputError, StoreError, sqlite3.Error):
        names = []
    for name in names:
        ctx.prompter.show(f"  {name}")
    if names and not any("," in name for name in names):
        ctx.state.set("files", tuple(names))
    else:
        ctx.state.set("select", expr)


@dataclass(frozen=True)
class _DropPreview:
    database: Path | None = None
    selection: DocumentSelection | None = None
    report: DropReport | None = None
    refused: str | None = None


def _preview_drop(spec: DropSpec) -> _DropPreview:
    try:
        database = find_database(spec.input)
        selection = select_stored(
            database, select=spec.select, files=spec.files, select_all=spec.all
        )
        report = asyncio.run(
            drop_documents(database, selection.documents, dry_run=True)
        )
    except (DropInputError, StoreError, sqlite3.Error, OSError) as exc:
        return _DropPreview(refused=str(exc))
    return _DropPreview(database, selection, report)


def _drop_notes(report: DropReport) -> list[str]:
    notes = [
        f"Would drop {plural(len(report.dropped), 'document')}, refuse "
        f"{len(report.refused)}; delete {plural(len(report.deleted_files), 'file')}, "
        f"keep {len(report.kept_files)}."
    ]
    notes += [
        f"Refused: {item.name or item.doc_id}: {item.reason}" for item in report.refused
    ]
    notes += [f"Kept: {item.path.name}: {item.reason}" for item in report.kept_files]
    return notes


def _drop_review(state: State, session: Session) -> Review:
    run = Line("run", ["dry run" if state.value("dry_run") else "drop"])
    try:
        spec = resolve_drop(state.answers)
    except SpecError as exc:
        current = state.value("input")
        shown = "none" if current is None else str(absolute(current))
        return Review([Line("input", [shown]), run], withheld=str(exc))
    path = absolute(spec.input)
    chosen = "all" if spec.all else spec.select or ", ".join(spec.files)
    preview = _preview_drop(spec)
    if preview.selection is None or preview.report is None:
        lines = [Line("input", [str(path)]), Line("documents", [chosen]), run]
        return Review(lines, withheld=preview.refused)
    picked = plural(len(preview.selection.documents), "document")
    lines = [Line("input", [str(path)]), Line("documents", [chosen, picked]), run]
    notes = [*preview.selection.warnings, *_drop_notes(preview.report)]
    return Review(lines, notes=notes)


def _report_drop(console: Console, report: DropReport) -> None:
    dropped = "would drop" if report.dry_run else "dropped"
    deleted = "would delete" if report.dry_run else "deleted"
    for name in report.dropped:
        _say(console, f"{name}: {dropped}")
    for refusal in report.refused:
        _say(console, f"Error: {refusal.name or refusal.doc_id}: {refusal.reason}")
    for path in report.deleted_files:
        _say(console, f"{deleted} {path.name}")
    for item in report.kept_files:
        _say(console, f"Warning: kept {item.path.name}: {item.reason}")
    if report.dropped and report.kept_files:
        _say(console, f"Warning: {KEPT_FILE_HINT}")
    files = plural(len(report.deleted_files), "file")
    _say(
        console,
        f"Drop: {dropped} {plural(len(report.dropped), 'document')}, refused "
        f"{len(report.refused)}; {deleted} {files}, kept {len(report.kept_files)}.",
    )


def _execute_drop(spec: DropSpec, console: Console) -> int:
    try:
        database = find_database(spec.input)
        selection = select_stored(
            database, select=spec.select, files=spec.files, select_all=spec.all
        )
    except DropInputError as exc:
        _say(console, f"Error: {exc}")
        return ExitCode.USAGE
    for warning in selection.warnings:
        _say(console, f"Warning: {warning}")
    report = asyncio.run(
        drop_documents(database, selection.documents, dry_run=spec.dry_run)
    )
    _report_drop(console, report)
    return drop_exit_code(report)


def _run_drop(request: RunRequest) -> int:
    try:
        spec = resolve_drop(dict(request.answers))
    except SpecError as exc:
        _say(request.console, f"Error: {exc}")
        return ExitCode.USAGE
    return _guarded(request.console, lambda: _execute_drop(spec, request.console))


def drop_flow() -> TaskFlow:
    """Return the drop task: input, documents and dry run."""

    def steps(session: Session) -> list[Step]:
        return [
            _input_step(f"Input (output folder or {DATABASE_NAME}):", _store_path),
            _documents_step(),
            _dry_run_step("drop the documents and delete their files"),
        ]

    return TaskFlow(
        name="drop",
        title=TASKS["drop"],
        program=DROP_PROGRAM,
        table=DROP_OPTIONS,
        steps=steps,
        review=_drop_review,
        command=_command(DROP_PROGRAM, DROP_OPTIONS),
        run=_run_drop,
    )
