"""The render and migrate tasks of the guided UI.

:func:`render_flow` renders the files of the stored documents at an input
path again; :func:`migrate_flow` imports the outputs of version 1.0 runs.
Their runs report like ``autoexcerpter render`` and ``autoexcerpter
migrate`` and return the same exit codes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from rich.console import Console

from autoexcerpter.common.contract import ExitCode
from autoexcerpter.common.options import OptionTable
from autoexcerpter.common.store import StoreError, VersionlessDatabaseError
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
    command_display,
)
from autoexcerpter.migrate import MigrationReport, migrate
from autoexcerpter.output.render import (
    DocumentRender,
    find_databases,
    preview_database,
    render_database,
)
from autoexcerpter.output.store import DATABASE_NAME, read_store
from autoexcerpter.spec import (
    MIGRATE_OPTIONS,
    RENDER_FORMATS,
    RENDER_OPTIONS,
    MigrateSpec,
    RenderSpec,
    SpecError,
    resolve_migrate,
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

__all__ = ["MIGRATE_PROGRAM", "RENDER_PROGRAM", "migrate_flow", "render_flow"]

logger = logging.getLogger(__name__)

RENDER_PROGRAM: Final = ("autoexcerpter", "render")
MIGRATE_PROGRAM: Final = ("autoexcerpter", "migrate")
PREVIEW_FILES: Final = 200
"""Migrate previews its dry run only for inputs of at most this many files."""

_MIGRATE_PREVIEW = "migrate_preview"
_KEEP = "keep"
_CHOOSE = "choose"


def _path_text(text: str) -> str | None:
    return "enter a path" if not text.strip() else None


def _existing_path(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    if not absolute(text).exists():
        return "no such file or folder"
    return None


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


def _migrate_command(root: Path) -> list[str]:
    return [*MIGRATE_PROGRAM, "--input", str(root)]


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
    versionless: Path | None = None
    refused: str | None = None


def _preview_render(path: Path, spec: RenderSpec) -> _RenderPreview:
    databases = find_databases(path)
    results: list[DocumentRender] = []
    for database in databases:
        try:
            results.extend(
                preview_database(database, formats=spec.formats, force=spec.force)
            )
        except VersionlessDatabaseError:
            return _RenderPreview(databases, results, versionless=database)
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
    if preview.versionless is not None:
        command = command_display(_migrate_command(preview.versionless.parent))
        return Review(
            lines,
            notes=["Convert it first with the migrate task:", *command],
            withheld=f"{preview.versionless} was written before the store",
        )
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
        try:
            read_store(database.parent)
        except VersionlessDatabaseError:
            command = " ".join(_migrate_command(database.parent))
            _say(
                console,
                f"Error: {database} was written before the store. Convert it "
                f"first: {command}",
            )
            return ExitCode.USAGE
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
# migrate
# =============================================================================


def _is_small(path: Path) -> bool:
    if path.is_file():
        return True
    count = 0
    for _root, _dirs, files in os.walk(path):
        count += len(files)
        if count > PREVIEW_FILES:
            return False
    return True


@dataclass(frozen=True)
class _MigratePreview:
    path: Path
    report: MigrationReport | None = None
    refused: str | None = None


def _preview_migrate(state: State, session: Session, path: Path) -> _MigratePreview:
    """Return the dry run of a small input, cached by path."""
    stored = state.data.get(_MIGRATE_PREVIEW)
    if isinstance(stored, _MigratePreview) and stored.path == path:
        return stored
    preview = _MigratePreview(path)
    if _is_small(path):
        state_dir = session.settings.state_directory()
        try:
            report = asyncio.run(migrate(path, dry_run=True, state_dir=state_dir))
            preview = _MigratePreview(path, report)
        except StoreError as exc:
            preview = _MigratePreview(path, refused=str(exc))
        except (OSError, sqlite3.Error, ValueError) as exc:
            logger.debug("No migrate preview of %s: %s", path, exc)
    state.data[_MIGRATE_PREVIEW] = preview
    return preview


def _found(report: MigrationReport) -> str:
    converted = sum(db.action == "convert" for db in report.databases)
    text = plural(len(report.databases), "database")
    if converted:
        text += f" ({converted} to convert)"
    return (
        f"{text}, {plural(len(report.documents), 'document')}, "
        f"{plural(report.total('imported'), 'unit')} to import"
    )


def _migrate_review(state: State, session: Session) -> Review:
    run = Line("run", ["dry run" if state.value("dry_run") else "import"])
    current = state.value("input")
    if current is None:
        return Review([Line("input", ["none"]), run], withheld="no input path")
    path = absolute(current)
    if not path.exists():
        return Review(
            [Line("input", [str(path)]), run],
            withheld=f"Input path does not exist: {path}",
        )
    preview = _preview_migrate(state, session, path)
    parts = [str(path)]
    if preview.report is not None:
        parts.append(_found(preview.report))
    lines = [Line("input", parts), run]
    if preview.refused is not None:
        return Review(lines, withheld=preview.refused)
    return Review(lines)


def _report_migration(console: Console, report: MigrationReport) -> None:
    for document in report.documents:
        label = f"{document.root.name}/{document.name}"
        for warning in document.warnings:
            _say(console, f"Warning: {label}: {warning}")
        for mismatch in document.mismatches:
            _say(console, f"Error: {label}: {mismatch}")
    for error in report.errors:
        _say(console, f"Error: {error}")
    for (before, after), count in report.matrix().items():
        _say(console, f"Resume state {before} -> {after}: {plural(count, 'document')}")
    verb = "would import" if report.dry_run else "imported"
    _say(
        console,
        f"Migrate of {plural(len(report.documents), 'document')}: {verb} "
        f"{plural(report.total('imported'), 'unit')}, "
        f"{plural(report.total('citations'), 'citation')}, "
        f"{plural(report.total('outputs'), 'file')}; "
        f"{plural(report.mismatches, 'mismatch', 'mismatches')}.",
    )


def _execute_migrate(spec: MigrateSpec, session: Session, console: Console) -> int:
    path = spec.input.expanduser().resolve()
    if not path.exists():
        _say(console, f"Error: Input path does not exist: {path}")
        return ExitCode.USAGE
    state_dir = session.settings.state_directory()
    report = asyncio.run(migrate(path, dry_run=spec.dry_run, state_dir=state_dir))
    _report_migration(console, report)
    return report.exit_code


def _run_migrate(request: RunRequest) -> int:
    try:
        spec = resolve_migrate(dict(request.answers))
    except SpecError as exc:
        _say(request.console, f"Error: {exc}")
        return ExitCode.USAGE
    return _guarded(
        request.console,
        lambda: _execute_migrate(spec, request.session, request.console),
    )


def migrate_flow() -> TaskFlow:
    """Return the migrate task: input and dry run."""

    def steps(session: Session) -> list[Step]:
        return [
            _input_step(
                "Input (output folder, working folder or database):", _existing_path
            ),
            _dry_run_step("import into the store"),
        ]

    return TaskFlow(
        name="migrate",
        title=TASKS["migrate"],
        program=MIGRATE_PROGRAM,
        table=MIGRATE_OPTIONS,
        steps=steps,
        review=_migrate_review,
        command=_command(MIGRATE_PROGRAM, MIGRATE_OPTIONS),
        run=_run_migrate,
    )
