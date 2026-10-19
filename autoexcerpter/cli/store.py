"""The store subcommands: render and drop documents.

``render`` exits 0 when every file holds its rendering or was left alone
because it was edited, 1 when a file could not be replaced or a document is
held by another run, 2 for a missing input, an invalid settings file or a
database it refuses. Both read the settings (``--settings FILE``) for the
log file. ``drop`` exits 0 when every selected
document was dropped (or would be, in a dry run), 1 when one was refused, 2
for a missing store, a selection that is malformed or matches nothing, or a
database it refuses. With ``--json`` each writes one JSON summary line on
every exit; ``--dry-run`` reports what it would write and writes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from autoexcerpter import ext
from autoexcerpter import settings as settings_module
from autoexcerpter.cli.parser import UsageError
from autoexcerpter.common.contract import ExitCode, SummaryEmitter
from autoexcerpter.common.log_setup import configure_logging
from autoexcerpter.common.store import DropReport, StoreError
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
    RENDER_OPTIONS,
    DropSpec,
    RenderSpec,
    SpecError,
    resolve_drop,
    resolve_render,
)

__all__ = [
    "drop_fields",
    "execute_drop",
    "execute_render",
    "render_fields",
    "report_usage_error",
]

logger = logging.getLogger(__name__)


def _write(line: str) -> None:
    print(line, file=sys.stderr)


def render_fields(
    *,
    databases: int = 0,
    documents: int = 0,
    written: int = 0,
    unchanged: int = 0,
    kept: int = 0,
    stale: int = 0,
    skipped: int = 0,
    outputs: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the fields of the render summary.

    ``kept`` counts files changed outside the tool and left alone, ``stale``
    files that could not be replaced and ``skipped`` documents another run
    holds; ``outputs`` lists the files written.
    """
    return {
        "command": "render",
        "databases": databases,
        "documents": documents,
        "written": written,
        "unchanged": unchanged,
        "kept": kept,
        "stale": stale,
        "skipped": skipped,
        "outputs": list(outputs),
    }


def _report(results: list[DocumentRender], dry_run: bool) -> None:
    prefix = "DRY-RUN: " if dry_run else ""
    verb = "would write" if dry_run else "wrote"
    for result in results:
        label = f"{result.database.parent.name}/{result.name}"
        if result.skipped is not None:
            _write(f"[ERROR] {label}: {result.skipped}")
        for path in result.written:
            _write(f"{prefix}{label}: {verb} {path}")
        for path in result.kept:
            _write(
                f"[WARNING] {label}: {path} was changed outside the tool and is "
                "left as it is; render with --force to replace it."
            )
        for warning in result.warnings:
            if "changed outside the tool" not in warning:
                _write(f"[WARNING] {label}: {warning}")
    written = sum(len(result.written) for result in results)
    unchanged = sum(len(result.unchanged) for result in results)
    kept = sum(len(result.kept) for result in results)
    _write(
        f"{prefix}Render of {len(results)} document(s): {verb} {written} "
        f"file(s), {unchanged} unchanged, {kept} left as edited."
    )


def _load_settings(path: Path | None) -> settings_module.Settings | str:
    """Load the settings and start the log file; return the error message."""
    try:
        loaded = settings_module.load(path, blocks=ext.settings_blocks())
    except settings_module.SettingsError as exc:
        return str(exc)
    configure_logging(log_file=loaded.log_file())
    return loaded


def _execute(spec: RenderSpec, emitter: SummaryEmitter) -> int:
    loaded = _load_settings(spec.settings)
    if isinstance(loaded, str):
        _write(f"[ERROR] {loaded}")
        emitter.emit(render_fields())
        return ExitCode.USAGE
    path = spec.input.expanduser().resolve()
    if not path.exists():
        _write(f"[ERROR] Input path does not exist: {path}")
        emitter.emit(render_fields())
        return ExitCode.USAGE
    if path.is_file() and path.name != DATABASE_NAME:
        _write(f"[ERROR] {path} is not a {DATABASE_NAME} file")
        emitter.emit(render_fields())
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
    _report(results, spec.dry_run)
    emitter.emit(
        render_fields(
            databases=len(databases),
            documents=len(results),
            written=sum(len(result.written) for result in results),
            unchanged=sum(len(result.unchanged) for result in results),
            kept=sum(len(result.kept) for result in results),
            stale=sum(len(result.stale) for result in results),
            skipped=sum(result.skipped is not None for result in results),
            outputs=[str(path) for result in results for path in result.written],
        )
    )
    failed = any(result.failed for result in results)
    return ExitCode.FAILURE if failed else ExitCode.OK


def drop_fields(
    *,
    database: Path | None = None,
    selected: int = 0,
    warnings: Sequence[str] = (),
    report: DropReport | None = None,
) -> dict[str, Any]:
    """Return the fields of the drop summary.

    ``selected`` counts the documents the selection chose; ``dropped``,
    ``refused`` (``doc_id``, ``name``, ``reason``), ``deleted_files`` and
    ``kept_files`` (``path``, ``reason``) come from the store's report.
    """
    fields: dict[str, Any] = {
        "command": "drop",
        "database": None if database is None else str(database),
        "selected": selected,
        "dropped": [],
        "refused": [],
        "deleted_files": [],
        "kept_files": [],
        "warnings": list(warnings),
    }
    if report is not None:
        fields.update(
            (key, value) for key, value in report.as_dict().items() if key != "dry_run"
        )
    return fields


def _report_drop(report: DropReport, warnings: Sequence[str]) -> None:
    dry_run = report.dry_run
    prefix = "DRY-RUN: " if dry_run else ""
    dropped = "would drop" if dry_run else "dropped"
    deleted = "would delete" if dry_run else "deleted"
    kept = "would keep" if dry_run else "kept"
    for warning in warnings:
        _write(f"[WARNING] {warning}")
    for name in report.dropped:
        _write(f"{prefix}{dropped} {name}")
    for refusal in report.refused:
        label = refusal.name if refusal.name is not None else f"#{refusal.doc_id}"
        _write(f"[ERROR] refused {label}: {refusal.reason}")
    for path in report.deleted_files:
        _write(f"{prefix}{deleted} {path}")
    for item in report.kept_files:
        _write(f"[WARNING] {kept} {item.path}: {item.reason}")
    if report.dropped and report.kept_files:
        _write(f"[WARNING] {KEPT_FILE_HINT}")
    total = len(report.dropped) + len(report.refused)
    _write(
        f"{prefix}Drop of {total} document(s): {dropped} {len(report.dropped)}, "
        f"refused {len(report.refused)}; {deleted} {len(report.deleted_files)} "
        f"file(s), {kept} {len(report.kept_files)}."
    )


def _execute_drop(spec: DropSpec, emitter: SummaryEmitter) -> int:
    loaded = _load_settings(spec.settings)
    if isinstance(loaded, str):
        _write(f"[ERROR] {loaded}")
        emitter.emit(drop_fields())
        return ExitCode.USAGE
    try:
        database = find_database(spec.input)
    except DropInputError as exc:
        _write(f"[ERROR] {exc}")
        emitter.emit(drop_fields())
        return ExitCode.USAGE
    try:
        selection = select_stored(
            database, select=spec.select, files=spec.files, select_all=spec.all
        )
    except DropInputError as exc:
        _write(f"[ERROR] {exc}")
        emitter.emit(drop_fields(database=database))
        return ExitCode.USAGE
    report = asyncio.run(
        drop_documents(database, selection.documents, dry_run=spec.dry_run)
    )
    _report_drop(report, selection.warnings)
    emitter.emit(
        drop_fields(
            database=database,
            selected=len(selection.documents),
            warnings=selection.warnings,
            report=report,
        )
    )
    return drop_exit_code(report)


_USAGE_FIELDS: dict[str, Callable[[], dict[str, Any]]] = {
    "render": render_fields,
    "drop": drop_fields,
}


def report_usage_error(error: UsageError, argv: Sequence[str]) -> int:
    """Write a parse error like argparse does, plus the JSON line if asked."""
    sys.stderr.write(error.usage)
    sys.stderr.write(f"{error.prog}: error: {error}\n")
    stream = sys.stdout if "--json" in argv else None
    fields = _USAGE_FIELDS.get(argv[0] if argv else "", render_fields)()
    SummaryEmitter(stream, dry_run="--dry-run" in argv).emit(fields)
    return ExitCode.USAGE


def _guarded(
    name: str,
    emitter: SummaryEmitter,
    empty: dict[str, Any],
    body: Callable[[], int],
) -> int:
    """Run *body*; map a refused database, Ctrl+C and a crash to exit codes.

    A refused database returns 2, Ctrl+C 130 and an unexpected exception 1;
    with ``--json`` the summary line is written on each of these exits.
    """
    try:
        return int(body())
    except SpecError as exc:
        _write(f"[ERROR] Invalid arguments: {exc}")
        emitter.emit(empty)
        return ExitCode.USAGE
    except StoreError as exc:
        _write(f"[ERROR] {exc}")
        emitter.emit(empty)
        return exc.exit_code
    except KeyboardInterrupt:
        _write("[ERROR] Interrupted by the user.")
        emitter.emit(empty)
        return ExitCode.INTERRUPTED
    except Exception as exc:
        logger.exception("%s failed: %s", name, exc)
        emitter.emit(empty)
        return ExitCode.FAILURE


def execute_render(args: argparse.Namespace) -> int:
    """Run the parsed render options and return the exit code."""
    configure_logging()
    flags = RENDER_OPTIONS.values(args)
    emitter = SummaryEmitter(
        sys.stdout if flags.get("json") else None,
        dry_run=bool(flags.get("dry_run")),
    )
    return _guarded(
        "Render",
        emitter,
        render_fields(),
        lambda: _execute(resolve_render(flags), emitter),
    )


def execute_drop(args: argparse.Namespace) -> int:
    """Run the parsed drop options and return the exit code."""
    configure_logging()
    flags = DROP_OPTIONS.values(args)
    emitter = SummaryEmitter(
        sys.stdout if flags.get("json") else None,
        dry_run=bool(flags.get("dry_run")),
    )
    return _guarded(
        "Drop",
        emitter,
        drop_fields(),
        lambda: _execute_drop(resolve_drop(flags), emitter),
    )
