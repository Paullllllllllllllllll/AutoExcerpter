"""The store subcommands: render stored documents and migrate version 1.0.

``render`` exits 0 when every file holds its rendering or was left alone
because it was edited, 1 when a file could not be replaced or a document is
held by another run, 2 for a missing input or a database it refuses; a
database written before the store names the migrate command. ``migrate``
exits 0 when every import verified, 1 on a verification mismatch or an
import that failed, 2 for a missing input or a database it refuses (newer,
unreadable or of an unknown layout). With ``--json`` both write one JSON
summary line on every exit; ``--dry-run`` reports what they would write and
writes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any

from autoexcerpter import ext
from autoexcerpter import settings as settings_module
from autoexcerpter.cli.parser import UsageError
from autoexcerpter.common.contract import ExitCode, SummaryEmitter
from autoexcerpter.common.log_setup import configure_logging
from autoexcerpter.common.store import StoreError, VersionlessDatabaseError
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
    RENDER_OPTIONS,
    MigrateSpec,
    RenderSpec,
    SpecError,
    resolve_migrate,
    resolve_render,
)

__all__ = [
    "execute_migrate",
    "execute_render",
    "migrate_fields",
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


def _execute(spec: RenderSpec, emitter: SummaryEmitter) -> int:
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
        try:
            read_store(database.parent)
        except VersionlessDatabaseError:
            command = f'autoexcerpter migrate --input "{database.parent}"'
            _write(
                f"[ERROR] {database} was written before the store. Convert it "
                f"first: {command}"
            )
            emitter.emit(render_fields())
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


def migrate_fields(  # noqa: PLR0913 - one keyword per summary field
    *,
    databases: int = 0,
    converted: int = 0,
    documents: int = 0,
    units: int = 0,
    citations: int = 0,
    outputs: int = 0,
    mismatches: int = 0,
    warnings: int = 0,
    snapshots: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the fields of the migrate summary.

    ``units``, ``citations`` and ``outputs`` count what was (or would be)
    imported; ``mismatches`` the verification differences that fail it.
    """
    return {
        "command": "migrate",
        "databases": databases,
        "converted": converted,
        "documents": documents,
        "units": units,
        "citations": citations,
        "outputs": outputs,
        "mismatches": mismatches,
        "warnings": warnings,
        "snapshots": list(snapshots),
    }


def _report_migration(report: MigrationReport) -> None:
    prefix = "DRY-RUN: " if report.dry_run else ""
    for database in report.databases:
        line = f"{prefix}{database.path}: {database.action}"
        if database.snapshot is not None:
            line += f" (snapshot {database.snapshot.name})"
        _write(line)
    for document in report.documents:
        label = f"{document.root.name}/{document.name}"
        if document.changed:
            verb = "would import" if report.dry_run else "imported"
            _write(
                f"{prefix}{label}: {verb} {document.imported} unit(s), "
                f"{document.citations} citation(s), {document.outputs} file(s)"
            )
        if document.kept:
            _write(f"{label}: {document.kept} unit(s) of later runs kept")
        for warning in document.warnings:
            _write(f"[WARNING] {label}: {warning}")
        for mismatch in document.mismatches:
            _write(f"[ERROR] {label}: {mismatch}")
    for warning in report.warnings:
        _write(f"[WARNING] {warning}")
    for error in report.errors:
        _write(f"[ERROR] {error}")
    for (before, after), count in report.matrix().items():
        _write(f"Resume state {before} -> {after}: {count} document(s)")
    if report.exit_code:
        _write(
            f"[ERROR] Migration finished with {report.mismatches} mismatch(es) and "
            f"{len(report.errors)} error(s)."
        )
    else:
        verb = "checked" if report.dry_run else "migrated; every import verified"
        _write(f"{prefix}{len(report.documents)} document(s) {verb}.")


def _migration_fields(report: MigrationReport) -> dict[str, Any]:
    return migrate_fields(
        databases=len(report.databases),
        converted=sum(db.action in ("convert", "converted") for db in report.databases),
        documents=len(report.documents),
        units=report.total("imported"),
        citations=report.total("citations"),
        outputs=report.total("outputs"),
        mismatches=report.mismatches,
        warnings=sum(len(document.warnings) for document in report.documents)
        + len(report.warnings),
        snapshots=[str(path) for path in report.snapshots],
    )


def _execute_migrate(spec: MigrateSpec, emitter: SummaryEmitter) -> int:
    path = spec.input.expanduser().resolve()
    if not path.exists():
        _write(f"[ERROR] Input path does not exist: {path}")
        emitter.emit(migrate_fields())
        return ExitCode.USAGE
    try:
        loaded = settings_module.load(spec.settings, blocks=ext.settings_blocks())
    except settings_module.SettingsError as exc:
        _write(f"[ERROR] {exc}")
        emitter.emit(migrate_fields())
        return ExitCode.USAGE
    report = asyncio.run(
        migrate(path, dry_run=spec.dry_run, state_dir=loaded.state_directory())
    )
    _report_migration(report)
    emitter.emit(_migration_fields(report))
    return report.exit_code


def report_usage_error(error: UsageError, argv: Sequence[str]) -> int:
    """Write a parse error like argparse does, plus the JSON line if asked."""
    sys.stderr.write(error.usage)
    sys.stderr.write(f"{error.prog}: error: {error}\n")
    stream = sys.stdout if "--json" in argv else None
    fields = migrate_fields() if argv[:1] == ["migrate"] else render_fields()
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


def execute_migrate(args: argparse.Namespace) -> int:
    """Run the parsed migrate options and return the exit code."""
    configure_logging()
    flags = MIGRATE_OPTIONS.values(args)
    emitter = SummaryEmitter(
        sys.stdout if flags.get("json") else None,
        dry_run=bool(flags.get("dry_run")),
    )
    return _guarded(
        "Migrate",
        emitter,
        migrate_fields(),
        lambda: _execute_migrate(resolve_migrate(flags), emitter),
    )
