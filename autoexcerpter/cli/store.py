"""The render subcommand: write the files of stored documents again.

``render`` exits 0 when every file holds its rendering or was left alone
because it was edited, 1 when a file could not be replaced or a document is
held by another run, 2 for a missing input or a database it refuses; a
database written before the store names the migrate command. With
``--json`` it writes one JSON summary line on every exit; ``--dry-run``
reports what it would write and writes no file.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from typing import Any

from autoexcerpter.cli.parser import UsageError
from autoexcerpter.common.contract import ExitCode, SummaryEmitter
from autoexcerpter.common.log_setup import configure_logging
from autoexcerpter.common.store import StoreError, VersionlessDatabaseError
from autoexcerpter.output.render import (
    DocumentRender,
    find_databases,
    preview_database,
    render_database,
)
from autoexcerpter.output.store import DATABASE_NAME, read_store
from autoexcerpter.spec import RENDER_OPTIONS, RenderSpec, SpecError, resolve_render

__all__ = ["execute_render", "render_fields", "report_usage_error"]

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


def report_usage_error(error: UsageError, argv: Sequence[str]) -> int:
    """Write a render parse error like argparse does, plus the JSON line if asked."""
    sys.stderr.write(error.usage)
    sys.stderr.write(f"{error.prog}: error: {error}\n")
    stream = sys.stdout if "--json" in argv else None
    SummaryEmitter(stream, dry_run="--dry-run" in argv).emit(render_fields())
    return ExitCode.USAGE


def execute_render(args: argparse.Namespace) -> int:
    """Run the parsed render options and return the exit code.

    A refused database returns 2, Ctrl+C 130 and an unexpected exception 1;
    with ``--json`` the summary line is written on each of these exits.
    """
    configure_logging()
    flags = RENDER_OPTIONS.values(args)
    emitter = SummaryEmitter(
        sys.stdout if flags.get("json") else None,
        dry_run=bool(flags.get("dry_run")),
    )
    try:
        spec = resolve_render(flags)
        return int(_execute(spec, emitter))
    except SpecError as exc:
        _write(f"[ERROR] Invalid arguments: {exc}")
        emitter.emit(render_fields())
        return ExitCode.USAGE
    except StoreError as exc:
        _write(f"[ERROR] {exc}")
        emitter.emit(render_fields())
        return exc.exit_code
    except KeyboardInterrupt:
        _write("[ERROR] Interrupted by the user.")
        emitter.emit(render_fields())
        return ExitCode.INTERRUPTED
    except Exception as exc:
        logger.exception("Render failed: %s", exc)
        emitter.emit(render_fields())
        return ExitCode.FAILURE
