"""The batch subcommands: check and cancel.

``batch check`` exits 0 when nothing failed, 1 when a submission or a
document failed or a submission waits unresolved for ``--adopt`` or
``--abandon``, and 2 when a database below the folder is refused
(versionless, newer or unreadable), the folder is missing, or ``--adopt``
or ``--abandon`` names nothing to act on. ``batch cancel`` exits 1 when a
cancellation or a provider listing failed, and 2 for a refused database or
a missing folder. With ``--json`` both write one JSON summary line on every
exit; the check's line adds ``adopted`` or ``abandoned`` when those options
are given.

``batch check --wait`` repeats the check until no batch work is pending or
``--wait-hours`` have passed. Its counts are read from the stores at the
end; it exits 1 as well when the deadline passed with work pending or a
document needs a run, and its line adds ``polls``, ``waited_seconds``,
``stopped``, ``needs_run`` and ``rerun``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any

from autoexcerpter import ext
from autoexcerpter import settings as settings_module
from autoexcerpter.batch.cancel import (
    CancelReport,
    cancel_batches,
    cancel_submissions,
    cancel_unfinished,
)
from autoexcerpter.batch.check import (
    ABANDONED,
    CheckResult,
    CheckTargetError,
    WaitProgress,
    WaitTotals,
    check_batches,
    check_until_settled,
    wait_totals,
)
from autoexcerpter.batch.finalize import FAILED
from autoexcerpter.batch.report import (
    cancel_lines,
    check_lines,
    complete_documents,
    pending_lines,
    retry_line,
    submitted_batches,
    wait_end_lines,
    wait_line,
)
from autoexcerpter.cli.heartbeat import BatchHeartbeat
from autoexcerpter.cli.parser import UsageError
from autoexcerpter.common.contract import (
    USAGE_KEY,
    ExitCode,
    SummaryEmitter,
    exit_code_for,
    usage_field,
)
from autoexcerpter.common.log_setup import configure_logging
from autoexcerpter.common.polling import Stop, WaitLimits, WaitTick
from autoexcerpter.common.usage import Usage
from autoexcerpter.settings import Settings
from autoexcerpter.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    BatchCheckSpec,
    CommandSpec,
    SpecError,
    resolve_batch_cancel,
    resolve_batch_check,
)

__all__ = [
    "cancel_fields",
    "check_fields",
    "execute_cancel",
    "execute_check",
    "report_usage_error",
]

logger = logging.getLogger(__name__)


def _write(line: str) -> None:
    print(line, file=sys.stderr)


def check_fields(  # noqa: PLR0913 - one keyword per summary field
    *,
    finalized: int = 0,
    pending: int = 0,
    failed: int = 0,
    unresolved: int = 0,
    submitted: int = 0,
    complete: int = 0,
    outputs: Sequence[str] = (),
    usage: Mapping[str, Usage] | None = None,
) -> dict[str, Any]:
    """Return the fields of the batch check summary.

    The first four count submissions by outcome; *submitted* counts the
    summary submissions the check opened and *complete* the documents it
    rendered complete. *usage* holds the tokens of the finalized results per
    role.
    """
    return {
        "command": "check_batches",
        "finalized": finalized,
        "pending": pending,
        "failed": failed,
        "unresolved": unresolved,
        "submitted": submitted,
        "complete": complete,
        "outputs": list(outputs),
        USAGE_KEY: usage_field(usage),
    }


def cancel_fields(*, cancelled: int = 0, failed: int = 0) -> dict[str, Any]:
    """Return the fields of the batch cancel summary."""
    return {"command": "cancel_batches", "cancelled": cancelled, "failed": failed}


def _base(flags: Mapping[str, Any]) -> dict[str, Any]:
    """Return the check summary's fields with zero counts.

    ``adopted`` and ``abandoned`` appear when their option is given.
    """
    base = check_fields()
    for option, key in (("adopt", "adopted"), ("abandon", "abandoned")):
        if flags.get(option):
            base[key] = False
    if flags.get("wait"):
        base.update(polls=0, waited_seconds=0, stopped=None, needs_run=0, rerun=[])
    return base


def _result_fields(spec: BatchCheckSpec, result: CheckResult) -> dict[str, Any]:
    stats = result.stats
    fields = check_fields(
        finalized=stats.finalized,
        pending=stats.pending,
        failed=stats.failed,
        unresolved=stats.unresolved,
        submitted=submitted_batches(result),
        complete=complete_documents(result),
        outputs=[str(path) for path in result.outputs],
        usage=result.usage,
    )
    if spec.adopt is not None:
        fields["adopted"] = any(
            check.batch_id == spec.adopt and check.outcome != FAILED
            for check in result.submissions
        )
    if spec.abandon is not None:
        fields["abandoned"] = any(
            check.submission_id == spec.abandon and check.outcome == ABANDONED
            for check in result.submissions
        )
    return fields


def _prepare(
    spec: CommandSpec,
    settings_path: Path | None,
    args: argparse.Namespace,
    directory: Path | None,
) -> Settings | str:
    """Load the settings, check the folder and run the extension startup.

    Returns the settings, or the message of what refused the command.
    """
    try:
        settings = settings_module.load(settings_path, blocks=ext.settings_blocks())
    except settings_module.SettingsError as exc:
        return str(exc)
    if directory is not None and not directory.is_dir():
        return f"Input folder does not exist: {directory}"
    try:
        ext.startup(spec, settings=settings, args=args)
    except settings_module.SettingsError as exc:
        return str(exc)
    return settings


def _execute_check(
    args: argparse.Namespace, flags: Mapping[str, Any], emitter: SummaryEmitter
) -> int:
    spec = resolve_batch_check(flags)
    directory = spec.directory.expanduser().resolve()
    prepared = _prepare(spec, spec.settings, args, directory)
    if isinstance(prepared, str):
        _write(f"[ERROR] {prepared}")
        emitter.emit(_base(flags))
        return ExitCode.USAGE
    if spec.wait:
        return _execute_wait(args, flags, spec, prepared, emitter)
    _write(f"Checking the batch work below '{directory}'...")
    try:
        result = asyncio.run(
            check_batches(
                directory,
                prepared,
                batch_ids=spec.batch_ids,
                adopt=spec.adopt,
                abandon=spec.abandon,
                args=args,
            )
        )
    except CheckTargetError as exc:
        _write(f"[ERROR] Invalid arguments: {exc}")
        emitter.emit(_base(flags))
        return ExitCode.USAGE
    for line in check_lines(result):
        _write(line)
    emitter.emit(_result_fields(spec, result))
    if result.refused:
        return ExitCode.USAGE
    return ExitCode.FAILURE if result.had_failure else ExitCode.OK


def _wait_fields(
    spec: BatchCheckSpec, progress: WaitProgress, totals: WaitTotals
) -> dict[str, Any]:
    """Return the summary fields of a wait from its store-based totals."""
    stats = totals.stats
    fields = check_fields(
        finalized=stats.finalized,
        pending=stats.pending,
        failed=stats.failed,
        unresolved=stats.unresolved,
        submitted=totals.submitted,
        complete=totals.complete,
        outputs=[str(path) for path in totals.outputs],
        usage=totals.usage,
    )
    targets = _result_fields(spec, progress.ledger.first or CheckResult())
    for key in ("adopted", "abandoned"):
        if key in targets:
            fields[key] = targets[key]
    fields.update(
        polls=progress.passes,
        waited_seconds=progress.waited_seconds(),
        stopped=progress.stop,
        needs_run=len(totals.needs_run),
        rerun=[hint.command for hint in totals.needs_run],
    )
    return fields


def _finish_wait(
    spec: BatchCheckSpec, progress: WaitProgress, emitter: SummaryEmitter
) -> WaitTotals:
    """Read the totals from the stores, write the closing lines and the
    summary line, and return the totals."""
    totals = wait_totals(progress.ledger)
    if progress.stop == Stop.TIMEOUT and progress.last is not None:
        for line in pending_lines(progress.last):
            _write(line)
    lines = wait_end_lines(
        totals,
        passes=progress.passes,
        waited=progress.waited_seconds(),
        stop=progress.stop or "error",
    )
    for line in lines:
        _write(line)
    emitter.emit(_wait_fields(spec, progress, totals))
    return totals


def _execute_wait(
    args: argparse.Namespace,
    flags: Mapping[str, Any],
    spec: BatchCheckSpec,
    settings: Settings,
    emitter: SummaryEmitter,
) -> int:
    """Check until no batch work is pending; see :func:`check_until_settled`.

    On Ctrl+C and on a crash the totals of the checks so far are written
    before the exception goes on to :func:`_guarded`.
    """
    directory = spec.directory.expanduser().resolve()
    limits = WaitLimits(spec.poll_minutes * 60, spec.wait_hours * 3600)
    progress = WaitProgress()
    _write(
        f"Checking the batch work below '{directory}' every "
        f"{spec.poll_minutes:g} min, for up to {spec.wait_hours:g} h..."
    )

    beat = BatchHeartbeat.from_env()

    def on_pass(number: int, result: CheckResult) -> None:
        for line in check_lines(result, pending=number == 1, totals=False):
            _write(line)
        if beat is not None:
            beat.on_pass(result)

    def on_wait(tick: WaitTick) -> None:
        _write(wait_line(tick))
        if beat is not None:
            beat.on_wait(tick)

    try:
        asyncio.run(
            check_until_settled(
                directory,
                settings,
                limits,
                batch_ids=spec.batch_ids,
                adopt=spec.adopt,
                abandon=spec.abandon,
                args=args,
                progress=progress,
                on_pass=on_pass,
                on_wait=on_wait,
                on_retry=lambda number, exc: _write(retry_line(number, exc)),
            )
        )
    except CheckTargetError as exc:
        _write(f"[ERROR] Invalid arguments: {exc}")
        emitter.emit(_base(flags))
        return ExitCode.USAGE
    except BaseException as exc:
        progress.stop = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        with contextlib.suppress(Exception):
            _finish_wait(spec, progress, emitter)
        raise
    totals = _finish_wait(spec, progress, emitter)
    if totals.refused:
        return ExitCode.USAGE
    if totals.had_failure or progress.stop == Stop.TIMEOUT:
        return ExitCode.FAILURE
    return ExitCode.OK


def _cancel(
    directory: Path | None,
    batch_ids: tuple[str, ...],
    settings: Settings,
    args: argparse.Namespace,
) -> Coroutine[Any, Any, CancelReport]:
    if directory is not None:
        _write(f"Cancelling the open batch submissions below '{directory}'...")
        return cancel_submissions(directory, settings, args=args)
    if batch_ids:
        _write(f"Cancelling {len(batch_ids)} batch(es)...")
        return cancel_batches(batch_ids, settings, args=args)
    _write("Listing the unfinished batches of every provider...")
    return cancel_unfinished(settings, args=args)


def _execute_cancel(
    args: argparse.Namespace, flags: Mapping[str, Any], emitter: SummaryEmitter
) -> int:
    spec = resolve_batch_cancel(flags)
    directory = None
    if spec.directory is not None:
        directory = spec.directory.expanduser().resolve()
    prepared = _prepare(spec, spec.settings, args, directory)
    if isinstance(prepared, str):
        _write(f"[ERROR] {prepared}")
        emitter.emit(cancel_fields())
        return ExitCode.USAGE
    report = asyncio.run(_cancel(directory, spec.batch_ids, prepared, args))
    for line in cancel_lines(report):
        _write(line)
    emitter.emit(cancel_fields(cancelled=report.cancelled, failed=report.failed))
    if report.refused:
        return ExitCode.USAGE
    return exit_code_for(failed=report.failed)


def _guarded(
    name: str,
    emitter: SummaryEmitter,
    empty: dict[str, Any],
    body: Callable[[], int],
) -> int:
    """Run *body*; map invalid options, Ctrl+C and a crash to exit codes.

    Invalid options return 2, Ctrl+C 130 and an unexpected exception 1; with
    ``--json`` the summary line is written on each of these exits.
    """
    try:
        return int(body())
    except SpecError as exc:
        _write(f"[ERROR] Invalid arguments: {exc}")
        emitter.emit(empty)
        return ExitCode.USAGE
    except KeyboardInterrupt:
        _write("[ERROR] Interrupted by the user.")
        emitter.emit(empty)
        return ExitCode.INTERRUPTED
    except Exception as exc:
        logger.exception("%s failed: %s", name, exc)
        emitter.emit(empty)
        return ExitCode.FAILURE


def execute_check(args: argparse.Namespace) -> int:
    """Run the parsed batch check options and return the exit code."""
    configure_logging()
    flags = BATCH_CHECK_OPTIONS.values(args)
    emitter = SummaryEmitter(sys.stdout if flags.get("json") else None)
    return _guarded(
        "Batch check",
        emitter,
        _base(flags),
        lambda: _execute_check(args, flags, emitter),
    )


def execute_cancel(args: argparse.Namespace) -> int:
    """Run the parsed batch cancel options and return the exit code."""
    configure_logging()
    flags = BATCH_CANCEL_OPTIONS.values(args)
    emitter = SummaryEmitter(sys.stdout if flags.get("json") else None)
    return _guarded(
        "Batch cancel",
        emitter,
        cancel_fields(),
        lambda: _execute_cancel(args, flags, emitter),
    )


def report_usage_error(error: UsageError, argv: Sequence[str]) -> int:
    """Write a parse error like argparse does, plus the JSON line if asked."""
    sys.stderr.write(error.usage)
    sys.stderr.write(f"{error.prog}: error: {error}\n")
    stream = sys.stdout if "--json" in argv else None
    fields = cancel_fields() if argv[1:2] == ["cancel"] else check_fields()
    SummaryEmitter(stream).emit(fields)
    return ExitCode.USAGE
