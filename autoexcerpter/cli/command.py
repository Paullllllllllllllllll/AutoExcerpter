"""The run subcommand: plan and run, report on stderr, summarize on stdout.

Exit codes: 0 when every item completed or, under ``--batch``, was
submitted, 1 when an item failed, 2 for usage and configuration errors (an
input without items and a selection that matches nothing included), 130 on
Ctrl+C. With ``--json`` exactly one JSON summary line goes to stdout on every
exit. The deprecated ``--transcription-format`` is passed on to the spec with
a warning.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from autoexcerpter import ext
from autoexcerpter import run as core
from autoexcerpter.cli.heartbeat import with_heartbeat
from autoexcerpter.cli.parser import UsageError
from autoexcerpter.cli.reporter import StderrReporter
from autoexcerpter.common.contract import (
    USAGE_KEY,
    ExitCode,
    SummaryEmitter,
    exit_code_for,
    usage_field,
)
from autoexcerpter.common.log_setup import configure_logging
from autoexcerpter.common.options import Condition, Option
from autoexcerpter.common.usage import Usage, UsageTotals
from autoexcerpter.llm import collect_usage
from autoexcerpter.settings import SettingsError, SpecError, load, resolve
from autoexcerpter.spec import (
    OPTIONS,
    TRANSCRIPTION_FORMAT_ALIAS,
    RunSpec,
    effective_spec,
)

__all__ = [
    "TRANSCRIPTION_FORMAT_DEPRECATION",
    "condition_warnings",
    "dry_run_fields",
    "execute",
    "plan_fields",
    "report_usage_error",
    "summary_fields",
]

logger = logging.getLogger(__name__)

TRANSCRIPTION_FORMAT_DEPRECATION = (
    "--transcription-format is deprecated and will be removed in version 4; "
    "use --formats with transcription-{alias} instead."
)


def summary_fields(
    *,
    total: int = 0,
    complete: int = 0,
    failed: int = 0,
    skipped: int = 0,
    submitted: int = 0,
    outputs: Sequence[str] = (),
    usage: Mapping[str, Usage] | None = None,
) -> dict[str, Any]:
    """Return the fields of the JSON run summary.

    *submitted* counts the items whose units went to batches under
    ``--batch``; *usage* holds the run's tokens per role; the summary adds
    their total.
    """
    return {
        "items_total": total,
        "items_complete": complete,
        "items_failed": failed,
        "items_skipped": skipped,
        "items_submitted": submitted,
        "outputs": list(outputs),
        USAGE_KEY: usage_field(usage),
    }


def plan_fields(run_plan: core.RunPlan) -> dict[str, Any]:
    """Return the fields a dry run adds to the counts of the run summary.

    ``to_process`` holds the items to process with their resume state and
    reused pages; ``skipped`` names the items left alone (complete, legacy
    and blocked); ``legacy``, ``blocked`` and ``refused`` list those items
    with the reason. ``spec`` holds the effective value and source of every
    run option.
    """
    return {
        "to_process": [
            {
                "name": item.name,
                "state": item.state,
                "completed_pages": item.completed_pages,
            }
            for item in run_plan.to_process
        ],
        "skipped": [item.name for item in run_plan.not_processed],
        "legacy": _reasons(run_plan.legacy),
        "blocked": _reasons(run_plan.blocked),
        "refused": _reasons(run_plan.refused),
        "spec": run_plan.effective_spec(),
    }


def dry_run_fields(run_plan: core.RunPlan) -> dict[str, Any]:
    """Return the JSON summary of a dry run: the counts and the plan."""
    counts = summary_fields(
        total=len(run_plan.to_process), skipped=len(run_plan.not_processed)
    )
    return {**counts, **plan_fields(run_plan)}


def _reasons(items: Sequence[core.PlannedItem]) -> list[dict[str, str]]:
    return [{"name": item.name, "reason": item.reason} for item in items]


def condition_warnings(
    flags: Mapping[str, Any], defaults: Mapping[str, Any], spec: RunSpec
) -> list[str]:
    """Return one warning per given option whose condition does not hold.

    Each conditional option of a phase is checked against that phase's
    resolved values; the warning names the flag or settings default that
    supplied the value. Values from code defaults never warn.
    """
    flat = effective_spec(spec)
    values = {
        option.name: flat[option.target]["value"]
        for option in OPTIONS
        if option.target is not None
    }
    offenders: dict[tuple[str, str], tuple[Condition, list[str]]] = {}
    for option in OPTIONS:
        condition = option.condition
        if condition is None or option.target is None or condition.holds(values):
            continue
        origin = _origin(option, (("flag", flags), ("settings", defaults)))
        if origin is None:
            continue
        rule = OPTIONS[origin[1]].condition or condition
        subject = condition.option.replace("_", " ")
        reason = f"the {subject} is {values[condition.option]}"
        offenders.setdefault(origin, (rule, []))[1].append(reason)
    messages = []
    for (level, name), (rule, reasons) in offenders.items():
        flag = OPTIONS[name].flag
        label = flag if level == "flag" else f"the settings default {name} ({flag})"
        messages.append(
            f"{label} applies {rule.describe()} and has no effect here: "
            + ", ".join(reasons)
        )
    return messages


def _origin(
    option: Option, levels: Sequence[tuple[str, Mapping[str, Any]]]
) -> tuple[str, str] | None:
    """Return the level and option name that supply *option*, if not a default."""
    chain = [option.name]
    while (fallback := OPTIONS[chain[-1]].fallback) is not None:
        if fallback in chain:
            break
        chain.append(fallback)
    for level, given in levels:
        for name in chain:
            if name in given:
                return level, name
    return None


def report_usage_error(error: UsageError, argv: Sequence[str]) -> int:
    """Write a parse error like argparse does, plus the JSON line if asked."""
    sys.stderr.write(error.usage)
    sys.stderr.write(f"{error.prog}: error: {error}\n")
    stream = sys.stdout if "--json" in argv else None
    SummaryEmitter(stream, dry_run="--dry-run" in argv).emit(summary_fields())
    return ExitCode.USAGE


def _execute(
    args: argparse.Namespace,
    flags: Mapping[str, Any],
    emitter: SummaryEmitter,
    reporter: StderrReporter,
) -> int:
    try:
        settings = load(flags.get("settings"), blocks=ext.settings_blocks())
        configure_logging(log_file=settings.log_file())
        resolution = resolve(settings, flags)
    except (SettingsError, SpecError) as exc:
        reporter.error(str(exc))
        emitter.emit(summary_fields())
        return ExitCode.USAGE
    spec = resolution.spec
    for message in condition_warnings(flags, settings.defaults, spec):
        reporter.warning(message)
    try:
        run_plan = core.plan(spec, settings, sources=resolution.sources, args=args)
    except core.PlanError as exc:
        reporter.error(str(exc))
        emitter.emit(summary_fields(total=exc.items_total))
        return exc.exit_code
    reporter.plan(run_plan)

    skipped = len(run_plan.not_processed)
    if not run_plan.dry_run and not run_plan.has_work:
        emitter.emit(summary_fields(skipped=skipped))
        return ExitCode.OK
    try:
        ext.startup(spec, settings=settings, args=args)
    except SettingsError as exc:
        reporter.error(str(exc))
        emitter.emit(summary_fields(total=len(run_plan.to_process), skipped=skipped))
        return ExitCode.USAGE
    if run_plan.dry_run:
        reporter.dry_run(run_plan)
        emitter.emit(dry_run_fields(run_plan))
        return ExitCode.OK

    events, progress = with_heartbeat(reporter)
    try:
        result = asyncio.run(core.run(run_plan, events, args=args))
    except core.PlanError as exc:
        reporter.error(str(exc))
        emitter.emit(summary_fields(total=exc.items_total, skipped=skipped))
        return exc.exit_code
    if progress is not None:
        progress.run_finished()
    reporter.completion(result)
    reporter.incomplete(result)
    emitter.emit(
        summary_fields(
            total=len(result.items),
            complete=result.complete,
            failed=result.failed,
            skipped=len(result.skipped),
            submitted=result.submitted,
            outputs=result.outputs,
            usage=result.usage,
        )
    )
    return exit_code_for(failed=result.failed)


def execute(args: argparse.Namespace) -> int:
    """Run the parsed run options and return the exit code.

    Ctrl+C returns 130 and an unexpected exception returns 1; with ``--json``
    the summary line is written on every exit path, and on those two it
    holds the tokens spent until then.
    """
    configure_logging()
    flags = OPTIONS.values(args)
    emitter = SummaryEmitter(
        sys.stdout if flags.get("json") else None,
        dry_run=bool(flags.get("dry_run")),
    )
    reporter = StderrReporter()
    alias = getattr(args, TRANSCRIPTION_FORMAT_ALIAS, None)
    if alias is not None:
        flags[TRANSCRIPTION_FORMAT_ALIAS] = alias
        reporter.warning(TRANSCRIPTION_FORMAT_DEPRECATION.format(alias=alias))
    with collect_usage() as spent:
        try:
            return int(_execute(args, flags, emitter, reporter))
        except KeyboardInterrupt:
            reporter.error("Interrupted by the user.")
            emitter.emit(summary_fields(usage=_by_role(spent)))
            return ExitCode.INTERRUPTED
        except Exception as exc:
            logger.exception("Critical error in main execution flow: %s", exc)
            emitter.emit(summary_fields(usage=_by_role(spent)))
            return ExitCode.FAILURE


def _by_role(totals: UsageTotals) -> dict[str, Usage]:
    return {role: totals.get(role) for role in totals.roles}
