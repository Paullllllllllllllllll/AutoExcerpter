"""The run task of the guided UI: transcribe and summarize documents.

:func:`flow` builds the task from the wizard steps of :mod:`.wizard` and
the review of :mod:`.review`. Its run plans and runs like ``autoexcerpter
run``, with the progress screen and the summary table, and returns the same
exit codes.
"""

from __future__ import annotations

import asyncio
from typing import Final

from autoexcerpter import ext
from autoexcerpter import run as core
from autoexcerpter.common.contract import ExitCode, exit_code_for
from autoexcerpter.common.wizard import State
from autoexcerpter.settings import SettingsError
from autoexcerpter.spec import OPTIONS
from autoexcerpter.ui.progress import RichProgress, plan_table, summary_table
from autoexcerpter.ui.review import build_review, equivalent_command
from autoexcerpter.ui.session import (
    TASKS,
    RunRequest,
    TaskFlow,
    plural,
    resolution,
)
from autoexcerpter.ui.wizard import build_steps

__all__ = ["PROGRAM", "flow"]

PROGRAM: Final = ("autoexcerpter", "run")


def _say(request: RunRequest, text: str) -> None:
    request.console.print(text, markup=False, highlight=False, soft_wrap=True)


def _run(request: RunRequest) -> int:
    session = request.session
    state = State(OPTIONS, session.settings.defaults, request.answers)
    try:
        resolved = resolution(state, session)
    except ValueError as exc:
        _say(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    spec = resolved.spec
    try:
        run_plan = core.plan(
            spec, session.settings, sources=resolved.sources, args=request.args
        )
    except core.PlanError as exc:
        _say(request, f"Error: {exc}")
        return int(exc.exit_code)
    if not run_plan.dry_run and not run_plan.has_work:
        skipped = plural(len(run_plan.skipped), "item")
        _say(request, f"Nothing to do: {skipped} already complete.")
        return int(ExitCode.OK)
    try:
        ext.startup(spec, settings=session.settings, args=request.args)
    except SettingsError as exc:
        _say(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    if run_plan.dry_run:
        request.console.print(plan_table(run_plan))
        return int(ExitCode.OK)
    try:
        with RichProgress(request.console) as progress:
            result = asyncio.run(core.run(run_plan, progress, args=request.args))
    except core.PlanError as exc:
        _say(request, f"Error: {exc}")
        return int(exc.exit_code)
    request.console.print(summary_table(result))
    return int(exit_code_for(failed=result.failed))


def flow() -> TaskFlow:
    """Return the run task: the wizard's steps, review and command."""
    return TaskFlow(
        name="run",
        title=TASKS["run"],
        program=PROGRAM,
        table=OPTIONS,
        steps=build_steps,
        review=build_review,
        command=equivalent_command,
        run=_run,
    )
