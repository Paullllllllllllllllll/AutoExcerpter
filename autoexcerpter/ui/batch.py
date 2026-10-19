"""The batch tasks of the guided UI: check and continue, or cancel, batches.

:func:`check_flow` asks the folder, the batch IDs and what to do with an
unresolved submission; :func:`cancel_flow` asks what to cancel: named batch
IDs, every unfinished batch, or the open submissions below a folder. Their
runs report like ``autoexcerpter batch check`` and ``batch cancel`` and
return the same exit codes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Final

from autoexcerpter import ext
from autoexcerpter.batch.cancel import (
    CancelReport,
    cancel_batches,
    cancel_submissions,
    cancel_unfinished,
)
from autoexcerpter.batch.check import CheckTargetError, check_batches, find_databases
from autoexcerpter.batch.report import cancel_lines, check_lines
from autoexcerpter.common.contract import ExitCode, exit_code_for
from autoexcerpter.common.options import OptionTable
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
from autoexcerpter.settings import SettingsError
from autoexcerpter.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    BatchCancelSpec,
    BatchCheckSpec,
    SpecError,
    resolve_batch_cancel,
    resolve_batch_check,
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

__all__ = ["CANCEL_PROGRAM", "CHECK_PROGRAM", "cancel_flow", "check_flow"]

CHECK_PROGRAM: Final = ("autoexcerpter", "batch", "check")
CANCEL_PROGRAM: Final = ("autoexcerpter", "batch", "cancel")

_REPORT = "report"
_ADOPT = "adopt"
_ABANDON = "abandon"
_IDS = "ids"
_ALL = "all"
_DIR = "dir"


def _existing_folder(text: str) -> str | None:
    if not text.strip():
        return "enter a folder"
    if not absolute(text).is_dir():
        return "no such folder"
    return None


def _given(text: str) -> str | None:
    return None if text.strip() else "enter an ID"


def _databases(folder: Path) -> int:
    return len(find_databases(folder)) if folder.is_dir() else 0


def _say(request: RunRequest, text: str) -> None:
    request.console.print(text, markup=False, highlight=False, soft_wrap=True)


def _startup(request: RunRequest, spec: BatchCheckSpec | BatchCancelSpec) -> bool:
    """Start the extension; print the refusal and return False on one."""
    try:
        ext.startup(spec, settings=request.session.settings, args=request.args)
    except SettingsError as exc:
        _say(request, f"Error: {exc}")
        return False
    return True


def _command(
    program: tuple[str, ...], table: OptionTable
) -> Callable[[State, Session], list[str]]:
    def command(state: State, session: Session) -> list[str]:
        answers = {
            name: value
            for name, value in state.answers.items()
            if name != "settings" and value not in (None, (), False)
        }
        return command_words(program, table.to_argv(answers), session, table)

    return command


# =============================================================================
# batch check
# =============================================================================


def _ask_folder(ctx: StepContext) -> Nav:
    state = ctx.state
    current = state.value("input")
    answer = ctx.prompter.path(
        "Folder with batch submissions (subfolders included):",
        default=str(absolute(current)) if current is not None else "",
        validate=_existing_folder,
        only_directories=True,
    )
    if answer is BACK:
        return BACK
    folder = absolute(answer)
    state.set("input", folder)
    ctx.prompter.show(f"  found {plural(_databases(folder), 'database')}")
    return DONE


def _ask_batch_ids(ctx: StepContext) -> Nav:
    return ctx.text_option(
        "batch_ids",
        "Batch IDs to finalize (comma-separated; empty = every batch):",
        empty=(),
    )


def _unresolved_action(state: State) -> str:
    if state.value("adopt") is not None:
        return _ADOPT
    if state.value("abandon") is not None:
        return _ABANDON
    return _REPORT


def _ask_unresolved(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def action() -> Nav:
        answer = ctx.prompter.select(
            "Unresolved submissions:",
            [
                Choice(_REPORT, "Report them", "with their candidate batches"),
                Choice(_ADOPT, "Adopt a provider batch ID", "then finalize it"),
                Choice(_ABANDON, "Abandon a submission ID", "its pages go pending"),
            ],
            default=_unresolved_action(state),
        )
        if answer is BACK:
            return BACK
        chosen["action"] = answer
        if answer == _REPORT:
            state.clear("adopt")
            state.clear("abandon")
        return DONE

    def target() -> Nav | None:
        name = chosen.get("action")
        if name not in (_ADOPT, _ABANDON):
            return None
        message = "Provider batch ID:" if name == _ADOPT else "Submission ID:"
        current = state.value(name)
        answer = ctx.prompter.text(message, default=current or "", validate=_given)
        if answer is BACK:
            return BACK
        state.clear(_ABANDON if name == _ADOPT else _ADOPT)
        state.set(name, answer.strip())
        return DONE

    return StepContext.sequence([action, target])


def _check_steps(session: Session) -> list[Step]:
    return [
        Step("folder", "Folder", ask=_ask_folder, sets=("input",)),
        Step("batches", "Batches", ask=_ask_batch_ids, sets=("batch_ids",)),
        Step(
            "unresolved", "Unresolved", ask=_ask_unresolved, sets=("adopt", "abandon")
        ),
    ]


def _unresolved_text(spec: BatchCheckSpec) -> str:
    if spec.adopt is not None:
        return f"adopt batch {spec.adopt}"
    if spec.abandon is not None:
        return f"abandon submission {spec.abandon}"
    return "report them"


def _check_review(state: State, session: Session) -> Review:
    try:
        spec = resolve_batch_check(state.answers)
    except SpecError as exc:
        return Review([], withheld=str(exc))
    folder = absolute(spec.directory)
    count = _databases(folder)
    lines = [
        Line("folder", [f"{folder}  ({plural(count, 'database')})"]),
        Line("batches", [", ".join(spec.batch_ids) or "every batch"]),
        Line("unresolved", [_unresolved_text(spec)]),
    ]
    if not folder.is_dir():
        return Review(lines, withheld=f"Input folder does not exist: {folder}")
    notes = [
        f"Checks the batch work of {plural(count, 'database')} below {folder}: "
        "finished batches are stored, the summaries of finished pages are "
        "submitted, and documents with nothing left are written as their run "
        "asked."
    ]
    if spec.batch_ids:
        notes.append(
            "Only the named batches are finalized; unresolved submissions are "
            "reconciled in every check."
        )
    return Review(lines, notes=notes)


def _run_check(request: RunRequest) -> int:
    try:
        spec = resolve_batch_check(dict(request.answers))
    except SpecError as exc:
        _say(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    directory = spec.directory.expanduser().resolve()
    if not directory.is_dir():
        _say(request, f"Error: Input folder does not exist: {directory}")
        return int(ExitCode.USAGE)
    if not _startup(request, spec):
        return int(ExitCode.USAGE)
    _say(request, f"Checking the batch work below '{directory}'...")
    try:
        result = asyncio.run(
            check_batches(
                directory,
                request.session.settings,
                batch_ids=spec.batch_ids,
                adopt=spec.adopt,
                abandon=spec.abandon,
                args=request.args,
            )
        )
    except CheckTargetError as exc:
        _say(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    for line in check_lines(result):
        _say(request, line)
    if result.refused:
        return int(ExitCode.USAGE)
    return int(ExitCode.FAILURE if result.had_failure else ExitCode.OK)


def check_flow() -> TaskFlow:
    """Return the batch check task: folder, batch IDs and unresolved action."""
    return TaskFlow(
        name="batch-check",
        title=TASKS["batch-check"],
        program=CHECK_PROGRAM,
        table=BATCH_CHECK_OPTIONS,
        steps=_check_steps,
        review=_check_review,
        command=_command(CHECK_PROGRAM, BATCH_CHECK_OPTIONS),
        run=_run_check,
    )


# =============================================================================
# batch cancel
# =============================================================================


def _cancel_target(state: State) -> str:
    if state.value("all"):
        return _ALL
    if state.value("input") is not None:
        return _DIR
    return _IDS


def _ask_target(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def target() -> Nav:
        answer = ctx.prompter.select(
            "Cancel:",
            [
                Choice(_IDS, "Named batch IDs"),
                Choice(_ALL, "Every unfinished batch", "of every provider"),
                Choice(_DIR, "The open submissions below a folder"),
            ],
            default=_cancel_target(state),
        )
        if answer is BACK:
            return BACK
        chosen["target"] = answer
        state.clear("all")
        if answer != _IDS:
            state.clear("batch_ids")
        if answer != _DIR:
            state.clear("input")
        if answer == _ALL:
            state.set("all", True)
        return DONE

    def batch_ids() -> Nav | None:
        if chosen.get("target") != _IDS:
            return None
        option = state.table["batch_ids"]
        current = ",".join(state.value("batch_ids") or ())

        def validate(text: str) -> str | None:
            if not text.strip(" ,"):
                return "enter at least one batch ID"
            return None

        answer = ctx.prompter.text(
            "Batch IDs (comma-separated):", default=current, validate=validate
        )
        if answer is BACK:
            return BACK
        state.set("batch_ids", option.convert(answer.strip()))
        return DONE

    def folder() -> Nav | None:
        if chosen.get("target") != _DIR:
            return None
        current = state.value("input")
        answer = ctx.prompter.path(
            "Folder with the open submissions (subfolders included):",
            default="" if current is None else str(current),
            validate=_existing_folder,
            only_directories=True,
        )
        if answer is BACK:
            return BACK
        path = absolute(answer)
        state.set("input", path)
        ctx.prompter.show(f"  found {plural(_databases(path), 'database')}")
        return DONE

    return StepContext.sequence([target, batch_ids, folder])


def _cancel_steps(session: Session) -> list[Step]:
    return [
        Step(
            "target",
            "What to cancel",
            ask=_ask_target,
            sets=("batch_ids", "all", "input"),
        )
    ]


def _cancel_review(state: State, session: Session) -> Review:
    try:
        spec = resolve_batch_cancel(state.answers)
    except SpecError as exc:
        return Review([], withheld=str(exc))
    withheld: str | None = None
    if spec.directory is not None:
        folder = absolute(spec.directory)
        count = _databases(folder)
        line = f"the open submissions below {folder}  ({plural(count, 'database')})"
        note = (
            f"Cancels the submitted batches recorded in {plural(count, 'database')} "
            f"below {folder}; their pages return to pending and their batch work "
            "ends."
        )
        if not folder.is_dir():
            withheld = f"Input folder does not exist: {folder}"
    elif spec.batch_ids:
        line = ", ".join(spec.batch_ids)
        note = (
            f"Cancels {plural(len(spec.batch_ids), 'batch', 'batches')} with the "
            "provider each ID's shape names."
        )
    else:
        line = "every unfinished batch of every provider"
        note = (
            "Lists the unfinished batches of every provider whose key variable is "
            "set and cancels each."
        )
    return Review([Line("target", [line])], notes=[note], withheld=withheld)


def _cancel(
    request: RunRequest, spec: BatchCancelSpec, directory: Path | None
) -> Coroutine[Any, Any, CancelReport]:
    settings = request.session.settings
    args = request.args
    if directory is not None:
        _say(request, f"Cancelling the open batch submissions below '{directory}'...")
        return cancel_submissions(directory, settings, args=args)
    if spec.batch_ids:
        _say(request, f"Cancelling {len(spec.batch_ids)} batch(es)...")
        return cancel_batches(spec.batch_ids, settings, args=args)
    _say(request, "Listing the unfinished batches of every provider...")
    return cancel_unfinished(settings, args=args)


def _run_cancel(request: RunRequest) -> int:
    try:
        spec = resolve_batch_cancel(dict(request.answers))
    except SpecError as exc:
        _say(request, f"Error: {exc}")
        return int(ExitCode.USAGE)
    directory = None
    if spec.directory is not None:
        directory = spec.directory.expanduser().resolve()
        if not directory.is_dir():
            _say(request, f"Error: Input folder does not exist: {directory}")
            return int(ExitCode.USAGE)
    if not _startup(request, spec):
        return int(ExitCode.USAGE)
    report = asyncio.run(_cancel(request, spec, directory))
    for line in cancel_lines(report):
        _say(request, line)
    if report.refused:
        return int(ExitCode.USAGE)
    return int(exit_code_for(failed=report.failed))


def cancel_flow() -> TaskFlow:
    """Return the batch cancel task: what to cancel."""
    return TaskFlow(
        name="batch-cancel",
        title=TASKS["batch-cancel"],
        program=CANCEL_PROGRAM,
        table=BATCH_CANCEL_OPTIONS,
        steps=_cancel_steps,
        review=_cancel_review,
        command=_command(CANCEL_PROGRAM, BATCH_CANCEL_OPTIONS),
        run=_run_cancel,
    )
