"""The ui subcommand: a guided session in an interactive terminal.

:func:`execute` is what ``autoexcerpter ui`` dispatches to; :func:`run_app`
does the same with an injectable prompter, console, command stream,
clipboard runner and environment, for tests and recorded sessions. The
session starts with the run task; choosing another task on the task step
rebuilds the wizard for it, starting at that task's first step. Exit codes
are those of the chosen task's command: 0 when every item completed or
nothing ran, 1 when an item failed, 2 for usage and configuration errors,
130 on Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, TextIO

from rich.console import Console

from autoexcerpter import ext
from autoexcerpter.common.contract import ExitCode
from autoexcerpter.common.log_setup import configure_logging
from autoexcerpter.common.prompts import QuestionaryPrompter
from autoexcerpter.common.wizard import (
    ClipboardRunner,
    Decision,
    Outcome,
    Prompter,
    command_line,
    copy_to_clipboard,
)
from autoexcerpter.settings import SettingsError, load
from autoexcerpter.ui import batch, run_task, store
from autoexcerpter.ui.session import (
    RunRequest,
    Session,
    TaskFlow,
    TaskSwitch,
    build_wizard,
)

__all__ = ["FLOWS", "NOT_INTERACTIVE", "execute", "run_app"]

logger = logging.getLogger(__name__)

NOT_INTERACTIVE: Final = (
    "autoexcerpter ui needs an interactive terminal (stdin and stdout). "
    "For scripts and agents, use 'autoexcerpter run' with flags; see "
    "'autoexcerpter run --help'."
)

FLOWS: dict[str, Callable[[], TaskFlow]] = {
    "run": run_task.flow,
    "batch-check": batch.check_flow,
    "batch-cancel": batch.cancel_flow,
    "render": store.render_flow,
    "migrate": store.migrate_flow,
}
"""The task flows by their :data:`autoexcerpter.ui.session.TASKS` names."""

FIRST_TASK: Final = "run"


def execute(args: argparse.Namespace) -> int:
    """Run the guided UI for the parsed ``ui`` arguments; return the exit code."""
    settings_path: Path | None = getattr(args, "settings", None)
    return run_app(settings_path, args=args)


def _interactive() -> bool:
    streams = (sys.stdin, sys.stdout)
    return all(stream is not None and stream.isatty() for stream in streams)


def _say(console: Console, text: str) -> None:
    console.print(text, markup=False, highlight=False, soft_wrap=True)


def run_app(
    settings_path: Path | None = None,
    *,
    args: argparse.Namespace | None = None,
    prompter: Prompter | None = None,
    console: Console | None = None,
    stdout: TextIO | None = None,
    clipboard: ClipboardRunner | None = None,
    environ: Mapping[str, str] | None = None,
    require_tty: bool = True,
) -> int:
    """Run the wizard, then the chosen action, and return the exit code.

    *args* holds the parsed ``ui`` arguments for the extension (default:
    none); *console* shows messages, the run screen and the summary
    (default: stderr); *stdout* receives the copied command; *clipboard*
    runs the clipboard program; *environ* is where the review looks up API
    key variables. ``require_tty=False`` skips the terminal check.
    """
    if require_tty and not _interactive():
        sys.stderr.write(NOT_INTERACTIVE + "\n")
        return int(ExitCode.USAGE)
    screen = console if console is not None else Console(stderr=True)
    try:
        try:
            settings = load(settings_path, blocks=ext.settings_blocks())
        except SettingsError as exc:
            _say(screen, f"Error: {exc}")
            return int(ExitCode.USAGE)
        configure_logging()
        namespace = args if args is not None else argparse.Namespace()
        session = Session(
            settings,
            settings_path,
            environ if environ is not None else os.environ,
            namespace,
        )
        return _guided(
            session,
            prompter if prompter is not None else QuestionaryPrompter(),
            screen,
            stdout if stdout is not None else sys.stdout,
            clipboard,
            namespace,
        )
    except KeyboardInterrupt:
        _say(screen, "Interrupted by the user.")
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:
        logger.exception("Critical error in the guided run: %s", exc)
        return int(ExitCode.FAILURE)


def _wizard_outcome(
    session: Session, prompter: Prompter, width: int
) -> tuple[TaskFlow, Outcome]:
    """Run the wizard, rebuilding it for each task the user switches to.

    A rebuilt wizard starts at the new task's first step.
    """
    flow = FLOWS[FIRST_TASK]()
    chosen = False
    while True:
        wizard = build_wizard(flow, session, prompter, width=width, chosen=chosen)
        try:
            return flow, wizard.run()
        except TaskSwitch as switch:
            flow = FLOWS[switch.task]()
            chosen = True


def _guided(
    session: Session,
    prompter: Prompter,
    console: Console,
    stdout: TextIO,
    clipboard: ClipboardRunner | None,
    args: argparse.Namespace,
) -> int:
    flow, outcome = _wizard_outcome(session, prompter, console.width)
    if outcome.action is Decision.QUIT:
        return int(ExitCode.OK)
    if outcome.action is Decision.COPY and outcome.command is not None:
        line = command_line(outcome.command)
        stdout.write(line + "\n")
        stdout.flush()
        if copy_to_clipboard(line, runner=clipboard):
            _say(console, "Copied the command to the clipboard.")
        else:
            _say(console, "Clipboard unavailable; copy the command above.")
        return int(ExitCode.OK)
    return int(flow.run(RunRequest(session, outcome.answers, console, args)))
