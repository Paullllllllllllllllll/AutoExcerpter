"""The ui subcommand: a guided run in an interactive terminal.

:func:`execute` is what ``autoexcerpter ui`` dispatches to; :func:`run_app`
does the same with an injectable prompter, console, command stream,
clipboard runner and environment, for tests and recorded sessions. Exit
codes are those of ``autoexcerpter run``: 0 when every item completed or
nothing ran, 1 when an item failed, 2 for usage and configuration errors,
130 on Ctrl+C.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, TextIO

from rich.console import Console

from autoexcerpter import ext
from autoexcerpter import run as core
from autoexcerpter.common.contract import ExitCode, exit_code_for
from autoexcerpter.common.log_setup import configure_logging
from autoexcerpter.common.prompts import QuestionaryPrompter
from autoexcerpter.common.wizard import (
    ClipboardRunner,
    Decision,
    Prompter,
    State,
    command_line,
    copy_to_clipboard,
)
from autoexcerpter.settings import SettingsError, load
from autoexcerpter.spec import OPTIONS
from autoexcerpter.ui.progress import RichProgress, plan_table, summary_table
from autoexcerpter.ui.session import Session, plural, resolution
from autoexcerpter.ui.wizard import build_wizard

__all__ = ["NOT_INTERACTIVE", "execute", "run_app"]

logger = logging.getLogger(__name__)

NOT_INTERACTIVE: Final = (
    "autoexcerpter ui needs an interactive terminal (stdin and stdout). "
    "For scripts and agents, use 'autoexcerpter run' with flags; see "
    "'autoexcerpter run --help'."
)


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
        session = Session(
            settings,
            settings_path,
            environ if environ is not None else os.environ,
        )
        return _guided(
            session,
            prompter if prompter is not None else QuestionaryPrompter(),
            screen,
            stdout if stdout is not None else sys.stdout,
            clipboard,
            args if args is not None else argparse.Namespace(),
        )
    except KeyboardInterrupt:
        _say(screen, "Interrupted by the user.")
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:
        logger.exception("Critical error in the guided run: %s", exc)
        return int(ExitCode.FAILURE)


def _guided(
    session: Session,
    prompter: Prompter,
    console: Console,
    stdout: TextIO,
    clipboard: ClipboardRunner | None,
    args: argparse.Namespace,
) -> int:
    outcome = build_wizard(session, prompter, width=console.width).run()
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
    return _run(session, outcome.answers, console, args)


def _run(
    session: Session,
    answers: Mapping[str, Any],
    console: Console,
    args: argparse.Namespace,
) -> int:
    state = State(OPTIONS, session.settings.defaults, answers)
    try:
        resolved = resolution(state, session)
    except ValueError as exc:
        _say(console, f"Error: {exc}")
        return int(ExitCode.USAGE)
    spec = resolved.spec
    try:
        run_plan = core.plan(spec, session.settings, sources=resolved.sources)
    except core.PlanError as exc:
        _say(console, f"Error: {exc}")
        return int(exc.exit_code)
    if not run_plan.dry_run and not run_plan.to_process:
        skipped = plural(len(run_plan.skipped), "item")
        _say(console, f"Nothing to do: {skipped} already complete.")
        return int(ExitCode.OK)
    try:
        ext.startup(spec, settings=session.settings, args=args)
    except SettingsError as exc:
        _say(console, f"Error: {exc}")
        return int(ExitCode.USAGE)
    if run_plan.dry_run:
        console.print(plan_table(run_plan))
        return int(ExitCode.OK)
    try:
        with RichProgress(console) as progress:
            result = asyncio.run(core.run(run_plan, progress))
    except core.PlanError as exc:
        _say(console, f"Error: {exc}")
        return int(exc.exit_code)
    console.print(summary_table(result))
    return int(exit_code_for(failed=result.failed))
