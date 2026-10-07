"""Argument parsing: the top-level parser and one function per subcommand."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, NoReturn

from autoexcerpter import ext
from autoexcerpter.common.options import HelpFormatter
from autoexcerpter.spec import OPTIONS

__all__ = [
    "RUN_DESCRIPTION",
    "SUBCOMMANDS",
    "UI_DESCRIPTION",
    "Parser",
    "UsageError",
    "add_run_arguments",
    "add_run_command",
    "add_ui_command",
    "build_parser",
]

PROG = "autoexcerpter"

DESCRIPTION = "Transcribe and summarize PDFs and image folders."

TOP_EPILOG = "Run 'autoexcerpter <command> --help' for the options of a command."

RUN_DESCRIPTION = (
    "Transcribe PDFs and image folders with a vision model and summarize them."
)

RUN_EPILOG = """\
Precedence: a flag wins over the settings defaults: block, which wins over the \
code default. --settings FILE reads another settings file instead of \
config/settings.yaml.

Exit codes:
  0    every item completed, or nothing was left to do
  1    an item failed, or no item could be processed
  2    usage or configuration error
  130  interrupted (Ctrl+C)

With --json, stdout carries exactly one JSON summary line on every exit; \
progress, warnings and errors go to stderr.

Examples:
  Transcribe and summarize one PDF:
    autoexcerpter run --input book.pdf
  Every item in a folder, outputs in one folder:
    autoexcerpter run --input scans --all --output excerpts
  Separate transcription and summary models:
    autoexcerpter run --input book.pdf --transcription-model gpt-5.6-sol
      --summary-model claude-sonnet-5
  Plan a folder run without API calls and print the plan as JSON:
    autoexcerpter run --input scans --all --dry-run --json
"""

UI_DESCRIPTION = (
    "Guided run: walks through input, outputs, models and image settings, "
    "shows a review screen with the equivalent command, then runs. Needs an "
    "interactive terminal."
)


class UsageError(Exception):
    """The command line is invalid; *usage* is the parser's usage text."""

    def __init__(self, message: str, *, prog: str, usage: str) -> None:
        super().__init__(message)
        self.prog = prog
        self.usage = usage


class Parser(argparse.ArgumentParser):
    """An argument parser that raises :class:`UsageError` instead of exiting.

    Subcommand parsers created from it share the class, so every parse
    error reaches the caller, which can still write the JSON summary.
    """

    def error(self, message: str) -> NoReturn:
        raise UsageError(message, prog=self.prog, usage=self.format_usage())


# The action ``add_subparsers`` returns; its class is private to argparse.
Commands = Any


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the option table and the extension arguments to *parser*."""
    OPTIONS.add_arguments(parser)
    ext.register_args(parser)


def add_run_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the run subcommand."""
    run: argparse.ArgumentParser = commands.add_parser(
        "run",
        help="transcribe and summarize documents",
        description=RUN_DESCRIPTION,
        epilog=RUN_EPILOG,
        usage="%(prog)s --input PATH [options]",
        formatter_class=HelpFormatter,
    )
    add_run_arguments(run)
    return run


def add_ui_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the ui subcommand: the settings option and the extension arguments."""
    ui: argparse.ArgumentParser = commands.add_parser(
        "ui",
        help="guided run in an interactive terminal",
        description=UI_DESCRIPTION,
        formatter_class=HelpFormatter,
    )
    OPTIONS.subset(("settings",)).add_arguments(ui)
    ext.register_args(ui)
    return ui


SUBCOMMANDS: tuple[Callable[[Commands], argparse.ArgumentParser], ...] = (
    add_run_command,
    add_ui_command,
)


def build_parser() -> Parser:
    """Return the top-level parser with every subcommand."""
    parser = Parser(
        prog=PROG,
        description=DESCRIPTION,
        epilog=TOP_EPILOG,
        formatter_class=HelpFormatter,
    )
    commands = parser.add_subparsers(
        dest="command", metavar="COMMAND", title="commands"
    )
    for add_command in SUBCOMMANDS:
        add_command(commands)
    return parser
