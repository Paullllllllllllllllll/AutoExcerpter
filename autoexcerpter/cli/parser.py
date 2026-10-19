"""Argument parsing: the top-level parser and one function per subcommand.

The commands of :data:`autoexcerpter.spec.EXTENSION_COMMANDS`, those that
call providers, also take the extension's arguments.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, NoReturn

from autoexcerpter import ext
from autoexcerpter.common.options import HelpFormatter, OptionTable
from autoexcerpter.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    DROP_OPTIONS,
    OPTIONS,
    RENDER_OPTIONS,
)

__all__ = [
    "BATCH_CANCEL_DESCRIPTION",
    "BATCH_CHECK_DESCRIPTION",
    "DROP_DESCRIPTION",
    "RENDER_DESCRIPTION",
    "RUN_DESCRIPTION",
    "SUBCOMMANDS",
    "UI_DESCRIPTION",
    "Parser",
    "UsageError",
    "add_batch_command",
    "add_drop_command",
    "add_render_command",
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
  0    every item completed or, under --batch, was submitted, or nothing
       was left to do
  1    an item failed, or another run leases one (a killed run keeps its
       lease until it expires)
  2    usage or configuration error, also an input without items and a
       --select or --files that matches nothing
  130  interrupted (Ctrl+C or Ctrl+Break)

With --json, stdout carries exactly one JSON summary line on every exit; \
progress, warnings and errors go to stderr.

Examples:
  Transcribe and summarize one PDF:
    autoexcerpter run --input book.pdf
  Every item in a folder, outputs in one folder:
    autoexcerpter run --input scans --all --output excerpts
  Two items of a folder by name, plain-text transcription and Markdown summary:
    autoexcerpter run --input scans --files a.pdf,b.pdf
      --formats transcription-txt,summary-md
  Separate transcription and summary models:
    autoexcerpter run --input book.pdf --transcription-model gpt-5.6-sol
      --summary-model claude-sonnet-5
  Plan a folder run without API calls and print the plan as JSON:
    autoexcerpter run --input scans --all --dry-run --json
  Submit a folder to the batch APIs, then collect it later:
    autoexcerpter run --input scans --all --batch
    autoexcerpter batch check --input scans
"""

UI_DESCRIPTION = (
    "Guided session in an interactive terminal. The first step chooses the "
    "task: a new run, checking or cancelling batches, rendering stored "
    "documents again, or dropping stored documents. Each task asks "
    "its options one step at a time, shows a review screen with the "
    "equivalent command, then runs it. Exit codes are those of the chosen "
    "task's command."
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
    """Add the option table and the extension arguments."""
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


RENDER_DESCRIPTION = (
    "Write the transcription and summary files of stored documents again from "
    "their autoexcerpter.sqlite, without model calls."
)

RENDER_EPILOG = """\
Without --formats each document is rendered in the formats it was rendered \
in before. A file changed outside the tool is left as it is unless --force.

Exit codes:
  0    every file holds its rendering or was left as edited
  1    a file could not be replaced, or another run holds a document
  2    usage error, a missing settings file, or a database the store refuses
  130  interrupted (Ctrl+C or Ctrl+Break)
"""


def add_render_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the render subcommand."""
    render: argparse.ArgumentParser = commands.add_parser(
        "render",
        help="render stored documents again without model calls",
        description=RENDER_DESCRIPTION,
        epilog=RENDER_EPILOG,
        formatter_class=HelpFormatter,
    )
    RENDER_OPTIONS.add_arguments(render)
    return render


DROP_DESCRIPTION = (
    "Delete stored documents from autoexcerpter.sqlite with their pages, "
    "summaries, citations and provider responses, and delete their rendered "
    "files, so the next run processes them from scratch."
)

DROP_EPILOG = """\
--select numbers the stored documents from 1 in name order; --files names \
documents by name or by the file name of their input. A rendered file is \
deleted only while it holds the tool's rendering; a file changed outside the \
tool is kept and named. A document that another live run leases, or whose \
pages an unfinished batch holds, is refused.

Exit codes:
  0    every selected document was dropped (or would be, with --dry-run)
  1    a document was refused
  2    usage error, no store at the input, a selection that is malformed or
       matches nothing, or a database the store refuses
  130  interrupted (Ctrl+C or Ctrl+Break)

With --json, stdout carries exactly one JSON summary line on every exit.
"""


def add_drop_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the drop subcommand."""
    drop: argparse.ArgumentParser = commands.add_parser(
        "drop",
        help="delete stored documents and their files",
        description=DROP_DESCRIPTION,
        epilog=DROP_EPILOG,
        usage="%(prog)s --input PATH (--select EXPR | --files LIST | --all) [options]",
        formatter_class=HelpFormatter,
    )
    DROP_OPTIONS.add_arguments(drop)
    return drop


BATCH_DESCRIPTION = (
    "Check and continue, or cancel, the batch work that run --batch submitted."
)

BATCH_CHECK_DESCRIPTION = (
    "Check the batch submissions found below a folder: store the results of "
    "every finished batch, submit the summaries of the pages it transcribed, "
    "and write the files of every document left with nothing to do."
)

BATCH_CHECK_EPILOG = """\
Without --wait a check makes one pass. A run --batch that summarizes then \
needs two checks after its batches finish: the first collects the \
transcriptions and submits the summaries, the second collects the summaries \
and writes the files. With --wait the check repeats every --poll-minutes \
until no batch work is pending; no check starts after --wait-hours, but a \
running one finishes.

Exit codes:
  0    nothing failed
  1    a submission or a document failed, or a submission is unresolved;
       with --wait also when the deadline passed with work pending or a
       document needs a run
  2    usage error, a missing folder, a database the store refuses, or an
       --adopt or --abandon that names nothing to act on
  130  interrupted (Ctrl+C or Ctrl+Break)

With --json, stdout carries exactly one JSON summary line on every exit.
"""

BATCH_CANCEL_DESCRIPTION = (
    "Cancel the named batches, every unfinished batch, or the open batch "
    "submissions below a folder."
)

BATCH_CANCEL_EPILOG = """\
Exit codes:
  0    every batch was cancelled
  1    a cancellation or a provider listing failed
  2    usage error, a missing folder or a database the store refuses
  130  interrupted (Ctrl+C or Ctrl+Break)

With --json, stdout carries exactly one JSON summary line on every exit.
"""


def _add_batch_action(
    actions: Commands,
    name: str,
    table: OptionTable,
    *,
    help: str,
    description: str,
    epilog: str,
) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = actions.add_parser(
        name,
        help=help,
        description=description,
        epilog=epilog,
        formatter_class=HelpFormatter,
    )
    table.add_arguments(parser)
    ext.register_args(parser)
    return parser


def add_batch_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the batch subcommand with its check and cancel subcommands."""
    batch: argparse.ArgumentParser = commands.add_parser(
        "batch",
        help="check or cancel batch submissions",
        description=BATCH_DESCRIPTION,
        formatter_class=HelpFormatter,
    )
    actions = batch.add_subparsers(
        dest="batch_command", metavar="COMMAND", title="commands"
    )
    batch.set_defaults(group_parser=batch)
    _add_batch_action(
        actions,
        "check",
        BATCH_CHECK_OPTIONS,
        help="collect finished batches and continue their documents",
        description=BATCH_CHECK_DESCRIPTION,
        epilog=BATCH_CHECK_EPILOG,
    )
    _add_batch_action(
        actions,
        "cancel",
        BATCH_CANCEL_OPTIONS,
        help="cancel batches",
        description=BATCH_CANCEL_DESCRIPTION,
        epilog=BATCH_CANCEL_EPILOG,
    )
    return batch


def add_ui_command(commands: Commands) -> argparse.ArgumentParser:
    """Add the ui subcommand: the settings option and the extension arguments."""
    ui: argparse.ArgumentParser = commands.add_parser(
        "ui",
        help="guided session in an interactive terminal",
        description=UI_DESCRIPTION,
        formatter_class=HelpFormatter,
    )
    OPTIONS.subset(("settings",)).add_arguments(ui)
    ext.register_args(ui)
    return ui


SUBCOMMANDS: tuple[Callable[[Commands], argparse.ArgumentParser], ...] = (
    add_run_command,
    add_batch_command,
    add_render_command,
    add_drop_command,
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
