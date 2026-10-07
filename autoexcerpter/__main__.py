"""Console entry point: dispatch the subcommand; bare ``autoexcerpter`` prints help."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from autoexcerpter.cli import command
from autoexcerpter.cli.parser import UsageError, build_parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the subcommand in *argv* (``sys.argv[1:]`` when None)."""
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help()
        return 0
    try:
        namespace = parser.parse_args(args)
    except UsageError as error:
        return command.report_usage_error(error, args)
    if namespace.command == "run":
        return command.execute(namespace)
    if namespace.command == "ui":
        from autoexcerpter.ui import app

        return app.execute(namespace)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
