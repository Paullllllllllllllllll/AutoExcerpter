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
        if args[0] in ("render", "migrate"):
            from autoexcerpter.cli import store

            return store.report_usage_error(error, args)
        if args[0] == "batch":
            from autoexcerpter.cli import batch

            return batch.report_usage_error(error, args)
        return command.report_usage_error(error, args)
    if namespace.command == "run":
        return command.execute(namespace)
    if namespace.command == "batch":
        from autoexcerpter.cli import batch

        if namespace.batch_command == "check":
            return batch.execute_check(namespace)
        if namespace.batch_command == "cancel":
            return batch.execute_cancel(namespace)
        namespace.group_parser.print_help()
        return 0
    if namespace.command == "render":
        from autoexcerpter.cli import store

        return store.execute_render(namespace)
    if namespace.command == "migrate":
        from autoexcerpter.cli import store

        return store.execute_migrate(namespace)
    if namespace.command == "ui":
        from autoexcerpter.ui import app

        return app.execute(namespace)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
