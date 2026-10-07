"""Run screen of the guided UI: live progress bars, plan and summary tables.

:class:`RichProgress` implements :class:`~autoexcerpter.events.RunEvents` on a
rich console. On a terminal it shows an overall bar for the items, a page bar
for the current item and a status line for waits; warnings and one result line
per item are printed above the bars, and so are log records written to stderr
while the display runs. On any other console it prints plain lines only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any, Self

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from autoexcerpter.common.usage import Usage
from autoexcerpter.events import ItemFinished, ItemStarted, PageDone, Waiting
from autoexcerpter.run import RunPlan, RunResult

__all__ = ["RichProgress", "plan_table", "summary_table"]

_PAGE_KEYS = ("pages_ok", "pages_failed", "pages_deferred", "summary_failures")


def _kind_label(kind: str) -> str:
    return "PDF" if kind == "pdf" else "image folder"


def _count(report: Mapping[str, Any] | None, key: str) -> int:
    if not report:
        return 0
    return int(report.get(key, 0) or 0)


def _wait_line(event: Waiting) -> str:
    seconds = f" {event.seconds:.0f}s" if event.seconds is not None else ""
    return f"Waiting{seconds}: {event.reason}"


class RichProgress:
    """Show run events on a rich console; use as a context manager."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console(stderr=True)
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[note]}"),
            TimeElapsedColumn(),
            console=self.console,
            auto_refresh=False,
        )
        self._items_task: TaskID | None = None
        self._page_task: TaskID | None = None
        self._status: str = ""
        self._live: Live | None = None

    @property
    def live(self) -> bool:
        """Whether the live display is running."""
        return self._live is not None

    def __enter__(self) -> Self:
        if self.console.is_terminal and not self.console.is_dumb_terminal:
            self._live = Live(
                console=self.console,
                get_renderable=self._render,
                refresh_per_second=8,
                redirect_stdout=False,
                redirect_stderr=True,
                transient=False,
            )
            self._live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._status = ""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render(self) -> RenderableType:
        bars = self._progress.get_renderable()
        if not self._status:
            return bars
        return Group(bars, Text(self._status, style="yellow"))

    def _print(self, text: Text) -> None:
        self.console.print(text, soft_wrap=True)

    def item_started(self, event: ItemStarted) -> None:
        """Start the page bar of the item and size the overall bar."""
        self._status = ""
        if self._items_task is None:
            self._items_task = self._progress.add_task(
                "Items", total=event.total, note=""
            )
        else:
            self._progress.update(self._items_task, total=event.total)
        if self._page_task is not None:
            self._progress.remove_task(self._page_task)
        self._page_task = self._progress.add_task(event.name, total=None, note="")
        if not self.live:
            self._print(
                Text(
                    f"[{event.index}/{event.total}] {event.name} "
                    f"({_kind_label(event.kind)})"
                )
            )

    def page_done(self, event: PageDone) -> None:
        """Advance the page bar to *done* of *total* pending pages."""
        self._status = ""
        if self._page_task is None:
            return
        note = (
            f"+{event.already_complete} from an earlier run"
            if event.already_complete
            else ""
        )
        self._progress.update(
            self._page_task, completed=event.done, total=event.total, note=note
        )

    def waiting(self, event: Waiting) -> None:
        """Show the wait on the status line, or print it."""
        line = _wait_line(event)
        if self.live:
            self._status = line
        else:
            self._print(Text(line))

    def warning(self, message: str) -> None:
        """Print a warning above the bars."""
        for line in message.splitlines() or [""]:
            self._print(Text(f"Warning: {line}", style="yellow"))

    def item_finished(self, event: ItemFinished) -> None:
        """Print the item's result line and advance the overall bar."""
        self._status = ""
        if self._page_task is not None:
            self._progress.remove_task(self._page_task)
            self._page_task = None
        if self._items_task is not None:
            self._progress.update(self._items_task, completed=event.index)
        verdict = "complete" if event.success else "incomplete"
        line = Text(f"[{event.index}/{event.total}] {event.name}: ")
        line.append(verdict, style="green" if event.success else "red")
        report = event.report
        if isinstance(report, Mapping):
            line.append(
                f"; {_count(report, 'pages_ok')} page(s) ok, "
                f"{_count(report, 'pages_failed')} failed, "
                f"{_count(report, 'pages_deferred')} deferred"
            )
            failures = _count(report, "summary_failures")
            if failures:
                line.append(f", {failures} summary failure(s)")
        self._print(line)
        if event.outputs:
            self._print(Text(f"    outputs: {', '.join(event.outputs)}"))


def _usage_table(usage: Mapping[str, Usage], total: Usage) -> Table:
    table = Table(title="Token usage", title_justify="left")
    table.add_column("Role")
    for header in ("Input", "Output", "Cached", "Reasoning", "Total"):
        table.add_column(header, justify="right")

    def cells(value: Usage) -> list[str]:
        return [
            f"{value.input_tokens:,}",
            f"{value.output_tokens:,}",
            f"{value.cached_tokens:,}",
            f"{value.reasoning_tokens:,}",
            f"{value.total_tokens:,}",
        ]

    for role, value in usage.items():
        table.add_row(role, *cells(value))
    table.add_section()
    table.add_row("Total", *cells(total), style="bold")
    return table


def _listing(label: str, values: Iterable[str]) -> Text:
    names = list(values)
    return Text(f"{label}: {', '.join(names) if names else 'none'}")


def summary_table(result: RunResult) -> RenderableType:
    """Return the run summary: items with totals, run facts and token usage."""
    table = Table(title="Run summary", title_justify="left")
    table.add_column("Item")
    table.add_column("Status")
    for header in ("Pages ok", "Failed", "Deferred", "Summary failures"):
        table.add_column(header, justify="right")
    totals = [0, 0, 0, 0]
    for item in result.items:
        counts = [_count(item.report, key) for key in _PAGE_KEYS]
        totals = [a + b for a, b in zip(totals, counts, strict=True)]
        table.add_row(
            item.name,
            Text(
                "complete" if item.success else "incomplete",
                style="green" if item.success else "red",
            ),
            *(str(count) for count in counts),
        )
    table.add_section()
    table.add_row(
        "Total",
        f"{result.complete} complete, {result.failed} incomplete",
        *(str(count) for count in totals),
        style="bold",
    )
    parts: list[RenderableType] = [
        table,
        Text(f"Run time: {result.seconds:.1f}s"),
        _listing("Skipped (already complete)", result.skipped),
        _listing("Databases written", result.databases),
    ]
    if result.usage:
        parts.append(_usage_table(result.usage, result.total_usage))
    else:
        parts.append(Text("Token usage: none reported"))
    return Group(*parts)


def plan_table(plan: RunPlan) -> Table:
    """Return the items a plan processes, with resume state, and its skips."""
    table = Table(
        title=(
            f"Plan: {len(plan.to_process)} to process, "
            f"{len(plan.skipped)} already complete"
        ),
        title_justify="left",
    )
    table.add_column("#", justify="right")
    table.add_column("Item")
    table.add_column("Kind")
    table.add_column("Resume state")
    table.add_column("Reused pages", justify="right")
    for index, planned in enumerate(plan.to_process, start=1):
        table.add_row(
            str(index),
            planned.name,
            _kind_label(planned.item.kind),
            planned.state.replace("_", " "),
            str(planned.completed_pages),
        )
    if plan.skipped:
        table.add_section()
        for planned in plan.skipped:
            table.add_row(
                "-",
                planned.name,
                _kind_label(planned.item.kind),
                "complete (skip)",
                "-",
            )
    return table
