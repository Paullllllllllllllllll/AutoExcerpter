"""Human-readable run reporting on stderr; stdout carries only the JSON line."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

from autoexcerpter.events import ItemFinished, ItemStarted, PageDone, Waiting
from autoexcerpter.run import RunPlan, RunResult

__all__ = ["MAX_LISTED_OUTPUTS", "PROGRESS_STEPS", "StderrReporter"]

# Progress lines per item: one each time another 1/PROGRESS_STEPS of the
# pending pages is done; failed and deferred pages are always reported.
PROGRESS_STEPS = 20
MAX_LISTED_OUTPUTS = 20


def _write(line: str) -> None:
    print(line, file=sys.stderr)


def _page_counts(reports: list[Mapping[str, Any] | None]) -> tuple[int, int, int, int]:
    """Sum ``(pages_ok, pages_failed, pages_deferred, summary_failures)``."""
    totals = [0, 0, 0, 0]
    keys = ("pages_ok", "pages_failed", "pages_deferred", "summary_failures")
    for report in reports:
        if not report:
            continue
        for position, key in enumerate(keys):
            totals[position] += int(report.get(key, 0) or 0)
    return totals[0], totals[1], totals[2], totals[3]


class StderrReporter:
    """Write run events and run summaries to stderr."""

    def item_started(self, event: ItemStarted) -> None:
        """Name the item about to be processed."""
        kind = "PDF" if event.kind == "pdf" else "image folder"
        _write(f"[{event.index}/{event.total}] {event.name} ({kind})")

    def page_done(self, event: PageDone) -> None:
        """Report failed and deferred pages, and progress in steps."""
        if event.status != "ok":
            _write(
                f"  {event.item}: page {event.page + 1} {event.status} "
                f"({event.done}/{event.total})"
            )
            return
        step = max(1, math.ceil(event.total / PROGRESS_STEPS))
        if event.done % step and event.done != event.total:
            return
        line = f"  {event.item}: {event.done}/{event.total} page(s) done"
        if event.already_complete:
            line += f" ({event.already_complete} from an earlier run)"
        _write(line)

    def waiting(self, event: Waiting) -> None:
        """Report a wait."""
        seconds = f" {event.seconds:.0f}s" if event.seconds is not None else ""
        _write(f"  waiting{seconds}: {event.reason}")

    def warning(self, message: str) -> None:
        """Write a warning."""
        for line in message.splitlines() or [""]:
            _write(f"[WARNING] {line}")

    def item_finished(self, event: ItemFinished) -> None:
        """Write the item's page counts, time and outputs."""
        report = event.report
        if not isinstance(report, Mapping):
            return
        line = (
            f"  {event.name}: {report.get('pages_ok', 0)}/"
            f"{report.get('pages_attempted', 0)} page(s) ok, "
            f"{report.get('pages_failed', 0)} failed, "
            f"{report.get('pages_deferred', 0)} deferred"
        )
        elapsed = report.get("elapsed_s")
        avg_api = report.get("avg_api_s")
        if elapsed is not None:
            line += f"; {elapsed:.1f}s"
        if avg_api is not None:
            line += f"; avg API {avg_api:.1f}s"
        _write(line)
        outputs = report.get("outputs") or []
        if outputs:
            _write(f"    outputs: {', '.join(outputs)}")

    def error(self, message: str) -> None:
        """Write an error."""
        for line in message.splitlines() or [""]:
            _write(f"[ERROR] {line}")

    def plan(self, run_plan: RunPlan) -> None:
        """Write the plan's warnings."""
        for message in run_plan.warnings:
            self.warning(message)

    def dry_run(self, run_plan: RunPlan) -> None:
        """Write one line per planned and per skipped item."""
        for item in run_plan.to_process:
            _write(
                f"DRY-RUN: {item.name} -> {item.state} "
                f"({item.completed_pages} completed page(s))"
            )
        for item in run_plan.skipped:
            _write(f"DRY-RUN: {item.name} -> already complete (skip)")

    def completion(self, result: RunResult) -> None:
        """Write the run overview: items, pages, run time and outputs."""
        ok, failed, deferred, summary_failures = _page_counts(
            [item.report for item in result.items]
        )
        pages = f"  pages: {ok} ok, {failed} failed, {deferred} deferred"
        if summary_failures:
            pages += f", {summary_failures} summary failure(s)"
        lines = [
            "Run complete:",
            f"  items: {result.complete} processed, {result.failed} failed, "
            f"{len(result.skipped)} skipped (of {len(result.items)})",
            pages,
            f"  run time: {result.seconds:.1f}s",
        ]
        outputs = result.outputs
        if outputs:
            lines.append(f"  outputs ({len(outputs)}):")
            lines.extend(f"    {out}" for out in outputs[:MAX_LISTED_OUTPUTS])
            if len(outputs) > MAX_LISTED_OUTPUTS:
                lines.append(f"    ... and {len(outputs) - MAX_LISTED_OUTPUTS} more")
        for line in lines:
            _write(line)

    def incomplete(self, result: RunResult) -> None:
        """Name the items that finished incomplete, with their page counts."""
        incomplete = [item for item in result.items if not item.success]
        if not incomplete:
            return
        lines = [
            f"{len(incomplete)} item(s) finished INCOMPLETE. Their "
            "transcription and summary files may contain error placeholders:"
        ]
        for item in incomplete:
            report = item.report
            if report:
                detail = (
                    f"{report.get('pages_failed', 0)} failed, "
                    f"{report.get('pages_deferred', 0)} deferred page(s)"
                )
                if report.get("summary_failures", 0):
                    detail += f", {report['summary_failures']} summary failure(s)"
                lines.append(f"    - {item.name} ({detail})")
            else:
                lines.append(
                    f"    - {item.name} (one or more pages failed or were deferred)"
                )
        lines.append(
            "Re-running resumes and retries only the missing pages; see each "
            "item's log for the exact failed/deferred page counts."
        )
        for line in lines:
            _write(f"[WARN] {line}")
