"""Run screen: progress events, summary table and plan table."""

from __future__ import annotations

import dataclasses
import io
import re
import sys
from collections.abc import Callable
from pathlib import Path

from rich.console import Console, RenderableType

import autoexcerpter.run as core
from autoexcerpter.common.usage import Usage
from autoexcerpter.events import (
    ItemFinished,
    ItemStarted,
    PageDone,
    RunEvents,
    Waiting,
)
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.resume import ProcessingState, ResumeResult
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.ui.progress import RichProgress, plan_table, summary_table
from tests.conftest import make_settings, make_spec

ESC = "\x1b"


def _console() -> Console:
    return Console(
        file=io.StringIO(), width=100, color_system=None, force_terminal=False
    )


def _output(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def _render(renderable: RenderableType) -> str:
    console = _console()
    console.print(renderable)
    return _output(console)


def _report(
    ok: int, failed: int = 0, deferred: int = 0, summary: int = 0
) -> dict[str, int]:
    return {
        "pages_ok": ok,
        "pages_failed": failed,
        "pages_deferred": deferred,
        "summary_failures": summary,
        "pages_attempted": ok + failed,
    }


def _drive(events: RunEvents) -> None:
    events.item_started(ItemStarted("alpha", "pdf", 1, 2))
    events.page_done(PageDone("alpha", 2, "ok", 1, 2, already_complete=2))
    events.waiting(Waiting("rate limit", 30.0))
    events.page_done(PageDone("alpha", 3, "ok", 2, 2, already_complete=2))
    events.item_finished(
        ItemFinished("alpha", 1, 2, True, ("/out/alpha.md",), _report(4))
    )
    events.item_started(ItemStarted("beta", "image_folder", 2, 2))
    events.warning("model refused page 1\nretrying later")
    events.page_done(PageDone("beta", 0, "failed", 1, 1))
    events.item_finished(
        ItemFinished("beta", 2, 2, False, (), _report(0, failed=1, summary=1))
    )


def test_non_terminal_console_prints_plain_lines() -> None:
    console = _console()
    with RichProgress(console) as progress:
        assert not progress.live
        _drive(progress)
    output = _output(console)
    assert ESC not in output
    lines = output.splitlines()
    assert "[1/2] alpha (PDF)" in lines
    assert "[2/2] beta (image folder)" in lines
    assert "Waiting 30s: rate limit" in lines
    assert "Warning: model refused page 1" in lines
    assert "Warning: retrying later" in lines
    assert "[1/2] alpha: complete; 4 page(s) ok, 0 failed, 0 deferred" in lines
    assert "    outputs: /out/alpha.md" in lines
    assert (
        "[2/2] beta: incomplete; 0 page(s) ok, 1 failed, 0 deferred, "
        "1 summary failure(s)"
    ) in lines


def test_item_without_report_prints_its_verdict() -> None:
    console = _console()
    with RichProgress(console) as progress:
        progress.item_started(ItemStarted("gamma", "pdf", 1, 1))
        progress.item_finished(ItemFinished("gamma", 1, 1, False))
    assert "[1/1] gamma: incomplete" in _output(console).splitlines()


def test_terminal_console_shows_bars_and_redirects_stderr() -> None:
    console = Console(
        file=io.StringIO(), width=100, force_terminal=True, color_system=None
    )
    with RichProgress(console) as progress:
        assert progress.live
        progress.item_started(ItemStarted("alpha", "pdf", 1, 1))
        progress.page_done(PageDone("alpha", 0, "ok", 1, 3, already_complete=2))
        progress.waiting(Waiting("rate limit", 12.0))
        assert "Waiting 12s: rate limit" in _render(progress._render())
        assert "+2 from an earlier run" in _render(progress._render())
        sys.stderr.write("a log record\n")
        progress.item_finished(ItemFinished("alpha", 1, 1, True, (), _report(3)))
    assert not progress.live
    output = _output(console)
    assert "a log record" in output
    assert "alpha: complete; 3 page(s) ok" in output
    assert "Items" in output


def _result() -> core.RunResult:
    paths = ItemPaths.for_item("x", Path("out"), "md")
    return core.RunResult(
        items=(
            core.ItemResult("alpha", True, paths, report=_report(4)),
            core.ItemResult("beta", False, paths, report=_report(2, 1, 3, 1)),
            core.ItemResult("gamma", False, paths),
        ),
        skipped=("delta",),
        seconds=12.34,
        usage={
            "transcription": Usage(1200, 300, 1500, 100, 0),
            "summary": Usage(800, 200, 1000, 0, 50),
        },
        databases=("/out/autoexcerpter.sqlite",),
    )


def _row(output: str, first: str) -> list[str]:
    for line in output.splitlines():
        cells = [cell.strip() for cell in re.split(r"[│┃]", line.strip("│┃ "))]
        if cells and cells[0] == first:
            return cells
    raise AssertionError(f"no row starting with {first!r} in:\n{output}")


def test_summary_table_lists_items_totals_and_usage() -> None:
    output = _render(summary_table(_result()))
    assert ESC not in output
    assert _row(output, "alpha") == ["alpha", "complete", "4", "0", "0", "0"]
    assert _row(output, "beta") == ["beta", "incomplete", "2", "1", "3", "1"]
    assert _row(output, "gamma") == ["gamma", "incomplete", "0", "0", "0", "0"]
    assert "1 complete, 2 incomplete" in output
    assert "Run time: 12.3s" in output
    assert "Skipped (already complete): delta" in output
    assert "Databases written: /out/autoexcerpter.sqlite" in output
    assert _row(output, "transcription") == [
        "transcription",
        "1,200",
        "300",
        "100",
        "0",
        "1,500",
    ]
    assert _row(output, "summary")[-1] == "1,000"
    totals = [
        line for line in output.splitlines() if line.strip("│ ").startswith("Total")
    ]
    assert len(totals) == 2
    assert "2,000" in totals[1] and "2,500" in totals[1]


def test_summary_table_without_usage_or_skips() -> None:
    output = _render(summary_table(core.RunResult(items=())))
    assert "Token usage: none reported" in output
    assert "Skipped (already complete): none" in output


def test_plan_table_shows_resume_state_and_skips(
    tmp_path: Path, make_pdf: Callable[..., Path]
) -> None:
    make_pdf("alpha.pdf", num_pages=4)
    make_pdf("beta.pdf", num_pages=1)
    run_plan = core.plan(make_spec(tmp_path, all=True), make_settings())
    alpha, beta = run_plan.to_process
    partial = dataclasses.replace(
        alpha,
        resume=ResumeResult(
            "alpha", ProcessingState.PARTIAL, completed_page_indices={0, 1}
        ),
    )
    skipped = dataclasses.replace(
        beta, resume=ResumeResult("beta", ProcessingState.COMPLETE)
    )
    gamma = core.PlannedItem(
        ItemSpec("image_folder", tmp_path / "gamma"),
        tmp_path,
        ResumeResult("gamma", ProcessingState.NONE),
    )
    run_plan = dataclasses.replace(
        run_plan, to_process=(partial, gamma), skipped=(skipped,)
    )
    output = _render(plan_table(run_plan))
    assert ESC not in output
    assert "Plan: 2 to process, 1 already complete" in output
    assert _row(output, "1") == ["1", "alpha", "PDF", "partial", "2"]
    assert _row(output, "2") == ["2", "gamma", "image folder", "none", "0"]
    assert _row(output, "-") == ["-", "beta", "PDF", "complete (skip)", "-"]
