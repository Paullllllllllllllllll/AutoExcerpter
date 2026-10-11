"""Resume through ``plan`` and ``run``: every start state, and a lost lease.

Each case runs a two-page PDF once or not at all, changes the input, the
output folder or the flags, and runs again. The second run's plan must give
the item the expected resume state, and its run must call the offline
managers for exactly the expected pages and leave every unit done when the
item is processed. The last test lets an item's lease expire, after its first
page is stored, while its second page is still with the model, and once
lets another run take the document over; the late page is not stored and
the item fails without writing its files.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.output.store import DATABASE_NAME
from autoexcerpter.pipeline.paths import ItemPaths
from tests.conftest import make_settings, make_spec
from tests.recording_events import RecordingEvents

MakePdf = Callable[..., Path]
OTHER_RUN = "other-run"


@dataclass
class Offline:
    """Run *pdf* into *out* with managers that record the pages they serve.

    Pages in *fail* fail their transcription; *before* maps a page index to
    a coroutine function awaited before that page's transcription returns.
    """

    pdf: Path
    out: Path
    fail: set[int] = field(default_factory=set)
    before: dict[int, Callable[[], Any]] = field(default_factory=dict)
    transcribed: list[int] = field(default_factory=list)
    summarized: list[int] = field(default_factory=list)

    async def transcribe(self, payload: Any) -> dict[str, Any]:
        index = payload.number - 1
        self.transcribed.append(index)
        if index in self.before:
            await self.before[index]()
        result = {
            "image": payload.image_name,
            "sequence_number": payload.number,
            "transcription": f"Text of page {index}.",
            "processing_time": 0.01,
            "provider": "openai",
        }
        if index in self.fail:
            result["error"] = "refused"
        return result

    async def summarize(self, transcription: str, page_num: int) -> dict[str, Any]:
        match = re.search(r"page (\d+)", transcription)
        assert match is not None, transcription
        self.summarized.append(int(match.group(1)))
        return {
            "page": page_num,
            "page_information": {
                "page_number_integer": page_num,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["A point."],
            "references": None,
            "processing_time": 0.01,
            "provider": "openai",
        }

    def plan(self, **flags: Any) -> core.RunPlan:
        spec = make_spec(self.pdf, output=self.out, openalex=False, **flags)
        return core.plan(spec, make_settings())

    def run(
        self, fail: set[int] | None = None, **flags: Any
    ) -> tuple[core.RunPlan, core.RunResult, RecordingEvents]:
        self.fail = set(fail or ())
        self.transcribed.clear()
        self.summarized.clear()
        run_plan = self.plan(**flags)
        events = RecordingEvents()
        return run_plan, asyncio.run(core.run(run_plan, events)), events

    @property
    def db(self) -> Path:
        return self.out / DATABASE_NAME

    def statuses(self) -> dict[str, list[str]]:
        units = read_rows(self.db, "units", order_by=["kind", "idx"])
        found: dict[str, list[str]] = {}
        for unit in units:
            found.setdefault(str(unit["kind"]), []).append(str(unit["status"]))
        return found


@pytest.fixture
def offline(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    mock_image_processing: dict[str, Any],
) -> Offline:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    runner = Offline(make_pdf("doc.pdf", num_pages=2), tmp_path / "out")
    transcriber = MagicMock()
    transcriber.transcribe_payload = AsyncMock(side_effect=runner.transcribe)
    summarizer = MagicMock()
    summarizer.generate_summary = AsyncMock(side_effect=runner.summarize)
    monkeypatch.setattr(
        managers_module, "TranscriptionManager", MagicMock(return_value=transcriber)
    )
    monkeypatch.setattr(
        managers_module, "SummaryManager", MagicMock(return_value=summarizer)
    )
    return runner


def _take_over(db: Path, until: float) -> None:
    """Lease every document of *db* to another run until *until*."""
    connection = sqlite3.connect(db, timeout=30)
    with contextlib.closing(connection), connection:
        connection.execute(
            "UPDATE documents SET lease_run_id = ?, lease_until = ?, "
            "lease_generation = lease_generation + 1",
            (OTHER_RUN, until),
        )


# ============================================================================
# Start states
# ============================================================================
def _fresh(offline: Offline) -> None:
    return None


def _complete(offline: Offline) -> None:
    offline.run()


def _complete_without_summary_file(offline: Offline) -> None:
    offline.run()
    ItemPaths.for_item("doc", offline.out).summary_md.unlink()


def _page_failed(offline: Offline) -> None:
    offline.run(fail={1})


def _transcription_only(offline: Offline) -> None:
    offline.run(summarize=False)


def _input_changed(offline: Offline) -> None:
    offline.run()
    offline.pdf.write_bytes(offline.pdf.read_bytes() + b"\n%changed\n")


def _leased_elsewhere(offline: Offline) -> None:
    offline.run(fail={1})
    _take_over(offline.db, time.time() + 600)


def _legacy(offline: Offline) -> None:
    offline.out.mkdir()
    path = ItemPaths.for_item("doc", offline.out).transcription
    path.write_text("Version 1.0 output.\n", encoding="utf-8")


@dataclass(frozen=True)
class Case:
    """A start state, the second run's flags and what that run must do.

    *processed* says whether the second run processes the item, so that its
    units end done; *failed* whether it counts the item failed.
    """

    start: Callable[[Offline], None]
    flags: dict[str, Any]
    state: str
    transcribed: list[int]
    summarized: list[int]
    processed: bool = True
    failed: bool = False


BOTH = [0, 1]
CASES: dict[str, Case] = {
    "fresh": Case(_fresh, {}, "none", BOTH, BOTH),
    "complete": Case(_complete, {}, "complete", [], [], processed=False),
    "complete_file_missing": Case(
        _complete_without_summary_file, {}, "complete", [], [], processed=False
    ),
    "partial": Case(_page_failed, {}, "partial", [1], [1]),
    "transcription_only": Case(_transcription_only, {}, "transcription_only", [], BOTH),
    "force": Case(_complete, {"force": True}, "none", BOTH, BOTH),
    "retranscribe_partial": Case(
        _page_failed, {"retranscribe": True}, "partial", BOTH, BOTH
    ),
    "input_changed": Case(_input_changed, {}, "none", BOTH, BOTH),
    "model_changed": Case(
        _page_failed, {"transcription_model": "gpt-5-mini"}, "partial", [1], [1]
    ),
    "image_setting_changed": Case(
        _page_failed,
        {"jpeg_quality": 80},
        "refused",
        [],
        [],
        processed=False,
        failed=True,
    ),
    "image_setting_changed_retranscribe": Case(
        _page_failed, {"jpeg_quality": 80, "retranscribe": True}, "partial", BOTH, BOTH
    ),
    "blocked": Case(_leased_elsewhere, {}, "blocked", [], [], processed=False),
    "legacy": Case(_legacy, {}, "legacy", [], [], processed=False),
}


def _planned_state(run_plan: core.RunPlan) -> str:
    (planned,) = (*run_plan.to_process, *run_plan.not_processed, *run_plan.refused)
    return planned.state


@pytest.mark.parametrize("name", list(CASES))
def test_a_second_run_resumes_from_each_start_state(
    offline: Offline, name: str
) -> None:
    case = CASES[name]
    case.start(offline)
    before = offline.statuses() if offline.db.exists() else None

    run_plan, result, events = offline.run(**case.flags)

    assert _planned_state(run_plan) == case.state
    assert sorted(offline.transcribed) == case.transcribed
    assert sorted(offline.summarized) == case.summarized
    assert result.failed == int(case.failed)
    if case.processed:
        assert result.complete == 1
        assert offline.statuses() == {
            "summary": ["done"] * 2,
            "transcription": ["done"] * 2,
        }
    else:
        assert result.complete == 0
        assert (offline.statuses() if offline.db.exists() else None) == before
    paths = ItemPaths.for_item("doc", offline.out)
    if name == "complete_file_missing":
        assert paths.summary_md.is_file()
    if name == "model_changed":
        assert any("Resume with another model" in m for m in events.of("warning"))
    if name == "image_setting_changed":
        assert any(
            "Image settings changed: jpeg_quality" in m for m in events.of("warning")
        )
    if case.state in ("blocked", "legacy", "complete"):
        assert result.skipped == ("doc",)


# ============================================================================
# A lease that expires while a page is with the model
# ============================================================================
def _expire(db: Path) -> None:
    """End the lease of every document of *db* now, as a missed renewal does."""
    connection = sqlite3.connect(db, timeout=30)
    with contextlib.closing(connection), connection:
        connection.execute("UPDATE documents SET lease_until = ?", (time.time(),))


@pytest.mark.parametrize("taken_over", [False, True], ids=["expired", "taken_over"])
def test_a_lease_holder_outliving_its_expiry_stores_no_later_unit(
    offline: Offline, taken_over: bool
) -> None:
    async def outlive_the_lease() -> None:
        deadline = time.monotonic() + 10
        while offline.statuses().get("transcription", [""])[0] != "done":
            assert time.monotonic() < deadline, "page 0 was never stored"
            await asyncio.sleep(0.02)
        _expire(offline.db)
        if taken_over:
            _take_over(offline.db, time.time() + 600)

    offline.before[1] = outlive_the_lease

    _, result, events = offline.run(summarize=False)

    assert offline.transcribed == [0, 1]
    assert result.failed == 1
    assert result.items[0].outputs == ()
    assert not ItemPaths.for_item("doc", offline.out).transcription.exists()
    assert offline.statuses() == {"transcription": ["done", "pending"]}
    (document,) = read_rows(offline.db, "documents")
    holder = OTHER_RUN if taken_over else None
    assert document["lease_run_id"] == holder
    assert any("lease was lost" in m for m in events.of("warning"))
