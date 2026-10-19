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
import json
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.item as item_module
import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.output.store import DATABASE_NAME
from autoexcerpter.pipeline.item import ItemFiles, ItemProcessor
from autoexcerpter.pipeline.outputs import OutputStatus
from autoexcerpter.pipeline.units import DocumentHandle
from autoexcerpter.rendering.citations import OpenAlexClient
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
    references: list[dict[str, Any]] | None = None

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
            "references": self.references,
            "processing_time": 0.01,
            "provider": "openai",
        }

    def plan(self, **flags: Any) -> core.RunPlan:
        spec = make_spec(self.pdf, output=self.out, **{"openalex": False, **flags})
        return core.plan(spec, make_settings(state_dir=self.out.parent / "state"))

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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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


def _unstored(offline: Offline) -> None:
    offline.out.mkdir()
    path = ItemPaths.for_item("doc", offline.out).transcription
    path.write_text("A file without stored units.\n", encoding="utf-8")


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
# A model whose family resolves its image settings from another section.
ANTHROPIC = {
    "transcription_provider": "anthropic",
    "transcription_model": "claude-haiku-5-5",
}
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
    "model_family_changed": Case(_page_failed, ANTHROPIC, "partial", [1], [1]),
    "model_family_and_image_setting_changed": Case(
        _page_failed,
        {**ANTHROPIC, "jpeg_quality": 80},
        "refused",
        [],
        [],
        processed=False,
        failed=True,
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
    "unstored": Case(_unstored, {}, "unstored", [], [], processed=False),
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
    if name in ("model_changed", "model_family_changed"):
        assert any("Resume with another model" in m for m in events.of("warning"))
    if name in ("image_setting_changed", "model_family_and_image_setting_changed"):
        assert any(
            "Image settings changed: jpeg_quality" in m for m in events.of("warning")
        )
    if case.state in ("blocked", "unstored", "complete"):
        assert result.skipped == ("doc",)
    if case.state == "blocked":
        assert result.blocked == result.leased == ("doc",)
        (reason,) = result.blocked_reasons
        assert "leased by run other-run until " in reason
        assert result.exit_code == 1


# ============================================================================
# Interruptions between a run's writes
# ============================================================================
async def _interrupted(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("interrupted")


def _interrupt(
    offline: Offline, monkeypatch: pytest.MonkeyPatch, owner: Any, name: str
) -> None:
    """Run once with *owner.name* raising, as an interruption there does."""
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, _interrupted)
    _, result, _ = offline.run()
    monkeypatch.setattr(owner, name, original)
    assert result.failed == 1


def test_a_repair_interrupted_before_its_render_renders_its_files_again(
    offline: Offline, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline.run(fail={1})
    transcription = ItemPaths.for_item("doc", offline.out).transcription
    with_failure = transcription.read_bytes()
    _interrupt(offline, monkeypatch, ItemProcessor, "_write_outputs")
    assert transcription.read_bytes() == with_failure

    run_plan, result, _ = offline.run()

    assert [planned.name for planned in run_plan.render_only] == ["doc"]
    assert result.failed == 0
    assert transcription.read_bytes() != with_failure


def test_an_input_change_interrupted_before_its_reset_is_detected_again(
    offline: Offline, monkeypatch: pytest.MonkeyPatch
) -> None:
    _input_changed(offline)
    _interrupt(offline, monkeypatch, DocumentHandle, "reset")

    run_plan, _, _ = offline.run()

    assert _planned_state(run_plan) == "none"
    assert sorted(offline.transcribed) == BOTH


def _no_lookup(self: OpenAlexClient, citation_text: str) -> Any:
    raise AssertionError(f"OpenAlex was asked about {citation_text!r}")


def test_citations_interrupted_in_their_enrichment_are_stored_by_the_render(
    offline: Offline, monkeypatch: pytest.MonkeyPatch
) -> None:
    citation = "Smith, J. (2020). A Cited Work. Journal of Tests, 1, 1-2."
    offline.references = [{"citation": citation, "is_partial": False}]
    _interrupt(offline, monkeypatch, ItemFiles, "store_citations")
    monkeypatch.setattr(OpenAlexClient, "lookup", _no_lookup)

    run_plan, result, _ = offline.run(openalex=True)

    assert [planned.name for planned in run_plan.render_only] == ["doc"]
    assert result.failed == 0
    assert [row["text"] for row in read_rows(offline.db, "citations")] == [citation]
    summary = ItemPaths.for_item("doc", offline.out).summary_md
    assert citation in summary.read_text(encoding="utf-8")


def test_a_render_only_item_with_citations_leaves_them(
    offline: Offline, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline.references = [{"citation": "Smith (2020).", "is_partial": False}]
    offline.run()
    ItemPaths.for_item("doc", offline.out).summary_md.unlink()
    consolidated: list[str] = []
    monkeypatch.setattr(
        item_module,
        "collect_citations",
        lambda *args, **kwargs: consolidated.append("called"),
    )
    monkeypatch.setattr(OpenAlexClient, "lookup", _no_lookup)

    _, result, _ = offline.run(openalex=True)

    assert result.failed == 0
    assert consolidated == []
    assert ItemPaths.for_item("doc", offline.out).summary_md.is_file()


def test_a_forced_run_without_summaries_succeeds_on_a_summarized_item(
    offline: Offline,
) -> None:
    offline.run()

    _, result, _ = offline.run(force=True, summarize=False)

    assert (result.complete, result.failed) == (1, 0)


def test_a_file_the_render_cannot_write_fails_the_run(
    offline: Offline, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline.run()
    ItemPaths.for_item("doc", offline.out).summary_md.unlink()

    async def unwritable(self: ItemFiles) -> OutputStatus:
        return OutputStatus(summaries_ok=False)

    monkeypatch.setattr(ItemFiles, "write", unwritable)

    _, result, _ = offline.run()

    assert result.failed == 1
    assert result.items[0].name == "doc"
    assert result.skipped == ()


def test_a_file_rendered_again_is_listed_with_the_outputs(offline: Offline) -> None:
    offline.run()
    paths = ItemPaths.for_item("doc", offline.out)
    paths.summary_md.unlink()

    _, result, _ = offline.run()

    assert (result.items, result.skipped, result.total) == ((), ("doc",), 1)
    assert str(paths.summary_md.resolve()) in result.rendered
    assert result.outputs[: len(result.rendered)] == list(result.rendered)
    assert result.outputs[-1] == str(offline.db.resolve())


def test_a_lease_taken_after_planning_blocks_the_item_and_exits_1(
    offline: Offline,
) -> None:
    offline.run(fail={1})
    offline.transcribed.clear()
    run_plan = offline.plan()
    _take_over(offline.db, time.time() + 600)

    result = asyncio.run(core.run(run_plan, RecordingEvents()))

    assert [planned.name for planned in run_plan.to_process] == ["doc"]
    assert offline.transcribed == []
    assert (result.items, result.skipped) == ((), ("doc",))
    assert result.blocked == result.leased == ("doc",)
    (reason,) = result.blocked_reasons
    assert "leased by run other-run until " in reason
    assert result.exit_code == 1


def test_a_complete_item_leased_before_its_render_counts_as_blocked(
    offline: Offline,
) -> None:
    offline.run()
    summary = ItemPaths.for_item("doc", offline.out).summary_md
    summary.unlink()
    run_plan = offline.plan()
    _take_over(offline.db, time.time() + 600)

    result = asyncio.run(core.run(run_plan))

    assert [planned.name for planned in run_plan.render_only] == ["doc"]
    assert (result.items, result.skipped) == ((), ("doc",))
    assert result.blocked == result.leased == ("doc",)
    assert result.exit_code == 1
    assert not summary.exists()


# ============================================================================
# Page selection
# ============================================================================
LONG_PAGES = 20


@pytest.fixture
def long_pdf(offline: Offline, make_pdf: MakePdf) -> Offline:
    """The offline runner on a PDF of LONG_PAGES pages, document ``long``."""
    offline.pdf = make_pdf("long.pdf", num_pages=LONG_PAGES)
    return offline


def _done(offline: Offline, kind: str = "transcription") -> list[int]:
    units = read_rows(offline.db, "units", order_by=["idx"])
    return [
        int(unit["idx"])
        for unit in units
        if unit["kind"] == kind and unit["status"] == "done"
    ]


def _rendered_pages(offline: Offline) -> list[int]:
    """Return the pages of the transcription file, in the order it holds them."""
    text = ItemPaths.for_item("long", offline.out).transcription.read_text(
        encoding="utf-8"
    )
    return [int(page) for page in re.findall(r"Text of page (\d+)\.", text)]


def test_pages_work_only_the_selected_pages_and_render_them_in_order(
    long_pdf: Offline,
) -> None:
    _, result, _ = long_pdf.run(pages="7,2,15-17")

    selected = [1, 6, 14, 15, 16]
    assert sorted(long_pdf.transcribed) == selected
    assert sorted(long_pdf.summarized) == selected
    assert (result.complete, result.partial, result.failed) == (0, 1, 0)
    assert _done(long_pdf) == selected
    assert _done(long_pdf, "summary") == selected
    rows = read_rows(long_pdf.db, "summaries", order_by=["idx"])
    assert [row["idx"] for row in rows] == selected
    assert _rendered_pages(long_pdf) == selected
    assert ItemPaths.for_item("long", long_pdf.out).summary_md.is_file()
    assert _planned_state(long_pdf.plan()) == "partial"


def test_later_page_ranges_add_pages_and_a_run_without_pages_completes(
    long_pdf: Offline,
) -> None:
    long_pdf.run(pages="3")
    assert sorted(long_pdf.transcribed) == [0, 1, 2]

    long_pdf.run(pages="last:3")
    assert sorted(long_pdf.transcribed) == [17, 18, 19]
    assert _rendered_pages(long_pdf) == [0, 1, 2, 17, 18, 19]

    run_plan, result, _ = long_pdf.run()

    assert _planned_state(run_plan) == "partial"
    assert sorted(long_pdf.transcribed) == list(range(3, 17))
    assert sorted(long_pdf.summarized) == list(range(3, 17))
    assert (result.complete, result.failed) == (1, 0)
    assert _rendered_pages(long_pdf) == list(range(LONG_PAGES))
    assert _planned_state(long_pdf.plan()) == "complete"


def test_a_range_beyond_the_last_page_skips_the_item_without_a_call(
    long_pdf: Offline,
) -> None:
    long_pdf.run(pages="1-2")
    before = long_pdf.statuses()

    _, result, events = long_pdf.run(pages="99-", force=True)

    assert long_pdf.transcribed == long_pdf.summarized == []
    assert result.skipped == ("long",)
    assert (result.complete, result.failed) == (0, 0)
    assert long_pdf.statuses() == before
    assert (
        "Page range 'pages 99-' yielded no pages for 'long' (20 pages). Skipping."
        in events.of("warning")
    )


def test_a_failed_page_outside_the_range_keeps_the_item_from_counting_complete(
    long_pdf: Offline,
) -> None:
    long_pdf.run(fail={5})

    _, result, _ = long_pdf.run(pages="1-2")

    assert long_pdf.transcribed == []
    assert (result.complete, result.partial, result.failed) == (0, 1, 0)


def test_a_range_beyond_the_last_page_of_a_complete_item_warns(
    long_pdf: Offline,
) -> None:
    long_pdf.run()

    run_plan, result, _ = long_pdf.run(pages="99-")

    assert long_pdf.transcribed == long_pdf.summarized == []
    assert result.skipped == ("long",)
    assert (
        "Page range 'pages 99-' yielded no pages for 'long' (20 pages), which is "
        "complete. Skipping." in run_plan.warnings
    )


def test_force_with_pages_resets_every_stored_page_of_the_item(
    long_pdf: Offline,
) -> None:
    long_pdf.run(pages="1-5")

    _, result, _ = long_pdf.run(pages="4-4", force=True)

    assert long_pdf.transcribed == [3]
    assert (result.complete, result.partial, result.failed) == (0, 1, 0)
    assert _done(long_pdf) == [3]
    assert _done(long_pdf, "summary") == [3]
    assert _rendered_pages(long_pdf) == [3]


def test_retranscribe_with_pages_resets_only_the_selected_pages(
    long_pdf: Offline,
) -> None:
    long_pdf.run(pages="1-5")

    _, result, _ = long_pdf.run(pages="4-4", retranscribe=True)

    assert long_pdf.transcribed == long_pdf.summarized == [3]
    assert (result.complete, result.partial, result.failed) == (0, 1, 0)
    assert _done(long_pdf) == _done(long_pdf, "summary") == [0, 1, 2, 3, 4]
    summaries = read_rows(long_pdf.db, "units", order_by=["idx"])
    assert [u["stale"] for u in summaries if u["kind"] == "summary"] == [0] * 20
    assert _rendered_pages(long_pdf) == [0, 1, 2, 3, 4]


def _image_settings(offline: Offline) -> dict[str, Any]:
    (document,) = read_rows(offline.db, "documents")
    value = document["image_settings"]
    return dict(json.loads(value) if isinstance(value, str) else value)


def test_retranscribe_with_pages_keeps_the_stored_image_settings(
    long_pdf: Offline, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_pdf.run(pages="1-5")
    stored = _image_settings(long_pdf)
    qualities: list[Any] = []
    real = item_module.open_source

    def spy(item: Any, job: Any) -> Any:
        qualities.append(job.images.jpeg_quality)
        return real(item, job)

    monkeypatch.setattr(item_module, "open_source", spy)

    run_plan, result, events = long_pdf.run(
        pages="2-2", retranscribe=True, jpeg_quality=80
    )

    assert _planned_state(run_plan) == "partial"
    assert long_pdf.transcribed == [1]
    assert (result.complete, result.partial, result.failed) == (0, 1, 0)
    assert qualities == [stored["jpeg_quality"]] != [80]
    assert _image_settings(long_pdf) == stored
    assert _done(long_pdf) == [0, 1, 2, 3, 4]
    assert any(
        "keeps the stored image settings (jpeg_quality)" in m
        for m in events.of("warning")
    )


@pytest.mark.parametrize("summarized", ["", "1-3"], ids=["no_summaries", "some"])
def test_retranscribe_with_pages_keeps_the_image_settings_of_a_transcribed_item(
    long_pdf: Offline, monkeypatch: pytest.MonkeyPatch, summarized: str
) -> None:
    long_pdf.run(summarize=False)
    if summarized:
        long_pdf.run(pages=summarized)
    stored = _image_settings(long_pdf)
    qualities: list[Any] = []
    real = item_module.open_source

    def spy(item: Any, job: Any) -> Any:
        qualities.append(job.images.jpeg_quality)
        return real(item, job)

    monkeypatch.setattr(item_module, "open_source", spy)

    run_plan, result, _ = long_pdf.run(pages="2-2", retranscribe=True, jpeg_quality=80)

    assert _planned_state(run_plan) == "transcription_only"
    assert long_pdf.transcribed == long_pdf.summarized == [1]
    assert (result.complete, result.partial, result.failed) == (0, 1, 0)
    assert qualities == [stored["jpeg_quality"]] != [80]
    assert _image_settings(long_pdf) == stored
    assert _done(long_pdf) == list(range(LONG_PAGES))


def _edit_image_record(db: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Apply *edit* to every stored image record."""
    connection = sqlite3.connect(db, timeout=30)
    with contextlib.closing(connection), connection:
        for doc_id, value in connection.execute(
            "SELECT doc_id, image_settings FROM documents"
        ).fetchall():
            record = json.loads(value)
            edit(record)
            connection.execute(
                "UPDATE documents SET image_settings = ? WHERE doc_id = ?",
                (json.dumps(record), doc_id),
            )


def _stored_with_low_detail(record: dict[str, Any]) -> None:
    """The record of pages sent with --image-detail low (the test sections
    ignore the run's detail, so the stored side carries the difference)."""
    record.update(llm_detail="low", request_detail="low", resolved_detail="low")


def test_retranscribe_with_pages_refuses_image_settings_it_cannot_keep(
    long_pdf: Offline,
) -> None:
    long_pdf.run(summarize=False)
    _edit_image_record(long_pdf.db, _stored_with_low_detail)
    stored = _image_settings(long_pdf)

    run_plan, result, events = long_pdf.run(pages="2-2", retranscribe=True)

    assert _planned_state(run_plan) == "refused"
    assert long_pdf.transcribed == long_pdf.summarized == []
    assert (result.complete, result.failed) == (0, 1)
    assert _image_settings(long_pdf) == stored
    assert _done(long_pdf) == list(range(LONG_PAGES))
    (warning,) = [m for m in events.of("warning") if "Image settings changed" in m]
    assert "--retranscribe without --pages" in warning


def test_retranscribe_with_pages_refuses_a_document_without_an_image_record(
    long_pdf: Offline,
) -> None:
    long_pdf.run(summarize=False)
    connection = sqlite3.connect(long_pdf.db, timeout=30)
    with contextlib.closing(connection), connection:
        connection.execute("UPDATE documents SET image_settings = NULL")

    run_plan, result, events = long_pdf.run(pages="2-2", retranscribe=True)

    assert _planned_state(run_plan) == "refused"
    assert long_pdf.transcribed == long_pdf.summarized == []
    assert (result.complete, result.failed) == (0, 1)
    (document,) = read_rows(long_pdf.db, "documents")
    assert document["image_settings"] is None
    (warning,) = [m for m in events.of("warning") if "are unknown" in m]
    assert "--retranscribe without --pages" in warning


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

    assert sorted(offline.transcribed) == [0, 1]
    assert result.failed == 1
    assert result.items[0].outputs == ()
    assert not ItemPaths.for_item("doc", offline.out).transcription.exists()
    assert offline.statuses() == {"transcription": ["done", "pending"]}
    (document,) = read_rows(offline.db, "documents")
    holder = OTHER_RUN if taken_over else None
    assert document["lease_run_id"] == holder
    assert any("lease was lost" in m for m in events.of("warning"))
