"""The summary work of a batch document: requests, placeholders, waits."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from autoexcerpter.batch.work import (
    needs_summary,
    summary_work,
    write_placeholders,
)
from autoexcerpter.common.store import PlannedUnit, Store, UnitResult
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    open_store,
    plan_pages,
)

MODEL = "gpt-5.6-terra"
TEXTS = {
    0: ("done", "## Page one"),
    1: ("failed", "[transcription error: boom]"),
    2: ("done", "[page_0003.jpg: no transcribable text]"),
    3: ("pending", None),
    4: ("done", ""),
}


async def _seed(store: Store) -> int:
    await store.begin_run("run")
    doc_id = await store.ensure_document("doc", input_path="doc.pdf", kind="pdf")
    lease = await store.acquire(doc_id)
    assert lease is not None
    pages = [PlannedUnit(idx, f"page_{idx + 1:04d}.jpg", idx) for idx in TEXTS]
    await plan_pages(
        store,
        lease,
        pages,
        transcription=(MODEL, "openai"),
        summary=(MODEL, "openai"),
    )
    for idx, (status, text) in TEXTS.items():
        if status == "pending":
            continue
        await store.write_unit(
            lease,
            UnitResult(
                kind=KIND_TRANSCRIPTION,
                idx=idx,
                status=status,
                model=MODEL,
                provider="openai",
                content=text,
                error="boom" if status == "failed" else None,
            ),
        )
    return doc_id


def test_the_work_holds_requests_placeholders_and_waits(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            doc_id = await _seed(store)
            work = summary_work(store, doc_id)
            assert [r.unit.idx for r in work.requests] == [0]
            assert work.requests[0].request.text == "## Page one"
            assert work.requests[0].request.custom_id == "summary-0"
            assert {p.unit.idx: p.outcome for p in work.placeholders} == {
                1: "placeholder",
                2: "blank",
                4: "blank",
            }
            failed = next(p for p in work.placeholders if p.unit.idx == 1)
            assert failed.entry["bullet_points"] == ["[Transcription failed: boom]"]

            waiting = summary_work(store, doc_id, waiting={0})
            assert waiting.requests == []
            none = summary_work(store, doc_id, selected=lambda unit: False)
            assert not none

            lease = store.held_leases()[doc_id]
            await write_placeholders(
                store, lease, work.placeholders, model=MODEL, provider="openai"
            )
            summaries = {u.idx: u for u in store.units(doc_id, kind=KIND_SUMMARY)}
            assert {idx: summaries[idx].status for idx in (1, 2, 4)} == dict.fromkeys(
                (1, 2, 4), "done"
            )
            assert summaries[1].outcome == "placeholder"
            assert not needs_summary(summaries[2])
            assert needs_summary(summaries[0])
            after = summary_work(store, doc_id)
            assert [r.unit.idx for r in after.requests] == [0]
            assert after.placeholders == []

    asyncio.run(main())
    with sqlite3.connect(tmp_path / "autoexcerpter.sqlite") as db:
        rows = db.execute("SELECT idx, page FROM summaries ORDER BY idx").fetchall()
    assert rows == [(1, 2), (2, 3), (4, 5)]
