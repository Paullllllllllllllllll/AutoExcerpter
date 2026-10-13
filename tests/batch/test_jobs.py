"""Batch jobs: rebuilt from the submitting run, kept while a part is open."""

from __future__ import annotations

import asyncio
from pathlib import Path

from autoexcerpter.batch.check import end_batch_jobs
from autoexcerpter.batch.jobs import load_jobs
from autoexcerpter.common.sqlite import read
from autoexcerpter.common.store import PlannedUnit, Store, UnitKey
from autoexcerpter.output.store import (
    KIND_TRANSCRIPTION,
    begin_run,
    open_store,
    plan_pages,
    read_batch_jobs,
    write_batch_job,
)
from autoexcerpter.settings import Settings
from autoexcerpter.spec import effective_spec, resolve_spec

MODEL = "gpt-5.6-terra"


async def _document(store: Store, spec_flags: dict[str, object]) -> int:
    spec = resolve_spec({"input": Path("doc.pdf"), **spec_flags}).spec
    await begin_run(store, "run", effective_spec(spec))
    doc_id = await store.ensure_document("doc", input_path="doc.pdf", kind="pdf")
    lease = await store.acquire(doc_id)
    assert lease is not None
    await plan_pages(
        store,
        lease,
        [PlannedUnit(idx, f"page_{idx + 1:04d}.jpg", idx) for idx in range(2)],
        transcription=(MODEL, "openai"),
        summary=(MODEL, "openai"),
    )
    await write_batch_job(store, lease, "Prices")
    return doc_id


async def _submitted(store: Store, doc_id: int, idx: int, batch_id: str) -> str:
    submission = await store.open_submission(
        provider="openai",
        model=MODEL,
        units=[UnitKey(doc_id, KIND_TRANSCRIPTION, idx)],
        part=idx,
    )
    await store.record_batch(submission.submission_id, batch_id)
    return submission.submission_id


def test_a_job_rebuilds_the_spec_and_formats_of_its_run(tmp_path: Path) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            doc_id = await _document(
                store, {"model": MODEL, "response_format": "text", "batch": True}
            )
            jobs = load_jobs(store, Settings())
            job = jobs[doc_id]
            assert job.context == "Prices"
            assert job.spec.run.batch
            assert job.transcription_format == "text"
            assert job.job.transcription.name == MODEL

    asyncio.run(main())


def test_ending_one_part_keeps_the_job_while_another_part_is_open(
    tmp_path: Path,
) -> None:
    async def main() -> None:
        async with open_store(tmp_path) as store:
            doc_id = await _document(store, {"model": MODEL, "batch": True})
            first = await _submitted(store, doc_id, 0, "batch_1")
            second = await _submitted(store, doc_id, 1, "batch_2")

            await store.mark_cancelled(first)
            await end_batch_jobs(store, first)
            assert [job.doc_id for job in read(store.path, read_batch_jobs)] == [doc_id]

            await store.mark_cancelled(second)
            await end_batch_jobs(store, second)
            assert read(store.path, read_batch_jobs) == []

    asyncio.run(main())
