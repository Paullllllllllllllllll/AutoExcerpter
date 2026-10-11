"""AttemptRows: every provider attempt becomes one response row with its role."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autoexcerpter.common.capture import AttemptRecord, record_attempt, use_recorder
from autoexcerpter.common.store import PlannedUnit, UnitKey, unit_context
from autoexcerpter.output.attempts import AttemptRows
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    open_store,
    plan_pages,
    read_only_connection,
)


def _record(role: str, attempt: int, error: str | None = None) -> AttemptRecord:
    return AttemptRecord(
        provider="",
        model="",
        role=role,
        attempt=attempt,
        label="page_0001.jpg",
        request={"image": "data:image/jpeg;base64," + "A" * 2048},
        raw='{"id": "resp_1"}',
        raw_kind="http",
        error=error,
        http_status=500 if error else 200,
    )


def test_attempts_are_stored_per_unit_with_their_role(tmp_path: Path) -> None:
    async def main() -> tuple[list[int | None], AttemptRows, AttemptRows]:
        async with open_store(tmp_path) as store:
            await store.begin_run("run")
            doc_id = await store.ensure_document("doc", kind="pdf")
            lease = await store.acquire(doc_id)
            assert lease is not None
            await plan_pages(
                store,
                lease,
                [PlannedUnit(0, label="page_0001.jpg", source_index=0)],
                transcription=("gpt-5.6-luna", "openai"),
                summary=("claude-sonnet-5-5", "anthropic"),
            )
            ids: list[int | None] = []
            transcribe = AttemptRows(store, provider="openai", model="gpt-5.6-luna")
            with (
                use_recorder(transcribe),
                unit_context(UnitKey(doc_id, KIND_TRANSCRIPTION, 0)),
            ):
                ids.append(await record_attempt(_record("transcription", 1, "boom")))
                ids.append(await record_attempt(_record("transcription", 2)))
            summarize = AttemptRows(
                store, provider="anthropic", model="claude-sonnet-5-5"
            )
            with (
                use_recorder(summarize),
                unit_context(UnitKey(doc_id, KIND_SUMMARY, 0)),
            ):
                ids.append(await record_attempt(_record("summary", 1)))
            return ids, transcribe, summarize

    ids, transcribe, summarize = asyncio.run(main())
    with read_only_connection(tmp_path) as connection:
        rows = connection.execute(
            "SELECT response_id, kind, order_index, role, provider, model, attempt, "
            "http_status FROM page_attempts"
        ).fetchall()
        (request,) = connection.execute(
            "SELECT request FROM responses WHERE response_id = ?", (ids[0],)
        ).fetchone()

    assert rows == [
        (ids[0], "transcription", 0, "transcription", "openai", "gpt-5.6-luna", 1, 500),
        (ids[1], "transcription", 0, "transcription", "openai", "gpt-5.6-luna", 2, 200),
        (ids[2], "summary", 0, "summary", "anthropic", "claude-sonnet-5-5", 1, 200),
    ]
    assert "base64" in request and "A" * 2048 not in request
    selected = transcribe.selected()
    assert selected is not None and selected.response_id == ids[1]
    assert summarize.selected() is not None


def test_a_failed_append_is_kept_and_raised(tmp_path: Path) -> None:
    async def main() -> AttemptRows:
        async with open_store(tmp_path) as store:
            await store.begin_run("run")
            rows = AttemptRows(store)
        with pytest.raises(RuntimeError, match="not open"):
            await rows.record(_record("summary", 1))
        return rows

    rows = asyncio.run(main())

    assert len(rows.failures) == 1
    assert rows.selected() is None
