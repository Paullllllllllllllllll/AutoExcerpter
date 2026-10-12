"""The public leased write, the document import and renderers that return bytes."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.common.store import (
    Component,
    DocumentImport,
    Lease,
    LeaseLostError,
    PlannedUnit,
    RenderTarget,
    ResponseRow,
    Store,
    UnitKey,
    UnitResult,
)
from tests.common.store_child import FakeClock

MINUTE = 60.0
EXTRA = Component(
    "tool",
    ("CREATE TABLE extra (doc_id INTEGER NOT NULL, text TEXT NOT NULL);",),
)
BINARY = b"PK\x03\x04\x00\xff\xfe binary body\r\n"


def _insert(text: str, lease: Lease) -> Callable[[sqlite3.Connection], int]:
    def func(connection: sqlite3.Connection) -> int:
        connection.execute(
            "INSERT INTO extra (doc_id, text) VALUES (?, ?)", (lease.doc_id, text)
        )
        (count,) = connection.execute("SELECT count(*) FROM extra").fetchone()
        return int(count)

    return func


async def _lease(store: Store) -> Lease:
    doc_id = await store.ensure_document("doc")
    lease = await store.acquire(doc_id)
    assert lease is not None
    return lease


def test_write_commits_the_function_under_the_lease(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    async def main() -> list[int]:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("run")
            lease = await _lease(store)
            first = await store.write(lease, _insert("one", lease))
            second = await store.write(lease, _insert("two", lease))
            return [first, second]

    assert asyncio.run(main()) == [1, 2]
    assert [row["text"] for row in read_rows(db, "extra")] == ["one", "two"]


def test_write_rolls_back_when_the_function_raises(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"

    def failing(connection: sqlite3.Connection) -> None:
        connection.execute("INSERT INTO extra (doc_id, text) VALUES (1, 'lost')")
        raise ValueError("refused")

    async def main() -> None:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("run")
            lease = await _lease(store)
            with pytest.raises(ValueError, match="refused"):
                await store.write(lease, failing)
            assert lease.doc_id in store.held_leases()

    asyncio.run(main())
    assert read_rows(db, "extra") == []


def test_write_refuses_an_expired_or_taken_lease(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    called: list[str] = []

    def func(connection: sqlite3.Connection) -> None:
        called.append("ran")
        connection.execute("INSERT INTO extra (doc_id, text) VALUES (1, 'late')")

    async def main() -> None:
        async with (
            Store(db, "tool", components=(EXTRA,), clock=clock) as first,
            Store(db, "tool", components=(EXTRA,), clock=clock) as second,
        ):
            await first.begin_run("first")
            await second.begin_run("second")
            lease = await _lease(first)
            clock.advance(11 * MINUTE)
            with pytest.raises(LeaseLostError, match="expired"):
                await first.write(lease, func)
            assert lease.doc_id not in first.held_leases()

            again = await first.acquire(lease.doc_id)
            assert again is not None
            clock.advance(11 * MINUTE)
            assert await second.acquire(lease.doc_id) is not None
            with pytest.raises(LeaseLostError, match="another holder"):
                await first.write(again, func)

    asyncio.run(main())
    assert called == []
    assert read_rows(db, "extra") == []


def test_a_unit_and_its_extra_row_commit_or_roll_back_together(
    tmp_path: Path,
) -> None:
    db = tmp_path / "tool.sqlite"

    def failing(connection: sqlite3.Connection) -> None:
        connection.execute("INSERT INTO extra (doc_id, text) VALUES (1, 'lost')")
        raise ValueError("refused")

    async def main() -> None:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("run")
            lease = await _lease(store)
            await store.plan_units(
                lease, "page", [PlannedUnit(0)], model="m", provider="p"
            )
            done = UnitResult("page", 0, "done", "m", "p", content="kept")
            await store.write_unit(lease, done, extra=_insert("row", lease))
            failed = UnitResult("page", 0, "done", "m", "p", content="lost")
            with pytest.raises(ValueError, match="refused"):
                await store.write_unit(lease, failed, extra=failing)

    asyncio.run(main())
    assert [row["text"] for row in read_rows(db, "extra")] == ["row"]
    assert [row["content"] for row in read_rows(db, "units")] == ["kept"]


_IGNORED = {"run_id", "first_run_id", "last_run_id", "lease_until"}


def _rows(db: Path) -> dict[str, list[dict[str, object]]]:
    tables = ("documents", "units", "responses", "outputs", "extra")
    return {
        table: [
            {
                key: value
                for key, value in row.items()
                if key not in _IGNORED and not key.endswith("_at")
            }
            for row in read_rows(db, table)
        ]
        for table in tables
    }


def test_an_import_writes_what_the_single_writes_write(tmp_path: Path) -> None:
    usage = {"input_tokens": 3}

    def response(lease: Lease) -> ResponseRow:
        unit = UnitKey(lease.doc_id, "page", 0)
        return ResponseRow("p", "m", unit, raw="body", raw_kind="legacy", usage=usage)

    def result(response_id: int) -> UnitResult:
        return UnitResult(
            "page", 0, "done", "m", "p", content="text", response_id=response_id
        )

    async def single(db: Path, output: Path) -> None:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("migrate")
            lease = await _lease(store)
            await store.set_image_settings(lease, {"dpi": 300})
            response_id = await store.append_response(response(lease))
            await store.write_unit(
                lease, result(response_id), extra=_insert("row", lease)
            )
            await store.write_unit(lease, UnitResult("page", 1, "pending", "m", "p"))
            await store.register_output(lease, output, format="txt")

    def batch_writes(batch: DocumentImport, output: Path) -> None:
        batch.set_image_settings({"dpi": 300})
        response_id = batch.append_response(response(batch.lease))
        batch.write_unit(result(response_id), extra=_insert("row", batch.lease))
        batch.write_unit(UnitResult("page", 1, "pending", "m", "p"))
        batch.register_output(output, format="txt")

    async def imported(db: Path, output: Path) -> None:
        async with Store(db, "tool", components=(EXTRA,)) as store:
            await store.begin_run("migrate")
            lease = await _lease(store)
            await store.import_document(
                lease, lambda batch: batch_writes(batch, output)
            )

    for name, write in (("single", single), ("import", imported)):
        (tmp_path / name).mkdir()
        output = tmp_path / name / "doc.txt"
        output.write_text("text\n", encoding="utf-8")
        asyncio.run(write(tmp_path / name / "tool.sqlite", output))

    expected = _rows(tmp_path / "single" / "tool.sqlite")
    assert expected["units"][0]["response_id"] == 1
    assert expected["documents"][0]["units_done"] == 1
    assert _rows(tmp_path / "import" / "tool.sqlite") == expected


def test_an_import_commits_all_or_nothing_under_the_lease(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    called: list[str] = []

    def failing(batch: DocumentImport) -> None:
        called.append("ran")
        batch.write_unit(UnitResult("page", 0, "done", "m", "p", content="lost"))
        raise ValueError("refused")

    async def main() -> None:
        async with Store(db, "tool", clock=clock) as store:
            await store.begin_run("migrate")
            lease = await _lease(store)
            with pytest.raises(ValueError, match="refused"):
                await store.import_document(lease, failing)
            clock.advance(11 * MINUTE)
            with pytest.raises(LeaseLostError, match="expired"):
                await store.import_document(lease, failing)

    asyncio.run(main())
    assert called == ["ran"]
    assert read_rows(db, "units") == []


def test_a_bytes_rendering_is_written_as_it_is(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    target = tmp_path / "out" / "doc.docx"
    text_target = tmp_path / "out" / "doc.txt"

    def binary(_connection: sqlite3.Connection, _doc_id: int) -> bytes:
        return BINARY

    def text(_connection: sqlite3.Connection, _doc_id: int) -> str:
        return "text\n"

    async def main() -> list[list[Path]]:
        async with Store(db, "tool") as store:
            await store.begin_run("run")
            lease = await _lease(store)
            spec = [
                RenderTarget(target, "docx", binary),
                RenderTarget(text_target, "txt", text),
            ]
            first = await store.render(lease, spec)
            second = await store.render(lease, spec)
            return [first.written, second.unchanged]

    written, unchanged = asyncio.run(main())

    assert written == [target, text_target]
    assert unchanged == [target, text_target]
    assert target.read_bytes() == BINARY
    assert text_target.read_bytes() == b"text\n"
    digests = {row["path"]: row["sha256"] for row in read_rows(db, "outputs")}
    assert digests["out/doc.docx"] == hashlib.sha256(BINARY).hexdigest()
