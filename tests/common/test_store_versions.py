"""Schema versions, refusals and the conversion of versionless databases.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_versions.py -q

The interrupted conversion starts ``tests/common/store_child.py`` with the
test's interpreter and the repository root as working directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from autoexcerpter.common.sqlite import read, read_rows, table_names
from autoexcerpter.common.store import (
    CORE_VERSION,
    Component,
    Converter,
    NewerSchemaError,
    Store,
    StoreError,
    UnreadableDatabaseError,
    UnrecognizedSchemaError,
    VersionlessDatabaseError,
    check_schema,
    convert_versionless,
    inspect_schema,
)
from tests.common.store_child import (
    P2_CONVERTER,
    build_ae_v1,
    build_p2,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD = "tests/common/store_child.py"
CHILD_TIMEOUT = 120.0


def _open(db: Path, components: tuple[Component, ...] = ()) -> None:
    async def main() -> None:
        async with Store(db, "tool", components=components):
            pass

    asyncio.run(main())


def _counts(db: Path, *tables: str) -> tuple[int, ...]:
    return read(
        db,
        lambda c: tuple(
            int(c.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in tables
        ),
    )


def _integrity(db: Path) -> str:
    return str(read(db, lambda c: c.execute("PRAGMA integrity_check").fetchone()[0]))


# =============================================================================
# Refusals
# =============================================================================


@pytest.mark.parametrize("build", [build_p2, build_ae_v1], ids=["ct-p2", "ae-v1"])
def test_a_versionless_database_is_refused_and_left_unchanged(
    tmp_path: Path, build: object
) -> None:
    db = tmp_path / "old.sqlite"
    build(db)  # type: ignore[operator]
    before = db.read_bytes()

    with pytest.raises(VersionlessDatabaseError) as raised:
        _open(db)

    assert raised.value.exit_code == 2
    assert isinstance(raised.value, StoreError)
    assert "migrate" in str(raised.value)
    assert db.read_bytes() == before
    assert not db.with_name(db.name + "-journal").exists()
    with pytest.raises(VersionlessDatabaseError):
        check_schema(db)


def test_a_newer_core_schema_is_refused_and_left_unchanged(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    _open(db)
    with contextlib.closing(sqlite3.connect(db)) as connection:
        connection.execute(
            "UPDATE meta SET version = ? WHERE component = 'core'",
            (CORE_VERSION + 1,),
        )
        connection.commit()
    before = db.read_bytes()

    with pytest.raises(NewerSchemaError) as raised:
        _open(db)

    assert raised.value.exit_code == 2
    assert db.read_bytes() == before


def test_a_newer_tool_component_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    two = Component("tool", ("CREATE TABLE a (x)", "CREATE TABLE b (x)"))
    one = Component("tool", ("CREATE TABLE a (x)",))
    _open(db, (two,))

    with pytest.raises(NewerSchemaError, match="'tool' version 2"):
        _open(db, (one,))


def test_a_file_that_is_not_a_database_is_refused_and_kept(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    content = b"This is a text file, not a database.\n" * 200
    db.write_bytes(content)

    with pytest.raises(UnreadableDatabaseError) as raised:
        _open(db)

    assert raised.value.exit_code == 2
    assert db.read_bytes() == content
    with pytest.raises(UnreadableDatabaseError):
        check_schema(db)


def test_only_writers_create_a_missing_database(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "tool.sqlite"

    state = check_schema(db)

    assert not state.exists
    assert not db.exists()

    async def without_create() -> None:
        await Store(db, "tool").open(create=False)

    with pytest.raises(FileNotFoundError):
        asyncio.run(without_create())
    assert not db.exists()
    _open(db)
    assert check_schema(db).versions == {"core": CORE_VERSION}


# =============================================================================
# Tool component migrations
# =============================================================================

V1 = "CREATE TABLE extras (id INTEGER PRIMARY KEY, note TEXT);"
V2 = """
ALTER TABLE extras ADD COLUMN detail TEXT;
CREATE TABLE more (id INTEGER PRIMARY KEY);
CREATE TRIGGER extras_touch AFTER INSERT ON extras
BEGIN
    INSERT INTO more (id) VALUES (NEW.id);
END;
"""


def _seed_v1(db: Path) -> None:
    async def main() -> None:
        async with Store(db, "tool", components=(Component("tool", (V1,)),)) as store:
            await store._write(
                lambda c: c.execute("INSERT INTO extras (note) VALUES ('kept')")
            )

    asyncio.run(main())


def test_a_tool_component_migrates_from_version_one_to_two(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    _seed_v1(db)
    seen: list[str] = []

    def step_three(connection: sqlite3.Connection) -> None:
        seen.append("ran")
        connection.execute("UPDATE extras SET detail = 'filled'")

    _open(db, (Component("tool", (V1, V2)),))

    assert check_schema(db).versions == {"core": CORE_VERSION, "tool": 2}
    assert read_rows(db, "extras") == [{"id": 1, "note": "kept", "detail": None}]
    assert "more" in table_names(db)

    _open(db, (Component("tool", (V1, V2, step_three)),))

    assert seen == ["ran"]
    assert read_rows(db, "extras")[0]["detail"] == "filled"
    _open(db, (Component("tool", (V1, V2, step_three)),))
    assert seen == ["ran"]


def test_a_failing_migration_step_rolls_the_whole_upgrade_back(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    _seed_v1(db)
    broken = "ALTER TABLE extras ADD COLUMN detail TEXT;\nCREATE TABLE extras (x);"

    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        _open(db, (Component("tool", (V1, broken)),))

    assert check_schema(db).versions == {"core": CORE_VERSION, "tool": 1}
    assert read_rows(db, "extras") == [{"id": 1, "note": "kept"}]


# =============================================================================
# Conversion
# =============================================================================


def test_conversion_rebuilds_a_populated_database_and_reruns_as_no_op(
    tmp_path: Path,
) -> None:
    db = tmp_path / "old.sqlite"
    saved = tmp_path / "snapshots" / "old.sqlite"
    build_p2(db, documents=3, pages=5)

    result = convert_versionless(db, [P2_CONVERTER], snapshot_path=saved)

    assert (result.status, result.converter, result.snapshot) == (
        "converted",
        "p2",
        saved,
    )
    assert sorted(table_names(saved)) == ["documents", "pages"]
    assert _counts(saved, "documents", "pages") == (3, 15)
    assert check_schema(db).versions == {"core": CORE_VERSION}
    assert not [name for name in table_names(db) if name.startswith("legacy_")]
    assert _counts(db, "documents", "units", "units WHERE status = 'failed'") == (
        3,
        15,
        3,
    )
    assert _integrity(db) == "ok"
    before = db.read_bytes()

    assert convert_versionless(db, [P2_CONVERTER], snapshot_path=saved).status == (
        "current"
    )
    assert db.read_bytes() == before

    async def resume() -> None:
        async with Store(db, "tool") as store:
            state = store.resume_state("doc0")
            assert state is not None
            assert state.counts == {"done": 4, "failed": 1}
            assert state.status == "failed"

    asyncio.run(resume())


def test_a_failed_verification_leaves_the_old_database(tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite"
    build_p2(db)
    before = db.read_bytes()

    def mismatch(connection: sqlite3.Connection, snapshot: Path) -> None:
        raise ValueError("Row counts differ")

    converter = Converter(
        "p2", P2_CONVERTER.matches, P2_CONVERTER.rebuild, verify=mismatch
    )

    with pytest.raises(ValueError, match="Row counts differ"):
        convert_versionless(db, [converter], snapshot_path=tmp_path / "s.sqlite")

    assert db.read_bytes() == before
    assert inspect_schema(db).versionless


def test_an_unrecognized_schema_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "other.sqlite"
    with contextlib.closing(sqlite3.connect(db)) as connection:
        connection.execute("CREATE TABLE things (x)")
        connection.execute("INSERT INTO things VALUES (1)")
        connection.commit()
    before = db.read_bytes()
    saved = tmp_path / "s.sqlite"

    with pytest.raises(UnrecognizedSchemaError) as raised:
        convert_versionless(db, [P2_CONVERTER], snapshot_path=saved)

    assert raised.value.exit_code == 2
    assert db.read_bytes() == before
    assert not saved.exists()

    ae = tmp_path / "ae.sqlite"
    build_ae_v1(ae)
    with pytest.raises(UnrecognizedSchemaError):
        convert_versionless(ae, [P2_CONVERTER], snapshot_path=saved)


def test_converting_an_empty_database_does_nothing(tmp_path: Path) -> None:
    db = tmp_path / "empty.sqlite"
    db.write_bytes(b"")

    result = convert_versionless(db, [P2_CONVERTER], snapshot_path=tmp_path / "s")

    assert result.status == "empty"
    assert db.read_bytes() == b""


@contextlib.contextmanager
def _child(*args: str) -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [sys.executable, CHILD, *args],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    watchdog = threading.Timer(CHILD_TIMEOUT, process.kill)
    watchdog.start()
    try:
        yield process
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.communicate()


def _expect(process: subprocess.Popen[str], word: str) -> None:
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if line != word:
        process.kill()
        _out, err = process.communicate()
        pytest.fail(f"child printed {line!r} instead of {word!r}:\n{err}")


def _old_layout(db: Path) -> Mapping[str, object]:
    return {
        "tables": sorted(table_names(db)),
        "counts": _counts(db, "documents", "pages", "pages WHERE status = 'failed'"),
        "rows": read_rows(db, "pages", order_by=["document", "page_index"]),
    }


def test_a_conversion_killed_inside_its_transaction_leaves_the_old_database(
    tmp_path: Path,
) -> None:
    db = tmp_path / "old.sqlite"
    saved = tmp_path / "snapshot.sqlite"
    journal = db.with_name(db.name + "-journal")
    build_p2(db, documents=4, pages=6)
    layout = _old_layout(db)

    with _child("convert", str(db), str(saved)) as process:
        _expect(process, "READY")
        assert journal.is_file()
        process.kill()
        process.wait(CHILD_TIMEOUT)

    assert journal.is_file()
    with pytest.raises(VersionlessDatabaseError):
        check_schema(db)
    assert not journal.exists()
    assert _old_layout(db) == layout
    assert _integrity(db) == "ok"

    result = convert_versionless(db, [P2_CONVERTER], snapshot_path=saved)

    assert result.status == "converted"
    assert _counts(db, "documents", "units", "units WHERE status = 'failed'") == (
        4,
        24,
        4,
    )
    assert _counts(db, "responses") == (0,)
    assert convert_versionless(db, [P2_CONVERTER], snapshot_path=saved).status == (
        "current"
    )
    assert _integrity(db) == "ok"
