"""migrate converts version 1.0 databases into the store schema.

A version 1.0 ``autoexcerpter.sqlite`` (``documents`` keyed by name,
``pages``, ``summaries``, ``citations``) is snapshotted and rebuilt in one
transaction; its row counts are verified against the snapshot, and a
failed verification leaves the old database as it was.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.migrate as migrate_module
from autoexcerpter import __main__ as entry
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.common.store import UnrecognizedSchemaError, inspect_schema
from autoexcerpter.migrate import MigrationReport, migrate
from autoexcerpter.output.store import DATABASE_NAME
from tests.migrate.conftest import METADATA
from tests.migrate.legacy import (
    CITATION,
    USAGE,
    Item,
    Page,
    make_item,
    write_database,
    write_files,
    write_logs,
)


def _migrate(root: Path, state_dir: Path, **options: Any) -> MigrationReport:
    return asyncio.run(migrate(root, state_dir=state_dir, **options))


def _items(root: Path, state_dir: Path) -> list[Item]:
    first = make_item(
        root,
        "alpha",
        [Page("One.", references=(CITATION,)), Page(error="timeout")],
        state_dir=state_dir,
    )
    second = make_item(root, "beta", [Page("Only page.")], state_dir=state_dir)
    for item in (first, second):
        write_files(item)
    return [first, second]


@pytest.fixture
def root(tmp_path: Path, state_dir: Path) -> Path:
    """An output folder with a version 1.0 database of two items, no logs."""
    folder = tmp_path / "out"
    write_database(folder, _items(folder, state_dir))
    return folder


def _units(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    documents = {
        row["doc_id"]: row["name"]
        for row in read_rows(root / DATABASE_NAME, "documents")
    }
    return {
        (documents[row["doc_id"]], row["kind"], row["idx"]): row
        for row in read_rows(root / DATABASE_NAME, "units")
    }


def test_a_version_1_database_is_converted_and_verified(
    root: Path, state_dir: Path
) -> None:
    report = _migrate(root, state_dir)

    assert report.exit_code == 0
    (database,) = report.databases
    assert database.action == "converted"
    assert database.snapshot is not None and database.snapshot.is_file()
    assert not inspect_schema(database.snapshot).versions
    assert inspect_schema(root / DATABASE_NAME).versions == {
        "core": 1,
        "autoexcerpter": 1,
    }
    units = _units(root)
    assert len(units) == 6
    failed = units[("alpha", "transcription", 1)]
    assert (failed["status"], failed["error"]) == ("failed", "timeout")
    assert units[("alpha", "summary", 1)]["outcome"] == "placeholder"
    assert json.loads(units[("beta", "transcription", 0)]["usage"]) == USAGE
    assert {d.before for d in report.documents} == {"versionless"}
    assert {d.after for d in report.documents} == {"complete", "partial"}
    assert all(d.warnings == [] for d in report.documents)


def test_the_citations_are_consolidated_from_the_cache_alone(
    root: Path, state_dir: Path
) -> None:
    report = _migrate(root, state_dir)

    rows = read_rows(root / DATABASE_NAME, "citations")
    assert [row["text"] for row in rows] == [CITATION]
    assert rows[0]["doi"] == METADATA["doi"]
    assert {d.name: d.citations for d in report.documents} == {"alpha": 1, "beta": 0}


def test_stored_citations_are_carried_over(tmp_path: Path, state_dir: Path) -> None:
    folder = tmp_path / "out"
    write_database(folder, _items(folder, state_dir), citations=True)

    report = _migrate(folder, state_dir)

    assert report.total("citations") == 0
    rows = read_rows(folder / DATABASE_NAME, "citations")
    assert [(row["text"], row["pages"]) for row in rows] == [(CITATION, "p. 1")] * 2


def test_the_converted_files_are_registered(root: Path, state_dir: Path) -> None:
    _migrate(root, state_dir)

    outputs = {row["path"] for row in read_rows(root / DATABASE_NAME, "outputs")}
    assert outputs == {
        f"{name}{suffix}"
        for name in ("alpha", "beta")
        for suffix in ("_transcription.md", "_summary.md", "_summary.docx")
    }


def test_logs_beside_a_database_add_provenance_and_keep_usage(
    root: Path, state_dir: Path
) -> None:
    write_logs(_items(root, state_dir)[0])

    report = _migrate(root, state_dir)

    assert report.exit_code == 0
    unit = _units(root)[("alpha", "transcription", 0)]
    assert json.loads(unit["usage"]) == USAGE
    assert json.loads(unit["provenance"])["image_provenance"] == {"dpi": 300, "page": 0}
    again = _migrate(root, state_dir)
    assert again.total("imported") == 0
    assert again.databases[0].action == "unchanged"


def test_a_second_run_leaves_the_converted_database_alone(
    root: Path, state_dir: Path
) -> None:
    _migrate(root, state_dir)
    before = (root / DATABASE_NAME).read_bytes()

    report = _migrate(root, state_dir)

    assert report.databases[0].action == "unchanged"
    assert report.total("imported") + report.total("citations") == 0
    assert (root / DATABASE_NAME).read_bytes() == before


def test_a_dry_run_converts_nothing(root: Path, state_dir: Path) -> None:
    report = _migrate(root, state_dir, dry_run=True)

    assert report.databases[0].action == "convert"
    assert inspect_schema(root / DATABASE_NAME).versionless


def test_a_failed_verification_leaves_the_old_database(
    root: Path, state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(migrate_module, "_convert_summaries", lambda *args: None)

    report = _migrate(root, state_dir)

    assert report.exit_code == 1
    assert "does not hold the rows" in report.errors[0]
    assert inspect_schema(root / DATABASE_NAME).versionless


def test_an_unknown_layout_is_refused(tmp_path: Path, state_dir: Path) -> None:
    folder = tmp_path / "out"
    folder.mkdir()
    connection = sqlite3.connect(folder / DATABASE_NAME)
    with connection:
        connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE pages (document TEXT)")
    connection.close()

    with pytest.raises(UnrecognizedSchemaError):
        _migrate(folder, state_dir)

    code = entry.main(["migrate", "--input", str(folder)])

    assert code == 2
    assert inspect_schema(folder / DATABASE_NAME).versionless
