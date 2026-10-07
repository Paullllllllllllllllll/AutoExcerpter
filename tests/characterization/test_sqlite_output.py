"""The SQLite output across runs: one file per output folder, rows upserted.

Each test runs ``autoexcerpter run`` more than once over the same output
folder through ``run_tool`` and reads ``autoexcerpter.sqlite`` back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoexcerpter.common.sqlite import read_rows, table_names
from autoexcerpter.rendering.sqlite import DATABASE_NAME, TABLES
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import FakeLLM
from tests.characterization.inputs import make_pdf

_KEYS = {spec.name: list(spec.key) for spec in TABLES}
# Columns that change between two runs of the same input; the spec records
# the second run's --force.
_PER_RUN = {"updated_at", "spec"}


def _tables(database: Path) -> dict[str, list[dict[str, Any]]]:
    """Return every table's rows ordered by key, without per-run columns."""
    return {
        table: [
            {key: value for key, value in row.items() if key not in _PER_RUN}
            for row in read_rows(database, table, order_by=_KEYS[table])
        ]
        for table in table_names(database)
    }


def _run(llm: FakeLLM, root: Path, input_path: Path, **kwargs: Any) -> RunResult:
    llm.reset_records()
    result = run_tool(input_path, llm=llm, root=root, output_dir=root / "out", **kwargs)
    assert result.exit_code == 0, result.stderr
    return result


def test_two_items_share_one_database_and_a_rerun_upserts(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    tree = tmp_path / "in"
    make_pdf(tree / "alpha.pdf", pages=2)
    make_pdf(tree / "beta.pdf", pages=3)
    database = tmp_path / "out" / DATABASE_NAME

    _run(fake_llm, tmp_path, tree)
    first = _tables(database)

    assert [path.name for path in (tmp_path / "out").glob("*.sqlite*")] == [
        DATABASE_NAME
    ]
    assert [row["name"] for row in first["documents"]] == ["alpha", "beta"]
    assert all(row["complete"] == 1 for row in first["documents"])
    pages = [(row["document"], row["page_index"]) for row in first["pages"]]
    assert pages == [("alpha", 0), ("alpha", 1), ("beta", 0), ("beta", 1), ("beta", 2)]
    assert len(first["summaries"]) == 5
    assert {row["document"] for row in first["citations"]} == {"alpha", "beta"}

    _run(fake_llm, tmp_path, tree, force=True)

    assert _tables(database) == first
    specs = [json.loads(row["spec"]) for row in read_rows(database, "documents")]
    assert {spec["run.force"]["value"] for spec in specs} == {True}


def test_a_shorter_replacement_input_drops_the_stale_page_rows(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    pdf = make_pdf(tmp_path / "in" / "doc.pdf", pages=3)
    database = tmp_path / "out" / DATABASE_NAME
    _run(fake_llm, tmp_path, pdf)

    make_pdf(pdf, pages=2)
    _run(fake_llm, tmp_path, pdf)

    tables = _tables(database)
    assert [row["page_index"] for row in tables["pages"]] == [0, 1]
    assert [row["page_index"] for row in tables["summaries"]] == [0, 1]
    (document,) = tables["documents"]
    assert document["pages_total"] == 2
    assert document["complete"] == 1
