"""The store across runs: one database per output folder, units kept or reset.

Each test runs ``autoexcerpter run`` through ``run_tool`` and reads
``autoexcerpter.sqlite`` back.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter import __main__ as entry
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.output.store import DATABASE_NAME
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import FakeLLM
from tests.characterization.inputs import make_pdf

# Columns that change between two runs of the same input.
_PER_RUN = {"run_id", "updated_at", "response_id", "attempts", "last_run_id"}


def _run(llm: FakeLLM, root: Path, input_path: Path, **kwargs: Any) -> RunResult:
    llm.reset_records()
    return run_tool(input_path, llm=llm, root=root, output_dir=root / "out", **kwargs)


def _units(database: Path) -> list[dict[str, Any]]:
    """Return the units in key order, without the columns of each run."""
    return [
        {key: value for key, value in row.items() if key not in _PER_RUN}
        for row in read_rows(database, "units", order_by=["doc_id", "kind", "idx"])
    ]


def _view(database: Path, view: str) -> list[dict[str, Any]]:
    order = ["document"] if view == "document_status" else ["document", "order_index"]
    return read_rows(database, view, order_by=order)


def _keys(database: Path) -> list[tuple[str, str, int]]:
    return [
        (str(row["document"]), str(row["kind"]), int(row["order_index"]))
        for row in _view(database, "page_text")
    ] + [
        (str(row["document"]), "summary", int(row["order_index"]))
        for row in _view(database, "page_summaries")
    ]


@contextlib.contextmanager
def formats_flag(value: str) -> Iterator[None]:
    """Make the adapter pass ``--formats <value>`` instead of its own list."""
    real = entry.main

    def main(argv: Sequence[str] | None = None) -> int:
        args = list(argv or ())
        args[args.index("--formats") + 1] = value
        return real(args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(entry, "main", main)
        yield


def test_two_items_share_one_database_and_a_forced_rerun_rewrites_the_units(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    tree = tmp_path / "in"
    make_pdf(tree / "alpha.pdf", pages=2)
    make_pdf(tree / "beta.pdf", pages=3)
    database = tmp_path / "out" / DATABASE_NAME

    first = _run(fake_llm, tmp_path, tree)
    assert first.exit_code == 0, first.stderr
    units = _units(database)

    assert [path.name for path in (tmp_path / "out").glob("*.sqlite*")] == [
        DATABASE_NAME
    ]
    documents = _view(database, "document_status")
    assert [(row["document"], row["status"]) for row in documents] == [
        ("alpha", "complete"),
        ("beta", "complete"),
    ]
    assert sorted(_keys(database)) == sorted(
        [
            (name, kind, idx)
            for name, pages in (("alpha", 2), ("beta", 3))
            for kind in ("transcription", "summary")
            for idx in range(pages)
        ]
    )
    citations = read_rows(database, "document_citations", order_by=["document"])
    assert {row["document"] for row in citations} == {"alpha", "beta"}

    second = _run(fake_llm, tmp_path, tree, force=True)

    assert second.exit_code == 0, second.stderr
    assert _units(database) == units
    runs = read_rows(database, "runs")
    assert [run["exit_code"] for run in runs] == [0, 0]
    assert len(read_rows(database, "responses")) == 2 * len(units)


def test_a_shorter_replacement_input_drops_the_stale_pages(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    pdf = make_pdf(tmp_path / "in" / "doc.pdf", pages=3)
    database = tmp_path / "out" / DATABASE_NAME
    assert _run(fake_llm, tmp_path, pdf).exit_code == 0

    make_pdf(pdf, pages=2)
    second = _run(fake_llm, tmp_path, pdf)

    assert second.exit_code == 0, second.stderr
    assert sorted(_keys(database)) == [
        ("doc", "summary", 0),
        ("doc", "summary", 1),
        ("doc", "transcription", 0),
        ("doc", "transcription", 1),
    ]
    (document,) = _view(database, "document_status")
    assert (document["units_total"], document["status"]) == (4, "complete")


def test_summaries_without_an_output_format_exit_2_before_any_call(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=1)
    with formats_flag(""):
        result = _run(fake_llm, tmp_path, pdf)

    assert result.exit_code == 2, result.stderr
    assert "--no-summarize" in result.stderr
    assert result.calls() == []
    assert result.llm.models == []
    assert result.created == []


def test_a_transcription_only_run_needs_no_output_format(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=1)
    with formats_flag(""):
        result = _run(fake_llm, tmp_path, pdf, summarize=False)

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 1
    assert result.paths("doc").transcription.is_file()
    assert (tmp_path / "out" / DATABASE_NAME).is_file()
