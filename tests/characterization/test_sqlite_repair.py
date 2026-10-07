"""A missing database row is repaired from the working logs, through ``run_tool``.

An item whose files are complete but whose ``documents`` row is missing or
incomplete is finished on the next run from its logs, without a model call.
The ``--formats`` cases rewrite the adapter's ``--formats`` value.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.run as core
from autoexcerpter import __main__ as entry
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.rendering.sqlite import DATABASE_NAME, write_document
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import FakeLLM
from tests.characterization.inputs import make_pdf

PAGES = 3


def _run(llm: FakeLLM, root: Path, pdf: Path, **kwargs: Any) -> RunResult:
    llm.reset_records()
    return run_tool(pdf, llm=llm, root=root, output_dir=root / "out", **kwargs)


def _counts(result: RunResult) -> dict[str, int]:
    assert result.summary is not None, result.stdout
    keys = ("total", "complete", "failed", "skipped")
    return {key: result.summary[f"items_{key}"] for key in keys}


def _rows(database: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = read_rows(database, "documents")
    pages = read_rows(database, "pages", order_by=["document", "page_index"])
    return documents, pages


def _assert_complete_rows(database: Path) -> None:
    documents, pages = _rows(database)
    assert [(row["name"], row["complete"]) for row in documents] == [("doc", 1)]
    assert [(row["document"], row["page_index"]) for row in pages] == [
        ("doc", index) for index in range(PAGES)
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


@pytest.fixture
def pdf(docs_dir: Path) -> Path:
    return make_pdf(docs_dir / "doc.pdf", pages=PAGES)


# ============================================================================
# A failed database write is repaired by the next run
# ============================================================================
def test_a_failed_database_write_is_repaired_without_a_model_call(
    tmp_path: Path, fake_llm: FakeLLM, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "out" / DATABASE_NAME
    failures = iter([True])

    async def write_once_failing(*args: Any) -> None:
        if next(failures, False):
            raise sqlite3.OperationalError("database is locked")
        await write_document(*args)

    monkeypatch.setattr(core, "write_document", write_once_failing)

    first = _run(fake_llm, tmp_path, pdf)

    assert first.exit_code == 1, first.stderr
    assert _counts(first) == {"total": 1, "complete": 0, "failed": 1, "skipped": 0}
    paths = first.paths("doc")
    for path in (paths.transcription, paths.summary_md, paths.summary_docx):
        assert path.is_file()
    assert first.summary is not None
    assert str(database.resolve()) not in first.summary["outputs"]
    assert _rows(database) == ([], [])

    second = _run(fake_llm, tmp_path, pdf)

    assert second.exit_code == 0, second.stderr
    assert second.calls() == []
    assert _counts(second) == {"total": 1, "complete": 1, "failed": 0, "skipped": 0}
    assert second.summary is not None
    assert second.summary["outputs"][-1] == str(database.resolve())
    _assert_complete_rows(database)

    third = _run(fake_llm, tmp_path, pdf)

    assert third.exit_code == 0, third.stderr
    assert third.calls() == []
    assert _counts(third) == {"total": 0, "complete": 0, "failed": 0, "skipped": 1}


# ============================================================================
# --formats
# ============================================================================
def test_an_item_finished_without_sqlite_gains_its_rows_without_a_model_call(
    tmp_path: Path, fake_llm: FakeLLM, pdf: Path
) -> None:
    database = tmp_path / "out" / DATABASE_NAME
    with formats_flag("summary-md,summary-docx"):
        first = _run(fake_llm, tmp_path, pdf)
    assert first.exit_code == 0, first.stderr
    assert first.calls()
    assert not database.exists()

    second = _run(fake_llm, tmp_path, pdf)

    assert second.exit_code == 0, second.stderr
    assert second.calls() == []
    assert _counts(second) == {"total": 1, "complete": 1, "failed": 0, "skipped": 0}
    _assert_complete_rows(database)


def test_summaries_without_an_output_format_exit_2_before_any_call(
    tmp_path: Path, fake_llm: FakeLLM, pdf: Path
) -> None:
    with formats_flag(""):
        result = _run(fake_llm, tmp_path, pdf)

    assert result.exit_code == 2, result.stderr
    assert "--no-summarize" in result.stderr
    assert result.calls() == []
    assert result.llm.models == []
    assert result.created == []


def test_a_transcription_only_run_needs_no_output_format(
    tmp_path: Path, fake_llm: FakeLLM, pdf: Path
) -> None:
    with formats_flag(""):
        result = _run(fake_llm, tmp_path, pdf, summarize=False)

    assert result.exit_code == 0, result.stderr
    assert _counts(result) == {"total": 1, "complete": 1, "failed": 0, "skipped": 0}
    assert result.paths("doc").transcription.is_file()
    assert not (tmp_path / "out" / DATABASE_NAME).exists()
