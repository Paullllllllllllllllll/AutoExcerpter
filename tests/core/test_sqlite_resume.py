"""Resume classification against the output folder's database.

An item whose files and logs are complete but whose ``documents`` row is
missing or incomplete is finished from its working logs; the plan reads the
database without writing it. The checker's classification of each database
case is in tests/pipeline/test_resume_matrix.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.pipeline.resume as resume_module
import autoexcerpter.run as core
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.constants import LOG_FORMAT_VERSION
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.resume import ProcessingState, ResumeChecker
from autoexcerpter.rendering.sqlite import (
    DATABASE_NAME,
    complete_documents,
    open_database,
)
from tests.conftest import make_settings, make_spec

MakePdf = Callable[..., Path]

PAGES = 2
NAME = "doc"


def _write_log(path: Path, log_type: str, entries: list[dict[str, Any]]) -> None:
    header = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": log_type,
        "input_item_name": NAME,
        "input_item_path": "/fake/doc.pdf",
        "input_type": "PDF",
        "total_images": len(entries),
        "model_name": "gpt-5-mini",
    }
    lines = [json.dumps(header), *(json.dumps(entry) for entry in entries)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finished_item(out: Path, *, logs: bool = True, name: str = NAME) -> ItemPaths:
    """Write the outputs and working logs a complete earlier run leaves."""
    paths = ItemPaths.for_item(name, out)
    out.mkdir(parents=True, exist_ok=True)
    for path in (paths.transcription, paths.summary_md, paths.summary_docx):
        path.write_text("content", encoding="utf-8")
    if logs:
        _write_log(
            paths.transcription_log,
            "transcription",
            [
                {"original_input_order_index": index, "transcription": f"Page {index}"}
                for index in range(PAGES)
            ],
        )
        _write_log(
            paths.summary_log,
            "summary",
            [
                {"original_input_order_index": index, "bullet_points": ["A point."]}
                for index in range(PAGES)
            ],
        )
    return paths


def _document(name: str, complete: bool) -> dict[str, Any]:
    return {
        "name": name,
        "path": f"/fake/{name}.pdf",
        "kind": "pdf",
        "complete": complete,
    }


def _write_documents(out: Path, *rows: Mapping[str, Any]) -> Path:
    """Write ``documents`` rows into *out*'s database; return its path."""

    async def write() -> None:
        async with open_database(out) as writer:
            await writer.upsert("documents", rows)

    asyncio.run(write())
    return out / DATABASE_NAME


def _checker(**changes: Any) -> ResumeChecker:
    options: dict[str, Any] = {"resume_mode": "skip", "check_database": True}
    return ResumeChecker(**(options | changes))


def _files(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def test_only_complete_rows_are_read(tmp_path: Path) -> None:
    _write_documents(tmp_path, _document("alpha", True), _document("beta", False))

    assert complete_documents(tmp_path) == {"alpha"}


def test_a_folder_without_a_database_reads_empty_and_creates_nothing(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "out"
    folder.mkdir()

    assert complete_documents(folder) == frozenset()
    assert _files(folder) == []


def test_an_unreadable_database_reads_empty_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / DATABASE_NAME).write_bytes(b"not a database" * 100)

    with caplog.at_level(logging.WARNING):
        assert complete_documents(tmp_path) == frozenset()

    assert DATABASE_NAME in caplog.text


def test_the_check_is_off_by_default(tmp_path: Path) -> None:
    _finished_item(tmp_path)

    result = ResumeChecker(resume_mode="skip").should_skip(NAME, tmp_path)

    assert result.state is ProcessingState.COMPLETE


def test_an_unreadable_database_counts_as_no_row(tmp_path: Path) -> None:
    _finished_item(tmp_path)
    (tmp_path / DATABASE_NAME).write_bytes(b"not a database" * 100)

    result = _checker().should_skip(NAME, tmp_path)

    assert result.state is ProcessingState.TRANSCRIPTION_ONLY


def test_the_database_is_read_once_per_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("alpha", "beta"):
        _finished_item(tmp_path, name=name)
    calls: list[Path] = []

    def read(folder: Path) -> frozenset[str]:
        calls.append(folder)
        return frozenset({"alpha", "beta"})

    monkeypatch.setattr(resume_module, "complete_documents", read)
    checker = _checker()

    states = [checker.should_skip(name, tmp_path).state for name in ("alpha", "beta")]

    assert states == [ProcessingState.COMPLETE, ProcessingState.COMPLETE]
    assert calls == [tmp_path]


def _plan(pdf: Path, out: Path, **flags: Any) -> core.RunPlan:
    return core.plan(make_spec(pdf, output=out, **flags), make_settings())


def test_the_plan_resumes_an_item_without_a_complete_row(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf", num_pages=PAGES)
    out = tmp_path / "out"
    _finished_item(out)

    run_plan = _plan(pdf, out)

    (planned,) = run_plan.to_process
    assert planned.state == "transcription_only"
    assert planned.completed_pages == PAGES
    resume = planned.item_resume()
    assert resume.completed_page_indices == frozenset(range(PAGES))
    assert resume.transcription_results is not None
    assert resume.summary_results is not None
    assert run_plan.skipped == ()
    assert run_plan.warnings == ()


def test_the_plan_ignores_the_database_without_the_sqlite_format(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf", num_pages=PAGES)
    out = tmp_path / "out"
    _finished_item(out)

    run_plan = _plan(pdf, out, formats=("summary-md", "summary-docx"))

    assert [item.name for item in run_plan.skipped] == [NAME]
    assert run_plan.to_process == ()


def test_the_plan_warns_about_a_complete_item_without_logs_or_row(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf", num_pages=PAGES)
    out = tmp_path / "out"
    _finished_item(out, logs=False)

    run_plan = _plan(pdf, out)

    assert [item.name for item in run_plan.skipped] == [NAME]
    (warning,) = run_plan.warnings
    assert f"'{NAME}'" in warning
    assert "--force rebuilds it" in warning


def test_a_dry_run_without_a_database_creates_no_file(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf", num_pages=PAGES)
    out = tmp_path / "out"
    _finished_item(out)
    before = _files(out)

    _plan(pdf, out, dry_run=True)

    assert _files(out) == before
    assert DATABASE_NAME not in before


def test_a_dry_run_leaves_an_existing_database_as_it_was(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf", num_pages=PAGES)
    out = tmp_path / "out"
    _finished_item(out)
    database = _write_documents(out, _document(NAME, False))
    rows = read_rows(database, "documents")
    before = _files(out)

    run_plan = _plan(pdf, out, dry_run=True)

    assert [item.state for item in run_plan.to_process] == ["transcription_only"]
    assert _files(out) == before
    assert not (out / f"{DATABASE_NAME}-wal").exists()
    assert not (out / f"{DATABASE_NAME}-shm").exists()
    assert read_rows(database, "documents") == rows


def test_summaries_without_an_output_format_are_refused(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf")

    with pytest.raises(core.ConfigurationError, match="--no-summarize") as raised:
        _plan(pdf, tmp_path / "out", formats=())

    assert raised.value.exit_code == 2
    assert _plan(pdf, tmp_path / "out", formats=(), summarize=False).to_process
