"""The resume matrix before and after migrate.

Each case is an item as version 1.0 left it, with the work version 1.0
would have done on resume: the pages it transcribed again, the pages it
summarized again and whether it only rendered its files. Before migrate the
store has no units, so every case with files or logs is ``legacy``; after
it, the store must classify each item to the same work. The state names
differ where version 1.0 named the work differently: it called a retry of
failed pages ``transcription_only`` (the store says ``partial``) and a
render of missing summary files from complete logs ``transcription_only``
(the store says ``complete`` and renders without model calls).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from autoexcerpter.migrate import migrate
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    ResumeRequest,
    classify,
)
from tests.migrate.legacy import Item, Page, make_item, write_files, write_logs


@dataclass(frozen=True)
class Case:
    """One row of the matrix: the item, version 1.0's state and work."""

    build: Callable[[Path, Path], Item]
    v1_state: str
    transcribe: tuple[int, ...]
    summarize: tuple[int, ...]
    render: bool
    after: str


def _complete(root: Path, state: Path) -> Item:
    item = make_item(root, "doc", [Page("A."), Page("B.")], state_dir=state)
    write_logs(item)
    write_files(item)
    return item


def _partial(root: Path, state: Path) -> Item:
    item = make_item(root, "doc", [Page("A."), Page("B.")], state_dir=state, total=3)
    write_logs(item)
    return item


def _failed_page(root: Path, state: Path) -> Item:
    item = make_item(root, "doc", [Page("A."), Page(error="timeout")], state_dir=state)
    write_logs(item)
    write_files(item)
    return item


def _failed_summary(root: Path, state: Path) -> Item:
    pages = [Page("A."), Page("B.", summary_error="refused")]
    item = make_item(root, "doc", pages, state_dir=state)
    write_logs(item)
    write_files(item)
    return item


def _no_summary_log(root: Path, state: Path) -> Item:
    pages = [Page("A."), Page("B.")]
    item = make_item(root, "doc", pages, state_dir=state, summarized=False)
    write_logs(item)
    write_files(item, summary=False)
    return item


def _summary_file_missing(root: Path, state: Path) -> Item:
    item = _complete(root, state)
    item.paths.summary_md.unlink()
    return item


def _files_without_logs(root: Path, state: Path) -> Item:
    item = make_item(root, "doc", [Page("A.")], state_dir=state)
    write_files(item)
    return item


CASES = {
    "complete": Case(_complete, "complete", (), (), False, "complete"),
    "partial": Case(_partial, "partial", (2,), (2,), False, "partial"),
    "failed-page": Case(
        _failed_page, "transcription_only", (1,), (1,), False, "partial"
    ),
    "failed-summary": Case(
        _failed_summary, "transcription_only", (), (1,), False, "transcription_only"
    ),
    "no-summary-log": Case(
        _no_summary_log, "transcription_only", (), (0, 1), False, "transcription_only"
    ),
    "summary-file-missing": Case(
        _summary_file_missing, "transcription_only", (), (), True, "complete"
    ),
    "files-without-logs": Case(
        _files_without_logs, "complete", (), (), False, "legacy"
    ),
}


def _request(item: Item) -> ResumeRequest:
    paths = item.paths
    return ResumeRequest(
        outputs=(paths.transcription, paths.summary_docx, paths.summary_md),
        legacy=paths.legacy_logs(),
    )


@pytest.mark.parametrize("name", list(CASES))
def test_migrate_keeps_the_resume_work_of_version_1(
    name: str, tmp_path: Path, state_dir: Path
) -> None:
    case = CASES[name]
    item = case.build(tmp_path / "out", state_dir)

    before = classify(item.root, item.name, _request(item))
    report = asyncio.run(migrate(item.root, state_dir=state_dir))
    after = classify(item.root, item.name, _request(item))

    assert before.state.value == "legacy"
    assert report.exit_code == 0
    assert after.state.value == case.after
    assert after.indices(KIND_TRANSCRIPTION) == case.transcribe
    assert after.indices(KIND_SUMMARY) == case.summarize
    assert after.render is case.render
    for document in report.documents:
        assert (document.before, document.after) == ("legacy", case.after)
