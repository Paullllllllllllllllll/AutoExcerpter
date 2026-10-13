"""Resume of an item stored before the image presets moved to ``common/``.

``tests/fixtures/resume_before_presets/`` holds a three-page PDF and the
output folder of a run by v2.4.0 before the move: page 2 refused, pages 1
and 3 transcribed and summarized. The test copies both into a fresh root and
runs again, which retries the failed page. Resume compares the stored image
settings with the current ones and refuses a partly done item whose settings
differ, so the rerun requesting only page 2 proves that the record is
unchanged; the test also compares the stored record text before and after.
The fixture keeps its absolute paths, which tie the item to its input, under
a placeholder root (``tests/fixture_paths.py``).

With ``AE_UPDATE_GOLDEN=1`` and no fixture folder, the test writes the
fixture instead. Never rewrite it after the move: it must stay a record of
the earlier code.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from autoexcerpter.common.testing import WriteGuard
from autoexcerpter.output.store import DATABASE_NAME
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM, Injection
from tests.characterization.golden import UPDATE_ENV
from tests.characterization.inputs import make_pdf
from tests.fixture_paths import ROOT_PLACEHOLDER, relocate

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "resume_before_presets"
FOLDERS = ("inputs", "out")
PAGES = 3
FAILED_PAGE = 1


def _run(llm: FakeLLM, root: Path) -> RunResult:
    llm.reset_records()
    return run_tool(
        root / "inputs" / "doc.pdf", llm=llm, root=root, output_dir=root / "out"
    )


def _image_settings(database: Path) -> list[str]:
    with contextlib.closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            "SELECT image_settings FROM documents ORDER BY doc_id"
        ).fetchall()
    return [str(row[0]) for row in rows]


def _write_fixture(llm: FakeLLM, root: Path, write_guard: WriteGuard) -> None:
    make_pdf(root / "inputs" / "doc.pdf", pages=PAGES)
    llm.inject(FAILED_PAGE, TRANSCRIPTION, 1, Injection(refusal="No."))
    first = _run(llm, root)
    llm.page(FAILED_PAGE).injections.clear()
    assert first.exit_code == 1, first.stderr
    for folder in FOLDERS:
        shutil.copytree(root / folder, FIXTURE / folder)
    relocate(FIXTURE, str(root), ROOT_PLACEHOLDER)
    write_guard.consume(FIXTURE.parent)


def test_rerun_resumes_an_item_stored_before_the_move(
    fake_llm: FakeLLM, tmp_path: Path, write_guard: WriteGuard
) -> None:
    root = tmp_path / "resume"
    root.mkdir()
    if not FIXTURE.is_dir():
        if os.environ.get(UPDATE_ENV) != "1":
            pytest.fail(f"No fixture {FIXTURE.name}; run with {UPDATE_ENV}=1.")
        _write_fixture(fake_llm, tmp_path / "write", write_guard)
    for folder in FOLDERS:
        shutil.copytree(FIXTURE / folder, root / folder)
    relocate(root, ROOT_PLACEHOLDER, str(root))
    database = root / "out" / DATABASE_NAME
    stored = _image_settings(database)

    rerun = _run(fake_llm, root)

    assert rerun.exit_code == 0, rerun.stderr
    assert [call["page"] for call in rerun.calls(TRANSCRIPTION)] == [FAILED_PAGE]
    assert [call["page"] for call in rerun.calls(SUMMARY)] == [FAILED_PAGE]
    assert rerun.summary is not None
    assert rerun.summary["items_complete"] == 1
    assert _image_settings(database) == stored
