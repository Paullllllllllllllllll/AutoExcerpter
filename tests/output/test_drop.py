"""The drop subcommand on a store that a run wrote: rows, files, exit codes.

Two two-page PDFs are run with offline managers into ``out/``, so each
document holds done units, summaries, citations and three rendered files.
``drop`` must delete one document's rows and unchanged files and leave the
other document alone; the cascade itself is tested in
``tests/common/test_store_drop.py``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter import __main__ as entry
from autoexcerpter.output.drop import KEPT_FILE_HINT
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.output.store import DATABASE_NAME, ItemState, classify
from tests.conftest import make_settings, make_spec

MakePdf = Callable[..., Path]
DOC_TABLES = ("units", "responses", "summaries", "citations", "outputs")


def _transcribe(payload: Any) -> dict[str, Any]:
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Text of page {payload.number}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _summarize(transcription: str, page_num: int) -> dict[str, Any]:
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": [f"A point of page {page_num}."],
        "references": [{"citation": "Smith, J. (2020). A Title.", "is_partial": False}],
        "processing_time": 0.01,
        "provider": "openai",
    }


@pytest.fixture
def out(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    mock_image_processing: dict[str, Any],
) -> Path:
    """Run ``doc.pdf`` and ``other.pdf`` into ``out/``; return the folder."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    transcriber = MagicMock()
    transcriber.transcribe_payload = AsyncMock(side_effect=_transcribe)
    summarizer = MagicMock()
    summarizer.generate_summary = AsyncMock(side_effect=_summarize)
    monkeypatch.setattr(
        managers_module, "TranscriptionManager", MagicMock(return_value=transcriber)
    )
    monkeypatch.setattr(
        managers_module, "SummaryManager", MagicMock(return_value=summarizer)
    )
    (tmp_path / "in").mkdir()
    make_pdf("in/doc.pdf", num_pages=2)
    make_pdf("in/other.pdf", num_pages=2)
    folder = tmp_path / "out"
    spec = make_spec(tmp_path / "in", output=folder, all=True, openalex=False)
    result = asyncio.run(core.run(core.plan(spec, make_settings())))
    assert result.complete == 2
    return folder


def _files(folder: Path, name: str) -> list[Path]:
    paths = ItemPaths.for_item(name, folder)
    return [paths.summary_docx, paths.summary_md, paths.transcription]


def _rows(folder: Path, name: str) -> dict[str, int]:
    connection = sqlite3.connect(folder / DATABASE_NAME)
    try:
        found = connection.execute(
            "SELECT doc_id FROM documents WHERE name = ?", (name,)
        ).fetchone()
        doc_id = found[0] if found else None
        counts = {
            table: connection.execute(
                f"SELECT count(*) FROM {table} WHERE doc_id IS ?", (doc_id,)
            ).fetchone()[0]
            for table in DOC_TABLES
        }
        counts["documents"] = int(found is not None)
        return counts
    finally:
        connection.close()


def _main(
    capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, dict[str, Any], str]:
    code = entry.main(["drop", *argv, "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out.strip().splitlines()[-1]), captured.err


def test_drop_deletes_one_document_and_its_files(
    out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _rows(out, "other")

    code, summary, err = _main(capsys, "--input", str(out), "--files", "doc.pdf")

    assert code == 0
    assert summary["command"] == "drop"
    assert summary["database"] == str((out / DATABASE_NAME).resolve())
    assert (summary["selected"], summary["dropped"]) == (1, ["doc"])
    assert summary["deleted_files"] == [str(path) for path in _files(out, "doc")]
    assert (summary["refused"], summary["kept_files"]) == ([], [])
    assert "dropped doc" in err
    assert set(_rows(out, "doc").values()) == {0}
    assert _rows(out, "other") == before
    assert not any(path.exists() for path in _files(out, "doc"))
    assert all(path.is_file() for path in _files(out, "other"))
    assert classify(out, "doc").state is ItemState.NONE


def test_a_dry_run_changes_nothing_and_an_edited_file_is_kept(
    out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edited = ItemPaths.for_item("doc", out).summary_md
    edited.write_text("edited by hand\n", encoding="utf-8")
    before = _rows(out, "doc")

    code, summary, err = _main(
        capsys, "--input", str(out / DATABASE_NAME), "--select", "doc", "--dry-run"
    )

    assert code == 0
    assert summary["dry_run"] is True
    assert summary["dropped"] == ["doc"]
    assert summary["kept_files"] == [
        {"path": str(edited), "reason": "it was changed outside the tool"}
    ]
    assert len(summary["deleted_files"]) == 2
    assert "DRY-RUN: would drop doc" in err
    assert _rows(out, "doc") == before
    assert all(path.is_file() for path in _files(out, "doc"))

    code, summary, err = _main(capsys, "--input", str(out), "--select", "doc")

    assert code == 0
    assert summary["kept_files"][0]["path"] == str(edited)
    assert edited.read_text(encoding="utf-8") == "edited by hand\n"
    assert KEPT_FILE_HINT in err
    assert classify(out, "doc").state is ItemState.NONE


def test_a_document_another_run_leases_is_refused_with_exit_1(
    out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    connection = sqlite3.connect(out / DATABASE_NAME)
    with connection:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'other-run', lease_until = ? "
            "WHERE name = 'doc'",
            (time.time() + 600,),
        )
    connection.close()

    code, summary, err = _main(capsys, "--input", str(out), "--all")

    assert code == 1
    assert summary["dropped"] == ["other"]
    (refusal,) = summary["refused"]
    assert refusal["name"] == "doc"
    assert "run other-run leases it" in refusal["reason"]
    assert "[ERROR] refused doc" in err
    assert _rows(out, "doc")["units"] > 0
    assert all(path.is_file() for path in _files(out, "doc"))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--select", "nothing-like-this"), "No stored document matches"),
        (("--files", "absent.pdf"), "--files names no stored document"),
        ((), "exactly one of --select, --files and --all"),
    ],
    ids=["select-matches-nothing", "files-match-nothing", "no-selection"],
)
def test_a_selection_that_chooses_nothing_exits_2(
    out: Path, capsys: pytest.CaptureFixture[str], argv: tuple[str, ...], message: str
) -> None:
    before = _rows(out, "doc")

    code, summary, err = _main(capsys, "--input", str(out), *argv)

    assert code == 2
    assert (summary["selected"], summary["dropped"]) == (0, [])
    assert message in err
    assert _rows(out, "doc") == before


def test_a_files_entry_that_names_two_documents_exits_2(
    out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    connection = sqlite3.connect(out / DATABASE_NAME)
    with connection:
        connection.execute("UPDATE documents SET name = 'doc.pdf' WHERE name = 'other'")
    connection.close()
    before = _rows(out, "doc")

    code, summary, err = _main(capsys, "--input", str(out), "--files", "doc.pdf")

    assert code == 2
    assert summary["dropped"] == []
    assert "names 2 documents (doc, doc.pdf)" in err
    assert _rows(out, "doc") == before


def test_a_folder_without_a_store_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, summary, err = _main(capsys, "--input", str(tmp_path), "--all")

    assert code == 2
    assert summary["command"] == "drop"
    assert f"No {DATABASE_NAME} in" in err
    assert not (tmp_path / DATABASE_NAME).exists()


def test_a_parse_error_writes_the_drop_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = entry.main(["drop", "--input", "x", "--all", "--select", "1", "--json"])

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out.strip())["command"] == "drop"
