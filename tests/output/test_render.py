"""Files rendered from the store, and the render subcommand.

A two-page PDF is run with offline managers, so its document holds done
transcription and summary units, summaries and citations. The renderers must
reproduce the files the run wrote from the store alone, to the byte and the
same on every render; ``render`` writes missing files again, leaves edited
ones unless forced, and refuses a database that is not a current store.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter import __main__ as entry
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.common.store import RenderReport
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.output.render import (
    RENDERERS,
    preview_database,
    render_database,
    render_item,
)
from autoexcerpter.output.store import DATABASE_NAME, open_store, read_only_connection
from tests.conftest import make_settings, make_spec

MakePdf = Callable[..., Path]
CITATION = "Smith, J. (2020). A Title. Journal, 1, 1-2."


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
        "references": [{"citation": CITATION, "is_partial": False}],
        "processing_time": 0.01,
        "provider": "openai",
    }


@pytest.fixture
def stored(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    mock_image_processing: dict[str, Any],
) -> ItemPaths:
    """Run a two-page PDF into ``out/``; return the item's paths."""
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
    out = tmp_path / "out"
    spec = make_spec(make_pdf("doc.pdf", num_pages=2), output=out, openalex=False)
    result = asyncio.run(core.run(core.plan(spec, make_settings())))
    assert result.complete == 1
    return ItemPaths.for_item("doc", out)


def _rendered(root: Path, fmt: str) -> str | bytes:
    with read_only_connection(root) as connection:
        return RENDERERS[fmt](connection, 1)


def _as_bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


@pytest.mark.parametrize(
    ("fmt", "attribute"),
    [
        ("transcription-md", "transcription"),
        ("summary-md", "summary_md"),
        ("summary-docx", "summary_docx"),
    ],
)
def test_the_store_renders_the_files_of_the_run_to_the_byte(
    stored: ItemPaths, fmt: str, attribute: str
) -> None:
    first = _as_bytes(_rendered(stored.output_dir, fmt))
    second = _as_bytes(_rendered(stored.output_dir, fmt))

    assert first == second
    assert getattr(stored, attribute).read_bytes() == first


def test_the_transcription_header_shows_the_document_time(stored: ItemPaths) -> None:
    text = stored.transcription.read_text(encoding="utf-8")

    assert "# Processed on: " in text
    assert "Total processing time" not in text
    assert text.endswith("Text of page 2.\n")


def test_the_header_time_is_the_document_time_in_utc(stored: ItemPaths) -> None:
    connection = sqlite3.connect(stored.output_dir / DATABASE_NAME)
    with connection:
        connection.execute(
            "UPDATE documents SET updated_at = '2026-01-01T10:00:00.000+02:00'"
        )
    connection.close()

    text = str(_rendered(stored.output_dir, "transcription-md"))
    summary = str(_rendered(stored.output_dir, "summary-md"))

    assert "# Processed on: 2026-01-01 08:00:00\n" in text
    assert "*Summary generated 2026-01-01 08:00 " in summary


def test_a_renderer_that_raises_marks_its_output_stale(
    stored: ItemPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = stored.output_dir / DATABASE_NAME
    before = {row["path"]: row["sha256"] for row in read_rows(database, "outputs")}

    def broken(connection: sqlite3.Connection, doc_id: int) -> str:
        raise ValueError("broken renderer")

    monkeypatch.setitem(RENDERERS, "summary-md", broken)

    async def main() -> RenderReport:
        async with open_store(stored.output_dir) as store:
            await store.begin_run("render")
            lease = await store.acquire(1)
            assert lease is not None
            try:
                return await render_item(
                    store, lease, stored, ["transcription-md", "summary-md"]
                )
            finally:
                await store.release(lease)

    report = asyncio.run(main())

    assert report.stale == [stored.summary_md]
    assert "broken renderer" in report.warnings[-1]
    rows = {row["path"]: row for row in read_rows(database, "outputs")}
    summary = rows[stored.summary_md.name]
    assert (summary["stale"], summary["sha256"]) == (1, before[stored.summary_md.name])
    assert rows[stored.transcription.name]["stale"] == 0


def test_the_summaries_render_the_stored_citations(stored: ItemPaths) -> None:
    text = stored.summary_md.read_text(encoding="utf-8")

    assert CITATION in text
    assert "*(pp. 1-2)*" in text


def _note(root: Path, message: str, *, current: bool) -> None:
    connection = sqlite3.connect(root / DATABASE_NAME)
    with connection:
        run_id = None
        if current:
            (run_id,) = connection.execute("SELECT run_id FROM units").fetchone()
        connection.execute(
            "INSERT INTO notes (doc_id, run_id, level, message, created_at) "
            "VALUES (1, ?, 'warning', ?, '2026-10-11T00:00:00.000+00:00')",
            (run_id, message),
        )
    connection.close()


def test_only_model_notes_of_runs_with_current_units_reach_the_header(
    stored: ItemPaths,
) -> None:
    current = "Resume with another model: 'doc' holds units of a; this run uses b."
    gone = "Resume with another model: 'doc' holds units of c; this run uses d."
    _note(stored.output_dir, current, current=True)
    _note(stored.output_dir, gone, current=False)
    _note(stored.output_dir, "Some other warning.", current=True)

    text = str(_rendered(stored.output_dir, "transcription-md"))

    assert f"# Note: {current}\n" in text
    assert gone not in text
    assert "Some other warning." not in text


def test_render_database_writes_missing_files_and_keeps_edited_ones(
    stored: ItemPaths,
) -> None:
    database = stored.output_dir / DATABASE_NAME
    stored.summary_md.unlink()
    stored.transcription.write_text("edited by hand\n", encoding="utf-8")

    preview = preview_database(database)
    (result,) = asyncio.run(render_database(database))

    assert preview[0].written == [stored.summary_md]
    assert preview[0].kept == [stored.transcription]
    assert result.written == [stored.summary_md]
    assert result.kept == [stored.transcription]
    assert result.unchanged == [stored.summary_docx]
    assert stored.transcription.read_text(encoding="utf-8") == "edited by hand\n"

    (forced,) = asyncio.run(render_database(database, force=True))

    assert forced.written == [stored.transcription]
    assert stored.transcription.read_bytes() == _as_bytes(
        _rendered(stored.output_dir, "transcription-md")
    )


def test_render_with_formats_adds_a_format_at_the_item_path(
    stored: ItemPaths,
) -> None:
    database = stored.output_dir / DATABASE_NAME
    txt = stored.transcription.with_suffix(".txt")

    (result,) = asyncio.run(render_database(database, formats=["transcription-txt"]))

    assert result.written == [txt]
    assert txt.read_bytes() == stored.transcription.read_bytes()
    formats = {row["format"] for row in read_rows(database, "outputs")}
    assert "transcription-txt" in formats


def _main(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = entry.main(["render", *argv, "--json"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    return code, json.loads(line)


def test_the_render_command_reports_what_it_wrote(
    stored: ItemPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    stored.summary_docx.unlink()

    code, summary = _main(capsys, "--input", str(stored.output_dir))

    assert code == 0
    assert summary["command"] == "render"
    assert (summary["documents"], summary["written"], summary["unchanged"]) == (
        1,
        1,
        2,
    )
    assert summary["outputs"] == [str(stored.summary_docx)]
    assert stored.summary_docx.is_file()


def test_a_dry_run_render_writes_nothing(
    stored: ItemPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    stored.summary_docx.unlink()

    code, summary = _main(capsys, "--input", str(stored.output_dir), "--dry-run")

    assert code == 0
    assert summary["dry_run"] is True
    assert summary["written"] == 1
    assert not stored.summary_docx.exists()


def test_a_versionless_database_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    connection = sqlite3.connect(out / DATABASE_NAME)
    with connection:
        connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
    connection.close()

    code = entry.main(["render", "--input", str(out), "--json"])

    captured = capsys.readouterr()
    assert code == 2
    assert "is not an AutoExcerpter store" in captured.err
    assert "migrate" not in captured.err
    assert json.loads(captured.out.strip())["written"] == 0
