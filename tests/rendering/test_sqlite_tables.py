"""AutoExcerpter's SQLite rows: documents, pages, summaries and citations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.common.usage import Usage, UsageTotals
from autoexcerpter.rendering.citations import Citation
from autoexcerpter.rendering.sqlite import (
    DATABASE_NAME,
    DocumentRecord,
    RunInfo,
    open_database,
    settings_fingerprint,
    write_document,
)

SPEC: dict[str, dict[str, Any]] = {
    "input.path": {"value": "C:/in/doc.pdf", "source": "flag"},
    "run.force": {"value": False, "source": "default"},
    "transcription_model.model": {"value": "gpt-5-mini", "source": "settings"},
}


def _page(index: int, **extra: Any) -> dict[str, Any]:
    return {
        "original_input_order_index": index,
        "image": f"page_{index + 1:04d}.jpg",
        "page_index": index,
        "transcription": f"Text {index}",
        **extra,
    }


def _summary(index: int, **extra: Any) -> dict[str, Any]:
    return {
        "original_input_order_index": index,
        "page_information": {"page_number_integer": index + 1},
        "bullet_points": [f"Point {index}"],
        "references": [],
        **extra,
    }


def _usage(transcription: int, summary: int = 0) -> UsageTotals:
    totals = UsageTotals()
    totals.add("transcription", Usage(total_tokens=transcription))
    if summary:
        totals.add("summary", Usage(total_tokens=summary))
    return totals


def _record(count: int, **overrides: Any) -> DocumentRecord:
    values: dict[str, Any] = {
        "name": "doc",
        "path": Path("C:/in/doc.pdf"),
        "kind": "pdf",
        "hash": "abc",
        "transcription_model": "gpt-5-mini",
        "summary_model": "gpt-5-mini",
        "complete": True,
        "report": {"pages_total": count, "outputs": ["C:/out/doc_summary.md"]},
        "pages": [_page(i) for i in range(count)],
        "summaries": [_summary(i) for i in range(count)],
        "usage": {i: _usage(100 + i, 10 + i) for i in range(count)},
    }
    values.update(overrides)
    return DocumentRecord(**values)


def _write(folder: Path, *records: DocumentRecord) -> Path:
    async def main() -> None:
        async with open_database(folder) as writer:
            for record in records:
                await write_document(writer, record, RunInfo(SPEC, "9.9.9"))

    asyncio.run(main())
    return folder / DATABASE_NAME


def _rows(database: Path, table: str) -> list[dict[str, Any]]:
    return read_rows(database, table, order_by=["page_index"])


def test_a_document_writes_one_row_per_page_and_summary(tmp_path: Path) -> None:
    database = _write(tmp_path, _record(2))

    (document,) = read_rows(database, "documents")
    assert document["hash"] == "abc"
    assert document["tool_version"] == "9.9.9"
    assert document["settings_fingerprint"] == settings_fingerprint(SPEC)
    assert json.loads(document["spec"]) == SPEC
    assert json.loads(document["outputs"]) == ["doc_summary.md"]
    pages = _rows(database, "pages")
    assert [(row["status"], row["model"]) for row in pages] == [
        ("ok", "gpt-5-mini"),
        ("ok", "gpt-5-mini"),
    ]
    assert json.loads(pages[1]["usage"])["total_tokens"] == 101
    summaries = _rows(database, "summaries")
    assert json.loads(summaries[0]["usage"])["total_tokens"] == 10
    assert json.loads(summaries[0]["bullet_points"]) == ["Point 0"]


def test_reused_pages_keep_the_usage_of_the_run_that_made_them(
    tmp_path: Path,
) -> None:
    first = _record(2, pages=[_page(0), _page(1, error="refused")])
    resumed = _record(
        2,
        usage={1: _usage(500, 50)},
        reused_pages=frozenset({0}),
        reused_model="gpt-5-mini",
    )

    database = _write(tmp_path, first, resumed)

    pages = _rows(database, "pages")
    assert [row["status"] for row in pages] == ["ok", "ok"]
    assert [json.loads(row["usage"])["total_tokens"] for row in pages] == [100, 500]
    summaries = _rows(database, "summaries")
    assert [json.loads(row["usage"])["total_tokens"] for row in summaries] == [10, 50]


def test_a_shorter_rerun_deletes_the_stale_rows(tmp_path: Path) -> None:
    database = _write(tmp_path, _record(3), _record(1))

    assert [row["page_index"] for row in _rows(database, "pages")] == [0]
    assert [row["page_index"] for row in _rows(database, "summaries")] == [0]


def test_a_run_without_summaries_leaves_the_summary_rows(tmp_path: Path) -> None:
    database = _write(tmp_path, _record(2), _record(2, summaries=None))

    assert len(_rows(database, "summaries")) == 2


def test_citations_are_replaced_by_each_rendered_summary(tmp_path: Path) -> None:
    old = Citation(raw_text="Old, A. (1900). Gone.")
    new = Citation(raw_text="New, B. (2000). Kept.", doi="10.1/x", partial=True)

    database = _write(
        tmp_path,
        _record(1, citations=[old]),
        _record(1, citations=[new]),
        _record(1, citations=None),
    )

    (row,) = read_rows(database, "citations")
    assert (row["position"], row["text"], row["doi"], row["partial"]) == (
        1,
        "New, B. (2000). Kept.",
        "10.1/x",
        1,
    )


def test_the_fingerprint_ignores_input_run_control_and_sources() -> None:
    moved: dict[str, dict[str, Any]] = {
        **SPEC,
        "input.path": {"value": "D:/elsewhere.pdf", "source": "flag"},
        "run.force": {"value": True, "source": "flag"},
        "output.directory": {"value": "D:/out", "source": "flag"},
    }
    other_model: dict[str, dict[str, Any]] = {
        **SPEC,
        "transcription_model.model": {"value": "gpt-5", "source": "settings"},
    }

    assert settings_fingerprint(moved) == settings_fingerprint(SPEC)
    assert settings_fingerprint(other_model) != settings_fingerprint(SPEC)
