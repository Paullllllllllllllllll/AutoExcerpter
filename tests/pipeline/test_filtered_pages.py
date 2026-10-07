"""A content-filtered page stays failed: resume retries it, summaries mark it."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.llm.summary import SummaryManager
from autoexcerpter.pipeline.log import (
    LogHeader,
    WorkingLog,
    append_to_log,
    initialize_log_file,
    read_log,
)
from tests.llm import helpers as llm
from tests.pipeline.helpers import make_runner, summarizer

_FILTERED_ENTRY: dict[str, Any] = {
    "image": "page_0002.png",
    "transcription": "[transcription error: content filter]",
    "error": "Response stopped by the provider's content filter",
    "error_type": "content_filter",
    "original_input_order_index": 1,
}


def test_completed_pages_skip_a_content_filter_entry() -> None:
    log = WorkingLog(
        {"_format_version": 2},
        [{"transcription": "ok", "original_input_order_index": 0}, _FILTERED_ENTRY],
    )
    assert log.completed_pages() == {0}


def test_completed_pages_from_log(tmp_path: Path) -> None:
    log_path = tmp_path / "transcription.jsonl"
    assert initialize_log_file(
        log_path, LogHeader("item", str(tmp_path / "in.pdf"), "PDF", 3, "gpt-5-mini")
    )
    append_to_log(log_path, {"transcription": "a", "original_input_order_index": 0})
    append_to_log(log_path, _FILTERED_ENTRY)
    append_to_log(log_path, {"transcription": "c", "original_input_order_index": 2})

    log = read_log(log_path)

    assert log is not None
    assert log.completed_pages() == {0, 2}


def test_the_summary_of_a_filtered_page_is_a_placeholder(tmp_path: Path) -> None:
    summary_manager = summarizer()
    runner = make_runner(tmp_path, summary_manager=summary_manager)

    result = asyncio.run(runner.summarize(dict(_FILTERED_ENTRY), 1, "page_0002.png"))

    assert result is not None
    summary_manager.generate_summary.assert_not_called()
    assert result["bullet_points"][0].startswith("[Transcription failed:")
    assert result["page"] == 2
    assert result["page_information"]["page_number_integer"] is None
    assert result["page_information"]["page_number_type"] == "none"


def test_a_filtered_summary_is_marked_for_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filtered = llm.message(
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
        }
    )
    llm.install(monkeypatch, [filtered])
    manager = SummaryManager(llm.phase(), llm.env())
    runner = make_runner(tmp_path, summary_manager=manager)
    entry = {"transcription": "Some text.", "original_input_order_index": 0}

    result = asyncio.run(runner.summarize(entry, 0, "page_0001.png"))

    assert result is not None
    assert "content filter" in result["error"]
    assert result["error_type"] == "content_filter"
