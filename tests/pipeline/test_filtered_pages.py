"""A content-filtered page stays failed and gets a placeholder summary.

Resume retries a failed page (see tests/output/test_ae_resume.py); here the
page runner's summaries of such a page are checked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.llm.summary import SummaryManager
from tests.llm import helpers as llm
from tests.pipeline.helpers import make_runner, summarizer

_FILTERED_ENTRY: dict[str, Any] = {
    "image": "page_0002.png",
    "transcription": "[transcription error: content filter]",
    "error": "Response stopped by the provider's content filter",
    "error_type": "content_filter",
    "original_input_order_index": 1,
}


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
