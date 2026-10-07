"""A scanned PDF under the default ``target_dpi: native`` for OpenAI.

Each page of the input holds one page-spanning grayscale scan at 144 DPI, so
native DPI resolution renders the page from its scan density instead of
``native_fallback_dpi``. Goldens live under ``golden/inputs/pdf_scanned_native/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM
from tests.characterization.inputs import make_pdf

Golden = Callable[[str, RunResult], None]

PAGES = 3
PAGE_SIZE = (200.0, 300.0)
SCAN_SIZE = (400, 600)
SCAN_DPI = 144.0


def _log_entries(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [record for record in records if "_format_version" not in record]


def test_pdf_scanned_native_dpi(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    from autoexcerpter.imaging.settings import IMAGE_PROCESSING

    assert IMAGE_PROCESSING["api_image_processing"]["target_dpi"] == "native"

    pdf = make_pdf(
        docs_dir / "doc.pdf", pages=PAGES, page_size=PAGE_SIZE, scan_size=SCAN_SIZE
    )

    result = run_tool(pdf, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out")

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 1
    calls = result.calls(TRANSCRIPTION)
    assert [call["page"] for call in calls] == list(range(PAGES))
    assert [call["page"] for call in result.calls(SUMMARY)] == list(range(PAGES))
    sizes = {
        (block["width"], block["height"])
        for call in calls
        for block in call["messages"][1]["blocks"]
    }
    assert sizes == {SCAN_SIZE}

    log = result.paths("doc").transcription_log
    assert log in result.created
    entries = _log_entries(log)
    assert len(entries) == PAGES
    for entry in entries:
        image = entry["image_provenance"]
        assert image["dpi_source"] == "image"
        assert image["source_dpi_x"] == SCAN_DPI
        assert image["source_dpi_y"] == SCAN_DPI
    golden("inputs/pdf_scanned_native", result)
