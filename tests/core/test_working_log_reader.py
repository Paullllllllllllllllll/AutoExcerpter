"""The tolerant reader of version 1.0 working logs that migrate imports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoexcerpter.migrate import WorkingLogError, read_working_log

HEADER = json.dumps({"_format_version": 2, "item_name": "doc.pdf"})


def _page(index: int) -> str:
    return json.dumps({"original_input_order_index": index, "transcription": "x"})


def test_a_truncated_tail_is_dropped_and_bad_interior_lines_skipped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcription.jsonl"
    lines = [HEADER, _page(0), "{not json", '{"note": 1}', _page(1), '{"original_']
    path.write_text("\n".join(lines), encoding="utf-8")

    log = read_working_log(path)

    assert log is not None
    assert log.header["item_name"] == "doc.pdf"
    assert [r["original_input_order_index"] for r in log.results] == [0, 1]
    assert log.dropped_tail
    assert log.skipped_lines == (3,)


def test_a_missing_or_blank_log_reads_as_none(tmp_path: Path) -> None:
    blank = tmp_path / "summary.jsonl"
    blank.write_text("\n \n", encoding="utf-8")

    assert read_working_log(tmp_path / "absent.jsonl") is None
    assert read_working_log(blank) is None


@pytest.mark.parametrize("first", ['{"_format_version": 1}', "not json", "[]"])
def test_a_log_without_the_version_2_header_is_refused(
    tmp_path: Path, first: str
) -> None:
    path = tmp_path / "transcription.jsonl"
    path.write_text(f"{first}\n{_page(0)}\n", encoding="utf-8")

    with pytest.raises(WorkingLogError):
        read_working_log(path)
