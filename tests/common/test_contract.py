"""Exit codes and the one-line JSON summary of the agent contract."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from autoexcerpter.common.contract import (
    USAGE_KEY,
    ExitCode,
    SummaryEmitter,
    exit_code_for,
    usage_field,
    write_json_line,
)
from autoexcerpter.common.usage import Usage

EMPTY = {
    "items_total": 0,
    "items_complete": 0,
    "items_failed": 0,
    "items_skipped": 0,
    "outputs": [],
}


def _lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_usage_field_sums_the_roles_under_total() -> None:
    field = usage_field(
        {
            "transcription": Usage(100, 20, 120, cached_tokens=10),
            "summary": Usage(50, 5, 55, reasoning_tokens=3),
        }
    )

    assert USAGE_KEY == "usage"
    assert list(field) == ["transcription", "summary", "total"]
    assert field["total"] == {
        "input_tokens": 150,
        "output_tokens": 25,
        "total_tokens": 175,
        "cached_tokens": 10,
        "reasoning_tokens": 3,
    }


def test_usage_field_without_usage_is_a_zero_total() -> None:
    assert usage_field() == {"total": Usage().as_dict()}
    with pytest.raises(ValueError, match="reserved"):
        usage_field({"total": Usage()})


def test_exit_code_values() -> None:
    assert [int(code) for code in ExitCode] == [0, 1, 2, 130]


@pytest.mark.parametrize(("failed", "expected"), [(0, 0), (1, 1), (5, 1)])
def test_exit_code_for_counts(failed: int, expected: int) -> None:
    assert exit_code_for(failed=failed) == expected


def test_summary_is_emitted_once_with_dry_run_first() -> None:
    stream = io.StringIO()
    emitter = SummaryEmitter(stream, dry_run=True)

    assert emitter.emit({**EMPTY, "items_total": 2}) is True
    assert emitter.emit(EMPTY) is False

    assert stream.getvalue().endswith("\n")
    [summary] = _lines(stream)
    assert list(summary)[0] == "dry_run"
    assert summary == {"dry_run": True, **EMPTY, "items_total": 2}
    assert emitter.emitted


def test_disabled_emitter_writes_nothing() -> None:
    emitter = SummaryEmitter(None)

    assert not emitter.enabled
    assert emitter.emit(EMPTY) is False
    assert not emitter.emitted


def test_non_ascii_is_written_verbatim_on_utf8() -> None:
    stream = io.StringIO()
    write_json_line(stream, {"outputs": ["C:/Bücher/Kochbuch.md"]})

    assert "Bücher" in stream.getvalue()


def test_legacy_code_page_falls_back_to_ascii_escapes() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", write_through=True)
    obj = {"outputs": ["C:/книги/本.md"]}

    write_json_line(stream, obj)

    text = raw.getvalue().decode("ascii")
    assert "\\u" in text
    assert text.count("\n") == 1
    assert json.loads(text) == obj


def test_late_fallback_after_summary_does_not_double_emit() -> None:
    stream = io.StringIO()
    emitter = SummaryEmitter(stream)
    emitter.emit({**EMPTY, "items_total": 3, "items_complete": 3})

    emitter.emit(EMPTY)

    [summary] = _lines(stream)
    assert summary["items_complete"] == 3
    assert summary["dry_run"] is False
