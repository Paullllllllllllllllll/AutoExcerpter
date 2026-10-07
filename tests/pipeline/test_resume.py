"""Tests for pipeline/resume.py beside the resume-state matrix.

Covers the txt transcription format, completed-page parsing (versioned JSONL,
incomplete log recovery), load_results, a migrated CRLF log and long item
names. The classification of on-disk states lives in test_resume_matrix.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.constants import LOG_FORMAT_VERSION
from autoexcerpter.pipeline.log import append_to_log, finalize_log_file, read_log
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.resume import (
    ProcessingState,
    ResumeChecker,
    load_results,
)
from tests.pipeline.helpers import create_log_with_entries, create_outputs, jsonl_log


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create and return a temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def item_name() -> str:
    """Return a standard test item name."""
    return "TestDocument"


@pytest.fixture
def checker_skip() -> ResumeChecker:
    """Create a ResumeChecker in skip mode with summarize enabled."""
    return ResumeChecker(
        resume_mode="skip",
        summarize=True,
        output_docx=True,
        output_markdown=True,
    )


def _create_file(path: Path, content: str = "content") -> None:
    """Helper to create a file with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_txt_format_reads_the_txt_transcription(
    output_dir: Path, item_name: str
) -> None:
    """With the txt format only ``X_transcription.txt`` counts."""
    checker = ResumeChecker("skip", summarize=False, transcription_format="txt")
    create_outputs(output_dir, item_name, ("transcription",))
    assert checker.should_skip(item_name, output_dir).state is ProcessingState.NONE

    paths = ItemPaths.for_item(item_name, output_dir, "txt")
    _create_file(paths.transcription)
    result = checker.should_skip(item_name, output_dir)
    assert result.state is ProcessingState.COMPLETE
    assert result.existing_outputs == [paths.transcription]


@pytest.mark.parametrize(
    ("summarized", "state"), [(1, "transcription_only"), (2, "complete")]
)
def test_a_summary_log_short_of_pages_is_not_complete(
    output_dir: Path,
    item_name: str,
    checker_skip: ResumeChecker,
    summarized: int,
    state: str,
) -> None:
    create_outputs(output_dir, item_name)
    pages = [{"original_input_order_index": i, "transcription": "t"} for i in (0, 1)]
    create_log_with_entries(output_dir, item_name, pages)
    paths = ItemPaths.for_item(item_name, output_dir)
    summaries = [{"original_input_order_index": i} for i in range(summarized)]
    paths.summary_log.write_text(jsonl_log(summaries, 2), encoding="utf-8")

    result = checker_skip.should_skip(item_name, output_dir)

    assert result.state.value == state


# ============================================================================
# Completed-page parsing Tests
# ============================================================================
def _completed(log_path: Path) -> set[int] | None:
    """Return the completed page indices parsed from a working log."""
    log = read_log(log_path)
    return log.completed_pages() if log is not None else None


def _jl(
    entries: list[dict[str, Any]], header_extra: dict[str, Any] | None = None
) -> str:
    """Build versioned JSONL log content (header line + one line per entry)."""
    header: dict[str, Any] = {
        "_format_version": LOG_FORMAT_VERSION,
        "input_item_name": "test",
    }
    if header_extra:
        header.update(header_extra)
    lines = [json.dumps(header)] + [json.dumps(e) for e in entries]
    return "\n".join(lines) + "\n"


class TestCompletedPagesFromLog:
    def test_valid_complete_log(self, tmp_path: Path) -> None:
        """Parse a valid, versioned JSONL log."""
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "a"},
            {"original_input_order_index": 1, "page": 2, "transcription": "b"},
            {"original_input_order_index": 2, "page": 3, "error": "fail"},
        ]
        log_path = tmp_path / "transcription.jsonl"
        log_path.write_text(_jl(entries), encoding="utf-8")

        result = _completed(log_path)
        assert result == {0, 1}  # index 2 has error, excluded

    def test_truncated_last_line_recovered(self, tmp_path: Path) -> None:
        """A crash truncating the final JSONL line drops only that line."""
        e1 = {"original_input_order_index": 0, "page": 1, "transcription": "a"}
        log_path = tmp_path / "transcription.jsonl"
        # Complete e1 line, then a torn partial line (no newline).
        log_path.write_text(
            _jl([e1]) + '{"original_input_order_index": 1, "transcr',
            encoding="utf-8",
        )

        result = _completed(log_path)
        assert result == {0}

    def test_interior_corrupt_line_skipped(self, tmp_path: Path) -> None:
        log_path = tmp_path / "transcription.jsonl"
        log_path.write_text(
            _jl([{"original_input_order_index": 0}]).rstrip("\n")
            + "\nnot-json\n"
            + json.dumps({"original_input_order_index": 2})
            + "\n",
            encoding="utf-8",
        )

        assert _completed(log_path) == {0, 2}

    def test_unversioned_log_refused(self, tmp_path: Path) -> None:
        """A JSON-array log without a version marker is refused."""
        header = {"input_item_name": "test"}
        e1 = {"original_input_order_index": 0, "page": 1, "transcription": "a"}
        log_path = tmp_path / "unversioned.json"
        log_path.write_text(json.dumps([header, e1]), encoding="utf-8")

        assert _completed(log_path) is None

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns None."""
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("", encoding="utf-8")
        assert _completed(log_path) is None

    def test_header_only(self, tmp_path: Path) -> None:
        """Log with only a header (no page entries) returns None."""
        log_path = tmp_path / "header_only.jsonl"
        log_path.write_text(_jl([]), encoding="utf-8")
        assert _completed(log_path) is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file returns None."""
        log_path = tmp_path / "does_not_exist.jsonl"
        assert _completed(log_path) is None

    def test_all_errors_returns_none(self, tmp_path: Path) -> None:
        """Log where all entries have errors returns None."""
        entries = [{"original_input_order_index": 0, "page": 1, "error": "fail"}]
        log_path = tmp_path / "errors.jsonl"
        log_path.write_text(_jl(entries), encoding="utf-8")
        assert _completed(log_path) is None

    def test_corrupted_json(self, tmp_path: Path) -> None:
        """Completely corrupted file returns None."""
        log_path = tmp_path / "corrupt.jsonl"
        log_path.write_text("not valid json at all {{{", encoding="utf-8")
        assert _completed(log_path) is None


# ============================================================================
# load_results Tests
# ============================================================================
class TestLoadResults:
    def test_returns_entries_excluding_header(self, tmp_path: Path) -> None:
        """Returns only entries with original_input_order_index (not header)."""
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "a"},
            {"original_input_order_index": 1, "page": 2, "transcription": "b"},
        ]
        log_path = tmp_path / "log.jsonl"
        log_path.write_text(_jl(entries), encoding="utf-8")

        result = load_results(log_path)
        assert result is not None
        assert len(result) == 2
        assert result[0]["transcription"] == "a"
        assert result[1]["transcription"] == "b"

    def test_includes_error_entries(self, tmp_path: Path) -> None:
        """Error entries are included (they are page results, just failed)."""
        entries = [{"original_input_order_index": 0, "page": 1, "error": "fail"}]
        log_path = tmp_path / "log.jsonl"
        log_path.write_text(_jl(entries), encoding="utf-8")

        result = load_results(log_path)
        assert result is not None
        assert len(result) == 1
        assert "error" in result[0]

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("", encoding="utf-8")
        assert load_results(log_path) is None

    def test_header_only_returns_none(self, tmp_path: Path) -> None:
        log_path = tmp_path / "header_only.jsonl"
        log_path.write_text(_jl([]), encoding="utf-8")
        assert load_results(log_path) is None

    def test_incomplete_log(self, tmp_path: Path) -> None:
        """Can recover entries from a log with a truncated final line."""
        e1 = {"original_input_order_index": 0, "page": 1, "transcription": "a"}
        log_path = tmp_path / "incomplete.jsonl"
        log_path.write_text(_jl([e1]) + '{"original_input_o', encoding="utf-8")

        result = load_results(log_path)
        assert result is not None
        assert len(result) == 1


def test_a_renamed_log_with_crlf_lines_resumes_and_takes_appends(
    checker_skip: ResumeChecker, output_dir: Path, item_name: str
) -> None:
    """A working log moved to ``X_autoexcerpter/`` by the migration resumes."""
    paths = ItemPaths.for_item(item_name, output_dir)
    paths.working_dir.mkdir(parents=True)
    entries = [
        {"original_input_order_index": i, "page": i + 1, "transcription": "t"}
        for i in range(2)
    ]
    crlf = jsonl_log(entries, total_images=3).replace("\n", "\r\n")
    paths.transcription_log.write_bytes(crlf.encode("utf-8"))

    result = checker_skip.should_skip(item_name, output_dir)
    assert result.state == ProcessingState.PARTIAL
    assert result.completed_page_indices == {0, 1}

    page = {"original_input_order_index": 2, "page": 3, "transcription": "t"}
    assert append_to_log(paths.transcription_log, page)
    finalize_log_file(paths.transcription_log)
    assert _completed(paths.transcription_log) == {0, 1, 2}


# ============================================================================
# Edge Cases
# ============================================================================
class TestEdgeCases:
    def test_long_item_name(
        self, checker_skip: ResumeChecker, output_dir: Path
    ) -> None:
        """Resume checker reads the outputs of a name that gets shortened."""
        long_name = "A" * 300
        create_outputs(output_dir, long_name)

        result = checker_skip.should_skip(long_name, output_dir)
        assert result.state == ProcessingState.COMPLETE

    def test_special_characters_in_name(
        self, checker_skip: ResumeChecker, output_dir: Path
    ) -> None:
        """Resume checker works with special characters in item names."""
        name = "Test (Doc) [2024] — v1.0"
        create_outputs(output_dir, name)

        result = checker_skip.should_skip(name, output_dir)
        assert result.state == ProcessingState.COMPLETE

    def test_partial_log_with_long_name(
        self, checker_skip: ResumeChecker, output_dir: Path
    ) -> None:
        """Partial log detection works with long item names."""
        long_name = "B" * 300
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "text"},
        ]
        create_log_with_entries(output_dir, long_name, entries)

        result = checker_skip.should_skip(long_name, output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0}
