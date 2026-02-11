"""Tests for the resume-aware processing utilities in core/resume.py.

Covers:
- ProcessingState enum
- ResumeResult dataclass
- ResumeChecker (skip/overwrite modes, item-level and page-level detection)
- load_completed_pages (JSON array parsing, incomplete log recovery)
- load_transcription_results_from_log
- Integration with main.py's _resolve_item_output_dir, _display_resume_info, _parse_execution_mode
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock, patch

import pytest

from core.resume import (
    ProcessingState,
    ResumeChecker,
    ResumeResult,
    load_completed_pages,
    load_transcription_results_from_log,
    _parse_log_entries,
)
from modules.path_utils import create_safe_directory_name, create_safe_log_filename


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


@pytest.fixture
def checker_overwrite() -> ResumeChecker:
    """Create a ResumeChecker in overwrite mode."""
    return ResumeChecker(
        resume_mode="overwrite",
        summarize=True,
        output_docx=True,
        output_markdown=True,
    )


@pytest.fixture
def checker_no_summary() -> ResumeChecker:
    """Create a ResumeChecker in skip mode with summarize disabled."""
    return ResumeChecker(
        resume_mode="skip",
        summarize=False,
        output_docx=False,
        output_markdown=False,
    )


def _create_file(path: Path, content: str = "content") -> None:
    """Helper to create a file with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_log_with_entries(
    output_dir: Path,
    item_name: str,
    entries: List[Dict[str, Any]],
    finalize: bool = True,
) -> Path:
    """Create a transcription log file in the working directory."""
    safe_working_dir_name = create_safe_directory_name(item_name, "_working_files")
    working_dir = output_dir / safe_working_dir_name
    working_dir.mkdir(parents=True, exist_ok=True)

    safe_log_name = create_safe_log_filename(item_name, "transcription")
    log_path = working_dir / safe_log_name

    # Mimic the JSON array log format
    header = {
        "input_item_name": item_name,
        "input_item_path": f"/fake/{item_name}.pdf",
        "input_type": "PDF",
        "total_images": len(entries),
    }

    parts = [json.dumps(header)]
    for entry in entries:
        parts.append(json.dumps(entry))

    content = "[\n" + ",\n".join(parts)
    if finalize:
        content += "\n]"
    # else: leave the array unclosed (simulating a crash)

    log_path.write_text(content, encoding="utf-8")
    return log_path


# ============================================================================
# ProcessingState Tests
# ============================================================================
class TestProcessingState:
    def test_enum_values(self):
        assert ProcessingState.COMPLETE.value == "complete"
        assert ProcessingState.TRANSCRIPTION_ONLY.value == "transcription_only"
        assert ProcessingState.PARTIAL.value == "partial"
        assert ProcessingState.NONE.value == "none"

    def test_enum_members(self):
        assert len(ProcessingState) == 4


# ============================================================================
# ResumeResult Tests
# ============================================================================
class TestResumeResult:
    def test_default_fields(self):
        result = ResumeResult(item_name="test", state=ProcessingState.NONE)
        assert result.item_name == "test"
        assert result.state == ProcessingState.NONE
        assert result.output_dir is None
        assert result.existing_outputs == []
        assert result.missing_outputs == []
        assert result.reason == ""
        assert result.completed_page_indices is None

    def test_with_completed_pages(self):
        pages = {0, 1, 5}
        result = ResumeResult(
            item_name="test",
            state=ProcessingState.PARTIAL,
            completed_page_indices=pages,
        )
        assert result.completed_page_indices == {0, 1, 5}


# ============================================================================
# ResumeChecker — Overwrite Mode Tests
# ============================================================================
class TestResumeCheckerOverwrite:
    def test_should_skip_always_returns_none(
        self, checker_overwrite: ResumeChecker, output_dir: Path, item_name: str
    ):
        """In overwrite mode, should_skip always returns NONE regardless of existing files."""
        # Create all output files
        _create_file(output_dir / f"{item_name}.txt")
        _create_file(output_dir / f"{item_name}.docx")
        _create_file(output_dir / f"{item_name}.md")

        result = checker_overwrite.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.NONE
        assert result.reason == "overwrite mode"

    def test_filter_items_returns_all(self, checker_overwrite: ResumeChecker, output_dir: Path):
        """In overwrite mode, filter_items returns all items."""
        items = ["a", "b", "c"]
        to_process, skipped = checker_overwrite.filter_items(
            items=items,
            output_dir_func=lambda _: output_dir,
            name_func=lambda x: x,
        )
        assert to_process == ["a", "b", "c"]
        assert skipped == []


# ============================================================================
# ResumeChecker — Skip Mode: COMPLETE State Tests
# ============================================================================
class TestResumeCheckerComplete:
    def test_all_outputs_exist_with_summaries(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """Item is COMPLETE when .txt, .docx, and .md all exist."""
        _create_file(output_dir / f"{item_name}.txt")
        _create_file(output_dir / f"{item_name}.docx")
        _create_file(output_dir / f"{item_name}.md")

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.COMPLETE
        assert len(result.existing_outputs) == 3
        assert len(result.missing_outputs) == 0

    def test_all_outputs_exist_no_summary(
        self, checker_no_summary: ResumeChecker, output_dir: Path, item_name: str
    ):
        """Item is COMPLETE when only .txt exists and summarize is disabled."""
        _create_file(output_dir / f"{item_name}.txt")

        result = checker_no_summary.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.COMPLETE
        assert len(result.existing_outputs) == 1

    def test_docx_only_mode(self, output_dir: Path, item_name: str):
        """Item is COMPLETE when .txt and .docx exist (markdown disabled)."""
        checker = ResumeChecker(
            resume_mode="skip", summarize=True, output_docx=True, output_markdown=False
        )
        _create_file(output_dir / f"{item_name}.txt")
        _create_file(output_dir / f"{item_name}.docx")

        result = checker.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.COMPLETE

    def test_markdown_only_mode(self, output_dir: Path, item_name: str):
        """Item is COMPLETE when .txt and .md exist (docx disabled)."""
        checker = ResumeChecker(
            resume_mode="skip", summarize=True, output_docx=False, output_markdown=True
        )
        _create_file(output_dir / f"{item_name}.txt")
        _create_file(output_dir / f"{item_name}.md")

        result = checker.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.COMPLETE

    def test_empty_txt_not_complete(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """An empty .txt file does not count as complete."""
        _create_file(output_dir / f"{item_name}.txt", content="")
        _create_file(output_dir / f"{item_name}.docx")
        _create_file(output_dir / f"{item_name}.md")

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state != ProcessingState.COMPLETE


# ============================================================================
# ResumeChecker — Skip Mode: TRANSCRIPTION_ONLY State Tests
# ============================================================================
class TestResumeCheckerTranscriptionOnly:
    def test_txt_exists_but_summaries_missing(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """State is TRANSCRIPTION_ONLY when .txt exists but summary files are missing."""
        _create_file(output_dir / f"{item_name}.txt")

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.TRANSCRIPTION_ONLY
        assert len(result.missing_outputs) == 2  # .docx and .md

    def test_txt_and_docx_exist_md_missing(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """State is TRANSCRIPTION_ONLY when .txt and .docx exist but .md is missing."""
        _create_file(output_dir / f"{item_name}.txt")
        _create_file(output_dir / f"{item_name}.docx")

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.TRANSCRIPTION_ONLY
        assert len(result.missing_outputs) == 1


# ============================================================================
# ResumeChecker — Skip Mode: PARTIAL State Tests
# ============================================================================
class TestResumeCheckerPartial:
    def test_partial_log_detected(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """State is PARTIAL when a transcription log with completed entries exists."""
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "text"},
            {"original_input_order_index": 1, "page": 2, "transcription": "text"},
        ]
        _create_log_with_entries(output_dir, item_name, entries)

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0, 1}

    def test_partial_log_with_errors(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """Entries with errors are excluded from completed page indices."""
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "text"},
            {"original_input_order_index": 1, "page": 2, "error": "API timeout"},
            {"original_input_order_index": 2, "page": 3, "transcription": "text"},
        ]
        _create_log_with_entries(output_dir, item_name, entries)

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0, 2}

    def test_incomplete_log_recovered(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """Incomplete JSON array (no closing bracket) is still parsed for page-level resume."""
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "text"},
        ]
        _create_log_with_entries(output_dir, item_name, entries, finalize=False)

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0}


# ============================================================================
# ResumeChecker — Skip Mode: NONE State Tests
# ============================================================================
class TestResumeCheckerNone:
    def test_no_outputs_at_all(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """State is NONE when no output files and no log exist."""
        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.NONE

    def test_empty_log_returns_none(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """An empty log file means NONE state."""
        safe_working_dir_name = create_safe_directory_name(item_name, "_working_files")
        working_dir = output_dir / safe_working_dir_name
        working_dir.mkdir(parents=True)
        safe_log_name = create_safe_log_filename(item_name, "transcription")
        log_path = working_dir / safe_log_name
        log_path.write_text("", encoding="utf-8")

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.NONE

    def test_log_with_only_header_returns_none(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """A log with only the header entry (no page results) means NONE state."""
        _create_log_with_entries(output_dir, item_name, entries=[])

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.NONE

    def test_log_with_only_errors_returns_none(
        self, checker_skip: ResumeChecker, output_dir: Path, item_name: str
    ):
        """A log where all entries have errors means NONE state (no completed pages)."""
        entries = [
            {"original_input_order_index": 0, "page": 1, "error": "timeout"},
            {"original_input_order_index": 1, "page": 2, "error": "rate limit"},
        ]
        _create_log_with_entries(output_dir, item_name, entries)

        result = checker_skip.should_skip(item_name, output_dir)
        assert result.state == ProcessingState.NONE


# ============================================================================
# ResumeChecker — filter_items Tests
# ============================================================================
class TestResumeCheckerFilterItems:
    def test_complete_items_are_skipped(
        self, checker_skip: ResumeChecker, output_dir: Path
    ):
        """Complete items are moved to skipped list."""
        # Create complete outputs for item "A"
        _create_file(output_dir / "A.txt")
        _create_file(output_dir / "A.docx")
        _create_file(output_dir / "A.md")

        items = ["A", "B"]
        to_process, skipped = checker_skip.filter_items(
            items=items,
            output_dir_func=lambda _: output_dir,
            name_func=lambda x: x,
        )
        assert to_process == ["B"]
        assert len(skipped) == 1
        assert skipped[0].item_name == "A"
        assert skipped[0].state == ProcessingState.COMPLETE

    def test_partial_items_are_not_skipped(
        self, checker_skip: ResumeChecker, output_dir: Path
    ):
        """Partial items remain in the to_process list (they need processing)."""
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "text"},
        ]
        _create_log_with_entries(output_dir, "PartialDoc", entries)

        items = ["PartialDoc"]
        to_process, skipped = checker_skip.filter_items(
            items=items,
            output_dir_func=lambda _: output_dir,
            name_func=lambda x: x,
        )
        assert to_process == ["PartialDoc"]
        assert skipped == []

    def test_mixed_states(self, checker_skip: ResumeChecker, output_dir: Path):
        """Mix of COMPLETE, PARTIAL, and NONE items are correctly partitioned."""
        # Complete: "Done"
        _create_file(output_dir / "Done.txt")
        _create_file(output_dir / "Done.docx")
        _create_file(output_dir / "Done.md")
        # Partial: "Half"
        _create_log_with_entries(
            output_dir, "Half",
            [{"original_input_order_index": 0, "page": 1, "transcription": "t"}],
        )
        # None: "New"

        items = ["Done", "Half", "New"]
        to_process, skipped = checker_skip.filter_items(
            items=items,
            output_dir_func=lambda _: output_dir,
            name_func=lambda x: x,
        )
        assert set(to_process) == {"Half", "New"}
        assert len(skipped) == 1
        assert skipped[0].item_name == "Done"


# ============================================================================
# load_completed_pages Tests
# ============================================================================
class TestLoadCompletedPages:
    def test_valid_complete_log(self, tmp_path: Path):
        """Parse a valid, finalized JSON log."""
        header = {"input_item_name": "test"}
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "a"},
            {"original_input_order_index": 1, "page": 2, "transcription": "b"},
            {"original_input_order_index": 2, "page": 3, "error": "fail"},
        ]
        log_path = tmp_path / "test_log.json"
        log_path.write_text(
            json.dumps([header] + entries), encoding="utf-8"
        )

        result = load_completed_pages(log_path)
        assert result == {0, 1}  # index 2 has error, excluded

    def test_incomplete_json_array(self, tmp_path: Path):
        """Recover from incomplete JSON array (no closing bracket)."""
        header = {"input_item_name": "test"}
        e1 = {"original_input_order_index": 0, "page": 1, "transcription": "a"}
        log_path = tmp_path / "test_log.json"
        log_path.write_text(
            "[\n" + json.dumps(header) + ",\n" + json.dumps(e1),
            encoding="utf-8",
        )

        result = load_completed_pages(log_path)
        assert result == {0}

    def test_incomplete_with_trailing_comma(self, tmp_path: Path):
        """Recover from incomplete JSON array with trailing comma."""
        header = {"input_item_name": "test"}
        e1 = {"original_input_order_index": 0, "page": 1, "transcription": "a"}
        log_path = tmp_path / "test_log.json"
        log_path.write_text(
            "[\n" + json.dumps(header) + ",\n" + json.dumps(e1) + ",",
            encoding="utf-8",
        )

        result = load_completed_pages(log_path)
        assert result == {0}

    def test_empty_file(self, tmp_path: Path):
        """Empty file returns None."""
        log_path = tmp_path / "empty.json"
        log_path.write_text("", encoding="utf-8")
        assert load_completed_pages(log_path) is None

    def test_header_only(self, tmp_path: Path):
        """Log with only a header (no page entries) returns None."""
        header = {"input_item_name": "test"}
        log_path = tmp_path / "header_only.json"
        log_path.write_text(json.dumps([header]), encoding="utf-8")
        assert load_completed_pages(log_path) is None

    def test_nonexistent_file(self, tmp_path: Path):
        """Nonexistent file returns None."""
        log_path = tmp_path / "does_not_exist.json"
        assert load_completed_pages(log_path) is None

    def test_all_errors_returns_none(self, tmp_path: Path):
        """Log where all entries have errors returns None."""
        header = {"input_item_name": "test"}
        entries = [
            {"original_input_order_index": 0, "page": 1, "error": "fail"},
        ]
        log_path = tmp_path / "errors.json"
        log_path.write_text(json.dumps([header] + entries), encoding="utf-8")
        assert load_completed_pages(log_path) is None

    def test_corrupted_json(self, tmp_path: Path):
        """Completely corrupted file returns None."""
        log_path = tmp_path / "corrupt.json"
        log_path.write_text("not valid json at all {{{", encoding="utf-8")
        assert load_completed_pages(log_path) is None


# ============================================================================
# load_transcription_results_from_log Tests
# ============================================================================
class TestLoadTranscriptionResultsFromLog:
    def test_returns_entries_excluding_header(self, tmp_path: Path):
        """Returns only entries with original_input_order_index (not header)."""
        header = {"input_item_name": "test"}
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "a"},
            {"original_input_order_index": 1, "page": 2, "transcription": "b"},
        ]
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps([header] + entries), encoding="utf-8")

        result = load_transcription_results_from_log(log_path)
        assert result is not None
        assert len(result) == 2
        assert result[0]["transcription"] == "a"
        assert result[1]["transcription"] == "b"

    def test_includes_error_entries(self, tmp_path: Path):
        """Error entries are included (they are page results, just failed)."""
        header = {"input_item_name": "test"}
        entries = [
            {"original_input_order_index": 0, "page": 1, "error": "fail"},
        ]
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps([header] + entries), encoding="utf-8")

        result = load_transcription_results_from_log(log_path)
        assert result is not None
        assert len(result) == 1
        assert "error" in result[0]

    def test_empty_file_returns_none(self, tmp_path: Path):
        log_path = tmp_path / "empty.json"
        log_path.write_text("", encoding="utf-8")
        assert load_transcription_results_from_log(log_path) is None

    def test_header_only_returns_none(self, tmp_path: Path):
        header = {"input_item_name": "test"}
        log_path = tmp_path / "header_only.json"
        log_path.write_text(json.dumps([header]), encoding="utf-8")
        assert load_transcription_results_from_log(log_path) is None

    def test_incomplete_log(self, tmp_path: Path):
        """Can recover entries from an incomplete log."""
        header = {"input_item_name": "test"}
        e1 = {"original_input_order_index": 0, "page": 1, "transcription": "a"}
        log_path = tmp_path / "incomplete.json"
        log_path.write_text(
            "[\n" + json.dumps(header) + ",\n" + json.dumps(e1),
            encoding="utf-8",
        )

        result = load_transcription_results_from_log(log_path)
        assert result is not None
        assert len(result) == 1


# ============================================================================
# _parse_log_entries Tests
# ============================================================================
class TestParseLogEntries:
    def test_valid_json_array(self):
        raw = json.dumps([{"a": 1}, {"b": 2}])
        result = _parse_log_entries(raw)
        assert result is not None
        assert len(result) == 2

    def test_not_a_list(self):
        raw = json.dumps({"key": "value"})
        result = _parse_log_entries(raw)
        assert result is None

    def test_empty_string(self):
        result = _parse_log_entries("")
        assert result is None

    def test_missing_closing_bracket(self):
        raw = '[{"a": 1},{"b": 2}'
        result = _parse_log_entries(raw)
        assert result is not None
        assert len(result) == 2

    def test_trailing_comma_and_missing_bracket(self):
        raw = '[{"a": 1},'
        result = _parse_log_entries(raw)
        assert result is not None
        assert len(result) == 1

    def test_completely_invalid(self):
        result = _parse_log_entries("{{not json}}")
        assert result is None


# ============================================================================
# Integration: main.py helper function tests
# ============================================================================
class TestMainIntegration:
    """Test main.py functions that interact with the resume system."""

    def test_parse_execution_mode_default_resume(self):
        """Default resume mode is 'skip'."""
        from main import _parse_execution_mode

        args = MagicMock()
        args.force = None
        args.resume = None
        args.input = "test_input"
        args.output = "test_output"
        args.all = False
        args.select = None
        args.context = None

        with patch("main.config") as mock_config:
            mock_config.CLI_MODE = True
            result = _parse_execution_mode(args)

        assert result[-1] == "skip"  # resume_mode is last element

    def test_parse_execution_mode_force(self):
        """--force flag sets resume_mode to 'overwrite'."""
        from main import _parse_execution_mode

        args = MagicMock()
        args.force = True
        args.resume = None
        args.input = "test_input"
        args.output = "test_output"
        args.all = False
        args.select = None
        args.context = None

        with patch("main.config") as mock_config:
            mock_config.CLI_MODE = True
            result = _parse_execution_mode(args)

        assert result[-1] == "overwrite"

    def test_display_resume_info_no_skipped(self):
        """_display_resume_info returns True when no items are skipped."""
        from main import _display_resume_info

        result = _display_resume_info("skip", [], [], [])
        assert result is True

    def test_display_resume_info_all_skipped_cli(self):
        """_display_resume_info returns False in CLI mode when all items are skipped."""
        from main import _display_resume_info

        skipped = [ResumeResult(item_name="A", state=ProcessingState.COMPLETE)]
        items = [MagicMock()]

        with patch("main.config") as mock_config:
            mock_config.CLI_MODE = True
            result = _display_resume_info("skip", items, skipped, [])

        assert result is False

    def test_resolve_item_output_dir_colocated(self, tmp_path: Path):
        """_resolve_item_output_dir returns input parent when INPUT_PATHS_IS_OUTPUT_PATH is True."""
        from main import _resolve_item_output_dir

        item = MagicMock()
        item.path = tmp_path / "subdir" / "test.pdf"
        item.path.parent.mkdir(parents=True, exist_ok=True)

        with patch("main.config") as mock_config:
            mock_config.INPUT_PATHS_IS_OUTPUT_PATH = True
            result = _resolve_item_output_dir(item, tmp_path / "output")

        assert result == tmp_path / "subdir"

    def test_resolve_item_output_dir_separate(self, tmp_path: Path):
        """_resolve_item_output_dir returns base_output_dir when INPUT_PATHS_IS_OUTPUT_PATH is False."""
        from main import _resolve_item_output_dir

        item = MagicMock()
        item.path = tmp_path / "subdir" / "test.pdf"
        base_out = tmp_path / "output"

        with patch("main.config") as mock_config:
            mock_config.INPUT_PATHS_IS_OUTPUT_PATH = False
            result = _resolve_item_output_dir(item, base_out)

        assert result == base_out


# ============================================================================
# Edge Cases
# ============================================================================
class TestEdgeCases:
    def test_long_item_name(self, checker_skip: ResumeChecker, output_dir: Path):
        """Resume checker works with long item names that trigger safe-name truncation."""
        long_name = "A" * 100
        _create_file(output_dir / f"{long_name}.txt")
        _create_file(output_dir / f"{long_name}.docx")
        _create_file(output_dir / f"{long_name}.md")

        result = checker_skip.should_skip(long_name, output_dir)
        assert result.state == ProcessingState.COMPLETE

    def test_special_characters_in_name(
        self, checker_skip: ResumeChecker, output_dir: Path
    ):
        """Resume checker works with special characters in item names."""
        name = "Test (Doc) [2024] — v1.0"
        _create_file(output_dir / f"{name}.txt")
        _create_file(output_dir / f"{name}.docx")
        _create_file(output_dir / f"{name}.md")

        result = checker_skip.should_skip(name, output_dir)
        assert result.state == ProcessingState.COMPLETE

    def test_partial_log_with_long_name(
        self, checker_skip: ResumeChecker, output_dir: Path
    ):
        """Partial log detection works with long item names."""
        long_name = "B" * 30
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "text"},
        ]
        _create_log_with_entries(output_dir, long_name, entries)

        result = checker_skip.should_skip(long_name, output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0}

    def test_filter_items_with_custom_objects(
        self, checker_skip: ResumeChecker, output_dir: Path
    ):
        """filter_items works with arbitrary objects via name_func and output_dir_func."""
        # Create complete output for item with stem "Alpha"
        _create_file(output_dir / "Alpha.txt")
        _create_file(output_dir / "Alpha.docx")
        _create_file(output_dir / "Alpha.md")

        class FakeItem:
            def __init__(self, stem: str):
                self.stem = stem

        items = [FakeItem("Alpha"), FakeItem("Beta")]
        to_process, skipped = checker_skip.filter_items(
            items=items,
            output_dir_func=lambda _: output_dir,
            name_func=lambda x: x.stem,
        )
        assert len(to_process) == 1
        assert to_process[0].stem == "Beta"
        assert len(skipped) == 1
        assert skipped[0].item_name == "Alpha"
