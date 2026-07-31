"""Tests for scripts/repair_layout - deterministic line-break repair."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.repair_layout import run_repair
from scripts.repair_layout.repair import is_passthrough_line, repair_text
from scripts.repair_layout.verifier import content_signature, page_markers, verify

# A page whose lines cluster just below a 62-character wrap ceiling, so the
# width estimator recognizes it as wrapped and the rejoin path actually runs.
_WRAPPED_PAGE = (
    "# Transcription of: test\n\n"
    "<page_number>1</page_number>\n"
    "This is a fairly long first line of prose that fills near the\n"
    "width here.\n"
    "Another long line of prose that also reaches close to the same\n"
    "limit.\n"
    "A third long prose line approaching the right hand margin now\n"
    "indeed.\n"
)
_WRAPPED_TAIL = (
    "More prose on the following page that runs to the same width\n"
    "again.\n"
    "Yet another long prose line reaching close to the right margin\n"
    "here.\n"
)


class TestContentSignature:
    """Tests for the content-preservation signature."""

    def test_ignores_whitespace(self) -> None:
        """Reflowing whitespace does not change the signature."""
        assert content_signature("hello world") == content_signature("hello\nworld")

    def test_ignores_line_break_hyphen(self) -> None:
        """De-hyphenation does not change the signature."""
        assert content_signature("Man-\nagement") == content_signature("Management")

    def test_detects_word_change(self) -> None:
        """A changed word changes the signature."""
        assert content_signature("the cat sat") != content_signature("the dog sat")

    def test_detects_dropped_punctuation(self) -> None:
        """Dropped punctuation changes the signature (dashes are content)."""
        assert content_signature("a—b") != content_signature("ab")

    def test_folds_ligature(self) -> None:
        """A ligature folds to its expansion."""
        assert content_signature("ﬁrm") == content_signature("firm")


class TestVerify:
    """Tests for the verification gate."""

    def test_passes_whitespace_reflow(self) -> None:
        """A pure reflow (joining lines) passes the gate."""
        result = verify("a b\nc d e", "a b c d e")
        assert result.passed

    def test_fails_content_change(self) -> None:
        """A content change fails the gate."""
        result = verify("the cat sat", "the dog sat")
        assert not result.passed
        assert not result.signature_ok

    def test_allows_reassembled_marker(self) -> None:
        """Gaining a marker by reassembling a split one is allowed."""
        original = "x <page_number>8\n3</page_number> y"
        repaired = "x <page_number>83</page_number> y"
        result = verify(original, repaired)
        assert result.passed
        assert result.markers_ok

    def test_fails_on_increased_line_count(self) -> None:
        """The repair must never increase the line count."""
        result = verify("one line here", "one\nline\nhere")
        assert not result.lines_ok
        assert not result.passed


class TestIsPassthroughLine:
    """Tests for structural-line detection."""

    def test_heading(self) -> None:
        assert is_passthrough_line("# Preface")

    def test_metadata_header(self) -> None:
        assert is_passthrough_line("# Source Path: C:/x/y.pdf")

    def test_page_marker(self) -> None:
        assert is_passthrough_line("<page_number>9</page_number>")

    def test_table_row(self) -> None:
        assert is_passthrough_line("| a | b | c |")

    def test_prose_is_not_passthrough(self) -> None:
        assert not is_passthrough_line("This is ordinary prose.")

    def test_split_marker_closing_half(self) -> None:
        """The closing half of a marker split by wrapping is not prose."""
        assert is_passthrough_line("3</page_number>")

    def test_trailing_inline_marker(self) -> None:
        """A marker at the end of a prose line still marks a boundary."""
        assert is_passthrough_line("Closing sentence. <page_number>12</page_number>")


class TestRepairText:
    """End-to-end tests for the repair."""

    def _wrapped_page(self) -> str:
        return _WRAPPED_PAGE

    def test_rejoins_orphan_remainder(self) -> None:
        """A short orphan remainder is rejoined onto its printed line."""
        repaired, _ = repair_text(self._wrapped_page())
        assert "near the width here." in repaired
        assert "\nwidth here." not in repaired

    def test_rejoin_preserves_content(self) -> None:
        """The rejoin passes the content gate."""
        text = self._wrapped_page()
        repaired, _ = repair_text(text)
        assert verify(text, repaired).passed

    def test_merges_hyphenation(self) -> None:
        """Ordinary line-break hyphenation is merged, keeping the break."""
        repaired, audit = repair_text("the concep-\ntion of value here\n")
        assert "conception" in repaired
        assert any(not d.kept for d in audit.hyphen_decisions)

    def test_keeps_compound_hyphen(self) -> None:
        """A genuine compound keeps its hyphen."""
        repaired, audit = repair_text("the co-\nordinate system used\n")
        assert "co-ordinate" in repaired
        assert any(d.kept for d in audit.hyphen_decisions)

    def test_preserves_structure(self) -> None:
        """Headings and page markers survive verbatim."""
        text = "# Transcription of: t\n\n# Preface\n\n<page_number>5</page_number>\n"
        repaired, _ = repair_text(text)
        assert "# Preface" in repaired
        assert "<page_number>5</page_number>" in repaired

    def test_does_not_glue_next_line_to_hyphen(self) -> None:
        """A hyphen-ending line never space-joins the next line ('func- tions')."""
        repaired, _ = repair_text("the func-\ntions are clear and useful here\n")
        assert "func- tions" not in repaired
        assert "functions" in repaired

    def test_standalone_split_marker_preserved(self) -> None:
        """A standalone split marker is a passthrough block: content preserved."""
        text = _WRAPPED_PAGE + "<page_number>8\n3</page_number>\n" + _WRAPPED_TAIL
        repaired, _ = repair_text(text)
        # The fixture is long enough for the width estimator to fire, so the
        # rejoin path really runs over the split marker rather than skipping it.
        assert "near the width here." in repaired
        assert verify(text, repaired).passed
        # The split marker stays unmatchable rather than being reassembled.
        assert "83" not in page_markers(text)
        assert page_markers(repaired) == page_markers(text)
        lines = repaired.split("\n")
        assert "<page_number>8" in lines
        assert "3</page_number>" in lines

    def test_inline_split_marker_halves_not_rejoined(self) -> None:
        """A marker split off a prose line is never space-joined back together.

        Rejoining would inject a space inside the marker
        ("<page_number>8 3</page_number>"), which the whitespace-blind content
        gate cannot detect.
        """
        text = (
            _WRAPPED_PAGE
            + "Prose that runs right up to the wrapped marker <page_number>8\n"
            + "3</page_number>\n"
            + _WRAPPED_TAIL
        )
        repaired, _ = repair_text(text)
        assert "<page_number>8 3</page_number>" not in repaired
        assert "3</page_number>" in repaired.split("\n")

    def test_spaceless_short_line_is_not_glued(self) -> None:
        """A short space-less line (URL) keeps a space when a remainder joins.

        The whitespace-blind gate cannot see word-gluing, so this is pinned
        here rather than in the verifier.
        """
        text = (
            _WRAPPED_PAGE
            + "https://example.org/very/long/path/to/a/document/identifier\n"
            + "Figure 3.\n"
        )
        repaired, _ = repair_text(text)
        assert "identifierFigure" not in repaired
        assert "identifier Figure 3." in repaired

    def test_width_filling_token_rejoins_without_space(self) -> None:
        """A long token broken *at* the wrap width still rejoins seamlessly."""
        token = "https://example.org/" + "x" * 42  # exactly the 62-char width
        text = _WRAPPED_PAGE + token + "\n0123.pdf\n"
        repaired, _ = repair_text(text)
        assert token + "0123.pdf" in repaired

    def test_dehyphenation_preserves_continuation_indent(self) -> None:
        """The continuation line keeps its own leading whitespace."""
        repaired, _ = repair_text("the concep-\n    tion of value here\n")
        assert "the conception" in repaired
        assert "\n    of value here" in repaired


class TestRunRepairIO:
    """Tests for the orchestrator's file I/O and run-level error handling."""

    def _write_target(self, path: Path, *, bom: bool = False) -> None:
        """Write a repairable transcription file, optionally BOM-prefixed."""
        prefix = "\ufeff" if bom else ""
        path.write_text(prefix + _WRAPPED_PAGE, encoding="utf-8", newline="")

    def test_bom_is_stripped_on_read_and_restored_on_write(
        self, tmp_path: Path
    ) -> None:
        """A leading BOM never reaches the repair, and survives the write."""
        target = tmp_path / "bom.txt"
        self._write_target(target, bom=True)

        text, had_crlf, had_bom = run_repair.read_text_preserve(target)
        assert had_bom
        assert not had_crlf
        assert text.startswith("# Transcription of:")

        result = run_repair.process_file(target, tmp_path, dry_run=False)
        assert result.written
        raw = target.read_text(encoding="utf-8", newline="")
        assert raw.startswith("\ufeff# Transcription of:")
        assert "near the width here." in raw

    def test_main_reports_unreadable_file_and_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One undecodable file does not abort the run or lose the reports."""
        self._write_target(tmp_path / "good.txt")
        bad = tmp_path / "bad.txt"
        # Valid UTF-8 header (so find_targets accepts it) with invalid bytes
        # far beyond the text-reader's first chunk.
        bad.write_bytes(b"# Transcription of: bad\n\n" + b"a" * 20000 + b"\xff\n")

        monkeypatch.setattr(sys, "argv", ["run_repair", "--root", str(tmp_path)])
        assert run_repair.main() == 1

        report = (tmp_path / "backup" / "repair_report.md").read_text(encoding="utf-8")
        assert "bad.txt" in report
        assert "SKIPPED" in report
        # The healthy file was still repaired despite the failure.
        assert "near the width here." in (tmp_path / "good.txt").read_text(
            encoding="utf-8"
        )

    def test_main_rejects_nonexistent_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A --root that is not a directory is a usage error, not a crash."""
        missing = tmp_path / "nope"
        monkeypatch.setattr(sys, "argv", ["run_repair", "--root", str(missing)])
        with pytest.raises(SystemExit) as excinfo:
            run_repair.main()
        assert excinfo.value.code == 2

    def test_main_skips_backup_when_no_targets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty tree writes no zip and no misleading backup_doc line."""
        monkeypatch.setattr(sys, "argv", ["run_repair", "--root", str(tmp_path)])
        assert run_repair.main() == 0
        backup_dir = tmp_path / "backup"
        assert not list(backup_dir.glob("*.zip"))
        assert not (backup_dir / "backup_doc.md").exists()

    def test_backup_doc_notes_limited_run(self, tmp_path: Path) -> None:
        """A truncated run says so, instead of implying a full-tree snapshot."""
        target = tmp_path / "one.txt"
        self._write_target(target)
        backup_dir = tmp_path / "backup"

        run_repair.create_backup(
            [target], tmp_path, backup_dir, "01_01_2026", total_candidates=3
        )

        doc = (backup_dir / "backup_doc.md").read_text(encoding="utf-8")
        assert "limited run: first 1 of 3 candidates" in doc
