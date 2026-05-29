"""Tests for scripts/repair_layout - deterministic line-break repair."""

from __future__ import annotations

from scripts.repair_layout.repair import is_passthrough_line, repair_text
from scripts.repair_layout.verifier import content_signature, page_markers, verify


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


class TestRepairText:
    """End-to-end tests for the repair."""

    def _wrapped_page(self) -> str:
        return (
            "# Transcription of: test\n\n"
            "<page_number>1</page_number>\n"
            "This is a fairly long first line of prose that fills near the\n"
            "width here.\n"
            "Another long line of prose that also reaches close to the same\n"
            "limit.\n"
            "A third long prose line approaching the right hand margin now\n"
            "indeed.\n"
        )

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
        text = (
            "# Transcription of: t\n\n"
            "<page_number>8\n"
            "3</page_number>\n"
            "body text follows here\n"
        )
        repaired, _ = repair_text(text)
        # Standalone marker lines pass through; the content gate still holds.
        assert verify(text, repaired).passed
        assert not page_markers(text)  # original split marker is unmatchable
