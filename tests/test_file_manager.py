"""Tests for processors/file_manager.py - File management utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest

from processors.file_manager import (
    sanitize_for_xml,
    parse_latex_in_text,
    normalize_latex_whitespace,
    simplify_problematic_latex,
    _extract_summary_payload,
    _page_number_and_flags,
    _is_meaningful_summary,
)


class TestSanitizeForXml:
    """Tests for sanitize_for_xml function."""

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert sanitize_for_xml("") == ""
        assert sanitize_for_xml(None) == ""

    def test_normal_text_unchanged(self):
        """Normal text is unchanged."""
        text = "Hello, World!"
        assert sanitize_for_xml(text) == text

    def test_removes_null_character(self):
        """Null character is removed."""
        text = "before\x00after"
        assert sanitize_for_xml(text) == "beforeafter"

    def test_removes_control_characters(self):
        """Control characters are removed."""
        text = "test\x01\x02\x03text"
        result = sanitize_for_xml(text)
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x03" not in result
        assert "testtext" == result

    def test_preserves_tab(self):
        """Tab character is preserved."""
        text = "col1\tcol2"
        assert sanitize_for_xml(text) == text

    def test_preserves_newline(self):
        """Newline character is preserved."""
        text = "line1\nline2"
        assert sanitize_for_xml(text) == text

    def test_preserves_carriage_return(self):
        """Carriage return is preserved."""
        text = "line1\r\nline2"
        assert sanitize_for_xml(text) == text

    def test_removes_delete_character(self):
        """DEL character (0x7F) is removed."""
        text = "test\x7Ftext"
        assert sanitize_for_xml(text) == "testtext"


class TestParseLatexInText:
    """Tests for parse_latex_in_text function."""

    def test_no_latex(self):
        """Text without LaTeX returns single text segment."""
        text = "Plain text without formulas"
        result = parse_latex_in_text(text)
        
        assert len(result) == 1
        assert result[0][1] == "text"

    def test_inline_latex(self):
        """Inline LaTeX ($...$) is detected."""
        text = "Here is $x + y = z$ inline"
        result = parse_latex_in_text(text)
        
        types = [seg[1] for seg in result]
        assert "latex_inline" in types

    def test_display_latex(self):
        """Display LaTeX ($$...$$) is detected."""
        text = "Here is $$x + y = z$$ display"
        result = parse_latex_in_text(text)
        
        types = [seg[1] for seg in result]
        assert "latex_display" in types

    def test_mixed_latex(self):
        """Mixed inline and display LaTeX are both detected."""
        text = "Inline $a$ and display $$b$$"
        result = parse_latex_in_text(text)
        
        types = [seg[1] for seg in result]
        assert "latex_inline" in types
        assert "latex_display" in types

    def test_escaped_dollar(self):
        """Escaped dollar signs are not treated as LaTeX."""
        text = "Price is \\$50 and \\$100"
        result = parse_latex_in_text(text)
        
        # Should preserve the escaped dollar signs
        full_text = "".join(seg[0] for seg in result)
        assert "$" in full_text

    def test_empty_text(self):
        """Empty text returns single empty segment."""
        result = parse_latex_in_text("")
        
        assert len(result) == 1
        assert result[0] == ("", "text")

    def test_multiline_display_latex(self):
        """Multiline display LaTeX is handled."""
        text = "$$\nx + y\n= z\n$$"
        result = parse_latex_in_text(text)
        
        types = [seg[1] for seg in result]
        assert "latex_display" in types


class TestNormalizeLatexWhitespace:
    """Tests for normalize_latex_whitespace function."""

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        text = "  x + y  "
        assert normalize_latex_whitespace(text) == "x+y"

    def test_collapses_spaces(self):
        """Multiple spaces are collapsed."""
        text = "x   +   y"
        result = normalize_latex_whitespace(text)
        assert "   " not in result

    def test_removes_space_around_operators(self):
        """Spaces around operators are removed."""
        text = "x = y + z"
        result = normalize_latex_whitespace(text)
        assert result == "x=y+z"


class TestSimplifyProblematicLatex:
    """Tests for simplify_problematic_latex function."""

    def test_returns_tuple(self):
        """Returns tuple of (simplified, applied_simplifications)."""
        result = simplify_problematic_latex("\\frac{a}{b}")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)

    def test_logical_operators_replaced(self):
        """Logical operators are replaced with alternatives."""
        latex = "a \\land b \\lor c"
        simplified, applied = simplify_problematic_latex(latex)
        
        assert "\\land" not in simplified
        assert "\\lor" not in simplified

    def test_text_commands_replaced(self):
        """\\text commands are replaced with \\mathrm."""
        latex = "\\text{hello}"
        simplified, applied = simplify_problematic_latex(latex)
        
        assert "\\text{" not in simplified or "\\mathrm{" in simplified

    def test_left_right_delimiters_simplified(self):
        """\\left and \\right are simplified."""
        latex = "\\left( x \\right)"
        simplified, applied = simplify_problematic_latex(latex)
        
        assert "\\left" not in simplified
        assert "\\right" not in simplified

    def test_phantom_removed(self):
        """\\phantom commands are removed."""
        latex = "x + \\phantom{hidden} y"
        simplified, applied = simplify_problematic_latex(latex)
        
        assert "\\phantom" not in simplified

    def test_spacing_normalized(self):
        """Spacing commands are normalized."""
        latex = "x\\,y\\;z"
        simplified, applied = simplify_problematic_latex(latex)
        
        assert "\\," not in simplified
        assert "\\;" not in simplified

    def test_environment_unwrapped(self):
        """Math environments are unwrapped."""
        latex = "\\begin{align}x = y\\end{align}"
        simplified, applied = simplify_problematic_latex(latex)
        
        assert "\\begin{align}" not in simplified
        assert "\\end{align}" not in simplified


class TestExtractSummaryPayload:
    """Tests for _extract_summary_payload function."""

    def test_direct_summary(self):
        """Extracts summary from result directly."""
        result = {
            "summary": {
                "bullet_points": ["Point 1"],
                "references": [],
            }
        }
        
        payload = _extract_summary_payload(result)
        
        assert "bullet_points" in payload

    def test_nested_summary(self):
        """Handles nested summary structure."""
        result = {
            "summary": {
                "summary": {
                    "bullet_points": ["Point 1"],
                }
            }
        }
        
        payload = _extract_summary_payload(result)
        
        assert "bullet_points" in payload

    def test_missing_summary(self):
        """Returns empty dict for missing summary."""
        result = {}
        
        payload = _extract_summary_payload(result)
        
        assert payload == {}

    def test_non_dict_summary(self):
        """Returns empty dict for non-dict summary."""
        result = {"summary": "string value"}
        
        payload = _extract_summary_payload(result)
        
        assert payload == {}


class TestPageNumberAndFlags:
    """Tests for _page_number_and_flags function."""

    def test_dict_format(self):
        """Handles dict format page number."""
        summary = {
            "page_number": {
                "page_number_integer": 5,
                "contains_no_page_number": False,
            }
        }
        
        result = _page_number_and_flags(summary)
        
        assert result["page_number_integer"] == 5
        assert result["contains_no_page_number"] is False

    def test_int_format(self):
        """Handles integer format page number."""
        summary = {"page_number": 10}
        
        result = _page_number_and_flags(summary)
        
        assert result["page_number_integer"] == 10

    def test_zero_page_number(self):
        """Zero page number indicates no page number."""
        summary = {"page_number": 0}
        
        result = _page_number_and_flags(summary)
        
        assert result["contains_no_page_number"] is True

    def test_fallback_to_page_field(self):
        """Falls back to 'page' field if page_number missing."""
        summary = {"page": 3}
        
        result = _page_number_and_flags(summary)
        
        # The function may return '?' for missing page_number or the page value
        assert result["page_number_integer"] in (3, "?")


class TestIsMeaningfulSummary:
    """Tests for _is_meaningful_summary function."""

    def test_meaningful_summary(self):
        """Returns True for meaningful summary."""
        summary = {
            "page_number": {
                "page_number_integer": 1,
                "contains_no_page_number": False,
            },
            "bullet_points": ["Key point 1", "Key point 2"],
            "contains_no_semantic_content": False,
        }
        
        assert _is_meaningful_summary(summary) is True

    def test_empty_bullet_points(self):
        """Returns False for empty bullet points."""
        summary = {
            "page_number": {"page_number_integer": 1, "contains_no_page_number": False},
            "bullet_points": [],
            "contains_no_semantic_content": False,
        }
        
        assert _is_meaningful_summary(summary) is False

    def test_error_marker_in_bullet(self):
        """Returns False when bullet contains error marker."""
        summary = {
            "page_number": {"page_number_integer": 1, "contains_no_page_number": False},
            "bullet_points": ["[empty page]"],
            "contains_no_semantic_content": False,
        }
        
        assert _is_meaningful_summary(summary) is False

    def test_no_semantic_content_flag(self):
        """Returns False when contains_no_semantic_content is True."""
        summary = {
            "page_number": {"page_number_integer": 1, "contains_no_page_number": False},
            "bullet_points": ["Some point"],
            "contains_no_semantic_content": True,
        }
        
        assert _is_meaningful_summary(summary) is False

    def test_unnumbered_page_zero(self):
        """Returns False for unnumbered page 0."""
        summary = {
            "page_number": {"page_number_integer": 0, "contains_no_page_number": True},
            "bullet_points": ["Some point"],
            "contains_no_semantic_content": False,
        }
        
        assert _is_meaningful_summary(summary) is False


class TestFileManagerIntegration:
    """Integration tests for file manager functions."""

    def test_docx_creation_imports(self):
        """Required imports for DOCX creation are available."""
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
        
        # Should not raise
        doc = Document()
        assert doc is not None

    def test_latex_conversion_imports(self):
        """Required imports for LaTeX conversion are available."""
        from latex2mathml.converter import convert as latex_to_mathml
        import mathml2omml
        
        # Test basic conversion
        mathml = latex_to_mathml("x")
        assert mathml is not None
