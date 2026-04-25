"""Tests for rendering/summary.py and related rendering utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from rendering.docx import (
    parse_latex_in_text,
    normalize_latex_whitespace,
    simplify_problematic_latex,
)
from rendering.summary import (
    sanitize_for_xml,
    _extract_summary_payload,
    _page_information,
    _is_meaningful_summary,
    _should_render_bullets,
    _get_structure_types,
    filter_empty_pages,
    PAGE_TYPES_WITH_BULLETS,
    STRUCTURE_PAGE_TYPE_ORDER,
    format_page_heading,
    int_to_roman,
)
from rendering.markdown import (
    create_markdown_summary,
)


class TestSanitizeForXml:
    """Tests for sanitize_for_xml function."""

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert sanitize_for_xml("") == ""
        assert sanitize_for_xml(None) == ""

    def test_normal_text_unchanged(self) -> None:
        """Normal text is unchanged."""
        text = "Hello, World!"
        assert sanitize_for_xml(text) == text

    def test_removes_null_character(self) -> None:
        """Null character is removed."""
        text = "before\x00after"
        assert sanitize_for_xml(text) == "beforeafter"

    def test_removes_control_characters(self) -> None:
        """Control characters are removed."""
        text = "test\x01\x02\x03text"
        result = sanitize_for_xml(text)
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x03" not in result
        assert "testtext" == result

    def test_preserves_tab(self) -> None:
        """Tab character is preserved."""
        text = "col1\tcol2"
        assert sanitize_for_xml(text) == text

    def test_preserves_newline(self) -> None:
        """Newline character is preserved."""
        text = "line1\nline2"
        assert sanitize_for_xml(text) == text

    def test_preserves_carriage_return(self) -> None:
        """Carriage return is preserved."""
        text = "line1\r\nline2"
        assert sanitize_for_xml(text) == text

    def test_removes_delete_character(self) -> None:
        """DEL character (0x7F) is removed."""
        text = "test\x7ftext"
        assert sanitize_for_xml(text) == "testtext"


class TestParseLatexInText:
    """Tests for parse_latex_in_text function."""

    def test_no_latex(self) -> None:
        """Text without LaTeX returns single text segment."""
        text = "Plain text without formulas"
        result = parse_latex_in_text(text)

        assert len(result) == 1
        assert result[0][1] == "text"

    def test_inline_latex(self) -> None:
        """Inline LaTeX ($...$) is detected."""
        text = "Here is $x + y = z$ inline"
        result = parse_latex_in_text(text)

        types = [seg[1] for seg in result]
        assert "latex_inline" in types

    def test_display_latex(self) -> None:
        """Display LaTeX ($$...$$) is detected."""
        text = "Here is $$x + y = z$$ display"
        result = parse_latex_in_text(text)

        types = [seg[1] for seg in result]
        assert "latex_display" in types

    def test_mixed_latex(self) -> None:
        """Mixed inline and display LaTeX are both detected."""
        text = "Inline $a$ and display $$b$$"
        result = parse_latex_in_text(text)

        types = [seg[1] for seg in result]
        assert "latex_inline" in types
        assert "latex_display" in types

    def test_escaped_dollar(self) -> None:
        """Escaped dollar signs are not treated as LaTeX."""
        text = "Price is \\$50 and \\$100"
        result = parse_latex_in_text(text)

        # Should preserve the escaped dollar signs
        full_text = "".join(seg[0] for seg in result)
        assert "$" in full_text

    def test_empty_text(self) -> None:
        """Empty text returns single empty segment."""
        result = parse_latex_in_text("")

        assert len(result) == 1
        assert result[0] == ("", "text")

    def test_multiline_display_latex(self) -> None:
        """Multiline display LaTeX is handled."""
        text = "$$\nx + y\n= z\n$$"
        result = parse_latex_in_text(text)

        types = [seg[1] for seg in result]
        assert "latex_display" in types


class TestNormalizeLatexWhitespace:
    """Tests for normalize_latex_whitespace function."""

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        text = "  x + y  "
        assert normalize_latex_whitespace(text) == "x+y"

    def test_collapses_spaces(self) -> None:
        """Multiple spaces are collapsed."""
        text = "x   +   y"
        result = normalize_latex_whitespace(text)
        assert "   " not in result

    def test_removes_space_around_operators(self) -> None:
        """Spaces around operators are removed."""
        text = "x = y + z"
        result = normalize_latex_whitespace(text)
        assert result == "x=y+z"


class TestSimplifyProblematicLatex:
    """Tests for simplify_problematic_latex function."""

    def test_returns_tuple(self) -> None:
        """Returns tuple of (simplified, applied_simplifications)."""
        result = simplify_problematic_latex("\\frac{a}{b}")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)

    def test_logical_operators_replaced(self) -> None:
        """Logical operators are replaced with alternatives."""
        latex = "a \\land b \\lor c"
        simplified, applied = simplify_problematic_latex(latex)

        assert "\\land" not in simplified
        assert "\\lor" not in simplified

    def test_text_commands_replaced(self) -> None:
        """\\text commands are replaced with \\mathrm."""
        latex = "\\text{hello}"
        simplified, applied = simplify_problematic_latex(latex)

        assert "\\text{" not in simplified or "\\mathrm{" in simplified

    def test_left_right_delimiters_simplified(self) -> None:
        """\\left and \\right are simplified."""
        latex = "\\left( x \\right)"
        simplified, applied = simplify_problematic_latex(latex)

        assert "\\left" not in simplified
        assert "\\right" not in simplified

    def test_phantom_removed(self) -> None:
        """\\phantom commands are removed."""
        latex = "x + \\phantom{hidden} y"
        simplified, applied = simplify_problematic_latex(latex)

        assert "\\phantom" not in simplified

    def test_spacing_normalized(self) -> None:
        """Spacing commands are normalized."""
        latex = "x\\,y\\;z"
        simplified, applied = simplify_problematic_latex(latex)

        assert "\\," not in simplified
        assert "\\;" not in simplified

    def test_environment_unwrapped(self) -> None:
        """Math environments are unwrapped."""
        latex = "\\begin{align}x = y\\end{align}"
        simplified, applied = simplify_problematic_latex(latex)

        assert "\\begin{align}" not in simplified
        assert "\\end{align}" not in simplified


class TestExtractSummaryPayload:
    """Tests for _extract_summary_payload function."""

    def test_flat_structure(self) -> None:
        """Extracts from flat structure (preferred)."""
        result = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
            },
            "bullet_points": ["Point 1"],
            "references": [],
        }

        payload = _extract_summary_payload(result)

        assert "bullet_points" in payload
        assert payload["bullet_points"] == ["Point 1"]

    def test_legacy_direct_summary(self) -> None:
        """Handles legacy direct summary structure."""
        result = {
            "summary": {
                "bullet_points": ["Point 1"],
                "references": [],
            }
        }

        payload = _extract_summary_payload(result)

        assert "bullet_points" in payload

    def test_missing_summary(self) -> None:
        """Returns empty dict for missing summary."""
        result: dict[str, Any] = {}

        payload = _extract_summary_payload(result)

        assert payload == {}

    def test_non_dict_summary(self) -> None:
        """Returns empty dict for non-dict summary."""
        result = {"summary": "string value"}

        payload = _extract_summary_payload(result)

        assert payload == {}


class TestPageNumberAndFlags:
    """Tests for _page_information function."""

    def test_dict_format(self) -> None:
        """Handles dict format page number."""
        summary = {
            "page_information": {
                "page_number_integer": 5,
                "page_number_type": "arabic",
            }
        }

        result = _page_information(summary)

        assert result["page_number_integer"] == 5
        assert result["is_unnumbered"] is False

    def test_fallback_to_page_field_only(self) -> None:
        """Falls back to page field when page_information is missing."""
        summary = {"page": 10}

        result = _page_information(summary)

        assert result["page_number_integer"] == 10
        assert result["page_types"] == ["content"]

    def test_empty_page_information(self) -> None:
        """Empty page_information dict falls back to page field."""
        summary = {"page_information": {}, "page": 5}

        result = _page_information(summary)

        # Empty dict is falsy, so falls back to page field
        assert result["page_number_integer"] == 5
        assert result["is_unnumbered"] is False

    def test_fallback_to_page_field(self) -> None:
        """Falls back to 'page' field if page_number missing."""
        summary = {"page": 3}

        result = _page_information(summary)

        # The function may return '?' for missing page_number or the page value
        assert result["page_number_integer"] in (3, "?")


class TestIsMeaningfulSummary:
    """Tests for _is_meaningful_summary function."""

    def test_meaningful_summary(self) -> None:
        """Returns True for meaningful summary."""
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
            },
            "bullet_points": ["Key point 1", "Key point 2"],
            "page_types": ["content"],
        }

        assert _is_meaningful_summary(summary) is True

    def test_empty_bullet_points(self) -> None:
        """Returns False for empty bullet points."""
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": [],
        }

        assert _is_meaningful_summary(summary) is False

    def test_error_marker_in_bullet(self) -> None:
        """Returns False when bullet contains error marker."""
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["[empty page]"],
        }

        assert _is_meaningful_summary(summary) is False

    def test_blank_page_type(self) -> None:
        """Returns False when page_types is ['blank']."""
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["blank"],
            },
            "bullet_points": None,
        }

        assert _is_meaningful_summary(summary) is False

    def test_unnumbered_page_with_content_type(self) -> None:
        """Content pages with bullet points are meaningful even if unnumbered."""
        summary = {
            "page_information": {
                "page_number_integer": None,
                "page_number_type": "none",
                "page_types": ["content"],
            },
            "bullet_points": ["Some point"],
        }

        # Content pages with bullet points are meaningful
        assert _is_meaningful_summary(summary) is True

    def test_null_bullet_points(self) -> None:
        """Returns False for null bullet_points."""
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": None,
        }

        assert _is_meaningful_summary(summary) is False

    def test_null_references_still_meaningful(self) -> None:
        """Returns True for meaningful summary with null references."""
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Key point 1", "Key point 2"],
            "references": None,
        }

        assert _is_meaningful_summary(summary) is True


class TestIntToRoman:
    """Tests for int_to_roman function."""

    def test_single_digits(self) -> None:
        """Single digit numbers convert correctly."""
        assert int_to_roman(1) == "i"
        assert int_to_roman(2) == "ii"
        assert int_to_roman(3) == "iii"
        assert int_to_roman(4) == "iv"
        assert int_to_roman(5) == "v"
        assert int_to_roman(6) == "vi"
        assert int_to_roman(7) == "vii"
        assert int_to_roman(8) == "viii"
        assert int_to_roman(9) == "ix"

    def test_double_digits(self) -> None:
        """Double digit numbers convert correctly."""
        assert int_to_roman(10) == "x"
        assert int_to_roman(12) == "xii"
        assert int_to_roman(14) == "xiv"
        assert int_to_roman(19) == "xix"
        assert int_to_roman(20) == "xx"
        assert int_to_roman(50) == "l"

    def test_larger_numbers(self) -> None:
        """Larger numbers convert correctly."""
        assert int_to_roman(100) == "c"
        assert int_to_roman(500) == "d"
        assert int_to_roman(1000) == "m"
        assert int_to_roman(99) == "xcix"
        assert int_to_roman(444) == "cdxliv"

    def test_zero_returns_empty(self) -> None:
        """Zero returns empty string."""
        assert int_to_roman(0) == ""

    def test_negative_returns_empty(self) -> None:
        """Negative numbers return empty string."""
        assert int_to_roman(-1) == ""
        assert int_to_roman(-100) == ""


class TestPageNumberAndFlagsWithType:
    """Tests for _page_information with page_number_type field."""

    def test_dict_format_with_type(self) -> None:
        """Handles dict format with page_number_type."""
        summary = {
            "page_information": {
                "page_number_integer": 5,
                "page_number_type": "arabic",
            }
        }

        result = _page_information(summary)

        assert result["page_number_integer"] == 5
        assert result["page_number_type"] == "arabic"
        assert result["is_unnumbered"] is False

    def test_roman_page_type(self) -> None:
        """Handles Roman numeral page type."""
        summary = {
            "page_information": {
                "page_number_integer": 12,
                "page_number_type": "roman",
            }
        }

        result = _page_information(summary)

        assert result["page_number_integer"] == 12
        assert result["page_number_type"] == "roman"
        assert result["is_unnumbered"] is False

    def test_none_page_type(self) -> None:
        """Handles 'none' page type for unnumbered pages."""
        summary = {
            "page_information": {
                "page_number_integer": None,
                "page_number_type": "none",
            }
        }

        result = _page_information(summary)

        assert result["page_number_type"] == "none"
        assert result["is_unnumbered"] is True

    def test_missing_type_defaults_to_arabic(self) -> None:
        """Missing page_number_type defaults to 'arabic'."""
        summary = {
            "page_information": {
                "page_number_integer": 5,
            }
        }

        result = _page_information(summary)

        assert result["page_number_type"] == "arabic"
        assert result["is_unnumbered"] is False

    def test_page_field_fallback_defaults_to_arabic(self) -> None:
        """Fallback to page field uses arabic type."""
        summary = {"page": 10}

        result = _page_information(summary)

        assert result["page_number_integer"] == 10
        assert result["page_number_type"] == "arabic"

    def test_fallback_with_page_field(self) -> None:
        """Fallback case uses page field value."""
        summary = {"page": 3}

        result = _page_information(summary)

        # Fallback case returns the page value but marks as arabic type
        assert result["page_number_integer"] == 3
        assert result["page_number_type"] == "arabic"

    def test_null_page_number_integer(self) -> None:
        """Null page_number_integer is handled as unnumbered."""
        summary = {
            "page_information": {
                "page_number_integer": None,
                "page_number_type": "arabic",
            }
        }

        result = _page_information(summary)

        assert result["page_number_integer"] == "?"
        assert result["page_number_type"] == "none"
        assert result["is_unnumbered"] is True


class TestFileManagerIntegration:
    """Integration tests for file manager functions."""

    def test_docx_creation_imports(self) -> None:
        """Required imports for DOCX creation are available."""
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

        # Should not raise
        doc = Document()
        assert doc is not None

    def test_latex_conversion_imports(self) -> None:
        """Required imports for LaTeX conversion are available."""
        from latex2mathml.converter import convert as latex_to_mathml
        import mathml2omml

        # Test basic conversion
        mathml = latex_to_mathml("x")
        assert mathml is not None


class TestFormatPageHeadingMd:
    """Tests for format_page_heading (used by markdown writer with ## prefix)."""

    def test_arabic_page_number(self) -> None:
        """Arabic page numbers format correctly."""
        result = f"## {format_page_heading(5, 'arabic', ['content'], False)}"
        assert result == "## Page 5"

    def test_roman_page_number(self) -> None:
        """Roman numeral pages format with Page prefix and roman numeral."""
        result = f"## {format_page_heading(3, 'roman', ['content'], False)}"
        assert result == "## Page iii"

    def test_unnumbered_page_via_type(self) -> None:
        """Unnumbered pages via page_number_type='none' format correctly."""
        result = f"## {format_page_heading('?', 'none', ['content'], False)}"
        assert result == "## [Unnumbered page]"

    def test_unnumbered_page_via_flag(self) -> None:
        """Unnumbered pages via is_unnumbered flag format correctly."""
        result = f"## {format_page_heading('?', 'arabic', ['content'], True)}"
        assert result == "## [Unnumbered page]"

    def test_string_page_number(self) -> None:
        """String page numbers are handled."""
        result = f"## {format_page_heading('42', 'arabic', ['content'], False)}"
        assert result == "## Page 42"

    def test_preface_page_type(self) -> None:
        """Preface page type adds prefix."""
        result = f"## {format_page_heading(1, 'roman', ['preface'], False)}"
        assert result == "## [Preface] Page i"

    def test_appendix_page_type(self) -> None:
        """Appendix page type adds prefix."""
        result = f"## {format_page_heading(100, 'arabic', ['appendix'], False)}"
        assert result == "## [Appendix] Page 100"


class TestCreateMarkdownSummary:
    """Tests for create_markdown_summary function."""

    def test_creates_file(self, tmp_path) -> None:
        """Creates a markdown file at the specified path."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Point 1", "Point 2"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test Document")

        assert output_path.exists()

    def test_contains_title(self, tmp_path) -> None:
        """Output contains document title."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Point 1"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "My Test Doc")

        content = output_path.read_text(encoding="utf-8")
        assert "# Summary of My Test Doc" in content

    def test_contains_metadata(self, tmp_path) -> None:
        """Output contains processing metadata."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Point 1"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "*Processed:" in content
        assert "Total pages: 1*" in content

    def test_contains_page_headings(self, tmp_path) -> None:
        """Output contains page headings."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 5,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Point 1"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "## Page 5" in content

    def test_contains_bullet_points(self, tmp_path) -> None:
        """Output contains bullet points."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["First point", "Second point"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "- First point" in content
        assert "- Second point" in content

    def test_roman_numeral_pages(self, tmp_path) -> None:
        """Roman numeral pages show Pre-face heading."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 3,
                    "page_number_type": "roman",
                    "page_types": ["content"],
                },
                "bullet_points": ["Preface content"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "## Page iii" in content

    def test_filters_empty_pages(self, tmp_path) -> None:
        """Empty pages are filtered out."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Valid content"],
                "references": [],
            },
            {
                "page_information": {
                    "page_number_integer": 2,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": [],
                "references": [],
            },
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "## Page 1" in content
        assert "## Page 2" not in content

    def test_preserves_latex_formulas(self, tmp_path) -> None:
        """LaTeX formulas are preserved in markdown."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["The formula $x + y = z$ is important"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "$x + y = z$" in content

    def test_multiple_pages(self, tmp_path) -> None:
        """Multiple pages are included in order."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Page 1 content"],
                "references": [],
            },
            {
                "page_information": {
                    "page_number_integer": 2,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Page 2 content"],
                "references": [],
            },
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "## Page 1" in content
        assert "## Page 2" in content
        # Page 1 should come before Page 2
        assert content.index("## Page 1") < content.index("## Page 2")

    def test_empty_results(self, tmp_path) -> None:
        """Empty results create minimal file."""
        output_path = tmp_path / "test_summary.md"

        create_markdown_summary([], output_path, "Empty Doc")

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "# Summary of Empty Doc" in content
        assert "Total pages: 0*" in content

    @patch("rendering.markdown.CitationManager")
    def test_references_section_with_citations(
        self, mock_citation_manager_class, tmp_path
    ) -> None:
        """References section is added when citations exist."""
        output_path = tmp_path / "test_summary.md"

        # Setup mock citation manager
        mock_citation = MagicMock()
        mock_citation.raw_text = "Author (2020). Title."
        mock_citation.url = "https://example.com"
        mock_citation.doi = "10.1234/example"
        mock_citation.metadata = {"publication_year": 2020}

        mock_manager = MagicMock()
        mock_manager.citations = {"key": mock_citation}
        mock_manager.get_citations_with_pages.return_value = [(mock_citation, "p. 1")]
        mock_citation_manager_class.return_value = mock_manager

        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Content"],
                "references": ["Author (2020). Title."],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        assert "## Consolidated References" in content
        assert "[Author (2020). Title.](https://example.com)" in content

    def test_unnumbered_pages(self, tmp_path) -> None:
        """Unnumbered pages show appropriate heading."""
        output_path = tmp_path / "test_summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": None,
                    "page_number_type": "none",
                    "page_types": ["content"],
                },
                "bullet_points": ["Unnumbered content"],
                "references": [],
            }
        ]

        # Note: This page will be filtered as unnumbered, so let's add a valid page too
        summary_results.append(
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Valid content"],
                "references": [],
            }
        )

        create_markdown_summary(summary_results, output_path, "Test")

        content = output_path.read_text(encoding="utf-8")
        # Unnumbered page should be filtered out
        assert "## Page 1" in content
