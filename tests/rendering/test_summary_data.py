"""Summary data preparation in rendering/summary.py.

Covers reference normalization, XML sanitizing, newline collapsing, page
information, the meaningful-summary and blank-transcription checks, page
headings (single pages and spreads), Document Structure page ranges and
``prepare_summary_data`` with malformed results.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoexcerpter.constants import is_blank_transcription
from autoexcerpter.rendering.citations import CitationManager
from autoexcerpter.rendering.summary import (
    _is_meaningful_summary,
    _normalize_references,
    _page_information,
    collapse_internal_newlines,
    filter_empty_pages,
    format_page_heading,
    format_structure_page_range,
    prepare_summary_data,
    sanitize_for_xml,
)


# ============================================================================
# References and text
# ============================================================================
class TestNormalizeReferences:
    """Tests for _normalize_references."""

    def test_object_items(self) -> None:
        """Schema objects yield (citation, is_partial) tuples."""
        refs = [
            {"citation": "Smith, J. (2020). Full Work. Journal.", "is_partial": False},
            {"citation": "  Doe, A. (1999).  ", "is_partial": True},
        ]
        assert _normalize_references(refs) == [
            ("Smith, J. (2020). Full Work. Journal.", False),
            ("Doe, A. (1999).", True),
        ]

    def test_bare_string_items(self) -> None:
        """Bare strings from prompted JSON are complete references."""
        refs = ["  Bare, B. (2001). String. Press. ", "Other, O. (1890). Book."]
        assert _normalize_references(refs) == [
            ("Bare, B. (2001). String. Press.", False),
            ("Other, O. (1890). Book.", False),
        ]

    def test_malformed_items_skipped(self) -> None:
        """Empty text, missing citation key and other types are skipped."""
        refs = [
            "",
            "   ",
            {"is_partial": True},
            {"citation": "", "is_partial": False},
            {"citation": None, "is_partial": True},
            42,
            None,
            {"citation": "Kept, K. (2003). Real. Press.", "is_partial": False},
        ]
        assert _normalize_references(refs) == [
            ("Kept, K. (2003). Real. Press.", False),
        ]

    def test_non_list_returns_empty(self) -> None:
        """None or a non-list value yields an empty list."""
        assert _normalize_references(None) == []
        assert _normalize_references("not a list") == []


class TestSanitizeForXml:
    """Tests for sanitize_for_xml."""

    def test_empty_string(self) -> None:
        """Empty string and None return an empty string."""
        assert sanitize_for_xml("") == ""
        assert sanitize_for_xml(None) == ""

    def test_normal_text_unchanged(self) -> None:
        text = "Hello, World!"
        assert sanitize_for_xml(text) == text

    def test_removes_null_character(self) -> None:
        assert sanitize_for_xml("before\x00after") == "beforeafter"

    def test_removes_control_characters(self) -> None:
        assert sanitize_for_xml("test\x01\x02\x03text") == "testtext"

    def test_preserves_tab(self) -> None:
        text = "col1\tcol2"
        assert sanitize_for_xml(text) == text

    def test_preserves_newline(self) -> None:
        text = "line1\nline2"
        assert sanitize_for_xml(text) == text

    def test_preserves_carriage_return(self) -> None:
        text = "line1\r\nline2"
        assert sanitize_for_xml(text) == text

    def test_removes_delete_character(self) -> None:
        """DEL (0x7F) is removed."""
        assert sanitize_for_xml("test\x7ftext") == "testtext"

    def test_removes_null_and_bell(self) -> None:
        assert sanitize_for_xml("a\x00b\x07c") == "abc"

    def test_removes_noncharacter_fffe(self) -> None:
        assert sanitize_for_xml("ab￾cd") == "abcd"

    def test_removes_surrogate(self) -> None:
        assert sanitize_for_xml("ab\ud800cd") == "abcd"

    def test_removes_fdd0_noncharacter_block(self) -> None:
        assert sanitize_for_xml("x﷐y﷯z") == "xyz"

    def test_preserves_non_ascii_text(self) -> None:
        text = "Normal text with accents: café — dash."
        assert sanitize_for_xml(text) == text


class TestCollapseInternalNewlines:
    """Line breaks inside a bullet or citation become single spaces."""

    def test_collapses_internal_newline(self) -> None:
        assert collapse_internal_newlines("Line one\nLine two") == "Line one Line two"
        assert collapse_internal_newlines("a  \n  b") == "a b"

    def test_lone_cr_collapses(self) -> None:
        assert collapse_internal_newlines("first\rsecond") == "first second"

    def test_padded_cr_collapses_to_single_space(self) -> None:
        assert collapse_internal_newlines("first  \r  second") == "first second"

    def test_lf_and_crlf_collapse(self) -> None:
        assert collapse_internal_newlines("first\nsecond") == "first second"
        assert collapse_internal_newlines("first\r\nsecond") == "first second"
        assert collapse_internal_newlines("first \n second") == "first second"

    def test_text_without_breaks_unchanged(self) -> None:
        assert collapse_internal_newlines("first second") == "first second"


# ============================================================================
# Page information
# ============================================================================
class TestPageInformation:
    """Tests for _page_information."""

    def test_dict_format_with_type(self) -> None:
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
        """The 'none' page type marks an unnumbered page."""
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
        summary = {"page_information": {"page_number_integer": 5}}

        result = _page_information(summary)

        assert result["page_number_type"] == "arabic"
        assert result["is_unnumbered"] is False

    def test_null_page_number_integer(self) -> None:
        """A null page_number_integer is unnumbered whatever its type says."""
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

    def test_string_page_types(self) -> None:
        """A single page type as a string (prompted JSON) becomes a list."""
        summary = {
            "page_information": {
                "page_number_integer": 7,
                "page_number_type": "arabic",
                "page_types": "bibliography",
            }
        }

        assert _page_information(summary)["page_types"] == ["bibliography"]

    def test_empty_page_information(self) -> None:
        """An empty page_information dict is treated as unnumbered."""
        summary = {"page_information": {}, "page": 5}

        result = _page_information(summary)

        assert result["page_number_integer"] == "?"
        assert result["is_unnumbered"] is True

    def test_page_field_only_is_unnumbered(self) -> None:
        """The ``page`` field is a scan position, never a printed number."""
        result = _page_information({"page": 10})

        assert result["page_number_integer"] == "?"
        assert result["page_number_type"] == "none"
        assert result["is_unnumbered"] is True
        assert result["page_types"] == ["content"]
        assert result["inferred"] is False


# ============================================================================
# Filtering
# ============================================================================
class TestIsMeaningfulSummary:
    """Tests for _is_meaningful_summary."""

    def test_meaningful_summary(self) -> None:
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
        """A blank-page sentinel bullet is not meaningful."""
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

        assert _is_meaningful_summary(summary) is True

    def test_null_bullet_points(self) -> None:
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

    def test_bullet_mentioning_error_is_meaningful(self) -> None:
        """Only placeholder bullets count as errors, not the word in prose."""
        summary = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Discusses measurement error in historical wage series."],
        }

        assert _is_meaningful_summary(summary) is True


class TestIsBlankTranscription:
    """constants.is_blank_transcription is gated by length."""

    def test_long_prose_mentioning_empty_page_is_not_blank(self) -> None:
        prose = (
            "The author reflects at length on the metaphor of the empty page "
            "as a site of possibility. " * 40
        )
        assert len(prose) > 2000
        assert is_blank_transcription(prose) is False

    def test_short_bracketed_sentinel_is_blank(self) -> None:
        sentinel = "[<img>: no transcribable text — blank scan]"
        assert is_blank_transcription(sentinel) is True

    @pytest.mark.parametrize(
        "marker",
        [
            "[p1.jpg: no transcribable text]",
            "[p1.jpg: transcription not possible]",
            "[p1.jpg: transcription not possible — torn scan]",
        ],
    )
    def test_emitted_markers_are_blank(self, marker: str) -> None:
        assert is_blank_transcription(marker) is True

    def test_short_prose_mentioning_a_phrase_is_not_blank(self) -> None:
        assert is_blank_transcription("Chapter 3: The empty page") is False


class TestFilterEmptyPages:
    """Tests for filter_empty_pages."""

    def test_filters_blank_pages(self) -> None:
        results: list[dict[str, Any]] = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["blank"],
                },
                "bullet_points": None,
            },
            {
                "page_information": {
                    "page_number_integer": 2,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["Real content"],
            },
        ]

        filtered = filter_empty_pages(results)

        assert filtered == [results[1]]

    def test_keeps_structure_pages_without_bullets(self) -> None:
        """Pages with structure types are kept even without bullets."""
        results = [
            {
                "page_information": {
                    "page_number_integer": 50,
                    "page_number_type": "arabic",
                    "page_types": ["bibliography"],
                },
                "bullet_points": [],
            },
        ]

        assert filter_empty_pages(results) == results

    def test_empty_input(self) -> None:
        assert filter_empty_pages([]) == []


# ============================================================================
# Page headings
# ============================================================================
class TestFormatPageHeading:
    """Tests for format_page_heading on single pages."""

    def test_arabic_content_page(self) -> None:
        assert format_page_heading(5, "arabic", ["content"], False) == "Page 5"

    def test_roman_page(self) -> None:
        assert format_page_heading(3, "roman", ["content"], False) == "Page iii"

    def test_string_page_number(self) -> None:
        assert format_page_heading("42", "arabic", ["content"], False) == "Page 42"

    def test_unnumbered_page(self) -> None:
        """page_number_type 'none' gives the unnumbered placeholder."""
        result = format_page_heading("?", "none", ["content"], True)
        assert result == "[Unnumbered page]"

    def test_unnumbered_page_via_flag(self) -> None:
        """The is_unnumbered flag wins over an arabic page_number_type."""
        result = format_page_heading("?", "arabic", ["content"], True)
        assert result == "[Unnumbered page]"

    def test_preface_prefix(self) -> None:
        result = format_page_heading(1, "roman", ["preface"], False)
        assert result == "[Preface] Page i"

    def test_appendix_prefix(self) -> None:
        result = format_page_heading(100, "arabic", ["appendix"], False)
        assert result == "[Appendix] Page 100"

    def test_abstract_prefix(self) -> None:
        """An abstract page without content gets the [Abstract] prefix."""
        result = format_page_heading(1, "arabic", ["abstract"], False)
        assert result == "[Abstract] Page 1"

    def test_abstract_with_content_no_prefix(self) -> None:
        result = format_page_heading(1, "arabic", ["abstract", "content"], False)
        assert "[Abstract]" not in result

    def test_figures_tables_prefix(self) -> None:
        result = format_page_heading(50, "arabic", ["figures_tables_sources"], False)
        assert result == "[Figures/Tables] Page 50"


class TestFormatPageHeadingSpread:
    """Tests for format_page_heading on two-page spreads."""

    def test_arabic_spread(self) -> None:
        result = format_page_heading(
            12, "arabic", ["content"], False, page_number_end=13, is_spread=True
        )
        assert result == "Pages 12-13"

    def test_roman_spread(self) -> None:
        result = format_page_heading(
            12, "roman", ["content"], False, page_number_end=13, is_spread=True
        )
        assert result == "Pages xii-xiii"

    def test_unnumbered_spread(self) -> None:
        result = format_page_heading(
            "?", "none", ["content"], True, page_number_end=None, is_spread=True
        )
        assert result == "[Unnumbered spread]"

    def test_spread_missing_end_derived(self) -> None:
        """A spread without an explicit end derives start + 1."""
        result = format_page_heading(12, "arabic", ["content"], False, is_spread=True)
        assert result == "Pages 12-13"

    def test_spread_keeps_type_prefix(self) -> None:
        result = format_page_heading(
            12, "roman", ["preface"], False, page_number_end=13, is_spread=True
        )
        assert result == "[Preface] Pages xii-xiii"


# ============================================================================
# Document Structure page ranges
# ============================================================================
class TestFormatStructurePageRange:
    """Tests for format_structure_page_range."""

    def test_empty(self) -> None:
        assert format_structure_page_range([]) == ""

    def test_single_arabic(self) -> None:
        assert format_structure_page_range([(5, "arabic")]) == "p. 5"

    def test_single_roman(self) -> None:
        assert format_structure_page_range([(3, "roman")]) == "p. iii"

    def test_roman_before_arabic(self) -> None:
        """Roman front matter is listed before arabic pages, each compacted."""
        entries = [
            (12, "roman"),
            (11, "roman"),
            (100, "arabic"),
            (101, "arabic"),
            (102, "arabic"),
        ]
        assert format_structure_page_range(entries) == "pp. xi-xii, 100-102"

    def test_roman_and_arabic_page_twelve_do_not_collide(self) -> None:
        entries = [(12, "roman"), (12, "arabic")]
        assert format_structure_page_range(entries) == "pp. xii, 12"

    def test_spread_pages_included(self) -> None:
        entries = [(12, "arabic"), (13, "arabic"), (14, "arabic")]
        assert format_structure_page_range(entries) == "pp. 12-14"


# ============================================================================
# prepare_summary_data
# ============================================================================
def _page(number: int, page_types: list[str], bullets: list[str]) -> dict[str, Any]:
    return {
        "page_information": {
            "page_number_integer": number,
            "page_number_type": "arabic",
            "page_types": page_types,
        },
        "bullet_points": bullets,
        "references": [],
    }


class TestPrepareSummaryData:
    """Tests for prepare_summary_data."""

    def test_page_types_decide_bullets_and_structure(self) -> None:
        """Prose types render bullets; structure types fill Document Structure."""
        results = [
            _page(1, ["content"], ["Body"]),
            _page(50, ["bibliography"], ["Not rendered"]),
            _page(60, ["content", "bibliography"], ["Both"]),
            _page(70, ["appendix"], ["Annex"]),
        ]

        data = prepare_summary_data(results, CitationManager())

        headings = [item.heading_text for item in data.page_render_items]
        assert headings == ["Page 1", "Page 60", "[Appendix] Page 70"]
        assert data.content_page_count == 3
        assert data.page_type_pages["bibliography"] == [(50, "arabic"), (60, "arabic")]
        assert data.page_type_pages["appendix"] == [(70, "arabic")]
        assert "content" not in data.page_type_pages

    def test_spread_records_both_pages_and_citations(self) -> None:
        """A numbered spread records both pages in structure and citations."""
        cm = CitationManager()
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 12,
                    "is_two_page_spread": True,
                    "page_number_integer_end": 13,
                    "page_number_type": "arabic",
                    "page_types": ["content", "bibliography"],
                },
                "bullet_points": ["A point"],
                "references": [
                    {"citation": "Author (2020). Title.", "is_partial": False}
                ],
            }
        ]

        data = prepare_summary_data(summary_results, cm)

        assert (12, "arabic") in data.page_type_pages["bibliography"]
        assert (13, "arabic") in data.page_type_pages["bibliography"]

        citations = list(cm.citations.values())
        assert len(citations) == 1
        assert citations[0].get_page_range_str() == "pp. 12-13"

        assert data.page_render_items[0].heading_text == "Pages 12-13"
        assert data.page_render_items[0].is_spread is True

    def test_flat_result_renders_as_unnumbered_page(self) -> None:
        """A result without page_information renders as an unnumbered page."""
        result = {"page": 1, "bullet_points": ["First point.", "Second point."]}

        assert filter_empty_pages([result]) == [result]

        data = prepare_summary_data([result], CitationManager())

        assert len(data.page_render_items) == 1
        item = data.page_render_items[0]
        assert item.is_unnumbered is True
        assert item.position == 1
        assert item.bullet_points == ["First point.", "Second point."]
        assert item.heading_text == format_page_heading(
            "?", "none", ["content"], True, scan=("pdf", 1)
        )

    def test_error_placeholder_page_is_rendered(self) -> None:
        """A failed page keeps its placeholder bullet and its heading."""
        result = _page(3, ["other"], ["[Error generating summary: request timed out]"])
        result["error"] = "request timed out"

        assert _is_meaningful_summary(result) is True
        data = prepare_summary_data([result], CitationManager())

        assert len(data.page_render_items) == 1
        item = data.page_render_items[0]
        assert item.bullet_points == ["[Error generating summary: request timed out]"]
        assert "Page 3" in item.heading_text

    def test_non_string_bullets_are_dropped(self) -> None:
        result = _page(1, ["content"], [])
        result["bullet_points"] = [None, 123, "real bullet"]

        assert _is_meaningful_summary(result) is True
        data = prepare_summary_data([result], CitationManager())

        assert data.page_render_items[0].bullet_points == ["real bullet"]

    def test_non_string_page_types_entry_is_ignored(self) -> None:
        """A dict in page_types, as from a corrupt log, does not raise."""
        result = _page(1, [], ["A finding."])
        result["page_information"]["page_types"] = [{"bad": True}]

        data = prepare_summary_data([result], CitationManager())

        assert len(data.page_render_items) == 1

    def test_string_page_types_survive_beside_debris(self) -> None:
        result = _page(1, [], ["A finding."])
        result["page_information"]["page_types"] = ["bibliography", {"bad": True}]

        data = prepare_summary_data([result], CitationManager())

        assert data.page_type_pages.get("bibliography") == [(1, "arabic")]
