"""Pages without a printed number stay locatable, and no number is invented.

- The ``.txt`` carries ``<page_break pdf="N"/>`` (``image="N"``) before a page
  without a ``<page_number>`` tag; numbered pages are written unchanged.
- A number inferred between two numbered neighbours is bracketed: ``Page [6]``
  in headings, ``p. [6]`` in citations.
- Unnumbered pages are headed and cited by scan position (``PDF p. N``).
- Citation locators are typed: printed arabic, printed roman, PDF, image.
- ``repair_layout`` treats ``<page_break`` as a page boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from docx import Document

from pipeline.page_numbering import PageNumberProcessor
from rendering.citations import CitationManager, _format_page_range
from rendering.docx import create_docx_summary
from rendering.markdown import create_markdown_summary
from rendering.summary import build_render_context, format_page_heading
from rendering.text import write_transcription_to_text
from scripts.repair_layout.repair import is_passthrough_line, repair_text
from scripts.repair_layout.verifier import verify
from tests.test_repair_layout import _WRAPPED_PAGE

_REF = "Smith, J. (1990). Atlantic Trade Networks. Oxford University Press."


def _body(path: Path) -> str:
    """Return the .txt content after the metadata header."""
    return path.read_text(encoding="utf-8").split("\n\n", 1)[1]


def _write_txt(
    tmp_path: Path, results: list[dict[str, Any]], item_type: str = "PDF"
) -> str:
    out = tmp_path / "doc.txt"
    assert write_transcription_to_text(
        results, out, "doc", item_type, 1.0, tmp_path / "doc.pdf"
    )
    return _body(out)


def _summary(
    index: int,
    page: int | None,
    ptype: str,
    page_types: list[str] | None = None,
    references: list[str] | None = None,
    bullets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "original_input_order_index": index,
        "page_information": {
            "page_number_integer": page,
            "page_number_type": ptype,
            "page_types": page_types or ["content"],
        },
        "bullet_points": bullets if bullets is not None else [f"Point {index}."],
        "references": references or [],
    }


# ============================================================================
# .txt page breaks
# ============================================================================


class TestTextPageBreaks:
    def test_mixed_numbered_and_unnumbered_pdf(self, tmp_path: Path) -> None:
        results = [
            {
                "original_input_order_index": 0,
                "page_index": 0,
                "transcription": "<page_number>5</page_number>\nFirst page.",
            },
            {
                "original_input_order_index": 1,
                "page_index": 1,
                "transcription": "Untagged plate.",
            },
            {
                "original_input_order_index": 2,
                "page_index": 2,
                "transcription": "Third page.\n<page_number>7</page_number>",
            },
        ]
        assert _write_txt(tmp_path, results) == (
            "<page_number>5</page_number>\nFirst page.\n"
            '<page_break pdf="2"/>\nUntagged plate.\n'
            "Third page.\n<page_number>7</page_number>"
        )

    def test_numbered_pages_are_byte_identical(self, tmp_path: Path) -> None:
        results = [
            {"original_input_order_index": i, "page_index": i, "transcription": t}
            for i, t in enumerate(
                ["<page_number>1</page_number>\nA", "B <page_number>2</page_number>"]
            )
        ]
        assert _write_txt(tmp_path, results) == (
            "<page_number>1</page_number>\nA\nB <page_number>2</page_number>"
        )

    def test_image_folder_uses_image_attribute(self, tmp_path: Path) -> None:
        results = [
            {"original_input_order_index": 0, "transcription": "Scan one."},
            {"original_input_order_index": 1, "transcription": "Scan two."},
        ]
        assert _write_txt(tmp_path, results, item_type="Image Folder") == (
            '<page_break image="1"/>\nScan one.\n<page_break image="2"/>\nScan two.'
        )

    def test_pdf_page_without_index_never_claims_pdf(self, tmp_path: Path) -> None:
        results = [
            {
                "original_input_order_index": 3,
                "transcription": "[preprocessing error: boom]",
                "error": "boom",
            }
        ]
        assert _write_txt(tmp_path, results).startswith('<page_break image="4"/>\n')


# ============================================================================
# Inferred numbers: 5, untagged, 7
# ============================================================================


class TestInferredPage:
    def _adjusted(self, ptype: str = "arabic") -> list[dict[str, Any]]:
        results = [
            _summary(0, 5, ptype),
            _summary(1, None, "none", references=[_REF]),
            _summary(2, 7, ptype),
        ]
        return PageNumberProcessor().adjust_and_sort_page_numbers(results)

    def test_inferred_flag_is_set_only_on_the_gap(self) -> None:
        infos = [r["page_information"] for r in self._adjusted()]
        assert [i["page_number_integer"] for i in infos] == [5, 6, 7]
        assert [bool(i.get("number_inferred")) for i in infos] == [
            False,
            True,
            False,
        ]

    def test_heading_and_citation_are_bracketed(self) -> None:
        manager, data = build_render_context(self._adjusted())
        headings = [item.heading_text for item in data.page_render_items]
        assert headings == ["Page 5", "Page [6]", "Page 7"]
        assert data.page_render_items[1].inferred is True
        (citation,) = manager.citations.values()
        assert citation.get_page_range_str() == "p. [6]"

    def test_roman_inferred_heading(self) -> None:
        _manager, data = build_render_context(self._adjusted("roman"))
        headings = [item.heading_text for item in data.page_render_items]
        assert headings == ["Page v", "Page [vi]", "Page vii"]

    def test_txt_marks_the_inferred_page(self, tmp_path: Path) -> None:
        results = [
            {"original_input_order_index": 0, "page_index": 0,
             "transcription": "A <page_number>5</page_number>"},
            {"original_input_order_index": 1, "page_index": 1,
             "transcription": "B"},
            {"original_input_order_index": 2, "page_index": 2,
             "transcription": "C <page_number>7</page_number>"},
        ]  # fmt: skip
        assert '\n<page_break pdf="2"/>\nB\n' in _write_txt(tmp_path, results)

    def test_unusable_tag_still_gets_marker(self, tmp_path: Path) -> None:
        results = [
            {"original_input_order_index": 0, "page_index": 0,
             "transcription": "A <page_number>?</page_number>"},
        ]  # fmt: skip
        assert _write_txt(tmp_path, results).startswith('<page_break pdf="1"/>\n')

    def test_flag_cleared_when_page_ends_unnumbered(self) -> None:
        result = _summary(0, 0, "arabic")
        result["page_information"]["number_inferred"] = True
        with patch.object(
            PageNumberProcessor,
            "_find_section_anchor",
            return_value=(None, None, None),
        ):
            (adjusted,) = PageNumberProcessor().adjust_and_sort_page_numbers([result])
        info = adjusted["page_information"]
        assert info["page_number_integer"] is None
        assert info["page_number_type"] == "none"
        assert "number_inferred" not in info


class TestNoAnchorFallback:
    def test_no_anchor_page_is_not_given_its_position(self) -> None:
        """Without anchor or usable model number the page stays unnumbered."""
        with patch.object(
            PageNumberProcessor,
            "_find_section_anchor",
            return_value=(None, None, None),
        ):
            adjusted = PageNumberProcessor().adjust_and_sort_page_numbers(
                [_summary(0, 0, "arabic"), _summary(1, -3, "roman")]
            )
        for result in adjusted:
            info = result["page_information"]
            assert info["page_number_integer"] is None
            assert info["page_number_type"] == "none"


# ============================================================================
# Headings for unnumbered pages
# ============================================================================


class TestUnnumberedHeadings:
    @pytest.mark.parametrize(
        ("label", "spread", "expected"),
        [
            ("pdf", False, "[No printed number; PDF p. 12]"),
            ("image", False, "[No printed number; Image 12]"),
            ("pdf", True, "[No printed numbers; PDF p. 12]"),
            ("image", True, "[No printed numbers; Image 12]"),
        ],
    )
    def test_position_heading(self, label: str, spread: bool, expected: str) -> None:
        heading = format_page_heading(
            "?",
            "none",
            ["content"],
            True,
            is_spread=spread,
            position=12,
            position_label=label,
        )
        assert heading == expected

    def test_type_prefix_is_kept(self) -> None:
        heading = format_page_heading(
            "?", "none", ["figures_tables_sources"], True, position=3
        )
        assert heading == "[Figures/Tables] [No printed number; PDF p. 3]"

    def test_inferred_spread(self) -> None:
        heading = format_page_heading(
            6, "arabic", ["content"], False, 7, is_spread=True, inferred=True
        )
        assert heading == "Pages [6]-[7]"

    def test_render_context_uses_scan_position(self) -> None:
        results = [_summary(11, None, "none", references=[_REF])]
        manager, data = build_render_context(results, position_label="image")
        assert data.page_render_items[0].heading_text == (
            "[No printed number; Image 12]"
        )
        assert data.page_render_items[0].position == 12
        (citation,) = manager.citations.values()
        assert citation.get_page_range_str() == "Image 12"

    def test_page_field_without_page_information_is_unnumbered(self) -> None:
        results = [
            {
                "original_input_order_index": 4,
                "summary": {"page": 5, "bullet_points": ["x"]},
            }
        ]
        _manager, data = build_render_context(results)
        assert data.page_render_items[0].heading_text == (
            "[No printed number; PDF p. 5]"
        )


# ============================================================================
# Typed citation locators
# ============================================================================


class TestCitationLocators:
    @pytest.mark.parametrize(
        ("locators", "expected"),
        [
            ([("printed-arabic", n, False) for n in (12, 13, 14)], "pp. 12-14"),
            ([("printed-roman", n, False) for n in (10, 11, 12)], "pp. x-xii"),
            ([("pdf", n, False) for n in (12, 13, 14)], "PDF pp. 12-14"),
            ([("printed-arabic", 6, True)], "p. [6]"),
            (
                [("printed-arabic", n, False) for n in (5, 6, 7)]
                + [("pdf", 12, False)],
                "pp. 5-7; PDF p. 12",
            ),
            (
                [("pdf", 12, False), ("printed-arabic", 12, False)]
                + [("printed-roman", 3, False), ("image", 2, False)],
                "p. iii; p. 12; PDF p. 12; Image 2",
            ),
            (
                [("printed-arabic", 5, False), ("printed-arabic", 6, True)]
                + [("printed-arabic", 7, False)],
                "pp. 5, [6], 7",
            ),
        ],
    )
    def test_format(self, locators: list[tuple[str, int, bool]], expected: str) -> None:
        assert _format_page_range(locators) == expected

    def test_roman_pages_stay_roman_through_prepare(self) -> None:
        results = [
            _summary(i, page, "roman", references=[_REF])
            for i, page in enumerate((10, 11, 12))
        ]
        manager, _data = build_render_context(results)
        (citation,) = manager.citations.values()
        assert citation.get_page_range_str() == "pp. x-xii"

    def test_mixture_through_prepare(self) -> None:
        results = [
            _summary(0, 5, "arabic", references=[_REF]),
            _summary(1, 6, "arabic", references=[_REF]),
            _summary(2, None, "none", references=[_REF]),
        ]
        manager, _data = build_render_context(results)
        (citation,) = manager.citations.values()
        assert citation.get_page_range_str() == "pp. 5-6; PDF p. 3"

    def test_merge_unions_locators(self) -> None:
        manager = CitationManager()
        manager.add_citations(
            ["Mennell, S. (1985). All Manners of Food. Blackwell."],
            ("printed-roman", 4, False),
        )
        manager.add_citations(
            ["Mennell, Stephen (1985). All Manners of Food. Blackwell."],
            ("pdf", 9, False),
        )
        manager.consolidate()
        (citation,) = manager.citations.values()
        assert citation.get_page_range_str() == "p. iv; PDF p. 9"


# ============================================================================
# Bibliography-only page through JSON, Markdown and DOCX
# ============================================================================


class TestBibliographyOnlyPage:
    def _results(self) -> list[dict[str, Any]]:
        results = [
            _summary(0, 20, "arabic", bullets=["A content point."]),
            _summary(
                1, 21, "arabic", page_types=["bibliography"], references=[_REF],
                bullets=[],
            ),
        ]  # fmt: skip
        results[1]["bullet_points"] = None
        # Round-trip through JSON as the summary log does.
        loaded: list[dict[str, Any]] = json.loads(json.dumps(results))
        return loaded

    def test_markdown_and_docx(self, tmp_path: Path) -> None:
        results = self._results()
        manager, data = build_render_context(results)
        assert [item.heading_text for item in data.page_render_items] == ["Page 20"]

        md_path = tmp_path / "doc.md"
        create_markdown_summary(
            results, md_path, "doc", citation_manager=manager, data=data
        )
        md = md_path.read_text(encoding="utf-8")
        assert "### Page 21" not in md
        assert "### Page 20" in md
        assert "- **Bibliography**: p. 21" in md
        assert "Atlantic Trade Networks" in md
        assert "*(p. 21)*" in md

        docx_path = tmp_path / "doc.docx"
        create_docx_summary(
            results, docx_path, "doc", citation_manager=manager, data=data
        )
        texts = [p.text for p in Document(str(docx_path)).paragraphs]
        assert "Page 20" in texts
        assert "Page 21" not in texts
        assert any("Atlantic Trade Networks" in t and "(p. 21)" in t for t in texts)


# ============================================================================
# repair_layout with <page_break>
# ============================================================================

# A page wrapped at a narrower width (41) than _WRAPPED_PAGE (62).
_NARROW_PAGE = (
    "Narrow column prose that wraps at a much\n"
    "smaller width.\n"
    "The second narrow line also fills its own\n"
    "column.\n"
    "A third narrow line of prose reaching the\n"
    "edge.\n"
    "And a fourth narrow line to fill the page\n"
    "out.\n"
)


class TestRepairPageBreak:
    def test_page_break_is_passthrough(self) -> None:
        assert is_passthrough_line('<page_break pdf="3"/>')
        assert is_passthrough_line('  <page_break image="3"/>')

    def test_page_break_separates_widths(self) -> None:
        text = _WRAPPED_PAGE + '<page_break pdf="2"/>\n' + _NARROW_PAGE
        repaired, _ = repair_text(text)
        assert "near the width here." in repaired
        assert "at a much smaller width." in repaired
        assert '\n<page_break pdf="2"/>\n' in repaired
        assert verify(text, repaired).passed

    def test_without_boundary_narrow_page_is_not_repaired(self) -> None:
        """Negative control: one region takes the wider page's width."""
        repaired, _ = repair_text(_WRAPPED_PAGE + "\n" + _NARROW_PAGE)
        assert "at a much smaller width." not in repaired

    def test_image_folder_txt(self) -> None:
        text = (
            "# Transcription of: scans\n# Type: Image Folder\n\n"
            '<page_break image="1"/>\n' + _NARROW_PAGE
            + '<page_break image="2"/>\n' + _NARROW_PAGE
        )  # fmt: skip
        repaired, _ = repair_text(text)
        assert repaired.count("at a much smaller width.") == 2
        assert '<page_break image="1"/>' in repaired.split("\n")
        assert '<page_break image="2"/>' in repaired.split("\n")
        assert verify(text, repaired).passed

    def test_dropped_page_break_fails_verification(self) -> None:
        original = 'Prose.\n<page_break pdf="2"/>\nMore prose.'
        result = verify(original, "Prose.\nMore prose.")
        assert not result.markers_ok
        assert not result.passed
