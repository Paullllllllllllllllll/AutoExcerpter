"""The Markdown summary writer in rendering/markdown.py.

Covers the title and metadata line, the Document Structure section, the page
summaries (headings, bullets, LaTeX, filtering, order, malformed input), the
linked reference line and the written file (LF newlines, final newline, atomic
write). The full layout of a rich document, including the references block, is
pinned by the read-back snapshot in test_readback_snapshots.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import autoexcerpter.rendering.markdown as markdown_module
from autoexcerpter.rendering.citations import CitationManager
from autoexcerpter.rendering.markdown import create_markdown_summary
from autoexcerpter.rendering.summary import build_render_context, prepare_summary_data


def _page(
    number: int | None,
    bullets: list[str],
    number_type: str = "arabic",
    page_types: list[str] | None = None,
    references: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "page_information": {
            "page_number_integer": number,
            "page_number_type": number_type,
            "page_types": page_types or ["content"],
        },
        "bullet_points": bullets,
        "references": references or [],
    }


def _render(tmp_path: Path, results: list[dict[str, Any]], name: str = "Test") -> str:
    output_path = tmp_path / "summary.md"
    create_markdown_summary(results, output_path, name)
    return output_path.read_text(encoding="utf-8")


# ============================================================================
# Title and Document Structure
# ============================================================================
class TestMarkdownStructure:
    """Title, metadata line and Document Structure section."""

    def test_h1_without_summary_prefix_and_page_grouping(self, tmp_path: Path) -> None:
        """H1 is the bare document name; pages are grouped under Page Summaries."""
        content = _render(tmp_path, [_page(1, ["Point A"])], "Doc Name")

        assert content.startswith("# Doc Name")
        assert "# Summary of" not in content
        assert "## Page Summaries" in content
        assert "### Page 1" in content
        # No horizontal rule when there are no references.
        assert "\n---\n" not in content

    def test_contains_metadata(self, tmp_path: Path) -> None:
        content = _render(tmp_path, [_page(1, ["Point 1"])])

        assert "*Summary generated" in content
        assert "1 total pages*" in content

    def test_empty_results(self, tmp_path: Path) -> None:
        """No results give the title and a zero page count."""
        content = _render(tmp_path, [], "Empty Doc")

        assert "# Empty Doc" in content
        assert "0 total pages*" in content

    def test_structure_section_for_toc_pages(self, tmp_path: Path) -> None:
        """Table of contents pages appear in the Document Structure section."""
        results = [
            _page(3, [], number_type="roman", page_types=["table_of_contents"]),
            _page(1, ["Some content"]),
        ]

        content = _render(tmp_path, results)

        assert "## Document Structure" in content
        assert "Table of Contents" in content


# ============================================================================
# Page summaries
# ============================================================================
class TestCreateMarkdownSummary:
    """Page headings and bullets of create_markdown_summary."""

    def test_contains_bullet_points(self, tmp_path: Path) -> None:
        content = _render(tmp_path, [_page(1, ["First point", "Second point"])])

        assert "- First point" in content
        assert "- Second point" in content

    def test_roman_numeral_pages(self, tmp_path: Path) -> None:
        """A roman page number is written as a lowercase numeral."""
        results = [_page(3, ["Preface content"], number_type="roman")]

        assert "### Page iii" in _render(tmp_path, results)

    def test_appendix_page_heading(self, tmp_path: Path) -> None:
        """Appendix pages get the [Appendix] prefix in their heading."""
        content = _render(
            tmp_path, [_page(100, ["Appendix data"], page_types=["appendix"])]
        )

        assert "### [Appendix] Page 100" in content
        assert "## Page Summaries" in content

    def test_unnumbered_pages(self, tmp_path: Path) -> None:
        """An unnumbered page is headed by its scan position."""
        results = [
            _page(None, ["Unnumbered content"], number_type="none"),
            _page(1, ["Valid content"]),
        ]

        content = _render(tmp_path, results)

        assert "### [No printed number; PDF p. 1]" in content
        assert "### Page 1" in content

    def test_filters_empty_pages(self, tmp_path: Path) -> None:
        results = [_page(1, ["Valid content"]), _page(2, [])]

        content = _render(tmp_path, results)

        assert "### Page 1" in content
        assert "### Page 2" not in content

    def test_preserves_latex_formulas(self, tmp_path: Path) -> None:
        results = [_page(1, ["The formula $x + y = z$ is important"])]

        assert "$x + y = z$" in _render(tmp_path, results)

    def test_multiple_pages(self, tmp_path: Path) -> None:
        """Pages are written in order."""
        results = [_page(1, ["Page 1 content"]), _page(2, ["Page 2 content"])]

        content = _render(tmp_path, results)

        assert content.index("### Page 1") < content.index("### Page 2")

    def test_total_pages_counts_filtered_pages(self, tmp_path: Path) -> None:
        """The metadata line reports the source page count before filtering."""
        results = [
            _page(1, ["Real content bullet"]),
            _page(2, [], page_types=["blank"]),
        ]
        results[1]["bullet_points"] = None
        data = prepare_summary_data(results, CitationManager())
        assert len(data.filtered_results) == 1
        assert data.source_page_count == 2

        assert "2 total pages*" in _render(tmp_path, results)


class TestMarkdownMalformedInput:
    """Failed pages, non-string bullets, stray newlines and control characters."""

    def test_error_placeholder_is_visible(self, tmp_path: Path) -> None:
        result = _page(3, ["[Error generating summary: request timed out]"])
        result["page_information"]["page_types"] = ["other"]
        result["error"] = "request timed out"

        content = _render(tmp_path, [result])

        assert "[Error generating summary: request timed out]" in content

    def test_only_string_bullets_are_rendered(self, tmp_path: Path) -> None:
        result = _page(1, [])
        result["bullet_points"] = [None, 123, "real bullet"]

        content = _render(tmp_path, [result])

        assert "- real bullet" in content
        assert "123" not in content

    def test_bullet_newline_collapsed(self, tmp_path: Path) -> None:
        content = _render(tmp_path, [_page(1, ["First part\nSecond part"])])

        assert "- First part Second part" in content

    def test_citation_newline_collapsed(self, tmp_path: Path) -> None:
        result = _page(
            1,
            ["ok"],
            references=[
                {
                    "citation": "Smith, J. (2020). *A Long\nBroken Title*. Press.",
                    "is_partial": False,
                }
            ],
        )

        with patch.object(markdown_module, "enrich_if_enabled"):
            content = _render(tmp_path, [result])

        assert "A Long Broken Title" in content

    def test_control_character_in_heading_is_removed(self, tmp_path: Path) -> None:
        result = _page(7, ["A bullet"])
        result["page_information"]["page_number_integer"] = "7\x0b"

        content = _render(tmp_path, [result])

        assert "### Page 7" in content
        assert "\x0b" not in content


# ============================================================================
# References
# ============================================================================
class TestMarkdownReferences:
    """The Consolidated References line of a linked citation."""

    def test_linked_reference_escapes_brackets_and_shows_metadata(
        self, tmp_path: Path
    ) -> None:
        """Brackets in the link text and spaces in the URL are escaped; the
        pages and the DOI and year follow the link."""
        results = [
            _page(
                1,
                ["Content"],
                references=[
                    {
                        "citation": "Author (2020). [Reprint] Title.",
                        "is_partial": False,
                    }
                ],
            )
        ]
        manager, data = build_render_context(results, openalex_enabled=False)
        (citation,) = manager.citations.values()
        citation.url = "https://example.com/a b"
        citation.doi = "10.1234/test"
        citation.metadata = {"publication_year": 2020}
        output_path = tmp_path / "summary.md"

        create_markdown_summary(results, output_path, "Test", manager, data)

        content = output_path.read_text(encoding="utf-8")
        assert "## Consolidated References" in content
        assert (
            "1. [Author (2020). \\[Reprint\\] Title.](https://example.com/a%20b) "
            "*(p. 1)* *[DOI: 10.1234/test, Year: 2020]*"
        ) in content

    @staticmethod
    def _render_with_url(tmp_path: Path, url: str) -> str:
        results = [
            _page(
                1,
                ["A bullet"],
                references=[
                    {
                        "citation": "Smith, J. (2020). *A Linked Work*. Press.",
                        "is_partial": False,
                    }
                ],
            )
        ]
        manager = CitationManager()
        data = prepare_summary_data(results, manager)
        for citation in manager.citations.values():
            citation.url = url
        output_path = tmp_path / "summary.md"

        create_markdown_summary(
            results, output_path, "Doc", citation_manager=manager, data=data
        )
        return output_path.read_text(encoding="utf-8")

    def test_angle_brackets_escaped_in_link_url(self, tmp_path: Path) -> None:
        content = self._render_with_url(
            tmp_path, "https://doi.org/10.1000/j.<seg>.2020"
        )

        assert "%3Cseg%3E" in content
        assert "<seg>" not in content

    def test_parentheses_escaped_in_link_url(self, tmp_path: Path) -> None:
        content = self._render_with_url(
            tmp_path, "https://doi.org/10.1016/S0140-6736(00)02800-3"
        )

        assert "%28" in content and "%29" in content


# ============================================================================
# The written file
# ============================================================================
class TestMarkdownFile:
    """Newlines and the atomic write of the .md file."""

    def test_written_file_contains_no_cr(self, tmp_path: Path) -> None:
        output_path = tmp_path / "summary.md"

        create_markdown_summary(
            [_page(1, ["A bullet point", "Another bullet point"])], output_path, "Test"
        )

        raw = output_path.read_bytes()
        assert b"\r" not in raw
        assert b"\n" in raw

    def test_file_ends_with_newline(self, tmp_path: Path) -> None:
        content = _render(tmp_path, [_page(1, ["A bullet"])], "Doc")

        assert content.endswith("\n")
        assert not content.endswith("\n\n\n")

    def test_empty_document_ends_with_newline(self, tmp_path: Path) -> None:
        assert _render(tmp_path, [], "Empty Doc").endswith("\n")

    def test_successful_write_leaves_no_temp_file(self, tmp_path: Path) -> None:
        out = tmp_path / "doc.md"

        create_markdown_summary([], out, "My Document")

        assert "# My Document" in out.read_text(encoding="utf-8")
        assert not out.with_name(out.name + ".tmp").exists()

    def test_failed_replace_keeps_previous_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "doc.md"
        out.write_text("PRIOR COMPLETE MD", encoding="utf-8")

        def _boom(_src: Any, _dst: Any) -> None:
            raise OSError("replace failed mid-swap")

        monkeypatch.setattr("autoexcerpter.rendering.markdown.os.replace", _boom)

        with pytest.raises(OSError, match="replace failed"):
            create_markdown_summary([], out, "My Document")

        assert out.read_text(encoding="utf-8") == "PRIOR COMPLETE MD"
        assert not out.with_name(out.name + ".tmp").exists()

    def test_non_os_error_removes_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_src: Any, _dst: Any) -> None:
            raise ValueError("not an OSError")

        output_path = tmp_path / "summary.md"
        monkeypatch.setattr("autoexcerpter.rendering.markdown.os.replace", _boom)

        with pytest.raises(ValueError, match="not an OSError"):
            create_markdown_summary([_page(1, ["A bullet"])], output_path, "Doc")

        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
