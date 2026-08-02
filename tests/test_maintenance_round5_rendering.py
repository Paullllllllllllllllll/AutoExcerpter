"""Round 5: rendering/page-numbering maintenance regression tests.

One focused regression per verified fix:

1. Non-string debris in ``page_information.page_types`` (a dict from a corrupt
   or reused log) is filtered in ``parse_page_information`` instead of becoming
   the primary section type and raising ``TypeError`` as a dict key.
2. ``_verify_citation_match`` rejects a truthy non-string title instead of
   raising inside ``_fold`` and aborting enrichment for the whole document.
3. ``_extract_metadata_from_response`` only accepts a string ``doi``/``id`` as
   the citation URL, so no non-string poisons the writers or the cache.
4. ``_resolve_partials`` drops a partial whose comparison text strips to an
   empty token set instead of merging it into the block's only full reference
   on zero evidence.
5. ``_extract_doi`` strips an unmatched trailing ``)`` from a parenthesized
   DOI while preserving the internal parens of ``10.1016/S0140-6736(00)…``.
6. ``collapse_internal_newlines`` also collapses a lone carriage return, which
   sanitize_for_xml keeps and which splits list items in most renderers.
7. The Markdown writer pins ``newline="\\n"``, so the .md file carries no CRLF
   on Windows.
8. ``_extract_volume`` scans past a match caught by the 1500-2099 year guard
   instead of returning None on the first implausible candidate.
9. ``parse_markdown_emphasis`` recognizes ``***x***`` as combined bold+italic
   instead of emitting stray literal asterisk runs around the bold text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from pipeline.page_numbering import PageNumberProcessor
from rendering.citations import CitationManager, _extract_volume
from rendering.docx import (
    add_formatted_text_to_paragraph,
    parse_markdown_emphasis,
    strip_markdown_emphasis,
)
from rendering.markdown import create_markdown_summary
from rendering.summary import collapse_internal_newlines


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _page(page_types: Any, index: int = 0) -> dict[str, Any]:
    """Build a minimal summary result carrying *page_types*."""
    return {
        "page": index + 1,
        "original_input_order_index": index,
        "page_information": {
            "page_number_visible": True,
            "page_types": page_types,
        },
        "bullet_points": [],
    }


def _mock_paragraph() -> MagicMock:
    """Return a paragraph mock whose ``add_run`` yields a fresh run each call."""
    paragraph = MagicMock()
    runs: list[MagicMock] = []

    def _add_run(text: str = "") -> MagicMock:
        run = MagicMock()
        run.text = text
        run.bold = None
        run.italic = None
        runs.append(run)
        return run

    paragraph.add_run.side_effect = _add_run
    paragraph.runs = runs
    return paragraph


# ---------------------------------------------------------------------------
# Fix 1: non-string page_types debris is filtered before it becomes a dict key
# ---------------------------------------------------------------------------
class TestPageTypesDebrisFiltered:
    """Dict debris must not reach the section-grouping dict key."""

    def test_dict_debris_does_not_crash_and_folds_to_content(self) -> None:
        processor = PageNumberProcessor()

        result = processor.adjust_and_sort_page_numbers([_page([{"bad": "debris"}])])

        assert len(result) == 1
        assert result[0]["page_information"]["page_types"] == ["content"]

    def test_string_entries_survive_alongside_debris(self) -> None:
        processor = PageNumberProcessor()

        _, _, page_types, _, _, _ = processor.parse_page_information(
            _page(["preface", {"bad": "debris"}, 42])
        )

        assert page_types == ["preface"]

    def test_valid_list_passes_through_unchanged(self) -> None:
        """Negative control: a clean list is not touched by the filter."""
        processor = PageNumberProcessor()

        _, _, page_types, _, _, _ = processor.parse_page_information(
            _page(["appendix", "content"])
        )

        assert page_types == ["appendix", "content"]


# ---------------------------------------------------------------------------
# Fix 2: a non-string OpenAlex title is rejected, not folded
# ---------------------------------------------------------------------------
class TestNonStringTitleRejected:
    """A JSON number title must fail the match rather than raise."""

    _CITATION = "Smith, J. (1990). Atlantic Trade Networks. Oxford University Press."

    def test_numeric_title_returns_false(self) -> None:
        manager = CitationManager()

        assert manager._verify_citation_match(self._CITATION, {"title": 12345}) is False

    def test_numeric_display_name_returns_false(self) -> None:
        manager = CitationManager()

        assert (
            manager._verify_citation_match(
                self._CITATION, {"title": None, "display_name": 7}
            )
            is False
        )

    def test_genuine_match_still_verifies(self) -> None:
        """Negative control: a well-formed payload still links."""
        manager = CitationManager()

        assert (
            manager._verify_citation_match(
                self._CITATION,
                {"title": "Atlantic Trade Networks", "publication_year": 1990},
            )
            is True
        )


# ---------------------------------------------------------------------------
# Fix 3: only a string doi/id becomes the citation URL
# ---------------------------------------------------------------------------
class TestCitationUrlTypeGuard:
    """A non-string identifier must not reach citation.url (nor the cache)."""

    def test_numeric_doi_and_null_id_yield_no_url(self) -> None:
        manager = CitationManager()

        metadata = manager._extract_metadata_from_response({"doi": 12345, "id": None})

        assert metadata["url"] is None

    def test_numeric_doi_falls_through_to_string_id(self) -> None:
        manager = CitationManager()

        metadata = manager._extract_metadata_from_response(
            {"doi": 123, "id": "https://openalex.org/W1"}
        )

        assert metadata["url"] == "https://openalex.org/W1"

    def test_string_doi_still_wins(self) -> None:
        """Negative control: the ordinary payload is unchanged."""
        manager = CitationManager()

        metadata = manager._extract_metadata_from_response(
            {"doi": "https://doi.org/10.1234/abc", "id": "https://openalex.org/W1"}
        )

        assert metadata["url"] == "https://doi.org/10.1234/abc"


# ---------------------------------------------------------------------------
# Fix 4: an empty-token partial never merges on containment
# ---------------------------------------------------------------------------
class TestEmptyTokenPartialDropped:
    """A bare-URL partial is a subset of every full; that is zero evidence."""

    _FULL = "Https, X. On Something. Journal"
    _BARE_URL = "https://example.com/foo"

    def test_bare_url_partial_does_not_merge_its_page(self) -> None:
        manager = CitationManager()
        manager.add_citations([self._FULL], 1)
        manager.add_citations([(self._BARE_URL, True)], 2)

        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.raw_text == self._FULL
        assert survivor.pages == {1}

    def test_legitimate_subset_partial_still_merges(self) -> None:
        """Negative control: an author-year stub still folds into its full."""
        manager = CitationManager()
        manager.add_citations(
            ["Smith, J. (1990). Atlantic Trade Networks. Oxford University Press."], 1
        )
        manager.add_citations([("(Smith 1990)", True)], 2)

        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.pages == {1, 2}


# ---------------------------------------------------------------------------
# Fix 5: balance-aware trailing-paren stripping in _extract_doi
# ---------------------------------------------------------------------------
class TestDoiParenBalance:
    """Unmatched closers go; internal parens stay."""

    def test_parenthesized_doi_loses_trailing_paren(self) -> None:
        manager = CitationManager()

        assert (
            manager._extract_doi("See (10.1234/abc.def) for details")
            == "10.1234/abc.def"
        )

    def test_trailing_paren_then_period(self) -> None:
        manager = CitationManager()

        assert manager._extract_doi("See (10.1234/abc.def).") == "10.1234/abc.def"

    def test_internal_parens_are_preserved(self) -> None:
        """Negative control: a Lancet-style DOI keeps its balanced parens."""
        manager = CitationManager()

        assert (
            manager._extract_doi("https://doi.org/10.1016/S0140-6736(00)57123-X")
            == "10.1016/S0140-6736(00)57123-X"
        )

    def test_internal_parens_preserved_inside_a_parenthesized_citation(self) -> None:
        manager = CitationManager()

        assert (
            manager._extract_doi("(10.1016/S0140-6736(00)57123-X)")
            == "10.1016/S0140-6736(00)57123-X"
        )

    def test_plain_doi_unchanged(self) -> None:
        manager = CitationManager()

        assert manager._extract_doi("doi: 10.1234/abcd") == "10.1234/abcd"


# ---------------------------------------------------------------------------
# Fix 6: a lone carriage return collapses like a newline
# ---------------------------------------------------------------------------
class TestCollapseLoneCarriageReturn:
    """A raw \\r must not survive into .md bullets or DOCX runs."""

    def test_lone_cr_collapses(self) -> None:
        assert collapse_internal_newlines("first\rsecond") == "first second"

    def test_padded_cr_collapses_to_single_space(self) -> None:
        assert collapse_internal_newlines("first  \r  second") == "first second"

    def test_lf_and_crlf_behavior_unchanged(self) -> None:
        """Negative control: the pre-existing cases keep their result."""
        assert collapse_internal_newlines("first\nsecond") == "first second"
        assert collapse_internal_newlines("first\r\nsecond") == "first second"
        assert collapse_internal_newlines("first \n second") == "first second"

    def test_text_without_breaks_unchanged(self) -> None:
        assert collapse_internal_newlines("first second") == "first second"


# ---------------------------------------------------------------------------
# Fix 7: the Markdown writer emits LF only
# ---------------------------------------------------------------------------
class TestMarkdownNewlines:
    """write_text must not translate "\\n" to CRLF on Windows."""

    @patch("rendering.markdown.CitationManager")
    def test_written_file_contains_no_cr(
        self, mock_cm_class: MagicMock, tmp_path: Path
    ) -> None:
        mock_cm = MagicMock()
        mock_cm.citations = {}
        mock_cm_class.return_value = mock_cm
        output_path = tmp_path / "summary.md"
        summary_results = [
            {
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["A bullet point", "Another bullet point"],
                "references": [],
            }
        ]

        create_markdown_summary(summary_results, output_path, "Test")

        raw = output_path.read_bytes()
        assert b"\r" not in raw
        assert b"\n" in raw


# ---------------------------------------------------------------------------
# Fix 8: the volume scan continues past a year-window match
# ---------------------------------------------------------------------------
class TestVolumeScanPastYearGuard:
    """An implausible first candidate is discarded, not returned."""

    def test_volume_found_after_year_window_match(self) -> None:
        assert _extract_volume("The Band 1968 Story, vol. 3") == 3

    def test_roman_volume_found_after_year_window_match(self) -> None:
        assert _extract_volume("The Band 1968 Story, t. II") == 2

    def test_year_alone_is_still_rejected(self) -> None:
        """Negative control: the year guard itself is intact."""
        assert _extract_volume("The Band 1968 Story") is None
        assert _extract_volume("Band 1750") is None

    def test_ordinary_volume_unchanged(self) -> None:
        assert _extract_volume("Vol. 5") == 5


# ---------------------------------------------------------------------------
# Fix 9: *** is bold+italic, not bold flanked by literal asterisks
# ---------------------------------------------------------------------------
class TestTripleEmphasis:
    """``***x***`` must not leak literal asterisk runs."""

    def test_triple_emphasis_is_one_segment(self) -> None:
        assert parse_markdown_emphasis("***Title***") == [("bold_italic", "Title")]

    def test_triple_emphasis_inside_prose(self) -> None:
        segments = parse_markdown_emphasis("a ***b*** c")

        assert segments == [("text", "a "), ("bold_italic", "b"), ("text", " c")]
        assert all("*" not in piece for _kind, piece in segments)

    def test_docx_run_carries_both_flags(self) -> None:
        paragraph = _mock_paragraph()

        add_formatted_text_to_paragraph(paragraph, "***Title***")

        assert len(paragraph.runs) == 1
        assert paragraph.runs[0].bold is True
        assert paragraph.runs[0].italic is True

    def test_strip_handles_triple_emphasis(self) -> None:
        assert strip_markdown_emphasis("***x*** and **b** and *i*") == "x and b and i"

    def test_single_and_double_emphasis_unchanged(self) -> None:
        """Negative control: the ordinary tokenizer output is untouched."""
        assert parse_markdown_emphasis("**word**") == [("bold", "word")]
        assert parse_markdown_emphasis("*word*") == [("italic", "word")]
        assert parse_markdown_emphasis("87* (2)") == [("text", "87* (2)")]
