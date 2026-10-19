"""Tests for DOI extraction, search terms, match verification and metadata.

Covers citations/matching.py. A candidate links only when its title overlaps
the citation and a year (+/-1) or an author surname corroborates it. Only a
string DOI or OpenAlex id becomes the citation URL.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoexcerpter.rendering.citations.matching import (
    extract_doi,
    extract_metadata,
    extract_search_terms,
    verify_citation_match,
)

_SEN = "Sen, S. 1981. Poverty and Famines. Oxford."
_LANCET = "10.1016/S0140-6736(00)57123-X"
_WILEY = "10.1002/(SICI)1097-0257(199804)17:8<873::AID-SIM779>3.0.CO"


class TestExtractDoi:
    def test_extract_doi_from_text(self) -> None:
        assert extract_doi("doi: 10.1234/test.2020") is not None
        assert extract_doi("https://doi.org/10.1234/test") is not None
        assert extract_doi("10.1234/test.paper") is not None

    def test_extract_doi_none_when_missing(self) -> None:
        assert extract_doi("Smith, J. (2020). Title.") is None

    @pytest.mark.parametrize(
        ("text", "doi"),
        [
            ("doi: 10.1234/abcd", "10.1234/abcd"),
            ("See (10.1234/abc.def) for details", "10.1234/abc.def"),
            ("See (10.1234/abc.def).", "10.1234/abc.def"),
            ("[doi:10.1234/abc123]", "10.1234/abc123"),
            ("<https://doi.org/10.5555/xyz9>", "10.5555/xyz9"),
            ("see (doi:10.1234/abc)]", "10.1234/abc"),
            (f"https://doi.org/{_LANCET}", _LANCET),
            (f"({_LANCET})", _LANCET),
            (f"doi:{_WILEY}", _WILEY),
        ],
        ids=[
            "plain",
            "parenthesized",
            "paren-then-period",
            "bracketed",
            "angle-wrapped-url",
            "mixed-tail",
            "internal-parens",
            "internal-parens-inside-parens",
            "internal-angles",
        ],
    )
    def test_unbalanced_closers_stripped_and_balanced_pairs_kept(
        self, text: str, doi: str
    ) -> None:
        assert extract_doi(text) == doi


class TestSearchTerms:
    def test_extract_search_terms(self) -> None:
        text = "Smith, J. (2020). A Very Long Title for Testing. Journal, 10, 1-20."
        result = extract_search_terms(text)

        assert len(result) > 0
        assert len(result) <= 100

    def test_search_terms_capture_accented_and_apostrophe_surnames(self) -> None:
        muller = extract_search_terms(
            "Müller, H. (2010). *A Long Enough Title*. Journal."
        )
        obrien = extract_search_terms(
            "O'Brien, P. (2010). *A Long Enough Title*. Journal."
        )

        assert "Müller" in muller
        assert "O'Brien" in obrien


class TestVerifyCitationMatch:
    def test_verify_citation_match_true(self) -> None:
        """Title overlap with a corroborating year links."""
        citation = "Smith, J. (2020). Introduction to Testing."
        work_data = {"title": "Introduction to Testing", "publication_year": 2020}

        assert verify_citation_match(citation, work_data) is True

    def test_verify_citation_match_requires_corroboration(self) -> None:
        """Title overlap alone (no year or author signal) does not link."""
        citation = "Smith, J. (2020). Introduction to Testing."
        work_data = {"title": "Introduction to Testing"}

        assert verify_citation_match(citation, work_data) is False

    def test_verify_citation_match_false(self) -> None:
        citation = "Smith, J. (2020). Something Completely Different."
        work_data = {"title": "Introduction to Testing"}

        assert verify_citation_match(citation, work_data) is False

    def test_year_after_initial_corroborates_match(self) -> None:
        """The year after an author initial ("Sen, S. 1981") corroborates."""
        work: dict[str, Any] = {
            "title": "Poverty and Famines",
            "publication_year": 1981,
        }
        assert verify_citation_match(_SEN, work) is True

    def test_gross_year_mismatch_is_rejected(self) -> None:
        """A year off by more than two rejects the link despite the author."""
        work: dict[str, Any] = {
            "title": "Poverty and Famines",
            "publication_year": 2015,
            "authorships": [{"author": {"display_name": "Amartya Sen"}}],
        }
        assert verify_citation_match(_SEN, work) is False

    def test_year_off_by_decades_is_rejected(self) -> None:
        work: dict[str, Any] = {
            "title": "Introduction to Testing",
            "publication_year": 2001,
            "authorships": [{"author": {"display_name": "John Smith"}}],
        }

        assert (
            verify_citation_match("Smith, J. (1950). Introduction to Testing.", work)
            is False
        )

    def test_single_token_title_rejected(self) -> None:
        """A candidate title with one substantive token does not link."""
        work: dict[str, Any] = {
            "title": "Nations",
            "publication_year": 1983,
            "authorships": [{"author": {"display_name": "Benedict Anderson"}}],
        }

        assert (
            verify_citation_match("Anderson, B. (1983). Nations and nationalism.", work)
            is False
        )


class TestNonStringTitle:
    """A non-string title fails the match instead of raising."""

    _CITATION = "Smith, J. (1990). Atlantic Trade Networks. Oxford University Press."

    def test_numeric_title_returns_false(self) -> None:
        assert verify_citation_match(self._CITATION, {"title": 12345}) is False

    def test_numeric_display_name_returns_false(self) -> None:
        assert (
            verify_citation_match(self._CITATION, {"title": None, "display_name": 7})
            is False
        )

    def test_well_formed_payload_verifies(self) -> None:
        assert (
            verify_citation_match(
                self._CITATION,
                {"title": "Atlantic Trade Networks", "publication_year": 1990},
            )
            is True
        )


class TestExtractMetadata:
    def test_extract_metadata(self, mock_openalex_response: dict[str, Any]) -> None:
        result = extract_metadata(mock_openalex_response)

        assert result["title"] == "Introduction to Testing"
        assert result["publication_year"] == 2020
        assert "10.1234/test.2020.001" in result["doi"]
        assert len(result["authors"]) > 0

    def test_numeric_doi_and_null_id_yield_no_url(self) -> None:
        assert extract_metadata({"doi": 12345, "id": None})["url"] is None

    def test_numeric_doi_falls_through_to_string_id(self) -> None:
        metadata = extract_metadata({"doi": 123, "id": "https://openalex.org/W1"})

        assert metadata["url"] == "https://openalex.org/W1"

    def test_string_doi_wins_over_id(self) -> None:
        metadata = extract_metadata(
            {"doi": "https://doi.org/10.1234/abc", "id": "https://openalex.org/W1"}
        )

        assert metadata["url"] == "https://doi.org/10.1234/abc"
