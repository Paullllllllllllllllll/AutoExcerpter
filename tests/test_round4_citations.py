"""Round 4: page-marker year-guard regression tests for rendering/citations.

Commit 2f1aa58 added German page markers (``S.``/``ss.``) to the page-marker
stripping regexes. That made an author initial adjoining a year — ``Sen, S.
1981`` or ``Krugman, P. 1991`` — match as a page marker, destroying year
extraction and (worse) collapsing distinct editions onto one normalized key.
These tests pin the year-guard fix: a marker followed by a lone plausible year
is left intact, while page ranges and non-year page numbers still strip.
"""

from __future__ import annotations

from typing import Any

from rendering.citations import Citation, CitationManager, _extract_year


class TestInitialBeforeYearNotAPageMarker:
    """An author initial ("S.", "P.") before a year must not read as a page."""

    def test_sen_initial_year_extracted(self) -> None:
        """ "Sen, S. 1981" yields year 1981, not None."""
        assert (
            _extract_year("Sen, S. 1981. Poverty and Famines. Oxford: Clarendon.")
            == 1981
        )

    def test_krugman_initial_year_extracted(self) -> None:
        """ "Krugman, P. 1991" yields year 1991."""
        assert _extract_year("Krugman, P. 1991. Geography and Trade. MIT.") == 1991

    def test_temin_initial_year_extracted(self) -> None:
        """ "Temin, P. 2002" yields year 2002 (20xx branch of the guard)."""
        assert _extract_year("Temin, P. 2002. Price Behavior. Journal.") == 2002

    def test_umlaut_initial_unaffected(self) -> None:
        """A non-marker initial after an umlaut surname keeps its year."""
        assert _extract_year("Müller, P. 1985. Ein Werk. Verlag.") == 1985

    def test_other_letter_initials_unaffected(self) -> None:
        """Initials that were never markers (e.g. "K.") still extract years."""
        assert _extract_year("Arrow, K. 1963. Social Choice. Wiley.") == 1963


class TestDistinctEditionsDoNotCollapse:
    """Different-year editions of one work must keep distinct normalized keys."""

    _E1 = "Sen, S. 1981. Poverty and Famines. Oxford."
    _E2 = "Sen, S. 1999. Poverty and Famines. Oxford."

    def test_editions_have_distinct_years(self) -> None:
        """Each edition resolves its own year."""
        assert Citation(raw_text=self._E1).year == 1981
        assert Citation(raw_text=self._E2).year == 1999

    def test_editions_have_distinct_keys(self) -> None:
        """The two Sen editions no longer share a normalized key."""
        assert (
            Citation(raw_text=self._E1).normalized_key
            != Citation(raw_text=self._E2).normalized_key
        )

    def test_editions_stay_separate_through_consolidate(self) -> None:
        """Distinct editions survive dedup + fuzzy consolidation as two entries."""
        manager = CitationManager()
        manager.add_citations([self._E1], 1)
        manager.add_citations([self._E2], 2)
        manager.consolidate()

        assert len(manager.citations) == 2
        years = sorted(c.year for c in manager.citations.values() if c.year)
        assert years == [1981, 1999]


class TestPageRangesStillStrip:
    """Page ranges (the motivating "S. 1066-1071" case) must still strip fully."""

    def test_german_page_range_not_read_as_year(self) -> None:
        """ "S. 1066-1071" yields no bogus year 1066 (range alternative first)."""
        assert _extract_year("Meyer. Some Work. S. 1066-1071.") is None

    def test_english_page_range_stripped(self) -> None:
        """ "pp. 123-145" strips before year scanning (no bogus year)."""
        assert _extract_year("Meyer. Some Work. pp. 123-145.") is None

    def test_real_year_survives_alongside_page_range(self) -> None:
        """A genuine year is still found next to a stripped German page range."""
        assert _extract_year("Meyer (1998). Some Work. S. 1066-1071.") == 1998

    def test_page_range_stripped_from_normalized_key(self) -> None:
        """A trailing page range does not enter the comparison text / key."""
        with_pages = Citation(raw_text="Meyer. Some Work. pp. 123-145.")
        without = Citation(raw_text="Meyer. Some Work.")
        assert with_pages.comparison_text == without.comparison_text
        assert with_pages.normalized_key == without.normalized_key

    def test_non_year_single_page_still_strips(self) -> None:
        """A lone non-year page ("S. 42") still strips (guard only spares years)."""
        assert _extract_year("Meyer (1998). Some Work. S. 42.") == 1998
        with_page = Citation(raw_text="Meyer. Some Work. S. 42.")
        without = Citation(raw_text="Meyer. Some Work.")
        assert with_page.comparison_text == without.comparison_text


class TestFourDigitPageTradeoff:
    """A four-digit single German page is read as a year — accepted trade-off."""

    def test_single_four_digit_page_reads_as_year(self) -> None:
        """ "S. 1815" (a big-volume page) is treated as a year, by design."""
        assert _extract_year("Meyer. Some Work. S. 1815.") == 1815


class TestVerifyMatchYearGuardUnaffected:
    """The initial-before-year fix must not weaken _verify_citation_match."""

    def test_initial_year_now_corroborates_match(self) -> None:
        """With the year now recoverable, an exact-year candidate links."""
        work: dict[str, Any] = {
            "title": "Poverty and Famines",
            "publication_year": 1981,
        }
        assert (
            CitationManager()._verify_citation_match(
                "Sen, S. 1981. Poverty and Famines. Oxford.", work
            )
            is True
        )

    def test_gross_year_mismatch_still_rejected(self) -> None:
        """A recovered year off by more than two still disqualifies the link."""
        work: dict[str, Any] = {
            "title": "Poverty and Famines",
            "publication_year": 2015,
            "authorships": [{"author": {"display_name": "Amartya Sen"}}],
        }
        assert (
            CitationManager()._verify_citation_match(
                "Sen, S. 1981. Poverty and Famines. Oxford.", work
            )
            is False
        )
