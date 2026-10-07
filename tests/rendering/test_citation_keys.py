"""Tests for the citation record and its deduplication keys (citations/model.py).

The normalized key folds accents, case and punctuation, drops URLs, DOIs and
page markers and editor markers, and keeps the year and the volume, so distinct
editions never share a key. An author initial before a year ("Sen, S. 1981") is
not a page marker.
"""

from __future__ import annotations

import pytest

from autoexcerpter.rendering.citations import Citation, CitationManager
from autoexcerpter.rendering.citations.model import (
    extract_volume,
    extract_year,
    first_author_surname,
    fold,
    token_set,
)


class TestCitation:
    def test_same_citations_same_key(self) -> None:
        citation1 = Citation(raw_text="Smith, J. (2020). Test Title. Publisher.")
        citation2 = Citation(raw_text="Smith, J. (2020). Test Title. Publisher.")

        assert citation1.normalized_key == citation2.normalized_key

    def test_add_page(self) -> None:
        citation = Citation(raw_text="Test citation")

        citation.add_page(5)
        citation.add_page(10)
        citation.add_page(5)

        # A bare int is a printed arabic page number.
        assert citation.locators == {
            ("printed-arabic", 5, False),
            ("printed-arabic", 10, False),
        }

    @pytest.mark.parametrize(
        ("pages", "expected"),
        [
            ([], ""),
            ([5], "p. 5"),
            ([5, 10], "pp. 5, 10"),
            ([5, 6, 7, 10], "pp. 5-7, 10"),
        ],
        ids=["empty", "single", "list", "consecutive-range"],
    )
    def test_page_range_str(self, pages: list[int], expected: str) -> None:
        citation = Citation(raw_text="Test citation")
        for page in pages:
            citation.add_page(page)

        assert citation.get_page_range_str() == expected


class TestExtractYear:
    @pytest.mark.parametrize(
        ("text", "year"),
        [
            # An author initial before a year is not a page marker.
            ("Sen, S. 1981. Poverty and Famines. Oxford: Clarendon.", 1981),
            ("Krugman, P. 1991. Geography and Trade. MIT.", 1991),
            ("Temin, P. 2002. Price Behavior. Journal.", 2002),
            ("Müller, P. 1985. Ein Werk. Verlag.", 1985),
            ("Arrow, K. 1963. Social Choice. Wiley.", 1963),
            ("Sraffa, P. 1951-1973. The Works of David Ricardo. Cambridge.", 1951),
            ("Sraffa, P. 1951-73. Works. Cambridge.", 1951),
            # Page markers and page ranges strip before the year scan.
            ("Meyer. Some Work. pp. 123-145.", None),
            ("Meyer (1998). Some Work. S. 42.", 1998),
            ("Meyer. Some Work. S. 1066-1071.", None),
            ("Meyer (1998). Some Work. S. 1066-1071.", 1998),
            ("Meyer (1998). Some Work. S. 12-34.", 1998),
            ("Meyer. Some Work. S. 1400-1600.", None),
            ("Some Title, S. 42-58, o.J.", None),
            # Multi-letter markers are never initials and strip any number.
            ("Meyer. Some Work. pp. 1850.", None),
            ("Meyer. Some Work. fol. 1200.", None),
            ("Meyer. Some Work. ss. 1980.", None),
            ("Meyer (1998). Some Work. pp. 1850.", 1998),
            ("Meyer. Some Work. pp. 1951-1973.", None),
            # Accepted trade-off: a lone four-digit page after "S." is a year.
            ("Meyer. Some Work. S. 1815.", 1815),
            # DOI registrant prefixes are not years.
            ("Muller. A Study of Something. 10.1016/foobar", None),
            ("Jones. Some Work on Genetics. doi:10.1111/abcd", None),
            (
                "Smith. A Title (2019). https://doi.org/10.1016/j.example.2018.05.001",
                2019,
            ),
        ],
        ids=[
            "initial-1981",
            "initial-1991",
            "initial-2002",
            "umlaut-surname",
            "other-initial",
            "initial-year-range",
            "initial-abbreviated-year-range",
            "pp-range",
            "s-page-after-year",
            "s-range-below-1500",
            "year-beside-s-range-below-1500",
            "s-ordinary-range",
            "s-mixed-range",
            "s-range-undated",
            "pp-four-digit-page",
            "fol-four-digit-page",
            "ss-four-digit-page",
            "year-with-pp-four-digit-page",
            "pp-year-shaped-range",
            "s-four-digit-page",
            "doi-prefix",
            "doi-colon",
            "year-with-doi-url",
        ],
    )
    def test_extract_year(self, text: str, year: int | None) -> None:
        assert extract_year(text) == year

    def test_reprint_year_folds_to_earliest_either_order(self) -> None:
        assert extract_year("Marx (1976 [1867]). Capital.") == 1867
        assert extract_year("Marx (1867 [1976]). Capital.") == 1867


class TestExtractVolume:
    @pytest.mark.parametrize(
        ("text", "volume"),
        [
            ("Smith (2020). *Collected Works*, vol. 1. CUP.", 1),
            ("Smith (2020). *W*, vol. 3. CUP.", 3),
            ("Vol. 5", 5),
            ("Smith (2020). *W*, vol. xiv. CUP.", 14),
            ("Dupont. *Oeuvres*, t. II. Paris.", 2),
            ("Dupont. *Oeuvres*, tome iv. Paris.", 4),
            ("Nature, Vol. 585, pp. 10-12", 585),
            ("Gesammelte Werke, Bd. 2, Berlin", 2),
            # The scan continues past a match in the publication-year window.
            ("The Band 1968 Story, vol. 3", 3),
            ("The Band 1968 Story, t. II", 2),
            # Words, years and author initials are not volumes.
            ("Meyer, H. Studien. Bd. im Erscheinen.", None),
            ("Dupont. Oeuvres, t. dix. Paris.", None),
            ("Smith. Volvic Waters, vol. vic.", None),
            ("The Band 1968 Story", None),
            ("Band 1750", None),
            ("Smith, T. 1990. The Wealth. CUP.", None),
            ("Smith, T. 190. Title.", None),
            ("Smith. It took 5 years to write.", None),
        ],
        ids=[
            "arabic-1",
            "arabic-3",
            "capitalized-vol",
            "roman-lowercase",
            "tome-uppercase",
            "tome-word-roman",
            "large-journal-volume",
            "german-band",
            "vol-after-year-window",
            "tome-after-year-window",
            "german-im",
            "french-dix",
            "malformed-roman",
            "year-after-band",
            "band-year",
            "initial-before-year",
            "initial-before-number",
            "bare-t-before-number",
        ],
    )
    def test_extract_volume(self, text: str, volume: int | None) -> None:
        assert extract_volume(text) == volume

    def test_german_bd_volumes(self) -> None:
        first = Citation(
            raw_text="Weber, M. (1920). *Gesammelte Aufsätze*, Bd. 1. Mohr."
        )
        second = Citation(
            raw_text="Weber, M. (1920). *Gesammelte Aufsätze*, Bd. 2. Mohr."
        )

        assert (first.volume, second.volume) == (1, 2)


class TestFirstAuthorSurname:
    """Particles, initials, brackets and non-ASCII letters around the surname."""

    def test_parenthesized_partial(self) -> None:
        assert first_author_surname("(Smith, 1990, S. 12)") == "smith"

    def test_particle_surname_orderings_agree(self) -> None:
        assert first_author_surname("van der Berg, J. (2000). X.") == "berg"
        assert first_author_surname("Berg, J. van der (2000). X.") == "berg"

    def test_non_decomposable_letter_is_transliterated(self) -> None:
        """Ł becomes l, so the surname also matches its ASCII spelling."""
        assert first_author_surname("Łukasz, K. (2010). X.") == "lukasz"

    def test_umlaut_surname_folds_to_ascii(self) -> None:
        assert first_author_surname("Müller, H. (2010). X.") == "muller"

    def test_leading_initial_is_skipped(self) -> None:
        assert first_author_surname("A. Smith (2020). X.") == "smith"


class TestEditorMarkers:
    """ed/eds/trans are stripped as standalone markers, never inside words."""

    def test_education_keeps_its_ed(self) -> None:
        citation = Citation(raw_text="Smith, J. 1990. Education Reform.")

        assert "education" in citation.comparison_text

    def test_trans_compound_keeps_its_trans(self) -> None:
        """'Trans-Atlantic' and 'Atlantic' titles get distinct keys."""
        trans = Citation(raw_text="Smith, J. (1990). Trans-Atlantic Trade Networks.")
        atlantic = Citation(raw_text="Smith, J. (1990). Atlantic Trade Networks.")

        assert "trans" in trans.comparison_text
        assert trans.comparison_text != atlantic.comparison_text
        assert trans.normalized_key != atlantic.normalized_key

        manager = CitationManager()
        manager.add_citations([(trans.raw_text, False)], 1)
        manager.add_citations([(atlantic.raw_text, False)], 1)
        assert len(manager.citations) == 2

    def test_standalone_editor_marker_is_stripped(self) -> None:
        citation = Citation(raw_text="Jones, A. (ed.). 1985. Collected Essays.")

        assert "ed" not in citation.comparison_text.split()


class TestNormalizedKey:
    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (
                "Sen, S. 1981. Poverty and Famines. Oxford.",
                "Sen, S. 1999. Poverty and Famines. Oxford.",
            ),
            (
                "Sraffa, P. 1951-1973. The Works of David Ricardo. Cambridge.",
                "Sraffa, P. 1960-1973. The Works of David Ricardo. Cambridge.",
            ),
            (
                "Smith, J. (2020). *Collected Works*, vol. 1. Cambridge University "
                "Press.",
                "Smith, J. (2020). *Collected Works*, vol. 2. Cambridge University "
                "Press.",
            ),
            (
                "https://example.org/records/alpha",
                "https://example.org/records/beta",
            ),
        ],
        ids=["editions", "year-ranges", "volumes", "url-only"],
    )
    def test_distinct_works_have_distinct_keys(self, first: str, second: str) -> None:
        assert Citation(raw_text=first).normalized_key != (
            Citation(raw_text=second).normalized_key
        )

    @pytest.mark.parametrize(
        "marker",
        ["pp. 123-145", "S. 42", "pp. 1850", "S. 1066-1071"],
        ids=["pp-range", "s-page", "pp-four-digit-page", "s-range-below-1500"],
    )
    def test_page_markers_do_not_enter_the_key(self, marker: str) -> None:
        with_pages = Citation(raw_text=f"Meyer. Some Work. {marker}.")
        without = Citation(raw_text="Meyer. Some Work.")

        assert with_pages.comparison_text == without.comparison_text
        assert with_pages.normalized_key == without.normalized_key

    def test_year_range_after_initial_stays_in_comparison_text(self) -> None:
        citation = Citation(
            raw_text="Sraffa, P. 1951-1973. The Works of David Ricardo. Cambridge."
        )

        assert "1951" in citation.comparison_text

    def test_url_only_citation_has_empty_comparison_text(self) -> None:
        citation = Citation(raw_text="https://example.org/records/alpha")

        assert citation.comparison_text == ""

    def test_comparison_text_has_no_asterisk(self) -> None:
        citation = Citation(raw_text="Doe, J. (2000). *A Starred Title*. Press.")

        assert "*" not in citation.comparison_text


class TestFold:
    """Letters that NFKD does not decompose transliterate."""

    def test_fold_transliterates_o_slash(self) -> None:
        assert fold("Møller") == "moller"

    def test_token_set_keeps_whole_surname(self) -> None:
        assert token_set(fold("Møller")) == {"moller"}

    def test_ligatures(self) -> None:
        assert fold("Ægir Œuvres Łódź") == "aegir oeuvres lodz"
