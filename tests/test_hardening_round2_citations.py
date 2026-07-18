"""Hardening round 2: citation, summary, and DOCX rendering regressions.

Each test pins a specific defect fixed in this round (German/French volume
designators, parenthesized in-text partials, particle/non-ASCII surnames,
German page markers, reprint years, OpenAlex false-link guards, the
overbrace/underbrace regex, and Markdown emphasis in DOCX bullets).
"""

from __future__ import annotations

from typing import Any

from docx import Document

from rendering.citations import (
    Citation,
    CitationManager,
    _extract_year,
    _first_author_surname,
)
from rendering.docx import (
    add_formatted_text_to_paragraph,
    simplify_problematic_latex,
)


class TestGermanFrenchVolumeDesignators:
    """Bd./Band/Teil/tome/t. volumes must be distinguished (English vol. too)."""

    _BD1 = "Weber, M. (1920). *Gesammelte Aufsätze*, Bd. 1. Mohr."
    _BD2 = "Weber, M. (1920). *Gesammelte Aufsätze*, Bd. 2. Mohr."

    def test_german_bd_volumes_extracted(self) -> None:
        """ "Bd. 1" and "Bd. 2" extract distinct volume numbers."""
        assert Citation(raw_text=self._BD1).volume == 1
        assert Citation(raw_text=self._BD2).volume == 2

    def test_german_bd_volumes_stay_separate(self) -> None:
        """Distinct German volumes never fuzzy-merge into one citation."""
        manager = CitationManager()
        manager.add_citations([self._BD1], 1)
        manager.add_citations([self._BD2], 2)
        manager.consolidate()

        assert len(manager.citations) == 2
        volumes = sorted(c.volume for c in manager.citations.values() if c.volume)
        assert volumes == [1, 2]

    def test_french_tome_roman_numeral(self) -> None:
        """A French "t. II" / "tome iv" volume parses via Roman numerals."""
        assert Citation(raw_text="Dupont. *Oeuvres*, t. II. Paris.").volume == 2
        assert Citation(raw_text="Dupont. *Oeuvres*, tome iv. Paris.").volume == 4

    def test_english_vol_still_arabic(self) -> None:
        """The original English "vol. 3" path is preserved."""
        assert Citation(raw_text="Smith (2020). *W*, vol. 3. CUP.").volume == 3

    def test_author_initial_before_year_is_not_a_volume(self) -> None:
        """"Smith, T. 1990. Title." must not read the year as a tome number."""
        assert Citation(raw_text="Smith, T. 1990. The Wealth. CUP.").volume is None

    def test_ordinary_words_do_not_false_positive(self) -> None:
        """A bare "t" without a period before a number is not a volume."""
        assert Citation(raw_text="Smith. It took 5 years to write.").volume is None


class TestParenthesizedPartials:
    """In-text partials like "(Smith, 1990, S. 12)" must not lose their page."""

    _FULL = "Smith, John. (1990). A Full Length Work On Something. Some Press."

    def test_parenthesized_partial_author_extracted(self) -> None:
        """The leading "(" no longer empties the author head."""
        assert _first_author_surname("(Smith, 1990, S. 12)") == "smith"

    def test_parenthesized_partial_resolves_into_full(self) -> None:
        """A parenthesized partial folds into its full citation; pages union."""
        manager = CitationManager()
        manager.add_citations([("(Smith, 1990, S. 12)", True)], 5)
        manager.add_citations([(self._FULL, False)], 7)
        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.raw_text == self._FULL
        assert survivor.get_sorted_pages() == [5, 7]


class TestParticleAndNonAsciiSurnames:
    """Particle surnames must fold consistently; non-ASCII letters must survive."""

    def test_particle_surname_orderings_merge(self) -> None:
        """ "van der Berg" and "Berg, J. van der" resolve to the same surname."""
        assert _first_author_surname("van der Berg, J. (2000). X.") == "berg"
        assert _first_author_surname("Berg, J. van der (2000). X.") == "berg"

    def test_particle_variants_merge_through_consolidate(self) -> None:
        """The two orderings share a block and merge into one citation."""
        manager = CitationManager()
        manager.add_citations(
            ["van der Berg, J. (2000). A Study Of Coastal Trade. Brill."], 1
        )
        manager.add_citations(
            ["Berg, J. van der (2000). A Study Of Coastal Trade. Brill."], 2
        )
        manager.consolidate()

        assert len(manager.citations) == 1
        assert next(iter(manager.citations.values())).get_sorted_pages() == [1, 2]

    def test_latin_extended_surname_not_mangled(self) -> None:
        """ "Łukasz" keeps its leading Ł-derived letter instead of losing it."""
        assert _first_author_surname("Łukasz, K. (2010). X.") == "łukasz"

    def test_umlaut_surname_folds_to_ascii(self) -> None:
        """ "Müller" folds to "muller" (accent stripped, not dropped)."""
        assert _first_author_surname("Müller, H. (2010). X.") == "muller"

    def test_initials_still_skipped(self) -> None:
        """A leading single-letter initial is not taken as the surname."""
        assert _first_author_surname("A. Smith (2020). X.") == "smith"

    def test_search_terms_capture_accented_and_apostrophe_surnames(self) -> None:
        """Search-term extraction keeps "Müller" and "O'Brien" whole."""
        manager = CitationManager()
        muller = manager._extract_search_terms(
            "Müller, H. (2010). *A Long Enough Title*. Journal."
        )
        obrien = manager._extract_search_terms(
            "O'Brien, P. (2010). *A Long Enough Title*. Journal."
        )
        assert "Müller" in muller
        assert "O'Brien" in obrien


class TestGermanPageMarkersAndReprintYears:
    """German page markers must not poison the year; reprints fold to earliest."""

    def test_german_page_range_not_read_as_year(self) -> None:
        """ "S. 1066-1071" no longer yields a bogus year 1066."""
        assert _extract_year("Meyer. Some Work. S. 1066-1071.") is None

    def test_real_year_survives_page_marker(self) -> None:
        """A genuine year is still found next to a German page range."""
        assert _extract_year("Meyer (1998). Some Work. S. 1066-1071.") == 1998

    def test_reprint_year_folds_to_earliest_either_order(self) -> None:
        """ "(1867 [1976])" and "(1976 [1867])" both resolve to 1867."""
        assert _extract_year("Marx (1976 [1867]). Capital.") == 1867
        assert _extract_year("Marx (1867 [1976]). Capital.") == 1867

    def test_reprint_variants_merge(self) -> None:
        """The two reprint orderings share a year block and merge."""
        manager = CitationManager()
        manager.add_citations(["Marx, K. (1976 [1867]). *Das Kapital*. Dietz."], 1)
        manager.add_citations(["Marx, K. (1867 [1976]). *Das Kapital*. Dietz."], 2)
        manager.consolidate()

        assert len(manager.citations) == 1


class TestVerifyCitationMatchGuards:
    """OpenAlex link verification must reject weak/contradictory candidates."""

    def _manager(self) -> CitationManager:
        return CitationManager()

    def test_single_token_title_rejected(self) -> None:
        """A one-substantive-token candidate title does not link on a word hit."""
        work: dict[str, Any] = {
            "title": "Nations",
            "publication_year": 1983,
            "authorships": [{"author": {"display_name": "Benedict Anderson"}}],
        }
        assert (
            self._manager()._verify_citation_match(
                "Anderson, B. (1983). Nations and nationalism.", work
            )
            is False
        )

    def test_gross_year_mismatch_rejected(self) -> None:
        """A known year off by more than two disqualifies despite an author hit."""
        work: dict[str, Any] = {
            "title": "Introduction to Testing",
            "publication_year": 2001,
            "authorships": [{"author": {"display_name": "John Smith"}}],
        }
        assert (
            self._manager()._verify_citation_match(
                "Smith, J. (1950). Introduction to Testing.", work
            )
            is False
        )

    def test_genuine_match_still_accepted(self) -> None:
        """A multi-token title with a corroborating year still links."""
        work: dict[str, Any] = {
            "title": "Introduction to Testing",
            "publication_year": 2020,
        }
        assert (
            self._manager()._verify_citation_match(
                "Smith, J. (2020). Introduction to Testing.", work
            )
            is True
        )


class TestDocxRenderingHardening:
    """Overbrace regex and Markdown emphasis in DOCX bullets."""

    def test_overbrace_preserves_trailing_content(self) -> None:
        """Simplifying \\overbrace must not devour following formula content."""
        out, _ = simplify_problematic_latex(r"\overbrace{a+b} + \frac{c}{d}")
        assert "\\frac{c}{d}" in out
        assert "a+b" in out
        assert "{d}" not in out.replace("\\frac{c}{d}", "")

    def test_underbrace_preserves_trailing_content(self) -> None:
        """The mirrored \\underbrace rule behaves the same way."""
        out, _ = simplify_problematic_latex(r"\underbrace{a+b}_{k} + \frac{c}{d}")
        assert "\\frac{c}{d}" in out
        assert "a+b" in out

    def test_bullet_bold_produces_bold_run_without_asterisks(self) -> None:
        """A DOCX bullet with **bold** yields a bold run and no literal "*"."""
        paragraph = Document().add_paragraph()
        add_formatted_text_to_paragraph(paragraph, "This is **bold** text")

        assert "*" not in paragraph.text
        bold_runs = [r for r in paragraph.runs if r.bold and r.text == "bold"]
        assert len(bold_runs) == 1
