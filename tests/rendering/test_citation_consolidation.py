"""Tests for citation collection and merging (citations/manager.py, consolidate.py).

Exact duplicates merge on insertion. ``consolidate()`` merges near-duplicates
within an (author, year) block, never across years or volumes, and resolves
partial (in-text) citations by containment. ``merge_into`` re-derives the
survivor's fields from the promoted text. Citations that resolve to the same
DOI merge after enrichment.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any
from unittest.mock import patch

import pytest

from autoexcerpter.rendering.citations import (
    Citation,
    CitationManager,
    LookupOutcome,
    LookupResult,
)
from autoexcerpter.rendering.citations.consolidate import merge_into
from autoexcerpter.rendering.citations.model import jaccard


def _full(*texts: str) -> list[tuple[str, bool]]:
    """Wrap citation texts as complete (non-partial) reference tuples."""
    return [(text, False) for text in texts]


def _consolidated(*texts: str) -> CitationManager:
    """Add each text as a full citation on its own page, then consolidate."""
    manager = CitationManager()
    for page, text in enumerate(texts, 1):
        manager.add_citations(_full(text), page)
    manager.consolidate()
    return manager


class TestAddCitations:
    def test_add_citations_basic(self) -> None:
        manager = CitationManager()

        manager.add_citations(_full("Citation 1", "Citation 2"), page_number=1)

        assert len(manager.citations) == 2

    def test_add_citations_rejects_bare_strings(self) -> None:
        """A bare string is not a (text, is_partial) pair."""
        manager = CitationManager()
        items: list[Any] = ["Smith, J. (2020). A Title."]

        with pytest.raises(TypeError, match=r"\(text, is_partial\) pairs"):
            manager.add_citations(items, page_number=1)

        assert manager.citations == {}

    def test_add_citations_deduplication(self, sample_citations: list[str]) -> None:
        manager = CitationManager()

        manager.add_citations(_full(*sample_citations), page_number=1)

        # The sample holds one exact duplicate.
        assert len(manager.citations) == 3

    def test_add_citations_tracks_pages(self) -> None:
        manager = CitationManager()

        manager.add_citations(_full("Same citation"), page_number=1)
        manager.add_citations(_full("Same citation"), page_number=5)
        manager.add_citations(_full("Same citation"), page_number=10)

        assert len(manager.citations) == 1
        citation = list(manager.citations.values())[0]
        assert citation.get_page_range_str() == "pp. 1, 5, 10"

    def test_add_citations_skips_empty(self) -> None:
        manager = CitationManager()

        items: list[Any] = [
            ("", False),
            ("  ", False),
            (None, False),
            ("Valid citation", False),
        ]
        manager.add_citations(items, page_number=1)

        assert len(manager.citations) == 1

    def test_unnumbered_page_citation_rendered(self) -> None:
        """A citation only on an unnumbered page renders its PDF position."""
        manager = CitationManager()
        manager.add_citations([("Doe, J. (2001). Untitled.", False)], ("pdf", 4, False))
        citation = next(iter(manager.citations.values()))
        assert citation.get_page_range_str() == "PDF p. 4"

    def test_citation_without_locator_is_kept(self) -> None:
        """A citation whose position is unknown is kept without a locator."""
        manager = CitationManager()
        manager.add_citations([("Doe, J. (2001). Untitled.", False)], None)
        citation = next(iter(manager.citations.values()))
        assert citation.get_page_range_str() == ""


class TestSortedCitations:
    def test_get_sorted_citations(self, sample_citations: list[str]) -> None:
        """Citations sort by first-author surname."""
        manager = CitationManager()
        manager.add_citations(_full(*sample_citations), page_number=1)

        authors = [c.author for c in manager.get_sorted_citations()]

        assert authors == ["brown", "johnson", "smith"]

    def test_get_citations_with_pages(self, sample_citations: list[str]) -> None:
        """Each sorted citation is paired with its page range."""
        manager = CitationManager()
        manager.add_citations(_full(*sample_citations), page_number=5)

        result = manager.get_citations_with_pages()

        assert result == [(c, "p. 5") for c in manager.get_sorted_citations()]


class TestMergePolicy:
    def test_accent_variants_merge(self) -> None:
        """Muller and Müller fold to one key before consolidation runs."""
        manager = CitationManager()
        manager.add_citations(
            _full("Muller, S. (1985). The All-Consuming Nature of Taste. Blackwell."),
            1,
        )
        manager.add_citations(
            _full("Müller, S. (1985). The All-Consuming Nature of Taste. Blackwell."),
            2,
        )

        assert len(manager.citations) == 1
        merged = next(iter(manager.citations.values()))
        assert merged.get_page_range_str() == "pp. 1-2"

    def test_fuzzy_variants_merge(self) -> None:
        """An abbreviated and a full first name of one author merge."""
        manager = CitationManager()
        manager.add_citations(
            _full("Mennell, S. (1985). All Manners of Food. Blackwell."), 1
        )
        manager.add_citations(
            _full("Mennell, Stephen (1985). All Manners of Food. Blackwell."), 2
        )
        # S. and Stephen give distinct keys, so both survive insertion.
        assert len(manager.citations) == 2

        manager.consolidate()

        assert len(manager.citations) == 1
        merged = next(iter(manager.citations.values()))
        assert merged.get_page_range_str() == "pp. 1-2"

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (
                "Smith, J. (1985). A Study of Things. Press.",
                "Smith, J. (1987). A Study of Things. Press.",
            ),
            (
                "Braudel, F. (1979). Civilization and Capitalism, Vol. 1. Harper.",
                "Braudel, F. (1979). Civilization and Capitalism, Vol. 2. Harper.",
            ),
            (
                "Smith, J. (2020). *Collected Works*, vol. 1. Cambridge University "
                "Press.",
                "Smith, J. (2020). *Collected Works*, vol. 2. Cambridge University "
                "Press.",
            ),
            (
                "Weber, M. (1920). *Gesammelte Aufsätze*, Bd. 1. Mohr.",
                "Weber, M. (1920). *Gesammelte Aufsätze*, Bd. 2. Mohr.",
            ),
            (
                "Sen, S. 1981. Poverty and Famines. Oxford.",
                "Sen, S. 1999. Poverty and Famines. Oxford.",
            ),
            (
                "https://example.org/records/alpha",
                "https://example.org/records/beta",
            ),
            (
                "Smith, J. (1990). Trans-Atlantic Trade Networks in the Early "
                "Modern Atlantic World. Oxford University Press.",
                "Smith, J. (1990). Atlantic Migration and Labor Systems, "
                "1500-1800. Cambridge University Press.",
            ),
        ],
        ids=[
            "years",
            "volumes",
            "near-identical-volumes",
            "german-volumes",
            "editions",
            "url-only",
            "distinct-titles",
        ],
    )
    def test_distinct_works_never_merge(self, first: str, second: str) -> None:
        manager = _consolidated(first, second)

        assert {c.raw_text for c in manager.citations.values()} == {first, second}

    def test_particle_variants_merge(self) -> None:
        """'van der Berg, J.' and 'Berg, J. van der' share a block and merge."""
        manager = _consolidated(
            "van der Berg, J. (2000). A Study Of Coastal Trade. Brill.",
            "Berg, J. van der (2000). A Study Of Coastal Trade. Brill.",
        )

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.get_page_range_str() == "pp. 1-2"

    def test_reprint_variants_merge(self) -> None:
        """Both reprint orderings fold to the earliest year and merge."""
        manager = _consolidated(
            "Marx, K. (1976 [1867]). *Das Kapital*. Dietz.",
            "Marx, K. (1867 [1976]). *Das Kapital*. Dietz.",
        )

        assert len(manager.citations) == 1

    def test_reordered_variant_merges_via_jaccard(self) -> None:
        """A low SequenceMatcher ratio with a high token-set Jaccard merges."""
        variant_a = (
            "Smith, John. (2020). Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota."
        )
        variant_b = (
            "Smith, John. (2020). Iota Theta Eta Zeta Epsilon Delta Gamma Beta Alpha."
        )
        manager = CitationManager()
        c_a = Citation(raw_text=variant_a)
        c_b = Citation(raw_text=variant_b)
        # Same block; the ratio gate fails and the Jaccard gate passes.
        assert (c_a.author, c_a.year) == (c_b.author, c_b.year)
        ratio = SequenceMatcher(None, c_a.comparison_text, c_b.comparison_text).ratio()
        assert ratio < manager._merge_ratio
        assert jaccard(c_a.comparison_text, c_b.comparison_text) >= (
            manager._merge_jaccard
        )

        manager.add_citations(_full(variant_a), 1)
        manager.add_citations(_full(variant_b), 2)
        assert len(manager.citations) == 2
        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.get_page_range_str() == "pp. 1-2"

    def test_survivor_is_longest_regardless_of_order(self) -> None:
        long = "Smith, John. (1990). The Complete Economic History of Trade. Press."
        short = "Smith, J. (1990). The Complete Economic History of Trade."

        forward = _consolidated(long, short)
        backward = _consolidated(short, long)

        assert [c.raw_text for c in forward.citations.values()] == [long]
        assert [c.raw_text for c in backward.citations.values()] == [long]


class TestMergeInto:
    def test_longer_variant_refreshes_comparison_text_not_key(self) -> None:
        """The promoted text re-derives the fields; the dict key stays."""
        survivor = Citation(raw_text="Smith, Cooking.")
        other = Citation(
            raw_text="Smith, John. The Art of Cooking. London: Test Press, 1850."
        )
        original_key = survivor.normalized_key

        merge_into(survivor, other)

        assert survivor.raw_text == other.raw_text
        assert survivor.comparison_text == other.comparison_text
        assert "art" in survivor.comparison_text
        assert survivor.year == 1850
        assert survivor.normalized_key == original_key


_FULL = (
    "Conrad, A. H., & Meyer, J. R. (1957). The Economics of Slavery in the "
    "Ante-Bellum South. Journal of Political Economy, 66(2), 95-130."
)
_FULL2 = (
    "Conrad, A. H., & Meyer, J. R. (1957). Studies in Econometric History. "
    "Cambridge University Press."
)
_PARTIAL = "Conrad, A. H., & Meyer, J. R. (1957)."


class TestPartialCitations:
    """A partial (in-text) citation merges into its one full match or is dropped."""

    def test_partial_merges_into_unique_full_pages_union(self) -> None:
        manager = CitationManager()
        manager.add_citations([(_PARTIAL, True)], 412)
        manager.add_citations([(_FULL, False)], 414)
        assert len(manager.citations) == 2

        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.raw_text == _FULL
        assert survivor.partial is False
        assert survivor.get_page_range_str() == "pp. 412, 414"

    def test_partial_with_no_full_match_dropped(self) -> None:
        manager = CitationManager()
        manager.add_citations([(_PARTIAL, True)], 5)

        manager.consolidate()

        assert len(manager.citations) == 0

    def test_partial_with_two_full_candidates_dropped(self) -> None:
        """An ambiguous partial (two same-author-year fulls) is dropped."""
        manager = CitationManager()
        manager.add_citations([(_FULL, False)], 1)
        manager.add_citations([(_FULL2, False)], 2)
        manager.add_citations([(_PARTIAL, True)], 3)

        manager.consolidate()

        assert len(manager.citations) == 2
        raws = {c.raw_text for c in manager.citations.values()}
        assert raws == {_FULL, _FULL2}
        assert all(c.partial is False for c in manager.citations.values())

    def test_full_citation_unaffected(self) -> None:
        manager = CitationManager()
        manager.add_citations([(_FULL, False)], 7)

        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.raw_text == _FULL
        assert survivor.partial is False
        assert survivor.get_page_range_str() == "p. 7"

    def test_full_wins_over_partial_same_key(self) -> None:
        """One key seen as partial, then as full, ends up full."""
        manager = CitationManager()
        same = "Same, S. (2000). A Complete Work. Some Press."
        manager.add_citations([(same, True)], 1)
        manager.add_citations([(same, False)], 2)

        assert len(manager.citations) == 1
        citation = next(iter(manager.citations.values()))
        assert citation.partial is False
        assert citation.get_page_range_str() == "pp. 1-2"

    def test_partial_with_reprint_year_is_merged_not_dropped(self) -> None:
        """A partial citing the reprint year merges into the canonicalized full."""
        manager = CitationManager()
        manager.add_citations(
            [("Smith, John. 1990 [1890]. *Great Work of Smith*. Somewhere.", False)], 3
        )
        manager.add_citations([("(Smith 1990)", True)], 17)
        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.year == 1890
        assert {("printed-arabic", 3, False), ("printed-arabic", 17, False)} <= (
            survivor.locators
        )

    def test_parenthesized_partial_resolves_into_full(self) -> None:
        """A partial like "(Smith, 1990, S. 12)" folds into its full citation."""
        full = "Smith, John. (1990). A Full Length Work On Something. Some Press."
        manager = CitationManager()
        manager.add_citations([("(Smith, 1990, S. 12)", True)], 5)
        manager.add_citations([(full, False)], 7)
        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.raw_text == full
        assert survivor.get_page_range_str() == "pp. 5, 7"

    def test_author_year_partial_merges(self) -> None:
        manager = CitationManager()
        full = "Smith, J. (1990). Atlantic Trade Networks. Oxford University Press."
        manager.add_citations([(full, False)], 1)
        manager.add_citations([("(Smith 1990)", True)], 2)
        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.get_page_range_str() == "pp. 1-2"

    def test_empty_token_partial_is_dropped(self) -> None:
        """A bare-URL partial has no tokens, so containment is no evidence."""
        full = "Https, X. On Something. Journal"
        manager = CitationManager()
        manager.add_citations([(full, False)], 1)
        manager.add_citations([("https://example.com/foo", True)], 2)
        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.raw_text == full
        assert survivor.get_page_range_str() == "p. 1"


class TestMergedPartialBecomesFull:
    """A partial survivor that absorbs a full variant is full."""

    _PARTIAL = "Smith, J. (1990). Atlantic Trade Networks. Oxford University Press."
    _FULL = "Smith, J. (1990). Atlantic Trade Networks. Oxford Univ. Press."

    def test_variants_are_near_duplicates(self) -> None:
        """The two variants fuzzy-merge, which the other tests rely on."""
        manager = _consolidated(self._PARTIAL, self._FULL)

        assert len(manager.citations) == 1

    def test_partial_survivor_absorbing_full_is_not_dropped(self) -> None:
        manager = CitationManager()
        manager.add_citations([(self._PARTIAL, True)], 1)
        manager.add_citations([(self._FULL, False)], 2)
        manager.consolidate()

        assert len(manager.citations) == 1
        merged = next(iter(manager.citations.values()))
        assert merged.partial is False
        assert merged.get_page_range_str() == "pp. 1-2"

    def test_two_partials_stay_partial(self) -> None:
        """Two merged partials without a full reference are dropped."""
        manager = CitationManager()
        manager.add_citations([(self._PARTIAL, True)], 1)
        manager.add_citations([(self._FULL, True)], 2)
        manager.consolidate()

        assert manager.citations == {}


class TestSharedIdentifierMerge:
    """Citations resolving to the same OpenAlex DOI collapse into one."""

    _SHORT = "Alpha, A. (2001). Short Variant. Journal A."
    _LONG = (
        "Beta, B. (2002). A Much Longer Differently Worded Variant Of The Same "
        "Underlying Work. Journal B."
    )

    def test_same_doi_collapses_pages_union_longest_canonical(self) -> None:
        manager = CitationManager()
        manager.add_citations(_full(self._SHORT), 1)
        manager.add_citations(_full(self._LONG), 2)
        assert len(manager.citations) == 2

        shared_metadata = {
            "doi": "10.5555/shared.work",
            "url": "https://doi.org/10.5555/shared.work",
            "title": "The Same Underlying Work",
            "publication_year": 2001,
            "authors": [],
            "venue": None,
        }

        def fake_lookup(_citation: Citation) -> LookupResult:
            return LookupResult(LookupOutcome.FOUND, dict(shared_metadata))

        with patch.object(manager, "lookup", side_effect=fake_lookup):
            manager.enrich_with_metadata(max_requests=None)

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.doi == "10.5555/shared.work"
        assert survivor.get_page_range_str() == "pp. 1-2"
        assert survivor.raw_text == self._LONG
