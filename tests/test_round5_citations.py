"""Round 5: citation-layer regression tests for rendering/citations.py.

One focused regression per verified fix:

1. ``_merge_into`` unions the partial flag, so a partial survivor that absorbs
   a full near-duplicate is no longer dropped by ``_resolve_partials``.
2. The p./s. year guard also covers year *ranges* ("Sraffa, P. 1951-1973"),
   while genuine German page ranges ("S. 1066-1071") still strip.
3. A citation stripped to nothing (a bare URL) keys on its folded raw text and
   never fuzzy-merges with another empty-comparison citation.
4. The volume Roman-numeral branch is anchored to real Roman grammar and a
   plausibility bound, so words like "im" and "dix" are not volumes.
5. A 429 with any retryAfter above the sleep ceiling latches the daily budget
   instead of firing one doomed request per remaining citation.
6. A blank polite-pool email omits the ``mailto`` parameter entirely.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rendering.citations import (
    Citation,
    CitationManager,
    _extract_volume,
    _extract_year,
    _is_budget_exhausted,
    _reset_budget_state_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_openalex_budget_latch() -> Generator[None]:
    """Clear the module-level budget latch around every test in this module."""
    _reset_budget_state_for_tests()
    yield
    _reset_budget_state_for_tests()


def _mock_429(retry_after: Any = None) -> MagicMock:
    """Build a MagicMock 429 response with an optional body retryAfter."""
    resp = MagicMock()
    resp.status_code = 429
    resp.json.return_value = {} if retry_after is None else {"retryAfter": retry_after}
    resp.headers = {}
    return resp


def _mock_200(payload: dict[str, Any]) -> MagicMock:
    """Build a MagicMock 200 response returning *payload* from ``.json()``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


# ---------------------------------------------------------------------------
# Fix 1: the fuzzy merge unions the partial flag
# ---------------------------------------------------------------------------
class TestMergedPartialBecomesFull:
    """A partial survivor absorbing a full variant must stop being partial."""

    _PARTIAL = "Smith, J. (1990). Atlantic Trade Networks. Oxford University Press."
    _FULL = "Smith, J. (1990). Atlantic Trade Networks. Oxford Univ. Press."

    def test_variants_are_near_duplicates(self) -> None:
        """The two variants really do fuzzy-merge (guards the fixture)."""
        manager = CitationManager()
        manager.add_citations([self._PARTIAL], 1)
        manager.add_citations([self._FULL], 2)
        assert len(manager.citations) == 2
        manager.consolidate()
        assert len(manager.citations) == 1

    def test_partial_survivor_absorbing_full_is_not_dropped(self) -> None:
        """The merged citation survives consolidation instead of vanishing."""
        manager = CitationManager()
        manager.add_citations([(self._PARTIAL, True)], 1)
        manager.add_citations([(self._FULL, False)], 2)
        manager.consolidate()

        assert len(manager.citations) == 1
        merged = next(iter(manager.citations.values()))
        assert merged.partial is False
        assert merged.pages == {1, 2}

    def test_two_partials_stay_partial(self) -> None:
        """Merging two partials leaves the survivor partial (and thus droppable)."""
        manager = CitationManager()
        manager.add_citations([(self._PARTIAL, True)], 1)
        manager.add_citations([(self._FULL, True)], 2)
        manager.consolidate()

        # No full reference exists for this block, so the merged partial is
        # dropped exactly as before.
        assert manager.citations == {}


# ---------------------------------------------------------------------------
# Fix 2: the year guard covers year ranges after a single-letter initial
# ---------------------------------------------------------------------------
class TestInitialBeforeYearRange:
    """ "Sraffa, P. 1951-1973" must keep its year range."""

    _SRAFFA = "Sraffa, P. 1951-1973. The Works of David Ricardo. Cambridge."

    def test_year_range_after_initial_is_not_a_page_range(self) -> None:
        assert _extract_year(self._SRAFFA) == 1951

    def test_year_range_survives_in_comparison_text(self) -> None:
        """The range is not stripped from the normalized-key material either."""
        assert "1951" in Citation(raw_text=self._SRAFFA).comparison_text

    def test_range_editions_keep_distinct_keys(self) -> None:
        """Two different Sraffa spans do not collapse onto one key."""
        other = "Sraffa, P. 1960-1973. The Works of David Ricardo. Cambridge."
        assert (
            Citation(raw_text=self._SRAFFA).normalized_key
            != Citation(raw_text=other).normalized_key
        )

    def test_german_page_range_still_strips(self) -> None:
        """Endpoints below 1500 are page numbers, not years: still stripped."""
        assert _extract_year("Meyer. Some Work. S. 1066-1071.") is None

    def test_ordinary_page_range_still_strips(self) -> None:
        assert _extract_year("Meyer (1998). Some Work. S. 12-34.") == 1998

    def test_mixed_endpoint_range_still_strips(self) -> None:
        """Only a range with BOTH endpoints in 1500-2099 is spared."""
        assert _extract_year("Meyer. Some Work. S. 1400-1600.") is None

    def test_german_page_range_absent_from_comparison_text(self) -> None:
        with_pages = Citation(raw_text="Meyer. Some Work. S. 1066-1071.")
        without = Citation(raw_text="Meyer. Some Work.")
        assert with_pages.comparison_text == without.comparison_text

    def test_multi_letter_marker_range_unaffected(self) -> None:
        """pp./ss./fol. are never initials, so a year-shaped range still strips."""
        assert _extract_year("Meyer. Some Work. pp. 1951-1973.") is None


# ---------------------------------------------------------------------------
# Fix 3: citations stripped to nothing keep distinct identities
# ---------------------------------------------------------------------------
class TestEmptyComparisonTextCitations:
    """Two URL-only citations must not collapse into one entry."""

    _URL_A = "https://example.org/records/alpha"
    _URL_B = "https://example.org/records/beta"

    def test_url_only_citation_has_empty_comparison_text(self) -> None:
        assert Citation(raw_text=self._URL_A).comparison_text == ""

    def test_distinct_urls_have_distinct_keys(self) -> None:
        assert (
            Citation(raw_text=self._URL_A).normalized_key
            != Citation(raw_text=self._URL_B).normalized_key
        )

    def test_identical_urls_still_share_a_key(self) -> None:
        assert (
            Citation(raw_text=self._URL_A).normalized_key
            == Citation(raw_text=self._URL_A).normalized_key
        )

    def test_both_urls_survive_consolidation(self) -> None:
        manager = CitationManager()
        manager.add_citations([self._URL_A], 1)
        manager.add_citations([self._URL_B], 2)
        manager.consolidate()

        assert len(manager.citations) == 2
        assert {c.raw_text for c in manager.citations.values()} == {
            self._URL_A,
            self._URL_B,
        }


# ---------------------------------------------------------------------------
# Fix 4: the Roman-numeral volume branch is anchored to real Roman grammar
# ---------------------------------------------------------------------------
class TestRomanVolumeGrammar:
    """Ordinary words must not parse as Roman volume numbers."""

    def test_german_im_is_not_a_volume(self) -> None:
        """ "Bd. im Erscheinen" (forthcoming) is not volume 999."""
        assert _extract_volume("Meyer, H. Studien. Bd. im Erscheinen.") is None

    def test_french_dix_is_not_a_volume(self) -> None:
        """ "t. dix" is not volume 509 (well-formed Roman, implausible value)."""
        assert _extract_volume("Dupont. Oeuvres, t. dix. Paris.") is None

    def test_malformed_roman_sequence_rejected(self) -> None:
        """ "vol. vic" is letter salad, not a Roman numeral."""
        assert _extract_volume("Smith. Volvic Waters, vol. vic.") is None

    def test_lowercase_roman_volume_still_parses(self) -> None:
        assert Citation(raw_text="Smith (2020). *W*, vol. xiv. CUP.").volume == 14

    def test_uppercase_tome_still_parses(self) -> None:
        assert Citation(raw_text="Dupont. *Oeuvres*, t. II. Paris.").volume == 2

    def test_arabic_volume_still_parses(self) -> None:
        assert Citation(raw_text="Smith (2020). *W*, vol. 3. CUP.").volume == 3

    def test_false_positive_volume_no_longer_enters_key(self) -> None:
        """The bogus volume is gone, so "Bd. im" keys like a volume-less cite."""
        assert Citation(raw_text="Meyer, H. Studien. Bd. im Erscheinen.").volume is None


# ---------------------------------------------------------------------------
# Fix 5: no 429 dead band between the sleep ceiling and the old threshold
# ---------------------------------------------------------------------------
class TestNo429DeadBand:
    """A medium retryAfter must latch rather than skip-and-retry-forever."""

    def test_enrichment_stops_after_medium_retry_after(self) -> None:
        manager = CitationManager()
        manager.add_citations(["Aaa, A. (2001). First Work Here. Press."], 1)
        manager.add_citations(["Bbb, B. (2002). Second Work Here. Press."], 2)
        manager.add_citations(["Ccc, C. (2003). Third Work Here. Press."], 3)
        assert len(manager.citations) == 3

        # enrich_with_metadata pools its GETs in a requests.Session, so the
        # session method is the seam that must be stubbed (never the network).
        with patch("requests.Session.get", return_value=_mock_429(120)) as mock_get:
            manager.enrich_with_metadata(max_requests=10)

        # Exactly one doomed request, then the latch stops the rest.
        assert mock_get.call_count == 1
        assert manager._openalex_budget_exhausted is True
        assert _is_budget_exhausted() is True


# ---------------------------------------------------------------------------
# Fix 6: the polite-pool mailto is omitted when unconfigured
# ---------------------------------------------------------------------------
class TestPolitePoolMailto:
    """A blank configured email must not send a placeholder mailto."""

    _CITATION = 'Smith, J. (2020). "A Sufficiently Long Title." Journal.'

    @staticmethod
    def _capture_params(manager: CitationManager, call: str) -> dict[str, Any]:
        """Run a query and return the params dict handed to requests.get."""
        captured: dict[str, Any] = {}

        def fake_get(url: str, params: Any = None, timeout: Any = None) -> MagicMock:
            captured.update({"url": url, "params": params or {}})
            return _mock_200({"results": []})

        with patch("rendering.citations.requests.get", side_effect=fake_get):
            if call == "doi":
                manager._query_openalex_by_doi("10.1234/abc")
            else:
                manager._query_openalex_by_text(TestPolitePoolMailto._CITATION)

        params = captured.get("params")
        assert isinstance(params, dict)
        return params

    def test_default_manager_email_is_blank(self) -> None:
        assert CitationManager().polite_pool_email == ""

    def test_whitespace_email_treated_as_blank(self) -> None:
        assert CitationManager(polite_pool_email="   ").polite_pool_email == ""

    def test_blank_email_omits_mailto_on_text_search(self) -> None:
        params = self._capture_params(CitationManager(), "text")
        assert "mailto" not in params
        assert params.get("search")

    def test_blank_email_omits_mailto_on_doi_lookup(self) -> None:
        params = self._capture_params(CitationManager(), "doi")
        assert "mailto" not in params

    def test_configured_email_sends_mailto_on_text_search(self) -> None:
        manager = CitationManager(polite_pool_email="scholar@example.edu")
        params = self._capture_params(manager, "text")
        assert params["mailto"] == "scholar@example.edu"

    def test_configured_email_sends_mailto_on_doi_lookup(self) -> None:
        manager = CitationManager(polite_pool_email="scholar@example.edu")
        params = self._capture_params(manager, "doi")
        assert params["mailto"] == "scholar@example.edu"
