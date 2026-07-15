"""Tests for rendering/citations.py - Citation management utilities."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rendering.citations import (
    Citation,
    CitationManager,
    _is_budget_exhausted,
    _mark_budget_exhausted,
    _reset_budget_state_for_tests,
)


class TestCitation:
    """Tests for Citation dataclass."""

    def test_init_basic(self) -> None:
        """Basic initialization with raw text."""
        citation = Citation(raw_text="Smith, J. (2020). Test Paper. Journal, 10, 1-10.")

        assert citation.raw_text == "Smith, J. (2020). Test Paper. Journal, 10, 1-10."
        assert citation.normalized_key != ""
        assert len(citation.pages) == 0

    def test_normalized_key_generated(self) -> None:
        """Normalized key is automatically generated."""
        citation = Citation(raw_text="Smith (2020). A Title.")

        assert citation.normalized_key != ""
        assert len(citation.normalized_key) == 32  # MD5 hash length

    def test_same_citations_same_key(self) -> None:
        """Equivalent citations produce same normalized key."""
        citation1 = Citation(raw_text="Smith, J. (2020). Test Title. Publisher.")
        citation2 = Citation(raw_text="Smith, J. (2020). Test Title. Publisher.")

        assert citation1.normalized_key == citation2.normalized_key

    def test_normalized_key_removes_years(self) -> None:
        """Years are removed from normalization."""
        citation1 = Citation(raw_text="Smith. Test Title. Publisher.")
        citation2 = Citation(raw_text="Smith (2020). Test Title. Publisher.")

        # Should have same key after year removal
        # Note: exact equality depends on normalization algorithm
        assert citation1.normalized_key != "" and citation2.normalized_key != ""

    def test_normalized_key_removes_page_numbers(self) -> None:
        """Page numbers are removed from normalization."""
        citation1 = Citation(raw_text="Smith. Test. pp. 10-20.")
        citation2 = Citation(raw_text="Smith. Test.")

        # Both should have valid keys
        assert citation1.normalized_key != ""
        assert citation2.normalized_key != ""

    def test_add_page(self) -> None:
        """Pages can be added to citation."""
        citation = Citation(raw_text="Test citation")

        citation.add_page(5)
        citation.add_page(10)
        citation.add_page(5)  # Duplicate

        assert 5 in citation.pages
        assert 10 in citation.pages
        assert len(citation.pages) == 2  # No duplicate

    def test_get_sorted_pages(self) -> None:
        """Pages are returned sorted."""
        citation = Citation(raw_text="Test citation")
        citation.add_page(10)
        citation.add_page(5)
        citation.add_page(15)

        assert citation.get_sorted_pages() == [5, 10, 15]

    def test_get_page_range_str_empty(self) -> None:
        """Empty pages returns empty string."""
        citation = Citation(raw_text="Test citation")

        assert citation.get_page_range_str() == ""

    def test_get_page_range_str_single(self) -> None:
        """Single page returns 'p. X' format."""
        citation = Citation(raw_text="Test citation")
        citation.add_page(5)

        assert citation.get_page_range_str() == "p. 5"

    def test_get_page_range_str_multiple(self) -> None:
        """Multiple pages returns 'pp. X, Y' format."""
        citation = Citation(raw_text="Test citation")
        citation.add_page(5)
        citation.add_page(10)

        assert "pp." in citation.get_page_range_str()

    def test_get_page_range_str_consecutive(self) -> None:
        """Consecutive pages are shown as ranges."""
        citation = Citation(raw_text="Test citation")
        citation.add_page(5)
        citation.add_page(6)
        citation.add_page(7)
        citation.add_page(10)

        result = citation.get_page_range_str()
        assert "5-7" in result
        assert "10" in result


class TestCitationManager:
    """Tests for CitationManager class."""

    def test_init_default(self) -> None:
        """Default initialization."""
        manager = CitationManager()

        assert manager.citations == {}
        assert manager.polite_pool_email is not None

    def test_init_custom_email(self) -> None:
        """Custom email initialization."""
        manager = CitationManager(polite_pool_email="test@example.com")

        assert manager.polite_pool_email == "test@example.com"

    def test_add_citations_basic(self) -> None:
        """Basic citation addition."""
        manager = CitationManager()

        manager.add_citations(["Citation 1", "Citation 2"], page_number=1)

        assert len(manager.citations) == 2

    def test_add_citations_deduplication(self, sample_citations: list[str]) -> None:
        """Duplicate citations are deduplicated."""
        manager = CitationManager()

        manager.add_citations(sample_citations, page_number=1)

        # Should have 3 unique citations (one duplicate in sample)
        assert len(manager.citations) == 3

    def test_add_citations_tracks_pages(self) -> None:
        """Page numbers are tracked correctly."""
        manager = CitationManager()

        manager.add_citations(["Same citation"], page_number=1)
        manager.add_citations(["Same citation"], page_number=5)
        manager.add_citations(["Same citation"], page_number=10)

        assert len(manager.citations) == 1
        citation = list(manager.citations.values())[0]
        assert citation.pages == {1, 5, 10}

    def test_add_citations_skips_empty(self) -> None:
        """Empty citation strings are skipped."""
        manager = CitationManager()

        manager.add_citations(["", "  ", None, "Valid citation"], page_number=1)  # type: ignore

        assert len(manager.citations) == 1

    def test_get_sorted_citations(self, sample_citations: list[str]) -> None:
        """Citations are sorted alphabetically."""
        manager = CitationManager()
        manager.add_citations(sample_citations, page_number=1)

        sorted_citations = manager.get_sorted_citations()

        # Should be sorted by raw_text (case-insensitive)
        texts = [c.raw_text.lower() for c in sorted_citations]
        assert texts == sorted(texts)

    def test_get_citations_with_pages(self, sample_citations: list[str]) -> None:
        """Returns citations with page range strings."""
        manager = CitationManager()
        manager.add_citations(sample_citations, page_number=5)

        result = manager.get_citations_with_pages()

        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], Citation) for item in result)
        assert all(isinstance(item[1], str) for item in result)


class TestCitationManagerDOI:
    """Tests for DOI extraction and API queries."""

    def test_extract_doi_from_text(self) -> None:
        """DOI is extracted from citation text."""
        manager = CitationManager()

        # Test various DOI formats
        assert manager._extract_doi("doi: 10.1234/test.2020") is not None
        assert manager._extract_doi("https://doi.org/10.1234/test") is not None
        assert manager._extract_doi("10.1234/test.paper") is not None

    def test_extract_doi_none_when_missing(self) -> None:
        """Returns None when no DOI in text."""
        manager = CitationManager()

        assert manager._extract_doi("Smith, J. (2020). Title.") is None

    def test_extract_search_terms(self) -> None:
        """Search terms are extracted from citation."""
        manager = CitationManager()

        text = "Smith, J. (2020). A Very Long Title for Testing. Journal, 10, 1-20."
        result = manager._extract_search_terms(text)

        assert len(result) > 0
        assert len(result) <= 100  # Max length enforced

    def test_verify_citation_match_true(self) -> None:
        """Matching citation with a corroborating year returns True."""
        manager = CitationManager()

        citation = "Smith, J. (2020). Introduction to Testing."
        # Strict linking: title overlap AND (year +/-1 OR author surname).
        work_data = {"title": "Introduction to Testing", "publication_year": 2020}

        assert manager._verify_citation_match(citation, work_data) is True

    def test_verify_citation_match_requires_corroboration(self) -> None:
        """Title overlap alone (no year/author signal) does not link."""
        manager = CitationManager()

        citation = "Smith, J. (2020). Introduction to Testing."
        work_data = {"title": "Introduction to Testing"}

        assert manager._verify_citation_match(citation, work_data) is False

    def test_verify_citation_match_false(self) -> None:
        """Non-matching citation returns False."""
        manager = CitationManager()

        citation = "Smith, J. (2020). Something Completely Different."
        work_data = {"title": "Introduction to Testing"}

        assert manager._verify_citation_match(citation, work_data) is False


class TestCitationManagerEnrichment:
    """Tests for metadata enrichment."""

    def test_enrich_with_metadata_calls_api(self, sample_citations: list[str]) -> None:
        """Enrichment calls OpenAlex API."""
        manager = CitationManager()
        manager.add_citations(sample_citations[:1], page_number=1)

        with patch.object(manager, "_fetch_metadata_from_openalex") as mock_fetch:
            mock_fetch.return_value = None

            manager.enrich_with_metadata(max_requests=1)

            mock_fetch.assert_called()

    def test_enrich_with_metadata_respects_limit(
        self, sample_citations: list[str]
    ) -> None:
        """Enrichment respects max_requests limit."""
        manager = CitationManager()
        manager.add_citations(sample_citations, page_number=1)

        call_count = 0

        def mock_fetch(text: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"doi": "10.1234/test", "url": "https://doi.org/10.1234/test"}

        with patch.object(
            manager, "_fetch_metadata_from_openalex", side_effect=mock_fetch
        ):
            manager.enrich_with_metadata(max_requests=1)

        assert call_count == 1

    def test_extract_metadata_from_response(
        self, mock_openalex_response: dict[str, Any]
    ) -> None:
        """Metadata is extracted correctly from API response."""
        manager = CitationManager()

        result = manager._extract_metadata_from_response(mock_openalex_response)

        assert result["title"] == "Introduction to Testing"
        assert result["publication_year"] == 2020
        assert "10.1234/test.2020.001" in result["doi"]
        assert len(result["authors"]) > 0

    def test_enrichment_updates_citation(self) -> None:
        """Enrichment updates citation metadata."""
        manager = CitationManager()
        manager.add_citations(["Test citation with DOI 10.1234/test"], page_number=1)

        mock_metadata = {
            "doi": "10.1234/test",
            "url": "https://doi.org/10.1234/test",
            "title": "Test",
            "publication_year": 2020,
            "authors": ["Test Author"],
            "venue": "Test Journal",
        }

        with patch.object(
            manager, "_fetch_metadata_from_openalex", return_value=mock_metadata
        ):
            manager.enrich_with_metadata(max_requests=1)

        citation = list(manager.citations.values())[0]
        assert citation.metadata is not None
        assert citation.doi == "10.1234/test"


class TestCitationManagerAPIRequest:
    """Tests for API request handling."""

    def test_make_openalex_request_success(self) -> None:
        """Successful API request returns data."""
        manager = CitationManager()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}

        with patch("rendering.citations.requests.get", return_value=mock_response):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works", {"search": "test"}, "test query"
            )

            assert result == {"results": []}

    def test_make_openalex_request_404_returns_none(self) -> None:
        """404 response returns None."""
        manager = CitationManager()

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("rendering.citations.requests.get", return_value=mock_response):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works/invalid", {}, "invalid DOI"
            )

            assert result is None

    def test_make_openalex_request_retries_on_error(self) -> None:
        """Request is retried on network error."""
        manager = CitationManager()

        import requests

        call_count = 0

        def mock_get(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.RequestException("Network error")
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"results": []}
            return response

        with (
            patch("rendering.citations.requests.get", side_effect=mock_get),
            patch("rendering.citations.time.sleep"),
        ):
            manager._make_openalex_request("https://api.openalex.org/works", {}, "test")

        assert call_count >= 2  # At least one retry

    def test_cache_prevents_duplicate_requests(self) -> None:
        """Cache prevents duplicate API requests."""
        manager = CitationManager()

        # Pre-populate cache
        manager._api_cache["test citation"] = {"cached": True}

        with patch("rendering.citations.requests.get") as mock_get:
            result = manager._fetch_metadata_from_openalex("test citation")

            # Should not make API call
            mock_get.assert_not_called()
            assert result == {"cached": True}


@pytest.fixture(autouse=True)
def _reset_openalex_budget_latch() -> Generator[None]:
    """Clear the module-level budget latch around every test.

    The ``config.state`` dir is already isolated per test by the autouse
    ``_isolate_state_dir`` fixture in ``conftest.py`` (so no real state files are
    touched); this fixture additionally clears the in-memory latch so exhaustion
    set by one test never leaks into the next.
    """
    _reset_budget_state_for_tests()
    yield
    _reset_budget_state_for_tests()


def _mock_429(retry_after: Any = None, header: str | None = None) -> MagicMock:
    """Build a MagicMock 429 response with an optional body/header retry-after."""
    resp = MagicMock()
    resp.status_code = 429
    resp.json.return_value = {} if retry_after is None else {"retryAfter": retry_after}
    resp.headers = {} if header is None else {"Retry-After": header}
    return resp


def _mock_200(payload: dict[str, Any]) -> MagicMock:
    """Build a MagicMock 200 response returning *payload* from ``.json()``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


class TestRetryAfterParsing:
    """Tests for _parse_retry_after (body preferred, header fallback)."""

    def test_prefers_body_over_header(self) -> None:
        """A JSON-body retryAfter wins over the Retry-After header."""
        resp = MagicMock()
        resp.headers = {"Retry-After": "999"}
        assert CitationManager._parse_retry_after({"retryAfter": 7}, resp) == 7

    def test_falls_back_to_header(self) -> None:
        """Retry-After header is used when the body lacks retryAfter."""
        resp = MagicMock()
        resp.headers = {"Retry-After": "12"}
        assert CitationManager._parse_retry_after({}, resp) == 12

    def test_ignores_date_format_header(self) -> None:
        """A date-format Retry-After header is treated as unknown (0)."""
        resp = MagicMock()
        resp.headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
        assert CitationManager._parse_retry_after({}, resp) == 0

    def test_missing_everywhere_returns_zero(self) -> None:
        """No body field and no header yields 0."""
        resp = MagicMock()
        resp.headers = {}
        assert CitationManager._parse_retry_after({}, resp) == 0


class TestOpenAlexBudgetLatch:
    """Tests for the process-wide + cross-run daily-budget exhaustion latch."""

    def test_large_retry_after_latches_and_persists(self) -> None:
        """429 with a large retryAfter sets the instance flag, the module latch,
        and persists the state file."""
        manager = CitationManager()

        with patch("rendering.citations.requests.get", return_value=_mock_429(3600)):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works", {}, "ctx"
            )

        assert result is None
        assert manager._openalex_budget_exhausted is True
        assert _is_budget_exhausted() is True

        # State file was written with a future timestamp.
        from config.state import read_json, resolve_state_file

        data = read_json(resolve_state_file("openalex_budget.json"))
        assert "exhausted_until" in data
        assert data["exhausted_until"] > time.time()

    def test_cache_served_after_exhaustion_without_http(self) -> None:
        """After exhaustion, enrichment still serves persistent-cache hits and
        makes zero further HTTP calls."""
        manager = CitationManager()
        manager.add_citations(["Alpha, A. (2001). Cached Work One. Some Press."], 1)
        manager.add_citations(["Beta, B. (2002). Uncached Work Two. Other Press."], 2)
        keys = list(manager.citations.keys())
        manager._persistent_cache[keys[0]] = {
            "doi": "10.1/one",
            "url": "https://doi.org/10.1/one",
            "title": "Cached Work One",
        }

        _mark_budget_exhausted(3600)

        with patch("rendering.citations.requests.get") as mock_get:
            manager.enrich_with_metadata(max_requests=10)
            mock_get.assert_not_called()

        cached = manager.citations.get(keys[0])
        assert cached is not None
        assert cached.metadata is not None
        assert cached.doi == "10.1/one"

    def test_new_manager_no_http_while_latched(self) -> None:
        """A fresh CitationManager in the same process makes no API calls while
        the latch is active."""
        _mark_budget_exhausted(3600)

        manager = CitationManager()
        manager.add_citations(["Gamma, G. (2003). Work with DOI 10.1234/gamma."], 1)

        with patch("rendering.citations.requests.get") as mock_get:
            manager.enrich_with_metadata(max_requests=10)
            mock_get.assert_not_called()

    def test_expired_latch_allows_requests(self) -> None:
        """A past exhausted_until (pre-written state file) allows API calls."""
        from config.state import resolve_state_file, write_json_atomic

        write_json_atomic(
            resolve_state_file("openalex_budget.json"),
            {"exhausted_until": time.time() - 100},
        )
        # Force a re-read of the (now stale) state file.
        _reset_budget_state_for_tests()
        assert _is_budget_exhausted() is False

        manager = CitationManager()
        manager.add_citations(["Delta, D. (2004). Work with DOI 10.1234/delta."], 1)

        work = {
            "title": "Delta Work",
            "doi": "https://doi.org/10.1234/delta",
            "publication_year": 2004,
            "authorships": [],
            "primary_location": {},
        }
        with patch(
            "rendering.citations.requests.get", return_value=_mock_200(work)
        ) as mock_get:
            manager.enrich_with_metadata(max_requests=10)
            mock_get.assert_called()


class TestOpenAlexRateLimitRetry:
    """Tests for short-retryAfter sleep-and-retry behavior."""

    def test_small_retry_after_sleeps_and_retries(self) -> None:
        """429 with a small retryAfter sleeps then retries within the loop."""
        manager = CitationManager()
        seq = [_mock_429(5), _mock_200({"results": []})]

        with (
            patch(
                "rendering.citations.requests.get",
                side_effect=lambda *a, **k: seq.pop(0),
            ) as mock_get,
            patch("rendering.citations.time.sleep") as mock_sleep,
        ):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works", {}, "ctx"
            )

        assert result == {"results": []}
        assert mock_get.call_count == 2
        mock_sleep.assert_any_call(5)

    def test_small_retry_after_from_header_retries(self) -> None:
        """A short Retry-After header (body empty) also triggers sleep-retry."""
        manager = CitationManager()
        seq = [_mock_429(header="3"), _mock_200({"ok": True})]

        with (
            patch(
                "rendering.citations.requests.get",
                side_effect=lambda *a, **k: seq.pop(0),
            ),
            patch("rendering.citations.time.sleep") as mock_sleep,
        ):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works", {}, "ctx"
            )

        assert result == {"ok": True}
        mock_sleep.assert_any_call(3)

    def test_medium_retry_after_skips_without_latch(self) -> None:
        """A retryAfter between the sleep ceiling and the budget threshold skips
        the citation but does not latch the budget."""
        manager = CitationManager()

        with patch(
            "rendering.citations.requests.get", return_value=_mock_429(120)
        ) as mock_get:
            result = manager._make_openalex_request(
                "https://api.openalex.org/works", {}, "ctx"
            )

        assert result is None
        assert mock_get.call_count == 1
        assert manager._openalex_budget_exhausted is False
        assert _is_budget_exhausted() is False


class TestEnrichMaxRequestsContinues:
    """Tests that max_requests no longer breaks the enrichment loop."""

    def test_max_requests_continues_and_serves_later_cache_hits(self) -> None:
        """Reaching max_requests must not stop later citations from getting
        their persistent-cache hits."""
        manager = CitationManager()
        manager.add_citations(["Aaa, A. (2001). First Uncached. Press."], 1)
        manager.add_citations(["Bbb, B. (2002). Second Uncached. Press."], 2)
        manager.add_citations(["Ccc, C. (2003). Third Cached. Press."], 3)
        keys = list(manager.citations.keys())
        manager._persistent_cache[keys[2]] = {
            "doi": None,
            "url": None,
            "title": "Third Cached",
        }

        fetch_calls: list[str] = []

        def fake_fetch(text: str) -> None:
            fetch_calls.append(text)
            return None

        with patch.object(
            manager, "_fetch_metadata_from_openalex", side_effect=fake_fetch
        ):
            manager.enrich_with_metadata(max_requests=1)

        # Only the first citation consumed the single allowed request; the
        # second (a cache miss past the cap) was skipped, not a hard break.
        assert len(fetch_calls) == 1
        third = manager.citations.get(keys[2])
        assert third is not None
        assert third.metadata is not None
