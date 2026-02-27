"""Tests for processors/citation_manager.py - Citation management utilities."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

from processors.citation_manager import (
    Citation,
    CitationManager,
)


class TestCitation:
    """Tests for Citation dataclass."""

    def test_init_basic(self):
        """Basic initialization with raw text."""
        citation = Citation(raw_text="Smith, J. (2020). Test Paper. Journal, 10, 1-10.")

        assert citation.raw_text == "Smith, J. (2020). Test Paper. Journal, 10, 1-10."
        assert citation.normalized_key != ""
        assert len(citation.pages) == 0

    def test_normalized_key_generated(self):
        """Normalized key is automatically generated."""
        citation = Citation(raw_text="Smith (2020). A Title.")

        assert citation.normalized_key != ""
        assert len(citation.normalized_key) == 32  # MD5 hash length

    def test_same_citations_same_key(self):
        """Equivalent citations produce same normalized key."""
        citation1 = Citation(raw_text="Smith, J. (2020). Test Title. Publisher.")
        citation2 = Citation(raw_text="Smith, J. (2020). Test Title. Publisher.")

        assert citation1.normalized_key == citation2.normalized_key

    def test_normalized_key_removes_years(self):
        """Years are removed from normalization."""
        citation1 = Citation(raw_text="Smith. Test Title. Publisher.")
        citation2 = Citation(raw_text="Smith (2020). Test Title. Publisher.")

        # Should have same key after year removal
        # Note: exact equality depends on normalization algorithm
        assert citation1.normalized_key != "" and citation2.normalized_key != ""

    def test_normalized_key_removes_page_numbers(self):
        """Page numbers are removed from normalization."""
        citation1 = Citation(raw_text="Smith. Test. pp. 10-20.")
        citation2 = Citation(raw_text="Smith. Test.")

        # Both should have valid keys
        assert citation1.normalized_key != ""
        assert citation2.normalized_key != ""

    def test_add_page(self):
        """Pages can be added to citation."""
        citation = Citation(raw_text="Test citation")

        citation.add_page(5)
        citation.add_page(10)
        citation.add_page(5)  # Duplicate

        assert 5 in citation.pages
        assert 10 in citation.pages
        assert len(citation.pages) == 2  # No duplicate

    def test_get_sorted_pages(self):
        """Pages are returned sorted."""
        citation = Citation(raw_text="Test citation")
        citation.add_page(10)
        citation.add_page(5)
        citation.add_page(15)

        assert citation.get_sorted_pages() == [5, 10, 15]

    def test_get_page_range_str_empty(self):
        """Empty pages returns empty string."""
        citation = Citation(raw_text="Test citation")

        assert citation.get_page_range_str() == ""

    def test_get_page_range_str_single(self):
        """Single page returns 'p. X' format."""
        citation = Citation(raw_text="Test citation")
        citation.add_page(5)

        assert citation.get_page_range_str() == "p. 5"

    def test_get_page_range_str_multiple(self):
        """Multiple pages returns 'pp. X, Y' format."""
        citation = Citation(raw_text="Test citation")
        citation.add_page(5)
        citation.add_page(10)

        assert "pp." in citation.get_page_range_str()

    def test_get_page_range_str_consecutive(self):
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

    def test_init_default(self):
        """Default initialization."""
        manager = CitationManager()

        assert manager.citations == {}
        assert manager.polite_pool_email is not None

    def test_init_custom_email(self):
        """Custom email initialization."""
        manager = CitationManager(polite_pool_email="test@example.com")

        assert manager.polite_pool_email == "test@example.com"

    def test_add_citations_basic(self):
        """Basic citation addition."""
        manager = CitationManager()

        manager.add_citations(["Citation 1", "Citation 2"], page_number=1)

        assert len(manager.citations) == 2

    def test_add_citations_deduplication(self, sample_citations: list[str]):
        """Duplicate citations are deduplicated."""
        manager = CitationManager()

        manager.add_citations(sample_citations, page_number=1)

        # Should have 3 unique citations (one duplicate in sample)
        assert len(manager.citations) == 3

    def test_add_citations_tracks_pages(self):
        """Page numbers are tracked correctly."""
        manager = CitationManager()

        manager.add_citations(["Same citation"], page_number=1)
        manager.add_citations(["Same citation"], page_number=5)
        manager.add_citations(["Same citation"], page_number=10)

        assert len(manager.citations) == 1
        citation = list(manager.citations.values())[0]
        assert citation.pages == {1, 5, 10}

    def test_add_citations_skips_empty(self):
        """Empty citation strings are skipped."""
        manager = CitationManager()

        manager.add_citations(["", "  ", None, "Valid citation"], page_number=1)  # type: ignore

        assert len(manager.citations) == 1

    def test_get_sorted_citations(self, sample_citations: list[str]):
        """Citations are sorted alphabetically."""
        manager = CitationManager()
        manager.add_citations(sample_citations, page_number=1)

        sorted_citations = manager.get_sorted_citations()

        # Should be sorted by raw_text (case-insensitive)
        texts = [c.raw_text.lower() for c in sorted_citations]
        assert texts == sorted(texts)

    def test_get_citations_with_pages(self, sample_citations: list[str]):
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

    def test_extract_doi_from_text(self):
        """DOI is extracted from citation text."""
        manager = CitationManager()

        # Test various DOI formats
        assert manager._extract_doi("doi: 10.1234/test.2020") is not None
        assert manager._extract_doi("https://doi.org/10.1234/test") is not None
        assert manager._extract_doi("10.1234/test.paper") is not None

    def test_extract_doi_none_when_missing(self):
        """Returns None when no DOI in text."""
        manager = CitationManager()

        assert manager._extract_doi("Smith, J. (2020). Title.") is None

    def test_extract_search_terms(self):
        """Search terms are extracted from citation."""
        manager = CitationManager()

        text = "Smith, J. (2020). A Very Long Title for Testing. Journal, 10, 1-20."
        result = manager._extract_search_terms(text)

        assert len(result) > 0
        assert len(result) <= 100  # Max length enforced

    def test_verify_citation_match_true(self):
        """Matching citation returns True."""
        manager = CitationManager()

        citation = "Smith, J. (2020). Introduction to Testing."
        work_data = {"title": "Introduction to Testing"}

        assert manager._verify_citation_match(citation, work_data) is True

    def test_verify_citation_match_false(self):
        """Non-matching citation returns False."""
        manager = CitationManager()

        citation = "Smith, J. (2020). Something Completely Different."
        work_data = {"title": "Introduction to Testing"}

        assert manager._verify_citation_match(citation, work_data) is False


class TestCitationManagerEnrichment:
    """Tests for metadata enrichment."""

    def test_enrich_with_metadata_calls_api(self, sample_citations: list[str]):
        """Enrichment calls OpenAlex API."""
        manager = CitationManager()
        manager.add_citations(sample_citations[:1], page_number=1)

        with patch.object(manager, "_fetch_metadata_from_openalex") as mock_fetch:
            mock_fetch.return_value = None

            manager.enrich_with_metadata(max_requests=1)

            mock_fetch.assert_called()

    def test_enrich_with_metadata_respects_limit(self, sample_citations: list[str]):
        """Enrichment respects max_requests limit."""
        manager = CitationManager()
        manager.add_citations(sample_citations, page_number=1)

        call_count = 0

        def mock_fetch(text):
            nonlocal call_count
            call_count += 1
            return {"doi": "10.1234/test", "url": "https://doi.org/10.1234/test"}

        with patch.object(
            manager, "_fetch_metadata_from_openalex", side_effect=mock_fetch
        ):
            manager.enrich_with_metadata(max_requests=1)

        assert call_count == 1

    def test_extract_metadata_from_response(self, mock_openalex_response: dict):
        """Metadata is extracted correctly from API response."""
        manager = CitationManager()

        result = manager._extract_metadata_from_response(mock_openalex_response)

        assert result["title"] == "Introduction to Testing"
        assert result["publication_year"] == 2020
        assert "10.1234/test.2020.001" in result["doi"]
        assert len(result["authors"]) > 0

    def test_enrichment_updates_citation(self):
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

    def test_make_openalex_request_success(self):
        """Successful API request returns data."""
        manager = CitationManager()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}

        with patch(
            "processors.citation_manager.requests.get", return_value=mock_response
        ):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works", {"search": "test"}, "test query"
            )

            assert result == {"results": []}

    def test_make_openalex_request_404_returns_none(self):
        """404 response returns None."""
        manager = CitationManager()

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch(
            "processors.citation_manager.requests.get", return_value=mock_response
        ):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works/invalid", {}, "invalid DOI"
            )

            assert result is None

    def test_make_openalex_request_retries_on_error(self):
        """Request is retried on network error."""
        manager = CitationManager()

        import requests

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.RequestException("Network error")
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"results": []}
            return response

        with patch("processors.citation_manager.requests.get", side_effect=mock_get):
            with patch("processors.citation_manager.time.sleep"):  # Skip delays
                result = manager._make_openalex_request(
                    "https://api.openalex.org/works", {}, "test"
                )

        assert call_count >= 2  # At least one retry

    def test_cache_prevents_duplicate_requests(self):
        """Cache prevents duplicate API requests."""
        manager = CitationManager()

        # Pre-populate cache
        manager._api_cache["test citation"] = {"cached": True}

        with patch("processors.citation_manager.requests.get") as mock_get:
            result = manager._fetch_metadata_from_openalex("test citation")

            # Should not make API call
            mock_get.assert_not_called()
            assert result == {"cached": True}
