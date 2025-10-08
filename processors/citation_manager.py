"""Citation management utilities for deduplication, tracking, and metadata enrichment."""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import quote

import requests

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Constants for API configuration
OPENALEX_API_BASE = "https://api.openalex.org"
OPENALEX_POLITE_POOL_EMAIL = "your-email@example.com"  # Users should update this
API_REQUEST_TIMEOUT = 10
API_RETRY_DELAY = 1.0
MAX_API_RETRIES = 3
API_POLITE_DELAY = 0.1  # Delay between API calls to be polite

# Constants for citation matching
MIN_AUTHOR_LENGTH = 3
MIN_TITLE_LENGTH = 10
MIN_YEAR_LENGTH = 4
MATCH_RATIO_THRESHOLD = 0.3  # Minimum word overlap ratio for citation matching
MAX_AUTHORS_TO_EXTRACT = 5  # Limit authors in metadata
SEARCH_QUERY_MAX_LENGTH = 100  # Maximum length for search queries
PROGRESS_LOG_INTERVAL = 5  # Log progress every N citations


@dataclass
class Citation:
    """Represents a single citation with metadata and page tracking."""
    
    raw_text: str
    pages: Set[int] = field(default_factory=set)
    normalized_key: str = ""
    metadata: Optional[Dict] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    
    def __post_init__(self):
        """Generate normalized key for deduplication."""
        if not self.normalized_key:
            self.normalized_key = self._generate_normalized_key()
    
    def _generate_normalized_key(self) -> str:
        """Create a normalized key for citation deduplication."""
        # Start with lowercase text
        text = self.raw_text.strip().lower()
        
        # Remove URLs (http/https)
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Remove DOIs
        text = re.sub(r'doi:\s*[\d.]+/[^\s]+', '', text)
        
        # Remove page numbers in various formats: p. 123, pp. 123-145, (p. 123), (pp. 123-145)
        text = re.sub(r'\(?\s*pp?\.\s*\d+[-–—]?\d*\s*\)?', '', text)
        
        # Remove years in parentheses: (2002), (n.d.), (1984/1996), (forthcoming)
        text = re.sub(r'\(\s*(?:\d{4}(?:[-/]\d{4})?|n\.?d\.?|forthcoming)\s*\)', '', text)
        
        # Remove standalone years: 1984, 2002
        text = re.sub(r'\b\d{4}\b', '', text)
        
        # Remove "n.d." standalone
        text = re.sub(r'\bn\.?\s*d\.?\b', '', text)
        
        # Remove editor references: (Ed.), (Eds.), (Trans.)
        text = re.sub(r'\(\s*(?:ed|eds|trans)\.?\s*\)', '', text, flags=re.IGNORECASE)
        
        # Remove volume references: Vol. 4, Volume 4
        text = re.sub(r'\b(?:vol|volume)\.?\s*\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove issue/page references in journals: 63(3), 101,
        text = re.sub(r'\d+\(\d+\)', '', text)
        
        # Remove common publisher locations and publishers
        text = re.sub(r'\b(?:london|cambridge|oxford|new york|berkeley|chicago|press|university|publisher)\b', '', text, flags=re.IGNORECASE)
        
        # Remove common citation elements: [publisher], [Place of publication not identified]
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove remaining punctuation except spaces
        text = re.sub(r'[,.:;()\[\]"\'–—]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create hash for efficient comparison
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def add_page(self, page: int) -> None:
        """Add a page number to this citation."""
        self.pages.add(page)
    
    def get_sorted_pages(self) -> List[int]:
        """Return sorted list of page numbers."""
        return sorted(self.pages)
    
    def get_page_range_str(self) -> str:
        """Return a formatted string of page numbers/ranges."""
        if not self.pages:
            return ""
        
        pages = self.get_sorted_pages()
        if len(pages) == 1:
            return f"p. {pages[0]}"
        
        # Group consecutive pages into ranges
        ranges = []
        start = pages[0]
        end = pages[0]
        
        for i in range(1, len(pages)):
            if pages[i] == end + 1:
                end = pages[i]
            else:
                if start == end:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{end}")
                start = pages[i]
                end = pages[i]
        
        # Add the last range
        if start == end:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{end}")
        
        return f"pp. {', '.join(ranges)}"


class CitationManager:
    """Manages citations across a document with deduplication and metadata enrichment."""
    
    def __init__(self, polite_pool_email: Optional[str] = None):
        """
        Initialize the citation manager.
        
        Args:
            polite_pool_email: Email for OpenAlex API polite pool access.
        """
        self.citations: Dict[str, Citation] = {}
        self.polite_pool_email = polite_pool_email or OPENALEX_POLITE_POOL_EMAIL
        self._api_cache: Dict[str, Optional[Dict]] = {}
    
    def add_citations(self, citations: List[str], page_number: int) -> None:
        """
        Add citations from a page, handling deduplication.
        
        Args:
            citations: List of citation strings from a page.
            page_number: The page number where these citations appear.
        """
        for citation_text in citations:
            if not citation_text or not citation_text.strip():
                continue
            
            # Create or retrieve citation
            citation = Citation(raw_text=citation_text.strip())
            normalized_key = citation.normalized_key
            
            if normalized_key in self.citations:
                # Citation already exists, just add the page
                self.citations[normalized_key].add_page(page_number)
            else:
                # New citation
                citation.add_page(page_number)
                self.citations[normalized_key] = citation
    
    def enrich_with_metadata(self, max_requests: Optional[int] = None) -> None:
        """
        Enrich citations with metadata from OpenAlex API.
        
        Args:
            max_requests: Maximum number of API requests to make (None for unlimited).
        """
        logger.info("Enriching %d unique citations with metadata", len(self.citations))
        
        requests_made = 0
        processed = 0
        for citation in self.citations.values():
            processed += 1
            
            # Log progress periodically
            if processed % PROGRESS_LOG_INTERVAL == 0:
                logger.info("Processed %d/%d citations, enriched %d with metadata", 
                          processed, len(self.citations), requests_made)
            
            if max_requests and requests_made >= max_requests:
                logger.info("Reached maximum API requests limit (%d)", max_requests)
                break
            
            # Try to extract identifiable information
            metadata = self._fetch_metadata_from_openalex(citation.raw_text)
            if metadata:
                citation.metadata = metadata
                citation.doi = metadata.get('doi')
                citation.url = metadata.get('url')
                requests_made += 1
                
                # Be polite to the API
                time.sleep(API_POLITE_DELAY)
        
        logger.info("Successfully enriched %d citations with metadata", requests_made)
    
    def _fetch_metadata_from_openalex(self, citation_text: str) -> Optional[Dict]:
        """
        Fetch metadata for a citation from OpenAlex API.
        
        Args:
            citation_text: The citation text to search for.
            
        Returns:
            Dictionary with metadata if found, None otherwise.
        """
        # Check cache first
        if citation_text in self._api_cache:
            return self._api_cache[citation_text]
        
        # Extract potential DOI from citation
        doi = self._extract_doi(citation_text)
        if doi:
            result = self._query_openalex_by_doi(doi)
            if result:
                self._api_cache[citation_text] = result
                return result
        
        # Try searching by citation text
        result = self._query_openalex_by_text(citation_text)
        self._api_cache[citation_text] = result
        return result
    
    def _extract_doi(self, citation_text: str) -> Optional[str]:
        """Extract DOI from citation text if present."""
        # Common DOI patterns
        doi_patterns = [
            r'doi:\s*(10\.\d{4,}/[^\s]+)',
            r'https?://doi\.org/(10\.\d{4,}/[^\s]+)',
            r'https?://dx\.doi\.org/(10\.\d{4,}/[^\s]+)',
            r'\b(10\.\d{4,}/[^\s,;]+)',
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, citation_text, re.IGNORECASE)
            if match:
                doi = match.group(1).rstrip('.,;')
                return doi
        
        return None
    
    def _make_openalex_request(
        self, 
        url: str, 
        params: Dict[str, Any],
        context_description: str = ""
    ) -> Optional[Dict]:
        """
        Make a request to OpenAlex API with retry logic and error handling.
        
        Args:
            url: The API endpoint URL.
            params: Query parameters.
            context_description: Description for logging (e.g., "DOI 10.1234/abc").
            
        Returns:
            Response data if successful, None otherwise.
        """
        for attempt in range(MAX_API_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=API_REQUEST_TIMEOUT)
                
                # Log request URL on first attempt if not successful
                if attempt == 0 and response.status_code != 200:
                    logger.debug("OpenAlex request URL: %s", response.url)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    # 404 is expected when resource not found
                    return None
                else:
                    # Log error details
                    try:
                        error_detail = response.json()
                        logger.warning(
                            "OpenAlex API returned status %d for %s: %s", 
                            response.status_code, context_description, error_detail
                        )
                    except Exception:
                        logger.warning(
                            "OpenAlex API returned status %d for %s", 
                            response.status_code, context_description
                        )
                    return None  # Don't retry on client errors
            except requests.RequestException as e:
                logger.warning(
                    "Error querying OpenAlex for %s (attempt %d/%d): %s", 
                    context_description, attempt + 1, MAX_API_RETRIES, str(e)
                )
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY)
            except Exception as e:
                logger.warning("Unexpected error querying OpenAlex for %s: %s", context_description, str(e))
                return None
        
        return None
    
    def _query_openalex_by_doi(self, doi: str) -> Optional[Dict]:
        """Query OpenAlex API using DOI."""
        url = f"{OPENALEX_API_BASE}/works/https://doi.org/{doi}"
        params = {"mailto": self.polite_pool_email}
        
        data = self._make_openalex_request(url, params, f"DOI {doi}")
        if data:
            return self._extract_metadata_from_response(data)
        return None
    
    def _query_openalex_by_text(self, citation_text: str) -> Optional[Dict]:
        """Query OpenAlex API using citation text search."""
        # Extract key terms for better search
        search_query = self._extract_search_terms(citation_text)
        if not search_query or len(search_query) < 10:
            return None
        
        url = f"{OPENALEX_API_BASE}/works"
        params = {
            "search": search_query,
            "mailto": self.polite_pool_email,
            "per-page": 1,
        }
        
        data = self._make_openalex_request(url, params, f"search query: {search_query[:50]}")
        if data:
            results = data.get('results', [])
            if results and len(results) > 0:
                # Verify the result matches reasonably well
                if self._verify_citation_match(citation_text, results[0]):
                    return self._extract_metadata_from_response(results[0])
        
        return None
    
    def _extract_search_terms(self, citation_text: str) -> str:
        """Extract key search terms from citation text."""
        # Remove common citation formatting
        text = re.sub(r'\([^)]*\)', '', citation_text)  # Remove parentheses
        text = re.sub(r'\[[^\]]*\]', '', text)  # Remove brackets
        text = re.sub(r'\d{4}', '', text)  # Remove years
        text = re.sub(r'[,.:;]', ' ', text)  # Replace punctuation with spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        
        # Take first N chars for search
        return text[:SEARCH_QUERY_MAX_LENGTH] if len(text) > SEARCH_QUERY_MAX_LENGTH else text
    
    def _verify_citation_match(self, citation_text: str, work_data: Dict) -> bool:
        """Verify that OpenAlex result matches the citation."""
        raw_title = work_data.get('title') or work_data.get('display_name') or ''
        title = raw_title.lower()
        citation_lower = citation_text.lower()

        # Check if title appears in citation (at least 50% of title words)
        if title:
            title_words = set(re.findall(r'\w+', title))
            title_words = {w for w in title_words if len(w) > 3}
            
            if title_words:
                citation_words = set(re.findall(r'\w+', citation_lower))
                common_words = title_words & citation_words
                match_ratio = len(common_words) / len(title_words)
                
                return match_ratio >= MATCH_RATIO_THRESHOLD
        
        return False
    
    def _extract_metadata_from_response(self, work_data: Dict) -> Dict:
        """Extract relevant metadata from OpenAlex API response."""
        metadata = {
            'title': work_data.get('title'),
            'doi': work_data.get('doi', '').replace('https://doi.org/', '') if work_data.get('doi') else None,
            'publication_year': work_data.get('publication_year'),
            'url': work_data.get('doi') or work_data.get('id'),
            'authors': [],
            'venue': None,
        }
        
        # Extract authors
        authorships = work_data.get('authorships', [])
        for authorship in authorships[:MAX_AUTHORS_TO_EXTRACT]:
            author = authorship.get('author', {})
            if author.get('display_name'):
                metadata['authors'].append(author['display_name'])
        
        # Extract venue
        primary_location = work_data.get('primary_location', {})
        if primary_location:
            source = primary_location.get('source', {})
            if source:
                metadata['venue'] = source.get('display_name')
        
        return metadata
    
    def get_sorted_citations(self) -> List[Citation]:
        """
        Return citations sorted alphabetically by raw text.
        
        Returns:
            List of Citation objects sorted by citation text.
        """
        return sorted(self.citations.values(), key=lambda c: c.raw_text.lower())
    
    def get_citations_with_pages(self) -> List[Tuple[Citation, str]]:
        """
        Return citations with formatted page information.
        
        Returns:
            List of tuples (Citation, page_range_string).
        """
        citations = self.get_sorted_citations()
        return [(citation, citation.get_page_range_str()) for citation in citations]
