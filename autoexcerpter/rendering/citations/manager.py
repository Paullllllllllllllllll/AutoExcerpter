"""Citation collection, deduplication and OpenAlex enrichment for one document."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoexcerpter.rendering.citations.consolidate import (
    MERGE_JACCARD,
    MERGE_RATIO,
    consolidate,
    merge_by_shared_identifier,
    sorted_citations,
)
from autoexcerpter.rendering.citations.model import Citation
from autoexcerpter.rendering.citations.openalex import (
    API_POLITE_DELAY,
    LookupOutcome,
    LookupResult,
    OpenAlexClient,
    sleep_unless_stopped,
)
from autoexcerpter.rendering.locators import Locator

logger = logging.getLogger(__name__)

# Citations looked up per document unless the settings say otherwise.
DEFAULT_MAX_API_REQUESTS = 300
PROGRESS_LOG_INTERVAL = 5  # Log progress every N citations
CHECKPOINT_INTERVAL = 25  # Write the cache every N lookups


class CitationManager:
    """Collects the citations of one document, deduplicates and enriches them."""

    def __init__(
        self,
        polite_pool_email: str | None = None,
        api_key: str | None = None,
        *,
        openalex_enabled: bool = True,
        max_requests: int = DEFAULT_MAX_API_REQUESTS,
        state_dir: Path | None = None,
        openalex: OpenAlexClient | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        """Create a manager.

        Args:
            polite_pool_email: OpenAlex contact when *openalex* is None.
            api_key: OpenAlex API key when *openalex* is None.
            openalex_enabled: Whether ``enrich_if_enabled`` looks citations up.
            max_requests: Citations looked up per document.
            state_dir: Folder of the OpenAlex cache when *openalex* is None.
            openalex: The run's OpenAlex client, shared by the managers of one
                run so that its cache and budget state carry across items;
                None creates a client for this manager alone.
            stop: Ends :meth:`enrich_with_metadata` early once set.
        """
        self.citations: dict[str, Citation] = {}
        self.openalex_enabled = openalex_enabled
        self.max_requests = max_requests
        self.stop = stop
        self.openalex = (
            openalex
            if openalex is not None
            else OpenAlexClient(polite_pool_email, api_key, state_dir=state_dir)
        )
        # Memoize the deterministic raw_text -> normalized_key derivation so a
        # repeated mention of the same citation skips the regex/NFKD/MD5 work
        # in Citation.__post_init__.
        self._normalized_key_cache: dict[str, str] = {}
        self._merge_ratio = MERGE_RATIO
        self._merge_jaccard = MERGE_JACCARD

    def add_citations(
        self,
        citations: Sequence[tuple[str, bool]],
        page_number: Locator | int | None,
    ) -> None:
        """Add the citations of one page, merging exact duplicates.

        Args:
            citations: ``(text, is_partial)`` pairs from a page, where
                ``is_partial`` marks an in-text-only stub (author-year without
                full bibliographic data).
            page_number: Where these citations appear: a locator, a bare int
                (a printed arabic page number), or None when the position is
                unknown (the citations are still recorded).

        Raises:
            TypeError: An item is a bare string instead of a pair.
        """
        for item in citations:
            if isinstance(item, str):
                raise TypeError(
                    "add_citations expects (text, is_partial) pairs, "
                    f"got the string {item[:60]!r}"
                )
            citation_text, is_partial = item
            if not citation_text or not citation_text.strip():
                continue

            stripped = citation_text.strip()

            # An identical raw text was already derived and its citation is
            # still present: only merge this mention's page and partial state.
            cached_key = self._normalized_key_cache.get(stripped)
            if cached_key is not None and cached_key in self.citations:
                existing = self.citations[cached_key]
                existing.add_page(page_number)
                existing.partial = existing.partial and is_partial
                continue

            citation = Citation(raw_text=stripped, partial=is_partial)
            normalized_key = citation.normalized_key
            self._normalized_key_cache[stripped] = normalized_key

            if normalized_key in self.citations:
                # Full wins over partial when one key is seen with both flags.
                existing = self.citations[normalized_key]
                existing.add_page(page_number)
                existing.partial = existing.partial and is_partial
            else:
                citation.add_page(page_number)
                self.citations[normalized_key] = citation

    def consolidate(self) -> None:
        """Fuzzy-merge near-duplicates and resolve partial citations.

        See :func:`autoexcerpter.rendering.citations.consolidate.consolidate`.
        """
        self.citations = consolidate(
            self.citations,
            merge_ratio=self._merge_ratio,
            merge_jaccard=self._merge_jaccard,
        )

    def lookup(self, citation: Citation) -> LookupResult:
        """Look one citation up in OpenAlex.

        :meth:`enrich_with_metadata` calls this for every citation the cache
        does not answer. Override it to skip or record lookups; the default
        delegates to ``self.openalex.lookup``.
        """
        return self.openalex.lookup(citation.raw_text)

    def enrich_with_metadata(self, max_requests: int | None = None) -> None:
        """Attach OpenAlex metadata to the citations and checkpoint the cache.

        Cached results are always served. A citation without a cached result
        is looked up unless the run's budget is exhausted or *max_requests*
        lookups (None: unlimited) were made in this call. One lookup runs a
        DOI query and/or a search, each retried on transient failures, so a
        single citation can issue several HTTP requests. Citations that
        resolve to the same work are merged afterwards.

        The cache is checkpointed every ``CHECKPOINT_INTERVAL`` lookups and
        when the method ends, also by an exception, so recorded results
        survive an interruption. Once ``self.stop`` is set, no further
        citation is processed and the method returns.
        """
        logger.info("Enriching %d unique citations with metadata", len(self.citations))
        openalex = self.openalex
        cache = openalex.cache
        tally = _Tally()
        try:
            with openalex.session(self.stop):
                if not self._enrich_all(openalex, tally, max_requests):
                    return
        finally:
            cache.checkpoint()

        self.citations = merge_by_shared_identifier(self.citations)
        logger.info(
            "Enriched %d citations with metadata (%d from the cache, %d via API); "
            "%d known as not found; skipped the lookup of %d citation(s).",
            tally.cache_hits + tally.api_enriched,
            tally.cache_hits,
            tally.api_enriched,
            tally.cached_misses,
            tally.skipped_api,
        )

    def _enrich_all(
        self, openalex: OpenAlexClient, tally: _Tally, max_requests: int | None
    ) -> bool:
        """Enrich every citation in turn; return False when stopped early."""
        stop = self.stop
        for processed, citation in enumerate(self.citations.values(), 1):
            if stop is not None and stop.is_set():
                logger.warning(
                    "OpenAlex enrichment stopped after %d lookup(s).",
                    tally.lookups_made,
                )
                return False
            if processed % PROGRESS_LOG_INTERVAL == 0:
                logger.info(
                    "Processed %d/%d citations, enriched %d with metadata",
                    processed,
                    len(self.citations),
                    tally.cache_hits + tally.api_enriched,
                )
            self._enrich_one(citation, openalex, tally, max_requests)
        return True

    def _enrich_one(
        self,
        citation: Citation,
        openalex: OpenAlexClient,
        tally: _Tally,
        max_requests: int | None,
    ) -> None:
        cache = openalex.cache
        cached = cache.get(citation.normalized_key)
        if cached is not None:
            if cached.metadata is not None:
                _apply_metadata(citation, cached.metadata)
                tally.cache_hits += 1
            else:
                tally.cached_misses += 1
            return

        # A skipped lookup does not end the loop, so later citations still
        # get their cached results.
        if not _may_look_up(openalex, tally, max_requests):
            tally.skipped_api += 1
            return

        # Every lookup counts against the cap, found or not.
        result = self.lookup(citation)
        tally.lookups_made += 1
        cache.record(citation.normalized_key, result)
        if result.outcome is LookupOutcome.FOUND and result.metadata:
            _apply_metadata(citation, result.metadata)
            tally.api_enriched += 1
        if tally.lookups_made % CHECKPOINT_INTERVAL == 0:
            cache.checkpoint()
        sleep_unless_stopped(API_POLITE_DELAY, self.stop)

    def get_sorted_citations(self) -> list[Citation]:
        """Return the citations in a stable, accent-folded order."""
        return sorted_citations(self.citations)

    def get_citations_with_pages(self) -> list[tuple[Citation, str]]:
        """Return ``(citation, page range)`` pairs in sorted order."""
        return [
            (citation, citation.get_page_range_str())
            for citation in self.get_sorted_citations()
        ]


@dataclass
class _Tally:
    """Counters and log-once flags of one enrichment call."""

    lookups_made: int = 0
    cache_hits: int = 0
    cached_misses: int = 0
    api_enriched: int = 0
    skipped_api: int = 0
    cap_notified: bool = False
    budget_notified: bool = False


def _may_look_up(
    openalex: OpenAlexClient, tally: _Tally, max_requests: int | None
) -> bool:
    """Return whether a lookup may run; log the first refusal of each kind."""
    if openalex.budget_exhausted:
        if not tally.budget_notified:
            logger.warning(
                "OpenAlex request limit reached: serving the "
                "remaining citations from the cache only."
            )
            tally.budget_notified = True
        return False
    if max_requests is not None and tally.lookups_made >= max_requests:
        if not tally.cap_notified:
            logger.info(
                "Reached the citation lookup cap (%d citations "
                "looked up; each may cost several HTTP "
                "requests): serving the remaining citations "
                "from the cache only.",
                max_requests,
            )
            tally.cap_notified = True
        return False
    return True


def _apply_metadata(citation: Citation, metadata: dict[str, Any]) -> None:
    """Attach fetched metadata to a citation."""
    citation.metadata = metadata
    citation.doi = metadata.get("doi")
    citation.url = metadata.get("url")


def enrich_if_enabled(citation_manager: CitationManager) -> None:
    """Run OpenAlex enrichment if the manager has it enabled.

    The manager's ``max_requests`` caps the number of citations looked up per
    document, not the number of HTTP calls.
    """
    if citation_manager.openalex_enabled:
        citation_manager.enrich_with_metadata(
            max_requests=citation_manager.max_requests
        )
    else:
        logger.info("OpenAlex enrichment disabled - skipping metadata lookup")


__all__ = [
    "CHECKPOINT_INTERVAL",
    "DEFAULT_MAX_API_REQUESTS",
    "CitationManager",
    "enrich_if_enabled",
]
