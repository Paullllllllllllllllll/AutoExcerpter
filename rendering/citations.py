"""Citation management utilities for deduplication, tracking, and metadata
enrichment."""

from __future__ import annotations

import hashlib
import re
import threading
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import requests

from config.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Text folding & structured-key helpers (shared by dedup and matching)
# ============================================================================

# Curly quotes/apostrophes and the several dash characters folded to ASCII so
# that "Müller's" / "Muller's" and en/em dashes do not defeat deduplication.
_CURLY_TRANSLATION = {
    ord("‘"): "'",
    ord("’"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord("–"): "-",
    ord("—"): "-",
    ord("−"): "-",
    ord("­"): "",  # soft hyphen
}

_YEAR_CORE = r"(1[0-9]{3}|20[0-9]{2})"
_YEAR_RE = re.compile(r"\b" + _YEAR_CORE + r"\b")
# Reprint form: a plain/parenthesized year adjoining a bracketed year, in either
# order ("1976 [1867]" or "[1867] 1976"). Both years are captured so the
# original (earliest) can win over the reprint.
_REPRINT_RE = re.compile(
    _YEAR_CORE + r"\s*\[\s*" + _YEAR_CORE + r"\s*\]"
    r"|\[\s*" + _YEAR_CORE + r"\s*\]\s*" + _YEAR_CORE
)
# Page-marker spans (English pp./p., German S./SS., folio fol.) stripped before
# year scanning so a page range such as "S. 1066-1071" is never read as a year.
# The bare-number alternative is guarded by a negative lookahead so an author
# initial adjoining a year ("Sen, S. 1981") is NOT swallowed as a page marker:
# a marker followed by a lone plausible year (1000-2099) is left intact, while
# page ranges and non-year page numbers still strip. The range alternative comes
# first so "S. 1066-1071" is consumed whole (never mistaken for a year). Trade-off:
# a genuine single-page German cite of a four-digit page ("S. 1815") is read as a
# year — a rare, accepted false positive versus the initial-vs-year collapse.
_PAGE_MARKER_RE = re.compile(
    r"\b(?:pp?|ss?|fol)\.\s*"
    r"(?:\d+\s*-\s*\d+|(?!(?:1[0-9]{3}|20[0-9]{2})\b)\d+)",
    re.IGNORECASE,
)
# Volume designators across English/German/French scholarship. The number may
# be Arabic or a Roman numeral. The period is optional for the spelled-out
# designators but required for the bare "t." so ordinary words never match;
# the bare "t." number is further capped at three digits and the "t" is matched
# case-sensitively as lowercase (via the scoped "(?-i:t)" flag) so an author
# initial ("Smith, T. 190. Title" or "Smith, T. 1990. Title") is never read as a
# tome/volume, while the lowercase French "t. II" abbreviation still parses.
_VOLUME_RE = re.compile(
    r"\b(?:vol|volume|bd|band|teil|tome)\.?\s*(\d+|[ivxlcdm]+)\b"
    r"|\b(?-i:t)\.\s*(\d{1,3}|[ivxlcdm]+)\b",
    re.IGNORECASE,
)


def _roman_to_int(text: str) -> int | None:
    """Convert a Roman numeral to an int (case-insensitive), or None if invalid."""
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    s = text.lower()
    if not s or any(ch not in values for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = values[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total or None


_TITLE_SPAN_RE = re.compile(r'\*([^*\n]{3,})\*|"([^"\n]{3,})"')
_STOPLIST_RE = re.compile(
    r"\b(?:london|cambridge|oxford|new york|berkeley|chicago"
    r"|press|university|publishers?)\b"
)


def _fold(text: str) -> str:
    """Fold text for robust comparison.

    NFKD-normalizes, strips combining marks (so ``Müller`` == ``Muller`` and
    ``Génin`` == ``Genin``), casefolds, unifies curly quotes and dashes to
    ASCII, and rewrites ``&`` as ``and``.
    """
    if not text:
        return ""
    text = text.translate(_CURLY_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = text.replace("&", " and ")
    return text


def _extract_year(text: str) -> int | None:
    """Return the first 4-digit publication year in *text*, or None.

    URLs and DOIs are stripped first: common DOI registrant prefixes such as
    ``10.1016/`` or ``10.1111/`` otherwise match the ``1[0-9]{3}`` year pattern
    and yield a bogus year that then mis-blocks deduplication.
    """
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"doi:\s*10\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b10\.\d{4,}/\S+", " ", text)
    # A reprint ("1976 [1867]") splits the same work across two blocks unless we
    # canonicalize on the earliest year; detect the adjoining pair and return the
    # minimum before the positional "first year" scan below.
    reprint = _REPRINT_RE.search(text)
    if reprint:
        years = [int(g) for g in reprint.groups() if g]
        if years:
            return min(years)
    # Drop page-marker spans so their numbers cannot be misread as a year.
    text = _PAGE_MARKER_RE.sub(" ", text)
    match = _YEAR_RE.search(text)
    return int(match.group(1)) if match else None


def _extract_volume(text: str) -> int | None:
    """Return the volume number in *text* (``Vol. 3``, ``Bd. 2``, ``t. II``), or
    None. Handles English/German/French designators and Roman numerals."""
    match = _VOLUME_RE.search(text)
    if not match:
        return None
    captured = match.group(1) or match.group(2)
    if captured is None:
        return None
    if captured.isdigit():
        return int(captured)
    return _roman_to_int(captured)


def _title_spans(text: str) -> list[str]:
    """Return italic (``*...*``) or quoted (``"..."``) title spans in *text*."""
    spans: list[str] = []
    for ital, quoted in _TITLE_SPAN_RE.findall(text):
        span = ital or quoted
        if span:
            spans.append(span)
    return spans


# Non-name tokens that must never be taken as a surname.
_SURNAME_STOPWORDS = {"the", "and", "of", "in", "on", "ed", "eds", "trans"}
# Lowercase nobiliary/name particles that precede the actual surname. When a
# name leads with one ("van der Berg"), the first non-particle token is the
# surname, so "van der Berg" and "Berg, J. van der" fold to the same block.
_NAME_PARTICLES = {
    "van",
    "von",
    "der",
    "de",
    "den",
    "du",
    "la",
    "le",
    "ter",
    "ten",
    "da",
    "di",
    "dos",
    "del",
}
# Unicode-aware word token: a letter (any script, incl. Latin Extended) followed
# by at least one more letter/apostrophe/hyphen. The two-character minimum skips
# single-letter initials ("J.").
_NAME_TOKEN_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’-])+", re.UNICODE)


def _first_author_surname(text: str) -> str:
    """Return the folded first-author surname (best-effort), or ``""``.

    Takes the first alphabetic token before the first year or opening
    parenthesis, skipping lowercase name particles so particle surnames fold
    consistently regardless of ordering.
    """
    # An in-text partial like "(Smith, 1990, S. 12)" leads with "(", which would
    # otherwise split to an empty head; drop leading brackets/whitespace first.
    stripped = text.lstrip("([ \t\r\n")
    head = re.split(r"\(|\d{4}", stripped, maxsplit=1)[0]
    tokens = [
        folded
        for tok in _NAME_TOKEN_RE.findall(head)
        if (folded := _fold(tok)) not in _SURNAME_STOPWORDS
    ]
    if not tokens:
        return ""
    for tok in tokens:
        if tok not in _NAME_PARTICLES:
            return tok
    return tokens[0]


def _token_set(text: str) -> set[str]:
    """Return the set of word tokens (length > 1) in folded *text*."""
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1}


def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two comparison strings."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ============================================================================
# Page Range Formatting
# ============================================================================


def _format_page_range(pages: list[int], has_unnumbered: bool = False) -> str:
    """Format a list of page numbers as a compact range string.

    Examples:
        [1, 2, 3, 5, 7, 8, 9] -> "pp. 1-3, 5, 7-9"
        [5] -> "p. 5"
        [] with has_unnumbered=True -> "unnumbered"
        [5] with has_unnumbered=True -> "pp. 5, unnumbered"
    """
    if not pages:
        return "unnumbered" if has_unnumbered else ""

    pages = sorted(set(pages))
    ranges = []
    start = pages[0]
    end = pages[0]

    for page in pages[1:]:
        if page == end + 1:
            end = page
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = page

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    if has_unnumbered:
        ranges.append("unnumbered")

    single = len(pages) == 1 and not has_unnumbered
    prefix = "p." if single else "pp."
    return f"{prefix} {', '.join(ranges)}"


# ============================================================================
# OpenAlex Enrichment Helper (shared by DOCX and Markdown writers)
# ============================================================================


def enrich_if_enabled(citation_manager: CitationManager) -> None:
    """Run OpenAlex metadata enrichment if enabled in config.

    Reads ``config.app.CITATION_ENABLE_OPENALEX`` and
    ``config.app.CITATION_MAX_API_REQUESTS``; logs the decision.
    """
    from config import app as config  # deferred to avoid import cycles

    if config.CITATION_ENABLE_OPENALEX:
        citation_manager.enrich_with_metadata(
            max_requests=config.CITATION_MAX_API_REQUESTS
        )
    else:
        logger.info("OpenAlex enrichment disabled - skipping metadata lookup")


# Constants for API configuration
OPENALEX_API_BASE = "https://api.openalex.org"
OPENALEX_POLITE_POOL_EMAIL = "your-email@example.com"  # Users should update this
API_REQUEST_TIMEOUT = 10
API_RETRY_DELAY = 1.0
MAX_API_RETRIES = 3
API_POLITE_DELAY = 0.1  # Delay between API calls to be polite
# 429 budget-exhaustion: skip remaining enrichment if retryAfter exceeds this (seconds)
BUDGET_EXHAUSTED_RETRY_AFTER_THRESHOLD = 300  # 5 minutes
# 429 short rate-limit: sleep-and-retry within the attempt loop when retryAfter
# is at or below this (seconds); larger values keep the skip behavior.
RATE_LIMIT_MAX_SLEEP = 30  # seconds

# Constants for citation matching
MIN_AUTHOR_LENGTH = 3
MIN_TITLE_LENGTH = 10
MIN_YEAR_LENGTH = 4
# Minimum fraction of candidate title words that must appear in the citation
# for an OpenAlex link to be considered (strict linking; see decision 9). The
# config value ``citation.match_title_overlap`` overrides this at runtime.
MATCH_RATIO_THRESHOLD = 0.5
MAX_AUTHORS_TO_EXTRACT = 5  # Limit authors in metadata
SEARCH_QUERY_MAX_LENGTH = 100  # Maximum length for search queries
SEARCH_RESULTS_PER_PAGE = 5  # Candidates fetched per search; first matching wins
PROGRESS_LOG_INTERVAL = 5  # Log progress every N citations


@dataclass
class Citation:
    """Represents a single citation with metadata and page tracking.

    The deduplication key is *structured*: it folds the text (accents, case,
    quotes, ``&``), keeps the year and volume as discriminators (so different
    editions never collapse), and applies the publisher/city stop-list only
    outside the title span. ``consolidate()`` on the manager performs a later,
    conservative fuzzy merge within (author, year) blocks.
    """

    raw_text: str
    pages: set[int] = field(default_factory=set)
    normalized_key: str = ""
    metadata: dict[str, Any] | None = None
    doi: str | None = None
    url: str | None = None
    unnumbered: bool = False
    partial: bool = False
    variants: list[str] = field(default_factory=list)
    # Structured discriminators / comparison material (populated in __post_init__)
    year: int | None = None
    volume: int | None = None
    author: str = ""
    comparison_text: str = ""

    def __post_init__(self) -> None:
        """Generate structured normalized key for deduplication."""
        self.year = _extract_year(self.raw_text)
        self.volume = _extract_volume(self.raw_text)
        self.author = _first_author_surname(self.raw_text)
        if not self.normalized_key:
            self.normalized_key = self._generate_normalized_key()

    def _generate_normalized_key(self) -> str:
        """Create a structured normalized key for citation deduplication."""
        text = _fold(self.raw_text.strip())

        # Remove URLs and DOIs (identifiers handled separately).
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"doi:\s*10\.\S+", " ", text)
        text = re.sub(r"\b10\.\d{4,}/\S+", " ", text)

        # Remove page numbers, including German (S., SS.) and folio (fol.) forms
        # (p. 123, pp. 123-145, (pp. 123), S. 12-34); text is already folded to
        # lowercase here, but \b keeps "words." from matching the bare "s.".
        # Mirrors _PAGE_MARKER_RE's year guard so an author initial before a year
        # ("Sen, S. 1981") is not stripped here either (which would otherwise
        # collapse distinct editions onto one key); range alternative comes first.
        text = re.sub(
            r"\(?\s*\b(?:pp?|ss?|fol)\.\s*"
            r"(?:\d+\s*-\s*\d+|(?!(?:1[0-9]{3}|20[0-9]{2})\b)\d+)\s*\)?",
            " ",
            text,
        )

        # Protect the folded title span so the stop-list only strips the
        # non-title parts (keeps "the cambridge world history of food" intact).
        placeholders: dict[str, str] = {}
        for i, span in enumerate(_title_spans(self.raw_text)):
            folded_span = _fold(span)
            token = f" __title{i}__ "
            if folded_span and folded_span in text:
                text = text.replace(folded_span, token)
                placeholders[token.strip()] = folded_span

        text = _STOPLIST_RE.sub(" ", text)

        for token, folded_span in placeholders.items():
            text = text.replace(token, f" {folded_span} ")

        # Editor / translator markers and bracketed clarifications. The marker
        # must be a standalone token: the (?<![\w-]) / (?![\w-]) guards stop it
        # from firing inside a word or a hyphenated compound, so "Education"
        # keeps its "ed" and "Trans-Atlantic" keeps its "trans" (a bare \b would
        # still strip the latter, since a hyphen is a word boundary).
        text = re.sub(r"\(?\s*(?<![\w-])(?:eds?|trans)(?![\w-])\.?\s*\)?", " ", text)
        text = re.sub(r"\[[^\]]*\]", " ", text)

        # Strip punctuation (straightened quotes, apostrophes, ASCII hyphen).
        text = re.sub(r"[,.:;()\[\]\"'\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        self.comparison_text = text

        # Keep year and volume in the key material so different editions never
        # collapse onto the same hash.
        key_material = f"{text}|y={self.year}|v={self.volume}"
        return hashlib.md5(key_material.encode("utf-8")).hexdigest()

    def add_page(self, page: int | None) -> None:
        """Add a page number (or mark unnumbered when *page* is None)."""
        if page is None:
            self.unnumbered = True
        else:
            self.pages.add(page)

    def get_sorted_pages(self) -> list[int]:
        """Return sorted list of page numbers."""
        return sorted(self.pages)

    def get_page_range_str(self) -> str:
        """Return a formatted string of page numbers/ranges."""
        return _format_page_range(self.get_sorted_pages(), self.unnumbered)


class CitationManager:
    """Manages citations across a document with deduplication and metadata
    enrichment."""

    def __init__(self, polite_pool_email: str | None = None):
        """
        Initialize the citation manager.

        Args:
            polite_pool_email: Email for OpenAlex API polite pool access.
        """
        self.citations: dict[str, Citation] = {}
        self.polite_pool_email = polite_pool_email or OPENALEX_POLITE_POOL_EMAIL
        self._api_cache: dict[str, dict[str, Any] | None] = {}
        # Memoize the deterministic raw_text -> normalized_key derivation so a
        # repeated mention of the same citation skips the regex/NFKD/MD5
        # pipeline in Citation.__post_init__ (add_citations is O(mentions) of
        # this work otherwise; with the cache it is O(unique raw texts)).
        self._normalized_key_cache: dict[str, str] = {}
        self._openalex_budget_exhausted: bool = False

        # Merge/linking thresholds (config-exposed under `citation:`).
        try:
            from config import app as config

            self._merge_ratio = config.CITATION_MERGE_RATIO
            self._merge_jaccard = config.CITATION_MERGE_JACCARD
            self._match_title_overlap = config.CITATION_MATCH_TITLE_OVERLAP
        except Exception:
            self._merge_ratio = 0.90
            self._merge_jaccard = 0.85
            self._match_title_overlap = MATCH_RATIO_THRESHOLD

        # Persistent cross-run OpenAlex cache (keyed by normalized citation key).
        self._persistent_cache: dict[str, dict[str, Any]] = (
            _load_persistent_openalex_cache()
        )

    def add_citations(
        self,
        citations: Sequence[str | tuple[str, bool]],
        page_number: int | None,
    ) -> None:
        """
        Add citations from a page, handling deduplication.

        Args:
            citations: List of citations from a page. Each item is either a
                plain citation string (treated as a complete, non-partial
                reference) or a ``(text, is_partial)`` tuple where
                ``is_partial`` marks an in-text-only stub (author-year without
                full bibliographic data).
            page_number: The page number where these citations appear, or None
                for an unnumbered page (still recorded; rendered "unnumbered").
        """
        for item in citations:
            if isinstance(item, tuple):
                citation_text, is_partial = item
            else:
                citation_text, is_partial = item, False

            if not citation_text or not citation_text.strip():
                continue

            stripped = citation_text.strip()

            # Fast path: an identical raw text was already derived and its
            # citation is still present. Skip rebuilding the Citation (regex +
            # NFKD fold + MD5 key) and only merge this mention's page/partial
            # state, exactly as the "already exists" branch below would.
            cached_key = self._normalized_key_cache.get(stripped)
            if cached_key is not None and cached_key in self.citations:
                existing = self.citations[cached_key]
                existing.add_page(page_number)
                existing.partial = existing.partial and is_partial
                continue

            # Create or retrieve citation
            citation = Citation(raw_text=stripped, partial=is_partial)
            normalized_key = citation.normalized_key
            self._normalized_key_cache[stripped] = normalized_key

            if normalized_key in self.citations:
                # Citation already exists, just add the page. Full wins over
                # partial when the same key is seen with both flags.
                existing = self.citations[normalized_key]
                existing.add_page(page_number)
                existing.partial = existing.partial and is_partial
            else:
                # New citation
                citation.add_page(page_number)
                self.citations[normalized_key] = citation

    def consolidate(self) -> None:
        """Conservatively fuzzy-merge near-duplicate citations.

        Blocks citations on (first-author surname, exact year) so different
        years never merge, then within a block merges variants whose
        SequenceMatcher ratio clears ``merge_ratio`` or whose token-set Jaccard
        clears ``merge_jaccard``. Differing volumes never merge. The longest
        variant becomes canonical; pages/unnumbered union; the first non-null
        DOI/metadata wins; every merge is logged with both variants.
        """
        from collections import defaultdict

        blocks: dict[tuple[str, int | None], list[Citation]] = defaultdict(list)
        for citation in self.citations.values():
            blocks[(citation.author, citation.year)].append(citation)

        merged: dict[str, Citation] = {}
        for block in blocks.values():
            survivors: list[Citation] = []
            for candidate in block:
                target = self._find_merge_target(candidate, survivors)
                if target is None:
                    survivors.append(candidate)
                else:
                    self._merge_into(target, candidate)
            for survivor in survivors:
                merged[survivor.normalized_key] = survivor

        self.citations = merged
        self._resolve_partials()

    def _resolve_partials(self) -> None:
        """Merge or drop partial (in-text-only) citations by containment.

        Runs after the fuzzy-merge pass. Within each (first-author surname,
        exact year) block, a partial citation's comparison-text token set is
        typically just author names + year; a full reference for the same work
        is its token superset. For each partial:

        - exactly one full (non-partial) superset candidate -> merge the
          partial into it (full raw_text stays canonical; pages/unnumbered
          union);
        - zero candidates, or more than one (ambiguous) -> drop the partial
          entirely (do not guess), logged at info level.

        Full citations are never dropped or altered by this step.
        """
        from collections import defaultdict

        partials = [c for c in self.citations.values() if c.partial]
        if not partials:
            return

        blocks: dict[tuple[str, int | None], list[Citation]] = defaultdict(list)
        for citation in self.citations.values():
            blocks[(citation.author, citation.year)].append(citation)

        for partial in partials:
            block = blocks[(partial.author, partial.year)]
            partial_tokens = _token_set(partial.comparison_text)
            candidates = [
                full
                for full in block
                if not full.partial
                and full is not partial
                and partial_tokens <= _token_set(full.comparison_text)
            ]
            if len(candidates) == 1:
                self._merge_partial_into(candidates[0], partial)
                self.citations.pop(partial.normalized_key, None)
            else:
                reason = (
                    "no full match"
                    if not candidates
                    else f"ambiguous ({len(candidates)} full matches)"
                )
                logger.info(
                    "Dropping partial citation (%s): %s",
                    reason,
                    partial.raw_text,
                )
                self.citations.pop(partial.normalized_key, None)

    @staticmethod
    def _merge_partial_into(full: Citation, partial: Citation) -> None:
        """Merge a partial citation into a full one; full stays canonical.

        Unlike :meth:`_merge_into`, the full reference's ``raw_text`` always
        remains canonical regardless of length — a partial stub must never
        become the displayed citation. Pages/unnumbered union.
        """
        logger.info(
            "Merging partial citation into full:\n  - full:    %s\n  - partial: %s",
            full.raw_text,
            partial.raw_text,
        )
        full.pages |= partial.pages
        full.unnumbered = full.unnumbered or partial.unnumbered

    def _find_merge_target(
        self, candidate: Citation, survivors: list[Citation]
    ) -> Citation | None:
        """Return an existing survivor *candidate* should merge into, or None."""
        for survivor in survivors:
            if (
                candidate.volume is not None
                and survivor.volume is not None
                and candidate.volume != survivor.volume
            ):
                # Different volumes are genuinely different works.
                continue
            ratio = SequenceMatcher(
                None, candidate.comparison_text, survivor.comparison_text
            ).ratio()
            jaccard = _jaccard(candidate.comparison_text, survivor.comparison_text)
            if ratio >= self._merge_ratio or jaccard >= self._merge_jaccard:
                return survivor
        return None

    @staticmethod
    def _merge_into(survivor: Citation, other: Citation) -> None:
        """Merge *other* into *survivor* (longest variant becomes canonical)."""
        logger.info(
            "Merging citation variants:\n  - %s\n  - %s",
            survivor.raw_text,
            other.raw_text,
        )
        survivor.variants.extend(other.variants)
        if other.raw_text != survivor.raw_text:
            survivor.variants.append(other.raw_text)
        if len(other.raw_text) > len(survivor.raw_text):
            survivor.variants.append(survivor.raw_text)
            survivor.raw_text = other.raw_text
            # Promoting a new canonical raw_text invalidates the survivor's
            # derived fields; re-derive them so later merge/partial-resolution
            # and sorting decisions use the promoted text. Call
            # _generate_normalized_key only for its side effect of refreshing
            # comparison_text — discard the returned key, since it is the dict
            # key and must stay stable.
            survivor.year = _extract_year(survivor.raw_text)
            survivor.volume = _extract_volume(survivor.raw_text)
            survivor.author = _first_author_surname(survivor.raw_text)
            survivor._generate_normalized_key()

        survivor.pages |= other.pages
        survivor.unnumbered = survivor.unnumbered or other.unnumbered
        if survivor.doi is None:
            survivor.doi = other.doi
        if survivor.metadata is None:
            survivor.metadata = other.metadata
        if survivor.url is None:
            survivor.url = other.url

    def enrich_with_metadata(self, max_requests: int | None = None) -> None:
        """
        Enrich citations with metadata from OpenAlex API.

        Args:
            max_requests: Maximum number of API requests to make (None for unlimited).
        """
        logger.info("Enriching %d unique citations with metadata", len(self.citations))

        # Consult the process-wide / cross-run latch up front so the once-only
        # "budget exhausted" message fires even when this manager never issues a
        # request itself (a prior item or run tripped the daily quota).
        if _is_budget_exhausted():
            self._openalex_budget_exhausted = True

        requests_made = 0
        cache_hits = 0
        api_enriched = 0
        skipped_api = 0
        max_requests_notified = False
        budget_notified = False
        for processed, citation in enumerate(self.citations.values(), start=1):
            # Log progress periodically
            if processed % PROGRESS_LOG_INTERVAL == 0:
                logger.info(
                    "Processed %d/%d citations, enriched %d with metadata",
                    processed,
                    len(self.citations),
                    cache_hits + api_enriched,
                )

            # Persistent cross-run cache: reuse a prior lookup without spending
            # a request against the daily budget. Cache hits are always served,
            # even after the API budget is exhausted or the request cap is hit.
            cached = self._persistent_cache.get(citation.normalized_key)
            if cached is not None:
                self._apply_metadata(citation, cached)
                cache_hits += 1
                continue

            # Cache miss: this citation would require an API request. When the
            # API is unavailable (budget exhausted) or the per-item cap is
            # reached, do NOT break — keep iterating so later citations still
            # get their persistent-cache hits; only the request itself is
            # skipped, and the reason is logged once (not per citation).
            budget_out = self._openalex_budget_exhausted or _is_budget_exhausted()
            if budget_out:
                if not budget_notified:
                    logger.warning(
                        "OpenAlex daily budget exhausted — serving remaining "
                        "citations from the persistent cache only; skipping "
                        "API lookups for the rest of this item."
                    )
                    budget_notified = True
                skipped_api += 1
                continue

            if max_requests is not None and requests_made >= max_requests:
                if not max_requests_notified:
                    logger.info(
                        "Reached maximum API requests limit (%d) — serving "
                        "remaining citations from the persistent cache only.",
                        max_requests,
                    )
                    max_requests_notified = True
                skipped_api += 1
                continue

            # Every attempt counts against the budget (hit or miss), so a
            # document of hard-to-match citations cannot silently exceed the cap.
            metadata = self._fetch_metadata_from_openalex(citation.raw_text)
            requests_made += 1
            if metadata:
                self._apply_metadata(citation, metadata)
                self._persistent_cache[citation.normalized_key] = metadata
                api_enriched += 1
                # Be polite to the API
                time.sleep(API_POLITE_DELAY)

        # Post-enrichment: merge any citations that resolved to the same work.
        # This (and the cache save) must run even in cache-only mode, so it lives
        # after the loop completes rather than behind an early break.
        self._merge_by_shared_identifier()
        _save_persistent_openalex_cache(self._persistent_cache)

        logger.info(
            "Successfully enriched %d citations with metadata "
            "(%d from persistent cache, %d via API); "
            "skipped API lookup for %d citation(s).",
            cache_hits + api_enriched,
            cache_hits,
            api_enriched,
            skipped_api,
        )

    @staticmethod
    def _apply_metadata(citation: Citation, metadata: dict[str, Any]) -> None:
        """Attach fetched metadata to a citation."""
        citation.metadata = metadata
        citation.doi = metadata.get("doi")
        citation.url = metadata.get("url")

    def _merge_by_shared_identifier(self) -> None:
        """Merge citations that OpenAlex resolved to the same DOI / work id."""
        ident_map: dict[str, Citation] = {}
        new_citations: dict[str, Citation] = {}
        for citation in list(self.citations.values()):
            ident: str | None = None
            if citation.doi:
                ident = f"doi:{citation.doi.strip().lower()}"
            elif citation.metadata and citation.metadata.get("url"):
                ident = f"url:{str(citation.metadata['url']).strip().lower()}"

            if ident and ident in ident_map:
                self._merge_into(ident_map[ident], citation)
                continue

            new_citations[citation.normalized_key] = citation
            if ident:
                ident_map[ident] = citation

        self.citations = new_citations

    def _fetch_metadata_from_openalex(
        self, citation_text: str
    ) -> dict[str, Any] | None:
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

    def _extract_doi(self, citation_text: str) -> str | None:
        """Extract DOI from citation text if present."""
        # Common DOI patterns
        doi_patterns = [
            r"doi:\s*(10\.\d{4,}/[^\s]+)",
            r"https?://doi\.org/(10\.\d{4,}/[^\s]+)",
            r"https?://dx\.doi\.org/(10\.\d{4,}/[^\s]+)",
            r"\b(10\.\d{4,}/[^\s,;]+)",
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, citation_text, re.IGNORECASE)
            if match:
                doi = match.group(1).rstrip(".,;")
                return doi

        return None

    @staticmethod
    def _parse_retry_after(error_detail: dict[str, Any], response: Any) -> int:
        """Return the 429 retry-after delay in whole seconds (0 if unknown).

        Prefers the OpenAlex JSON body's ``retryAfter`` field; falls back to the
        standard HTTP ``Retry-After`` response header when the body lacks it.
        Only integer-second values are honored; a date-format ``Retry-After``
        header (or any unparsable value) is treated as unknown (``0``).
        """
        raw: Any = error_detail.get("retryAfter")
        if raw is None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                try:
                    raw = headers.get("Retry-After")
                except Exception:
                    raw = None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _make_openalex_request(
        self, url: str, params: dict[str, Any], context_description: str = ""
    ) -> dict[str, Any] | None:
        """
        Make a request to OpenAlex API with retry logic and error handling.

        Args:
            url: The API endpoint URL.
            params: Query parameters.
            context_description: Description for logging (e.g., "DOI 10.1234/abc").

        Returns:
            Response data if successful, None otherwise.
        """
        # Single choke point for both the DOI and text-search paths: if the
        # daily budget is known to be exhausted (this run or a prior run, via
        # the process-wide / cross-run latch), do not spend a request.
        if _is_budget_exhausted():
            self._openalex_budget_exhausted = True
            logger.debug(
                "OpenAlex daily budget latched as exhausted; skipping request for %s.",
                context_description,
            )
            return None

        for attempt in range(MAX_API_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=API_REQUEST_TIMEOUT)

                # Log request URL on first attempt if not successful
                if attempt == 0 and response.status_code != 200:
                    logger.debug("OpenAlex request URL: %s", response.url)

                if response.status_code == 200:
                    result: dict[str, Any] = response.json()
                    return result
                elif response.status_code == 404:
                    # 404 is expected when resource not found
                    return None
                elif response.status_code == 429:
                    # Rate-limit or paid-tier budget exhaustion.
                    try:
                        error_detail = response.json()
                    except Exception:
                        error_detail = {}
                    retry_after = self._parse_retry_after(error_detail, response)
                    if retry_after > BUDGET_EXHAUSTED_RETRY_AFTER_THRESHOLD:
                        # Long retryAfter means the daily budget is depleted;
                        # latch it process-wide + cross-run so no further items
                        # (this run or the next) re-hammer the API.
                        self._openalex_budget_exhausted = True
                        _mark_budget_exhausted(retry_after)
                        logger.warning(
                            "OpenAlex daily budget exhausted (retryAfter=%ds). "
                            "Disabling OpenAlex enrichment for %.0f minute(s).",
                            retry_after,
                            retry_after / 60.0,
                        )
                        return None
                    elif 0 < retry_after <= RATE_LIMIT_MAX_SLEEP:
                        # Short rate-limit: wait it out and retry within the
                        # existing attempt budget rather than dropping the
                        # citation.
                        if attempt < MAX_API_RETRIES - 1:
                            logger.warning(
                                "OpenAlex rate limit hit for %s (retryAfter=%ds; "
                                "attempt %d/%d). Sleeping and retrying.",
                                context_description,
                                retry_after,
                                attempt + 1,
                                MAX_API_RETRIES,
                            )
                            time.sleep(retry_after)
                            continue
                        logger.warning(
                            "OpenAlex rate limit hit for %s (retryAfter=%ds) "
                            "after %d attempts; giving up.",
                            context_description,
                            retry_after,
                            MAX_API_RETRIES,
                        )
                        return None
                    else:
                        # No usable retryAfter, or one between the short-sleep
                        # ceiling and the budget threshold: skip this citation.
                        logger.warning(
                            "OpenAlex rate limit hit for %s (retryAfter=%ds). "
                            "Skipping this citation.",
                            context_description,
                            retry_after,
                        )
                        return None
                elif response.status_code == 500:
                    # Transient server error — retry with exponential backoff.
                    delay = API_RETRY_DELAY * (2**attempt)
                    if attempt < MAX_API_RETRIES - 1:
                        logger.debug(
                            "OpenAlex server error 500 for %s (attempt %d/%d); "
                            "retrying in %.1fs.",
                            context_description,
                            attempt + 1,
                            MAX_API_RETRIES,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.warning(
                            "OpenAlex API returned status 500 for %s "
                            "after %d attempts; giving up.",
                            context_description,
                            MAX_API_RETRIES,
                        )
                        return None
                else:
                    # Other unexpected client/server errors — log once and skip.
                    try:
                        error_detail = response.json()
                        logger.warning(
                            "OpenAlex API returned status %d for %s: %s",
                            response.status_code,
                            context_description,
                            error_detail,
                        )
                    except Exception:
                        logger.warning(
                            "OpenAlex API returned status %d for %s",
                            response.status_code,
                            context_description,
                        )
                    return None
            except requests.RequestException as e:
                logger.warning(
                    "Error querying OpenAlex for %s (attempt %d/%d): %s",
                    context_description,
                    attempt + 1,
                    MAX_API_RETRIES,
                    str(e),
                )
                if attempt < MAX_API_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY)
            except Exception as e:
                logger.warning(
                    "Unexpected error querying OpenAlex for %s: %s",
                    context_description,
                    str(e),
                )
                return None

        return None

    def _query_openalex_by_doi(self, doi: str) -> dict[str, Any] | None:
        """Query OpenAlex API using DOI."""
        url = f"{OPENALEX_API_BASE}/works/https://doi.org/{doi}"
        params = {"mailto": self.polite_pool_email}

        data = self._make_openalex_request(url, params, f"DOI {doi}")
        if data:
            return self._extract_metadata_from_response(data)
        return None

    def _query_openalex_by_text(self, citation_text: str) -> dict[str, Any] | None:
        """Query OpenAlex API using citation text search.

        Fetches the top ``SEARCH_RESULTS_PER_PAGE`` candidates and returns
        the first one whose title passes :meth:`_verify_citation_match`.
        Requesting multiple candidates matters because OpenAlex's relevance
        ranking is noisy for long-tail citations: the true match is often
        rank 2-5 rather than rank 1.
        """
        # Extract key terms for better search
        search_query = self._extract_search_terms(citation_text)
        if not search_query or len(search_query) < 10:
            return None

        url = f"{OPENALEX_API_BASE}/works"
        params = {
            "search": search_query,
            "mailto": self.polite_pool_email,
            "per-page": SEARCH_RESULTS_PER_PAGE,
        }

        data = self._make_openalex_request(
            url, params, f"search query: {search_query[:50]}"
        )
        if not data:
            return None

        results = data.get("results") or []
        for candidate in results:
            if self._verify_citation_match(citation_text, candidate):
                return self._extract_metadata_from_response(candidate)

        return None

    def _extract_search_terms(self, citation_text: str) -> str:
        """Extract key search terms from citation text.

        Preferred strategy: take up to three author surnames followed by the
        italicized (``*title*``) or quoted (``"title"``) title. Including
        venue/publisher tokens (``Stanford University Press``) dilutes
        relevance on the OpenAlex ``/works`` endpoint and can push the true
        match off the first page of results. When no markup-delimited title
        is present, fall back to the naive cleanup as a last resort.
        """
        # Extract italicized title (markdown *...*) or quoted title ("...")
        title = ""
        ital = re.search(r"\*([^*\n]{5,})\*", citation_text)
        quoted = re.search(r'"([^"\n]{5,})"', citation_text)
        if ital:
            title = ital.group(1)
        elif quoted:
            title = quoted.group(1)

        # Extract author surnames: first few Capitalized words (>=3 chars)
        # appearing before the first year or opening parenthesis.
        head = re.split(r"\(|\d{4}", citation_text, maxsplit=1)[0]
        # Unicode-aware capitalized tokens (>=3 chars) so accented surnames
        # ("Müller") and apostrophe forms ("O'Brien") are captured rather than
        # skipped or truncated.
        surnames = [
            tok
            for tok in re.findall(r"[^\W\d_](?:[^\W\d_]|['’-])*", head)
            if len(tok) >= 3
            and tok[:1].isupper()
            and tok.lower() not in {"ed", "eds", "the", "and", "of", "in", "on"}
        ][:3]

        if title and surnames:
            combined = " ".join(surnames) + " " + title
            # Strip OpenAlex search operators (`*`, `"`, `?`, `!`) that
            # otherwise trigger 500 errors on /works.
            combined = re.sub(r'[*"?!]', " ", combined)
            combined = re.sub(r"\s+", " ", combined).strip()
            return combined[:SEARCH_QUERY_MAX_LENGTH]

        # Fallback: naive cleanup
        text = re.sub(r"\([^)]*\)", "", citation_text)
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"\d{4}", "", text)
        text = re.sub(r'[*"?!]', " ", text)
        text = re.sub(r"[,.:;]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:SEARCH_QUERY_MAX_LENGTH]

    def _verify_citation_match(
        self, citation_text: str, work_data: dict[str, Any]
    ) -> bool:
        """Verify that an OpenAlex result matches the citation (strict linking).

        Requires title-word overlap at or above the configured threshold AND a
        corroborating signal — the candidate's publication year within +/-1 of a
        year cited in the text, or the candidate's author surname appearing in
        the citation. This prefers no link over a wrong one (a permissive
        title-only match assigns wrong DOIs to look-alike titles).
        """
        raw_title = work_data.get("title") or work_data.get("display_name") or ""
        if not raw_title:
            return False

        citation_folded = _fold(citation_text)
        title_words = {w for w in _token_set(_fold(raw_title)) if len(w) > 3}
        if not title_words:
            return False

        citation_words = _token_set(citation_folded)

        # A grossly mismatched known year disqualifies outright: a coincidental
        # author surname must not link a 1950 citation to a 2001 candidate.
        cand_year = work_data.get("publication_year")
        cited_year = _extract_year(citation_text)
        year_diff: int | None = None
        if isinstance(cand_year, int) and cited_year is not None:
            year_diff = abs(cand_year - cited_year)
        if year_diff is not None and year_diff > 2:
            return False

        # Title check. A single substantive title token ("Nations") scores 1.0
        # against any citation containing that word, so demand it appear as an
        # explicit quoted/italic title span; otherwise require at least two
        # substantive title tokens in the overlap plus the ratio threshold.
        if len(title_words) < 2:
            folded_title = _fold(raw_title).strip()
            citation_title_spans = {
                _fold(span).strip() for span in _title_spans(citation_text)
            }
            if folded_title not in citation_title_spans:
                return False
        else:
            overlap_tokens = title_words & citation_words
            if len(overlap_tokens) < 2:
                return False
            if len(overlap_tokens) / len(title_words) < self._match_title_overlap:
                return False

        # Corroborating signal 1: publication year within +/-1 of a cited year.
        if year_diff is not None and year_diff <= 1:
            return True

        # Corroborating signal 2: candidate author surname present in citation.
        for authorship in work_data.get("authorships", []):
            author = (
                authorship.get("author", {}) if isinstance(authorship, dict) else {}
            )
            name = author.get("display_name") or ""
            parts = _fold(name).split()
            if parts and parts[-1] in citation_words:
                return True

        return False

    def _extract_metadata_from_response(
        self, work_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract relevant metadata from OpenAlex API response."""
        metadata: dict[str, Any] = {
            "title": work_data.get("title"),
            "doi": (
                work_data.get("doi", "").replace("https://doi.org/", "")
                if work_data.get("doi")
                else None
            ),
            "publication_year": work_data.get("publication_year"),
            "url": work_data.get("doi") or work_data.get("id"),
            "authors": [],
            "venue": None,
        }

        # Extract authors
        authorships = work_data.get("authorships", [])
        for authorship in authorships[:MAX_AUTHORS_TO_EXTRACT]:
            author = authorship.get("author", {})
            if author.get("display_name"):
                metadata["authors"].append(author["display_name"])

        # Extract venue
        primary_location = work_data.get("primary_location", {})
        if primary_location:
            source = primary_location.get("source", {})
            if source:
                metadata["venue"] = source.get("display_name")

        return metadata

    def get_sorted_citations(self) -> list[Citation]:
        """
        Return citations in a stable, accent-folded order.

        Sorts by (folded first-author surname, year, folded title/comparison
        text) so the order is deterministic across runs and does not push
        accented names after ``z`` (as a raw-text sort would).

        Returns:
            List of Citation objects in stable order.
        """

        def sort_key(c: Citation) -> tuple[str, int, str]:
            return (
                c.author or "~",
                c.year if c.year is not None else 9999,
                c.comparison_text or _fold(c.raw_text),
            )

        return sorted(self.citations.values(), key=sort_key)

    def get_citations_with_pages(self) -> list[tuple[Citation, str]]:
        """
        Return citations with formatted page information.

        Returns:
            List of tuples (Citation, page_range_string).
        """
        citations = self.get_sorted_citations()
        return [(citation, citation.get_page_range_str()) for citation in citations]


# ============================================================================
# Persistent cross-run OpenAlex cache
# ============================================================================

_OPENALEX_CACHE_FILE = "openalex_cache.json"


def _load_persistent_openalex_cache() -> dict[str, dict[str, Any]]:
    """Load the persistent OpenAlex cache (normalized key -> metadata)."""
    try:
        from config.state import read_json, resolve_state_file

        path = resolve_state_file(_OPENALEX_CACHE_FILE)
        data = read_json(path)
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load persistent OpenAlex cache: %s", exc)
        return {}


def _save_persistent_openalex_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Persist the OpenAlex cache atomically to the user state dir."""
    if not cache:
        return
    try:
        from config.state import resolve_state_file, write_json_atomic

        path = resolve_state_file(_OPENALEX_CACHE_FILE)
        write_json_atomic(path, cache)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not save persistent OpenAlex cache: %s", exc)


# ============================================================================
# Process-wide + cross-run OpenAlex daily-budget exhaustion latch
# ============================================================================
#
# When OpenAlex signals daily-quota exhaustion (a 429 with a long ``retryAfter``),
# a single ``CitationManager`` disabling itself is not enough: the pipeline builds
# a fresh manager per processed item, so a multi-document run would otherwise
# re-hammer the API after the quota is gone. This module-level latch records an
# ``exhausted_until`` epoch timestamp guarded by a lock, mirrors it to a state
# file so a subsequent process honors it too, and is consulted before every
# OpenAlex request.

_BUDGET_STATE_FILE = "openalex_budget.json"

_budget_lock = threading.Lock()
_budget_exhausted_until: float | None = None
_budget_state_loaded: bool = False


def _read_budget_state_file() -> float | None:
    """Read the persisted ``exhausted_until`` timestamp, or None on any failure."""
    try:
        from config.state import read_json, resolve_state_file

        path = resolve_state_file(_BUDGET_STATE_FILE)
        data = read_json(path)
        value = data.get("exhausted_until")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load OpenAlex budget state: %s", exc)
    return None


def _persist_budget_state(until: float) -> None:
    """Persist the ``exhausted_until`` timestamp atomically to the state dir."""
    try:
        from config.state import resolve_state_file, write_json_atomic

        path = resolve_state_file(_BUDGET_STATE_FILE)
        write_json_atomic(path, {"exhausted_until": until})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not save OpenAlex budget state: %s", exc)


def _openalex_budget_exhausted_until() -> float | None:
    """Return the epoch timestamp until which the OpenAlex budget is exhausted.

    Lazily loads the persisted state file exactly once per process (under the
    lock) so a run started after the quota tripped honors the remaining
    cooldown; subsequent calls read the cached in-memory value.
    """
    global _budget_exhausted_until, _budget_state_loaded
    with _budget_lock:
        if not _budget_state_loaded:
            _budget_state_loaded = True
            _budget_exhausted_until = _read_budget_state_file()
        return _budget_exhausted_until


def _is_budget_exhausted() -> bool:
    """Return True while the OpenAlex daily budget is latched as exhausted.

    A timestamp at or before the current time is treated as expired (the daily
    quota has reset), so requests are allowed again.
    """
    until = _openalex_budget_exhausted_until()
    return until is not None and until > time.time()


def _mark_budget_exhausted(retry_after: float) -> None:
    """Latch the OpenAlex budget as exhausted for *retry_after* seconds.

    Updates the in-memory state (under the lock) and mirrors it to the state
    file so a later process in the same daily window also backs off.
    """
    global _budget_exhausted_until, _budget_state_loaded
    until = time.time() + retry_after
    with _budget_lock:
        _budget_state_loaded = True
        _budget_exhausted_until = until
    _persist_budget_state(until)


def _reset_budget_state_for_tests() -> None:
    """Clear the in-memory budget latch (test hook only)."""
    global _budget_exhausted_until, _budget_state_loaded
    with _budget_lock:
        _budget_exhausted_until = None
        _budget_state_loaded = False
