"""OpenAlex lookups with explicit outcomes, and the lookup cache.

A lookup ends in one of four outcomes: found, not found, transient failure,
or budget exhausted (a 429 whose wait exceeds what a run can absorb, as when
the daily request limit is spent). After a budget-exhausted outcome the client
makes no further requests. :class:`OpenAlexCache` keeps found metadata and
not-found keys, nothing else, and persists them only on ``checkpoint()``.
A stop signal given to :meth:`OpenAlexClient.session` ends retry and
rate-limit waits early; the interrupted lookup ends as a transient failure.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from autoexcerpter.rendering.citations.matching import (
    MATCH_RATIO_THRESHOLD,
    SEARCH_QUERY_MAX_LENGTH,
    extract_doi,
    extract_metadata,
    extract_search_terms,
    extract_title,
    looks_like_review,
    publication_year_filter,
    truncate_at_word,
    verify_citation_match,
)
from autoexcerpter.rendering.citations.model import cited_years
from autoexcerpter.state import read_json, resolve_state_file, write_json_atomic

logger = logging.getLogger(__name__)

OPENALEX_API_BASE = "https://api.openalex.org"
API_REQUEST_TIMEOUT = 10
API_RETRY_DELAY = 1.0
MAX_API_RETRIES = 3
API_POLITE_DELAY = 0.1  # Pause after every lookup
# A 429 whose retryAfter (seconds) is at or below this is slept off and
# retried; a longer wait cannot be absorbed by a run, so the lookup ends as
# budget exhausted.
RATE_LIMIT_MAX_SLEEP = 30
SEARCH_RESULTS_PER_PAGE = 5  # Candidates fetched per search; first matching wins
STOP_POLL_INTERVAL = 0.1  # Longest sleep slice between two stop checks

CACHE_FILE = "openalex_cache.json"
MISSES_FILE = "openalex_misses.json"


def sleep_unless_stopped(seconds: float, stop: threading.Event | None) -> bool:
    """Sleep *seconds* through ``time.sleep``, checking *stop* every 0.1 s.

    With a stop signal the wait runs in slices of at most 0.1 s, counted
    rather than timed, so a patched ``time.sleep`` that returns at once still
    ends the wait at once; without one it is a single sleep. Returns False
    when *stop* is set before or during the wait.
    """
    if stop is None:
        time.sleep(seconds)
        return True
    remaining = seconds
    while remaining > 0:
        if stop.is_set():
            return False
        step = min(STOP_POLL_INTERVAL, remaining)
        time.sleep(step)
        remaining -= step
    return not stop.is_set()


class LookupOutcome(StrEnum):
    """How one OpenAlex lookup ended."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    TRANSIENT_FAILURE = "transient_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class LookupResult:
    """The outcome of one lookup and, when found, the work's metadata."""

    outcome: LookupOutcome
    metadata: dict[str, Any] | None = None


_NOT_FOUND = LookupResult(LookupOutcome.NOT_FOUND)
_TRANSIENT = LookupResult(LookupOutcome.TRANSIENT_FAILURE)
_EXHAUSTED = LookupResult(LookupOutcome.BUDGET_EXHAUSTED)

type _Reply = tuple[LookupOutcome, dict[str, Any] | None]
_TRANSIENT_REPLY: _Reply = (LookupOutcome.TRANSIENT_FAILURE, None)


class OpenAlexCache:
    """Lookup results keyed by normalized citation key.

    Only found and not-found outcomes are cached. The state directory is read
    on first use and written only by :meth:`checkpoint`, which merges with the
    files on disk so entries another process saved meanwhile are kept.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = state_dir
        self._found: dict[str, dict[str, Any]] = {}
        self._not_found: set[str] = set()
        self._loaded = False
        self._dirty = False

    def _read(self) -> tuple[dict[str, dict[str, Any]], set[str]]:
        found = {
            key: value
            for key, value in read_json(
                resolve_state_file(CACHE_FILE, self.state_dir)
            ).items()
            if isinstance(value, dict)
        }
        listed = read_json(resolve_state_file(MISSES_FILE, self.state_dir)).get(
            "not_found"
        )
        not_found = (
            {key for key in listed if isinstance(key, str) and key}
            if isinstance(listed, list)
            else set()
        )
        return found, not_found - set(found)

    def _merge_disk(self) -> None:
        found, not_found = self._read()
        self._found = found | self._found
        self._not_found = (not_found | self._not_found) - set(self._found)

    def get(self, key: str) -> LookupResult | None:
        """Return the cached result for *key*, or None when it is not cached."""
        if not self._loaded:
            self._loaded = True
            self._merge_disk()
        if key in self._found:
            return LookupResult(LookupOutcome.FOUND, self._found[key])
        if key in self._not_found:
            return _NOT_FOUND
        return None

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None

    def record(self, key: str, result: LookupResult) -> None:
        """Cache *result* under *key* when it is found or not found."""
        cached = self.get(key)
        if result.outcome is LookupOutcome.FOUND and result.metadata is not None:
            self._found[key] = result.metadata
            self._not_found.discard(key)
        elif result.outcome is LookupOutcome.NOT_FOUND and cached is None:
            self._not_found.add(key)
        else:
            return
        self._dirty = True

    def checkpoint(self) -> None:
        """Write the cache to the state directory if it changed."""
        if not self._dirty:
            return
        self._merge_disk()
        if self._found:
            write_json_atomic(
                resolve_state_file(CACHE_FILE, self.state_dir), self._found
            )
        if self._not_found:
            write_json_atomic(
                resolve_state_file(MISSES_FILE, self.state_dir),
                {"not_found": sorted(self._not_found)},
            )
        self._dirty = False


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
            except (AttributeError, TypeError):
                raw = None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


class OpenAlexClient:
    """Look citations up in OpenAlex; one client serves one run.

    The client holds the lookup cache and the budget state of the run: once a
    lookup ends as budget exhausted, every later :meth:`lookup` returns that
    outcome without a request. The API key is redacted from every log line.
    """

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        *,
        state_dir: Path | None = None,
        cache: OpenAlexCache | None = None,
    ) -> None:
        """Create a client.

        Args:
            email: Contact sent as ``mailto``; blank sends none.
            api_key: Sent as ``api_key``; blank or non-string sends none.
            state_dir: Folder of the cache when *cache* is None; None takes
                the default state directory.
            cache: The lookup cache; None creates one for *state_dir*.
        """
        self.email = (email or "").strip()
        self.api_key = api_key.strip() if isinstance(api_key, str) else ""
        self.cache = cache if cache is not None else OpenAlexCache(state_dir)
        self.title_overlap = MATCH_RATIO_THRESHOLD
        self._budget_exhausted = False
        # Connection pool while session() is open; None outside it, in which
        # case requests go through the module-level requests.get.
        self._session: requests.Session | None = None
        self._stop: threading.Event | None = None

    @property
    def budget_exhausted(self) -> bool:
        """Whether a lookup of this client ended as budget exhausted."""
        return self._budget_exhausted

    @contextmanager
    def session(self, stop: threading.Event | None = None) -> Iterator[None]:
        """Share one connection pool among the lookups inside the block.

        Once *stop* is set, no further request starts and retry waits end;
        the lookup in progress ends as a transient failure.
        """
        self._stop = stop
        try:
            with requests.Session() as session:
                self._session = session
                yield
        finally:
            self._session = None
            self._stop = None

    def _stopped(self) -> bool:
        return self._stop is not None and self._stop.is_set()

    def lookup(self, citation_text: str) -> LookupResult:
        """Look *citation_text* up and return the outcome.

        A DOI in the text is queried first; when it finds nothing, a title
        search (or a free-text search for citations without an isolable
        title) follows. The method neither reads nor writes the cache.
        Subclasses may override it to skip or record lookups.
        """
        if self._budget_exhausted:
            return _EXHAUSTED
        doi_outcome: LookupOutcome | None = None
        doi = extract_doi(citation_text)
        if doi:
            result = self._query_by_doi(doi)
            if result.outcome in (LookupOutcome.FOUND, LookupOutcome.BUDGET_EXHAUSTED):
                return result
            doi_outcome = result.outcome
        result = self._query_by_text(citation_text)
        if (
            result.outcome is LookupOutcome.NOT_FOUND
            and doi_outcome is LookupOutcome.TRANSIENT_FAILURE
        ):
            return _TRANSIENT
        return result

    def _request_params(self) -> dict[str, Any]:
        """Return the ``mailto`` and ``api_key`` parameters that are configured.

        OpenAlex treats ``mailto`` as an opt-in courtesy; sending a blank or
        placeholder address is worse than sending none, so an unset email
        simply omits the parameter. The same holds for an unset API key.
        """
        params: dict[str, Any] = {}
        if self.email:
            params["mailto"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _redact(self, text: str) -> str:
        """Replace the API key in *text* (URL, error message, response body).

        Covers the raw key, its URL-encoded form, and any ``api_key=`` query
        value, so a key echoed back by the server or an intermediary is also
        masked.
        """
        if self.api_key:
            for form in {self.api_key, quote(self.api_key, safe="")}:
                text = text.replace(form, "***")
        return re.sub(r"(api_key=)[^&\s'\"]+", r"\1***", text)

    def _query_by_doi(self, doi: str) -> LookupResult:
        """Look a work up by DOI."""
        url = f"{OPENALEX_API_BASE}/works/https://doi.org/{doi}"
        outcome, data = self._request(url, self._request_params(), f"DOI {doi}")
        if outcome is not LookupOutcome.FOUND:
            return LookupResult(outcome)
        if not data:
            return _NOT_FOUND
        return LookupResult(LookupOutcome.FOUND, extract_metadata(data))

    def _query_by_text(self, citation_text: str) -> LookupResult:
        """Look a work up by citation text.

        Fetches the top ``SEARCH_RESULTS_PER_PAGE`` candidates and returns
        the first one that passes :func:`verify_citation_match`. Requesting
        multiple candidates matters because OpenAlex's relevance ranking is
        noisy for long-tail citations: the true match is often rank 2-5.

        When a title can be isolated, the query is a ``title.search`` filter
        restricted to the cited year +/-1 (for a reprint, either cited year
        +/-1). OpenAlex runs its ``search`` parameter against full text, so
        sending the whole citation (authors, venue, volume, pages) ranks
        unrelated works first; the title filter finds the cited work at the
        same cost. Citations without an isolable title keep the free-text
        search. An empty title-filter result is final: no second query.
        """
        url = f"{OPENALEX_API_BASE}/works"
        title = extract_title(citation_text)
        params: dict[str, Any]
        if title:
            # Commas separate filters and "|" separates OR values, so all
            # punctuation is dropped from the title before it enters a filter.
            query = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", title)).strip()
            query = truncate_at_word(query, SEARCH_QUERY_MAX_LENGTH)
            filters = [f"title.search:{query}"]
            year_filter = publication_year_filter(cited_years(citation_text))
            if year_filter:
                filters.append(year_filter)
            params = {"filter": ",".join(filters)}
            description = f"title search: {query[:50]}"
        else:
            search_query = extract_search_terms(citation_text)
            if not search_query or len(search_query) < 10:
                return _NOT_FOUND
            params = {"search": search_query}
            description = f"search query: {search_query[:50]}"
        params.update({"per-page": SEARCH_RESULTS_PER_PAGE, **self._request_params()})

        outcome, data = self._request(url, params, description)
        if outcome is not LookupOutcome.FOUND:
            return LookupResult(outcome)
        if not data:
            return _NOT_FOUND

        for candidate in data.get("results") or []:
            if not isinstance(candidate, dict):
                continue
            if looks_like_review(citation_text, candidate):
                continue
            if verify_citation_match(citation_text, candidate, self.title_overlap):
                return LookupResult(LookupOutcome.FOUND, extract_metadata(candidate))

        return _NOT_FOUND

    def _request(
        self, url: str, params: dict[str, Any], context_description: str = ""
    ) -> tuple[LookupOutcome, dict[str, Any] | None]:
        """GET *url* with retries; return the outcome and the JSON body.

        FOUND means a 200 with a JSON object (the body may still hold no
        match); 404 is NOT_FOUND; a long 429 is BUDGET_EXHAUSTED; everything
        else that does not succeed within ``MAX_API_RETRIES`` attempts is
        TRANSIENT_FAILURE, as is a request skipped or a wait cut short by the
        session's stop signal. Requests go through the pooled session while
        :meth:`session` is open.
        """
        if self._budget_exhausted:
            return LookupOutcome.BUDGET_EXHAUSTED, None

        session = self._session
        http_get = requests.get if session is None else session.get

        for attempt in range(MAX_API_RETRIES):
            if self._stopped():
                return _TRANSIENT_REPLY
            try:
                response = http_get(url, params=params, timeout=API_REQUEST_TIMEOUT)
                if attempt == 0 and response.status_code != 200:
                    logger.debug(
                        "OpenAlex request URL: %s", self._redact(str(response.url))
                    )
                reply = self._handle_response(response, attempt, context_description)
            except requests.RequestException as e:
                reply = self._handle_request_error(e, attempt, context_description)
            except Exception as e:  # noqa: BLE001 - one lookup must not fail the run
                logger.warning(
                    "Unexpected error querying OpenAlex for %s: %s",
                    context_description,
                    self._redact(str(e)),
                )
                return _TRANSIENT_REPLY
            if reply is not None:
                return reply

        return _TRANSIENT_REPLY

    def _handle_response(
        self, response: Any, attempt: int, context_description: str
    ) -> _Reply | None:
        """Map one HTTP response to a reply, or None to retry."""
        status = response.status_code
        if status == 200:
            # A proxy, captive portal, or API change can return 200 with a
            # non-object JSON body.
            result = response.json()
            if isinstance(result, dict):
                return LookupOutcome.FOUND, result
            return _TRANSIENT_REPLY
        if status == 404:
            return LookupOutcome.NOT_FOUND, None
        if status == 429:
            return self._handle_rate_limit(response, attempt, context_description)
        if status == 500:
            return self._handle_server_error(attempt, context_description)
        try:
            detail = self._redact(str(response.json()))[:500]
        except ValueError:
            detail = ""
        logger.warning(
            "OpenAlex API returned status %d for %s%s",
            status,
            context_description,
            f": {detail}" if detail else "",
        )
        return _TRANSIENT_REPLY

    def _handle_rate_limit(
        self, response: Any, attempt: int, context_description: str
    ) -> _Reply | None:
        """Map a 429 to budget exhausted, a retry after a wait, or a failure."""
        try:
            error_detail = response.json()
        except ValueError:
            error_detail = {}
        if not isinstance(error_detail, dict):
            error_detail = {}
        retry_after = _parse_retry_after(error_detail, response)
        if retry_after > RATE_LIMIT_MAX_SLEEP:
            self._budget_exhausted = True
            logger.warning(
                "OpenAlex request limit reached (retryAfter=%ds); "
                "no further OpenAlex lookups in this run.",
                retry_after,
            )
            return LookupOutcome.BUDGET_EXHAUSTED, None
        if attempt < MAX_API_RETRIES - 1:
            # A short wait is absorbed; without a usable retryAfter, back off
            # a fixed delay so a 429 storm does not burn one citation per
            # response.
            delay = retry_after if retry_after > 0 else API_RETRY_DELAY
            logger.warning(
                "OpenAlex rate limit hit for %s (retryAfter=%s; "
                "attempt %d/%d). Sleeping %.1fs and retrying.",
                context_description,
                retry_after or "unknown",
                attempt + 1,
                MAX_API_RETRIES,
                delay,
            )
            return self._wait_to_retry(delay)
        logger.warning(
            "OpenAlex rate limit hit for %s after %d attempts; skipping this citation.",
            context_description,
            MAX_API_RETRIES,
        )
        return _TRANSIENT_REPLY

    def _handle_server_error(
        self, attempt: int, context_description: str
    ) -> _Reply | None:
        """Retry a 500 with exponential backoff until the attempts run out."""
        delay = API_RETRY_DELAY * (2**attempt)
        if attempt < MAX_API_RETRIES - 1:
            logger.debug(
                "OpenAlex server error 500 for %s (attempt %d/%d); retrying in %.1fs.",
                context_description,
                attempt + 1,
                MAX_API_RETRIES,
                delay,
            )
            return self._wait_to_retry(delay)
        logger.warning(
            "OpenAlex API returned status 500 for %s after %d attempts; giving up.",
            context_description,
            MAX_API_RETRIES,
        )
        return _TRANSIENT_REPLY

    def _handle_request_error(
        self, error: requests.RequestException, attempt: int, context_description: str
    ) -> _Reply | None:
        """Log a network error; wait before the next attempt, if one is left."""
        logger.warning(
            "Error querying OpenAlex for %s (attempt %d/%d): %s",
            context_description,
            attempt + 1,
            MAX_API_RETRIES,
            self._redact(str(error)),
        )
        if attempt < MAX_API_RETRIES - 1:
            return self._wait_to_retry(API_RETRY_DELAY)
        return None

    def _wait_to_retry(self, delay: float) -> _Reply | None:
        """Wait *delay* seconds; a stop signal ends the lookup as transient."""
        if sleep_unless_stopped(delay, self._stop):
            return None
        return _TRANSIENT_REPLY


__all__ = [
    "CACHE_FILE",
    "MAX_API_RETRIES",
    "MISSES_FILE",
    "LookupOutcome",
    "LookupResult",
    "OpenAlexCache",
    "OpenAlexClient",
    "sleep_unless_stopped",
]
