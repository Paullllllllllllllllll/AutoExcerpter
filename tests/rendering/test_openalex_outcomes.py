"""OpenAlex lookup outcomes and the cache checkpoint, from recorded payloads.

Every lookup ends as found, not found, transient failure or budget exhausted.
Only found and not found are cached; the cache reaches the state directory on
``checkpoint()``; a budget-exhausted outcome stops the lookups of the client
that saw it and nothing else.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import requests

from autoexcerpter.rendering.citations import (
    Citation,
    CitationManager,
    LookupOutcome,
    LookupResult,
    OpenAlexCache,
    OpenAlexClient,
)
from autoexcerpter.rendering.citations.openalex import (
    CACHE_FILE,
    MAX_API_RETRIES,
    MISSES_FILE,
)
from autoexcerpter.state import read_json

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "openalex"

ALLEN = (
    "Allen, R. C. (2001). The great divergence in European wages and prices "
    "from the Middle Ages to the First World War. Explorations in Economic "
    "History, 38(4), 411-447. https://doi.org/10.1006/exeh.2001.0775"
)
BRAUDEL = (
    "Braudel, F. (1979). *Civilisation matérielle, économie et capitalisme*. "
    "Armand Colin."
)
UNKNOWN = "Nobody, N. (1923). An entirely unknown pamphlet on bread prices. Basel."
STALE_DOI = (
    "Ghost, G. (1990). A paper with a stale identifier. https://doi.org/10.9999/gone"
)
FLAKY = "Server, S. (1950). A lookup that meets a failing server. Zurich."
FLAKY_DOI = (
    "Shaky, S. (1991). An article behind a flaky resolver. "
    "https://doi.org/10.9999/flaky"
)
LIMITED = "Limit, L. (1960). The lookup that hits the daily limit. Bern."

# Substring of the request (URL and parameters) -> recorded payload.
ROUTES = {
    "10.1006/exeh.2001.0775": "allen_doi",
    "Civilisation": "braudel_title_search",
    "unknown pamphlet": "empty_search",
    "10.9999/gone": "doi_not_found",
    "stale identifier": "empty_search",
    "failing server": "server_error",
    "10.9999/flaky": "server_error",
    "flaky resolver": "empty_search",
    "daily limit": "daily_limit",
}


class _Response:
    """The subset of ``requests.Response`` the client reads."""

    def __init__(self, url: str, recorded: dict[str, Any]) -> None:
        self.url = url
        self.status_code = int(recorded["status"])
        self.headers: dict[str, str] = dict(recorded.get("headers", {}))
        self._body = recorded["body"]

    def json(self) -> Any:
        return json.loads(json.dumps(self._body))


class RecordedOpenAlex:
    """Answer OpenAlex GET requests from the recorded payloads."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def get(self, url: str, params: Any = None, **_: Any) -> _Response:
        target = f"{url} {json.dumps(params or {}, ensure_ascii=False)}"
        self.requests.append(target)
        for marker, name in ROUTES.items():
            if marker in target:
                path = FIXTURES / f"{name}.json"
                return _Response(url, json.loads(path.read_text(encoding="utf-8")))
        raise AssertionError(f"unexpected OpenAlex request: {target}")

    def session(self) -> _Session:
        return _Session(self)

    def asked(self, marker: str) -> int:
        """Return how many requests contained *marker*."""
        return sum(marker in target for target in self.requests)


class _Session:
    def __init__(self, server: RecordedOpenAlex) -> None:
        self._server = server

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> _Response:
        return self._server.get(url, **kwargs)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> RecordedOpenAlex:
    """Serve recorded payloads instead of the network; sleeping returns at once."""
    recorded = RecordedOpenAlex()
    monkeypatch.setattr(requests, "get", recorded.get)
    monkeypatch.setattr(requests, "Session", recorded.session)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    return recorded


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


def _key(text: str) -> str:
    return Citation(raw_text=text).normalized_key


def _full(*texts: str) -> list[tuple[str, bool]]:
    return [(text, False) for text in texts]


class TestLookupOutcomes:
    def test_doi_payload_is_found(self, server: RecordedOpenAlex) -> None:
        result = OpenAlexClient().lookup(ALLEN)

        assert result.outcome is LookupOutcome.FOUND
        assert result.metadata is not None
        assert result.metadata["doi"] == "10.1006/exeh.2001.0775"
        assert server.asked("10.1006/exeh.2001.0775") == 1

    def test_title_search_payload_is_found(self, server: RecordedOpenAlex) -> None:
        result = OpenAlexClient().lookup(BRAUDEL)

        assert result.outcome is LookupOutcome.FOUND
        assert result.metadata is not None
        assert result.metadata["url"] == "https://openalex.org/W1000000002"

    def test_empty_search_is_not_found(self, server: RecordedOpenAlex) -> None:
        assert OpenAlexClient().lookup(UNKNOWN) == LookupResult(LookupOutcome.NOT_FOUND)

    def test_unknown_doi_and_empty_search_is_not_found(
        self, server: RecordedOpenAlex
    ) -> None:
        result = OpenAlexClient().lookup(STALE_DOI)

        assert result.outcome is LookupOutcome.NOT_FOUND
        assert server.asked("10.9999/gone") == 1
        assert server.asked("stale identifier") == 1

    def test_server_errors_are_a_transient_failure(
        self, server: RecordedOpenAlex
    ) -> None:
        result = OpenAlexClient().lookup(FLAKY)

        assert result.outcome is LookupOutcome.TRANSIENT_FAILURE
        assert server.asked("failing server") == MAX_API_RETRIES

    def test_failed_doi_query_keeps_a_later_miss_transient(
        self, server: RecordedOpenAlex
    ) -> None:
        """The DOI might have resolved, so the empty search proves nothing."""
        result = OpenAlexClient().lookup(FLAKY_DOI)

        assert result.outcome is LookupOutcome.TRANSIENT_FAILURE
        assert server.asked("flaky resolver") == 1

    def test_connection_errors_are_a_transient_failure(
        self, server: RecordedOpenAlex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(url: str, **_: Any) -> Any:
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(requests, "get", refuse)

        result = OpenAlexClient().lookup(UNKNOWN)

        assert result.outcome is LookupOutcome.TRANSIENT_FAILURE

    def test_daily_limit_exhausts_the_budget_and_stops_lookups(
        self, server: RecordedOpenAlex
    ) -> None:
        client = OpenAlexClient()

        first = client.lookup(LIMITED)
        second = client.lookup(ALLEN)

        assert first.outcome is LookupOutcome.BUDGET_EXHAUSTED
        assert second.outcome is LookupOutcome.BUDGET_EXHAUSTED
        assert client.budget_exhausted is True
        assert server.asked("daily limit") == 1
        assert server.asked("10.1006/exeh.2001.0775") == 0


class TestEnrichmentCache:
    def test_only_not_found_is_cached_as_a_miss(
        self, server: RecordedOpenAlex, state_dir: Path
    ) -> None:
        manager = CitationManager(state_dir=state_dir)
        manager.add_citations(_full(ALLEN, UNKNOWN, FLAKY, LIMITED, BRAUDEL), 1)

        manager.enrich_with_metadata(max_requests=None)

        assert set(read_json(state_dir / CACHE_FILE)) == {_key(ALLEN)}
        assert read_json(state_dir / MISSES_FILE) == {"not_found": [_key(UNKNOWN)]}
        # The exhausted budget stopped the lookups before Braudel.
        assert server.asked("Civilisation") == 0

    def test_next_run_serves_the_cache_and_retries_the_rest(
        self, server: RecordedOpenAlex, state_dir: Path
    ) -> None:
        first = CitationManager(state_dir=state_dir)
        first.add_citations(_full(ALLEN, UNKNOWN, FLAKY, LIMITED, BRAUDEL), 1)
        first.enrich_with_metadata(max_requests=None)
        server.requests.clear()

        second = CitationManager(state_dir=state_dir)
        second.add_citations(_full(ALLEN, UNKNOWN, FLAKY, BRAUDEL), 1)
        second.enrich_with_metadata(max_requests=None)

        assert server.asked("10.1006/exeh.2001.0775") == 0
        assert server.asked("unknown pamphlet") == 0
        assert server.asked("failing server") == MAX_API_RETRIES
        assert server.asked("Civilisation") == 1
        assert second.citations[_key(ALLEN)].doi == "10.1006/exeh.2001.0775"
        assert set(read_json(state_dir / CACHE_FILE)) == {
            _key(ALLEN),
            _key(BRAUDEL),
        }

    def test_shared_client_stops_lookups_across_items(
        self, server: RecordedOpenAlex, state_dir: Path
    ) -> None:
        client = OpenAlexClient(state_dir=state_dir)
        first = CitationManager(openalex=client)
        first.add_citations(_full(LIMITED), 1)
        first.enrich_with_metadata()

        second = CitationManager(openalex=client)
        second.add_citations(_full(ALLEN), 1)
        second.enrich_with_metadata()

        assert server.asked("10.1006/exeh.2001.0775") == 0
        assert second.citations[_key(ALLEN)].metadata is None

    def test_lookup_is_overridable(
        self, server: RecordedOpenAlex, state_dir: Path
    ) -> None:
        """A subclass can skip lookups; a skipped citation is not cached."""

        class OnlyModernWorks(CitationManager):
            def lookup(self, citation: Citation) -> LookupResult:
                if citation.year is not None and citation.year < 1950:
                    return LookupResult(LookupOutcome.TRANSIENT_FAILURE)
                return super().lookup(citation)

        manager = OnlyModernWorks(state_dir=state_dir)
        manager.add_citations(_full(UNKNOWN, ALLEN), 1)
        manager.enrich_with_metadata()

        assert server.asked("unknown pamphlet") == 0
        assert manager.openalex.cache.get(_key(UNKNOWN)) is None
        assert not (state_dir / MISSES_FILE).exists()
        assert _key(ALLEN) in manager.openalex.cache


class TestCacheCheckpoint:
    _FOUND = LookupResult(LookupOutcome.FOUND, {"doi": "10.1/a", "url": None})
    _MISS = LookupResult(LookupOutcome.NOT_FOUND)

    def test_nothing_is_written_before_checkpoint(self, state_dir: Path) -> None:
        cache = OpenAlexCache(state_dir)
        cache.record("found", self._FOUND)

        assert not (state_dir / CACHE_FILE).exists()
        cache.checkpoint()
        assert read_json(state_dir / CACHE_FILE) == {"found": self._FOUND.metadata}

    def test_checkpoint_without_changes_writes_nothing(self, state_dir: Path) -> None:
        OpenAlexCache(state_dir).checkpoint()

        assert not (state_dir / CACHE_FILE).exists()
        assert not (state_dir / MISSES_FILE).exists()

    @pytest.mark.parametrize(
        "outcome",
        [LookupOutcome.TRANSIENT_FAILURE, LookupOutcome.BUDGET_EXHAUSTED],
    )
    def test_failures_are_not_cached(
        self, state_dir: Path, outcome: LookupOutcome
    ) -> None:
        cache = OpenAlexCache(state_dir)
        cache.record("key", LookupResult(outcome))
        cache.checkpoint()

        assert cache.get("key") is None
        assert not (state_dir / CACHE_FILE).exists()
        assert not (state_dir / MISSES_FILE).exists()

    def test_found_replaces_an_earlier_miss(self, state_dir: Path) -> None:
        cache = OpenAlexCache(state_dir)
        cache.record("key", self._MISS)
        cache.record("key", self._FOUND)
        cache.record("key", self._MISS)
        cache.checkpoint()

        assert OpenAlexCache(state_dir).get("key") == self._FOUND
        assert not (state_dir / MISSES_FILE).exists()

    def test_checkpoint_keeps_entries_saved_meanwhile(self, state_dir: Path) -> None:
        mine = OpenAlexCache(state_dir)
        theirs = OpenAlexCache(state_dir)
        assert mine.get("a") is None and theirs.get("b") is None

        theirs.record("b", self._MISS)
        theirs.checkpoint()
        mine.record("a", self._FOUND)
        mine.checkpoint()

        fresh = OpenAlexCache(state_dir)
        assert fresh.get("a") == self._FOUND
        assert fresh.get("b") == self._MISS
