"""Tests for the OpenAlex client at the request level (citations/openalex.py).

Covers the HTTP request and its outcomes, Retry-After parsing and every wait
of the client (polite delay, rate-limit waits, backoff), the exhausted budget,
the pooled session, the per-document lookup cap, the request parameters,
malformed payloads and the enrichment options of the manager.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from autoexcerpter.rendering.citations import (
    CitationManager,
    LookupOutcome,
    LookupResult,
    OpenAlexClient,
    enrich_if_enabled,
)
from autoexcerpter.rendering.citations.matching import (
    extract_metadata,
    verify_citation_match,
)
from autoexcerpter.rendering.citations.openalex import (
    API_POLITE_DELAY,
    API_RETRY_DELAY,
    MAX_API_RETRIES,
    RATE_LIMIT_MAX_SLEEP,
    _parse_retry_after,
)
from autoexcerpter.state import read_json

_OPENALEX = "autoexcerpter.rendering.citations.openalex"
_URL = "https://api.openalex.org/works"
_NOT_FOUND = LookupResult(LookupOutcome.NOT_FOUND)


def _full(*texts: str) -> list[tuple[str, bool]]:
    """Wrap citation texts as complete (non-partial) reference tuples."""
    return [(text, False) for text in texts]


def _found(metadata: dict[str, Any]) -> LookupResult:
    return LookupResult(LookupOutcome.FOUND, metadata)


def _mock_429(retry_after: Any = None, header: str | None = None) -> MagicMock:
    """Build a 429 response with an optional body or header retry-after."""
    resp = MagicMock()
    resp.status_code = 429
    resp.json.return_value = {} if retry_after is None else {"retryAfter": retry_after}
    resp.headers = {} if header is None else {"Retry-After": header}
    return resp


def _mock_200(payload: Any) -> MagicMock:
    """Build a 200 response returning *payload* from ``.json()``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


class TestRequest:
    def test_request_success(self) -> None:
        """A 200 with a JSON object is a response to evaluate."""
        with patch(
            f"{_OPENALEX}.requests.get", return_value=_mock_200({"results": []})
        ):
            result = OpenAlexClient()._request(_URL, {"search": "test"}, "test query")

        assert result == (LookupOutcome.FOUND, {"results": []})

    def test_request_404_is_not_found(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch(f"{_OPENALEX}.requests.get", return_value=mock_response):
            result = OpenAlexClient()._request(f"{_URL}/invalid", {}, "invalid DOI")

        assert result == (LookupOutcome.NOT_FOUND, None)


class TestRetryAfterParsing:
    @pytest.mark.parametrize(
        ("body", "header", "seconds"),
        [
            ({"retryAfter": 7}, "999", 7),
            ({}, "12", 12),
            ({}, "Wed, 21 Oct 2015 07:28:00 GMT", 0),
            ({}, None, 0),
        ],
        ids=["body-over-header", "header", "date-header", "missing"],
    )
    def test_parse_retry_after(
        self, body: dict[str, Any], header: str | None, seconds: int
    ) -> None:
        resp = MagicMock()
        resp.headers = {} if header is None else {"Retry-After": header}

        assert _parse_retry_after(body, resp) == seconds


class TestWaits:
    """Every wait of the client, one test per behavior."""

    @pytest.mark.parametrize(
        "result",
        [_found({"doi": None, "url": None}), _NOT_FOUND],
        ids=["found", "not-found"],
    )
    def test_polite_delay_follows_every_lookup(self, result: LookupResult) -> None:
        manager = CitationManager()
        manager.add_citations(_full("Aaa, A. (2001). First Work. Press."), 1)
        manager.add_citations(_full("Bbb, B. (2002). Second Work. Press."), 2)

        with (
            patch.object(manager, "lookup", return_value=result) as mock_lookup,
            patch(f"{_OPENALEX}.time.sleep") as mock_sleep,
            patch.object(manager.openalex.cache, "checkpoint"),
        ):
            manager.enrich_with_metadata(max_requests=10)

        assert mock_lookup.call_count == 2
        assert mock_sleep.call_args_list == [call(API_POLITE_DELAY)] * 2

    @pytest.mark.parametrize(
        ("response", "seconds"),
        [
            ({"retry_after": 5}, 5),
            ({"header": "3"}, 3),
            ({"retry_after": RATE_LIMIT_MAX_SLEEP}, RATE_LIMIT_MAX_SLEEP),
        ],
        ids=["body", "header", "ceiling"],
    )
    def test_short_retry_after_is_slept_off_and_retried(
        self, response: dict[str, Any], seconds: int
    ) -> None:
        replies = [_mock_429(**response), _mock_200({"results": []})]

        with (
            patch(f"{_OPENALEX}.requests.get", side_effect=replies) as mock_get,
            patch(f"{_OPENALEX}.time.sleep") as mock_sleep,
        ):
            result = OpenAlexClient()._request(_URL, {}, "ctx")

        assert result == (LookupOutcome.FOUND, {"results": []})
        assert mock_get.call_count == 2
        assert mock_sleep.call_args_list == [call(seconds)]

    @pytest.mark.parametrize("retry_after", [120, 3600], ids=["medium", "large"])
    def test_long_retry_after_exhausts_the_budget(
        self, tmp_path: Path, retry_after: int
    ) -> None:
        """One request, no wait, and nothing persisted for the next run."""
        state_dir = tmp_path / "state"
        client = OpenAlexClient(state_dir=state_dir)

        with (
            patch(
                f"{_OPENALEX}.requests.get", return_value=_mock_429(retry_after)
            ) as mock_get,
            patch(f"{_OPENALEX}.time.sleep") as mock_sleep,
        ):
            result = client._request(_URL, {}, "ctx")

        assert result == (LookupOutcome.BUDGET_EXHAUSTED, None)
        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()
        assert client.budget_exhausted is True
        assert not state_dir.exists()

    def test_unknown_retry_after_backs_off_retries_then_fails(self) -> None:
        """A 429 without a usable retryAfter ends as a transient failure."""
        client = OpenAlexClient()

        with (
            patch(f"{_OPENALEX}.requests.get", return_value=_mock_429()) as mock_get,
            patch(f"{_OPENALEX}.time.sleep") as mock_sleep,
        ):
            result = client._request(_URL, {}, "ctx")

        assert result == (LookupOutcome.TRANSIENT_FAILURE, None)
        assert mock_get.call_count == MAX_API_RETRIES
        assert mock_sleep.call_args_list == [call(API_RETRY_DELAY)] * (
            MAX_API_RETRIES - 1
        )
        assert client.budget_exhausted is False

    def test_connection_error_backs_off_and_retries(self) -> None:
        replies: list[Any] = [
            requests.ConnectionError("refused"),
            requests.ConnectionError("refused"),
            _mock_200({"results": []}),
        ]

        with (
            patch(f"{_OPENALEX}.requests.get", side_effect=replies) as mock_get,
            patch(f"{_OPENALEX}.time.sleep") as mock_sleep,
        ):
            result = OpenAlexClient()._request(_URL, {}, "t")

        assert result == (LookupOutcome.FOUND, {"results": []})
        assert mock_get.call_count == 3
        assert mock_sleep.call_args_list == [call(API_RETRY_DELAY)] * 2


class TestBudgetExhausted:
    """An exhausted budget stops the lookups of one client and nothing else."""

    def test_cache_served_after_exhaustion_without_http(self) -> None:
        manager = CitationManager()
        manager.add_citations(_full("Alpha, A. (2001). Doomed Lookup. Some Press."), 1)
        manager.add_citations(_full("Beta, B. (2002). Cached Work. Other Press."), 2)
        manager.add_citations(_full("Gamma, G. (2003). Never Asked. Third Press."), 3)
        keys = list(manager.citations)
        metadata = {"doi": "10.1/two", "url": "https://doi.org/10.1/two"}
        manager.openalex.cache.record(keys[1], _found(metadata))

        with (
            patch("requests.Session.get", return_value=_mock_429(3600)) as session_get,
            patch(f"{_OPENALEX}.requests.get") as module_get,
        ):
            manager.enrich_with_metadata(max_requests=10)

        assert session_get.call_count == 1
        module_get.assert_not_called()
        assert manager.citations[keys[1]].doi == "10.1/two"
        assert manager.citations[keys[2]].metadata is None

    def test_a_new_client_looks_up_again(self) -> None:
        """Exhaustion is a state of one client, not of the process or disk."""
        exhausted = OpenAlexClient()
        with patch(f"{_OPENALEX}.requests.get", return_value=_mock_429(3600)):
            exhausted._request(_URL, {}, "ctx")

        manager = CitationManager()
        manager.add_citations(_full("Delta, D. (2004). Work with DOI 10.1234/d."), 1)
        work = {
            "title": "Delta Work",
            "doi": "https://doi.org/10.1234/d",
            "publication_year": 2004,
            "authorships": [],
            "primary_location": {},
        }
        with patch("requests.Session.get", return_value=_mock_200(work)) as mock_get:
            manager.enrich_with_metadata(max_requests=10)

        mock_get.assert_called()
        assert manager.openalex.budget_exhausted is False


class TestSessionReuse:
    """One pooled connection per enrichment run, reused and then closed."""

    _ONE = 'Aaa, Alfred. (2001). "A Sufficiently Long First Title." Some Press.'
    _TWO = 'Bbb, Bertha. (2002). "A Sufficiently Long Second Title." Other Press.'

    def test_single_session_serves_all_lookups_and_is_closed(self) -> None:
        """Both lookups share one requests.Session, which is closed at the end.

        A real ``requests.Session`` is instantiated (its ``get`` and ``close``
        are patched at class level, so nothing reaches the network) to keep the
        genuine context-manager semantics: ``__exit__`` must call ``close``.
        """
        manager = CitationManager()
        manager.add_citations(_full(self._ONE), 1)
        manager.add_citations(_full(self._TWO), 2)
        assert len(manager.citations) == 2

        real_session_cls = requests.Session
        created: list[requests.Session] = []

        def factory(*args: Any, **kwargs: Any) -> requests.Session:
            session = real_session_cls(*args, **kwargs)
            created.append(session)
            return session

        with (
            patch(f"{_OPENALEX}.requests.Session", side_effect=factory),
            patch.object(
                real_session_cls, "get", return_value=_mock_200({"results": []})
            ) as mock_get,
            patch.object(real_session_cls, "close") as mock_close,
            patch("time.sleep"),
        ):
            manager.enrich_with_metadata(max_requests=10)

        assert len(created) == 1
        assert mock_get.call_count == 2
        mock_close.assert_called_once()
        assert manager.openalex._session is None

    def test_session_closed_and_cleared_when_enrichment_raises(self) -> None:
        manager = CitationManager()
        manager.add_citations(_full(self._ONE), 1)

        real_session_cls = requests.Session

        with (
            patch.object(real_session_cls, "close") as mock_close,
            patch.object(manager, "lookup", side_effect=RuntimeError("x")),
            pytest.raises(RuntimeError),
        ):
            manager.enrich_with_metadata(max_requests=10)

        mock_close.assert_called_once()
        assert manager.openalex._session is None

    def test_direct_query_without_session_uses_module_level_get(self) -> None:
        """Outside an enrichment run there is no session; requests.get is used."""
        with patch(
            f"{_OPENALEX}.requests.get", return_value=_mock_200({"ok": True})
        ) as mock_get:
            result = OpenAlexClient()._request(_URL, {}, "ctx")

        assert result == (LookupOutcome.FOUND, {"ok": True})
        assert mock_get.call_count == 1


class TestLookupCap:
    """``max_requests`` caps the lookups of one enrichment call."""

    @pytest.mark.parametrize(
        "result",
        [_found({"doi": None, "url": None}), _NOT_FOUND],
        ids=["found", "not-found"],
    )
    def test_every_lookup_counts_against_the_cap(self, result: LookupResult) -> None:
        manager = CitationManager()
        manager.add_citations(_full("First, A. (2001). One. Press."), 1)
        manager.add_citations(_full("Second, B. (2002). Two. Press."), 2)
        manager.add_citations(_full("Third, C. (2003). Three. Press."), 3)

        with (
            patch.object(manager, "lookup", return_value=result) as mock_lookup,
            patch(f"{_OPENALEX}.time.sleep"),
        ):
            manager.enrich_with_metadata(max_requests=2)

        assert mock_lookup.call_count == 2

    def test_max_requests_continues_and_serves_later_cache_hits(self) -> None:
        """Citations after the cap still get their cached results."""
        manager = CitationManager()
        manager.add_citations(_full("Aaa, A. (2001). First Uncached. Press."), 1)
        manager.add_citations(_full("Bbb, B. (2002). Second Uncached. Press."), 2)
        manager.add_citations(_full("Ccc, C. (2003). Third Cached. Press."), 3)
        keys = list(manager.citations.keys())
        manager.openalex.cache.record(
            keys[2], _found({"doi": None, "url": None, "title": "Third Cached"})
        )

        with patch.object(manager, "lookup", return_value=_NOT_FOUND) as mock_lookup:
            manager.enrich_with_metadata(max_requests=1)

        # The first citation used the one lookup; the second was skipped.
        assert mock_lookup.call_count == 1
        third = manager.citations.get(keys[2])
        assert third is not None
        assert third.metadata is not None


class TestRequestParameters:
    """``mailto`` and ``api_key`` are sent only when configured."""

    _CITATION = 'Smith, J. (2020). "A Sufficiently Long Title." Journal.'
    _KEY = "test-key-123"

    @classmethod
    def _capture_params(cls, client: OpenAlexClient, query: str) -> dict[str, Any]:
        """Run a DOI or text query and return the params handed to requests.get."""
        captured: dict[str, Any] = {}

        def fake_get(url: str, params: Any = None, timeout: Any = None) -> MagicMock:
            captured.update(params or {})
            return _mock_200({"results": []})

        with patch(f"{_OPENALEX}.requests.get", side_effect=fake_get):
            if query == "doi":
                client._query_by_doi("10.1234/abc")
            else:
                client._query_by_text(cls._CITATION)
        return captured

    @pytest.mark.parametrize(
        ("email", "query", "mailto"),
        [
            (None, "text", None),
            (None, "doi", None),
            ("   ", "text", None),
            ("scholar@example.edu", "text", "scholar@example.edu"),
            ("scholar@example.edu", "doi", "scholar@example.edu"),
        ],
        ids=["unset-text", "unset-doi", "blank-text", "set-text", "set-doi"],
    )
    def test_mailto_param(
        self, email: str | None, query: str, mailto: str | None
    ) -> None:
        client = CitationManager(polite_pool_email=email).openalex

        params = self._capture_params(client, query)

        assert params.get("mailto") == mailto

    @pytest.mark.parametrize(
        ("api_key", "query", "sent"),
        [
            (None, "text", None),
            (MagicMock(), "text", None),
            (f"  {_KEY} ", "text", _KEY),
            (f"  {_KEY} ", "doi", _KEY),
        ],
        ids=["unset", "non-string", "set-text", "set-doi"],
    )
    def test_api_key_param(self, api_key: Any, query: str, sent: str | None) -> None:
        client = CitationManager(api_key=api_key).openalex

        params = self._capture_params(client, query)

        assert params.get("api_key") == sent

    def test_key_redacted_from_error_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = OpenAlexClient(api_key=self._KEY)
        error = requests.ConnectionError(f"GET {_URL}?api_key={self._KEY}")
        with (
            patch(f"{_OPENALEX}.requests.get", side_effect=error),
            patch(f"{_OPENALEX}.time.sleep"),
            caplog.at_level("DEBUG"),
        ):
            client._query_by_doi("10.1234/abc")
        assert caplog.text
        assert self._KEY not in caplog.text

    def test_key_redacted_from_echoed_error_body(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = OpenAlexClient(api_key=self._KEY)
        resp = MagicMock()
        resp.status_code = 403
        resp.json.return_value = {
            "error": f"bad key {self._KEY}",
            "url": "?api_key=x%2B",
        }
        with (
            patch(f"{_OPENALEX}.requests.get", return_value=resp),
            caplog.at_level("DEBUG"),
        ):
            client._query_by_doi("10.1234/abc")
        assert "status 403" in caplog.text
        assert self._KEY not in caplog.text
        assert "x%2B" not in caplog.text


class TestMalformedPayloads:
    """A proxy, a captive portal or an API change does not crash the run."""

    _CITATION = "Smith, John. *Introduction to Modern Testing*. Test Press, 2020."

    def test_non_object_200_body_is_a_transient_failure(self) -> None:
        with patch(f"{_OPENALEX}.requests.get", return_value=_mock_200([1, 2, 3])):
            result = OpenAlexClient()._request(_URL, {"search": "test"}, "test query")

        assert result == (LookupOutcome.TRANSIENT_FAILURE, None)

    def test_non_dict_candidates_are_skipped(self) -> None:
        body = {"results": [None, "junk", 42]}

        with patch(f"{_OPENALEX}.requests.get", return_value=_mock_200(body)):
            result = OpenAlexClient()._query_by_text(self._CITATION)

        assert result.outcome is LookupOutcome.NOT_FOUND

    def test_valid_candidate_after_junk_still_matches(self) -> None:
        body = {
            "results": [
                None,
                "junk",
                {
                    "title": "Introduction to Modern Testing",
                    "publication_year": 2020,
                    "doi": "https://doi.org/10.1234/test",
                },
            ]
        }

        with patch(f"{_OPENALEX}.requests.get", return_value=_mock_200(body)):
            result = OpenAlexClient()._query_by_text(self._CITATION)

        assert result.metadata is not None
        assert result.metadata["doi"] == "10.1234/test"

    def test_verify_match_tolerates_null_authorships(self) -> None:
        work = {"title": "Introduction to Modern Testing", "authorships": None}

        assert (
            verify_citation_match(
                "Smith, John. *Introduction to Modern Testing*. Test Press.", work
            )
            is False
        )

    def test_extract_metadata_tolerates_null_fields(self) -> None:
        work: dict[str, Any] = {
            "title": "A Title",
            "doi": None,
            "authorships": None,
            "primary_location": None,
        }

        metadata = extract_metadata(work)

        assert metadata["doi"] is None
        assert metadata["authors"] == []
        assert metadata.get("venue") is None

    def test_extract_metadata_tolerates_wrong_types(self) -> None:
        work: dict[str, Any] = {
            "title": "A Title",
            "doi": 12345,
            "authorships": [None, "junk", {"author": None}, {"author": {}}],
            "primary_location": {"source": None},
        }

        metadata = extract_metadata(work)

        assert metadata["doi"] is None
        assert metadata["authors"] == []
        assert metadata.get("venue") is None


class TestEnrichment:
    """The manager applies metadata, serves its cache and honors its options."""

    def test_enrichment_updates_citation(self) -> None:
        manager = CitationManager()
        manager.add_citations(
            _full("Test citation with DOI 10.1234/test"), page_number=1
        )
        metadata = {
            "doi": "10.1234/test",
            "url": "https://doi.org/10.1234/test",
            "title": "Test",
            "publication_year": 2020,
            "authors": ["Test Author"],
            "venue": "Test Journal",
        }

        with patch.object(manager, "lookup", return_value=_found(metadata)):
            manager.enrich_with_metadata(max_requests=1)

        citation = list(manager.citations.values())[0]
        assert citation.metadata is not None
        assert citation.doi == "10.1234/test"

    def test_cached_results_need_no_lookup(self) -> None:
        """A cached found result or miss is served without a lookup."""
        manager = CitationManager()
        manager.add_citations(_full("Alpha, A. (2001). Cached Work. Press."), 1)
        manager.add_citations(_full("Beta, B. (2002). Known Miss. Press."), 2)
        first, second = manager.citations
        manager.openalex.cache.record(first, _found({"doi": "10.1/a", "url": None}))
        manager.openalex.cache.record(second, _NOT_FOUND)

        with patch.object(manager, "lookup") as mock_lookup:
            manager.enrich_with_metadata(max_requests=10)

        mock_lookup.assert_not_called()
        assert manager.citations[first].doi == "10.1/a"
        assert manager.citations[second].metadata is None

    def test_enrich_if_enabled_honors_the_switch_and_the_cap(self) -> None:
        enabled = CitationManager(max_requests=7)
        with patch.object(enabled, "enrich_with_metadata") as enrich:
            enrich_if_enabled(enabled)
        enrich.assert_called_once_with(max_requests=7)

        disabled = CitationManager(openalex_enabled=False)
        with patch.object(disabled, "enrich_with_metadata") as enrich:
            enrich_if_enabled(disabled)
        enrich.assert_not_called()

    def test_cache_lives_in_the_given_state_dir(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state-elsewhere"
        manager = CitationManager(state_dir=state_dir)
        manager.add_citations(_full("Alpha, A. (2001). Cached Work. Press."), 1)
        key = next(iter(manager.citations))
        metadata = {"doi": "10.1/x", "url": "https://doi.org/10.1/x"}

        with patch.object(manager, "lookup", return_value=_found(metadata)):
            manager.enrich_with_metadata(max_requests=5)

        cache = read_json(state_dir / "openalex_cache.json")
        assert cache[key] == metadata
        reloaded = CitationManager(state_dir=state_dir).openalex.cache
        assert reloaded.get(key) == _found(metadata)
