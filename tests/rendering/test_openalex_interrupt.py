"""Interruptible OpenAlex waits and the enrichment's cache checkpoints.

A stop signal ends the polite delay and the client's retry and 429 waits
within 0.1 s while they still sleep through ``time.sleep``. The enrichment
checks the signal before every lookup and checkpoints the cache every
``CHECKPOINT_INTERVAL`` lookups and when it ends, also by an exception.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from autoexcerpter.rendering.citations import (
    Citation,
    CitationManager,
    LookupOutcome,
    LookupResult,
    OpenAlexClient,
)
from autoexcerpter.rendering.citations.manager import CHECKPOINT_INTERVAL
from autoexcerpter.rendering.citations.openalex import (
    CACHE_FILE,
    MISSES_FILE,
    STOP_POLL_INTERVAL,
    sleep_unless_stopped,
)
from autoexcerpter.state import read_json

URL = "https://api.openalex.org/works"


class Sleeps:
    """Record ``time.sleep`` calls; set *stop* after *stop_after* of them."""

    def __init__(self, stop: threading.Event, stop_after: int | None = None) -> None:
        self.stop = stop
        self.stop_after = stop_after
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            self.stop.set()


@pytest.fixture
def stop() -> threading.Event:
    return threading.Event()


def _response(status: int, body: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.url = URL
    response.headers = {}
    response.json.return_value = body
    return response


def _citations(count: int) -> list[tuple[str, bool]]:
    """Return *count* full citations with distinct authors and titles."""
    citations = []
    for index in range(count):
        name = "".join(chr(ord("a") + int(digit)) for digit in f"{index:02d}")
        text = f"{name.title()}son, A. ({1900 + index}). On the {name} question."
        citations.append((text, False))
    return citations


class _Session:
    """A ``requests.Session`` stand-in whose GETs go to *get*."""

    def __init__(self, get: MagicMock) -> None:
        self.get = get

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _serve(monkeypatch: pytest.MonkeyPatch, get: MagicMock) -> None:
    monkeypatch.setattr(requests, "Session", lambda: _Session(get))


def _key(text: str) -> str:
    return Citation(raw_text=text).normalized_key


class TestSleepUnlessStopped:
    def test_a_patched_sleep_ends_the_wait_at_once(self, stop: threading.Event) -> None:
        sleeps = Sleeps(stop)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(time, "sleep", sleeps)
            assert sleep_unless_stopped(30, stop) is True

        assert sum(sleeps.calls) == pytest.approx(30)
        assert max(sleeps.calls) <= STOP_POLL_INTERVAL

    def test_a_set_signal_ends_the_wait_within_one_slice(
        self, stop: threading.Event
    ) -> None:
        sleeps = Sleeps(stop, stop_after=3)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(time, "sleep", sleeps)
            assert sleep_unless_stopped(30, stop) is False

        assert len(sleeps.calls) == 3

    def test_without_a_signal_the_wait_is_one_sleep(self) -> None:
        sleeps = Sleeps(threading.Event())
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(time, "sleep", sleeps)
            assert sleep_unless_stopped(5, None) is True

        assert sleeps.calls == [5]


class TestClientWaits:
    @pytest.mark.parametrize(
        "reply",
        [
            {"return_value": _response(429, {"retryAfter": 20})},
            {"return_value": _response(429, {})},
            {"return_value": _response(500, {})},
            {"side_effect": requests.ConnectionError("refused")},
        ],
        ids=[
            "rate-limit",
            "rate-limit-unknown-wait",
            "server-error",
            "connection-error",
        ],
    )
    def test_a_stop_ends_the_retry_wait(
        self,
        stop: threading.Event,
        monkeypatch: pytest.MonkeyPatch,
        reply: dict[str, Any],
    ) -> None:
        get = MagicMock(**reply)
        _serve(monkeypatch, get)
        sleeps = Sleeps(stop, stop_after=2)
        monkeypatch.setattr(time, "sleep", sleeps)
        client = OpenAlexClient()

        with client.session(stop):
            result = client._request(URL, {}, "ctx")

        assert result == (LookupOutcome.TRANSIENT_FAILURE, None)
        assert get.call_count == 1
        assert len(sleeps.calls) == 2
        assert max(sleeps.calls) <= STOP_POLL_INTERVAL
        assert client.budget_exhausted is False

    def test_no_request_starts_once_stopped(
        self, stop: threading.Event, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        get = MagicMock()
        _serve(monkeypatch, get)
        stop.set()
        client = OpenAlexClient()

        with client.session(stop):
            result = client._request(URL, {}, "ctx")

        assert result == (LookupOutcome.TRANSIENT_FAILURE, None)
        get.assert_not_called()


class Recording(CitationManager):
    """Answer lookups in order: found, not found, then whatever *then* does."""

    def __init__(
        self,
        state_dir: Path,
        then: Callable[[Recording], LookupResult] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(state_dir=state_dir, **kwargs)
        self.then = then
        self.looked_up: list[str] = []

    def lookup(self, citation: Citation) -> LookupResult:
        self.looked_up.append(citation.raw_text)
        number = len(self.looked_up)
        if number == 1:
            return LookupResult(LookupOutcome.FOUND, {"doi": "10.1/a", "url": None})
        if number == 2 or self.then is None:
            return LookupResult(LookupOutcome.NOT_FOUND)
        return self.then(self)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


@pytest.mark.usefixtures("no_sleep")
class TestEnrichmentStop:
    def test_a_set_signal_prevents_every_lookup(
        self, tmp_path: Path, stop: threading.Event
    ) -> None:
        stop.set()
        manager = Recording(tmp_path / "state", stop=stop)
        manager.add_citations(_citations(3), 1)

        manager.enrich_with_metadata()

        assert manager.looked_up == []

    def test_a_stop_during_a_lookup_ends_the_enrichment(
        self, tmp_path: Path, stop: threading.Event
    ) -> None:
        def stop_now(manager: Recording) -> LookupResult:
            stop.set()
            return LookupResult(LookupOutcome.NOT_FOUND)

        state_dir = tmp_path / "state"
        manager = Recording(state_dir, then=stop_now, stop=stop)
        manager.add_citations(_citations(5), 1)

        manager.enrich_with_metadata()

        assert len(manager.looked_up) == 3
        assert set(read_json(state_dir / CACHE_FILE)) == {_key(manager.looked_up[0])}
        assert read_json(state_dir / MISSES_FILE) == {
            "not_found": sorted(_key(text) for text in manager.looked_up[1:])
        }


@pytest.mark.usefixtures("no_sleep")
class TestEnrichmentCheckpoints:
    def test_an_exception_keeps_the_earlier_results(self, tmp_path: Path) -> None:
        def fail(manager: Recording) -> LookupResult:
            raise RuntimeError("lookup exploded")

        state_dir = tmp_path / "state"
        manager = Recording(state_dir, then=fail)
        manager.add_citations(_citations(4), 1)

        with pytest.raises(RuntimeError, match="lookup exploded"):
            manager.enrich_with_metadata()

        assert len(manager.looked_up) == 3
        first, second = manager.looked_up[:2]
        assert set(read_json(state_dir / CACHE_FILE)) == {_key(first)}
        assert read_json(state_dir / MISSES_FILE) == {"not_found": [_key(second)]}

    def test_the_cache_is_checkpointed_every_interval(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        manager = Recording(state_dir)
        manager.add_citations(_citations(CHECKPOINT_INTERVAL + 1), 1)
        written: list[int] = []
        checkpoint = manager.openalex.cache.checkpoint

        def spy() -> None:
            written.append(len(manager.looked_up))
            checkpoint()

        manager.openalex.cache.checkpoint = spy  # type: ignore[method-assign]

        manager.enrich_with_metadata()

        assert written == [CHECKPOINT_INTERVAL, CHECKPOINT_INTERVAL + 1]
        assert len(read_json(state_dir / MISSES_FILE)["not_found"]) == (
            CHECKPOINT_INTERVAL
        )
