"""Tests for the progress-bar heartbeat used by the page-processing loop.

The heartbeat keeps the tqdm bar refreshing while nothing completes, so a run
stuck in a long retry ladder still reads as alive-but-stuck.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Generator
from typing import Any

import pytest

from pipeline.transcriber import _HEARTBEAT_INTERVAL_S, _ProgressHeartbeat

TICK = 0.05
# Generous upper bound for "a tick should have happened by now".
WAIT = 5.0
_AGE_RE = re.compile(r"last page (\d+)s ago")


class FakePbar:
    """Minimal tqdm stand-in recording postfix and refresh calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.postfixes: list[str] = []
        self.refresh_count = 0
        self.ticked = threading.Event()

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        with self._lock:
            self.postfixes.append(text)

    def refresh(self) -> None:
        with self._lock:
            self.refresh_count += 1
        self.ticked.set()

    def snapshot(self) -> tuple[list[str], int]:
        with self._lock:
            return list(self.postfixes), self.refresh_count


def _ages(postfixes: list[str]) -> list[int]:
    ages = []
    for text in postfixes:
        match = _AGE_RE.fullmatch(text)
        assert match is not None, f"unexpected postfix: {text!r}"
        ages.append(int(match.group(1)))
    return ages


@pytest.fixture
def heartbeats() -> Generator[list[_ProgressHeartbeat]]:
    """Track heartbeats so every test stops its threads, even on failure."""
    created: list[_ProgressHeartbeat] = []
    yield created
    for heartbeat in created:
        heartbeat.stop()


def _start(
    heartbeats: list[_ProgressHeartbeat], pbar: Any, interval: float = TICK
) -> _ProgressHeartbeat:
    heartbeat = _ProgressHeartbeat(pbar, interval=interval)
    heartbeats.append(heartbeat)
    heartbeat.start()
    return heartbeat


def test_default_interval_is_thirty_seconds() -> None:
    assert _HEARTBEAT_INTERVAL_S == 30.0


def test_refresh_fires_while_running(
    heartbeats: list[_ProgressHeartbeat],
) -> None:
    pbar = FakePbar()
    _start(heartbeats, pbar)

    assert pbar.ticked.wait(WAIT), "heartbeat never refreshed the bar"
    _, count = pbar.snapshot()
    assert count >= 1


def test_postfix_before_first_completion(
    heartbeats: list[_ProgressHeartbeat],
) -> None:
    pbar = FakePbar()
    _start(heartbeats, pbar)

    assert pbar.ticked.wait(WAIT)
    postfixes, _ = pbar.snapshot()
    assert postfixes[0] == "no page completed yet"


def test_postfix_reports_staleness_and_resets_on_progress(
    heartbeats: list[_ProgressHeartbeat],
) -> None:
    pbar = FakePbar()
    heartbeat = _ProgressHeartbeat(pbar, interval=TICK)
    heartbeats.append(heartbeat)
    # Backdate the last completion instead of waiting for real staleness.
    heartbeat._last_progress = time.monotonic() - 250.0
    heartbeat.start()

    assert pbar.ticked.wait(WAIT)
    stale_postfixes, _ = pbar.snapshot()
    assert _ages(stale_postfixes)[0] >= 249

    pbar.ticked.clear()
    heartbeat.mark_progress()
    before = len(stale_postfixes)
    assert pbar.ticked.wait(WAIT)
    fresh_postfixes, _ = pbar.snapshot()
    assert len(fresh_postfixes) > before
    assert _ages(fresh_postfixes[before:])[-1] < 5


def test_stop_terminates_thread_promptly() -> None:
    pbar = FakePbar()
    heartbeat = _ProgressHeartbeat(pbar, interval=10.0)
    heartbeat.start()
    thread = heartbeat._thread
    assert thread is not None

    started = time.monotonic()
    heartbeat.stop()
    elapsed = time.monotonic() - started

    assert not thread.is_alive()
    assert thread.daemon
    assert elapsed < 2.0
    assert heartbeat._thread is None


def test_no_refresh_after_stop_returns(
    heartbeats: list[_ProgressHeartbeat],
) -> None:
    pbar = FakePbar()
    heartbeat = _start(heartbeats, pbar)
    assert pbar.ticked.wait(WAIT)

    heartbeat.stop()
    _, count_at_stop = pbar.snapshot()

    time.sleep(TICK * 10)
    _, count_later = pbar.snapshot()
    assert count_later == count_at_stop


def test_stop_without_start_is_safe() -> None:
    heartbeat = _ProgressHeartbeat(FakePbar(), interval=TICK)
    heartbeat.stop()
    heartbeat.stop()


def test_start_is_idempotent(heartbeats: list[_ProgressHeartbeat]) -> None:
    pbar = FakePbar()
    heartbeat = _start(heartbeats, pbar, interval=10.0)
    thread = heartbeat._thread
    heartbeat.start()
    assert heartbeat._thread is thread


def test_tick_survives_a_failing_bar(heartbeats: list[_ProgressHeartbeat]) -> None:
    class ExplodingPbar(FakePbar):
        def refresh(self) -> None:
            super().refresh()
            raise RuntimeError("bar closed")

    pbar = ExplodingPbar()
    _start(heartbeats, pbar)

    assert pbar.ticked.wait(WAIT)
    pbar.ticked.clear()
    # The thread keeps ticking despite the exception.
    assert pbar.ticked.wait(WAIT)
