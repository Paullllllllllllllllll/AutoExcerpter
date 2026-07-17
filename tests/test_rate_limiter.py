"""Tests for llm/rate_limit.py - Rate limiting with adaptive backoff."""

from __future__ import annotations

import threading
import time

import pytest

from config.constants import MAX_SLEEP_TIME
from llm.rate_limit import RateLimiter


class _SleepInterrupt(Exception):
    """Raised by the fake sleep to abort wait_for_capacity's blocking loop.

    Recording the requested sleep duration and then unwinding lets the tests
    assert that a real wait was demanded, without ever sleeping in real time.
    """


def _record_first_sleep(monkeypatch: pytest.MonkeyPatch, store: list[float]) -> None:
    """Patch the rate limiter's ``time.sleep`` to capture the first requested
    sleep duration and abort the wait loop via ``_SleepInterrupt``.

    A limiter that never blocks never calls sleep, so ``store`` stays empty and
    the caller's enforcement assertions fail — exactly what should happen for a
    broken (no-op) limiter.

    The limiter calls ``time.sleep`` via ``import time``, so patching the
    stdlib ``time`` module's ``sleep`` intercepts its blocking wait.
    """

    def fake_sleep(seconds: float) -> None:
        store.append(seconds)
        raise _SleepInterrupt

    monkeypatch.setattr(time, "sleep", fake_sleep)


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_with_single_limit(self) -> None:
        """Initializes with a single rate limit."""
        limits = [(10, 1)]  # 10 requests per second
        limiter = RateLimiter(limits)

        assert limiter.limits == limits
        assert len(limiter.request_timestamps) == 1
        assert limiter.total_requests == 0

    def test_init_with_multiple_limits(self) -> None:
        """Initializes with multiple rate limits."""
        limits = [(120, 1), (15000, 60)]  # 120/sec, 15000/min
        limiter = RateLimiter(limits)

        assert limiter.limits == limits
        assert len(limiter.request_timestamps) == 2

    def test_initial_state(self) -> None:
        """Initial state is correctly set."""
        limiter = RateLimiter([(10, 1)])

        assert limiter.total_requests == 0
        assert limiter.total_wait_time == 0.0
        assert limiter.consecutive_errors == 0
        assert limiter.error_multiplier == 1.0


class TestWaitForCapacity:
    """Tests for wait_for_capacity method."""

    def test_first_request_no_wait(self) -> None:
        """First request has no wait time."""
        limiter = RateLimiter([(10, 1)])

        wait_time = limiter.wait_for_capacity()

        assert wait_time < 0.1  # Should be near-instant
        assert limiter.total_requests == 1

    def test_requests_recorded(self) -> None:
        """Requests are recorded in timestamps."""
        limiter = RateLimiter([(10, 1)])

        limiter.wait_for_capacity()
        limiter.wait_for_capacity()

        assert len(limiter.request_timestamps[0]) == 2

    def test_error_penalty_delays_but_still_admits(self, monkeypatch) -> None:
        """An elevated multiplier must impose a real delay even when no window
        is saturated (the multiplier used to scale a zero wait, a silent
        no-op) — and wait_for_capacity must still RETURN. Regression: a
        perpetual penalty floor would livelock here, because the multiplier
        only decays via report_success, which needs an admission first.
        """
        import llm.rate_limit as rl

        monkeypatch.setattr(rl, "ERROR_BASE_PENALTY_SECONDS", 0.05)
        limiter = RateLimiter([(1000, 1)])
        limiter.report_error(is_rate_limit=True)
        assert limiter.error_multiplier > 1.0

        start = time.monotonic()
        waited = limiter.wait_for_capacity()
        elapsed = time.monotonic() - start

        # (multiplier - 1.0) * base = 0.5 * 0.05 = 0.025 s minimum delay.
        assert elapsed >= 0.02
        assert elapsed < 5.0
        assert waited >= 0.0
        assert limiter.total_requests == 1

    def test_rate_limit_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A saturated window forces the third request to sleep.

        The fake sleep records the requested duration and aborts the wait loop,
        so we assert that a real wait was demanded without sleeping in real
        time. A no-op limiter that admitted the third request immediately would
        never call sleep, leaving ``sleeps`` empty and failing the test.
        """
        limiter = RateLimiter([(2, 1)])  # 2 requests per second

        # Fill the window; these two admit instantly.
        limiter.wait_for_capacity()
        limiter.wait_for_capacity()

        sleeps: list[float] = []
        _record_first_sleep(monkeypatch, sleeps)

        # Third request is over the limit: it must wait rather than admit.
        with pytest.raises(_SleepInterrupt):
            limiter.wait_for_capacity()

        # A wait was actually requested (empty list => no enforcement).
        assert sleeps
        # The window wants a ~1 s wait; the request is capped at MAX_SLEEP_TIME.
        assert sleeps[0] == pytest.approx(MAX_SLEEP_TIME)
        # The blocked request was NOT admitted.
        assert limiter.total_requests == 2

    def test_multiple_limits_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tighter of several windows triggers the block.

        The per-minute window (2/60) saturates after two requests while the
        per-second window (100/1) stays wide open; the third request must still
        wait, proving every window is checked, not merely the first.
        """
        limiter = RateLimiter([(100, 1), (2, 60)])  # 100/sec, 2/min

        limiter.wait_for_capacity()
        limiter.wait_for_capacity()

        sleeps: list[float] = []
        _record_first_sleep(monkeypatch, sleeps)

        # Third request is under the per-second cap but over the per-minute cap.
        with pytest.raises(_SleepInterrupt):
            limiter.wait_for_capacity()

        assert sleeps
        # The per-minute window wants ~60 s; the request is capped at the max.
        assert sleeps[0] == pytest.approx(MAX_SLEEP_TIME)
        assert limiter.total_requests == 2


class TestReportSuccess:
    """Tests for report_success method."""

    def test_resets_consecutive_errors(self) -> None:
        """Success resets consecutive error count."""
        limiter = RateLimiter([(10, 1)])
        limiter.consecutive_errors = 5

        limiter.report_success()

        assert limiter.consecutive_errors == 0

    def test_decreases_error_multiplier(self) -> None:
        """Success gradually decreases error multiplier."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 3.0

        limiter.report_success()

        assert limiter.error_multiplier < 3.0
        assert limiter.error_multiplier >= 1.0

    def test_multiplier_does_not_go_below_one(self) -> None:
        """Error multiplier never goes below 1.0."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 1.0

        limiter.report_success()

        assert limiter.error_multiplier == 1.0


class TestReportError:
    """Tests for report_error method."""

    def test_increments_consecutive_errors(self) -> None:
        """Error increments consecutive error count."""
        limiter = RateLimiter([(10, 1)])

        limiter.report_error()

        assert limiter.consecutive_errors == 1

    def test_rate_limit_error_increases_multiplier(self) -> None:
        """Rate limit error increases multiplier significantly."""
        limiter = RateLimiter([(10, 1)])
        original = limiter.error_multiplier

        limiter.report_error(is_rate_limit=True)

        assert limiter.error_multiplier > original

    def test_other_error_increases_multiplier_after_threshold(self) -> None:
        """Non-rate-limit errors increase multiplier after threshold."""
        limiter = RateLimiter([(10, 1)])
        limiter.consecutive_errors = 3  # Above threshold
        original = limiter.error_multiplier

        limiter.report_error(is_rate_limit=False)

        assert limiter.error_multiplier > original

    def test_multiplier_capped_at_max(self) -> None:
        """Error multiplier is capped at max value."""
        limiter = RateLimiter([(10, 1)])

        # Report many errors
        for _ in range(50):
            limiter.report_error(is_rate_limit=True)

        assert limiter.error_multiplier <= limiter.max_error_multiplier


class TestGetStats:
    """Tests for get_stats method."""

    def test_returns_dict(self) -> None:
        """get_stats returns a dictionary."""
        limiter = RateLimiter([(10, 1)])

        stats = limiter.get_stats()

        assert isinstance(stats, dict)

    def test_contains_required_keys(self) -> None:
        """Stats dictionary contains all required keys."""
        limiter = RateLimiter([(10, 1)])

        stats = limiter.get_stats()

        assert "total_requests" in stats
        assert "total_wait_time" in stats
        assert "average_wait" in stats
        assert "current_rate" in stats
        assert "current_queue_lengths" in stats
        assert "error_multiplier" in stats

    def test_reflects_actual_state(self) -> None:
        """Stats reflect actual limiter state."""
        limiter = RateLimiter([(10, 1)])
        limiter.wait_for_capacity()
        limiter.wait_for_capacity()
        limiter.error_multiplier = 2.5

        stats = limiter.get_stats()

        assert stats["total_requests"] == 2
        assert stats["error_multiplier"] == 2.5

    def test_resets_rate_tracking(self) -> None:
        """Getting stats resets rate tracking counters."""
        limiter = RateLimiter([(10, 1)])
        limiter.wait_for_capacity()

        limiter.get_stats()

        assert limiter.request_count_since_last_update == 0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_requests(self) -> None:
        """Concurrent requests are handled safely."""
        limiter = RateLimiter([(1000, 1)])  # High limit to avoid blocking
        request_count = 100
        results = []

        def make_request() -> None:
            wait = limiter.wait_for_capacity()
            results.append(wait)

        threads = [threading.Thread(target=make_request) for _ in range(request_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert limiter.total_requests == request_count
        assert len(results) == request_count

    def test_concurrent_success_reports(self) -> None:
        """Concurrent success reports are handled safely."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 3.0
        report_count = 50

        def report() -> None:
            limiter.report_success()

        threads = [threading.Thread(target=report) for _ in range(report_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have decreased multiplier
        assert limiter.error_multiplier <= 3.0
        assert limiter.consecutive_errors == 0


class TestAdaptiveBackoff:
    """Tests for adaptive backoff behavior."""

    def test_error_multiplier_affects_wait(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A higher error multiplier lengthens the enforced wait.

        With the sleep cap lifted, the recorded sleep for a saturated window
        grows in proportion to ``error_multiplier``. An unchanged wait would
        prove the multiplier was ignored, so the growth assertion catches a
        limiter that fails to scale its backoff.
        """
        import llm.rate_limit as rl

        # Lift the MAX_SLEEP_TIME cap so the multiplier's effect stays visible
        # in the requested sleep (a saturated window otherwise saturates it).
        monkeypatch.setattr(rl, "MAX_SLEEP_TIME", 100.0)

        def saturated_wait(multiplier: float) -> float:
            limiter = RateLimiter([(2, 1)])
            limiter.wait_for_capacity()
            limiter.wait_for_capacity()
            limiter.error_multiplier = multiplier
            sleeps: list[float] = []
            _record_first_sleep(monkeypatch, sleeps)
            with pytest.raises(_SleepInterrupt):
                limiter.wait_for_capacity()
            assert sleeps
            return sleeps[0]

        baseline = saturated_wait(1.0)
        boosted = saturated_wait(3.0)

        # The elevated multiplier must lengthen the wait (roughly 3x here).
        assert boosted > baseline
        assert boosted >= 2 * baseline

    def test_recovery_after_success_streak(self) -> None:
        """Error multiplier recovers after streak of successes."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 3.0

        # Report many successes
        for _ in range(20):
            limiter.report_success()

        # Should have recovered toward 1.0
        assert limiter.error_multiplier < 3.0
