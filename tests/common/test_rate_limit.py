"""Tests for common/rate_limit.py: windows, adaptive backoff, registry."""

from __future__ import annotations

import asyncio
import threading

import pytest

from autoexcerpter.common import rate_limit
from autoexcerpter.common.rate_limit import (
    DEFAULT_RATE_LIMITS,
    RateLimiter,
    RateLimiterRegistry,
    RateLimitWaitAborted,
    parse_rate_limits,
)


class _Clock:
    """A manual monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> list[float]:
    """Record the limiter's sleeps and advance the manual clock by each."""
    record: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        record.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return record


class TestAcquire:
    def test_first_request_has_no_wait(
        self, clock: _Clock, sleeps: list[float]
    ) -> None:
        limiter = RateLimiter([(10, 1)], clock=clock)
        assert asyncio.run(limiter.acquire()) == 0.0
        assert len(limiter.request_timestamps[0]) == 1
        assert sleeps == []

    def test_saturated_window_waits_until_it_slides(
        self, clock: _Clock, sleeps: list[float]
    ) -> None:
        limiter = RateLimiter([(2, 1)], clock=clock)

        async def three() -> float:
            await limiter.acquire()
            await limiter.acquire()
            return await limiter.acquire()

        waited = asyncio.run(three())

        assert sleeps
        assert sleeps[0] == pytest.approx(rate_limit.MAX_SLEEP_TIME)
        assert waited >= 1.0
        assert limiter.total_requests == 3

    def test_every_window_is_checked(self, clock: _Clock) -> None:
        limiter = RateLimiter([(100, 1), (2, 60)], clock=clock)
        assert limiter.try_acquire() == 0.0
        assert limiter.try_acquire() == 0.0
        assert limiter.try_acquire() == pytest.approx(60.0)
        assert len(limiter.request_timestamps[1]) == 2

    def test_window_slides(self, clock: _Clock) -> None:
        limiter = RateLimiter([(1, 10)], clock=clock)
        assert limiter.try_acquire() == 0.0
        clock.now += 5
        assert limiter.try_acquire() == pytest.approx(5.0)
        clock.now += 5.01
        assert limiter.try_acquire() == 0.0

    def test_error_penalty_delays_but_still_admits(
        self, clock: _Clock, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rate_limit, "ERROR_BASE_PENALTY_SECONDS", 0.05)
        limiter = RateLimiter([(1000, 1)], clock=clock)
        limiter.report_error(is_rate_limit=True)
        assert limiter.error_multiplier > 1.0

        waited = asyncio.run(limiter.acquire())

        # (1.5 - 1.0) * 0.05 = 0.025 s of penalty, then admission.
        assert waited >= 0.025
        assert len(limiter.request_timestamps[0]) == 1

    def test_error_multiplier_lengthens_the_wait(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def saturated_wait(multiplier: float) -> float:
            limiter = RateLimiter([(2, 1)], clock=clock)
            limiter.try_acquire()
            limiter.try_acquire()
            limiter.error_multiplier = multiplier
            return limiter.try_acquire()

        baseline = saturated_wait(1.0)
        boosted = saturated_wait(3.0)
        assert boosted >= 2 * baseline > 0

    def test_abort_raises_without_recording(
        self, clock: _Clock, sleeps: list[float]
    ) -> None:
        limiter = RateLimiter([(1, 60)], clock=clock)
        limiter.try_acquire()

        with pytest.raises(RateLimitWaitAborted):
            asyncio.run(limiter.acquire(should_abort=lambda: True))

        assert len(limiter.request_timestamps[0]) == 1
        assert sleeps == []

    def test_abort_checked_between_slices(
        self, clock: _Clock, sleeps: list[float]
    ) -> None:
        limiter = RateLimiter([(1, 60)], clock=clock)
        limiter.try_acquire()
        checks = iter([False, False, True])

        with pytest.raises(RateLimitWaitAborted):
            asyncio.run(limiter.acquire(should_abort=lambda: next(checks)))

        assert len(sleeps) == 2

    def test_cancellation_records_nothing(self) -> None:
        limiter = RateLimiter([(1, 60)])
        limiter.try_acquire()

        async def main() -> None:
            task = asyncio.create_task(limiter.acquire())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(main())
        assert len(limiter.request_timestamps[0]) == 1

    def test_concurrent_tasks(self) -> None:
        limiter = RateLimiter([(1000, 1)])

        async def many() -> list[float]:
            return list(await asyncio.gather(*(limiter.acquire() for _ in range(100))))

        assert len(asyncio.run(many())) == 100
        assert len(limiter.request_timestamps[0]) == 100

    def test_concurrent_threads(self) -> None:
        limiter = RateLimiter([(1000, 1)])
        threads = [threading.Thread(target=limiter.try_acquire) for _ in range(100)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert limiter.total_requests == 100


class TestReports:
    def test_success_resets_and_relaxes(self) -> None:
        limiter = RateLimiter([(10, 1)])
        limiter.consecutive_errors = 5
        limiter.error_multiplier = 3.0
        limiter.report_success()
        assert limiter.consecutive_errors == 0
        assert 1.0 <= limiter.error_multiplier < 3.0

    def test_multiplier_floor_is_one(self) -> None:
        limiter = RateLimiter([(10, 1)])
        for _ in range(20):
            limiter.report_success()
        assert limiter.error_multiplier == 1.0

    def test_rate_limit_error_raises_multiplier_at_once(self) -> None:
        limiter = RateLimiter([(10, 1)])
        limiter.report_error(is_rate_limit=True)
        assert limiter.consecutive_errors == 1
        assert limiter.error_multiplier == pytest.approx(1.5)

    def test_other_errors_raise_after_threshold(self) -> None:
        limiter = RateLimiter([(10, 1)])
        for _ in range(rate_limit.CONSECUTIVE_ERRORS_THRESHOLD):
            limiter.report_error()
        assert limiter.error_multiplier == 1.0
        limiter.report_error()
        assert limiter.error_multiplier == pytest.approx(1.2)

    def test_multiplier_capped(self) -> None:
        limiter = RateLimiter([(10, 1)])
        for _ in range(50):
            limiter.report_error(is_rate_limit=True)
        assert limiter.error_multiplier == limiter.max_error_multiplier

    def test_stats(self, clock: _Clock) -> None:
        limiter = RateLimiter([(10, 1)], clock=clock)
        limiter.try_acquire()
        stats = limiter.stats()
        assert stats["total_requests"] == 1
        assert stats["queue_lengths"] == [1]
        assert stats["error_multiplier"] == 1.0


class TestParseRateLimits:
    def test_valid(self) -> None:
        assert parse_rate_limits([[10, 1], (600, 60)]) == [(10, 1), (600, 60)]

    def test_skips_bad_entries(self) -> None:
        raw = [[10, 1], [1, 2, 3], ["x", 1], [0, 1], [5, 0], "nope"]
        assert parse_rate_limits(raw) == [(10, 1)]

    @pytest.mark.parametrize("raw", [None, "x", {}, [], [[0, 0]]])
    def test_default_when_unusable(self, raw: object) -> None:
        assert parse_rate_limits(raw) == list(DEFAULT_RATE_LIMITS)


class TestRegistry:
    def test_one_limiter_per_provider(self) -> None:
        registry = RateLimiterRegistry([(10, 1)])
        first = registry.get("openai")
        assert registry.get("openai") is first
        assert registry.get("anthropic") is not first
        assert registry.get(None) is registry.get("default")
        assert "openai" in registry

    def test_per_provider_limits(self) -> None:
        registry = RateLimiterRegistry([(10, 1)], per_provider={"google": [(5, 60)]})
        assert registry.get("google").limits == [(5, 60)]
        assert registry.get("openai").limits == [(10, 1)]

    def test_registries_are_independent(self) -> None:
        assert RateLimiterRegistry().get("openai") is not RateLimiterRegistry().get(
            "openai"
        )
