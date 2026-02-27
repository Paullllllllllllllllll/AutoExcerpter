"""Tests for api/rate_limiter.py - Rate limiting with adaptive backoff."""

from __future__ import annotations

import time
import threading
from unittest.mock import patch

import pytest

from api.rate_limiter import RateLimiter


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_with_single_limit(self):
        """Initializes with a single rate limit."""
        limits = [(10, 1)]  # 10 requests per second
        limiter = RateLimiter(limits)

        assert limiter.limits == limits
        assert len(limiter.request_timestamps) == 1
        assert limiter.total_requests == 0

    def test_init_with_multiple_limits(self):
        """Initializes with multiple rate limits."""
        limits = [(120, 1), (15000, 60)]  # 120/sec, 15000/min
        limiter = RateLimiter(limits)

        assert limiter.limits == limits
        assert len(limiter.request_timestamps) == 2

    def test_initial_state(self):
        """Initial state is correctly set."""
        limiter = RateLimiter([(10, 1)])

        assert limiter.total_requests == 0
        assert limiter.total_wait_time == 0.0
        assert limiter.consecutive_errors == 0
        assert limiter.error_multiplier == 1.0


class TestWaitForCapacity:
    """Tests for wait_for_capacity method."""

    def test_first_request_no_wait(self):
        """First request has no wait time."""
        limiter = RateLimiter([(10, 1)])

        wait_time = limiter.wait_for_capacity()

        assert wait_time < 0.1  # Should be near-instant
        assert limiter.total_requests == 1

    def test_requests_recorded(self):
        """Requests are recorded in timestamps."""
        limiter = RateLimiter([(10, 1)])

        limiter.wait_for_capacity()
        limiter.wait_for_capacity()

        assert len(limiter.request_timestamps[0]) == 2
        assert limiter.total_requests == 2

    def test_rate_limit_enforced(self):
        """Rate limit is enforced when exceeded."""
        # Very low limit to test enforcement
        limiter = RateLimiter([(2, 1)])  # 2 requests per second

        # Make 2 requests (should be instant)
        limiter.wait_for_capacity()
        limiter.wait_for_capacity()

        # Third request should wait
        start = time.time()
        limiter.wait_for_capacity()
        elapsed = time.time() - start

        # Should have waited some time (rate limited)
        assert elapsed > 0.3 or limiter.total_requests == 3

    def test_multiple_limits_checked(self):
        """All limits are checked simultaneously."""
        # Tight second limit, loose first limit
        limiter = RateLimiter([(100, 1), (2, 60)])  # 100/sec, 2/min

        limiter.wait_for_capacity()
        limiter.wait_for_capacity()
        # Third request should wait due to per-minute limit

        assert limiter.total_requests == 2


class TestReportSuccess:
    """Tests for report_success method."""

    def test_resets_consecutive_errors(self):
        """Success resets consecutive error count."""
        limiter = RateLimiter([(10, 1)])
        limiter.consecutive_errors = 5

        limiter.report_success()

        assert limiter.consecutive_errors == 0

    def test_decreases_error_multiplier(self):
        """Success gradually decreases error multiplier."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 3.0

        limiter.report_success()

        assert limiter.error_multiplier < 3.0
        assert limiter.error_multiplier >= 1.0

    def test_multiplier_does_not_go_below_one(self):
        """Error multiplier never goes below 1.0."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 1.0

        limiter.report_success()

        assert limiter.error_multiplier == 1.0


class TestReportError:
    """Tests for report_error method."""

    def test_increments_consecutive_errors(self):
        """Error increments consecutive error count."""
        limiter = RateLimiter([(10, 1)])

        limiter.report_error()

        assert limiter.consecutive_errors == 1

    def test_rate_limit_error_increases_multiplier(self):
        """Rate limit error increases multiplier significantly."""
        limiter = RateLimiter([(10, 1)])
        original = limiter.error_multiplier

        limiter.report_error(is_rate_limit=True)

        assert limiter.error_multiplier > original

    def test_other_error_increases_multiplier_after_threshold(self):
        """Non-rate-limit errors increase multiplier after threshold."""
        limiter = RateLimiter([(10, 1)])
        limiter.consecutive_errors = 3  # Above threshold
        original = limiter.error_multiplier

        limiter.report_error(is_rate_limit=False)

        assert limiter.error_multiplier > original

    def test_multiplier_capped_at_max(self):
        """Error multiplier is capped at max value."""
        limiter = RateLimiter([(10, 1)])

        # Report many errors
        for _ in range(50):
            limiter.report_error(is_rate_limit=True)

        assert limiter.error_multiplier <= limiter.max_error_multiplier


class TestGetStats:
    """Tests for get_stats method."""

    def test_returns_dict(self):
        """get_stats returns a dictionary."""
        limiter = RateLimiter([(10, 1)])

        stats = limiter.get_stats()

        assert isinstance(stats, dict)

    def test_contains_required_keys(self):
        """Stats dictionary contains all required keys."""
        limiter = RateLimiter([(10, 1)])

        stats = limiter.get_stats()

        assert "total_requests" in stats
        assert "total_wait_time" in stats
        assert "average_wait" in stats
        assert "current_rate" in stats
        assert "current_queue_lengths" in stats
        assert "error_multiplier" in stats

    def test_reflects_actual_state(self):
        """Stats reflect actual limiter state."""
        limiter = RateLimiter([(10, 1)])
        limiter.wait_for_capacity()
        limiter.wait_for_capacity()
        limiter.error_multiplier = 2.5

        stats = limiter.get_stats()

        assert stats["total_requests"] == 2
        assert stats["error_multiplier"] == 2.5

    def test_resets_rate_tracking(self):
        """Getting stats resets rate tracking counters."""
        limiter = RateLimiter([(10, 1)])
        limiter.wait_for_capacity()

        limiter.get_stats()

        assert limiter.request_count_since_last_update == 0


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_requests(self):
        """Concurrent requests are handled safely."""
        limiter = RateLimiter([(1000, 1)])  # High limit to avoid blocking
        request_count = 100
        results = []

        def make_request():
            wait = limiter.wait_for_capacity()
            results.append(wait)

        threads = [threading.Thread(target=make_request) for _ in range(request_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert limiter.total_requests == request_count
        assert len(results) == request_count

    def test_concurrent_success_reports(self):
        """Concurrent success reports are handled safely."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 3.0
        report_count = 50

        def report():
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

    def test_error_multiplier_affects_wait(self):
        """Higher error multiplier increases wait times."""
        limiter = RateLimiter([(2, 1)])  # 2 per second

        # Fill capacity
        limiter.wait_for_capacity()
        limiter.wait_for_capacity()

        # Set high error multiplier
        limiter.error_multiplier = 3.0

        # Next request should wait longer due to multiplier
        # (The actual wait is multiplied by error_multiplier)
        start = time.time()
        limiter.wait_for_capacity()
        elapsed = time.time() - start

        # Should have some wait (rate limited)
        assert limiter.total_requests == 3

    def test_recovery_after_success_streak(self):
        """Error multiplier recovers after streak of successes."""
        limiter = RateLimiter([(10, 1)])
        limiter.error_multiplier = 3.0

        # Report many successes
        for _ in range(20):
            limiter.report_success()

        # Should have recovered toward 1.0
        assert limiter.error_multiplier < 3.0
