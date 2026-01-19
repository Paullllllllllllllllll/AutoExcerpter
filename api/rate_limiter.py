"""Rate limiter for API calls with adaptive backoff.

This module implements an intelligent rate limiter with adaptive backoff for API calls.
It tracks requests across multiple time windows and automatically adjusts wait times
based on error patterns.

Key Features:
1. **Multi-Window Rate Limiting**: Enforces multiple concurrent rate limits (e.g., 
   per-second, per-minute, per-hour) to comply with API quotas

2. **Adaptive Backoff**: Dynamically adjusts wait times based on consecutive errors:
   - Increases multiplier on rate limit/server errors
   - Gradually decreases on successful requests
   - Prevents thundering herd with jitter

3. **Thread-Safe**: Uses locks to ensure thread-safe operation in concurrent environments

4. **Statistics Tracking**: Monitors total requests, wait times, current rates, and
   error patterns for performance analysis

Usage:
    >>> limits = [(120, 1), (15000, 60)]  # 120/sec, 15000/min
    >>> limiter = RateLimiter(limits)
    >>> 
    >>> limiter.wait_for_capacity()  # Blocks until capacity available
    >>> # ... make API call ...
    >>> limiter.report_success()  # or limiter.report_error(is_rate_limit=True)
    >>> 
    >>> stats = limiter.get_stats()  # Get performance metrics

The adaptive error multiplier starts at 1.0 and can increase up to 5.0 based on
error frequency, providing intelligent backoff during API issues.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from modules.constants import (
    MIN_SLEEP_TIME,
    MAX_SLEEP_TIME,
    ERROR_MULTIPLIER_DECREASE_RATE,
    ERROR_MULTIPLIER_INCREASE_RATE_LIMIT,
    ERROR_MULTIPLIER_INCREASE_OTHER,
    CONSECUTIVE_ERRORS_THRESHOLD,
)


# ============================================================================
# Rate Limiter Class
# ============================================================================
class RateLimiter:
    """
    Manages API call rates with adaptive backoff.

    This rate limiter tracks requests across multiple time windows and
    automatically adjusts wait times based on error patterns.

    Attributes:
        limits: List of (max_requests, time_window_seconds) tuples
        request_timestamps: Deques tracking recent request times per limit
        total_requests: Total number of requests made
        total_wait_time: Cumulative wait time across all requests
        consecutive_errors: Count of consecutive errors
        error_multiplier: Dynamic multiplier for wait times based on errors
    """

    def __init__(self, limits: list[tuple[int, int]]) -> None:
        """
        Initialize the rate limiter.

        Args:
            limits: List of (max_requests, time_window_seconds) tuples defining rate limits
        """
        self.limits = limits
        self.request_timestamps: list[deque[float]] = [deque(maxlen=limit[0]) for limit in limits]
        self.lock = threading.Lock()

        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.last_stats_update = time.time()
        self.request_count_since_last_update = 0

        # Adaptive backoff
        self.consecutive_errors = 0
        self.error_multiplier = 1.0
        self.max_error_multiplier = 5.0

    def wait_for_capacity(self) -> float:
        """
        Wait until there's capacity to make a request under all rate limits.

        This method blocks until all rate limit windows have capacity, then
        records the request timestamp.

        Returns:
            Total time waited in seconds
        """
        wait_start = time.time()

        while True:
            wait_time = 0.0

            with self.lock:
                now = time.time()

                # Check all rate limit windows
                for i, (max_requests, seconds) in enumerate(self.limits):
                    cutoff = now - seconds

                    # Remove timestamps outside the window
                    while (
                        self.request_timestamps[i]
                        and self.request_timestamps[i][0] < cutoff
                    ):
                        self.request_timestamps[i].popleft()

                    # Check if we've hit the limit
                    if len(self.request_timestamps[i]) >= max_requests:
                        oldest_request_time = self.request_timestamps[i][0]
                        required_wait = oldest_request_time + seconds - now
                        wait_time = max(wait_time, required_wait)

                # Apply error multiplier to wait time
                wait_time *= self.error_multiplier

                # If no wait needed, record the request and return
                if wait_time <= 0:
                    for timestamps in self.request_timestamps:
                        timestamps.append(now)
                    self.total_requests += 1
                    self.request_count_since_last_update += 1
                    total_wait = time.time() - wait_start
                    self.total_wait_time += total_wait
                    return total_wait

            # Sleep with jitter to avoid thundering herd
            sleep_time = min(wait_time + MIN_SLEEP_TIME, MAX_SLEEP_TIME)
            time.sleep(sleep_time)

    def report_success(self) -> None:
        """
        Report a successful API request.

        Resets consecutive error counter and gradually decreases error multiplier.
        """
        with self.lock:
            self.consecutive_errors = 0
            if self.error_multiplier > 1.0:
                self.error_multiplier = max(1.0, self.error_multiplier * ERROR_MULTIPLIER_DECREASE_RATE)

    def report_error(self, is_rate_limit: bool = False) -> None:
        """
        Report a failed API request.

        Increases error multiplier based on error type to implement adaptive backoff.

        Args:
            is_rate_limit: Whether the error was a rate limit or server error
        """
        with self.lock:
            self.consecutive_errors += 1

            if is_rate_limit:
                self.error_multiplier = min(
                    self.max_error_multiplier,
                    self.error_multiplier * ERROR_MULTIPLIER_INCREASE_RATE_LIMIT,
                )
            elif self.consecutive_errors > CONSECUTIVE_ERRORS_THRESHOLD:
                self.error_multiplier = min(
                    self.max_error_multiplier,
                    self.error_multiplier * ERROR_MULTIPLIER_INCREASE_OTHER,
                )

    def get_stats(self) -> dict[str, Any]:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary containing statistics about requests, wait times, and current state
        """
        with self.lock:
            now = time.time()
            time_since_last_update = now - self.last_stats_update
            requests_per_second = self.request_count_since_last_update / max(
                1, time_since_last_update
            )

            stats = {
                "total_requests": self.total_requests,
                "total_wait_time": round(self.total_wait_time, 2),
                "average_wait": round(
                    self.total_wait_time / max(1, self.total_requests), 4
                ),
                "current_rate": round(requests_per_second, 2),
                "current_queue_lengths": [len(ts) for ts in self.request_timestamps],
                "error_multiplier": round(self.error_multiplier, 2),
            }

            # Reset stats tracking
            self.last_stats_update = now
            self.request_count_since_last_update = 0

            return stats


# ============================================================================
# Public API
# ============================================================================
__all__ = ["RateLimiter"]
