import threading
import time
from collections import deque
from typing import Any, Dict, List, Tuple

class RateLimiter:
    """Manages API call rates with adaptive backoff."""

    def __init__(self, limits: List[Tuple[int, int]]):
        self.limits = limits
        self.request_timestamps = [deque(maxlen=limit[0]) for limit in limits]
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.last_stats_update = time.time()
        self.request_count_since_last_update = 0
        self.consecutive_errors = 0
        self.error_multiplier = 1.0
        self.max_error_multiplier = 5.0

    def wait_for_capacity(self) -> float:
        wait_start = time.time()
        while True:
            wait_time = 0.0
            with self.lock:
                now = time.time()
                for i, (max_requests, seconds) in enumerate(self.limits):
                    cutoff = now - seconds
                    while (
                            self.request_timestamps[i]
                            and self.request_timestamps[i][0] < cutoff
                    ):
                        self.request_timestamps[i].popleft()
                    if len(self.request_timestamps[i]) >= max_requests:
                        oldest_request_time = self.request_timestamps[i][0]
                        required_wait = oldest_request_time + seconds - now
                        wait_time = max(wait_time, required_wait)
                wait_time *= self.error_multiplier
                if wait_time <= 0:
                    for timestamps in self.request_timestamps:
                        timestamps.append(now)
                    self.total_requests += 1
                    self.request_count_since_last_update += 1
                    total_wait = time.time() - wait_start
                    self.total_wait_time += total_wait
                    return total_wait
            sleep_time = min(wait_time + 0.05, 0.50)  # Add small jitter
            time.sleep(sleep_time)

    def report_success(self):
        with self.lock:
            self.consecutive_errors = 0
            if self.error_multiplier > 1.0:
                self.error_multiplier = max(1.0, self.error_multiplier * 0.9)

    def report_error(self, is_rate_limit: bool = False):
        with self.lock:
            self.consecutive_errors += 1
            if is_rate_limit:
                self.error_multiplier = min(
                    self.max_error_multiplier, self.error_multiplier * 1.5
                )
            elif self.consecutive_errors > 2:  # Increase more slowly for other errors
                self.error_multiplier = min(
                    self.max_error_multiplier, self.error_multiplier * 1.2
                )

    def get_stats(self) -> Dict[str, Any]:
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
                "current_queue_lengths": [
                    len(ts) for ts in self.request_timestamps
                ],
                "error_multiplier": round(self.error_multiplier, 2),
            }
            self.last_stats_update = now
            self.request_count_since_last_update = 0
            return stats
