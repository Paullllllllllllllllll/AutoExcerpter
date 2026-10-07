"""Page counter and ETA estimate for one item."""

from __future__ import annotations

import threading
import time

from autoexcerpter.constants import (
    ETA_BLEND_WEIGHT_OVERALL,
    ETA_BLEND_WEIGHT_RECENT,
    MIN_SAMPLES_FOR_ETA,
    RECENT_SAMPLES_FOR_ETA,
)


class PageProgress:
    """Thread-safe count of the pages processed in one item's page pool.

    *total* is the number of pending pages of this run (not the source page
    count), so progress and ETA are not skewed by pages completed on a prior
    run; *already_complete* is that prior-run count, reported for context.
    *workers* is the pool size: a per-page API time measures one worker, so
    the recent rate is scaled by it.
    """

    def __init__(
        self,
        total: int,
        *,
        already_complete: int = 0,
        workers: int = 1,
        started_at: float | None = None,
    ) -> None:
        self.total = total
        self.already_complete = already_complete
        self.workers = workers
        self.started_at = started_at if started_at is not None else time.time()
        self.processed = 0
        # API times of successful transcriptions, in completion order.
        self.transcription_times: list[float] = []
        self._lock = threading.Lock()

    def advance(self) -> int:
        """Count one finished page and return the new count."""
        with self._lock:
            self.processed += 1
            return self.processed

    def record_time(self, seconds: float) -> None:
        """Record the API time of one successful transcription."""
        with self._lock:
            self.transcription_times.append(seconds)

    def eta(self, processed_count: int) -> str:
        """Return the formatted estimate of the time left for this pool."""
        if processed_count <= MIN_SAMPLES_FOR_ETA or not self.started_at:
            return "ETA: N/A"

        elapsed_total = time.time() - self.started_at
        if elapsed_total <= 0:
            return "ETA: N/A"
        items_per_sec_overall = processed_count / elapsed_total
        if items_per_sec_overall <= 0:
            return "ETA: N/A"

        blended_rate = self._blended_rate(items_per_sec_overall)
        if blended_rate <= 0:
            return "ETA: N/A"

        eta_seconds = (self.total - processed_count) / blended_rate
        # divmod rather than time.gmtime, which wraps ETAs above 24 hours.
        hours, rem = divmod(int(eta_seconds), 3600)
        minutes, seconds = divmod(rem, 60)
        return f"ETA: {hours:02d}:{minutes:02d}:{seconds:02d}"

    def _blended_rate(self, overall_rate: float) -> float:
        """Blend the overall rate with the rate of the recent samples."""
        with self._lock:
            recent_samples = self.transcription_times[-RECENT_SAMPLES_FOR_ETA:]
        if not recent_samples:
            return overall_rate
        recent_avg_time = sum(recent_samples) / len(recent_samples)
        recent_rate = (
            self.workers / recent_avg_time if recent_avg_time > 0 else overall_rate
        )
        return (
            ETA_BLEND_WEIGHT_OVERALL * overall_rate
            + ETA_BLEND_WEIGHT_RECENT * recent_rate
        )


__all__ = ["PageProgress"]
