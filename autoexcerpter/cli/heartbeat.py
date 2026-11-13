"""The launcher's progress file, written beside the stderr reports.

:func:`with_heartbeat` puts a heartbeat sink beside the run's stderr
reporter when ``CHRONO_HEARTBEAT_FILE`` names a file: each item's pages
processed, the item position and, during the last item, an ETA.
:class:`BatchHeartbeat` writes the submissions a ``batch check --wait`` has
seen finish, once before each sleep.
"""

from __future__ import annotations

from autoexcerpter.batch.check import CheckResult
from autoexcerpter.batch.finalize import PENDING
from autoexcerpter.common.heartbeat import Heartbeat, ItemProgress
from autoexcerpter.common.polling import WaitTick
from autoexcerpter.events import (
    ItemFinished,
    ItemStarted,
    PageDone,
    RunEvents,
    Waiting,
)

__all__ = ["BatchHeartbeat", "HeartbeatEvents", "with_heartbeat"]

UNIT = "pages"


class HeartbeatEvents:
    """Pass every event to *inner*, then record it in *progress*.

    Each open item keeps its own page counts, keyed by its position in the
    run; a page event without a position belongs to the item opened last.
    While several items are open, the file shows the pages of all of them
    together and the position of the one opened last; the item end is
    written once no item is open.
    """

    def __init__(self, inner: RunEvents, progress: ItemProgress) -> None:
        self.inner = inner
        self.progress = progress
        self._open: dict[int, tuple[int, int]] = {}
        self._last = 0

    def _units(self) -> None:
        done = sum(counts[0] for counts in self._open.values())
        total = sum(counts[1] for counts in self._open.values())
        self.progress.units(done, total)

    def item_started(self, event: ItemStarted) -> None:
        """Forward the event; start the item's units at zero."""
        self.inner.item_started(event)
        self._open[event.index] = (0, 0)
        self._last = event.index
        self.progress.item_started(event.index, event.total)
        if len(self._open) > 1:
            self._units()

    def page_done(self, event: PageDone) -> None:
        """Forward the event; record the item's pages done."""
        self.inner.page_done(event)
        self._open[event.index or self._last] = (event.done, event.total)
        self._units()

    def waiting(self, event: Waiting) -> None:
        """Forward the event."""
        self.inner.waiting(event)

    def warning(self, message: str) -> None:
        """Forward the warning."""
        self.inner.warning(message)

    def item_finished(self, event: ItemFinished) -> None:
        """Forward the event; write the finished item, or the ones still open."""
        self.inner.item_finished(event)
        self._open.pop(event.index, None)
        if self._open:
            self._units()
        else:
            self.progress.item_finished()


def with_heartbeat(reporter: RunEvents) -> tuple[RunEvents, ItemProgress | None]:
    """Return the events to run with and the progress to finish, if any.

    Without ``CHRONO_HEARTBEAT_FILE`` the reporter alone receives the events.
    """
    heartbeat = Heartbeat.from_env()
    if heartbeat is None:
        return reporter, None
    progress = ItemProgress(heartbeat, UNIT)
    return HeartbeatEvents(reporter, progress), progress


class BatchHeartbeat:
    """The submissions of a wait that finished, of those its checks saw."""

    def __init__(self, heartbeat: Heartbeat) -> None:
        self.heartbeat = heartbeat
        self._outcomes: dict[tuple[object, str], str] = {}

    @classmethod
    def from_env(cls) -> BatchHeartbeat | None:
        """Return the writer for ``CHRONO_HEARTBEAT_FILE``; None when unset."""
        heartbeat = Heartbeat.from_env()
        return None if heartbeat is None else cls(heartbeat)

    def on_pass(self, result: CheckResult) -> None:
        """Keep each submission's latest outcome."""
        try:
            for check in result.submissions:
                self._outcomes[(check.database, check.submission_id)] = check.outcome
        except Exception as exc:  # noqa: BLE001 - a heartbeat never fails the run
            self.heartbeat.failed(exc)

    def on_wait(self, tick: WaitTick) -> None:
        """Write the finished submissions before the wait sleeps."""
        try:
            done = sum(outcome != PENDING for outcome in self._outcomes.values())
            self.heartbeat.update(done, len(self._outcomes), "batches", force=True)
        except Exception as exc:  # noqa: BLE001 - a heartbeat never fails the run
            self.heartbeat.failed(exc)
