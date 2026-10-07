"""Run events: how the core reports progress to a front end.

The core never prints. It calls a :class:`RunEvents` implementation instead:
the CLI writes the events to stderr, a test records them, and
:class:`NullEvents` drops them. Events are called on the event-loop thread.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "ItemFinished",
    "ItemStarted",
    "NullEvents",
    "PageDone",
    "RunEvents",
    "Waiting",
]


@dataclass(frozen=True)
class ItemStarted:
    """An item is about to be processed; *index* counts from 1 to *total*."""

    name: str
    kind: str
    index: int
    total: int


@dataclass(frozen=True)
class PageDone:
    """One pending page of an item finished.

    *page* is the 0-based page index and *status* is ``ok``, ``failed`` or
    ``deferred`` (left for a later run). *done* of *total* counts the pages
    pending in this run; *already_complete* pages came from an earlier run.
    """

    item: str
    page: int
    status: str
    done: int
    total: int
    already_complete: int = 0


@dataclass(frozen=True)
class Waiting:
    """The run waits; *cancel*, when given, ends the wait early."""

    reason: str
    seconds: float | None = None
    cancel: Callable[[], None] | None = None


@dataclass(frozen=True)
class ItemFinished:
    """An item finished; *report* holds its page statistics, if any."""

    name: str
    index: int
    total: int
    success: bool
    outputs: tuple[str, ...] = ()
    report: Mapping[str, Any] | None = None


class RunEvents(Protocol):
    """The events a run reports."""

    def item_started(self, event: ItemStarted) -> None: ...

    def page_done(self, event: PageDone) -> None: ...

    def waiting(self, event: Waiting) -> None: ...

    def warning(self, message: str) -> None: ...

    def item_finished(self, event: ItemFinished) -> None: ...


class NullEvents:
    """Drop every event."""

    def item_started(self, event: ItemStarted) -> None:
        """Ignore the event."""

    def page_done(self, event: PageDone) -> None:
        """Ignore the event."""

    def waiting(self, event: Waiting) -> None:
        """Ignore the event."""

    def warning(self, message: str) -> None:
        """Ignore the warning."""

    def item_finished(self, event: ItemFinished) -> None:
        """Ignore the event."""
