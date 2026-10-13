"""The parts of ``batch check --wait`` outside a check: final outcomes,
closing the batch clients and the wait line."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from autoexcerpter.batch.check import (
    ABANDONED,
    UNRESOLVED,
    WaitLedger,
    _submission_outcome,
)
from autoexcerpter.batch.clients import BatchClients
from autoexcerpter.batch.finalize import FAILED, FINALIZED_OUTCOME, PENDING
from autoexcerpter.batch.report import wait_line
from autoexcerpter.common.polling import WaitTick
from autoexcerpter.settings import Settings

KEY = (Path("out") / "autoexcerpter.sqlite", "sub-1")


@pytest.mark.parametrize(
    ("status", "seen", "expected"),
    [
        ("finalized", "failed", FINALIZED_OUTCOME),
        ("submitted", "failed", FAILED),
        ("failed", None, FAILED),
        ("failed", "abandoned", ABANDONED),
        ("unresolved", None, UNRESOLVED),
        ("submitted", None, PENDING),
        ("submitting", None, PENDING),
        ("cancelled", None, None),
    ],
)
def test_a_reported_failure_stays_unless_the_store_shows_recovery(
    status: str, seen: str | None, expected: str | None
) -> None:
    ledger = WaitLedger(submissions={KEY: None})
    if seen == "failed":
        ledger.failed.add(KEY)
    elif seen == "abandoned":
        ledger.abandoned.add(KEY)

    assert _submission_outcome(KEY, status, ledger) == expected


class Closable:
    def __init__(self, fail: bool = False) -> None:
        self.closed = 0
        self.fail = fail

    def close(self) -> None:
        self.closed += 1
        if self.fail:
            raise OSError("socket already closed")


def test_closing_the_clients_closes_each_once_and_never_raises() -> None:
    clients = BatchClients(Settings(), "batch")
    good, bad = Closable(), Closable(fail=True)
    clients._clients = {("openai", "A"): good, ("google", "B"): bad, ("x", "C"): 1}

    clients.close()
    clients.close()

    assert (good.closed, bad.closed) == (1, 1)


def test_the_wait_line_names_the_next_check_and_never_says_until() -> None:
    tick = WaitTick(passes=3, pending=2, elapsed=1800, next_in=900, timeout=50 * 3600)

    line = wait_line(tick, now=datetime(2026, 10, 13, 14, 30))

    assert line == (
        "Waiting for batches: 2 pending; next check at 14:45 (check 3; 0.50 of 50 h)."
    )
    assert "until" not in line.lower()
