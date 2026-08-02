"""Round-5 maintenance regressions for the LLM client base and token tracker.

Three fixes, all deterministic and offline:

1. ``LLMClientBase._invoke_with_retry``: an abort that arrives DURING the
   rate-limiter wait must prevent the API call. ``RateLimiter.wait_for_capacity``
   returns early on abort with the contract that the caller must not fire the
   call, so the abort event is re-checked after the wait (and outside the try,
   so the raise is never classified as a retryable API error).
2. ``DailyTokenTracker.sync_ledger_now``: a day rollover landing between the
   under-lock snapshot and the ledger call, or while the ledger call is in
   flight, must abandon the sync instead of leaking old-day deltas into the
   fresh day and re-latching stale state.
3. ``DailyTokenTracker.add_tokens``: with the shared budget enabled but the
   ledger DEGRADED, the private state file must be written during the run
   (degraded mode enforces from the private counter, so a hard kill would
   otherwise let the next run double-spend).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm.base import LLMClientBase, clear_abort, request_abort
from llm.shared_ledger import BucketKey, UsageSnapshot
from llm.token_tracker import DailyTokenTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**overrides: Any) -> LLMClientBase:
    """Create a bare LLMClientBase bypassing __init__ (see test_retry_window)."""
    client = LLMClientBase.__new__(LLMClientBase)
    defaults: dict[str, Any] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "timeout": 300,
        "rate_limiter": None,
        "max_retries": 5,
        "max_elapsed": 0.0,
        "chat_model": MagicMock(),
        "_stats_lock": threading.Lock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "model_config": {},
        "service_tier": "auto",
        "schema_retry_config": {},
        "_output_schema": None,
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(client, attr, val)
    client._extract_tokens_from_exception = MagicMock()  # type: ignore[method-assign]
    client._report_error = MagicMock()  # type: ignore[method-assign]
    client._calculate_backoff = MagicMock(return_value=0.0)  # type: ignore[method-assign]
    return client


class _AbortingLimiter:
    """Rate limiter whose wait returns early because the abort arrived mid-wait.

    Mirrors ``RateLimiter.wait_for_capacity``'s abort contract: it returns the
    elapsed wait without recording a request, and the caller must not then fire
    the API call.
    """

    def __init__(self) -> None:
        self.waits = 0

    def wait_for_capacity(self, should_abort: Any = None) -> float:
        self.waits += 1
        request_abort()
        return 0.0


class _RolloverLedger:
    """Ledger stand-in that can trigger a day rollover from inside a call."""

    def __init__(self) -> None:
        self.seed_calls = 0
        self.sync_calls: list[dict[BucketKey, int]] = []
        self.on_sync: Any = None
        self.combined = 0

    def _snapshot(self) -> UsageSnapshot:
        bucket = BucketKey("openai", "OPENAI_API_KEY", None)
        return UsageSnapshot(
            combined=self.combined,
            own_total=self.combined,
            buckets={bucket: self.combined},
            own_buckets={bucket: self.combined},
        )

    def seed_usage(
        self, own_total: int, own_buckets: dict[BucketKey, int] | None = None
    ) -> UsageSnapshot | None:
        self.seed_calls += 1
        return self._snapshot()

    def sync_usage(self, deltas: dict[BucketKey, int]) -> UsageSnapshot | None:
        self.sync_calls.append(dict(deltas))
        self.combined += sum(deltas.values())
        if self.on_sync is not None:
            self.on_sync()
        return self._snapshot()


class _DeadLedger:
    """Ledger stand-in that always fails, forcing degraded mode."""

    def seed_usage(
        self, own_total: int, own_buckets: dict[BucketKey, int] | None = None
    ) -> UsageSnapshot | None:
        return None

    def sync_usage(self, deltas: dict[BucketKey, int]) -> UsageSnapshot | None:
        return None


def _make_tracker(tmp_path: Path, state_name: str = "state.json") -> DailyTokenTracker:
    """Shared-budget tracker pinned to a scratch ledger directory."""
    return DailyTokenTracker(
        daily_limit=10_000_000,
        enabled=True,
        state_file=tmp_path / state_name,
        chunk_estimate_seed=1,
        estimate_smoothing=0.0,
        shared_enabled=True,
        shared_ledger_dir=tmp_path / "ledger",
    )


_BUCKET = BucketKey("openai", "OPENAI_API_KEY", None)


# ---------------------------------------------------------------------------
# Fix 1: abort arriving during the rate-limiter wait
# ---------------------------------------------------------------------------
class TestAbortDuringRateLimitWait:
    """The limiter's early return must not be followed by an API call."""

    def test_no_api_call_when_abort_lands_mid_wait(self) -> None:
        limiter = _AbortingLimiter()
        client = _make_client(rate_limiter=limiter)
        structured_model = MagicMock()

        clear_abort()
        try:
            with pytest.raises(RuntimeError, match="Abort requested"):
                client._invoke_with_retry(
                    structured_model, [], {}, "Transcription for page_001.png"
                )
        finally:
            clear_abort()

        assert limiter.waits == 1
        structured_model.invoke.assert_not_called()

    def test_abort_raise_is_not_retried_or_classified(self) -> None:
        """The raise happens outside the try: no retry, no error report."""
        limiter = _AbortingLimiter()
        client = _make_client(rate_limiter=limiter, max_retries=5, max_elapsed=900.0)
        structured_model = MagicMock()

        clear_abort()
        try:
            with pytest.raises(RuntimeError, match="Abort requested"):
                client._invoke_with_retry(structured_model, [], {}, "Summary for x")
        finally:
            clear_abort()

        # A single pass through the loop: no backoff, no error accounting.
        assert limiter.waits == 1
        client._report_error.assert_not_called()  # type: ignore[attr-defined]
        client._calculate_backoff.assert_not_called()  # type: ignore[attr-defined]
        client._extract_tokens_from_exception.assert_not_called()  # type: ignore[attr-defined]

    def test_normal_wait_still_reaches_the_model(self) -> None:
        """Control: without an abort the wait is followed by the invoke."""
        limiter = MagicMock()
        limiter.wait_for_capacity.return_value = 0.0
        client = _make_client(rate_limiter=limiter)
        structured_model = MagicMock()
        structured_model.invoke.return_value = "ok"

        clear_abort()
        assert client._invoke_with_retry(structured_model, [], {}, "ctx") == "ok"
        limiter.wait_for_capacity.assert_called_once()
        structured_model.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 2: day rollover during an in-flight ledger sync
# ---------------------------------------------------------------------------
class TestRolloverDuringLedgerSync:
    """A rollover mid-sync must not mix the two budget days' accounting."""

    def _arm(self, t: DailyTokenTracker, ledger: Any, day: str) -> None:
        """Put the tracker into a seeded, healthy, mid-run shared state."""
        with t._lock:
            t._ledger = ledger
            t._current_date = day
            t._seeded = True
            t._ledger_degraded = False
            t._combined_total = 5_000
            t._bucket_totals = {_BUCKET: 5_000}
            t._unsynced_deltas = {_BUCKET: 400}
            t._tokens_used_today = 400
            t._own_buckets = {_BUCKET: 400}
            t._last_ledger_sync_monotonic = 0.0

    def test_rollover_before_ledger_call_abandons_the_sync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = _make_tracker(tmp_path)
        ledger = _RolloverLedger()
        self._arm(t, ledger, "2026-08-02")

        # The clock crosses the budget day right after the snapshot was taken.
        monkeypatch.setattr(t, "_get_current_date_str", lambda: "2026-08-03")

        t.sync_ledger_now()

        assert ledger.sync_calls == []  # nothing pushed into the new day
        assert ledger.seed_calls == 0
        assert t._unsynced_deltas == {_BUCKET: 400}  # still queued
        assert t._ledger_sync_in_flight is False

    def test_rollover_mid_call_skips_the_writeback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = _make_tracker(tmp_path)
        ledger = _RolloverLedger()
        self._arm(t, ledger, "2026-08-02")

        today = {"value": "2026-08-02"}
        monkeypatch.setattr(t, "_get_current_date_str", lambda: today["value"])

        def _rollover() -> None:
            # The real rollover path, fired while the ledger call is in flight
            # (the tracker lock is released across the call by design).
            today["value"] = "2026-08-03"
            with t._lock:
                t._check_and_reset_if_new_day()

        ledger.on_sync = _rollover

        t.sync_ledger_now()

        assert len(ledger.sync_calls) == 1  # the call did go out on the old day
        # The writeback is skipped entirely: the fresh day's zeroed mirrors and
        # its re-seed request survive intact.
        assert t._current_date == "2026-08-03"
        assert t._unsynced_deltas == {}
        assert t._combined_total == 0
        assert t._bucket_totals == {}
        assert t._seeded is False
        assert t._tokens_used_today == 0
        assert t._ledger_sync_in_flight is False

    def test_same_day_sync_still_writes_back(self, tmp_path: Path) -> None:
        """Control: without a rollover the snapshot is adopted as before."""
        t = _make_tracker(tmp_path)
        ledger = _RolloverLedger()
        ledger.combined = 5_000
        self._arm(t, ledger, t._get_current_date_str())

        t.sync_ledger_now()

        assert len(ledger.sync_calls) == 1
        assert ledger.sync_calls[0] == {_BUCKET: 400}
        assert t._unsynced_deltas == {}  # pushed deltas cleared
        assert t._combined_total == 5_400
        assert t._ledger_degraded is False
        assert t._ledger_sync_in_flight is False


# ---------------------------------------------------------------------------
# Fix 3: degraded shared budget must persist to the private state file
# ---------------------------------------------------------------------------
class TestDegradedPersistence:
    """Degraded mode enforces from the private counter, so it must persist."""

    def _degrade(self, t: DailyTokenTracker) -> None:
        with t._lock:
            t._ledger = _DeadLedger()  # type: ignore[assignment]
            t._ledger_degraded = True
            t._seeded = False
            t._combined_total = 0
            t._unsynced_deltas = {}
            t._tokens_used_today = 0
            t._own_buckets = {}
            t._last_save_time = 0.0  # bypass the write debounce

    def test_add_tokens_writes_state_file_while_degraded(self, tmp_path: Path) -> None:
        t = _make_tracker(tmp_path)
        state_file = tmp_path / "state.json"
        state_file.unlink(missing_ok=True)
        self._degrade(t)

        t.add_tokens(750, provider="openai", key_env="OPENAI_API_KEY")

        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 750
        assert t._ledger_degraded is True

    def test_healthy_ledger_still_skips_the_private_write(self, tmp_path: Path) -> None:
        """Control: while healthy the ledger stays the sole in-run persistence."""
        t = _make_tracker(tmp_path)
        state_file = tmp_path / "state.json"
        ledger = _RolloverLedger()
        with t._lock:
            t._ledger = ledger  # type: ignore[assignment]
            t._ledger_degraded = False
            t._seeded = True
            t._unsynced_deltas = {}
            t._last_save_time = 0.0
            t._last_ledger_sync_monotonic = 0.0
        state_file.unlink(missing_ok=True)

        t.add_tokens(750, provider="openai", key_env="OPENAI_API_KEY")

        assert not state_file.exists()
