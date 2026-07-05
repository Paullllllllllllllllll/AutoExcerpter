"""Token usage tracking with daily limits and timezone-aware reset.

This module provides a thread-safe token tracker that:
- Counts total tokens used from OpenAI API responses
- Enforces configurable daily token limits
- Automatically resets at 00:01 UTC, one minute after OpenAI's 00:00 UTC
  free-tier reset
- Persists state to disk to survive application restarts
- Thread-safe for concurrent API calls

Usage:
    from llm.token_tracker import get_token_tracker

    tracker = get_token_tracker()

    # Check if we can proceed
    if tracker.can_use_tokens():
        # Make API call
        response = api_call()

        # Report usage
        tokens = response.usage.total_tokens
        tracker.add_tokens(tokens)

    # Check if limit is reached
    if tracker.is_limit_reached():
        wait_time = tracker.get_seconds_until_reset()
        # Wait or defer processing
"""

from __future__ import annotations

import atexit
import contextlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import app as config
from config.logger import setup_logger

if TYPE_CHECKING:
    from llm.shared_ledger import SharedTokenLedger

__all__ = [
    "DailyTokenTracker",
    "get_token_tracker",
    "wait_for_token_reset",
    "config",
]

logger = setup_logger(__name__)

# Legacy (pre-state-dir) token state file in the working directory. Adopted
# once into the user-level state dir when the user-level file is absent.
_LEGACY_TOKEN_TRACKER_FILE = Path.cwd() / ".autoexcerpter_token_state.json"
# Fallback default when the state dir cannot be resolved (kept for callers that
# construct a tracker without an explicit state_file).
_TOKEN_TRACKER_FILE = _LEGACY_TOKEN_TRACKER_FILE
_TOKEN_STATE_FILENAME = "token_state.json"

# Minimum seconds between debounced state writes (a flush at exit persists the
# final value). Prevents a per-API-call write storm during large runs.
_MIN_SAVE_INTERVAL_S = 1.0

# Minimum seconds between shared-ledger syncs on the debounced (add_tokens) path.
# Deliberately longer than the private-file debounce: a ledger sync takes an OS
# file lock, so it is heavier than a private write. Forced syncs (near-limit
# admission, wait-loop polls, exit flush) bypass this interval.
_LEDGER_SYNC_DEBOUNCE_S = 2.0

# This tool's field name in the shared cross-tool ledger.
_LEDGER_TOOL_NAME = "autoexcerpter"

# One-minute safety buffer past OpenAI's 00:00 UTC free-tier reset, so the
# tracker never frees its budget before the upstream quota has actually reset.
# Mirrors llm.shared_ledger._RESET_BUFFER (kept as a separate constant since
# this module is not vendored and may import the ledger only under
# TYPE_CHECKING).
_RESET_BUFFER = timedelta(minutes=1)

# Singleton instance
_tracker_instance: DailyTokenTracker | None = None
_tracker_lock = threading.Lock()


class DailyTokenTracker:
    """
    Thread-safe daily token usage tracker with persistent state.

    Tracks token usage across API calls and enforces daily limits with
    automatic reset at 00:01 UTC (one minute after OpenAI's 00:00 UTC
    free-tier reset).
    """

    daily_limit: int
    enabled: bool
    state_file: Path
    _lock: threading.Lock
    _current_date: str
    _tokens_used_today: int

    def __init__(
        self,
        daily_limit: int,
        enabled: bool = True,
        state_file: Path | None = None,
        chunk_estimate_seed: int = 25_000,
        estimate_smoothing: float = 0.3,
        shared_enabled: bool = False,
        shared_ledger_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the token tracker.

        Args:
            daily_limit: Maximum tokens allowed per day.
            enabled: Whether token limiting is enabled.
            state_file: Path to persistent state file
                (default: .autoexcerpter_token_state.json).
            chunk_estimate_seed: Cold-start estimate (in tokens) of how many
                tokens one page (transcription + optional summary) consumes,
                used by try_reserve() before any usage has been observed.
            estimate_smoothing: EWMA smoothing factor (0-1) applied to observed
                per-call token usage; higher reacts faster to recent calls.
            shared_enabled: Opt-in cross-tool combined budget. When True the
                daily limit is enforced against the COMBINED usage of every
                participating ChronoPipeline tool via a shared on-disk ledger,
                and that ledger replaces the private state file as persistence.
                When False (default) behaviour is bit-for-bit the private
                per-tool tracker with no ledger I/O whatsoever.
            shared_ledger_dir: Directory holding the shared ledger. Empty/None
                means the ledger default (``~/.chronopipeline``).
        """
        self.daily_limit = daily_limit
        self.enabled = enabled
        self.state_file = state_file or _TOKEN_TRACKER_FILE

        # Thread safety
        self._lock = threading.Lock()

        # Debounced-write bookkeeping.
        self._last_save_time: float = 0.0
        self._pending_save: bool = False

        # Token tracking state
        self._current_date: str = ""  # Format: YYYY-MM-DD
        self._tokens_used_today: int = 0

        # Page-level reservation state (in-memory only; transient per run).
        # _tokens_reserved is headroom claimed by in-flight pages that have not
        # yet committed actual usage via add_tokens(); the admission check in
        # try_reserve() subtracts both committed and reserved tokens so that
        # concurrent worker threads cannot collectively overshoot the limit.
        self._tokens_reserved: int = 0
        self._seed: int = max(1, int(chunk_estimate_seed))
        self._alpha: float = min(1.0, max(0.0, float(estimate_smoothing)))
        self._ewma: float = float(self._seed)

        # Per-page usage accumulator (thread-local). A page spends one or more
        # API calls (transcription + optional summary + retries) on a single
        # worker thread; each add_tokens() adds to this thread's running page
        # total, and record_page_usage() feeds that ONE per-page observation to
        # the EWMA. The EWMA therefore tracks per-page cost (what try_reserve
        # gates on) rather than per-call cost, which under-reserved ~2x when a
        # page made two calls.
        self._page_local = threading.local()

        # Shared cross-tool ledger state (only touched when shared_enabled).
        # The ledger is constructed lazily on first use so a disabled tracker
        # performs zero ledger I/O. _unsynced_delta accumulates committed tokens
        # not yet pushed to the ledger; _combined_total caches the last-known
        # combined usage across all tools. Budget math while enabled uses
        # (_combined_total + _unsynced_delta) as the effective usage. The ledger
        # sync rides the existing debounced save: add_tokens accumulates the
        # delta and the debounced path pushes ledger.sync(delta) INSTEAD of
        # writing the private state file.
        self._shared_enabled: bool = bool(shared_enabled)
        self._shared_ledger_dir: str | Path | None = shared_ledger_dir or None
        self._ledger: SharedTokenLedger | None = None
        self._ledger_construct_failed: bool = False
        self._ledger_tool_name: str = _LEDGER_TOOL_NAME
        self._unsynced_delta: int = 0
        self._combined_total: int = 0
        self._seeded: bool = False
        self._ledger_degraded: bool = False
        self._ledger_sync_in_flight: bool = False
        self._last_ledger_sync_monotonic: float = 0.0

        # Load existing state from disk
        self._load_state()

        # Seed the shared ledger once at init so the combined baseline (this
        # tool's prior same-day usage plus any concurrent tools) is known before
        # the first admission check. Best-effort: a degraded ledger simply leaves
        # the tracker in standalone mode. Init is off any hot path, so this runs
        # inline and the combined total is visible immediately.
        if self._shared_enabled:
            with contextlib.suppress(Exception):
                self.sync_ledger_now()

        logger.info(
            f"Token tracker initialized: enabled={enabled}, "
            f"daily_limit={daily_limit:,}, "
            f"current_usage={self._tokens_used_today:,}, "
            f"shared_budget={self._shared_enabled}"
        )

    def _get_current_date_str(self) -> str:
        """Get the current budget-day key (YYYY-MM-DD), buffered UTC.

        The day rolls over at 00:01 UTC rather than at exact UTC midnight, so
        the reset never fires before OpenAI's own 00:00 UTC free-tier reset
        has actually happened.
        """
        return (datetime.now(UTC) - _RESET_BUFFER).strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        """Load token usage state from disk."""
        if not self.state_file.exists():
            # No existing state, initialize fresh
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0
            logger.debug("No existing token state file found, starting fresh")
            return

        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            saved_date = state.get("date", "")
            saved_tokens = state.get("tokens_used", 0)

            current_date = self._get_current_date_str()

            if saved_date == current_date:
                # Same day, restore token count
                self._current_date = saved_date
                self._tokens_used_today = saved_tokens
                logger.info(
                    f"Loaded token state for {current_date}: "
                    f"{self._tokens_used_today:,} tokens used"
                )
            else:
                # Different day, reset counter
                self._current_date = current_date
                self._tokens_used_today = 0
                logger.info(
                    f"New day detected (was {saved_date}, now {current_date}). "
                    "Token counter reset to 0."
                )
                # Save the reset state
                self._save_state(force=True)

        except Exception as e:
            logger.warning(f"Error loading token state from {self.state_file}: {e}")
            # Initialize fresh on error
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0

    def _save_state(self, force: bool = False) -> None:
        """Save current token usage state to disk (debounced).

        Writes are coalesced to at most one per ``_MIN_SAVE_INTERVAL_S`` unless
        *force* is set (used for day-rollover and the exit flush), so a large
        run does not write the state file on every API call. A pending write is
        persisted by :meth:`flush`. The on-disk write goes through the shared
        :func:`config.state.write_json_atomic` helper (per-process-unique temp
        name plus PermissionError/FileNotFoundError retry).
        """
        now = time.time()
        if not force and (now - self._last_save_time) < _MIN_SAVE_INTERVAL_S:
            self._pending_save = True
            return

        state = {
            "date": self._current_date,
            "tokens_used": self._tokens_used_today,
            "last_updated": datetime.now().isoformat(),
        }
        from config.state import write_json_atomic

        write_json_atomic(self.state_file, state)
        self._last_save_time = now
        self._pending_save = False

    def set_daily_limit(self, new_limit: int) -> None:
        """Update the daily token limit at runtime.

        Used by the wait loops so a user editing ``app.yaml`` mid-wait (raising
        ``daily_token_limit.daily_tokens``) lifts the cap without a restart. A
        no-op when the value is unchanged.
        """
        new_limit = int(new_limit)
        with self._lock:
            if new_limit != self.daily_limit:
                logger.info(
                    "Daily token limit updated: %s -> %s",
                    f"{self.daily_limit:,}",
                    f"{new_limit:,}",
                )
                self.daily_limit = new_limit

    def flush(self) -> None:
        """Persist any debounced pending state write.

        When the shared budget is enabled this instead forces a final ledger
        sync so the last accumulated delta lands before exit; if a sync is in
        flight, wait briefly (bounded) for it to clear so the final push is not
        skipped by the in-flight guard. The private state file is not written
        while the ledger is the active persistence.
        """
        if self._shared_enabled:
            for _ in range(50):
                with self._lock:
                    busy = self._ledger_sync_in_flight
                if not busy:
                    break
                time.sleep(0.02)
            with contextlib.suppress(Exception):
                self.sync_ledger_now()
            return
        with self._lock:
            if self._pending_save:
                self._save_state(force=True)

    # ------------------------------------------------------------------
    # Shared cross-tool ledger integration
    # ------------------------------------------------------------------

    def _get_or_create_ledger_locked(self) -> SharedTokenLedger | None:
        """Construct the shared ledger lazily. Must hold ``self._lock``.

        Construction touches no filesystem (the ledger defers all I/O to its
        locked merge), so a bad directory cannot fail here; a genuinely invalid
        tool name is the only ValueError, and it is latched so we do not retry
        forever.
        """
        if self._ledger is None and not self._ledger_construct_failed:
            try:
                from llm.shared_ledger import SharedTokenLedger

                self._ledger = SharedTokenLedger(
                    self._ledger_tool_name,
                    ledger_dir=self._shared_ledger_dir,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._ledger_construct_failed = True
                logger.warning("Could not construct shared token ledger: %s", exc)
        return self._ledger

    def _effective_used_locked(self) -> int:
        """Return the usage figure the budget is enforced against. Hold the lock.

        Enabled and healthy: the last-known combined total across all tools plus
        this tool's not-yet-synced delta, so our own in-flight usage is never
        undercounted. Disabled or degraded: the private per-tool count, i.e.
        exactly today's standalone semantics.
        """
        if self._shared_enabled and not self._ledger_degraded:
            return self._combined_total + self._unsynced_delta
        return self._tokens_used_today

    def _due_for_ledger_sync_locked(self, force: bool) -> bool:
        """Whether a ledger sync should be dispatched now. Must hold the lock."""
        if not self._shared_enabled:
            return False
        if self._ledger_sync_in_flight:
            return False
        if force:
            return True
        elapsed = time.monotonic() - self._last_ledger_sync_monotonic
        return elapsed >= _LEDGER_SYNC_DEBOUNCE_S

    def _perform_ledger_sync(self) -> None:
        """Run a ledger sync inline on the calling thread.

        This repo is fully threaded with no asyncio event loop, so ledger I/O
        never risks blocking a loop; the sync runs inline (outside the tracker
        lock) on whichever worker thread found a sync due. The in-flight guard
        in :meth:`sync_ledger_now` keeps concurrent workers from stacking syncs,
        so the hot path never blocks on file I/O beyond the debounce gate.
        """
        self.sync_ledger_now()

    def sync_ledger_now(self) -> None:
        """Seed-or-sync the shared ledger, writing back the combined total.

        Discipline: snapshot the delta under the tracker lock, call the ledger
        (seed or sync) with the lock RELEASED, then write the returned combined
        total back under the lock. The ledger has its own internal mutex; we
        never hold the tracker lock across a ledger call so the hot path cannot
        stall on ledger I/O. Degradation (ledger returns None) leaves the tracker
        in standalone mode with the unsynced delta preserved so a transient
        failure self-heals and the full accumulated amount replays on a later
        sync.
        """
        if not self._shared_enabled:
            return

        with self._lock:
            if self._ledger_sync_in_flight:
                return
            self._ledger_sync_in_flight = True
            ledger = self._get_or_create_ledger_locked()
            need_seed = not self._seeded
            own_committed = self._tokens_used_today
            delta = self._unsynced_delta
            self._last_ledger_sync_monotonic = time.monotonic()

        try:
            if ledger is None:
                with self._lock:
                    self._ledger_degraded = True
                return

            own_field: int | None = None
            if need_seed:
                combined = ledger.seed(own_committed)
                if combined is not None:
                    breakdown = ledger.read_breakdown()
                    if breakdown is not None:
                        own_field = int(
                            breakdown.get(self._ledger_tool_name, own_committed)
                        )
            else:
                combined = ledger.sync(delta)

            with self._lock:
                if combined is None:
                    # Degraded: keep the unsynced delta so the full accumulated
                    # amount is pushed once the ledger recovers.
                    self._ledger_degraded = True
                else:
                    self._ledger_degraded = False
                    self._combined_total = combined
                    if need_seed:
                        self._seeded = True
                        baseline = own_field if own_field is not None else own_committed
                        if baseline > self._tokens_used_today:
                            self._tokens_used_today = baseline
                        # Any delta committed during the seed round is preserved
                        # for the next sync; the baseline is now in the ledger.
                        self._unsynced_delta = max(
                            0, self._tokens_used_today - baseline
                        )
                    else:
                        # Subtract only what we pushed; deltas that arrived
                        # mid-sync remain queued for the next push.
                        self._unsynced_delta = max(0, self._unsynced_delta - delta)
        finally:
            with self._lock:
                self._ledger_sync_in_flight = False

    def _maybe_forced_refresh_before_admit(self) -> None:
        """Force a ledger refresh before a reservation when it matters.

        Triggers a forced (debounce-bypassing) sync when the shared budget is not
        yet seeded, or when the cached combined total already exceeds 80% of the
        daily limit, so admission near the cap sees the freshest cross-tool usage.
        Runs inline (fresh value visible to the immediately following check). A
        no-op when the shared budget is disabled.
        """
        if not self._shared_enabled:
            return
        trigger = False
        with self._lock:
            near_limit = (
                self.daily_limit > 0 and self._combined_total > 0.8 * self.daily_limit
            )
            if (not self._seeded or near_limit) and self._due_for_ledger_sync_locked(
                force=True
            ):
                trigger = True
        if trigger:
            self._perform_ledger_sync()

    def _check_and_reset_if_new_day(self) -> None:
        """Check if it's a new day and reset counter if needed."""
        current_date = self._get_current_date_str()

        if current_date != self._current_date:
            logger.info(
                f"New day detected: {current_date} (was {self._current_date}). "
                f"Resetting token counter from {self._tokens_used_today:,} to 0."
            )
            self._current_date = current_date
            self._tokens_used_today = 0
            if self._shared_enabled:
                # The ledger rolls over internally; reset the local mirror and
                # force a re-seed on the next sync. The private file is left
                # untouched while the shared budget is the active persistence.
                self._unsynced_delta = 0
                self._combined_total = 0
                self._seeded = False
            else:
                self._save_state(force=True)

    def add_tokens(self, tokens: int) -> None:
        """
        Add tokens to the daily count.

        Args:
            tokens: Number of tokens to add.
        """
        if not self.enabled or tokens <= 0:
            return

        do_ledger_sync = False
        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens

            if self._shared_enabled:
                # Ledger is the active persistence: accumulate the delta and
                # decide (under the lock) whether a debounced sync is due. The
                # ledger I/O itself runs OUTSIDE the lock below, so the hot path
                # never takes the OS file lock while holding the tracker lock.
                self._unsynced_delta += tokens
                do_ledger_sync = self._due_for_ledger_sync_locked(force=False)
            else:
                # Debounced private-file write (unchanged standalone behaviour).
                self._save_state()

            logger.debug(
                f"Added {tokens:,} tokens. "
                f"Daily total: {self._tokens_used_today:,}/{self.daily_limit:,}"
            )

        if do_ledger_sync:
            self._perform_ledger_sync()

        # Accumulate this call's tokens into the current thread's page total
        # (outside the lock; thread-local is single-owner). The EWMA is updated
        # once per page via record_page_usage(), not per call.
        self._page_local.total = getattr(self._page_local, "total", 0) + tokens

    def record_page_usage(self, total: int | None = None) -> None:
        """Feed one page's total actual token usage to the reservation EWMA.

        Called once per page (from the worker's finally block). When *total* is
        None the accumulated per-thread sum from :meth:`add_tokens` for the
        just-finished page is used, then reset. A single per-page observation
        (transcription + optional summary + retries) drives the EWMA that
        :meth:`try_reserve` gates on, so the estimate converges to per-page
        cost. When summarization is disabled a page is one call and the EWMA
        tracks per-call automatically. No-op when limiting is disabled.
        """
        if not self.enabled:
            self._page_local.total = 0
            return

        if total is None:
            total = int(getattr(self._page_local, "total", 0))
        self._page_local.total = 0

        if total <= 0:
            return
        with self._lock:
            self._ewma = self._alpha * total + (1.0 - self._alpha) * self._ewma

    def try_reserve(self, estimate: int | None = None) -> int | None:
        """Reserve estimated tokens for one page before launching it.

        The estimate is the larger of the caller-supplied hint and the rolling
        EWMA of observed per-call usage. Image pages have no cheap pre-count, so
        callers typically pass no hint and rely on the EWMA.

        Returns the reserved amount, ``0`` when limiting is disabled (admit
        freely, nothing to release), or ``None`` when the remaining budget
        cannot cover the estimate (caller should stop admitting new work). A
        non-zero reservation must be matched by a later :meth:`release` of the
        same amount once the page completes.
        """
        if not self.enabled:
            return 0

        # Forced pre-admission refresh when near the combined cap (or not yet
        # seeded). Runs outside the tracker lock, inline, so the fresh combined
        # total is visible to the admission check below. No-op when the shared
        # budget is disabled.
        self._maybe_forced_refresh_before_admit()

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))
            available = (
                self.daily_limit - self._effective_used_locked() - self._tokens_reserved
            )
            if est > available:
                return None
            self._tokens_reserved += est
            return est

    def _current_estimate(self) -> int:
        """Current per-page token estimate (EWMA floor of 1)."""
        return max(1, round(self._ewma))

    def can_admit_page(self, estimate: int | None = None) -> bool:
        """Whether the remaining budget can admit one more page.

        Unlike :meth:`is_limit_reached` (which only fires when the budget is
        fully spent), this returns False when a positive budget remains but is
        too small to cover a page's estimated usage. Callers use it to decide
        whether to keep admitting work or wait for the daily reset, so a partial
        remainder that cannot fit a page is treated as wait-worthy rather than
        spun on.
        """
        if not self.enabled:
            return True

        # Forced pre-admission refresh when near the combined cap (or not yet
        # seeded); no-op when the shared budget is disabled.
        self._maybe_forced_refresh_before_admit()

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), self._current_estimate())
            available = (
                self.daily_limit - self._effective_used_locked() - self._tokens_reserved
            )
            return est <= available

    def release(self, amount: int) -> None:
        """Release a reservation made by :meth:`try_reserve` after the page.

        Actual usage is committed separately via :meth:`add_tokens`; releasing
        only frees the transient headroom the reservation was holding.
        """
        if not self.enabled or amount <= 0:
            return

        with self._lock:
            self._tokens_reserved = max(0, self._tokens_reserved - amount)

    def get_tokens_used_today(self) -> int:
        """
        Get the number of tokens used today.

        With the shared budget enabled this is the COMBINED usage across all
        participating tools (the figure the daily limit is enforced against);
        otherwise it is this tool's private count. See
        :meth:`get_own_tokens_used_today` for the per-tool figure.

        Returns:
            Token count for current day.
        """
        with self._lock:
            self._check_and_reset_if_new_day()
            return self._effective_used_locked()

    def get_own_tokens_used_today(self) -> int:
        """Return this tool's private token count for today (never combined)."""
        with self._lock:
            self._check_and_reset_if_new_day()
            return self._tokens_used_today

    def get_tokens_remaining(self) -> int:
        """
        Get the number of tokens remaining for today.

        Returns:
            Remaining token count (0 if limit exceeded).
        """
        if not self.enabled:
            return self.daily_limit  # Unlimited

        with self._lock:
            self._check_and_reset_if_new_day()
            remaining = self.daily_limit - self._effective_used_locked()
            return max(0, remaining)

    def is_limit_reached(self) -> bool:
        """
        Check if the daily token limit has been reached.

        Returns:
            True if limit is reached or exceeded, False otherwise.
        """
        if not self.enabled:
            return False

        return self.get_tokens_remaining() == 0

    def can_use_tokens(self, estimated_tokens: int = 0) -> bool:
        """
        Check if we can use a certain number of tokens.

        Args:
            estimated_tokens: Estimated tokens needed (default: 0 for any usage).

        Returns:
            True if we can proceed, False if limit would be exceeded.
        """
        if not self.enabled:
            return True

        remaining = self.get_tokens_remaining()

        if estimated_tokens > 0:
            return remaining >= estimated_tokens
        else:
            # Just check if any tokens remain
            return remaining > 0

    def get_seconds_until_reset(self) -> int:
        """
        Get the number of seconds until the counter resets (00:01 UTC).

        Returns:
            Seconds until the next 00:01 UTC reset.
        """
        now = datetime.now(UTC)
        delta = self.get_reset_time() - now
        return max(0, int(delta.total_seconds()))

    def get_reset_time(self) -> datetime:
        """
        Get the datetime when the counter will reset.

        Returns:
            Timezone-aware UTC datetime of the next 00:01 UTC reset (one
            minute after OpenAI's 00:00 UTC free-tier reset).
        """
        now = datetime.now(UTC)
        anchor = now - _RESET_BUFFER
        # datetime.min.time() is time(0, 0), i.e. midnight, without importing
        # the datetime.time class (which would shadow the stdlib time module
        # imported above for time.monotonic()/time.sleep()).
        next_midnight = datetime.combine(
            anchor.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        return next_midnight + _RESET_BUFFER

    def get_usage_percentage(self) -> float:
        """
        Get current usage as percentage of daily limit.

        Returns:
            Percentage (0-100+) of daily limit used.
        """
        if not self.enabled or self.daily_limit == 0:
            return 0.0

        used = self.get_tokens_used_today()
        return (used / self.daily_limit) * 100.0

    def get_stats(self) -> dict[str, Any]:
        """
        Get comprehensive token usage statistics.

        Returns:
            Dictionary with usage stats.
        """
        used = self.get_tokens_used_today()
        remaining = self.get_tokens_remaining()
        percentage = self.get_usage_percentage()
        seconds_until_reset = self.get_seconds_until_reset()
        reset_time = self.get_reset_time()

        stats: dict[str, Any] = {
            "enabled": self.enabled,
            "daily_limit": self.daily_limit,
            "tokens_used_today": used,
            "tokens_remaining": remaining,
            "usage_percentage": round(percentage, 2),
            "limit_reached": self.is_limit_reached(),
            "seconds_until_reset": seconds_until_reset,
            "reset_time": reset_time.isoformat(),
            "current_date": self._current_date,
        }
        stats.update(self._shared_stats())
        return stats

    def _shared_stats(self) -> dict[str, Any]:
        """Shared-budget stats: combined total, own count, and per-tool split.

        Returns an empty dict when the shared budget is disabled so callers see
        no change. ``read_breakdown`` is a lock-free ledger read run outside the
        tracker lock; ``tokens_used_today`` above already reflects the combined
        figure when enabled.
        """
        if not self._shared_enabled:
            return {}
        with self._lock:
            ledger = self._ledger
            own = self._tokens_used_today
            combined = self._effective_used_locked()
            degraded = self._ledger_degraded
        breakdown: dict[str, int] | None = None
        if ledger is not None:
            breakdown = ledger.read_breakdown()
        return {
            "shared_budget_enabled": True,
            "shared_budget_degraded": degraded,
            "own_tokens_used_today": own,
            "combined_tokens_used_today": combined,
            "shared_breakdown": breakdown or {},
        }


def _describe_reset_time(reset_time: datetime) -> str:
    """Render an aware-UTC reset instant for user-facing messages.

    The actual reset always happens at 00:01 UTC regardless of the local
    offset, so the UTC anchor is always shown alongside the more readable
    local wall-clock time.
    """
    local = reset_time.astimezone()
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} local (00:01 UTC)"


def get_token_tracker() -> DailyTokenTracker:
    """
    Get the singleton token tracker instance.

    Returns:
        Global DailyTokenTracker instance.
    """
    global _tracker_instance

    if _tracker_instance is None:
        with _tracker_lock:
            # Double-check locking
            if _tracker_instance is None:
                _tracker_instance = DailyTokenTracker(
                    daily_limit=config.DAILY_TOKEN_LIMIT,
                    enabled=config.DAILY_TOKEN_LIMIT_ENABLED,
                    state_file=_resolve_state_file(),
                    chunk_estimate_seed=config.DAILY_TOKEN_CHUNK_ESTIMATE_SEED,
                    estimate_smoothing=config.DAILY_TOKEN_ESTIMATE_SMOOTHING,
                    shared_enabled=config.SHARED_TOKEN_BUDGET_ENABLED,
                    shared_ledger_dir=config.SHARED_TOKEN_BUDGET_LEDGER_DIR or None,
                )
                atexit.register(_tracker_instance.flush)

    return _tracker_instance


def _resolve_state_file() -> Path:
    """Resolve the user-level token state file, adopting a legacy CWD file once."""
    try:
        from config.state import resolve_state_file

        return resolve_state_file(
            _TOKEN_STATE_FILENAME, legacy_path=_LEGACY_TOKEN_TRACKER_FILE
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Falling back to legacy token state file: %s", exc)
        return _TOKEN_TRACKER_FILE


def wait_for_token_reset() -> bool:
    """Block until the daily token limit resets, or the user interrupts.

    Lives here (not in cli/) so the threaded transcription pipeline can wait
    mid-item without importing the CLI layer (which imports the pipeline).
    Returns True when the limit has reset and processing may resume, or
    immediately when the limit is not currently reached; returns False if a
    KeyboardInterrupt cancels the wait.
    """
    tracker = get_token_tracker()
    # Wait-worthy when a page cannot fit the remaining budget, even if that
    # budget is still positive (a partial remainder too small for one page).
    if not tracker.enabled or tracker.can_admit_page():
        return True

    seconds_until_reset = tracker.get_seconds_until_reset()
    reset_time = tracker.get_reset_time()
    stats = tracker.get_stats()
    logger.warning(
        "Daily token budget cannot fit another page: %s/%s tokens used. "
        "Waiting until %s for reset...",
        f"{stats['tokens_used_today']:,}",
        f"{stats['daily_limit']:,}",
        _describe_reset_time(reset_time),
    )

    elapsed = 0
    try:
        while elapsed < seconds_until_reset:
            interval = min(1, max(0, seconds_until_reset - elapsed))
            time.sleep(interval)
            elapsed += interval

            # Forced ledger refresh each poll so another tool's usage or its
            # 00:01 UTC reset is observed while we wait. A no-op when the shared
            # budget is disabled, so single-tool waits are unchanged.
            if tracker._shared_enabled:
                with contextlib.suppress(Exception):
                    tracker.sync_ledger_now()

            # Live re-read of the configured daily limit: a user raising
            # daily_token_limit.daily_tokens mid-wait lifts the cap without a
            # restart. A read failure keeps the current limit.
            new_limit = config.reload_daily_token_limit()
            if new_limit is not None:
                tracker.set_daily_limit(new_limit)

            if tracker.can_admit_page():
                logger.info("Token budget has reset. Resuming processing.")
                return True
        logger.info("Token budget has reset. Resuming processing.")
        return True
    except KeyboardInterrupt:
        logger.info("Token-limit wait cancelled by user.")
        return False
