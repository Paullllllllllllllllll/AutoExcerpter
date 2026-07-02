"""Token usage tracking with daily limits and timezone-aware reset.

This module provides a thread-safe token tracker that:
- Counts total tokens used from OpenAI API responses
- Enforces configurable daily token limits
- Automatically resets at midnight in the local timezone
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
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config import app as config
from config.logger import setup_logger

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

# Singleton instance
_tracker_instance: DailyTokenTracker | None = None
_tracker_lock = threading.Lock()


class DailyTokenTracker:
    """
    Thread-safe daily token usage tracker with persistent state.

    Tracks token usage across API calls and enforces daily limits with
    automatic reset at midnight in the local timezone.
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

        # Load existing state from disk
        self._load_state()

        logger.info(
            f"Token tracker initialized: enabled={enabled}, "
            f"daily_limit={daily_limit:,}, "
            f"current_usage={self._tokens_used_today:,}"
        )

    def _get_current_date_str(self) -> str:
        """Get current date as string in YYYY-MM-DD format."""
        return datetime.now().strftime("%Y-%m-%d")

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
        persisted by :meth:`flush`.
        """
        now = time.time()
        if not force and (now - self._last_save_time) < _MIN_SAVE_INTERVAL_S:
            self._pending_save = True
            return
        try:
            state = {
                "date": self._current_date,
                "tokens_used": self._tokens_used_today,
                "last_updated": datetime.now().isoformat(),
            }

            # Write atomically using a temp file
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            # Replace original file
            temp_file.replace(self.state_file)
            self._last_save_time = now
            self._pending_save = False

        except Exception as e:
            logger.error(f"Error saving token state to {self.state_file}: {e}")

    def flush(self) -> None:
        """Persist any debounced pending state write."""
        with self._lock:
            if self._pending_save:
                self._save_state(force=True)

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
            self._save_state(force=True)

    def add_tokens(self, tokens: int) -> None:
        """
        Add tokens to the daily count.

        Args:
            tokens: Number of tokens to add.
        """
        if not self.enabled or tokens <= 0:
            return

        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            # Update the rolling per-call estimate used by try_reserve().
            self._ewma = self._alpha * tokens + (1.0 - self._alpha) * self._ewma
            self._save_state()

            logger.debug(
                f"Added {tokens:,} tokens. "
                f"Daily total: {self._tokens_used_today:,}/{self.daily_limit:,}"
            )

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

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))
            available = (
                self.daily_limit - self._tokens_used_today - self._tokens_reserved
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

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), self._current_estimate())
            available = (
                self.daily_limit - self._tokens_used_today - self._tokens_reserved
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

        Returns:
            Token count for current day.
        """
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
            remaining = self.daily_limit - self._tokens_used_today
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
        Get the number of seconds until the counter resets (midnight).

        Returns:
            Seconds until midnight (00:00:00 local time).
        """
        now = datetime.now()
        # Calculate next midnight
        tomorrow = now.date() + timedelta(days=1)
        midnight = datetime.combine(tomorrow, datetime.min.time())

        delta = midnight - now
        return int(delta.total_seconds())

    def get_reset_time(self) -> datetime:
        """
        Get the datetime when the counter will reset.

        Returns:
            Datetime of next midnight.
        """
        now = datetime.now()
        tomorrow = now.date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time())

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

        return {
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
        reset_time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    elapsed = 0
    try:
        while elapsed < seconds_until_reset:
            interval = min(1, max(0, seconds_until_reset - elapsed))
            time.sleep(interval)
            elapsed += interval
            if tracker.can_admit_page():
                logger.info("Token budget has reset. Resuming processing.")
                return True
        logger.info("Token budget has reset. Resuming processing.")
        return True
    except KeyboardInterrupt:
        logger.info("Token-limit wait cancelled by user.")
        return False
