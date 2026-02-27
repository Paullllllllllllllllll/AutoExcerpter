"""Extended tests for modules/token_tracker.py - coverage gap filling.

Covers:
- _load_state: same day restore, different day reset, corrupted file, missing fields
- _save_state: normal save, write error
- _check_and_reset_if_new_day: same day no-op, different day resets
- get_tokens_remaining: disabled, enabled with usage
- is_limit_reached: disabled, at limit, over limit, under limit
- can_use_tokens: disabled, enough tokens, not enough, estimated_tokens
- get_seconds_until_reset: returns positive integer
- get_reset_time: returns tomorrow midnight
- get_usage_percentage: disabled, zero limit, normal usage
- get_stats: returns comprehensive dict
- get_token_tracker singleton: first call creates, second returns same
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import modules.token_tracker as token_tracker
from modules.token_tracker import DailyTokenTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    """Reset the module-level singleton before every test."""
    monkeypatch.setattr(token_tracker, "_tracker_instance", None)


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    """Return a fresh state-file path inside tmp_path."""
    return tmp_path / "token_state.json"


def _make_tracker(
    state_file: Path,
    daily_limit: int = 1_000_000,
    enabled: bool = True,
) -> DailyTokenTracker:
    """Shorthand for building a tracker pointing at *state_file*."""
    return DailyTokenTracker(
        daily_limit=daily_limit,
        enabled=enabled,
        state_file=state_file,
    )


# ============================================================================
# _load_state
# ============================================================================
class TestLoadState:
    """Tests for DailyTokenTracker._load_state()."""

    def test_no_state_file_initializes_fresh(self, state_file: Path):
        """When no state file exists, tracker starts with zero usage."""
        tracker = _make_tracker(state_file)
        assert tracker._tokens_used_today == 0
        assert tracker._current_date == datetime.now().strftime("%Y-%m-%d")

    def test_same_day_restores_tokens(self, state_file: Path):
        """State file from the same day restores the token count."""
        today = datetime.now().strftime("%Y-%m-%d")
        state = {
            "date": today,
            "tokens_used": 42_000,
            "last_updated": datetime.now().isoformat(),
        }
        state_file.write_text(json.dumps(state), encoding="utf-8")

        tracker = _make_tracker(state_file)
        assert tracker._tokens_used_today == 42_000
        assert tracker._current_date == today

    def test_different_day_resets_counter(self, state_file: Path):
        """State file from a previous day resets the token count to zero."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        state = {
            "date": yesterday,
            "tokens_used": 99_999,
            "last_updated": datetime.now().isoformat(),
        }
        state_file.write_text(json.dumps(state), encoding="utf-8")

        tracker = _make_tracker(state_file)
        assert tracker._tokens_used_today == 0
        assert tracker._current_date == datetime.now().strftime("%Y-%m-%d")

    def test_corrupted_json_initializes_fresh(self, state_file: Path):
        """Corrupted (non-JSON) state file falls back to fresh state."""
        state_file.write_text("{{{NOT VALID JSON", encoding="utf-8")

        tracker = _make_tracker(state_file)
        assert tracker._tokens_used_today == 0

    def test_missing_fields_treated_as_defaults(self, state_file: Path):
        """State file missing 'date' or 'tokens_used' uses defaults."""
        # Write JSON with no expected keys
        state_file.write_text(json.dumps({"extra_key": "value"}), encoding="utf-8")

        tracker = _make_tracker(state_file)
        # "date" defaults to "" which differs from today, so counter resets to 0
        assert tracker._tokens_used_today == 0


# ============================================================================
# _save_state
# ============================================================================
class TestSaveState:
    """Tests for DailyTokenTracker._save_state()."""

    def test_normal_save_writes_valid_json(self, state_file: Path):
        """_save_state writes a valid JSON file with expected keys."""
        tracker = _make_tracker(state_file)
        tracker._tokens_used_today = 500
        tracker._save_state()

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 500
        assert "date" in data
        assert "last_updated" in data

    def test_save_state_handles_write_error(self, tmp_path: Path):
        """_save_state logs an error but does not crash on write failure."""
        # Use a directory path as the state file so that writing fails
        bad_path = tmp_path / "nonexistent_dir" / "subdir" / "state.json"
        tracker = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=bad_path,
        )
        # Manually call save; should not raise
        tracker._save_state()


# ============================================================================
# _check_and_reset_if_new_day
# ============================================================================
class TestCheckAndResetIfNewDay:
    """Tests for DailyTokenTracker._check_and_reset_if_new_day()."""

    def test_same_day_no_reset(self, state_file: Path, monkeypatch):
        """No reset occurs when the date has not changed."""
        tracker = _make_tracker(state_file)
        tracker._current_date = "2026-02-26"
        tracker._tokens_used_today = 200
        monkeypatch.setattr(tracker, "_get_current_date_str", lambda: "2026-02-26")

        tracker._check_and_reset_if_new_day()

        assert tracker._tokens_used_today == 200

    def test_different_day_resets(self, state_file: Path, monkeypatch):
        """Counter resets to zero when a new day is detected."""
        tracker = _make_tracker(state_file)
        tracker._current_date = "2026-02-25"
        tracker._tokens_used_today = 500
        monkeypatch.setattr(tracker, "_get_current_date_str", lambda: "2026-02-26")

        tracker._check_and_reset_if_new_day()

        assert tracker._tokens_used_today == 0
        assert tracker._current_date == "2026-02-26"


# ============================================================================
# get_tokens_remaining
# ============================================================================
class TestGetTokensRemaining:
    """Tests for DailyTokenTracker.get_tokens_remaining()."""

    def test_disabled_returns_full_limit(self, state_file: Path):
        """When disabled, remaining equals the daily limit (unlimited)."""
        tracker = _make_tracker(state_file, daily_limit=5_000, enabled=False)
        assert tracker.get_tokens_remaining() == 5_000

    def test_enabled_with_usage(self, state_file: Path):
        """When enabled, remaining is limit minus usage."""
        tracker = _make_tracker(state_file, daily_limit=1_000, enabled=True)
        tracker.add_tokens(300)
        assert tracker.get_tokens_remaining() == 700

    def test_over_limit_returns_zero(self, state_file: Path):
        """When usage exceeds limit, remaining is clamped to zero."""
        tracker = _make_tracker(state_file, daily_limit=100, enabled=True)
        # Bypass the enabled guard by directly mutating internal state
        tracker._tokens_used_today = 200
        assert tracker.get_tokens_remaining() == 0


# ============================================================================
# is_limit_reached
# ============================================================================
class TestIsLimitReached:
    """Tests for DailyTokenTracker.is_limit_reached()."""

    def test_disabled_never_reached(self, state_file: Path):
        """When disabled, limit is never reached."""
        tracker = _make_tracker(state_file, daily_limit=0, enabled=False)
        assert tracker.is_limit_reached() is False

    def test_under_limit(self, state_file: Path):
        """Under the limit returns False."""
        tracker = _make_tracker(state_file, daily_limit=100, enabled=True)
        tracker.add_tokens(50)
        assert tracker.is_limit_reached() is False

    def test_at_limit(self, state_file: Path):
        """Exactly at the limit returns True."""
        tracker = _make_tracker(state_file, daily_limit=100, enabled=True)
        tracker.add_tokens(100)
        assert tracker.is_limit_reached() is True

    def test_over_limit(self, state_file: Path):
        """Over the limit returns True."""
        tracker = _make_tracker(state_file, daily_limit=100, enabled=True)
        tracker._tokens_used_today = 200
        assert tracker.is_limit_reached() is True


# ============================================================================
# can_use_tokens
# ============================================================================
class TestCanUseTokens:
    """Tests for DailyTokenTracker.can_use_tokens()."""

    def test_disabled_always_true(self, state_file: Path):
        """When disabled, can_use_tokens always returns True."""
        tracker = _make_tracker(state_file, daily_limit=0, enabled=False)
        assert tracker.can_use_tokens() is True
        assert tracker.can_use_tokens(estimated_tokens=999_999) is True

    def test_enough_tokens_available(self, state_file: Path):
        """Returns True when enough tokens remain."""
        tracker = _make_tracker(state_file, daily_limit=1_000, enabled=True)
        tracker.add_tokens(500)
        assert tracker.can_use_tokens(estimated_tokens=500) is True

    def test_not_enough_tokens(self, state_file: Path):
        """Returns False when estimated tokens exceed remaining."""
        tracker = _make_tracker(state_file, daily_limit=1_000, enabled=True)
        tracker.add_tokens(900)
        assert tracker.can_use_tokens(estimated_tokens=200) is False

    def test_zero_estimated_checks_any_remaining(self, state_file: Path):
        """With estimated_tokens=0, checks that any tokens remain."""
        tracker = _make_tracker(state_file, daily_limit=10, enabled=True)
        assert tracker.can_use_tokens() is True

        tracker.add_tokens(10)
        assert tracker.can_use_tokens() is False


# ============================================================================
# get_seconds_until_reset
# ============================================================================
class TestGetSecondsUntilReset:
    """Tests for DailyTokenTracker.get_seconds_until_reset()."""

    def test_returns_positive_integer(self, state_file: Path):
        """Value is a positive integer no greater than 86400."""
        tracker = _make_tracker(state_file)
        secs = tracker.get_seconds_until_reset()
        assert isinstance(secs, int)
        assert 0 < secs <= 86_400

    def test_consistent_with_mocked_time(self, state_file: Path):
        """With a fixed time, seconds until reset is predictable."""
        fake_now = datetime(2026, 2, 26, 22, 0, 0)
        with patch("modules.token_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.min = datetime.min
            mock_dt.combine = datetime.combine
            # We also need side_effect for strftime inside _get_current_date_str
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            tracker = _make_tracker(state_file)
            secs = tracker.get_seconds_until_reset()
            # From 22:00 to midnight is 7200 seconds
            assert secs == 7200


# ============================================================================
# get_reset_time
# ============================================================================
class TestGetResetTime:
    """Tests for DailyTokenTracker.get_reset_time()."""

    def test_returns_tomorrow_midnight(self, state_file: Path):
        """Reset time is midnight of the next day."""
        fake_now = datetime(2026, 2, 26, 15, 30, 0)
        with patch("modules.token_tracker.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.min = datetime.min
            mock_dt.combine = datetime.combine

            tracker = _make_tracker(state_file)
            reset_time = tracker.get_reset_time()
            assert reset_time == datetime(2026, 2, 27, 0, 0, 0)


# ============================================================================
# get_usage_percentage
# ============================================================================
class TestGetUsagePercentage:
    """Tests for DailyTokenTracker.get_usage_percentage()."""

    def test_disabled_returns_zero(self, state_file: Path):
        """When disabled, usage percentage is 0.0."""
        tracker = _make_tracker(state_file, enabled=False)
        assert tracker.get_usage_percentage() == 0.0

    def test_zero_limit_returns_zero(self, state_file: Path):
        """When daily_limit is zero, usage percentage is 0.0."""
        tracker = _make_tracker(state_file, daily_limit=0, enabled=True)
        assert tracker.get_usage_percentage() == 0.0

    def test_normal_usage(self, state_file: Path):
        """50% usage returns 50.0."""
        tracker = _make_tracker(state_file, daily_limit=1_000, enabled=True)
        tracker.add_tokens(500)
        assert tracker.get_usage_percentage() == pytest.approx(50.0)

    def test_over_100_percent(self, state_file: Path):
        """Usage exceeding the limit yields > 100%."""
        tracker = _make_tracker(state_file, daily_limit=100, enabled=True)
        tracker._tokens_used_today = 150
        assert tracker.get_usage_percentage() == pytest.approx(150.0)


# ============================================================================
# get_stats
# ============================================================================
class TestGetStats:
    """Tests for DailyTokenTracker.get_stats()."""

    def test_returns_comprehensive_dict(self, state_file: Path):
        """get_stats returns a dict with all expected keys."""
        tracker = _make_tracker(state_file, daily_limit=10_000, enabled=True)
        tracker.add_tokens(2_500)

        stats = tracker.get_stats()

        assert stats["enabled"] is True
        assert stats["daily_limit"] == 10_000
        assert stats["tokens_used_today"] == 2_500
        assert stats["tokens_remaining"] == 7_500
        assert stats["usage_percentage"] == pytest.approx(25.0)
        assert stats["limit_reached"] is False
        assert isinstance(stats["seconds_until_reset"], int)
        assert "reset_time" in stats
        assert "current_date" in stats

    def test_disabled_stats(self, state_file: Path):
        """get_stats reflects disabled state correctly."""
        tracker = _make_tracker(state_file, daily_limit=10_000, enabled=False)

        stats = tracker.get_stats()

        assert stats["enabled"] is False
        assert stats["limit_reached"] is False
        assert stats["usage_percentage"] == 0.0


# ============================================================================
# get_token_tracker singleton
# ============================================================================
class TestGetTokenTrackerSingleton:
    """Tests for the module-level get_token_tracker() singleton."""

    def test_first_call_creates_instance(self, monkeypatch, tmp_path: Path):
        """First call creates a new DailyTokenTracker instance."""
        monkeypatch.setattr(token_tracker, "_TOKEN_TRACKER_FILE", tmp_path / "s.json")
        monkeypatch.setattr(
            token_tracker.config, "DAILY_TOKEN_LIMIT", 500, raising=False
        )
        monkeypatch.setattr(
            token_tracker.config, "DAILY_TOKEN_LIMIT_ENABLED", True, raising=False
        )

        t = token_tracker.get_token_tracker()
        assert isinstance(t, DailyTokenTracker)
        assert t.daily_limit == 500
        assert t.enabled is True

    def test_second_call_returns_same_instance(self, monkeypatch, tmp_path: Path):
        """Second call returns the same singleton instance."""
        monkeypatch.setattr(token_tracker, "_TOKEN_TRACKER_FILE", tmp_path / "s.json")
        monkeypatch.setattr(
            token_tracker.config, "DAILY_TOKEN_LIMIT", 500, raising=False
        )
        monkeypatch.setattr(
            token_tracker.config, "DAILY_TOKEN_LIMIT_ENABLED", True, raising=False
        )

        t1 = token_tracker.get_token_tracker()
        t2 = token_tracker.get_token_tracker()
        assert t1 is t2
