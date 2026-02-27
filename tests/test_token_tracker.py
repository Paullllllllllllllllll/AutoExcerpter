"""Tests for modules/token_tracker.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import modules.token_tracker as token_tracker


class TestDailyTokenTracker:
    def test_add_tokens_persists_state(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        tracker = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=state_file,
        )

        tracker.add_tokens(10)
        assert tracker.get_tokens_used_today() == 10
        assert state_file.exists()

        # Reload from disk
        tracker2 = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=state_file,
        )
        assert tracker2.get_tokens_used_today() == 10

    def test_add_tokens_ignored_when_disabled_or_non_positive(self, tmp_path: Path):
        tracker = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=False,
            state_file=tmp_path / "state.json",
        )
        tracker.add_tokens(10)
        assert tracker.get_tokens_used_today() == 0

        tracker_enabled = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "state2.json",
        )
        tracker_enabled.add_tokens(0)
        tracker_enabled.add_tokens(-5)
        assert tracker_enabled.get_tokens_used_today() == 0

    def test_new_day_resets_counter(self, tmp_path: Path, monkeypatch):
        tracker = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "state.json",
        )

        # Force an initial "day"
        monkeypatch.setattr(tracker, "_get_current_date_str", lambda: "2026-01-15")
        tracker._current_date = "2026-01-15"
        tracker.add_tokens(25)
        assert tracker.get_tokens_used_today() == 25

        # Advance to next day and ensure reset
        monkeypatch.setattr(tracker, "_get_current_date_str", lambda: "2026-01-16")
        assert tracker.get_tokens_used_today() == 0
        assert tracker.get_tokens_remaining() == 100

    def test_can_use_tokens_logic(self, tmp_path: Path):
        tracker = token_tracker.DailyTokenTracker(
            daily_limit=10,
            enabled=True,
            state_file=tmp_path / "state.json",
        )
        assert tracker.can_use_tokens() is True
        assert tracker.can_use_tokens(estimated_tokens=10) is True

        tracker.add_tokens(7)
        assert tracker.can_use_tokens(estimated_tokens=2) is True
        assert tracker.can_use_tokens(estimated_tokens=3) is True
        assert tracker.can_use_tokens(estimated_tokens=4) is False

    def test_get_seconds_until_reset_is_reasonable(self, tmp_path: Path):
        tracker = token_tracker.DailyTokenTracker(
            daily_limit=10,
            enabled=True,
            state_file=tmp_path / "state.json",
        )
        secs = tracker.get_seconds_until_reset()
        assert 0 < secs <= 86400


class TestGetTokenTrackerSingleton:
    def test_get_token_tracker_returns_singleton_and_uses_config(
        self, tmp_path: Path, monkeypatch
    ):
        # Reset singleton state
        monkeypatch.setattr(token_tracker, "_tracker_instance", None)

        # Ensure we don't write to cwd
        monkeypatch.setattr(
            token_tracker, "_TOKEN_TRACKER_FILE", tmp_path / "state.json"
        )

        monkeypatch.setattr(
            token_tracker.config, "DAILY_TOKEN_LIMIT", 123, raising=False
        )
        monkeypatch.setattr(
            token_tracker.config, "DAILY_TOKEN_LIMIT_ENABLED", True, raising=False
        )

        t1 = token_tracker.get_token_tracker()
        t2 = token_tracker.get_token_tracker()

        assert t1 is t2
        assert t1.daily_limit == 123
        assert t1.enabled is True
