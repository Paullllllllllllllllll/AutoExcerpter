"""Tests for llm/token_tracker.py."""

from __future__ import annotations

from pathlib import Path

import llm.token_tracker as token_tracker


class TestDailyTokenTracker:
    def test_add_tokens_persists_state(self, tmp_path: Path) -> None:
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

    def test_add_tokens_ignored_when_disabled_or_non_positive(
        self, tmp_path: Path
    ) -> None:
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

    def test_new_day_resets_counter(self, tmp_path: Path, monkeypatch) -> None:
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

    def test_can_use_tokens_logic(self, tmp_path: Path) -> None:
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

    def test_get_seconds_until_reset_is_reasonable(self, tmp_path: Path) -> None:
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
    ) -> None:
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


class TestChunkReservation:
    """Page-level reservation gate: try_reserve / release / EWMA estimate."""

    def test_disabled_returns_zero_and_never_denies(self, tmp_path: Path) -> None:
        t = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=False,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
        )
        assert t.try_reserve() == 0
        assert t.try_reserve(999_999) == 0
        t.release(0)  # no-op when disabled (must not raise)

    def test_reserve_within_and_beyond_budget(self, tmp_path: Path) -> None:
        t = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
        )
        # Seed EWMA is 10, so a bare reservation claims 10.
        assert t.try_reserve() == 10
        # An explicit estimate above the EWMA claims the estimate.
        assert t.try_reserve(50) == 50
        # reserved is now 60; remaining headroom is 40, so a 50 will not fit.
        assert t.try_reserve(50) is None
        # 40 fits exactly, bringing reserved to the limit.
        assert t.try_reserve(40) == 40
        assert t.try_reserve(1) is None

    def test_release_restores_headroom(self, tmp_path: Path) -> None:
        t = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
        )
        assert t.try_reserve(100) == 100
        assert t.try_reserve(1) is None
        t.release(100)
        assert t.try_reserve(50) == 50

    def test_committed_plus_reserved_never_exceeds_limit(self, tmp_path: Path) -> None:
        # smoothing=0 freezes the EWMA at the seed so this test isolates the
        # committed-plus-reserved arithmetic from estimate drift.
        t = token_tracker.DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
            estimate_smoothing=0.0,
        )
        t.add_tokens(60)  # committed usage
        assert t.try_reserve(30) == 30  # 60 + 30 = 90, fits
        assert t.try_reserve(20) is None  # 90 + 20 = 110, denied
        assert t.try_reserve(10) == 10  # 90 + 10 = 100, exact fit

    def test_add_tokens_does_not_update_ewma(self, tmp_path: Path) -> None:
        # Per-call committed usage must NOT move the EWMA (which is per-page).
        t = token_tracker.DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=100,
            estimate_smoothing=0.5,
        )
        t.add_tokens(1100)
        # EWMA still at the seed; the reservation reflects per-page cost only.
        assert t.try_reserve() == 100

    def test_record_page_usage_updates_ewma(self, tmp_path: Path) -> None:
        t = token_tracker.DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=100,
            estimate_smoothing=0.5,
        )
        # One page's total (e.g. 800 transcription + 300 summary) is one sample.
        # EWMA = 0.5 * 1100 + 0.5 * 100 = 600.
        t.record_page_usage(1100)
        assert t.try_reserve() == 600

    def test_record_page_usage_from_accumulated_calls(self, tmp_path: Path) -> None:
        # Summarize-on shape: two calls on the page accumulate; record feeds
        # the summed per-page total as ONE observation.
        t = token_tracker.DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=100,
            estimate_smoothing=1.0,  # EWMA snaps to the latest sample
        )
        t.add_tokens(800)  # transcription call
        t.add_tokens(300)  # summary call
        t.record_page_usage()  # total=1100 fed to EWMA
        assert t.try_reserve() == 1100

    def test_record_page_usage_single_call_when_summary_off(
        self, tmp_path: Path
    ) -> None:
        # Summarize-off shape: a page is one call; the EWMA tracks per-call,
        # which equals per-page here.
        t = token_tracker.DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=100,
            estimate_smoothing=1.0,
        )
        t.add_tokens(500)  # only the transcription call
        t.record_page_usage()
        assert t.try_reserve() == 500

    def test_reserve_uses_max_of_estimate_and_ewma(self, tmp_path: Path) -> None:
        t = token_tracker.DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=100,
        )
        assert t.try_reserve(5000) == 5000  # estimate above EWMA wins
        assert t.try_reserve(10) == 100  # estimate below EWMA floors at EWMA
