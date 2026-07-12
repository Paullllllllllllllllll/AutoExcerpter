"""Tests for the shared cross-tool token-budget integration.

These exercise ``DailyTokenTracker``'s optional ``shared_enabled`` path: the
vendored :mod:`llm.shared_ledger` wired in as the daily budget's persistence and
combined-total source. The DISABLED path (default) is covered by
``tests/test_llm_token_tracker.py`` and must stay bit-for-bit today's behaviour;
here every tracker is constructed with an explicit scratch ledger directory so
nothing touches ``~/.chronopipeline``.

This repo is fully threaded with no asyncio event loop, so ledger syncs run
inline and the scenarios below can drive them deterministically.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import llm.token_tracker as tt
from llm.shared_ledger import (
    LEDGER_FILENAME,
    BucketKey,
    SharedTokenLedger,
    UsageSnapshot,
    _today,
)
from llm.token_tracker import DailyTokenTracker, wait_for_token_reset


def _write_ledger(ledger_dir: Path, tools: dict[str, int]) -> None:
    """Write the ledger JSON directly (simulates another process/tool)."""
    ledger_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "date": _today(),
        "tools": tools,
        "last_updated": datetime.now().isoformat(),
    }
    (ledger_dir / LEDGER_FILENAME).write_text(json.dumps(data), encoding="utf-8")


def _make(
    *,
    ledger_dir: Path,
    state_file: Path,
    daily_limit: int = 10_000_000,
    enabled: bool = True,
    shared_enabled: bool = True,
    seed: int = 1,
    smoothing: float = 0.0,
) -> DailyTokenTracker:
    return DailyTokenTracker(
        daily_limit=daily_limit,
        enabled=enabled,
        state_file=state_file,
        chunk_estimate_seed=seed,
        estimate_smoothing=smoothing,
        shared_enabled=shared_enabled,
        shared_ledger_dir=ledger_dir,
    )


class TestDisabledUnchanged:
    def test_no_ledger_file_and_private_state_used(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "state.json"
        t = _make(
            ledger_dir=ledger_dir,
            state_file=state_file,
            daily_limit=100,
            shared_enabled=False,
        )
        t.add_tokens(50)
        t.flush()

        # Disabled: zero ledger I/O, private state file carries the count.
        assert not (ledger_dir / LEDGER_FILENAME).exists()
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 50
        assert t.get_tokens_used_today() == 50
        # No shared-budget keys leak into stats when disabled.
        assert "shared_budget_enabled" not in t.get_stats()


class TestEnabledPersistence:
    def test_add_then_flush_pushes_delta(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "s.json"
        t = _make(ledger_dir=ledger_dir, state_file=state_file)

        t.add_tokens(1234)
        t.flush()  # forces the accumulated delta into the ledger

        ledger = SharedTokenLedger("autoexcerpter", ledger_dir=ledger_dir)
        assert ledger.read_breakdown() == {"autoexcerpter": 1234}
        assert ledger.read_combined() == 1234
        # Combined view (only this tool present) equals our own usage.
        assert t.get_tokens_used_today() == 1234
        assert t.get_own_tokens_used_today() == 1234
        # The private state file is NOT used as persistence while enabled.
        assert not state_file.exists()

    def test_stats_expose_breakdown(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(700)
        t = _make(ledger_dir=ledger_dir, state_file=tmp_path / "s.json")
        t.add_tokens(300)
        t.flush()

        stats = t.get_stats()
        assert stats["shared_budget_enabled"] is True
        assert stats["shared_budget_degraded"] is False
        assert stats["own_tokens_used_today"] == 300
        assert stats["combined_tokens_used_today"] == 1000
        assert stats["tokens_used_today"] == 1000  # enforced figure is combined
        assert stats["shared_breakdown"] == {
            "autoexcerpter": 300,
            "chronotranscriber": 700,
        }


class TestForeignUsageClosesGate:
    def test_combined_total_enforced(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        # Another tool has already burned most of the shared daily budget.
        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(98)

        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=100,
        )
        # Seeded at init: combined baseline is the foreign 98.
        assert t.get_tokens_used_today() == 98
        assert t.get_own_tokens_used_today() == 0

        # A small own contribution tips the COMBINED total over the cap.
        t.add_tokens(5)
        assert t.is_limit_reached() is True
        # Admission refreshes near the cap and denies both reservation gates.
        assert t.try_reserve(1) is None
        assert t.can_admit_page(1) is False


class TestForcedAndDebouncedRefresh:
    def test_eager_refresh_near_limit(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=100,
        )
        t.add_tokens(85)
        t.flush()  # own field = 85; cached combined = 85

        # Another process raises the combined total behind our back.
        SharedTokenLedger("chronominer", ledger_dir=ledger_dir).sync(10)

        # Cached value is stale (still 85 -> remaining 15).
        assert t.get_tokens_remaining() == 15
        # Cached combined (85) exceeds 80% of 100 -> try_reserve refreshes first.
        t.try_reserve(1)
        assert t.get_tokens_remaining() == 5  # 100 - refreshed combined 95

    def test_below_80_picked_up_on_debounced_sync(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=1000,
        )
        t.add_tokens(100)
        t.flush()  # own field 100; cached combined 100 (< 80% of 1000)

        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(200)

        # Make the debounce interval look elapsed so the next add syncs inline.
        t._last_ledger_sync_monotonic = 0.0
        t.add_tokens(1)  # debounced sync fires inline, refreshing combined

        assert t.get_tokens_used_today() == 301  # 101 own + 200 foreign
        assert t.get_own_tokens_used_today() == 101


class TestSeeding:
    def test_seeds_from_legacy_private_state(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "s.json"
        # Legacy private state for TODAY with prior usage.
        state_file.write_text(
            json.dumps({"date": _today(), "tokens_used": 500}),
            encoding="utf-8",
        )

        _make(ledger_dir=ledger_dir, state_file=state_file)

        ledger = SharedTokenLedger("autoexcerpter", ledger_dir=ledger_dir)
        assert ledger.read_breakdown() == {"autoexcerpter": 500}

    def test_seed_takes_max_of_ledger_and_private(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "s.json"
        # Ledger already carries a higher own field than the private file.
        SharedTokenLedger("autoexcerpter", ledger_dir=ledger_dir).sync(900)
        state_file.write_text(
            json.dumps({"date": _today(), "tokens_used": 500}),
            encoding="utf-8",
        )

        t = _make(ledger_dir=ledger_dir, state_file=state_file)
        # Seed adopts max(ledger 900, private 500) without double-counting.
        assert t.get_own_tokens_used_today() == 900
        assert t.get_tokens_used_today() == 900

    def test_repeated_init_does_not_double_seed(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "s.json"
        state_file.write_text(
            json.dumps({"date": _today(), "tokens_used": 500}),
            encoding="utf-8",
        )

        _make(ledger_dir=ledger_dir, state_file=state_file)
        _make(ledger_dir=ledger_dir, state_file=state_file)  # second process

        ledger = SharedTokenLedger("autoexcerpter", ledger_dir=ledger_dir)
        assert ledger.read_breakdown() == {"autoexcerpter": 500}


class _FakeLedger:
    """Ledger stand-in (v2) whose I/O can be toggled to degrade and recover.

    Tracks this tool's own total and per-bucket rows so the degraded-mode
    per-bucket preservation can be asserted precisely.
    """

    def __init__(self) -> None:
        self.field = 0
        self.fail = True
        self.foreign = 0
        self.buckets: dict[BucketKey, int] = {}

    def _snapshot(self) -> UsageSnapshot:
        attributed = {
            b: v for b, v in self.buckets.items() if b.provider != "unattributed"
        }
        return UsageSnapshot(
            combined=self.field + self.foreign,
            own_total=self.field,
            buckets=dict(attributed),
            own_buckets=dict(self.buckets),
        )

    def seed_usage(
        self, own_total: int, own_buckets: dict[BucketKey, int] | None = None
    ) -> UsageSnapshot | None:
        if self.fail:
            return None
        self.field = max(self.field, int(own_total))
        for bucket, amount in (own_buckets or {}).items():
            self.buckets[bucket] = max(self.buckets.get(bucket, 0), int(amount))
        return self._snapshot()

    def sync_usage(self, deltas: dict[BucketKey, int]) -> UsageSnapshot | None:
        if self.fail:
            return None
        for bucket, amount in deltas.items():
            self.buckets[bucket] = self.buckets.get(bucket, 0) + max(0, int(amount))
        self.field += sum(max(0, int(v)) for v in deltas.values())
        return self._snapshot()

    def read_breakdown(self) -> dict[str, int] | None:
        if self.fail:
            return None
        return {"autoexcerpter": self.field}

    def read_combined(self) -> int | None:
        if self.fail:
            return None
        return self.field + self.foreign


class TestDegradedMode:
    def test_accumulates_while_degraded_then_lands_on_recovery(
        self, tmp_path: Path
    ) -> None:
        t = _make(
            ledger_dir=tmp_path / "ledger",
            state_file=tmp_path / "s.json",
            daily_limit=10_000_000,
        )
        # Swap in a degraded ledger and reset shared state to pre-seed.
        fake = _FakeLedger()
        with t._lock:
            t._ledger = fake  # type: ignore[assignment]
            t._seeded = False
            t._combined_total = 0
            t._unsynced_deltas = {}
            t._ledger_degraded = False
            t._last_ledger_sync_monotonic = 0.0

        t.add_tokens(100)  # debounced sync fires; seed fails -> degraded
        assert t._ledger_degraded is True
        # Standalone fallback keeps the tracker fully functional.
        assert t.get_tokens_used_today() == 100
        assert t.is_limit_reached() is False

        t.add_tokens(50)  # delta keeps accumulating while degraded
        assert t.get_own_tokens_used_today() == 150

        # Ledger recovers; a forced sync must land the full accumulated amount.
        fake.fail = False
        t.sync_ledger_now()

        assert fake.field == 150  # accumulated 100 + 50 landed
        assert t._ledger_degraded is False
        assert t.get_tokens_used_today() == 150

    def test_per_bucket_deltas_preserved_across_degradation(
        self, tmp_path: Path
    ) -> None:
        """Per-bucket unsynced deltas survive a degraded window and all land."""
        t = _make(
            ledger_dir=tmp_path / "ledger",
            state_file=tmp_path / "s.json",
            daily_limit=10_000_000,
        )
        fake = _FakeLedger()
        with t._lock:
            t._ledger = fake  # type: ignore[assignment]
            t._seeded = True  # skip seed so we exercise the sync path
            t._combined_total = 0
            t._bucket_totals = {}
            t._unsynced_deltas = {}
            t._ledger_degraded = False
            t._last_ledger_sync_monotonic = 0.0

        b_open = BucketKey("openai", "OPENAI_API_KEY", "large")
        t.add_tokens(100, provider="openai", key_env="OPENAI_API_KEY", model="gpt-5")
        # Degraded: nothing pushed, per-bucket delta retained.
        assert t._ledger_degraded is True
        assert t._unsynced_deltas.get(b_open) == 100

        t.add_tokens(40, provider="openai", key_env="OPENAI_API_KEY", model="gpt-5")
        t.add_tokens(7, provider="google", key_env="GOOGLE_API_KEY", model="gemini")
        assert t._unsynced_deltas.get(b_open) == 140

        fake.fail = False
        t.sync_ledger_now()

        # Every degraded-window delta lands on its own bucket, none lost.
        assert fake.buckets[b_open] == 140
        assert fake.field == 147
        assert not t._unsynced_deltas  # fully drained


class TestWaitLoopForcedSync:
    def test_foreign_reset_unblocks_within_a_poll(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        ledger_dir = tmp_path / "ledger"
        # Another tool has exhausted the shared budget.
        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(120)

        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=100,
        )
        monkeypatch.setattr(tt, "_tracker_instance", t)
        assert t.is_limit_reached() is True  # combined 120 > 100

        # Keep the configured limit fixed so only the ledger can unblock us.
        monkeypatch.setattr(tt.config, "reload_daily_token_limit", lambda: None)

        calls = {"n": 0}

        def fake_sleep(_seconds: float) -> None:
            calls["n"] += 1
            # Simulate the other tool resetting (its field drops to 0).
            _write_ledger(ledger_dir, {"autoexcerpter": 0, "chronotranscriber": 0})

        monkeypatch.setattr("llm.token_tracker.time.sleep", fake_sleep)

        result = wait_for_token_reset()

        assert result is True
        assert calls["n"] >= 1
        assert t.is_limit_reached() is False
