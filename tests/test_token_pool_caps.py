"""Tests for per-key-pool token accounting and enforcement.

Covers the primary (per-key-pool cap) and secondary (combined, scoped) gates on
:class:`DailyTokenTracker`, bucket stamping per role, private-state bucket
persistence with legacy adoption, and the ``config.app`` parsing/reload of the
new ``daily_token_limit.scope`` / ``per_key_pool_caps`` settings. Every tracker
is standalone (no shared ledger) unless noted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.shared_ledger import UNATTRIBUTED_BUCKET, BucketKey
from llm.token_tracker import DailyTokenTracker

# A tiny cap map so a couple of add_tokens exhausts a pool deterministically.
_CAPS = {"openai": {"large": 1000, "small": 5000}}


def _tracker(state_file: Path, **kw: object) -> DailyTokenTracker:
    defaults: dict[str, object] = {
        "daily_limit": 10**9,  # combined guard effectively out of the way
        "enabled": True,
        "state_file": state_file,
        "chunk_estimate_seed": 1,
        "estimate_smoothing": 0.0,
        "pool_caps": _CAPS,
    }
    defaults.update(kw)
    return DailyTokenTracker(**defaults)  # type: ignore[arg-type]


# ============================================================================
# Bucket stamping per role
# ============================================================================
class TestBucketStamping:
    def test_stamp_lands_on_derived_pool_bucket(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path / "s.json")
        t.add_tokens(300, provider="openai", key_env="OPENAI_API_KEY", model="gpt-5")
        b = BucketKey("openai", "OPENAI_API_KEY", "large")
        assert t._own_buckets[b] == 300
        # A gpt-*-mini model on the same key lands in the SMALL pool bucket.
        t.add_tokens(
            50, provider="openai", key_env="OPENAI_API_KEY", model="gpt-5-mini"
        )
        assert t._own_buckets[BucketKey("openai", "OPENAI_API_KEY", "small")] == 50

    def test_unstamped_usage_is_unattributed(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path / "s.json")
        t.add_tokens(120)  # no stamp
        assert t._own_buckets[UNATTRIBUTED_BUCKET] == 120
        # A non-openai provider derives no pool (pool None) but keeps its bucket.
        t.add_tokens(9, provider="google", key_env="GOOGLE_API_KEY", model="gemini")
        assert t._own_buckets[BucketKey("google", "GOOGLE_API_KEY", None)] == 9


# ============================================================================
# Primary per-key-pool gate
# ============================================================================
class TestPoolGate:
    def test_pool_cap_blocks_when_exhausted(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path / "s.json")
        stamp = {"provider": "openai", "key_env": "OPENAI_API_KEY", "model": "gpt-5"}
        t.add_tokens(1000, **stamp)  # exactly the large cap
        assert t.try_reserve(1, **stamp) is None
        assert t.can_admit_page(1, **stamp) is False

    def test_two_keys_enforced_independently(self, tmp_path: Path) -> None:
        """Key-1's exhausted pool must not block key-2's fresh pool."""
        t = _tracker(tmp_path / "s.json")
        k1 = {"provider": "openai", "key_env": "OPENAI_API_KEY", "model": "gpt-5"}
        k2 = {"provider": "openai", "key_env": "OPENAI_API_KEY_2", "model": "gpt-5"}
        t.add_tokens(1000, **k1)  # exhaust key-1 large pool
        assert t.try_reserve(1, **k1) is None  # key-1 blocked
        assert t.try_reserve(1, **k2) == 1  # key-2 admits

    def test_paid_summary_does_not_block_free_transcription(
        self, tmp_path: Path
    ) -> None:
        """An exhausted paid (pooled) summary key must not block a free/local
        transcription endpoint (pool None) in the same process."""
        t = _tracker(tmp_path / "s.json")
        summary = {
            "provider": "openai",
            "key_env": "SUM_KEY",  # remapped openai key -> still openai/large pool
            "model": "gpt-5",
        }
        # Give the summary key its own openai pool budget and exhaust it.
        t.set_pool_settings(pool_caps={"openai": {"large": 500, "small": 5000}})
        t.add_tokens(500, **summary)
        assert t.try_reserve(1, **summary) is None  # summary pool exhausted

        transcription = {
            "provider": "custom",  # local endpoint -> derive_pool None
            "key_env": "TRANS_KEY",
            "model": "local-vlm",
        }
        # The free transcription bucket (pool None) is never blocked.
        assert t.try_reserve(1, **transcription) == 1
        assert t.can_admit_page(1, **transcription) is True

    def test_pool_caps_disabled_never_blocks(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path / "s.json", pool_caps_enabled=False)
        stamp = {"provider": "openai", "key_env": "OPENAI_API_KEY", "model": "gpt-5"}
        t.add_tokens(10_000, **stamp)  # far above the 1000 large cap
        assert t.try_reserve(1, **stamp) == 1  # gate off -> admitted


# ============================================================================
# Secondary combined gate + scope
# ============================================================================
class TestCombinedScope:
    def test_pooled_scope_never_blocks_stamped_pool_none(self, tmp_path: Path) -> None:
        """scope=pooled: a stamped pool-None bucket bypasses the combined cap."""
        t = _tracker(tmp_path / "s.json", daily_limit=100, combined_scope="pooled")
        # Burn the combined budget with unstamped usage.
        t.add_tokens(100)
        stamp = {"provider": "custom", "key_env": "TRANS_KEY", "model": "local"}
        # Pool None + pooled scope -> combined gate does not apply.
        assert t.try_reserve(1, **stamp) == 1

    def test_all_scope_blocks_stamped_pool_none(self, tmp_path: Path) -> None:
        """scope=all: every stamped bucket obeys the combined cap (legacy)."""
        t = _tracker(tmp_path / "s.json", daily_limit=100, combined_scope="all")
        t.add_tokens(100)
        stamp = {"provider": "custom", "key_env": "TRANS_KEY", "model": "local"}
        assert t.try_reserve(1, **stamp) is None

    def test_unstamped_always_obeys_combined(self, tmp_path: Path) -> None:
        """Unstamped usage keeps legacy combined-only semantics under pooled."""
        t = _tracker(tmp_path / "s.json", daily_limit=100, combined_scope="pooled")
        t.add_tokens(100)
        assert t.try_reserve(1) is None  # combined exhausted, unstamped blocked


# ============================================================================
# Private-state bucket persistence + legacy adoption
# ============================================================================
class TestPrivateStatePersistence:
    def test_buckets_persist_and_reload(self, tmp_path: Path) -> None:
        state_file = tmp_path / "s.json"
        t = _tracker(state_file)
        t.add_tokens(300, provider="openai", key_env="OPENAI_API_KEY", model="gpt-5")
        t.add_tokens(20)  # unstamped
        t.flush()

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 320
        assert data["buckets"]["openai|OPENAI_API_KEY|large"] == 300
        assert data["buckets"]["unattributed|unattributed|"] == 20

        # A fresh tracker restores the per-bucket split, so the pool gate still
        # sees prior usage.
        t2 = _tracker(state_file)
        b = BucketKey("openai", "OPENAI_API_KEY", "large")
        assert t2._own_buckets[b] == 300
        stamp = {"provider": "openai", "key_env": "OPENAI_API_KEY", "model": "gpt-5"}
        # 300 already used against a 1000 cap: a 701 reservation would exceed it.
        assert t2.try_reserve(701, **stamp) is None
        assert t2.try_reserve(700, **stamp) == 700

    def test_legacy_state_without_buckets_adopts_unattributed(
        self, tmp_path: Path
    ) -> None:
        from llm.shared_ledger import _today

        state_file = tmp_path / "s.json"
        state_file.write_text(
            json.dumps({"date": _today(), "tokens_used": 500}), encoding="utf-8"
        )
        t = _tracker(state_file)
        assert t._own_buckets == {UNATTRIBUTED_BUCKET: 500}
        assert t.get_own_tokens_used_today() == 500


# ============================================================================
# Stats surface the pool buckets
# ============================================================================
class TestPoolStats:
    def test_stats_report_pool_used_and_remaining(self, tmp_path: Path) -> None:
        t = _tracker(tmp_path / "s.json")
        t.add_tokens(400, provider="openai", key_env="OPENAI_API_KEY", model="gpt-5")
        stats = t.get_stats()
        assert stats["per_key_pool_caps_enabled"] is True
        assert stats["combined_scope"] == "pooled"
        rows = {r["key_env"]: r for r in stats["pool_buckets"]}
        assert rows["OPENAI_API_KEY"]["used"] == 400
        assert rows["OPENAI_API_KEY"]["cap"] == 1000
        assert rows["OPENAI_API_KEY"]["remaining"] == 600


# ============================================================================
# Config parsing / reload
# ============================================================================
class TestConfigParsing:
    def test_parse_pool_caps_bare_int_unchanged(self) -> None:
        """Bare-int pool entries keep the original cap-only behavior."""
        from config.app import _parse_pool_caps

        caps, models = _parse_pool_caps(
            {"enabled": True, "openai": {"small": "9_750_000", "large": 975_000}}
        )
        assert caps == {"openai": {"small": 9_750_000, "large": 975_000}}
        assert models == {}

    def test_parse_pool_caps_mapping_form(self) -> None:
        """Mapping entries yield caps and/or model prefix definitions."""
        from config.app import _parse_pool_caps

        caps, models = _parse_pool_caps(
            {
                "openai": {
                    "small": 9_750_000,  # bare int alongside mapping forms
                    "large": {"cap": "975_000"},
                },
                "myhost": {
                    "standard": {"cap": 5_000_000, "models": ["my-model", " x "]},
                    "tracked_only": {"models": ["other-model"]},
                },
            }
        )
        assert caps == {
            "openai": {"small": 9_750_000, "large": 975_000},
            "myhost": {"standard": 5_000_000},
        }
        assert models == {
            "myhost": {
                "standard": ["my-model", "x"],
                "tracked_only": ["other-model"],
            }
        }

    def test_parse_pool_caps_non_dict_returns_empty(self) -> None:
        from config.app import _parse_pool_caps

        assert _parse_pool_caps(None) == ({}, {})
        assert _parse_pool_caps([1, 2, 3]) == ({}, {})

    def test_reload_pool_settings_shape(self) -> None:
        from config import app as config

        cfg = config.reload_pool_settings()
        # The bundled app.example.yaml (or real app.yaml) always parses.
        assert cfg is not None
        assert cfg["scope"] in ("pooled", "all")
        assert isinstance(cfg["pool_caps_enabled"], bool)
        assert isinstance(cfg["pool_caps"], dict)
        assert isinstance(cfg["pool_models"], dict)

    def test_module_constants_present(self) -> None:
        from config import app as config

        assert hasattr(config, "DAILY_TOKEN_LIMIT_SCOPE")
        assert hasattr(config, "PER_KEY_POOL_CAPS_ENABLED")
        assert hasattr(config, "PER_KEY_POOL_CAPS")
        assert hasattr(config, "PER_KEY_POOL_MODELS")
        assert isinstance(config.PER_KEY_POOL_CAPS, dict)
        assert isinstance(config.PER_KEY_POOL_MODELS, dict)


# ============================================================================
# Custom pool definitions (config-defined pools via compile_pools)
# ============================================================================
class TestCustomPools:
    def test_custom_provider_pool_derived_capped_enforced(self, tmp_path: Path) -> None:
        """A mapping-form pool for a custom provider derives, caps, enforces.

        Models a custom summary endpoint with its own daily pool while the
        transcription endpoint (unlisted model, same provider) stays uncapped.
        """
        t = _tracker(
            tmp_path / "s.json",
            pool_caps={"myhost": {"standard": 1000}},
            pool_models={"myhost": {"standard": ["my-model"]}},
        )
        summary = {"provider": "myhost", "key_env": "SUM_KEY", "model": "my-model"}
        t.add_tokens(400, **summary)
        # Derived into the custom pool.
        assert t._own_buckets[BucketKey("myhost", "SUM_KEY", "standard")] == 400
        t.add_tokens(600, **summary)  # cap (1000) now exhausted
        assert t.try_reserve(1, **summary) is None

        # An unlisted model on the same provider derives no pool: uncapped.
        transcription = {
            "provider": "myhost",
            "key_env": "TRANS_KEY",
            "model": "other-vlm",
        }
        assert t.try_reserve(1, **transcription) == 1
        assert t._bucket_for("myhost", "TRANS_KEY", "other-vlm")[0].pool is None

    def test_custom_pool_prefix_matches_suffixed_model(self, tmp_path: Path) -> None:
        t = _tracker(
            tmp_path / "s.json",
            pool_models={"myhost": {"standard": ["my-model"]}},
        )
        bucket, stamped = t._bucket_for("myhost", "KEY", "my-model-2024-06")
        assert stamped is True
        assert bucket.pool == "standard"

    def test_configured_openai_models_replace_builtins(self, tmp_path: Path) -> None:
        """A configured openai pool list REPLACES the built-in model lists."""
        t = _tracker(
            tmp_path / "s.json",
            pool_models={"openai": {"tiny": ["gpt-5-mini"]}},
        )
        # gpt-5-mini now lands in the configured "tiny" pool, not "small".
        assert t._bucket_for("openai", "K", "gpt-5-mini")[0].pool == "tiny"
        # gpt-5 (built-in "large") no longer matches: configured pools replace
        # the built-ins for the covered provider entirely.
        assert t._bucket_for("openai", "K", "gpt-5")[0].pool is None

    def test_capless_models_list_tracked_but_uncapped(self, tmp_path: Path) -> None:
        """A pool defined only by a models list is tracked but never blocked."""
        t = _tracker(
            tmp_path / "s.json",
            pool_caps=None,  # no config caps at all
            pool_models={"myhost": {"bulk": ["my-model"]}},
        )
        stamp = {"provider": "myhost", "key_env": "KEY", "model": "my-model"}
        # Usage far beyond any built-in cap ("bulk" is not in DEFAULT_POOL_CAPS).
        t.add_tokens(50_000_000, **stamp)
        assert t._own_buckets[BucketKey("myhost", "KEY", "bulk")] == 50_000_000
        assert t.try_reserve(1, **stamp) == 1  # uncapped -> admitted
        # Stats surface the tracked pool with cap/remaining None.
        rows = {r["pool"]: r for r in t.get_stats()["pool_buckets"]}
        assert rows["bulk"]["used"] == 50_000_000
        assert rows["bulk"]["cap"] is None
        assert rows["bulk"]["remaining"] is None

    def test_set_pool_settings_refreshes_compiled_pools(self, tmp_path: Path) -> None:
        """The mid-wait reload path swaps custom pool definitions live."""
        t = _tracker(tmp_path / "s.json")
        assert t._bucket_for("myhost", "K", "my-model")[0].pool is None
        t.set_pool_settings(pool_models={"myhost": {"standard": ["my-model"]}})
        assert t._bucket_for("myhost", "K", "my-model")[0].pool == "standard"
        # Empty mapping drops custom pools, reverting to built-ins.
        t.set_pool_settings(pool_models={})
        assert t._bucket_for("myhost", "K", "my-model")[0].pool is None
        assert t._bucket_for("openai", "K", "gpt-5")[0].pool == "large"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
