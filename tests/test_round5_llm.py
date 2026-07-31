"""Regression tests for the round-5 LLM fixes.

One focused test class per fix:

1.  ``TranscriptionManager._parse_transcription_from_text`` — an explicit
    ``"transcription": null`` (the schema's own no-text signal) yields the
    bracketed placeholder instead of the raw JSON blob, and the final fallback
    returns the fence-stripped text, never the fenced original.
2.  ``llm.shared_ledger.derive_pool`` — "o3-mini" lands in the small pool
    instead of matching the "o3" large-pool prefix.
3.  ``llm.token_tracker.wait_for_token_reset`` — the shared-ledger force-sync
    and the YAML re-reads are throttled (they ran every second for up to a
    day), and the post-deadline log line reports the real budget state.
4.  The shared rate limiter is keyed by the RESOLVED provider, so a manager
    built with ``provider=None`` draws on the same limiter as one built with
    ``provider="openai"``.
5.  ``llm.client.get_chat_model`` — an OpenAI fine-tuned id ("ft:...") is no
    longer mangled by provider-prefix stripping.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import llm.shared_ledger as sl
import llm.token_tracker as tt
from llm.client import LLMConfig, get_chat_model
from llm.summary import SummaryManager
from llm.transcription import TranscriptionManager

# ============================================================================
# Fix 1: explicit null transcription and the fence-strip fallback
# ============================================================================
_SCHEMA = {
    "name": "transcription_schema",
    "schema": {"type": "object", "properties": {"transcription": {"type": "string"}}},
}


def _bare_manager(**overrides: Any) -> TranscriptionManager:
    """Bare TranscriptionManager bypassing __init__."""
    mgr = TranscriptionManager.__new__(TranscriptionManager)
    defaults: dict[str, Any] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "custom_capabilities": None,
        "transcription_schema": _SCHEMA,
        "system_prompt": "sys",
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(mgr, attr, val)
    return mgr


class TestNullTranscriptionIsNoTranscribableText:
    def test_explicit_null_returns_placeholder(self) -> None:
        mgr = _bare_manager()
        data = json.dumps(
            {
                "image_analysis": "blank leaf",
                "transcription": None,
                "no_transcribable_text": False,
                "transcription_not_possible": False,
            }
        )
        result = mgr._parse_transcription_from_text(data, "page_007.png")
        assert result == "[page_007.png: no transcribable text]"
        assert "{" not in result  # never the raw JSON blob

    def test_null_with_flag_still_uses_flag_message(self) -> None:
        """An explicit flag keeps its richer message (analysis included)."""
        mgr = _bare_manager()
        data = json.dumps(
            {
                "image_analysis": "Photographic plate only.",
                "transcription": None,
                "no_transcribable_text": True,
                "transcription_not_possible": False,
            }
        )
        result = mgr._parse_transcription_from_text(data, "page_008.png")
        assert "Photographic plate only" in result

    def test_unknown_json_falls_back_to_stripped_text(self) -> None:
        """The fallback honors the fence-strip contract."""
        mgr = _bare_manager()
        data = json.dumps({"unknown_key": "value"})
        assert mgr._parse_transcription_from_text(data) == data
        fenced = f"```json\n{data}\n```"
        assert mgr._parse_transcription_from_text(fenced) == data


# ============================================================================
# Fix 2: o3-mini belongs to the small pool
# ============================================================================
class TestO3MiniPool:
    def test_o3_mini_is_small_pool(self) -> None:
        assert sl.derive_pool("openai", "o3-mini") == sl.POOL_SMALL

    def test_o3_itself_stays_large_pool(self) -> None:
        assert sl.derive_pool("openai", "o3") == sl.POOL_LARGE
        assert sl.derive_pool("openai", "o3-2025-04-16") == sl.POOL_LARGE


# ============================================================================
# Fix 3: wait_for_token_reset throttling and accurate exit logging
# ============================================================================
def _blocked_tracker(*, admits: bool = False) -> MagicMock:
    tracker = MagicMock()
    tracker.enabled = True
    tracker._shared_enabled = True
    tracker.can_admit_page.return_value = admits
    tracker.get_seconds_until_reset.return_value = 35
    tracker.get_reset_time.return_value = datetime.now(UTC)
    tracker.get_stats.return_value = {
        "tokens_used_today": 1_000,
        "daily_limit": 900,
    }
    return tracker


class TestWaitForTokenResetThrottling:
    def test_sync_and_reload_are_throttled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracker = _blocked_tracker()
        monkeypatch.setattr(tt, "get_token_tracker", lambda: tracker)
        monkeypatch.setattr("llm.token_tracker.time.sleep", lambda _s: None)

        calls = {"limit": 0, "pool": 0}
        monkeypatch.setattr(
            tt.config,
            "reload_daily_token_limit",
            lambda: calls.__setitem__("limit", calls["limit"] + 1),
        )
        monkeypatch.setattr(
            tt.config,
            "reload_pool_settings",
            lambda: calls.__setitem__("pool", calls["pool"] + 1),
        )

        assert tt.wait_for_token_reset() is True
        # 35 s wait: ledger sync fires early then every ~10 s (elapsed
        # 1/11/21/31) -> 4 times, not 35.
        assert tracker.sync_ledger_now.call_count == 4
        # Config re-read fires early then every ~30 s (elapsed 1/31) -> 2 times.
        assert calls["limit"] == 2
        assert calls["pool"] == 2

    def test_exhausted_deadline_without_headroom_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The old code always claimed 'Token budget has reset'."""
        tracker = _blocked_tracker()
        tracker.get_seconds_until_reset.return_value = 3
        monkeypatch.setattr(tt, "get_token_tracker", lambda: tracker)
        monkeypatch.setattr("llm.token_tracker.time.sleep", lambda _s: None)
        monkeypatch.setattr(tt.config, "reload_daily_token_limit", lambda: None)
        monkeypatch.setattr(tt.config, "reload_pool_settings", lambda: None)

        with caplog.at_level(logging.INFO, logger=tt.logger.name):
            assert tt.wait_for_token_reset() is True

        messages = [record.getMessage() for record in caplog.records]
        assert any("still cannot fit a page" in m for m in messages)
        assert not any("Token budget has reset" in m for m in messages)

    def test_budget_freed_mid_wait_logs_reset(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracker = _blocked_tracker()
        tracker.get_seconds_until_reset.return_value = 5
        # Blocked on entry, then admitted on the first in-loop re-check.
        tracker.can_admit_page.side_effect = [False, True]
        monkeypatch.setattr(tt, "get_token_tracker", lambda: tracker)
        monkeypatch.setattr("llm.token_tracker.time.sleep", lambda _s: None)
        monkeypatch.setattr(tt.config, "reload_daily_token_limit", lambda: None)
        monkeypatch.setattr(tt.config, "reload_pool_settings", lambda: None)

        with caplog.at_level(logging.INFO, logger=tt.logger.name):
            assert tt.wait_for_token_reset() is True

        assert any(
            "Token budget has reset" in record.getMessage() for record in caplog.records
        )


# ============================================================================
# Fix 4: the shared rate limiter is keyed by the resolved provider
# ============================================================================
def _write_transcription_assets(tmp_path: Path) -> tuple[Path, Path]:
    schemas_dir = tmp_path / "schemas"
    prompts_dir = tmp_path / "prompts"
    schemas_dir.mkdir()
    prompts_dir.mkdir()
    (schemas_dir / "transcription_schema.json").write_text(
        json.dumps(_SCHEMA), encoding="utf-8"
    )
    (prompts_dir / "transcription_system_prompt.txt").write_text(
        "Transcribe. {{SCHEMA}}", encoding="utf-8"
    )
    return schemas_dir, prompts_dir


def _write_summary_assets(tmp_path: Path) -> tuple[Path, Path]:
    schemas_dir = tmp_path / "schemas"
    prompts_dir = tmp_path / "prompts"
    schemas_dir.mkdir()
    prompts_dir.mkdir()
    (schemas_dir / "summary_schema.json").write_text(
        json.dumps(_SCHEMA), encoding="utf-8"
    )
    (prompts_dir / "summary_system_prompt.txt").write_text(
        "Summarize. {{SCHEMA}}", encoding="utf-8"
    )
    return schemas_dir, prompts_dir


def _stub_loader(section: str) -> MagicMock:
    loader = MagicMock()
    loader.get_model_config.return_value = {section: {"name": "gpt-5-mini"}}
    loader.get_concurrency_config.return_value = {
        "retry": {"schema_retries": {section.split("_")[0]: {}}},
        "api_requests": {section.split("_")[0]: {"service_tier": "flex"}},
    }
    return loader


class TestSharedLimiterKeyedByResolvedProvider:
    def test_transcription_manager_keys_resolved_provider(self, tmp_path: Path) -> None:
        schemas_dir, prompts_dir = _write_transcription_assets(tmp_path)
        keys: list[str | None] = []

        def recorder(provider: str | None) -> MagicMock:
            keys.append(provider)
            return MagicMock()

        with (
            patch("llm.base.get_chat_model", return_value=MagicMock()),
            patch("llm.base.get_api_timeout", return_value=300),
            patch(
                "llm.base.get_config_loader",
                return_value=_stub_loader("transcription_model"),
            ),
            patch("llm.transcription.SCHEMAS_DIR", schemas_dir),
            patch("llm.transcription.PROMPTS_DIR", prompts_dir),
            patch("llm.transcription.get_shared_rate_limiter", recorder),
        ):
            TranscriptionManager(model_name="gpt-5-mini", api_key="k")

        assert keys == ["openai"]  # not None -> not the "default" bucket

    def test_summary_manager_keys_resolved_provider(self, tmp_path: Path) -> None:
        schemas_dir, prompts_dir = _write_summary_assets(tmp_path)
        keys: list[str | None] = []

        def recorder(provider: str | None) -> MagicMock:
            keys.append(provider)
            return MagicMock()

        with (
            patch("llm.base.get_chat_model", return_value=MagicMock()),
            patch("llm.base.get_api_timeout", return_value=300),
            patch("llm.summary.get_api_timeout", return_value=300),
            patch(
                "llm.base.get_config_loader", return_value=_stub_loader("summary_model")
            ),
            patch("llm.summary.SCHEMAS_DIR", schemas_dir),
            patch("llm.summary.PROMPTS_DIR", prompts_dir),
            patch("llm.summary.get_shared_rate_limiter", recorder),
        ):
            SummaryManager(model_name="gpt-5-mini", api_key="k")

        assert keys == ["openai"]

    def test_none_and_explicit_provider_share_one_limiter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm import rate_limit

        monkeypatch.setattr(rate_limit, "_SHARED_LIMITERS", {})
        monkeypatch.setattr(
            "config.accessors.get_rate_limits", lambda: [(10, 1)], raising=False
        )
        from llm.client import get_provider_for_model

        implicit = rate_limit.get_shared_rate_limiter(
            get_provider_for_model("gpt-5-mini")
        )
        explicit = rate_limit.get_shared_rate_limiter("openai")
        assert implicit is explicit


# ============================================================================
# Fix 5: fine-tuned OpenAI model ids survive prefix stripping
# ============================================================================
class TestFineTunedModelName:
    def test_ft_prefix_is_not_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        model_id = "ft:gpt-4o-mini-2024-07-18:acme:culinary:abc123"
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(LLMConfig(model=model_id, provider="openai"))
        assert mock_class.call_args[1]["model"] == model_id

    def test_supported_provider_prefix_is_still_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(LLMConfig(model="openai:gpt-5-mini"))
        assert mock_class.call_args[1]["model"] == "gpt-5-mini"

    def test_sdk_retries_are_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with patch("langchain_openai.ChatOpenAI") as mock_class:
            mock_class.return_value = MagicMock()
            get_chat_model(LLMConfig(model="gpt-5-mini", provider="openai"))
        assert mock_class.call_args[1]["max_retries"] == 0
