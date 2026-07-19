"""Regression tests for the Step 1C AutoExcerpter hardening work package.

Covers, per work item:
1. CTRL+C cancellation: executor created once per item, shut down with
   cancel_futures on a fatal error; completion-order processing.
2. Progress-counter race: lock-guarded increment loses no updates.
3. Per-page EWMA (further shapes tested in test_llm_token_tracker.py).
4. Defensive cache-token capture (three provider shapes).
5. Live re-read of the daily limit during both wait paths.
6. Retry-After honoring + backoff cap.
7. Unified atomic writer: per-PID temp name + race retries.
8. Dead-config removal: accessors gone.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import cli.loop as loop_module
import config.accessors as accessors
import config.state as state_mod
import llm.token_tracker as token_tracker
import pipeline.transcriber as transcriber
from config import app as app_config
from llm.base import BACKOFF_CAP_S, LLMClientBase
from llm.token_tracker import DailyTokenTracker
from pipeline.transcriber import ItemTranscriber


# ============================================================================
# Item 4: defensive cache-token capture — three provider shapes
# ============================================================================
def _report_and_capture(usage_metadata: dict[str, Any]) -> int:
    """Run _report_token_usage over a response and return the committed total."""
    client = LLMClientBase.__new__(LLMClientBase)
    response = SimpleNamespace(usage_metadata=usage_metadata)
    with patch("llm.base.get_token_tracker") as mock_tt:
        tracker = MagicMock()
        tracker.get_tokens_used_today.return_value = 0
        mock_tt.return_value = tracker
        client._report_token_usage(response, "ctx")
    assert tracker.add_tokens.call_count == 1
    return int(tracker.add_tokens.call_args[0][0])


class TestCacheTokenCapture:
    def test_raw_anthropic_shape_adds_cache_at_full_weight(self) -> None:
        """input excludes cache; cache_*_input_tokens added on top."""
        committed = _report_and_capture(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 100,
            }
        )
        # 150 (input+output) + 900 cache = 1050.
        assert committed == 1050

    def test_langchain_normalized_shape_no_addition(self) -> None:
        """input_token_details signals cache already folded into the total."""
        committed = _report_and_capture(
            {
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
                "input_token_details": {"cache_read": 800, "cache_creation": 0},
            }
        )
        assert committed == 1200

    def test_openai_cached_tokens_shape_no_addition(self) -> None:
        """OpenAI cached prompt tokens are a subset already in the total."""
        committed = _report_and_capture(
            {
                "input_tokens": 1500,
                "output_tokens": 100,
                "total_tokens": 1600,
                "input_token_details": {"cache_read": 1200},
            }
        )
        assert committed == 1600

    def test_raw_anthropic_cache_on_unwrapped_response_metadata(self) -> None:
        """Cache read from response_metadata['usage'] when not folded."""
        client = LLMClientBase.__new__(LLMClientBase)
        response = SimpleNamespace(
            usage_metadata={"input_tokens": 100, "output_tokens": 50},
            response_metadata={
                "usage": {
                    "cache_read_input_tokens": 400,
                    "cache_creation_input_tokens": 0,
                }
            },
        )
        with patch("llm.base.get_token_tracker") as mock_tt:
            tracker = MagicMock()
            tracker.get_tokens_used_today.return_value = 0
            mock_tt.return_value = tracker
            client._report_token_usage(response, "ctx")
        assert tracker.add_tokens.call_args[0][0] == 550


# ============================================================================
# Item 6: Retry-After honoring + backoff cap
# ============================================================================
def _retry_client(max_retries: int = 2) -> LLMClientBase:
    client = LLMClientBase.__new__(LLMClientBase)
    client.rate_limiter = None
    client.max_retries = max_retries
    client.model_name = "gpt-5-mini"
    client.provider = "openai"
    return client


def _exc_with_retry_after(value: str, status: int = 429) -> Exception:
    exc = Exception("rate limited")
    exc.status_code = status  # type: ignore[attr-defined]
    exc.response = SimpleNamespace(headers={"retry-after": value})  # type: ignore[attr-defined]
    return exc


class TestRetryAfter:
    def test_parse_seconds_form(self) -> None:
        exc = _exc_with_retry_after("45")
        assert LLMClientBase._parse_retry_after(exc) == pytest.approx(45.0)

    def test_parse_missing_returns_none(self) -> None:
        assert LLMClientBase._parse_retry_after(Exception("no headers")) is None

    def test_parse_http_date_form(self) -> None:
        exc = Exception("x")
        exc.response = SimpleNamespace(  # type: ignore[attr-defined]
            headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}
        )
        parsed = LLMClientBase._parse_retry_after(exc)
        assert parsed is not None and parsed > 0

    def test_backoff_never_exceeds_cap(self) -> None:
        client = _retry_client()
        with patch(
            "llm.base._RETRY_CONFIG",
            {"backoff_base": 10.0, "backoff_multipliers": {"timeout": 10.0}},
        ):
            # 10 * 10**5 would be huge; must be clamped to the cap.
            assert client._calculate_backoff(5, "timeout") <= BACKOFF_CAP_S

    def test_invoke_honors_retry_after(self) -> None:
        client = _retry_client(max_retries=2)
        model = MagicMock()
        model.invoke.side_effect = [_exc_with_retry_after("45"), "ok"]
        slept: list[float] = []
        with (
            patch("llm.base.time.sleep", side_effect=slept.append),
            patch("llm.base.get_token_tracker"),
        ):
            result = client._invoke_with_retry(model, [], {}, "ctx")
        assert result == "ok"
        # Waited at least the server-requested 45 s (backoff was far smaller).
        assert slept[0] == pytest.approx(45.0)

    def test_invoke_caps_hostile_retry_after(self) -> None:
        client = _retry_client(max_retries=2)
        model = MagicMock()
        model.invoke.side_effect = [_exc_with_retry_after("100000"), "ok"]
        slept: list[float] = []
        with (
            patch("llm.base.time.sleep", side_effect=slept.append),
            patch("llm.base.get_token_tracker"),
        ):
            client._invoke_with_retry(model, [], {}, "ctx")
        assert slept[0] == pytest.approx(BACKOFF_CAP_S)

    def test_retry_after_feeds_rate_limit_signal(self) -> None:
        """A Retry-After on an otherwise-unclassified error still signals the
        limiter as a rate-limit event."""
        limiter = MagicMock()
        client = _retry_client(max_retries=0)
        client.rate_limiter = limiter
        exc = Exception("weird 200 body")  # not classified retryable
        exc.response = SimpleNamespace(headers={"retry-after": "5"})  # type: ignore[attr-defined]
        model = MagicMock()
        model.invoke.side_effect = exc
        with (
            patch("llm.base.time.sleep"),
            patch("llm.base.get_token_tracker"),
            pytest.raises(Exception, match="weird"),
        ):
            client._invoke_with_retry(model, [], {}, "ctx")
        limiter.report_error.assert_called_once_with(True)


# ============================================================================
# Item 5: live re-read of the daily limit during both wait paths
# ============================================================================
class TestLiveLimitReread:
    def test_set_daily_limit_lifts_cap(self, tmp_path: Path) -> None:
        t = DailyTokenTracker(
            daily_limit=10, enabled=True, state_file=tmp_path / "s.json"
        )
        t.add_tokens(10)
        assert t.is_limit_reached() is True
        t.set_daily_limit(10_000)
        assert t.is_limit_reached() is False

    def test_reload_daily_token_limit_reads_fresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "app.yaml"
        cfg.write_text(
            "daily_token_limit:\n  daily_tokens: 7_500_000\n", encoding="utf-8"
        )
        monkeypatch.setattr(app_config, "_APP_CONFIG_PATH", cfg)
        assert app_config.reload_daily_token_limit() == 7_500_000

    def test_reload_returns_none_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "app.yaml"
        cfg.write_text("summarize: true\n", encoding="utf-8")
        monkeypatch.setattr(app_config, "_APP_CONFIG_PATH", cfg)
        assert app_config.reload_daily_token_limit() is None

    def test_wait_for_token_reset_rereads_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mid-item wait path (llm.token_tracker.wait_for_token_reset)."""
        t = DailyTokenTracker(
            daily_limit=10,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=1000,  # cannot admit a page within 10
        )
        monkeypatch.setattr(token_tracker, "_tracker_instance", t)
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        monkeypatch.setattr(app_config, "reload_daily_token_limit", lambda: 10**9)
        assert token_tracker.wait_for_token_reset() is True
        assert t.daily_limit == 10**9

    def test_cli_wait_loop_rereads_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-item CLI gate (cli.loop._wait_for_token_reset)."""
        t = DailyTokenTracker(
            daily_limit=10, enabled=True, state_file=tmp_path / "s.json"
        )
        t.add_tokens(10)  # at limit
        monkeypatch.setattr(loop_module, "_user_requested_cancel", lambda: False)
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        monkeypatch.setattr(app_config, "CLI_MODE", True, raising=False)
        monkeypatch.setattr(app_config, "reload_daily_token_limit", lambda: 10**9)
        assert loop_module._wait_for_token_reset(t, seconds_until_reset=5) is True
        assert t.daily_limit == 10**9


# ============================================================================
# Item 7: unified atomic writer — per-PID temp name + race retries
# ============================================================================
class TestAtomicWriter:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state_mod.write_json_atomic(path, {"a": 1, "b": "x"})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": "x"}

    def test_temp_name_is_per_process_unique(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "state.json"
        seen: list[str] = []
        real_replace = os.replace

        def spy(src: Any, dst: Any) -> None:
            seen.append(str(src))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        state_mod.write_json_atomic(path, {"a": 1})
        assert str(os.getpid()) in seen[0]
        assert seen[0].endswith(".tmp")

    def test_retries_on_permission_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "state.json"
        calls = {"n": 0}
        real_replace = os.replace

        def flaky(src: Any, dst: Any) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError("locked by AV")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky)
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        state_mod.write_json_atomic(path, {"a": 1})
        assert calls["n"] == 2
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_retries_on_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "state.json"
        calls = {"n": 0}
        real_replace = os.replace

        def flaky(src: Any, dst: Any) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise FileNotFoundError("temp vanished")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky)
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        state_mod.write_json_atomic(path, {"a": 1})
        assert calls["n"] == 2

    def test_no_temp_file_left_behind(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state_mod.write_json_atomic(path, {"a": 1})
        leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_tracker_save_uses_shared_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[Path] = []
        real = state_mod.write_json_atomic

        def spy(path: Path, data: dict[str, Any]) -> None:
            called.append(path)
            real(path, data)

        monkeypatch.setattr(state_mod, "write_json_atomic", spy)
        t = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        t._save_state(force=True)
        assert called and called[-1] == tmp_path / "s.json"


# ============================================================================
# Item 8: dead-config removal — accessors gone
# ============================================================================
class TestRemovedConfig:
    def test_image_processing_concurrency_accessor_removed(self) -> None:
        import config as config_pkg

        assert not hasattr(accessors, "get_image_processing_concurrency")
        assert "get_image_processing_concurrency" not in accessors.__all__
        assert "get_image_processing_concurrency" not in config_pkg.__all__


# ============================================================================
# Items 1 + 2: executor lifecycle, cancel_futures, counter race
# ============================================================================
class _FakeSource:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


def _repass_transcriber(tmp_path: Path) -> ItemTranscriber:
    obj = ItemTranscriber.__new__(ItemTranscriber)
    obj.completed_page_indices = set()
    obj._budget_exhausted = threading.Event()
    obj._count_lock = threading.Lock()
    obj._token_tracker = DailyTokenTracker(
        daily_limit=10**9, enabled=True, state_file=tmp_path / "s.json"
    )
    return obj


class TestExecutorLifecycle:
    def test_pool_created_once_across_repasses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _repass_transcriber(tmp_path)
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (2, None)
        )

        real_executor = concurrent.futures.ThreadPoolExecutor
        instances: list[Any] = []
        shutdowns: list[tuple[bool, bool]] = []

        class SpyExecutor(real_executor):  # type: ignore[misc,valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                instances.append(self)
                super().__init__(*a, **k)

            def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
                shutdowns.append((wait, cancel_futures))
                super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", SpyExecutor)

        state = {"reset": False}

        def fake_process(
            idx, source, t_res, s_res, total, count_ref, already_complete=0
        ):
            if not state["reset"] and idx >= 3:
                obj._budget_exhausted.set()
                return None
            t_res.append({"original_input_order_index": idx})
            with obj._count_lock:
                count_ref[0] += 1
            return {"original_input_order_index": idx}

        def fake_wait() -> bool:
            state["reset"] = True
            return True

        obj._process_single_page = fake_process  # type: ignore[method-assign]
        monkeypatch.setattr(transcriber, "wait_for_token_reset", fake_wait)

        t_res, _ = obj._transcribe_and_summarize(_FakeSource(6))  # type: ignore[arg-type]

        # Exactly ONE executor for the whole item, across the budget re-pass.
        assert len(instances) == 1
        # Deterministic teardown with cancel_futures on the way out.
        assert (False, True) in shutdowns
        assert sorted(r["original_input_order_index"] for r in t_res) == list(range(6))

    def test_fatal_error_cancels_queued_futures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _repass_transcriber(tmp_path)
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber, "get_transcription_concurrency", lambda: (1, None)
        )

        real_executor = concurrent.futures.ThreadPoolExecutor
        shutdowns: list[tuple[bool, bool]] = []

        class SpyExecutor(real_executor):  # type: ignore[misc,valid-type]
            def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
                shutdowns.append((wait, cancel_futures))
                super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", SpyExecutor)

        def boom(idx, *a, **k):
            raise RuntimeError("fatal")

        obj._process_single_page = boom  # type: ignore[method-assign,assignment]

        with pytest.raises(RuntimeError, match="fatal"):
            obj._transcribe_and_summarize(_FakeSource(4))  # type: ignore[arg-type]

        # The finally shut the pool down cancelling queued work.
        assert (False, True) in shutdowns


class TestCounterRace:
    def test_concurrent_increments_lose_no_updates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = ItemTranscriber.__new__(ItemTranscriber)
        obj._budget_exhausted = threading.Event()
        obj._count_lock = threading.Lock()
        obj._token_tracker = DailyTokenTracker(
            daily_limit=1, enabled=False, state_file=tmp_path / "s.json"
        )
        obj.transcription_times = []
        obj.transcription_provider = "openai"
        obj.summary_manager = None
        obj.log_path = tmp_path / "log.jsonl"
        obj.summary_log_path = tmp_path / "slog.jsonl"
        obj.start_time_processing = time.time()

        tx = MagicMock()
        tx.transcribe_payload.return_value = {
            "transcription": "hello",
            "processing_time": 0.001,
            "provider": "openai",
        }
        obj.transcribe_manager = tx

        class _Src:
            def image_name(self, idx: int) -> str:
                return f"page_{idx}.jpg"

            def build_payload(self, idx: int) -> Any:
                return SimpleNamespace(source_file="f", provenance={}, page_index=None)

        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(transcriber, "append_to_log", lambda *a, **k: None)

        n = 64
        count_ref = [0]
        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        src = _Src()

        def work(idx: int) -> None:
            obj._process_single_page(idx, src, t_results, s_results, n, count_ref)  # type: ignore[arg-type]

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert count_ref[0] == n
        assert len(t_results) == n
