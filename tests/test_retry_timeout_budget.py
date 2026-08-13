"""Tests for the timeout/connection split, the timeout budget and the deadline.

Three features of ``llm.base`` land here:

- ``_is_timeout_error`` / ``_is_connection_error`` and the reworked
  ``LLMClientBase._classify_error``: timeouts and connection failures are now
  DISTINCT classes, and both are decided before the HTTP status checks so a
  gateway timeout carrying a 5xx status lands in the small timeout budget.
- ``retry.timeout_attempts`` -> ``TIMEOUT_ATTEMPTS`` -> per-instance
  ``self.timeout_attempts``: a separate, smaller retry budget for
  timeout-class failures. ``0`` (the class default used by ``__new__``-built
  test clients) restores the old full-ladder behavior.
- ``retry.call_timeout`` -> ``CALL_TIMEOUT_S`` -> per-instance
  ``self.call_deadline_s``: a wall-clock ceiling for one ``_invoke_with_retry``
  call across all attempts, checked before each attempt (raising
  ``CallDeadlineExceeded``) and after each failure (re-raising the provider
  error), and clamping the backoff sleep.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, NoReturn, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest

import llm.base as base_module
from llm.base import CallDeadlineExceeded, LLMClientBase


def _make_client(**overrides: object) -> LLMClientBase:
    """Create a bare LLMClientBase bypassing __init__ for isolated unit tests."""
    client = LLMClientBase.__new__(LLMClientBase)
    defaults: dict[str, object] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "timeout": 300,
        "rate_limiter": None,
        "max_retries": 5,
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
    # Neutralize collaborators so only the retry control flow is exercised.
    client._wait_for_rate_limit = MagicMock()  # type: ignore[method-assign]
    client._extract_tokens_from_exception = MagicMock()  # type: ignore[method-assign]
    client._report_error = MagicMock()  # type: ignore[method-assign]
    client._parse_retry_after = MagicMock(return_value=None)  # type: ignore[method-assign]
    client._calculate_backoff = MagicMock(return_value=0.0)  # type: ignore[method-assign]
    return client


def _read_timeout() -> Exception:
    """A bare httpx read timeout, the shape a hung generation produces."""
    return httpx.ReadTimeout("read timed out")


def _connect_error() -> Exception:
    """A bare httpx connection failure (cheap, unbilled)."""
    return httpx.ConnectError("connection refused")


def _server_error() -> Exception:
    """A retryable (HTTP 500) exception carrying no transport cause."""
    exc = Exception("Internal Server Error")
    exc.status_code = 500  # type: ignore[attr-defined]
    return exc


def _raise_read_timeout(*_args: Any, **_kwargs: Any) -> NoReturn:
    """side_effect that raises a FRESH read timeout on every call.

    A callable side_effect that merely *returns* an exception would be handed
    back as the mock's return value; it has to raise.
    """
    raise httpx.ReadTimeout("read timed out")


def _raise_connect_error(*_args: Any, **_kwargs: Any) -> NoReturn:
    """side_effect that raises a FRESH connection failure on every call."""
    raise httpx.ConnectError("connection refused")


class APITimeoutError(Exception):
    """Stand-in for an SDK timeout type raised without an httpx cause.

    Only its class NAME can classify it, which is exactly the fallback path
    ``_TIMEOUT_CLASS_NAMES`` exists for.
    """


# ============================================================================
# Classifier: timeout vs connection vs status vs string fallbacks
# ============================================================================
class TestClassifierSplit:
    """``_classify_error`` after the timeout/connection split."""

    def test_bare_read_timeout_is_timeout(self) -> None:
        assert LLMClientBase._classify_error(_read_timeout()) == (True, "timeout")

    def test_bare_connect_timeout_is_timeout(self) -> None:
        """ConnectTimeout is an httpx.TimeoutException, not a ConnectError."""
        exc = httpx.ConnectTimeout("connect timed out")
        assert LLMClientBase._classify_error(exc) == (True, "timeout")

    def test_wrapped_timeout_found_via_cause(self) -> None:
        """SDKs wrap the transport error; the __cause__ chain must be walked."""
        wrapper = Exception("Request failed.")
        wrapper.__cause__ = _read_timeout()
        assert LLMClientBase._classify_error(wrapper) == (True, "timeout")

    def test_wrapped_timeout_found_via_context(self) -> None:
        wrapper = Exception("Request failed.")
        wrapper.__context__ = _read_timeout()
        assert LLMClientBase._classify_error(wrapper) == (True, "timeout")

    def test_class_name_fallback_without_cause(self) -> None:
        """A bare SDK APITimeoutError has no httpx cause; the name table wins."""
        exc = APITimeoutError("Request timed out.")
        assert exc.__cause__ is None
        assert base_module._is_timeout_error(exc) is True
        assert LLMClientBase._classify_error(exc) == (True, "timeout")

    def test_builtin_timeout_error_is_timeout(self) -> None:
        assert LLMClientBase._classify_error(TimeoutError("nope")) == (True, "timeout")

    def test_timeout_wins_over_5xx_status(self) -> None:
        """A gateway timeout (504/524) must land in the small timeout budget."""
        exc = Exception("Gateway Timeout")
        exc.status_code = 504  # type: ignore[attr-defined]
        exc.__cause__ = _read_timeout()
        assert LLMClientBase._classify_error(exc) == (True, "timeout")

    def test_connect_error_is_connection_not_timeout(self) -> None:
        assert base_module._is_timeout_error(_connect_error()) is False
        assert LLMClientBase._classify_error(_connect_error()) == (True, "connection")

    def test_wrapped_connect_error_found_via_cause(self) -> None:
        wrapper = Exception("Connection error.")
        wrapper.__cause__ = _connect_error()
        assert LLMClientBase._classify_error(wrapper) == (True, "connection")

    def test_connection_string_fallback_is_connection(self) -> None:
        """The message-only path also yields the new ``connection`` class."""
        exc = ConnectionResetError("Connection reset by peer")
        assert LLMClientBase._classify_error(exc) == (True, "connection")

    def test_value_error_is_other(self) -> None:
        assert LLMClientBase._classify_error(ValueError("bad schema")) == (
            False,
            "other",
        )

    def test_status_checks_still_win_for_plain_exceptions(self) -> None:
        """Bare status-carrying exceptions match neither predicate; unchanged."""
        rate = Exception("Too Many Requests")
        rate.status_code = 429  # type: ignore[attr-defined]
        server = Exception("Internal Server Error")
        server.status_code = 500  # type: ignore[attr-defined]
        assert LLMClientBase._classify_error(rate) == (True, "rate_limit")
        assert LLMClientBase._classify_error(server) == (True, "server_error")

    def test_cause_cycle_terminates(self) -> None:
        """A self-referential chain must not spin forever."""
        first = Exception("first")
        second = Exception("second")
        first.__cause__ = second
        second.__cause__ = first
        assert base_module._is_timeout_error(first) is False
        assert base_module._is_connection_error(first) is False
        assert LLMClientBase._classify_error(first) == (False, "other")


# ============================================================================
# Timeout budget: self.timeout_attempts
# ============================================================================
class TestTimeoutBudget:
    """Timeout-class failures ride a smaller ladder than connection failures."""

    def test_perpetual_timeout_stops_at_timeout_attempts(self) -> None:
        """8 general attempts, 3 timeout attempts -> exactly 3 invoke calls."""
        client = _make_client(max_retries=7, timeout_attempts=3)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _raise_read_timeout

        with (
            patch("llm.base.time.sleep"),
            pytest.raises(httpx.ReadTimeout),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 3

    def test_perpetual_connection_error_rides_full_ladder(self) -> None:
        """The same ladder with connection failures uses all 8 attempts."""
        client = _make_client(max_retries=7, timeout_attempts=3)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _raise_connect_error

        with (
            patch("llm.base.time.sleep"),
            pytest.raises(httpx.ConnectError),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 8

    def test_timeout_attempts_zero_restores_full_ladder(self) -> None:
        """The class default (0) means "no separate budget"."""
        assert LLMClientBase.timeout_attempts == 0
        client = _make_client(max_retries=7)
        assert client.timeout_attempts == 0
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _raise_read_timeout

        with (
            patch("llm.base.time.sleep"),
            pytest.raises(httpx.ReadTimeout),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 8

    def test_budget_of_one_fails_on_the_first_timeout(self) -> None:
        client = _make_client(max_retries=7, timeout_attempts=1)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _raise_read_timeout

        with (
            patch("llm.base.time.sleep"),
            pytest.raises(httpx.ReadTimeout),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 1

    def test_non_timeout_failures_do_not_consume_the_budget(self) -> None:
        """Only timeout-class failures count against timeout_attempts."""
        client = _make_client(max_retries=7, timeout_attempts=2)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = [
            _server_error(),
            _read_timeout(),
            _server_error(),
            _server_error(),
            "success",
        ]

        with patch("llm.base.time.sleep"):
            result = client._invoke_with_retry(mock_model, [], {}, "test")

        assert result == "success"
        assert mock_model.invoke.call_count == 5


class TestTimeoutAttemptsResolution:
    """``retry.timeout_attempts`` loading and clamping."""

    def test_default_when_absent(self) -> None:
        assert base_module._resolve_timeout_attempts({}) == 3

    def test_clamped_to_at_least_one(self) -> None:
        assert base_module._resolve_timeout_attempts({"timeout_attempts": 0}) == 1
        assert base_module._resolve_timeout_attempts({"timeout_attempts": -5}) == 1

    def test_clamped_to_max_attempts(self) -> None:
        max_attempts = base_module.DEFAULT_MAX_RETRIES + 1
        resolved = base_module._resolve_timeout_attempts({"timeout_attempts": 999})
        assert resolved == max_attempts

    def test_garbage_falls_back_to_default(self) -> None:
        assert base_module._resolve_timeout_attempts({"timeout_attempts": "fast"}) == 3

    def test_module_constant_matches_config(self) -> None:
        expected = base_module._resolve_timeout_attempts(base_module._RETRY_CONFIG)
        assert expected == base_module.TIMEOUT_ATTEMPTS
        max_attempts = base_module.DEFAULT_MAX_RETRIES + 1
        assert 1 <= base_module.TIMEOUT_ATTEMPTS <= max_attempts


# ============================================================================
# Call deadline: self.call_deadline_s
# ============================================================================
class TestCallDeadline:
    """Wall-clock ceiling across all attempts of one call."""

    def test_watchdog_stops_a_failing_ladder(self) -> None:
        """Attempts remain, but the deadline has passed -> re-raise."""
        client = _make_client(max_retries=7, call_deadline_s=100.0)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _server_error()

        # monotonic: start=0; pre-attempt 1 = 0; post-failure 1 = 10 (retry);
        # pre-attempt 2 = 10; post-failure 2 = 500 (>= 100 -> raise).
        with (
            patch("llm.base.time.monotonic", side_effect=[0.0, 0.0, 10.0, 10.0, 500.0]),
            patch("llm.base.time.sleep"),
            pytest.raises(Exception, match="Internal Server Error"),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 2

    def test_pre_attempt_check_raises_call_deadline_exceeded(self) -> None:
        """A deadline already passed before an attempt fires no API call."""
        client = _make_client(max_retries=7, call_deadline_s=100.0)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _server_error()

        # start=0; pre-attempt 1 = 0; post-failure 1 = 50 (< 100 -> retry);
        # pre-attempt 2 = 500 (>= 100) -> CallDeadlineExceeded, no 2nd invoke.
        with (
            patch("llm.base.time.monotonic", side_effect=[0.0, 0.0, 50.0, 500.0]),
            patch("llm.base.time.sleep"),
            pytest.raises(CallDeadlineExceeded, match="call watchdog"),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 1

    def test_disabled_deadline_adds_no_monotonic_calls(self) -> None:
        """With the ceiling off, the clock is sampled once per failure only."""
        client = _make_client(max_retries=1, call_deadline_s=0.0)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _server_error()

        # Exactly three samples: the start, plus one per failed attempt. An
        # extra pre-attempt sample would exhaust this side_effect list.
        with (
            patch("llm.base.time.monotonic", side_effect=[0.0, 1.0, 2.0]),
            patch("llm.base.time.sleep"),
            pytest.raises(Exception, match="Internal Server Error"),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 2

    def test_class_default_is_disabled(self) -> None:
        assert LLMClientBase.call_deadline_s == 0.0
        assert _make_client().call_deadline_s == 0.0

    def test_backoff_clamped_to_deadline_despite_retry_after(self) -> None:
        """A large Retry-After may not push the sleep past the ceiling."""
        client = _make_client(max_retries=7, call_deadline_s=100.0)
        client._parse_retry_after = MagicMock(return_value=90.0)  # type: ignore[method-assign]
        client._calculate_backoff = MagicMock(return_value=5.0)  # type: ignore[method-assign]
        rate = Exception("Too Many Requests")
        rate.status_code = 429  # type: ignore[attr-defined]
        mock_model = MagicMock()
        mock_model.invoke.side_effect = [rate, "success"]

        slept: list[float] = []

        def _record(seconds: float) -> None:
            slept.append(seconds)

        # start=0; pre-attempt 1 = 0; post-failure 1 = 60 -> remaining 40 s.
        with (
            patch("llm.base.time.monotonic", side_effect=[0.0, 0.0, 60.0, 60.0]),
            patch("llm.base.time.sleep", side_effect=_record),
        ):
            result = client._invoke_with_retry(mock_model, [], {}, "test")

        assert result == "success"
        # Retry-After would have asked for 90 s; the ceiling leaves only 40 s.
        assert sum(slept) == pytest.approx(40.0)


class TestCallTimeoutResolution:
    """``retry.call_timeout`` value table (auto / off / numeric / garbage)."""

    @staticmethod
    def _auto_value(cfg: dict[str, Any]) -> float:
        from config.accessors import get_api_timeout

        return float(get_api_timeout()) * base_module._resolve_timeout_attempts(cfg) + (
            300.0
        )

    def test_absent_key_is_auto(self) -> None:
        cfg: dict[str, Any] = {"timeout_attempts": 3}
        assert base_module._resolve_call_timeout(cfg) == self._auto_value(cfg)

    def test_auto_math_uses_api_timeout_and_timeout_attempts(self) -> None:
        cfg: dict[str, Any] = {"call_timeout": "auto", "timeout_attempts": 3}
        with patch("llm.base.get_api_timeout", return_value=900):
            assert base_module._resolve_call_timeout(cfg) == 900 * 3 + 300

    def test_unrecognised_string_falls_back_to_auto(self) -> None:
        cfg: dict[str, Any] = {"call_timeout": "soonish", "timeout_attempts": 3}
        with patch("llm.base.get_api_timeout", return_value=900):
            assert base_module._resolve_call_timeout(cfg) == 3000.0

    @pytest.mark.parametrize(
        "raw",
        ["off", "OFF", "none", "disabled", 0, -1, -0.5, None, False, ["nope"], {}],
    )
    def test_disabling_values(self, raw: Any) -> None:
        assert base_module._resolve_call_timeout({"call_timeout": raw}) == 0.0

    @pytest.mark.parametrize(("raw", "expected"), [(1200, 1200.0), ("450.5", 450.5)])
    def test_numeric_values(self, raw: Any, expected: float) -> None:
        assert base_module._resolve_call_timeout({"call_timeout": raw}) == expected

    def test_true_is_auto(self) -> None:
        """YAML 1.1 parses a bare ``on`` as boolean true."""
        cfg: dict[str, Any] = {"call_timeout": True, "timeout_attempts": 3}
        with patch("llm.base.get_api_timeout", return_value=900):
            assert base_module._resolve_call_timeout(cfg) == 3000.0

    def test_config_exception_disables(self) -> None:
        broken = MagicMock()
        broken.get.side_effect = RuntimeError("config on fire")
        cfg = cast(dict[str, Any], broken)
        assert base_module._resolve_call_timeout(cfg) == 0.0

    def test_api_timeout_exception_falls_back_to_900(self) -> None:
        cfg: dict[str, Any] = {"call_timeout": "auto", "timeout_attempts": 2}
        with patch("llm.base.get_api_timeout", side_effect=RuntimeError("boom")):
            assert base_module._resolve_call_timeout(cfg) == 900 * 2 + 300

    def test_module_constant_matches_config(self) -> None:
        expected = base_module._resolve_call_timeout(base_module._RETRY_CONFIG)
        assert expected == base_module.CALL_TIMEOUT_S
