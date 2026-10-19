"""Tests for common/retry.py: classification, Retry-After, backoff, ladder."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from autoexcerpter.common.retry import (
    Decision,
    ErrorDecision,
    ErrorKind,
    RetryEvent,
    RetryHooks,
    RetryPolicy,
    call_with_retry,
    classify_error,
    default_decision,
    http_status,
    is_connection_error,
    is_rate_limit_message,
    is_server_error_message,
    is_timeout_error,
    parse_retry_after,
)
from autoexcerpter.common.usage import Usage


def _status_error(status: int, message: str = "error") -> Exception:
    exc = Exception(message)
    exc.status_code = status  # type: ignore[attr-defined]
    return exc


class APITimeoutError(Exception):
    """An SDK timeout class raised without an httpx cause."""


class _ValidationFailure(Exception):
    """Stands for unparseable model output."""


class TestClassifyError:
    @pytest.mark.parametrize(
        ("status", "kind"),
        [
            (429, ErrorKind.RATE_LIMIT),
            (500, ErrorKind.SERVER_ERROR),
            (503, ErrorKind.SERVER_ERROR),
            (529, ErrorKind.SERVER_ERROR),
            (408, ErrorKind.OTHER),
            (409, ErrorKind.OTHER),
            (400, ErrorKind.CLIENT_ERROR),
            (401, ErrorKind.CLIENT_ERROR),
            (404, ErrorKind.CLIENT_ERROR),
        ],
    )
    def test_status_table(self, status: int, kind: ErrorKind) -> None:
        assert classify_error(_status_error(status)) is kind

    def test_client_status_beats_a_transient_looking_message(self) -> None:
        exc = _status_error(400, "context too long; request timed out upstream")
        assert classify_error(exc) is ErrorKind.CLIENT_ERROR

    def test_rate_limit_with_timeout_word_is_rate_limit(self) -> None:
        assert classify_error(_status_error(429, "timeout")) is ErrorKind.RATE_LIMIT

    def test_builtin_timeout_error_is_timeout(self) -> None:
        assert classify_error(TimeoutError("Connection timed out")) is ErrorKind.TIMEOUT

    @pytest.mark.parametrize("link", ["__cause__", "__context__"])
    def test_wrapped_timeout_found_in_chain(self, link: str) -> None:
        wrapper = Exception("Request failed.")
        setattr(wrapper, link, httpx.ReadTimeout("read timed out"))
        assert classify_error(wrapper) is ErrorKind.TIMEOUT

    def test_class_name_fallback(self) -> None:
        exc = APITimeoutError("Request timed out.")
        assert is_timeout_error(exc)
        assert classify_error(exc) is ErrorKind.TIMEOUT

    def test_timeout_wins_over_5xx_status(self) -> None:
        exc = _status_error(504, "Gateway Timeout")
        exc.__cause__ = httpx.ReadTimeout("read timed out")
        assert classify_error(exc) is ErrorKind.TIMEOUT

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("Connection refused"),
            ConnectionResetError("Connection reset by peer"),
            Exception("Connection error."),
        ],
    )
    def test_connection_failures(self, exc: Exception) -> None:
        assert classify_error(exc) is ErrorKind.CONNECTION

    def test_status_on_wrapped_cause(self) -> None:
        wrapper = Exception("provider failed")
        wrapper.__cause__ = _status_error(429)
        assert http_status(wrapper) == 429
        assert classify_error(wrapper) is ErrorKind.RATE_LIMIT

    def test_string_status_is_ignored(self) -> None:
        exc = Exception("something benign")
        exc.status = "RESOURCE_EXHAUSTED"  # type: ignore[attr-defined]
        assert http_status(exc) is None
        assert classify_error(exc) is ErrorKind.CLIENT_ERROR

    def test_bool_status_is_ignored(self) -> None:
        exc = Exception("benign")
        exc.status_code = True  # type: ignore[attr-defined]
        assert http_status(exc) is None

    def test_google_api_error_int_code(self) -> None:
        from google.genai.errors import APIError

        exhausted = APIError(
            429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}
        )
        internal = APIError(500, {"error": {"message": "boom", "status": "INTERNAL"}})
        assert classify_error(exhausted) is ErrorKind.RATE_LIMIT
        assert classify_error(internal) is ErrorKind.SERVER_ERROR

    @pytest.mark.parametrize(
        ("message", "kind"),
        [
            ("Quota exceeded for the day", ErrorKind.RATE_LIMIT),
            ("Resource has been exhausted", ErrorKind.RATE_LIMIT),
            ("rate limit reached for requests", ErrorKind.RATE_LIMIT),
            ("Error code: 429 - too many", ErrorKind.RATE_LIMIT),
            ("The server is overloaded", ErrorKind.SERVER_ERROR),
            ("Internal error encountered", ErrorKind.SERVER_ERROR),
            ("Internal Server Error", ErrorKind.SERVER_ERROR),
            ("status code: 502", ErrorKind.SERVER_ERROR),
            ("503 Service Unavailable", ErrorKind.SERVER_ERROR),
            ("bad schema", ErrorKind.CLIENT_ERROR),
        ],
    )
    def test_message_fallbacks(self, message: str, kind: ErrorKind) -> None:
        assert classify_error(Exception(message)) is kind

    @pytest.mark.parametrize(
        "message", ["see line 502 of file.py", "you requested 132429 tokens"]
    )
    def test_stray_numbers_do_not_match(self, message: str) -> None:
        assert not is_server_error_message(message)
        assert not is_rate_limit_message(message)
        assert classify_error(ValueError(message)) is ErrorKind.CLIENT_ERROR

    def test_value_error_is_client_error(self) -> None:
        assert classify_error(ValueError("Invalid argument")) is ErrorKind.CLIENT_ERROR

    def test_cause_cycle_terminates(self) -> None:
        first = Exception("first")
        second = Exception("second")
        first.__cause__ = second
        second.__cause__ = first
        assert not is_timeout_error(first)
        assert not is_connection_error(first)
        assert classify_error(first) is ErrorKind.CLIENT_ERROR

    def test_retryable_property_and_default_decision(self) -> None:
        assert ErrorKind.OTHER.retryable
        assert not ErrorKind.CLIENT_ERROR.retryable
        assert default_decision(_status_error(503)) is Decision.RETRY
        assert default_decision(_status_error(400)) is Decision.FAIL


def _with_headers(headers: dict[str, str], on_response: bool = True) -> Exception:
    exc = _status_error(429, "rate limited")
    if on_response:
        exc.response = SimpleNamespace(headers=headers)  # type: ignore[attr-defined]
    else:
        exc.headers = headers  # type: ignore[attr-defined]
    return exc


class TestParseRetryAfter:
    def test_seconds_form(self) -> None:
        assert parse_retry_after(_with_headers({"retry-after": "45"})) == 45.0

    def test_title_case_header_on_exception(self) -> None:
        exc = _with_headers({"Retry-After": "2.5"}, on_response=False)
        assert parse_retry_after(exc) == 2.5

    def test_milliseconds_header_wins(self) -> None:
        exc = _with_headers({"retry-after-ms": "1500", "retry-after": "9"})
        assert parse_retry_after(exc) == 1.5

    def test_http_date_form(self) -> None:
        exc = _with_headers({"retry-after": "Wed, 21 Oct 2099 07:28:00 GMT"})
        now = dt.datetime(2099, 10, 21, 7, 27, 0, tzinfo=dt.UTC)
        assert parse_retry_after(exc, now=now) == 60.0

    def test_past_date_is_zero(self) -> None:
        exc = _with_headers({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
        assert parse_retry_after(exc) == 0.0

    def test_negative_seconds_are_zero(self) -> None:
        assert parse_retry_after(_with_headers({"retry-after": "-3"})) == 0.0

    @pytest.mark.parametrize(
        "exc",
        [
            None,
            Exception("no headers"),
            _with_headers({}),
            _with_headers({"retry-after": "soon"}),
        ],
    )
    def test_unusable_values_are_none(self, exc: Exception | None) -> None:
        assert parse_retry_after(exc) is None


class TestRetryPolicy:
    def test_formula(self) -> None:
        policy = RetryPolicy(backoff_base=0.5)
        # 0.5 * 2.0**0 + 0.75
        assert policy.backoff(0, "rate_limit", jitter=0.75) == pytest.approx(1.25)

    def test_exponential_growth(self) -> None:
        policy = RetryPolicy(backoff_base=1.0, multipliers={"timeout": 2.0})
        delays = [policy.backoff(n, "timeout", jitter=0.0) for n in range(3)]
        assert delays == pytest.approx([1.0, 2.0, 4.0])

    def test_unknown_kind_uses_two(self) -> None:
        policy = RetryPolicy(backoff_base=0.5, multipliers={})
        assert policy.backoff(1, "unknown", jitter=0.5) == pytest.approx(1.5)

    def test_random_jitter_within_bounds(self) -> None:
        policy = RetryPolicy(backoff_base=0.0, jitter_min=0.5, jitter_max=1.0)
        for _ in range(20):
            assert 0.5 <= policy.backoff(0, "timeout") <= 1.0

    def test_cap_applies(self) -> None:
        policy = RetryPolicy(backoff_base=10.0, multipliers={"timeout": 10.0})
        assert policy.backoff(5, "timeout") == policy.backoff_cap

    def test_huge_attempt_is_finite_and_capped(self) -> None:
        policy = RetryPolicy(multipliers={"server_error": 1e10})
        result = policy.backoff(5000, "server_error")
        assert result == policy.backoff_cap

    def test_retry_after_raises_the_delay_under_the_cap(self) -> None:
        policy = RetryPolicy()
        assert policy.delay(0, "rate_limit", 45.0, jitter=0.0) == 45.0
        assert policy.delay(0, "rate_limit", 9999.0, jitter=0.0) == 120.0
        assert policy.delay(0, "rate_limit", 0.1, jitter=0.0) == 0.5

    def test_from_mapping(self) -> None:
        policy = RetryPolicy.from_mapping(
            {
                "max_attempts": 5,
                "validation_attempts": 2,
                "backoff_base": 1.0,
                "backoff_cap": 60,
                "backoff_multipliers": {"timeout": 3.0},
                "jitter": {"min": 0.0, "max": 0.1},
            }
        )
        assert policy.max_attempts == 5
        assert policy.validation_attempts == 2
        assert policy.backoff_cap == 60.0
        assert policy.multipliers["timeout"] == 3.0
        assert policy.multipliers["rate_limit"] == 2.0
        assert (policy.jitter_min, policy.jitter_max) == (0.0, 0.1)

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            {"backoff_base": "x", "backoff_multipliers": None, "jitter": None},
            {"max_attempts": "fast", "validation_attempts": True},
        ],
    )
    def test_from_mapping_tolerates_malformed_values(self, raw: Any) -> None:
        policy = RetryPolicy.from_mapping(raw)
        assert policy == RetryPolicy()
        assert 0.0 <= policy.backoff(3, "server_error") <= policy.backoff_cap

    def test_from_mapping_keeps_at_least_one_attempt(self) -> None:
        policy = RetryPolicy.from_mapping({"max_attempts": 0})
        assert policy.max_attempts == 1


class _Calls:
    """An async callable that raises or returns the queued outcomes in order."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.count = 0

    async def __call__(self) -> Any:
        self.count += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Gate:
    def __init__(self) -> None:
        self.acquired = 0
        self.successes = 0
        self.errors: list[bool] = []

    async def acquire(self) -> float:
        self.acquired += 1
        return 0.0

    def report_success(self) -> None:
        self.successes += 1

    def report_error(self, is_rate_limit: bool = False) -> None:
        self.errors.append(is_rate_limit)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def _no_jitter() -> RetryPolicy:
    return RetryPolicy(jitter_min=0.0, jitter_max=0.0)


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the ladder's sleeps instead of sleeping."""
    record: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        record.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return record


class TestCallWithRetry:
    def test_success_on_first_attempt(self, slept: list[float]) -> None:
        calls = _Calls("ok")
        assert _run(call_with_retry(calls)) == "ok"
        assert calls.count == 1
        assert slept == []

    def test_retries_a_rate_limit_and_recovers_usage(self, slept: list[float]) -> None:
        exc = _status_error(429, "Rate limit")
        exc.body = {"usage": {"total_tokens": 50}}  # type: ignore[attr-defined]
        calls = _Calls(exc, "success")
        usages: list[Usage] = []

        result = _run(
            call_with_retry(
                calls,
                policy=_no_jitter(),
                hooks=RetryHooks(on_usage=usages.append),
            )
        )

        assert result == "success"
        assert calls.count == 2
        assert usages == [Usage(total_tokens=50)]
        assert slept == [0.5]

    def test_non_retryable_error_raises_at_once(self, slept: list[float]) -> None:
        calls = _Calls(ValueError("Invalid input"))
        with pytest.raises(ValueError, match="Invalid input"):
            _run(call_with_retry(calls))
        assert calls.count == 1
        assert slept == []

    def test_raises_after_max_attempts(self, slept: list[float]) -> None:
        calls = _Calls(_status_error(500, "Server error"))
        with pytest.raises(Exception, match="Server error"):
            _run(call_with_retry(calls, policy=RetryPolicy(max_attempts=3)))
        assert calls.count == 3
        assert len(slept) == 2

    def test_backoff_grows_per_transient_failure(self, slept: list[float]) -> None:
        calls = _Calls(_status_error(500), _status_error(500), _status_error(500), 1)
        _run(call_with_retry(calls, policy=_no_jitter()))
        assert slept == [0.5, 1.0, 2.0]

    def test_honors_retry_after(self, slept: list[float]) -> None:
        calls = _Calls(_with_headers({"retry-after": "45"}), "ok")
        assert _run(call_with_retry(calls, policy=_no_jitter())) == "ok"
        assert slept == [45.0]

    def test_gate_sees_every_attempt(self, slept: list[float]) -> None:
        gate = _Gate()
        calls = _Calls(_status_error(429), httpx.ConnectError("refused"), "ok")
        _run(call_with_retry(calls, hooks=RetryHooks(gate=gate)))
        assert gate.acquired == 3
        assert gate.errors == [True, False]
        assert gate.successes == 1

    def test_retry_after_counts_as_rate_signal(self, slept: list[float]) -> None:
        gate = _Gate()
        exc = _with_headers({"retry-after": "1"})
        exc.status_code = 408  # type: ignore[attr-defined]
        _run(call_with_retry(_Calls(exc, "ok"), hooks=RetryHooks(gate=gate)))
        assert gate.errors == [True]

    def test_validation_errors_use_their_own_budget(self, slept: list[float]) -> None:
        calls = _Calls(_ValidationFailure("bad json"))
        with pytest.raises(_ValidationFailure):
            _run(
                call_with_retry(
                    calls,
                    policy=RetryPolicy(max_attempts=8, validation_attempts=3),
                    validation_errors=(_ValidationFailure,),
                )
            )
        assert calls.count == 3

    def test_validation_and_transient_budgets_are_separate(
        self, slept: list[float]
    ) -> None:
        calls = _Calls(
            _ValidationFailure("bad"),
            _status_error(503),
            _status_error(503),
            _ValidationFailure("bad"),
            "ok",
        )
        result = _run(
            call_with_retry(
                calls,
                policy=RetryPolicy(validation_attempts=3),
                validation_errors=(_ValidationFailure,),
            )
        )
        assert result == "ok"
        assert calls.count == 5

    def test_validation_error_skips_on_error(self, slept: list[float]) -> None:
        seen: list[BaseException] = []

        def on_error(exc: BaseException) -> Decision:
            seen.append(exc)
            return Decision.FAIL

        calls = _Calls(_ValidationFailure("bad"), "ok")
        result = _run(
            call_with_retry(
                calls,
                validation_errors=(_ValidationFailure,),
                hooks=RetryHooks(on_error=on_error),
            )
        )
        assert result == "ok"
        assert seen == []

    def test_validation_usage_attached_to_the_error(self, slept: list[float]) -> None:
        failure = _ValidationFailure("bad")
        failure.usage = Usage(total_tokens=7)  # type: ignore[attr-defined]
        usages: list[Usage] = []
        _run(
            call_with_retry(
                _Calls(failure, "ok"),
                validation_errors=(_ValidationFailure,),
                hooks=RetryHooks(on_usage=usages.append),
            )
        )
        assert usages == [Usage(total_tokens=7)]

    def test_on_error_fail_stops_a_retryable_error(self, slept: list[float]) -> None:
        calls = _Calls(_status_error(429))
        with pytest.raises(Exception, match="^error$"):
            _run(
                call_with_retry(
                    calls, hooks=RetryHooks(on_error=lambda exc: Decision.FAIL)
                )
            )
        assert calls.count == 1

    def test_on_error_retry_overrides_a_client_error(self, slept: list[float]) -> None:
        calls = _Calls(_status_error(400), "ok")
        result = _run(
            call_with_retry(
                calls, hooks=RetryHooks(on_error=lambda exc: Decision.RETRY)
            )
        )
        assert result == "ok"

    def test_switch_key_retries_without_sleep(self, slept: list[float]) -> None:
        calls = _Calls(_status_error(429), "ok")
        _run(
            call_with_retry(
                calls, hooks=RetryHooks(on_error=lambda exc: Decision.SWITCH_KEY)
            )
        )
        assert calls.count == 2
        assert slept == []

    def test_switch_key_counts_against_the_budget(self, slept: list[float]) -> None:
        calls = _Calls(_status_error(429))
        with pytest.raises(Exception, match="^error$"):
            _run(
                call_with_retry(
                    calls,
                    policy=RetryPolicy(max_attempts=3),
                    hooks=RetryHooks(on_error=lambda exc: Decision.SWITCH_KEY),
                )
            )
        assert calls.count == 3

    def test_wait_uses_the_given_delay_and_no_attempt(self, slept: list[float]) -> None:
        decisions = [ErrorDecision(Decision.WAIT, 3600.0)] * 3

        def on_error(exc: BaseException) -> ErrorDecision:
            return decisions.pop(0)

        calls = _Calls(_status_error(429), _status_error(429), _status_error(429), 1)
        result = _run(
            call_with_retry(
                calls,
                policy=RetryPolicy(max_attempts=2),
                hooks=RetryHooks(on_error=on_error),
            )
        )
        assert result == 1
        assert calls.count == 4
        assert slept == [3600.0, 3600.0, 3600.0]

    def test_wait_without_delay_falls_back_to_retry_after(
        self, slept: list[float]
    ) -> None:
        calls = _Calls(_with_headers({"retry-after": "30"}), "ok")
        _run(
            call_with_retry(
                calls,
                policy=_no_jitter(),
                hooks=RetryHooks(on_error=lambda exc: Decision.WAIT),
            )
        )
        assert slept == [30.0]

    def test_async_on_error_hook(self, slept: list[float]) -> None:
        async def on_error(exc: BaseException) -> Decision:
            return Decision.SWITCH_KEY

        calls = _Calls(_status_error(500), "ok")
        assert _run(call_with_retry(calls, hooks=RetryHooks(on_error=on_error))) == "ok"
        assert slept == []

    def test_on_retry_receives_events(self, slept: list[float]) -> None:
        events: list[RetryEvent] = []
        calls = _Calls(_status_error(503), "ok")
        _run(
            call_with_retry(
                calls,
                policy=_no_jitter(),
                hooks=RetryHooks(on_retry=events.append),
                label="page 1",
            )
        )
        assert len(events) == 1
        event = events[0]
        assert (event.attempt, event.kind, event.decision) == (
            1,
            ErrorKind.SERVER_ERROR,
            Decision.RETRY,
        )
        assert event.delay == 0.5
        assert event.label == "page 1"

    def test_injected_sleep(self) -> None:
        record: list[float] = []

        async def sleep(seconds: float) -> None:
            record.append(seconds)

        calls = _Calls(_status_error(500), "ok")
        _run(call_with_retry(calls, policy=_no_jitter(), hooks=RetryHooks(sleep=sleep)))
        assert record == [0.5]

    def test_cancellation_is_not_retried(self, slept: list[float]) -> None:
        cancelled = asyncio.CancelledError()
        cancelled.__context__ = httpx.ReadTimeout("read timed out")
        gate = _Gate()
        calls = _Calls(cancelled, "ok")
        with pytest.raises(asyncio.CancelledError):
            _run(call_with_retry(calls, hooks=RetryHooks(gate=gate)))
        assert calls.count == 1
        assert gate.errors == []

    def test_cancelled_sleep_ends_the_ladder(self) -> None:
        async def main() -> None:
            calls = _Calls(_status_error(500), "ok")
            task = asyncio.create_task(
                call_with_retry(calls, policy=RetryPolicy(backoff_base=60.0))
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert calls.count == 1

        asyncio.run(main())
