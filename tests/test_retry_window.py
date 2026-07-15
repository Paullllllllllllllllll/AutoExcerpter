"""Tests for the time-based retry window and the incomplete-run warning.

Covers two features:

- ``LLMClientBase._invoke_with_retry`` time horizon (``retry.max_elapsed`` ->
  ``MAX_ELAPSED_S`` -> per-instance ``self.max_elapsed``): a retryable error is
  retried while EITHER attempts remain OR the window is still open, and stops
  only when BOTH are exhausted. ``self.max_elapsed == 0`` (the class default,
  used by ``__new__``-built test clients) restores attempts-only behavior.
- ``main._warn_incomplete_items``: a prominent stdout warning naming items that
  finished incomplete, silent when every item completed.
"""

from __future__ import annotations

import threading
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

import llm.base as base_module
from llm.base import LLMClientBase
from main import _warn_incomplete_items


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
    # Neutralize collaborators so only the retry-window control flow is exercised
    # (and so the number of time.monotonic() calls stays deterministic).
    client._wait_for_rate_limit = MagicMock()  # type: ignore[method-assign]
    client._extract_tokens_from_exception = MagicMock()  # type: ignore[method-assign]
    client._report_error = MagicMock()  # type: ignore[method-assign]
    client._parse_retry_after = MagicMock(return_value=None)  # type: ignore[method-assign]
    client._calculate_backoff = MagicMock(return_value=0.0)  # type: ignore[method-assign]
    return client


def _server_error() -> Exception:
    """Build a retryable (HTTP 500) exception."""
    exc = Exception("Internal Server Error")
    exc.status_code = 500  # type: ignore[attr-defined]
    return exc


# ============================================================================
# Retry window: _invoke_with_retry
# ============================================================================
class TestRetryWindow:
    """Time-based retry horizon semantics."""

    def test_retries_past_max_attempts_while_inside_window(self) -> None:
        """A retryable error keeps retrying past max_attempts while the elapsed
        time is still inside max_elapsed."""
        # max_retries=1 (legacy would allow only 2 attempts); window open.
        client = _make_client(max_retries=1, max_elapsed=900.0)
        mock_model = MagicMock()
        # Three failures then success: legacy attempts-only would raise on the
        # 2nd; the open window must carry it through to the 4th call.
        mock_model.invoke.side_effect = [
            _server_error(),
            _server_error(),
            _server_error(),
            "success",
        ]

        with (
            patch("llm.base.time.monotonic", return_value=0.0),
            patch("llm.base.time.sleep"),
        ):
            result = client._invoke_with_retry(mock_model, [], {}, "test")

        assert result == "success"
        assert mock_model.invoke.call_count == 4

    def test_raises_when_window_exhausted(self) -> None:
        """Once both attempts and the window are exhausted, the error raises."""
        # No attempt budget; the window alone governs continuation.
        client = _make_client(max_retries=0, max_elapsed=900.0)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _server_error()

        # monotonic: start=0; after 1st failure elapsed=500 (< 900, retry);
        # after 2nd failure elapsed=1000 (>= 900, raise).
        with (
            patch("llm.base.time.monotonic", side_effect=[0.0, 500.0, 1000.0]),
            patch("llm.base.time.sleep"),
            pytest.raises(Exception, match="Internal Server Error"),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 2

    def test_legacy_attempts_only_when_window_disabled(self) -> None:
        """max_elapsed absent/0 -> attempts-only: exactly max_attempts tries."""
        # Window disabled (0.0); 3 total attempts from max_retries=2.
        client = _make_client(max_retries=2, max_elapsed=0.0)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _server_error()

        with (
            patch("llm.base.time.monotonic", return_value=0.0),
            patch("llm.base.time.sleep"),
            pytest.raises(Exception, match="Internal Server Error"),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        # max_retries=2 -> attempts 0, 1, 2 then raise.
        assert mock_model.invoke.call_count == 3

    def test_window_disabled_is_the_class_default(self) -> None:
        """A ``__new__``-built client (no __init__) inherits the class default of
        0.0, so unit-test clients are legacy attempts-only by default."""
        client = _make_client(max_retries=2)  # no explicit max_elapsed
        assert client.max_elapsed == 0.0
        mock_model = MagicMock()
        mock_model.invoke.side_effect = _server_error()

        with (
            patch("llm.base.time.monotonic", return_value=0.0),
            patch("llm.base.time.sleep"),
            pytest.raises(Exception, match="Internal Server Error"),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        assert mock_model.invoke.call_count == 3

    def test_non_retryable_raises_immediately(self) -> None:
        """A non-retryable error raises on the first attempt even with an open
        window."""
        client = _make_client(max_retries=5, max_elapsed=900.0)
        mock_model = MagicMock()
        mock_model.invoke.side_effect = ValueError("Invalid input")

        with (
            patch("llm.base.time.monotonic", return_value=0.0),
            patch("llm.base.time.sleep"),
            pytest.raises(ValueError, match="Invalid input"),
        ):
            client._invoke_with_retry(mock_model, [], {}, "test")

        mock_model.invoke.assert_called_once()

    def test_config_constant_matches_yaml_and_class_default_is_disabled(self) -> None:
        """MAX_ELAPSED_S mirrors the configured retry.max_elapsed, while the class
        default stays 0.0 (legacy) for __init__-bypassing clients."""
        expected = float(base_module._RETRY_CONFIG.get("max_elapsed", 0))
        assert expected == base_module.MAX_ELAPSED_S
        assert LLMClientBase.max_elapsed == 0.0


# ============================================================================
# Incomplete-run warning: main._warn_incomplete_items
# ============================================================================
class TestIncompleteRunWarning:
    """Prominent end-of-run warning for incomplete items."""

    def test_warning_printed_when_items_incomplete(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Names each incomplete item and points at --resume / error placeholders."""
        _warn_incomplete_items(["Book_A", "Book_B"])
        out = capsys.readouterr().out
        assert "INCOMPLETE" in out
        assert "Book_A" in out
        assert "Book_B" in out
        assert "--resume" in out
        assert "placeholder" in out.lower()

    def test_no_warning_when_all_complete(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Silent when there are no incomplete items."""
        _warn_incomplete_items([])
        assert capsys.readouterr().out == ""
