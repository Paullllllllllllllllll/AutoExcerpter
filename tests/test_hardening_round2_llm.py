"""Regression tests for the round-2 hardening fixes.

Covers the config and LLM robustness fixes applied in this round:

1. config.app._get_float — malformed ``estimate_smoothing`` falls back to the
   default rather than raising at import time.
3. config.accessors.get_rate_limits — non-positive rate-limit entries are
   filtered out; an all-invalid list falls back to the defaults.
4. llm.base.LLMClientBase._calculate_backoff — a huge attempt count returns a
   finite value bounded by ``BACKOFF_CAP_S`` instead of raising OverflowError.
5. llm.prompts.strip_markdown_code_block — an uppercase ```JSON fence is
   stripped case-insensitively.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from config.accessors import get_rate_limits
from config.app import DAILY_TOKEN_ESTIMATE_SMOOTHING, _get_float
from llm.base import BACKOFF_CAP_S, LLMClientBase
from llm.prompts import strip_markdown_code_block


# ============================================================================
# Fix 1: _get_float robustness (estimate_smoothing)
# ============================================================================
class TestGetFloatEstimateSmoothing:
    """The guarded float helper must never raise on malformed config."""

    def test_non_numeric_falls_back_to_default(self) -> None:
        """A non-numeric value returns the supplied default, not an error."""
        data: dict[str, Any] = {"estimate_smoothing": "not-a-number"}
        assert _get_float(data, "estimate_smoothing", 0.3) == 0.3

    def test_none_value_falls_back_to_default(self) -> None:
        """A None value returns the default."""
        data: dict[str, Any] = {"estimate_smoothing": None}
        assert _get_float(data, "estimate_smoothing", 0.3) == 0.3

    def test_valid_value_parsed(self) -> None:
        """A valid numeric value is parsed as a float."""
        data: dict[str, Any] = {"estimate_smoothing": "0.5"}
        assert _get_float(data, "estimate_smoothing", 0.3) == 0.5

    def test_module_constant_imports_cleanly(self) -> None:
        """config.app imported without raising and produced a float constant."""
        assert isinstance(DAILY_TOKEN_ESTIMATE_SMOOTHING, float)


# ============================================================================
# Fix 3: get_rate_limits filters non-positive entries
# ============================================================================
def _loader_with_rate_limits(raw_limits: Any) -> MagicMock:
    """Build a mock ConfigLoader returning the given rate_limits config."""
    loader = MagicMock()
    loader.get_concurrency_config.return_value = {
        "api_requests": {"rate_limits": raw_limits}
    }
    return loader


class TestGetRateLimitsFiltering:
    """Non-positive counts/windows must be dropped before reaching RateLimiter."""

    def test_zero_and_negative_entries_filtered(self, monkeypatch) -> None:
        """Zero/negative entries are skipped, valid entries retained."""
        import config.accessors as accessors

        loader = _loader_with_rate_limits(
            [[120, 1], [0, 60], [15000, 0], [-5, 10], [10, -1]]
        )
        monkeypatch.setattr(accessors, "get_config_loader", lambda: loader)

        assert get_rate_limits() == [(120, 1)]

    def test_all_invalid_falls_back_to_defaults(self, monkeypatch) -> None:
        """When nothing valid remains, the defaults are returned."""
        import config.accessors as accessors
        from config.constants import DEFAULT_RATE_LIMITS

        loader = _loader_with_rate_limits([[0, 0], [-1, 5], [5, -1]])
        monkeypatch.setattr(accessors, "get_config_loader", lambda: loader)

        assert get_rate_limits() == list(DEFAULT_RATE_LIMITS)


# ============================================================================
# Fix 4: _calculate_backoff cannot overflow
# ============================================================================
class TestCalculateBackoffOverflow:
    """A very large attempt count must not raise and must stay capped."""

    def test_huge_attempt_returns_finite_capped_value(self) -> None:
        """attempt=5000 returns a finite value <= BACKOFF_CAP_S without raising."""
        # ``self`` is unused by the method, so any object suffices.
        dummy = cast(LLMClientBase, object())
        result = LLMClientBase._calculate_backoff(dummy, 5000, "server_error")
        assert result <= BACKOFF_CAP_S
        assert result == result  # not NaN
        assert result != float("inf")


# ============================================================================
# Fix 5: uppercase ```JSON fence stripped
# ============================================================================
class TestStripMarkdownCodeBlock:
    """The JSON fence must be stripped regardless of case."""

    def test_uppercase_json_fence_stripped(self) -> None:
        """```JSON ... ``` yields the bare payload."""
        payload = '{"key": "value"}'
        wrapped = f"```JSON\n{payload}\n```"
        assert strip_markdown_code_block(wrapped) == payload

    def test_lowercase_json_fence_still_stripped(self) -> None:
        """Lowercase behavior is preserved."""
        payload = '{"key": "value"}'
        wrapped = f"```json\n{payload}\n```"
        assert strip_markdown_code_block(wrapped) == payload
