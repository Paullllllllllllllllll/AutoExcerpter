"""Tests for common/http_timeouts.py: the per-phase httpx timeout builder."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from autoexcerpter.common.http_timeouts import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    build_httpx_timeout,
)


class TestBuildHttpxTimeout:
    """Tests for build_httpx_timeout."""

    def test_none_is_passed_through(self) -> None:
        """No request timeout means the SDK defaults stay untouched."""
        assert build_httpx_timeout(None) is None

    def test_defaults_apply_to_the_other_phases(self) -> None:
        """The scalar lands on read only; connect/write/pool use defaults."""
        timeout = build_httpx_timeout(900)

        assert timeout is not None
        assert timeout.read == 900
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT == 10.0
        assert timeout.write == DEFAULT_WRITE_TIMEOUT == 30.0
        assert timeout.pool == DEFAULT_POOL_TIMEOUT == 30.0

    def test_phase_values_are_honored(self) -> None:
        """Given per-phase values replace the built-in defaults."""
        timeout = build_httpx_timeout(120, connect=3, write=7.5, pool=12)

        assert timeout is not None
        assert timeout.read == 120
        assert timeout.connect == 3.0
        assert timeout.write == 7.5
        assert timeout.pool == 12.0

    @pytest.mark.parametrize("bad", [0, -1, "nonsense", None, True, False, [5]])
    def test_invalid_phase_values_fall_back_to_defaults(self, bad: Any) -> None:
        """Zero, negative, non-numeric and bool values are ignored."""
        timeout = build_httpx_timeout(900, connect=bad, write=bad, pool=bad)

        assert timeout is not None
        assert timeout.read == 900
        assert timeout.connect == DEFAULT_CONNECT_TIMEOUT
        assert timeout.write == DEFAULT_WRITE_TIMEOUT
        assert timeout.pool == DEFAULT_POOL_TIMEOUT

    def test_timeout_is_unhashable(self) -> None:
        """httpx.Timeout is unhashable, bypassing langchain's client cache.

        Each chat model therefore builds its own httpx client. The bypass
        lives in ``langchain_openai/chat_models/_client_utils.py``; a failure
        here means httpx made Timeout hashable and the client-sharing
        semantics changed.
        """
        timeout = build_httpx_timeout(900)

        assert isinstance(timeout, httpx.Timeout)
        with pytest.raises(TypeError):
            hash(timeout)
