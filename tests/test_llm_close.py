"""Regression tests for ``LLMClientBase.close`` teardown semantics.

Managers are built per item and their ``close`` is invoked at item end. An
earlier ``close`` closed the httpx client behind the chat model, poisoning
the connection pool so that every subsequent item failed with
``APIConnectionError``. These tests pin the fix: ``close`` must never close
the httpx client a manager still holds.

The OpenAI-family managers now hand ChatOpenAI an ``httpx.Timeout`` (per-phase
timeouts), which is unhashable and therefore bypasses langchain-openai's
process-wide ``@lru_cache`` client sharing: each manager gets its OWN httpx
client, closed on GC. The scalar-timeout providers (anthropic, google) still
share the cached client, so ``close`` staying a no-op remains load-bearing.

All offline: managers are built with a dummy api_key and no API call is made.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
from langchain_openai import ChatOpenAI

from llm.base import LLMClientBase


def _build(model: str = "gpt-5-mini") -> LLMClientBase:
    """Build a bare base manager for the openai path without any API call."""
    return LLMClientBase(model, provider="openai", api_key="test-dummy-key")


def _httpx_client(mgr: LLMClientBase) -> Any:
    """Return the httpx client behind an openai chat model.

    ``chat_model`` is typed as the abstract ``BaseChatModel``; the openai path
    builds a concrete ``ChatOpenAI`` whose ``root_client`` holds the openai SDK
    client that wraps the httpx client.
    """
    return cast(ChatOpenAI, mgr.chat_model).root_client._client


def test_openai_managers_get_per_manager_uncached_clients() -> None:
    """Per-phase ``httpx.Timeout`` bypasses the lru_cache client sharing.

    Each openai-family manager builds its own httpx client; ``close`` must
    leave every manager's client open regardless.
    """
    mgr1 = _build()
    mgr2 = _build()

    client1 = _httpx_client(mgr1)
    client2 = _httpx_client(mgr2)
    # The unhashable httpx.Timeout skips langchain-openai's cache, so the
    # managers no longer share one client object.
    assert client1 is not client2, "expected per-manager uncached httpx clients"
    assert isinstance(cast(ChatOpenAI, mgr1.chat_model).request_timeout, httpx.Timeout)
    assert not client1.is_closed
    assert not client2.is_closed

    mgr1.close()

    # close() is a no-op: both clients must remain open and usable.
    assert not client1.is_closed
    assert not client2.is_closed
    assert _httpx_client(mgr2) is client2


def test_close_is_idempotent_and_returns_none() -> None:
    """close() is a no-op: it returns None and never raises, even repeated."""
    mgr = _build()
    client = _httpx_client(mgr)

    # ``close`` is typed ``-> None``, so a value-context check such as
    # ``assert mgr.close() is None`` is statically redundant; call it directly.
    # The guarantees under test are that repeated calls never raise and that
    # the manager's httpx client is left open.
    mgr.close()
    mgr.close()
    assert not client.is_closed
