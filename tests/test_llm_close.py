"""Regression tests for ``LLMClientBase.close`` teardown semantics.

Managers are built per item and their ``close`` is invoked at item end. The
langchain providers we use (openai, anthropic, custom, openrouter -> ChatOpenAI;
anthropic -> ChatAnthropic) draw their httpx client from a process-wide
``@lru_cache``, so every manager for a given provider shares ONE httpx client.
An earlier ``close`` closed that shared client, poisoning the pool so that every
subsequent item failed with ``APIConnectionError``. These tests pin the fix:
``close`` must never close the shared httpx client.

All offline: managers are built with a dummy api_key and no API call is made.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_openai import ChatOpenAI

from llm.base import LLMClientBase


def _build(model: str = "gpt-5-mini") -> LLMClientBase:
    """Build a bare base manager for the openai path without any API call."""
    return LLMClientBase(model, provider="openai", api_key="test-dummy-key")


def _shared_httpx_client(mgr: LLMClientBase) -> Any:
    """Return the process-shared httpx client behind an openai chat model.

    ``chat_model`` is typed as the abstract ``BaseChatModel``; the openai path
    builds a concrete ``ChatOpenAI`` whose ``root_client`` holds the openai SDK
    client that wraps the cached httpx client.
    """
    return cast(ChatOpenAI, mgr.chat_model).root_client._client


def test_close_leaves_shared_httpx_client_usable() -> None:
    """mgr1.close() must not close the httpx client shared with mgr2.

    Reproduces the sequential-item bug: item 1's teardown once closed the
    cached httpx client that item 2 depends on.
    """
    mgr1 = _build()
    mgr2 = _build()

    # ChatOpenAI draws its httpx client from langchain-openai's process-wide
    # cache, so both managers reference the same underlying client object.
    client1 = _shared_httpx_client(mgr1)
    client2 = _shared_httpx_client(mgr2)
    assert client1 is client2, "expected a process-shared cached httpx client"
    assert not client2.is_closed

    mgr1.close()

    # The shared client must remain open and usable for the second manager.
    assert not client2.is_closed
    assert _shared_httpx_client(mgr2) is client2
    assert not _shared_httpx_client(mgr2).is_closed


def test_close_is_idempotent_and_returns_none() -> None:
    """close() is a no-op: it returns None and never raises, even repeated."""
    mgr = _build()
    client = _shared_httpx_client(mgr)

    # ``close`` is typed ``-> None``, so a value-context check such as
    # ``assert mgr.close() is None`` is statically redundant; call it directly.
    # The guarantees under test are that repeated calls never raise and that the
    # shared httpx client is left open.
    mgr.close()
    mgr.close()
    assert not client.is_closed
