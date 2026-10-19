"""Closing the call env closes the hooked httpx clients of every caller built."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from autoexcerpter.llm import CallEnv
from tests.llm import helpers

HttpxClient = httpx.Client | httpx.AsyncClient


def _httpx_clients(model: Any) -> list[HttpxClient]:
    """Return the httpx clients under the SDK clients of a chat model."""
    found: list[HttpxClient] = []
    for attr in ("root_client", "root_async_client", "_client", "_async_client"):
        inner = getattr(getattr(model, attr, None), "_client", None)
        if isinstance(inner, httpx.Client | httpx.AsyncClient) and inner not in found:
            found.append(inner)
    return found


@pytest.mark.parametrize(
    ("name", "provider", "key"),
    [
        ("gpt-5.6-terra", "openai", "OPENAI_API_KEY"),
        ("claude-sonnet-4-5", "anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_closing_the_env_closes_every_hooked_httpx_client(
    monkeypatch: pytest.MonkeyPatch, name: str, provider: str, key: str
) -> None:
    monkeypatch.setenv(key, "test-key")
    env = CallEnv.create()
    env.prepare("transcription", helpers.phase(name, provider))
    caller = env.caller("summary", helpers.phase(name, provider))
    clients = [
        *_httpx_clients(env.prepared["transcription"].model),
        *_httpx_clients(caller.model),
    ]
    assert len(clients) == 4
    assert not any(client.is_closed for client in clients)

    asyncio.run(env.aclose())

    assert all(client.is_closed for client in clients)
    assert env.callers == []
    assert env.prepared == {}
