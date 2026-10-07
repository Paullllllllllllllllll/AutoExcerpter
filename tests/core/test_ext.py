"""Tests for ext.py: pass-through hooks and the attempt context."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter import ext
from autoexcerpter.common.usage import Usage
from autoexcerpter.ext import Decision, Identity
from autoexcerpter.settings import Endpoint, Settings
from autoexcerpter.spec import resolve_spec

_SETTINGS = Settings(
    endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
)


class _Builder:
    def __init__(self) -> None:
        self.identities: list[Identity] = []

    def __call__(self, identity: Identity) -> dict[str, Any]:
        self.identities.append(identity)
        return {"key_env": identity.api_key_env, "n": len(self.identities)}


def _status_error(status: int) -> Exception:
    exc = Exception(f"status {status}")
    exc.status_code = status  # type: ignore[attr-defined]
    return exc


def test_startup_hooks_pass_through() -> None:
    parser = argparse.ArgumentParser()
    ext.register_args(parser)
    assert set(vars(parser.parse_args([])).values()) <= {None}
    assert list(ext.wizard_steps()) == []
    spec = resolve_spec({"input": Path("x.pdf")}).spec
    ext.startup(spec, settings=_SETTINGS, args=argparse.Namespace())
    assert ext.batch_allowed(spec) is True


def test_identity_uses_the_settings_key_variable() -> None:
    attempt = ext.attempt(
        "transcription", "openai", 100, settings=_SETTINGS, build_client=_Builder()
    )
    assert attempt.identity == Identity("openai", "OPENAI_API_KEY", "transcription")
    assert attempt.identity.group is None
    custom = ext.attempt(
        "summary",
        "custom",
        settings=_SETTINGS,
        build_client=_Builder(),
        endpoint="local",
    )
    assert custom.identity.api_key_env == "LOCAL_KEY"


def test_success_builds_the_client_commits_and_releases() -> None:
    builder = _Builder()

    async def run() -> ext.Attempt[dict[str, Any]]:
        async with ext.attempt(
            "summary", "openai", settings=_SETTINGS, build_client=builder
        ) as call:
            assert call.client == {"key_env": "OPENAI_API_KEY", "n": 1}
            assert not call.released
            call.commit(Usage(input_tokens=10, output_tokens=5, total_tokens=15))
            return call

    call = asyncio.run(run())
    assert call.released
    assert call.committed == Usage(10, 5, 15)
    assert call.usage == Usage(10, 5, 15)
    assert builder.identities == [call.identity]


def test_refresh_rebuilds_the_client_for_the_same_identity() -> None:
    seen: list[Identity] = []

    def build(identity: Identity) -> int:
        seen.append(identity)
        return len(seen)

    async def run() -> None:
        async with ext.attempt(
            "transcription", "google", settings=_SETTINGS, build_client=build
        ) as call:
            first = call.identity
            assert call.client == 1
            assert await call.refresh() == 2
            assert call.client == 2
            assert call.identity is first
            with pytest.raises(AttributeError):
                call.identity = Identity("x", "Y", "z")  # type: ignore[misc]

    asyncio.run(run())
    assert len(set(seen)) == 1


def test_exception_releases_and_propagates() -> None:
    holder: list[ext.Attempt[Any]] = []

    async def run() -> None:
        async with ext.attempt(
            "summary", "openai", settings=_SETTINGS, build_client=_Builder()
        ) as call:
            holder.append(call)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run())
    assert holder[0].released


def test_cancellation_releases() -> None:
    holder: list[ext.Attempt[Any]] = []

    async def body(started: asyncio.Event) -> None:
        async with ext.attempt(
            "transcription", "openai", settings=_SETTINGS, build_client=_Builder()
        ) as call:
            holder.append(call)
            started.set()
            await asyncio.sleep(3600)

    async def run() -> None:
        started = asyncio.Event()
        task = asyncio.create_task(body(started))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert holder[0].released


def test_failed_client_build_releases() -> None:
    def broken(identity: Identity) -> object:
        raise ValueError("no client")

    attempt = ext.attempt("summary", "openai", settings=_SETTINGS, build_client=broken)

    async def run() -> None:
        async with attempt:
            pass

    with pytest.raises(ValueError, match="no client"):
        asyncio.run(run())
    assert attempt.released


@pytest.mark.parametrize(
    ("error", "decision"),
    [
        (_status_error(429), Decision.RETRY),
        (_status_error(503), Decision.RETRY),
        (TimeoutError("timed out"), Decision.RETRY),
        (_status_error(400), Decision.FAIL),
        (_status_error(401), Decision.FAIL),
    ],
)
def test_on_error_returns_the_classifier_decision(
    error: Exception, decision: Decision
) -> None:
    attempt = ext.attempt(
        "summary", "openai", settings=_SETTINGS, build_client=_Builder()
    )
    assert attempt.on_error(error) is decision
    assert attempt.errors == [error]


def test_on_error_records_recovered_usage() -> None:
    error = _status_error(500)
    error.body = {  # type: ignore[attr-defined]
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    }
    attempt = ext.attempt(
        "transcription", "openai", settings=_SETTINGS, build_client=_Builder()
    )
    assert attempt.on_error(error) is Decision.RETRY
    attempt.on_error(_status_error(502))
    attempt.commit(Usage(input_tokens=1, output_tokens=1, total_tokens=2))
    assert attempt.recovered == Usage(7, 3, 10)
    assert attempt.committed == Usage(1, 1, 2)
    assert attempt.usage == Usage(8, 4, 12)


def test_client_before_entry_is_an_error() -> None:
    attempt = ext.attempt(
        "summary", "openai", settings=_SETTINGS, build_client=_Builder()
    )
    with pytest.raises(RuntimeError):
        _ = attempt.client
