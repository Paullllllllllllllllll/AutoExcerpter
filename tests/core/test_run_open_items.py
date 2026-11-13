"""Cancelling a run that keeps several items open, and the open-items window."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter.output.store import DATABASE_NAME
from tests.conftest import make_settings, make_spec

MakePdf = Callable[..., Path]


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable the run checks before its first item."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _plan(input_path: Path, **flags: Any) -> core.RunPlan:
    return core.plan(
        make_spec(input_path, all=True, summarize=False, **flags), make_settings()
    )


def test_cancelling_the_run_releases_the_leases_of_every_open_item(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    mock_image_processing: dict[str, Any],
) -> None:
    (tmp_path / "in").mkdir()
    for name in ("alpha", "beta", "gamma"):
        make_pdf(f"in/{name}.pdf", num_pages=2)
    in_flight = 0
    both_open = asyncio.Event()

    async def transcribe(payload: Any) -> dict[str, Any]:
        nonlocal in_flight
        in_flight += 1
        if in_flight == 4:
            both_open.set()
        await asyncio.Event().wait()
        raise AssertionError("the call is cancelled before it answers")

    manager = MagicMock()
    manager.transcribe_payload = AsyncMock(side_effect=transcribe)
    monkeypatch.setattr(
        managers_module, "TranscriptionManager", MagicMock(return_value=manager)
    )
    out = tmp_path / "out"
    run_plan = _plan(tmp_path / "in", output=out, concurrency=4, open_items=3)

    async def cancel_when_two_are_open() -> None:
        task = asyncio.ensure_future(core.run(run_plan))
        await asyncio.wait_for(both_open.wait(), timeout=30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_when_two_are_open())

    with contextlib.closing(sqlite3.connect(out / DATABASE_NAME)) as db:
        rows = db.execute("SELECT name, lease_run_id FROM documents").fetchall()
    assert sorted(name for name, _ in rows) == ["alpha", "beta"]
    assert [holder for _, holder in rows] == [None, None]


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"open_items": 4}, 4),
        ({"open_items": 4, "select": "1"}, 1),
        ({}, 1),
    ],
    ids=["several", "single-item", "default"],
)
def test_the_plan_opens_several_items_only_for_several(
    tmp_path: Path, make_pdf: MakePdf, flags: dict[str, Any], expected: int
) -> None:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")
    make_pdf("in/beta.pdf")
    if "select" in flags:
        spec = make_spec(tmp_path / "in", summarize=False, **flags)
        run_plan = core.plan(spec, make_settings())
    else:
        run_plan = _plan(tmp_path / "in", **flags)

    assert run_plan.open_items == expected
    assert ("run.open_items" in run_plan.effective_spec()) is ("open_items" in flags)
