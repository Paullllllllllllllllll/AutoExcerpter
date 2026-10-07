"""Final outputs are written off the event loop and stop on cancellation.

The OpenAlex enrichment runs in a worker thread: while a lookup blocks, other
tasks on the loop keep running. Cancelling the item or the run during the
enrichment waits for the in-flight lookup only, writes no summary file, keeps
the working logs and leaves the recorded lookups in the OpenAlex cache.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter.common.images import PagePayload
from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.rendering.citations import (
    LookupOutcome,
    LookupResult,
    OpenAlexClient,
)
from autoexcerpter.rendering.citations.openalex import CACHE_FILE
from autoexcerpter.state import read_json
from tests.conftest import make_settings, make_spec
from tests.pipeline.helpers import make_processor, summarizer, transcriber

MakePdf = Callable[..., Path]
pytestmark = pytest.mark.usefixtures("mock_api_keys")

REFERENCES = [
    "Allen, Robert C. (2001). The great divergence in European wages and "
    "prices. Explorations in Economic History.",
    "Braudel, Fernand (1979). Civilisation materielle, economie et "
    "capitalisme. Paris: Armand Colin.",
    "Clark, Gregory (2007). A Farewell to Alms. Princeton University Press.",
]


class GatedOpenAlex(OpenAlexClient):
    """Find every citation; lookup number *gate_at* waits for ``release``."""

    def __init__(self, state_dir: Path, gate_at: int) -> None:
        super().__init__(state_dir=state_dir)
        self.gate_at = gate_at
        self.looked_up: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def lookup(self, citation_text: str) -> LookupResult:
        self.looked_up.append(citation_text)
        number = len(self.looked_up)
        if number == self.gate_at:
            self.entered.set()
            if not self.release.wait(10):
                raise TimeoutError("the test never released the lookup")
        return LookupResult(
            LookupOutcome.FOUND,
            {
                "doi": f"10.1/{number}",
                "url": f"https://openalex.org/W{number}",
                "publication_year": 2000 + number,
            },
        )


def _transcribe(payload: PagePayload) -> dict[str, Any]:
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _summarize(transcription: str, page_num: int) -> dict[str, Any]:
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": ["A point."],
        "references": [{"citation": text, "is_partial": False} for text in REFERENCES],
        "processing_time": 0.01,
        "provider": "openai",
    }


@pytest.fixture(autouse=True)
def _instant_polite_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


def _processor(pdf: Path, output_dir: Path, client: OpenAlexClient) -> ItemProcessor:
    return make_processor(
        pdf,
        output_dir,
        transcribe_manager=transcriber(_transcribe),
        summary_manager=summarizer(_summarize),
        summarize=True,
        openalex=client,
        openalex_enabled=True,
    )


def _found_dois(state_dir: Path) -> set[str]:
    return {entry["doi"] for entry in read_json(state_dir / CACHE_FILE).values()}


def _assert_stopped_cleanly(paths: ItemPaths, client: GatedOpenAlex) -> None:
    assert len(client.looked_up) == 2
    assert not paths.summary_docx.exists()
    assert not paths.summary_md.exists()
    assert paths.transcription_log.exists()
    assert paths.summary_log.exists()
    assert _found_dois(client.cache.state_dir or Path()) == {"10.1/1", "10.1/2"}


def test_the_loop_keeps_running_while_a_lookup_blocks(
    make_pdf: MakePdf, tmp_path: Path, mock_image_processing: dict[str, Any]
) -> None:
    client = GatedOpenAlex(tmp_path / "state", gate_at=1)
    processor = _processor(make_pdf("doc.pdf"), tmp_path / "out", client)

    async def scenario() -> tuple[int, bool, bool]:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        ticking = asyncio.create_task(ticker())
        item = asyncio.create_task(processor.process())
        await _until(client.entered.is_set)
        before = ticks
        await asyncio.sleep(0.1)
        during = ticks - before
        blocked = not item.done()
        client.release.set()
        success = await item
        ticking.cancel()
        return during, blocked, success

    during, blocked, success = asyncio.run(scenario())

    assert blocked
    assert during > 0
    assert success
    assert processor.paths.summary_docx.exists()
    assert processor.paths.summary_md.exists()


def test_cancelling_the_item_waits_only_for_the_inflight_lookup(
    make_pdf: MakePdf, tmp_path: Path, mock_image_processing: dict[str, Any]
) -> None:
    client = GatedOpenAlex(tmp_path / "state", gate_at=2)
    processor = _processor(make_pdf("doc.pdf"), tmp_path / "out", client)

    async def scenario() -> tuple[bool, float]:
        item = asyncio.create_task(processor.process())
        await _until(client.entered.is_set)
        item.cancel()
        await asyncio.sleep(0.05)
        waited = not item.done()
        released = time.monotonic()
        client.release.set()
        with pytest.raises(asyncio.CancelledError):
            await item
        return waited, time.monotonic() - released

    waited, seconds = asyncio.run(scenario())

    assert waited
    assert seconds < 1.0
    assert processor.stop.is_set()
    _assert_stopped_cleanly(processor.paths, client)


def test_cancelling_the_run_stops_the_enrichment(
    make_pdf: MakePdf,
    tmp_path: Path,
    mock_image_processing: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcription = MagicMock()
    transcription.transcribe_payload = AsyncMock(side_effect=_transcribe)
    summary = MagicMock()
    summary.generate_summary = AsyncMock(side_effect=_summarize)
    monkeypatch.setattr(
        managers_module, "TranscriptionManager", MagicMock(return_value=transcription)
    )
    monkeypatch.setattr(
        managers_module, "SummaryManager", MagicMock(return_value=summary)
    )
    pdf = make_pdf("doc.pdf")
    output_dir = tmp_path / "out"
    planned = core.plan(
        make_spec(pdf, output=output_dir, openalex=True),
        make_settings(state_dir=tmp_path / "state"),
    )
    client = GatedOpenAlex(tmp_path / "state", gate_at=2)
    run_plan = dataclasses.replace(
        planned, job=dataclasses.replace(planned.job, openalex=client)
    )

    async def scenario() -> float:
        running = asyncio.create_task(core.run(run_plan))
        await _until(client.entered.is_set)
        running.cancel()
        await asyncio.sleep(0.05)
        released = time.monotonic()
        client.release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        return time.monotonic() - released

    seconds = asyncio.run(scenario())

    assert seconds < 1.0
    _assert_stopped_cleanly(ItemPaths.for_item("doc", output_dir), client)
