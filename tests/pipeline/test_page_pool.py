"""The item's page pool: one render executor, bounded pages, cancellation.

The progress total counts only the pages pending in this run.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.progress import PageProgress
from tests.pipeline.helpers import (
    make_processor,
    make_progress,
    make_runner,
    statuses,
    store_pages,
    stored_document,
    transcriber,
)


class _FakeSource:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


class _RenderSource:
    """Pages render on a worker thread; the thread names are recorded."""

    def __init__(self) -> None:
        self.threads: list[str] = []

    def image_name(self, index: int) -> str:
        return f"page_{index}.jpg"

    def build_payload(self, index: int) -> Any:
        self.threads.append(threading.current_thread().name)
        return SimpleNamespace(
            source_file="doc.pdf", provenance={}, page_index=index, image_name="p"
        )


def _processor(tmp_path: Path, concurrency: int, pages: int = 6) -> ItemProcessor:
    processor = make_processor(
        tmp_path / "doc.pdf", tmp_path / "out", concurrency=concurrency
    )
    processor.page_count = pages
    return processor


def _pool(processor: ItemProcessor, pending: list[int]) -> Any:
    return processor._transcribe_pending(
        _FakeSource(processor.page_count),  # type: ignore[arg-type]
        pending,
    )


async def _fake_run(idx: int, source: Any, progress: PageProgress) -> str:
    progress.advance()
    return "ok"


class _SpyExecutor(concurrent.futures.ThreadPoolExecutor):
    instances: list[_SpyExecutor] = []
    shutdowns: list[tuple[bool, bool]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        type(self).instances.append(self)
        super().__init__(*args, **kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        type(self).shutdowns.append((wait, cancel_futures))
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


@pytest.fixture
def spy_executor(monkeypatch: pytest.MonkeyPatch) -> type[_SpyExecutor]:
    _SpyExecutor.instances = []
    _SpyExecutor.shutdowns = []
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _SpyExecutor)
    return _SpyExecutor


def test_one_render_executor_per_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy_executor: type[_SpyExecutor]
) -> None:
    processor = _processor(tmp_path, 2)
    seen: list[int] = []

    async def run(idx: int, source: Any, progress: PageProgress) -> str:
        seen.append(idx)
        return await _fake_run(idx, source, progress)

    monkeypatch.setattr(processor.runner, "run", run)

    asyncio.run(_pool(processor, list(range(6))))

    assert len(spy_executor.instances) == 1
    assert (False, True) in spy_executor.shutdowns
    assert processor.runner.executor is None
    assert sorted(seen) == list(range(6))


def test_the_pool_bounds_pages_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path, 2)
    state = {"active": 0, "peak": 0, "done": 0}

    async def slow_run(*args: Any) -> str:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        state["done"] += 1
        return await _fake_run(*args)

    monkeypatch.setattr(processor.runner, "run", slow_run)

    asyncio.run(_pool(processor, list(range(6))))

    assert state["done"] == 6
    assert state["peak"] == 2


@pytest.mark.parametrize(
    ("provider", "alone"), [("anthropic", True), ("openai", False)]
)
def test_anthropic_runs_the_first_page_alone_to_warm_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, alone: bool
) -> None:
    processor = _processor(tmp_path, 4, pages=4)
    processor.runner.transcribe_manager = SimpleNamespace(provider=provider)
    events: list[str] = []

    async def run(idx: int, *args: Any) -> str:
        events.append(f"start {idx}")
        await asyncio.sleep(0.01)
        events.append(f"end {idx}")
        return await _fake_run(idx, *args)

    monkeypatch.setattr(processor.runner, "run", run)

    asyncio.run(_pool(processor, list(range(4))))

    assert (events[:2] == ["start 0", "end 0"]) is alone
    assert sorted(events) == sorted(
        f"{e} {i}" for i in range(4) for e in ("start", "end")
    )


def test_cancelling_the_item_cancels_its_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path, 2, pages=4)
    started: list[int] = []
    cancelled: list[int] = []

    async def blocking_run(idx: int, *_args: Any) -> None:
        started.append(idx)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(idx)
            raise

    monkeypatch.setattr(processor.runner, "run", blocking_run)

    async def scenario() -> None:
        task = asyncio.ensure_future(_pool(processor, list(range(4))))
        while len(started) < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert sorted(started) == [0, 1]
    assert sorted(cancelled) == [0, 1]


def test_a_fatal_page_error_cancels_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy_executor: type[_SpyExecutor]
) -> None:
    processor = _processor(tmp_path, 1, pages=4)

    async def boom(*_args: Any) -> None:
        raise RuntimeError("fatal")

    monkeypatch.setattr(processor.runner, "run", boom)

    with pytest.raises(RuntimeError, match="fatal"):
        asyncio.run(_pool(processor, list(range(4))))

    assert (False, True) in spy_executor.shutdowns


def test_pages_render_off_the_loop_and_count_every_page(tmp_path: Path) -> None:
    manager = transcriber(
        return_value={
            "transcription": "hello",
            "processing_time": 0.001,
            "provider": "openai",
        }
    )
    source = _RenderSource()
    progress = make_progress(16)

    async def run_all() -> list[str]:
        async with stored_document(tmp_path) as document:
            await store_pages(document, 16, {})
            runner = make_runner(
                tmp_path, transcribe_manager=manager, document=document
            )
            return await asyncio.gather(
                *(runner.run(i, source, progress) for i in range(16))
            )

    results = asyncio.run(run_all())

    assert results == ["ok"] * 16
    assert progress.processed == 16
    assert len(progress.transcription_times) == 16
    assert statuses(tmp_path) == ["done"] * 16
    assert threading.main_thread().name not in source.threads


def test_progress_denominator_is_pending_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path, 2, pages=5)
    captured: list[tuple[int, int]] = []

    async def spy(idx: int, source: Any, progress: PageProgress) -> str:
        captured.append((progress.total, progress.already_complete))
        return "ok"

    monkeypatch.setattr(processor.runner, "run", spy)

    # Pages 0 and 1 were completed on a prior run.
    asyncio.run(_pool(processor, [2, 3, 4]))

    assert captured, "spy was never called"
    assert all(total == 3 for total, _ in captured)
    assert all(done == 2 for _, done in captured)
