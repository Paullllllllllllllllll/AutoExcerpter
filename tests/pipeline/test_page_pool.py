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

import autoexcerpter.pipeline.item as item_module
import autoexcerpter.pipeline.pages as pages_module
from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.progress import PageProgress
from tests.pipeline.helpers import (
    make_processor,
    make_progress,
    make_runner,
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


def _processor(tmp_path: Path, concurrency: int) -> ItemProcessor:
    return make_processor(
        tmp_path / "doc.pdf", tmp_path / "out", concurrency=concurrency
    )


async def _fake_run(
    idx: int, source: Any, t_res: list[dict[str, Any]], s_res: Any, progress: Any
) -> dict[str, Any]:
    t_res.append({"original_input_order_index": idx})
    progress.advance()
    return {"original_input_order_index": idx}


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
    monkeypatch.setattr(processor.runner, "run", _fake_run)

    results, _ = asyncio.run(processor._transcribe_and_summarize(_FakeSource(6)))  # type: ignore[arg-type]

    assert len(spy_executor.instances) == 1
    assert (False, True) in spy_executor.shutdowns
    assert processor.runner.executor is None
    assert sorted(r["original_input_order_index"] for r in results) == list(range(6))


def test_the_pool_bounds_pages_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path, 2)
    state = {"active": 0, "peak": 0}

    async def slow_run(*args: Any) -> dict[str, Any]:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return await _fake_run(*args)

    monkeypatch.setattr(processor.runner, "run", slow_run)

    results, _ = asyncio.run(processor._transcribe_and_summarize(_FakeSource(6)))  # type: ignore[arg-type]

    assert len(results) == 6
    assert state["peak"] == 2


def test_cancelling_the_item_cancels_its_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = _processor(tmp_path, 2)
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
        task = asyncio.ensure_future(
            processor._transcribe_and_summarize(_FakeSource(4))  # type: ignore[arg-type]
        )
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
    processor = _processor(tmp_path, 1)

    async def boom(*_args: Any) -> None:
        raise RuntimeError("fatal")

    monkeypatch.setattr(processor.runner, "run", boom)

    with pytest.raises(RuntimeError, match="fatal"):
        asyncio.run(processor._transcribe_and_summarize(_FakeSource(4)))  # type: ignore[arg-type]

    assert (False, True) in spy_executor.shutdowns


def test_pages_render_off_the_loop_and_count_every_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = transcriber(
        return_value={
            "transcription": "hello",
            "processing_time": 0.001,
            "provider": "openai",
        }
    )
    runner = make_runner(tmp_path, transcribe_manager=manager)
    monkeypatch.setattr(pages_module, "append_to_log", lambda *a, **k: True)
    source = _RenderSource()
    progress = make_progress(16)
    results: list[dict[str, Any]] = []

    async def run_all() -> None:
        await asyncio.gather(
            *(runner.run(i, source, results, [], progress) for i in range(16))
        )

    asyncio.run(run_all())

    assert progress.processed == 16
    assert len(progress.transcription_times) == 16
    assert len(results) == 16
    assert threading.main_thread().name not in source.threads


def test_progress_denominator_is_pending_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = make_processor(
        tmp_path / "doc.pdf", tmp_path / "out", completed={0, 1}, concurrency=2
    )

    # Pages 0 and 1 were completed on a prior run.
    async def fake_reload(
        _eligible: Any,
        _prior: Any,
        _runner: Any,
        t_results: list[dict[str, Any]],
        _s: list[dict[str, Any]],
    ) -> None:
        t_results.extend({"original_input_order_index": i} for i in (0, 1))

    monkeypatch.setattr(item_module, "reload_completed_pages", fake_reload)

    captured: list[tuple[int, int]] = []

    async def spy(
        idx: int, source: Any, t: Any, s: Any, progress: PageProgress
    ) -> dict[str, Any]:
        captured.append((progress.total, progress.already_complete))
        t.append({"original_input_order_index": idx})
        return {"original_input_order_index": idx}

    monkeypatch.setattr(processor.runner, "run", spy)

    asyncio.run(processor._transcribe_and_summarize(_FakeSource(5)))  # type: ignore[arg-type]

    # 5 total, 2 already complete -> 3 pending pages this run.
    assert captured, "spy was never called"
    assert all(total == 3 for total, _ in captured)
    assert all(done == 2 for _, done in captured)
