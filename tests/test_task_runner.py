"""Tests for modules/task_runner.py."""

from __future__ import annotations

import time

import pytest

from modules.task_runner import ConcurrentTaskRunner


@pytest.fixture
def no_tqdm(monkeypatch):
    """Disable tqdm output for deterministic tests."""

    def _passthrough(iterable, **_kwargs):
        return iterable

    monkeypatch.setattr("modules.task_runner.tqdm", _passthrough)


class TestConcurrentTaskRunnerRun:
    def test_run_preserves_input_order_with_progress(self, no_tqdm):
        runner = ConcurrentTaskRunner[int, int](max_workers=4, show_progress=True)
        items = [0, 1, 2, 3, 4]

        def task(x: int) -> int:
            time.sleep(0.01 * (len(items) - x))
            return x * x

        results = runner.run(task, items)
        assert results == [0, 1, 4, 9, 16]

    def test_run_preserves_input_order_without_progress(self):
        runner = ConcurrentTaskRunner[int, int](max_workers=4, show_progress=False)
        items = [5, 4, 3, 2, 1]

        def task(x: int) -> int:
            return x + 1

        results = runner.run(task, items)
        assert results == [6, 5, 4, 3, 2]

    def test_run_exception_results_aligned_to_input_order(self, no_tqdm):
        runner = ConcurrentTaskRunner[int, int | None](max_workers=4, show_progress=True)
        items = [0, 1, 2, 3]

        def task(x: int) -> int:
            if x == 2:
                raise ValueError("boom")
            time.sleep(0.01 * (len(items) - x))
            return x

        results = runner.run(task, items)
        assert results[0] == 0
        assert results[1] == 1
        assert results[2] is None
        assert results[3] == 3


class TestConcurrentTaskRunnerEta:
    def test_calculate_eta_returns_na_when_not_enough_samples(self):
        runner = ConcurrentTaskRunner[int, int](max_workers=2, show_progress=False)
        runner.start_time = time.time() - 10
        assert runner.calculate_eta(processed_count=1, total_items=10) == "ETA: N/A"

    def test_calculate_eta_returns_value_when_enough_samples(self):
        runner = ConcurrentTaskRunner[int, int](max_workers=2, show_progress=False)
        runner.start_time = time.time() - 10
        for _ in range(10):
            runner.record_processing_time(0.2)

        eta = runner.calculate_eta(processed_count=6, total_items=10)
        assert eta.startswith("ETA: ")
        assert eta != "ETA: N/A"

    def test_calculate_blended_rate_falls_back_to_overall_when_no_samples(self):
        runner = ConcurrentTaskRunner[int, int](max_workers=2, show_progress=False)
        assert runner._calculate_blended_rate(2.0) == 2.0
