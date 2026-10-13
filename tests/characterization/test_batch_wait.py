"""``batch check --wait`` through the fake batch services.

The wait's clock and sleep are replaced by a clock whose sleep advances it,
so the checks follow one another at once; the fake batches complete after
a set number of status calls.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.batch import check as check_module
from tests.characterization.fakes import TRANSCRIPTION, FakeLLM
from tests.characterization.fakes_batch import FakeBatchService, Provider
from tests.characterization.test_batch import DOC, PAGES, Case


class WaitClock:
    """A monotonic clock whose ``sleep`` advances it instead of waiting."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> WaitClock:
    fake = WaitClock()
    monkeypatch.setattr(check_module, "_clock", fake)
    monkeypatch.setattr(check_module, "_sleep", fake.sleep)
    return fake


@pytest.fixture
def run(tmp_path: Path, fake_llm: FakeLLM, fake_batch: FakeBatchService) -> Case:
    provider: Provider = "openai"
    return Case(tmp_path, fake_llm, fake_batch, provider)


def _wait_fields(summary: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        summary[key] for key in ("polls", "stopped", "needs_run", "rerun", "pending")
    )


def test_one_wait_collects_both_rounds_and_writes_the_files(
    run: Case, clock: WaitClock
) -> None:
    run.service.polls_until_complete = 2
    run.submit()

    waited = run.check("wait", "--wait")

    assert waited.result.exit_code == 0, waited.result.stderr
    summary = waited.summary
    assert _wait_fields(summary) == (4, "settled", 0, [], 0)
    assert (summary["finalized"], summary["submitted"], summary["complete"]) == (
        2,
        1,
        1,
    )
    assert summary["waited_seconds"] == 3 * 900
    assert clock.sleeps == [900, 900, 900]
    assert summary["usage"]["transcription"]["total_tokens"] == 3 * 1900
    assert summary["usage"]["summary"]["total_tokens"] == 3 * 1150
    assert {Path(path).name for path in summary["outputs"]} == {
        DOC.transcription.name,
        DOC.summary_md.name,
        DOC.summary_docx.name,
    }
    transcript = (run.output / DOC.transcription).read_text(encoding="utf-8")
    for page in range(PAGES):
        assert run.llm.page(page).key in transcript
    assert run.jobs() == 0
    stderr = waited.result.stderr
    assert "Waiting for batches: 1 pending; next check at" in stderr
    assert "Waiting until" not in stderr
    assert "Stopped after 4 check(s) and 0.75 h: settled." in stderr


def test_the_wait_stops_at_the_deadline_with_the_batch_pending(
    run: Case, clock: WaitClock
) -> None:
    run.service.polls_until_complete = 100
    run.submit()

    waited = run.check("wait", "--wait", "--wait-hours", "1", "--poll-minutes", "30")

    assert waited.result.exit_code == 1
    assert _wait_fields(waited.summary) == (3, "timeout", 0, [], 1)
    assert clock.sleeps == [1800, 1800]
    assert "not finished yet" in waited.result.stderr
    assert run.rows("SELECT status FROM submissions") == [("submitted",)]


def _cancel_submissions(run: Case) -> None:
    with contextlib.closing(sqlite3.connect(run.database)) as db:
        db.execute("UPDATE submissions SET status = 'cancelled'")
        db.commit()


def test_a_document_that_needs_a_run_fails_only_the_wait(
    run: Case, clock: WaitClock
) -> None:
    run.submit()
    _cancel_submissions(run)

    once = run.check("once")
    assert once.result.exit_code == 0, once.result.stderr
    assert "needs a run" in once.result.stderr
    assert "needs_run" not in once.summary

    waited = run.check("wait", "--wait")

    assert waited.result.exit_code == 1
    polls, stopped, needs_run, rerun, _ = _wait_fields(waited.summary)
    assert (polls, stopped, needs_run) == (1, "settled", 1)
    (command,) = rerun
    assert command == (
        f'autoexcerpter run --input "{run.pdf}" --output "{run.output}" --batch'
    )
    assert f"needs a run: {command}" in waited.result.stderr
    assert clock.sleeps == []


def test_a_check_that_raises_midway_keeps_what_it_did(
    run: Case, clock: WaitClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = check_module._check_database
    calls: list[int] = []

    async def flaky(*args: Any, **kwargs: Any) -> None:
        await original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(check_module, "_check_database", flaky)
    run.submit()

    waited = run.check("wait", "--wait", "--poll-minutes", "1")

    assert waited.result.exit_code == 0, waited.result.stderr
    summary = waited.summary
    assert (summary["polls"], summary["finalized"], summary["submitted"]) == (2, 2, 1)
    assert summary["usage"]["transcription"]["total_tokens"] == 3 * 1900
    assert "[WARNING] Check 1 failed (connection)" in waited.result.stderr
    assert clock.sleeps == [60]


def test_an_interrupt_reports_the_checks_made(run: Case, clock: WaitClock) -> None:
    async def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt

    run.service.polls_until_complete = 100
    run.submit()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(check_module, "_sleep", interrupted)
        waited = run.check("wait", "--wait")

    assert waited.result.exit_code == 130
    assert _wait_fields(waited.summary) == (1, "interrupted", 0, [], 1)


def test_wait_values_without_wait_are_a_usage_error(run: Case) -> None:
    run.submit()

    refused = run.check("refused", "--poll-minutes", "5")

    assert refused.result.exit_code == 2
    assert "--poll-minutes needs --wait" in refused.result.stderr
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "pending")
