"""A run that a provider stops: exit 3, unprocessed units left pending.

The attempt's ``on_error`` returns ``Decision.STOP`` for one scripted failure,
as an extension does when a provider will serve no further calls of the run.
Two three-page PDFs run with one worker, so the order of the calls is fixed.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter import ext
from autoexcerpter.common.capture import HttpAttempt
from autoexcerpter.common.retry import Decision, StopSignal
from autoexcerpter.common.store import LeaseLostError
from autoexcerpter.output.store import DATABASE_NAME, KIND_SUMMARY, KIND_TRANSCRIPTION
from autoexcerpter.pipeline.item import ItemProcessor
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM, Injection
from tests.characterization.inputs import make_pdf

STOPPED_PAGE = 1
UNITS_PER_ITEM = 6


class ProviderGone(Exception):
    """The scripted failure after which the provider serves no further calls."""


@pytest.fixture
def stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every attempt decide STOP on :class:`ProviderGone`."""
    decide = ext.Attempt.on_error

    def on_error(self: ext.Attempt[object], exc: BaseException) -> Decision:
        decision = decide(self, exc)
        return Decision.STOP if isinstance(exc, ProviderGone) else decision

    monkeypatch.setattr(ext.Attempt, "on_error", on_error)


def _run(docs: Path, root: Path, llm: FakeLLM) -> RunResult:
    llm.reset_records()
    return run_tool(docs, llm=llm, root=root, concurrency=1)


def _units(output_dir: Path, name: str, kind: str) -> dict[int, str]:
    with contextlib.closing(sqlite3.connect(output_dir / DATABASE_NAME)) as db:
        rows = db.execute(
            "SELECT u.idx, u.status FROM units u JOIN documents d USING (doc_id) "
            "WHERE d.name = ? AND u.kind = ? ORDER BY u.idx",
            (name, kind),
        ).fetchall()
    return dict(rows)


def _leased(output_dir: Path) -> list[str]:
    with contextlib.closing(sqlite3.connect(output_dir / DATABASE_NAME)) as db:
        rows = db.execute(
            "SELECT name FROM documents WHERE lease_run_id IS NOT NULL"
        ).fetchall()
    return [name for (name,) in rows]


@pytest.mark.parametrize(
    ("role", "transcribed", "summarized", "calls"),
    [
        (TRANSCRIPTION, {0}, {0}, 3),
        (SUMMARY, {0, 1}, {0}, 4),
    ],
)
@pytest.mark.usefixtures("stopping")
def test_a_stop_leaves_the_rest_pending_and_a_rerun_completes_it(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    role: str,
    transcribed: set[int],
    summarized: set[int],
    calls: int,
) -> None:
    make_pdf(docs_dir / "doc_a.pdf")
    make_pdf(docs_dir / "doc_b.pdf")
    fake_llm.inject(STOPPED_PAGE, role, 1, Injection(exception=ProviderGone("gone")))
    out = tmp_path / "out"

    stopped = _run(docs_dir, tmp_path, fake_llm)

    assert stopped.exit_code == 3, stopped.stderr
    assert stopped.summary is not None
    assert stopped.summary["stopped"] == "ProviderGone: gone"
    assert stopped.summary["items_failed"] == 0
    assert stopped.summary["items_complete"] == 0
    # The stopped call is the last request; doc_b sends none.
    assert len(stopped.calls()) == calls
    last = fake_llm.calls[-1]
    assert (last["role"], last["page"]) == (role, STOPPED_PAGE)
    expected = {
        KIND_TRANSCRIPTION: transcribed,
        KIND_SUMMARY: summarized,
    }
    for kind, done in expected.items():
        units = _units(out, "doc_a", kind)
        assert {idx for idx, status in units.items() if status == "done"} == done
        assert set(units.values()) <= {"done", "pending"}
    assert _units(out, "doc_b", KIND_TRANSCRIPTION) == {}
    assert _leased(out) == []
    left = UNITS_PER_ITEM - len(transcribed) - len(summarized)
    assert (
        f"[ERROR] Run stopped: ProviderGone: gone. {left} unit(s) of 'doc_a' and "
        "1 item(s) not started were left unprocessed; a rerun resumes them."
    ) in stopped.stderr

    fake_llm.page(STOPPED_PAGE).injections.clear()
    resumed = _run(docs_dir, tmp_path, fake_llm)

    assert resumed.exit_code == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["stopped"] is None
    assert resumed.summary["items_complete"] == 2
    assert len(resumed.calls()) == left + UNITS_PER_ITEM
    for name in ("doc_a", "doc_b"):
        for kind in (KIND_TRANSCRIPTION, KIND_SUMMARY):
            assert set(_units(out, name, kind).values()) == {"done"}


@pytest.mark.usefixtures("stopping")
def test_the_stop_is_set_before_the_failed_attempt_is_stored(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_pdf(docs_dir / "doc_a.pdf")
    fake_llm.inject(
        STOPPED_PAGE, TRANSCRIPTION, 1, Injection(exception=ProviderGone("gone"))
    )
    order: list[str] = []
    set_stop, store = StopSignal.set, HttpAttempt.store

    def stopping(self: StopSignal, reason: str) -> None:
        order.append("stop")
        set_stop(self, reason)

    async def storing(self: HttpAttempt, **record: Any) -> None:
        if isinstance(record.get("error"), ProviderGone):
            order.append("store")
        await store(self, **record)

    monkeypatch.setattr(StopSignal, "set", stopping)
    monkeypatch.setattr(HttpAttempt, "store", storing)

    assert _run(docs_dir, tmp_path, fake_llm).exit_code == 3
    assert order[:2] == ["stop", "store"]


def _stale(output_dir: Path, table: str) -> set[int]:
    with contextlib.closing(sqlite3.connect(output_dir / DATABASE_NAME)) as db:
        return {stale for (stale,) in db.execute(f"SELECT stale FROM {table}")}


@pytest.mark.usefixtures("stopping")
def test_a_stop_withholds_the_files_of_a_page_transcribed_again(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM
) -> None:
    make_pdf(docs_dir / "doc_a.pdf")
    out = tmp_path / "out"

    def run(pages: str, *, retranscribe: bool = False) -> RunResult:
        fake_llm.reset_records()
        return run_tool(
            docs_dir,
            llm=fake_llm,
            root=tmp_path,
            concurrency=1,
            pages=pages,
            retranscribe=retranscribe,
        )

    # --retranscribe works a partial document; the last page stays pending.
    assert run("1-2").exit_code == 0
    fake_llm.inject(STOPPED_PAGE, SUMMARY, 1, Injection(exception=ProviderGone("gone")))

    stopped = run(f"{STOPPED_PAGE + 1}-{STOPPED_PAGE + 1}", retranscribe=True)

    assert stopped.exit_code == 3, stopped.stderr
    # The page's new transcription is stored and its summary is the old one,
    # so the files of the first run stay and are still to be rendered.
    assert _units(out, "doc_a", KIND_SUMMARY)[STOPPED_PAGE] == "done"
    assert _stale(out, "units") == {0, 1}
    assert _stale(out, "outputs") == {1}

    fake_llm.page(STOPPED_PAGE).injections.clear()
    resumed = _run(docs_dir, tmp_path, fake_llm)

    assert resumed.exit_code == 0, resumed.stderr
    # The stale summary and both units of the last page.
    assert len(resumed.calls()) == 3
    assert _stale(out, "units") == {0}
    assert _stale(out, "outputs") == {0}


def test_a_stop_outlasts_an_item_that_also_lost_its_lease(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_pdf(docs_dir / "doc_a.pdf")
    make_pdf(docs_dir / "doc_b.pdf")

    async def process(self: ItemProcessor) -> bool:
        self.stop_signal.set("ProviderGone: gone")
        raise LeaseLostError("another run took the document")

    monkeypatch.setattr(ItemProcessor, "process", process)

    stopped = _run(docs_dir, tmp_path, fake_llm)

    assert stopped.exit_code == 3, stopped.stderr
    assert stopped.summary is not None
    assert stopped.summary["stopped"] == "ProviderGone: gone"
    assert stopped.summary["items_failed"] == 1
    assert stopped.calls() == []
    assert _units(tmp_path / "out", "doc_b", KIND_TRANSCRIPTION) == {}
    assert (
        "[ERROR] Run stopped: ProviderGone: gone. 1 item(s) not started were "
        "left unprocessed; a rerun resumes them."
    ) in stopped.stderr
