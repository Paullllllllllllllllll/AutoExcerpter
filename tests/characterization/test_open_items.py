"""Several items open at once under one request limit (``--open-items``).

Every model call waits a short real delay inside the HTTP step and records
the document of the item that sent it, so the tests see how many calls are
in flight and which items overlap. Each item is a two-page PDF: two units,
each a transcription call followed by a summary call in one request slot.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.llm.caller as caller_module
import autoexcerpter.run as core
from autoexcerpter import ext
from autoexcerpter.common.retry import Decision
from autoexcerpter.common.structured import default_invoke
from autoexcerpter.output.store import DATABASE_NAME, open_store
from autoexcerpter.pipeline.item import ItemProcessor
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import FakeLLM
from tests.characterization.inputs import make_pdf

REAL_SLEEP = asyncio.sleep
# Long against an item's planning and page rendering, also on a busy machine,
# so that the calls the limit admits are in flight together.
DELAY = 0.3
DOCUMENT: contextvars.ContextVar[str] = contextvars.ContextVar("document")


class ProviderGone(Exception):
    """The scripted failure after which the provider serves no further calls."""


class PageRefused(Exception):
    """A permanent error of one item's own calls."""


@dataclass
class Calls:
    """The model calls of a run: in flight now, the peak, and their spans.

    *fail* returns the error the *n*-th call of a document raises, if any;
    *delay* its delay in seconds. A call named in *waits* first waits until
    the document it maps to has a call in flight.
    """

    fail: Callable[[str, int], BaseException | None] = lambda doc, n: None
    delay: Callable[[str, int], float] = lambda doc, n: DELAY
    waits: dict[tuple[str, int], str] = field(default_factory=dict)
    now: dict[str, int] = field(default_factory=dict)
    peak: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    spans: list[tuple[str, float, float]] = field(default_factory=list)

    def reset(self) -> None:
        """Forget the calls and their scripts."""
        self.fail = lambda doc, n: None
        self.delay = lambda doc, n: DELAY
        self.waits.clear()
        self.now.clear()
        self.counts.clear()
        self.spans.clear()
        self.peak = 0

    def sent(self, doc: str) -> int:
        """Return the calls *doc* has sent."""
        return self.counts.get(doc, 0)

    def overlap(self) -> bool:
        """Whether calls of two documents were in flight at one moment."""
        return any(
            a[0] != b[0] and a[1] < b[2] and b[1] < a[2]
            for a in self.spans
            for b in self.spans
        )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """Delay and record every model call; name its document."""
    record = Calls()
    process = ItemProcessor.process

    async def named(self: ItemProcessor) -> bool:
        DOCUMENT.set(self.name)
        return await process(self)

    async def delayed(model: Any, messages: Any) -> Any:
        doc = DOCUMENT.get()
        number = record.counts[doc] = record.counts.get(doc, 0) + 1
        record.now[doc] = record.now.get(doc, 0) + 1
        record.peak = max(record.peak, sum(record.now.values()))
        start = time.monotonic()
        try:
            error = record.fail(doc, number)
            other = record.waits.get((doc, number))
            deadline = start + 10
            while other and not record.now.get(other) and time.monotonic() < deadline:
                await REAL_SLEEP(0.005)
            await REAL_SLEEP(record.delay(doc, number))
            if error is not None:
                raise error
            return await default_invoke(model, messages)
        finally:
            record.now[doc] -= 1
            record.spans.append((doc, start, time.monotonic()))

    monkeypatch.setattr(ItemProcessor, "process", named)
    monkeypatch.setattr(caller_module, "default_invoke", delayed)
    return record


@pytest.fixture
def stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every attempt decide STOP on :class:`ProviderGone`."""
    decide = ext.Attempt.on_error

    def on_error(self: ext.Attempt[object], exc: BaseException) -> Decision:
        decision = decide(self, exc)
        return Decision.STOP if isinstance(exc, ProviderGone) else decision

    monkeypatch.setattr(ext.Attempt, "on_error", on_error)


def _docs(folder: Path, count: int) -> list[str]:
    names = [f"doc_{chr(ord('a') + index)}" for index in range(count)]
    for name in names:
        make_pdf(folder / f"{name}.pdf", pages=2)
    return names


def _run(docs: Path, root: Path, llm: FakeLLM, **flags: Any) -> RunResult:
    llm.reset_records()
    return run_tool(docs, llm=llm, root=root, concurrency=4, **flags)


def _statuses(output_dir: Path) -> dict[str, set[str]]:
    with contextlib.closing(sqlite3.connect(output_dir / DATABASE_NAME)) as db:
        rows = db.execute(
            "SELECT d.name, u.status FROM units u JOIN documents d USING (doc_id)"
        ).fetchall()
    statuses: dict[str, set[str]] = {}
    for name, status in rows:
        statuses.setdefault(name, set()).add(status)
    return statuses


def _leased(output_dir: Path) -> list[str]:
    with contextlib.closing(sqlite3.connect(output_dir / DATABASE_NAME)) as db:
        rows = db.execute(
            "SELECT name FROM documents WHERE lease_run_id IS NOT NULL"
        ).fetchall()
    return [name for (name,) in rows]


def test_open_items_share_the_limit_and_keep_the_item_order(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, calls: Calls
) -> None:
    names = _docs(docs_dir, 6)

    result = _run(docs_dir, tmp_path, fake_llm, open_items=3, docx=False)

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 6
    assert calls.peak == 4
    assert calls.overlap()
    assert _statuses(tmp_path / "out") == {name: {"done"} for name in names}
    transcriptions = [
        Path(path).name.removesuffix("_transcription.md")
        for path in result.summary["outputs"]
        if path.endswith("_transcription.md")
    ]
    assert transcriptions == names
    started = [
        line.split()[1]
        for line in result.stderr.splitlines()
        if re.match(r"\[\d+/6\] ", line)
    ]
    assert started == names


def test_one_open_item_runs_the_items_one_after_another(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, calls: Calls
) -> None:
    names = _docs(docs_dir, 3)

    result = _run(docs_dir, tmp_path, fake_llm, open_items=1)

    assert result.exit_code == 0, result.stderr
    assert calls.peak == 2
    for earlier, later in zip(names, names[1:], strict=False):
        ended = max(end for doc, _, end in calls.spans if doc == earlier)
        began = min(start for doc, start, _ in calls.spans if doc == later)
        assert began >= ended


@pytest.mark.usefixtures("stopping")
def test_a_stop_closes_the_open_items_and_opens_no_further_one(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, calls: Calls
) -> None:
    names = _docs(docs_dir, 3)
    out = tmp_path / "out"

    def fail(doc: str, number: int) -> BaseException | None:
        return ProviderGone("gone") if (doc, number) == ("doc_a", 2) else None

    # The stop falls once doc_b has a call in flight and before any unit
    # ends, so that no unit frees room for doc_c.
    calls.fail = fail
    calls.delay = lambda doc, number: 0.01 if (doc, number) == ("doc_a", 2) else 0.3
    calls.waits[("doc_a", 2)] = "doc_b"

    stopped = _run(docs_dir, tmp_path, fake_llm, open_items=3)

    assert stopped.exit_code == 3, stopped.stderr
    assert stopped.summary is not None
    assert stopped.summary["stopped"] == "ProviderGone: gone"
    assert stopped.summary["items_failed"] == 0
    assert calls.sent("doc_b") > 0
    assert calls.sent("doc_c") == 0
    statuses = _statuses(out)
    assert "doc_c" not in statuses
    assert all(found <= {"done", "pending"} for found in statuses.values())
    assert _leased(out) == []
    assert "'doc_a', 'doc_b' and 1 item(s) not started" in stopped.stderr

    calls.reset()
    resumed = _run(docs_dir, tmp_path, fake_llm, open_items=3)

    assert resumed.exit_code == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["items_complete"] == 3
    assert _statuses(out) == {name: {"done"} for name in names}


@pytest.mark.usefixtures("stopping")
def test_an_item_that_opens_as_the_run_stops_keeps_its_stored_pages(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    calls: Calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _docs(docs_dir, 2)
    out = tmp_path / "out"
    assert _run(docs_dir, tmp_path, fake_llm, open_items=2).exit_code == 0
    calls.reset()
    calls.fail = lambda doc, number: ProviderGone("gone") if doc == "doc_a" else None
    seen: dict[str, Any] = {"held": False}
    run_item, beyond = core._run_item, core._beyond_pages

    async def held_until_the_stop(run_plan: Any, planned: Any) -> str | None:
        # doc_b has its lease and waits here while doc_a stops the run.
        if planned.name == "doc_b":
            seen["held"] = True
            deadline = time.monotonic() + 10
            while seen["stop"].reason is None and time.monotonic() < deadline:
                await REAL_SLEEP(0.005)
        return await beyond(run_plan, planned)

    async def recording_item(*args: Any, **kwargs: Any) -> Any:
        seen["stop"] = args[3].stop
        return await run_item(*args, **kwargs)

    monkeypatch.setattr(core, "_run_item", recording_item)
    monkeypatch.setattr(core, "_beyond_pages", held_until_the_stop)

    stopped = _run(docs_dir, tmp_path, fake_llm, open_items=2, force=True)

    assert stopped.exit_code == 3, stopped.stderr
    assert seen["held"]
    assert calls.sent("doc_b") == 0
    assert _statuses(out)["doc_b"] == {"done"}
    assert _leased(out) == []


def test_an_item_that_fails_on_its_own_leaves_the_others_to_complete(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, calls: Calls
) -> None:
    _docs(docs_dir, 3)
    calls.fail = lambda doc, number: PageRefused("refused") if doc == "doc_b" else None

    result = _run(docs_dir, tmp_path, fake_llm, open_items=3)

    assert result.exit_code == 1, result.stderr
    assert result.summary is not None
    assert (result.summary["items_complete"], result.summary["items_failed"]) == (2, 1)
    statuses = _statuses(tmp_path / "out")
    assert statuses["doc_a"] == statuses["doc_c"] == {"done"}
    assert "failed" in statuses["doc_b"]
    assert _leased(tmp_path / "out") == []


def test_open_items_of_one_folder_share_one_store(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    calls: Calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = _docs(docs_dir, 2)
    opened: list[Path] = []

    def counting(root: Path, **kwargs: Any) -> Any:
        opened.append(root)
        return open_store(root, **kwargs)

    monkeypatch.setattr(core, "open_store", counting)

    result = _run(docs_dir, tmp_path, fake_llm, open_items=2)

    assert result.exit_code == 0, result.stderr
    assert calls.overlap()
    assert len(opened) == 1
    assert _statuses(tmp_path / "out") == {name: {"done"} for name in names}
