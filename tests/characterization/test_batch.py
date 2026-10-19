"""Characterization of batch mode through the fake batch services.

Each case submits with ``run --batch`` and continues with ``batch check``;
the real OpenAI, Anthropic and Google backends run against the fake SDK
clients of ``fakes_batch``, and a batch completes on its first status call.
A summarizing round trip takes three commands: the run submits the
transcriptions, the first check stores them and submits the summaries, the
second stores the summaries and writes the files. Every command is
snapshotted under ``golden/batch/<case>/<stage>/``: the output folder as it
stands after the command, the JSON summary line and ``batch.json``, with the
SDK client constructions, the API calls and the request bodies the command
submitted (images described, prompts as their length and digest).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.pipeline.item as pipeline_item
from autoexcerpter.output.store import DATABASE_NAME
from tests.characterization.adapter import (
    ResponseFormat,
    RunResult,
    item_paths,
    run_batch_command,
    run_tool,
)
from tests.characterization.fakes import (
    SUMMARY,
    TRANSCRIPTION,
    FakeLLM,
    Injection,
)
from tests.characterization.fakes_batch import (
    FakeAPIError,
    FakeBatchService,
    Provider,
    permanent_error,
)
from tests.characterization.golden import (
    GOLDEN_DIR,
    Normalizer,
    check_files,
    recorded_ids,
    replace_ids,
    snapshot,
)
from tests.characterization.inputs import make_pdf
from tests.conftest import WriteGuard

PAGES = 3
FAILED_PAGE = 1
MODELS: dict[Provider, str] = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.5-flash-lite",
}
CREATE_CALLS: dict[Provider, str] = {
    "openai": "openai.batches.create",
    "anthropic": "anthropic.messages.batches.create",
    "google": "google.batches.create",
}
DOC = item_paths("doc", Path())


@dataclasses.dataclass
class Stage:
    """One command of a case: its result, its snapshot and its API calls."""

    result: RunResult
    files: dict[str, str]
    batch: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        summary = self.result.summary
        assert summary is not None, self.result.stdout + self.result.stderr
        return summary

    @property
    def calls(self) -> list[str]:
        return [str(call["call"]) for call in self.batch["calls"]]

    def requests(self, role: str | None = None) -> list[dict[str, Any]]:
        """Return the submitted requests, of *role* when given."""
        return [
            request
            for request in self.batch["requests"]
            if role is None or request["role"] == role
        ]


def _digested(records: dict[str, Any]) -> dict[str, Any]:
    prompts = {
        name: {
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        }
        for name, text in records["prompts"].items()
    }
    return {**records, "prompts": prompts}


class Case:
    """Run the commands of one batch case over one output folder."""

    def __init__(
        self,
        root: Path,
        llm: FakeLLM,
        service: FakeBatchService,
        provider: Provider,
    ) -> None:
        self.root = root
        self.llm = llm
        self.service = service
        self.provider = provider
        self.output = root / "out"
        self.pdf = make_pdf(root / "docs" / "doc.pdf", pages=PAGES)
        self.stages: dict[str, Stage] = {}

    @property
    def database(self) -> Path:
        return self.output / DATABASE_NAME

    def _stage(self, name: str, command: Callable[[], RunResult]) -> Stage:
        self.llm.reset_records()
        self.service.reset_records()
        mark = self.service.mark()
        result = command()
        assert self.llm.calls == [], "a batch command made a direct model call"
        ids = recorded_ids(self.database) if self.database.is_file() else {}
        norm = Normalizer(self.root)
        records = replace_ids(
            norm.data(_digested(self.service.request_records(mark))), ids
        )
        state = dataclasses.replace(
            result,
            created=sorted(p for p in self.output.rglob("*") if p.is_file()),
        )
        files = {
            name: content
            for name, content in snapshot(state).items()
            if name != "requests.json"
        }
        files["batch.json"] = (
            json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        stage = Stage(result, files, records)
        self.stages[name] = stage
        return stage

    def submit(self, name: str = "submit", **kwargs: Any) -> Stage:
        """Run ``run --batch`` with both phases on the case's provider."""
        model = MODELS[self.provider]
        options: dict[str, Any] = {
            "transcription_provider": self.provider,
            "transcription_model": model,
            "summary_provider": self.provider,
            "summary_model": model,
            **kwargs,
        }
        return self._stage(
            name,
            lambda: run_tool(
                self.pdf,
                llm=self.llm,
                root=self.root,
                output_dir=self.output,
                batch=True,
                **options,
            ),
        )

    def check(self, name: str, *extra: str) -> Stage:
        """Run ``batch check`` over the output folder."""
        return self._stage(
            name,
            lambda: run_batch_command(
                "check", llm=self.llm, root=self.root, extra=extra
            ),
        )

    def cancel(self, name: str, *extra: str) -> Stage:
        """Run ``batch cancel --input`` over the output folder."""
        return self._stage(
            name,
            lambda: run_batch_command(
                "cancel", llm=self.llm, root=self.root, extra=extra
            ),
        )

    def golden(self, case: str, guard: WriteGuard) -> None:
        """Compare every stage with ``golden/batch/<case>/<stage>/``."""
        for name, stage in self.stages.items():
            check_files(
                f"batch/{case}/{name}",
                stage.files,
                on_write=lambda _dir: guard.consume(GOLDEN_DIR / "batch"),
            )

    def rows(self, sql: str) -> list[tuple[Any, ...]]:
        """Return the rows of a query on the case's database."""
        with contextlib.closing(sqlite3.connect(self.database)) as db:
            return db.execute(sql).fetchall()

    def statuses(self, kind: str) -> dict[int, str]:
        """Return the status of every unit of *kind*, by page."""
        rows = self.rows(f"SELECT idx, status FROM units WHERE kind = '{kind}'")
        return {int(idx): str(status) for idx, status in rows}

    def jobs(self) -> int:
        """Return the number of batch jobs in the store."""
        return int(self.rows("SELECT COUNT(*) FROM batch_jobs")[0][0])


@pytest.fixture
def case(
    tmp_path: Path, fake_llm: FakeLLM, fake_batch: FakeBatchService
) -> Callable[[Provider], Case]:
    def build(provider: Provider) -> Case:
        return Case(tmp_path, fake_llm, fake_batch, provider)

    return build


def _assert_complete(stage: Stage) -> None:
    assert stage.result.exit_code == 0, stage.result.stdout + stage.result.stderr
    assert stage.summary["complete"] == 1
    outputs = {Path(path).name for path in stage.summary["outputs"]}
    assert outputs == {
        DOC.transcription.name,
        DOC.summary_md.name,
        DOC.summary_docx.name,
    }


def _transcription_requests_are_images(stage: Stage) -> None:
    for request in stage.requests(TRANSCRIPTION):
        assert "image" in json.dumps(request["body"])


@pytest.mark.parametrize("provider", list(MODELS))
def test_round_trip(
    provider: Provider, case: Callable[[Provider], Case], write_guard: WriteGuard
) -> None:
    run = case(provider)

    submitted = run.submit()
    assert submitted.result.exit_code == 0, submitted.result.stderr
    assert submitted.summary["items_submitted"] == 1
    assert submitted.summary["items_complete"] == 0
    assert CREATE_CALLS[provider] in submitted.calls
    assert [r["page"] for r in submitted.requests(TRANSCRIPTION)] == [0, 1, 2]
    assert submitted.requests(SUMMARY) == []
    _transcription_requests_are_images(submitted)
    assert not (run.output / DOC.transcription).exists()
    assert run.jobs() == 1

    first = run.check("check1")
    assert first.result.exit_code == 0, first.result.stdout + first.result.stderr
    assert (first.summary["finalized"], first.summary["submitted"]) == (1, 1)
    assert first.summary["complete"] == 0
    assert [r["page"] for r in first.requests(SUMMARY)] == [0, 1, 2]
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "done")
    assert first.summary["usage"]["transcription"]["total_tokens"] == 3 * 1900

    second = run.check("check2")
    _assert_complete(second)
    assert second.summary["finalized"] == 1
    assert second.summary["usage"]["summary"]["total_tokens"] == 3 * 1150
    assert run.statuses(SUMMARY) == dict.fromkeys(range(PAGES), "done")
    assert run.jobs() == 0
    transcript = (run.output / DOC.transcription).read_text(encoding="utf-8")
    for page in range(PAGES):
        assert run.llm.page(page).key in transcript

    again = run.check("check3")
    assert again.result.exit_code == 0
    assert again.calls == []
    assert (again.summary["finalized"], again.summary["complete"]) == (0, 0)

    run.golden(f"{provider}_round_trip", write_guard)


@pytest.mark.parametrize("provider", list(MODELS))
def test_failed_request_gets_a_placeholder_and_a_rerun_retries_it(
    provider: Provider, case: Callable[[Provider], Case], write_guard: WriteGuard
) -> None:
    run = case(provider)
    run.llm.inject(
        FAILED_PAGE, TRANSCRIPTION, 1, Injection(exception=permanent_error())
    )

    run.submit()
    first = run.check("check1")
    assert first.result.exit_code == 0, first.result.stdout + first.result.stderr
    assert run.statuses(TRANSCRIPTION)[FAILED_PAGE] == "failed"
    assert [r["page"] for r in first.requests(SUMMARY)] == [0, 2]
    second = run.check("check2")
    assert second.result.exit_code == 1
    assert second.summary["complete"] == 0
    assert (run.output / DOC.transcription).is_file()
    assert run.jobs() == 0
    assert run.statuses(SUMMARY)[FAILED_PAGE] == "done"

    resubmitted = run.submit("resubmit")
    assert resubmitted.result.exit_code == 0, resubmitted.result.stderr
    assert [r["page"] for r in resubmitted.requests()] == [FAILED_PAGE]
    third = run.check("check3")
    assert [r["page"] for r in third.requests(SUMMARY)] == [FAILED_PAGE]
    fourth = run.check("check4")
    _assert_complete(fourth)
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "done")

    run.golden(f"{provider}_failed_request", write_guard)


def test_pages_submit_only_the_selected_pages_and_a_run_without_completes(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")

    submitted = run.submit(pages="3,1")
    assert submitted.result.exit_code == 0, submitted.result.stderr
    assert [r["page"] for r in submitted.requests(TRANSCRIPTION)] == [0, 2]
    first = run.check("check1")
    assert [r["page"] for r in first.requests(SUMMARY)] == [0, 2]
    assert run.statuses(TRANSCRIPTION) == {0: "done", 1: "pending", 2: "done"}
    second = run.check("check2")
    assert second.result.exit_code == 0, second.result.stderr
    assert second.summary["complete"] == 0
    assert "doc: left partial by --pages -> " in second.result.stderr
    assert "1 document(s) complete" not in second.result.stderr
    assert "0 document(s) complete, 1 left partial by --pages." in second.result.stderr
    assert run.jobs() == 0
    transcript = (run.output / DOC.transcription).read_text(encoding="utf-8")
    assert run.llm.page(0).key in transcript
    assert run.llm.page(2).key in transcript
    assert run.llm.page(1).key not in transcript

    rest = run.submit("rest")
    assert [r["page"] for r in rest.requests()] == [1]
    third = run.check("check3")
    assert [r["page"] for r in third.requests(SUMMARY)] == [1]
    _assert_complete(run.check("check4"))
    assert run.statuses(SUMMARY) == dict.fromkeys(range(PAGES), "done")


def test_pages_with_nothing_left_to_submit_count_the_partial_item_neither_way(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.submit(pages="1-2")
    run.check("check1")
    run.check("check2")

    again = run.submit("again", pages="1-2")

    assert again.result.exit_code == 0, again.result.stderr
    assert again.requests() == []
    counts = ("items_total", "items_complete", "items_failed", "items_submitted")
    assert [again.summary[key] for key in counts] == [1, 0, 0, 0]


def test_retranscribe_with_pages_submits_only_the_selected_pages(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.submit(pages="1-2")
    run.check("check1")
    run.check("check2")
    assert run.statuses(SUMMARY) == {0: "done", 1: "done", 2: "pending"}

    redo = run.submit("redo", retranscribe=True, pages="2-2")

    assert redo.result.exit_code == 0, redo.result.stderr
    assert [r["page"] for r in redo.requests()] == [1]
    assert run.statuses(TRANSCRIPTION) == {0: "done", 1: "pending", 2: "pending"}
    stale = run.rows("SELECT idx, stale FROM units WHERE kind = 'summary'")
    assert dict(stale)[0] == 0
    assert [r["page"] for r in run.check("check3").requests(SUMMARY)] == [1]
    fourth = run.check("check4")
    assert fourth.result.exit_code == 0, fourth.result.stderr
    assert fourth.summary["complete"] == 0
    assert "1 left partial by --pages." in fourth.result.stderr
    assert run.statuses(TRANSCRIPTION) == {0: "done", 1: "done", 2: "pending"}


def test_retranscribe_with_pages_keeps_the_stored_image_settings(
    case: Callable[[Provider], Case], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = case("openai")
    run.submit(pages="1-2")
    run.check("check1")
    run.check("check2")
    (stored,) = run.rows("SELECT image_settings FROM documents")
    qualities: list[Any] = []
    real = pipeline_item.open_source

    def spy(item: Any, job: Any) -> Any:
        qualities.append(job.images.jpeg_quality)
        return real(item, job)

    monkeypatch.setattr("autoexcerpter.batch.item.open_source", spy)

    redo = run.submit("redo", retranscribe=True, pages="2-2", jpeg_quality=80)

    assert redo.result.exit_code == 0, redo.result.stderr
    assert [r["page"] for r in redo.requests()] == [1]
    assert qualities == [json.loads(stored[0])["jpeg_quality"]] != [80]
    assert run.rows("SELECT image_settings FROM documents") == [stored]
    assert "keeps the stored image settings (jpeg_quality)" in redo.result.stderr


@pytest.mark.parametrize("summarized", [False, True], ids=["no_summaries", "some"])
def test_retranscribe_with_pages_keeps_the_image_settings_of_a_transcribed_item(
    case: Callable[[Provider], Case], monkeypatch: pytest.MonkeyPatch, summarized: bool
) -> None:
    run = case("openai")
    run.submit(summarize=False, docx=False, markdown=False)
    run.check("check1")
    if summarized:
        run.submit("summaries", pages="1-1")
        run.check("check2")
        assert run.statuses(SUMMARY) == {0: "done", 1: "pending", 2: "pending"}
    (stored,) = run.rows("SELECT image_settings FROM documents")
    qualities: list[Any] = []
    real = pipeline_item.open_source

    def spy(item: Any, job: Any) -> Any:
        qualities.append(job.images.jpeg_quality)
        return real(item, job)

    monkeypatch.setattr("autoexcerpter.batch.item.open_source", spy)

    redo = run.submit("redo", retranscribe=True, pages="2-2", jpeg_quality=80)

    assert redo.result.exit_code == 0, redo.result.stderr
    assert [r["page"] for r in redo.requests()] == [1]
    assert redo.requests(TRANSCRIPTION) == redo.requests()
    assert qualities == [json.loads(stored[0])["jpeg_quality"]] != [80]
    assert run.rows("SELECT image_settings FROM documents") == [stored]
    assert run.statuses(TRANSCRIPTION) == {0: "done", 1: "pending", 2: "done"}


def test_retranscribe_with_pages_refuses_image_settings_it_cannot_keep(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.submit(summarize=False, docx=False, markdown=False)
    run.check("check1")
    (stored,) = run.rows("SELECT image_settings FROM documents")

    redo = run.submit("redo", retranscribe=True, pages="2-2", image_detail="low")

    assert redo.result.exit_code == 1, redo.result.stderr
    assert redo.calls == []
    assert redo.summary["items_failed"] == 1
    assert "--retranscribe without --pages" in redo.result.stderr
    assert run.rows("SELECT image_settings FROM documents") == [stored]
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "done")


def test_retranscribe_with_pages_refuses_a_document_without_an_image_record(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.submit(summarize=False, docx=False, markdown=False)
    run.check("check1")
    with contextlib.closing(sqlite3.connect(run.database)) as db, db:
        db.execute("UPDATE documents SET image_settings = NULL")

    redo = run.submit("redo", retranscribe=True, pages="2-2")

    assert redo.result.exit_code == 1, redo.result.stderr
    assert redo.calls == []
    assert redo.summary["items_failed"] == 1
    assert "are unknown" in redo.result.stderr
    assert run.rows("SELECT image_settings FROM documents") == [(None,)]
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "done")


def test_missing_result_fails_the_page(
    case: Callable[[Provider], Case], write_guard: WriteGuard
) -> None:
    run = case("openai")
    run.service.drop_units = {(TRANSCRIPTION, FAILED_PAGE)}

    run.submit()
    first = run.check("check1")
    assert first.result.exit_code == 0, first.result.stdout + first.result.stderr
    assert run.statuses(TRANSCRIPTION)[FAILED_PAGE] == "failed"
    errors = run.rows(
        f"SELECT error FROM units WHERE kind = 'transcription' AND idx = {FAILED_PAGE}"
    )
    assert errors == [("[transcription error: no result returned]",)]
    run.check("check2")
    run.golden("openai_missing_result", write_guard)


def test_refused_submission_fails_the_item_and_leaves_no_batch_job(
    case: Callable[[Provider], Case], write_guard: WriteGuard
) -> None:
    run = case("anthropic")
    run.service.fail_on[CREATE_CALLS["anthropic"]] = permanent_error()

    submitted = run.submit()
    assert submitted.result.exit_code == 1
    assert submitted.summary["items_failed"] == 1
    assert submitted.summary["items_submitted"] == 0
    assert run.service.batch_states() == []
    assert run.jobs() == 0
    assert run.rows("SELECT status FROM submissions") == [("failed",)]
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "pending")

    checked = run.check("check")
    assert checked.result.exit_code == 0
    assert checked.calls == []
    run.golden("anthropic_submission_failure", write_guard)


def test_without_summaries_one_check_writes_the_transcription(
    case: Callable[[Provider], Case], write_guard: WriteGuard
) -> None:
    run = case("google")

    submitted = run.submit(summarize=False, docx=False, markdown=False)
    assert submitted.result.exit_code == 0, submitted.result.stderr
    first = run.check("check1")
    assert first.result.exit_code == 0, first.result.stdout + first.result.stderr
    assert first.summary["complete"] == 1
    assert first.summary["submitted"] == 0
    assert [Path(p).name for p in first.summary["outputs"]] == [DOC.transcription.name]
    assert run.statuses(SUMMARY) == {}
    run.golden("google_no_summary", write_guard)


def test_a_pending_batch_waits_for_a_later_check(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.service.polls_until_complete = 2

    run.submit()
    pending = run.check("pending")
    assert pending.result.exit_code == 0
    assert (pending.summary["pending"], pending.summary["finalized"]) == (1, 0)
    assert run.jobs() == 1
    finalized = run.check("finalize")
    assert finalized.summary["finalized"] == 1


def test_a_second_batch_run_skips_an_item_in_flight(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.submit()

    again = run.submit("again")
    # Pages waiting for a batch are submitted work: named, counted, exit 0.
    assert again.result.exit_code == 0, again.result.stderr
    assert (again.summary["items_skipped"], again.summary["items_blocked"]) == (1, 1)
    assert "Not processed (blocked): 'doc':" in again.result.stderr
    assert again.calls == []


def test_a_direct_run_takes_over_a_document_without_open_batches(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.service.fail_on[CREATE_CALLS["openai"]] = permanent_error()
    run.submit()
    run.service.fail_on.clear()
    assert run.jobs() == 0

    direct = run_tool(run.pdf, llm=run.llm, root=run.root, output_dir=run.output)
    assert direct.exit_code == 0, direct.stderr
    assert direct.summary is not None
    assert direct.summary["items_complete"] == 1
    assert run.jobs() == 0


def test_cancel_by_folder_ends_the_batch_work(
    case: Callable[[Provider], Case],
) -> None:
    run = case("anthropic")
    run.service.polls_until_complete = 3
    run.submit()

    cancelled = run.cancel("cancel")
    assert cancelled.result.exit_code == 0, cancelled.result.stdout
    assert cancelled.summary == {
        "dry_run": False,
        "command": "cancel_batches",
        "cancelled": 1,
        "failed": 0,
    }
    assert "anthropic.messages.batches.cancel" in cancelled.calls
    assert run.rows("SELECT status FROM submissions") == [("cancelled",)]
    assert run.jobs() == 0
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "pending")

    checked = run.check("check")
    assert checked.calls == []
    assert checked.summary["complete"] == 0


def test_an_unresolved_submission_is_abandoned(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.service.fail_on[CREATE_CALLS["openai"]] = TimeoutError("no answer")

    submitted = run.submit()
    assert submitted.result.exit_code == 1
    assert run.rows("SELECT status FROM submissions") == [("unresolved",)]
    assert run.jobs() == 1
    (submission_id,) = run.rows("SELECT submission_id FROM submissions")[0]

    reported = run.check("report")
    assert reported.result.exit_code == 1
    assert reported.summary["unresolved"] == 1

    abandoned = run.check("abandon", "--abandon", str(submission_id))
    assert abandoned.result.exit_code == 0, abandoned.result.stdout
    assert abandoned.summary["abandoned"] is True
    assert run.jobs() == 0
    assert run.rows("SELECT status FROM submissions") == [("abandoned",)]


def test_a_batch_whose_results_are_lost_can_be_abandoned_and_resubmitted(
    case: Callable[[Provider], Case],
) -> None:
    run = case("openai")
    run.submit()
    run.service.fail_on["openai.files.content"] = FakeAPIError(
        404, "Error code: 404 - file not found"
    )

    failed = run.check("failed")
    assert failed.result.exit_code == 1
    assert failed.summary["failed"] == 1
    assert "--abandon" in failed.result.stderr
    assert run.rows("SELECT status FROM submissions") == [("submitted",)]
    (submission_id,) = run.rows("SELECT submission_id FROM submissions")[0]

    abandoned = run.check(
        "abandon", "--abandon", str(submission_id), "--batch-ids", "batch_other"
    )
    assert abandoned.result.exit_code == 0, abandoned.result.stdout
    assert abandoned.summary["abandoned"] is True
    assert run.rows("SELECT status FROM submissions") == [("abandoned",)]
    assert run.jobs() == 0
    assert run.statuses(TRANSCRIPTION) == dict.fromkeys(range(PAGES), "pending")

    run.service.fail_on.clear()
    resubmitted = run.submit("resubmit")
    assert resubmitted.result.exit_code == 0, resubmitted.result.stderr
    assert [r["page"] for r in resubmitted.requests()] == [0, 1, 2]


def test_dry_run_reports_the_batch_plan_without_calls(
    case: Callable[[Provider], Case],
) -> None:
    run = case("google")

    planned = run.submit(dry_run=True)
    assert planned.result.exit_code == 0, planned.result.stderr
    assert planned.calls == []
    assert planned.summary["spec"]["run.batch"] == {"value": True, "source": "flag"}
    assert "batch mode: 1 item(s) would be submitted" in planned.result.stderr
    assert not run.output.exists()


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        ("openrouter", "provider openrouter"),
        ("custom", "custom endpoint"),
    ],
)
def test_a_provider_without_a_batch_api_exits_2_before_any_write(
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    provider: str,
    message: str,
) -> None:
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=1)
    model = "anthropic/claude-sonnet-4.5" if provider == "openrouter" else "fake"
    fmt: ResponseFormat = "schema" if provider == "openrouter" else "json"

    for dry_run in (False, True):
        result = run_tool(
            pdf,
            llm=fake_llm,
            root=tmp_path,
            transcription_provider=provider,
            transcription_model=model,
            transcription_format=fmt,
            batch=True,
            dry_run=dry_run,
        )
        assert result.exit_code == 2, result.stderr
        assert message in result.stderr
    assert fake_batch.calls == []
    assert not (tmp_path / "out").exists()


def test_check_with_nothing_below_the_folder_exits_0(
    tmp_path: Path, fake_llm: FakeLLM, fake_batch: FakeBatchService
) -> None:
    (tmp_path / "out").mkdir()
    result = run_batch_command("check", llm=fake_llm, root=tmp_path)
    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.summary is not None
    assert result.summary["finalized"] == 0
    assert fake_batch.calls == []
