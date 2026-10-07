"""Core entry points: discover, plan with typed errors, run with events."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests

import autoexcerpter.llm.caller as caller_module
import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.run as core
from autoexcerpter.common.contract import ExitCode
from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.constants import LOG_FORMAT_VERSION
from autoexcerpter.events import ItemFinished, ItemStarted, PageDone
from autoexcerpter.pipeline.job import job_from_spec
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.rendering.citations import OpenAlexClient
from autoexcerpter.rendering.sqlite import DATABASE_NAME
from autoexcerpter.settings import resolve as resolve_settings
from tests.conftest import make_settings, make_spec
from tests.recording_events import RecordingEvents

MakePdf = Callable[..., Path]


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variables the run checks before its first item."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "test-key")


def _plan(input_path: Path, **flags: Any) -> core.RunPlan:
    return core.plan(make_spec(input_path, **flags), make_settings())


def _transcribe(payload: Any) -> dict[str, Any]:
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


@pytest.fixture
def transcriber(
    monkeypatch: pytest.MonkeyPatch, mock_image_processing: dict[str, Any]
) -> MagicMock:
    """Replace the transcription manager with an offline stand-in."""
    manager = MagicMock()
    manager.transcribe_payload = AsyncMock(side_effect=_transcribe)
    monkeypatch.setattr(
        managers_module, "TranscriptionManager", MagicMock(return_value=manager)
    )
    return manager


def _fake_processor(
    success: bool, working_dir: Path, written: list[Path] | None = None
) -> MagicMock:
    processor = MagicMock()
    processor.process = AsyncMock(return_value=success)
    processor.written_outputs = written or []
    processor.last_run_report = {"pages_ok": 1}
    processor.record = None
    processor.paths.working_dir = working_dir
    return processor


def test_discover_finds_pdfs_and_image_folders(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    (tmp_path / "in" / "scans").mkdir(parents=True)
    make_pdf("in/a.pdf")
    (tmp_path / "in" / "scans" / "p1.png").write_bytes(b"png")

    items = core.discover(tmp_path / "in")

    assert sorted((item.kind, item.path.name) for item in items) == [
        ("image_folder", "scans"),
        ("pdf", "a.pdf"),
    ]
    assert all(item.path.is_absolute() for item in items)


def _empty_dir(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    (tmp_path / "in").mkdir()
    return tmp_path / "in", {}


def _two_items(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")
    make_pdf("in/beta.pdf")
    return tmp_path / "in", {}


def _select_nothing(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    path, _ = _two_items(tmp_path, make_pdf)
    return path, {"select": "gamma"}


def _same_stem(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    for sub in ("a", "b"):
        (tmp_path / "in" / sub).mkdir(parents=True)
        make_pdf(f"in/{sub}/book.pdf")
    return tmp_path / "in", {"all": True, "output": tmp_path / "out"}


def _empty_context_file(
    tmp_path: Path, make_pdf: MakePdf
) -> tuple[Path, dict[str, Any]]:
    context = tmp_path / "topics.txt"
    context.write_text("  \n", encoding="utf-8")
    return make_pdf("doc.pdf"), {"context": str(context)}


def _summaries_without_format(
    tmp_path: Path, make_pdf: MakePdf
) -> tuple[Path, dict[str, Any]]:
    return make_pdf("doc.pdf"), {"formats": ()}


@pytest.mark.parametrize(
    ("setup", "error", "code", "message"),
    [
        (_empty_dir, core.NoItemsError, 1, "No items found to process"),
        (_select_nothing, core.SelectionError, 1, "No items found matching 'gamma'"),
        (_two_items, core.AmbiguousSelectionError, 2, "neither --all nor --select"),
        (_same_stem, core.DuplicateOutputError, 2, "Duplicate output target"),
        (_empty_context_file, core.ConfigurationError, 2, "empty or unreadable"),
        (_summaries_without_format, core.ConfigurationError, 2, "--no-summarize"),
    ],
)
def test_plan_errors_carry_their_exit_code(
    tmp_path: Path,
    make_pdf: MakePdf,
    setup: Callable[[Path, MakePdf], tuple[Path, dict[str, Any]]],
    error: type[core.PlanError],
    code: int,
    message: str,
) -> None:
    input_path, flags = setup(tmp_path, make_pdf)

    with pytest.raises(error) as raised:
        _plan(input_path, **flags)

    assert raised.value.exit_code == code
    assert isinstance(raised.value.exit_code, ExitCode)
    assert message in str(raised.value)


@pytest.mark.parametrize("fmt", ["schema", "json", "text"])
def test_every_response_format_plans_on_a_registry_provider(
    tmp_path: Path, make_pdf: MakePdf, fmt: str
) -> None:
    run_plan = _plan(
        make_pdf("doc.pdf"),
        transcription_provider="anthropic",
        transcription_model="claude-sonnet-4-5",
        response_format=fmt,
    )

    assert run_plan.job.transcription.response_format == fmt
    assert run_plan.job.transcription.endpoint is None
    assert run_plan.job.summary.response_format == fmt


def test_duplicate_output_error_counts_the_selected_items(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    input_path, flags = _same_stem(tmp_path, make_pdf)
    with pytest.raises(core.DuplicateOutputError) as raised:
        _plan(input_path, **flags)
    assert raised.value.items_total == 2


def test_duplicate_guard_runs_before_the_resume_check(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, flags = _same_stem(tmp_path, make_pdf)
    monkeypatch.setattr(
        core, "ResumeChecker", MagicMock(side_effect=AssertionError("resume first"))
    )
    with pytest.raises(core.DuplicateOutputError):
        _plan(input_path, **flags)


def _log_from_other_input(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, Path]:
    """Return inputs A and B named doc.pdf; out/ holds A's working log."""
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    first, second = make_pdf("a/doc.pdf"), make_pdf("b/doc.pdf")
    log = ItemPaths.for_item("doc", tmp_path / "out").transcription_log
    log.parent.mkdir(parents=True)
    header = {"_format_version": LOG_FORMAT_VERSION, "input_item_path": str(first)}
    log.write_text(json.dumps(header) + "\n", encoding="utf-8")
    return first, second


def test_a_log_of_another_existing_input_stops_the_plan(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    first, second = _log_from_other_input(tmp_path, make_pdf)
    with pytest.raises(core.DuplicateOutputError) as raised:
        _plan(second, output=tmp_path / "out")
    assert raised.value.exit_code == 2
    assert str(first) in str(raised.value) and str(second) in str(raised.value)


def test_a_log_of_a_moved_input_keeps_the_resume(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    first, second = _log_from_other_input(tmp_path, make_pdf)
    first.unlink()
    run_plan = _plan(second, output=tmp_path / "out")
    assert [item.name for item in run_plan.to_process] == ["doc"]


@pytest.fixture
def library(tmp_path: Path, make_pdf: MakePdf) -> Path:
    (tmp_path / "Shared").mkdir()
    for name in ("Mennell_Food_History", "Laudan_Cuisine", "Pilcher_Food_History"):
        make_pdf(f"Shared/{name}.pdf")
    return tmp_path / "Shared"


def _names(run_plan: core.RunPlan) -> list[str]:
    return sorted(item.name for item in run_plan.to_process)


def test_select_by_number_range_and_name(library: Path) -> None:
    assert _names(_plan(library, all=True)) == [
        "Laudan_Cuisine",
        "Mennell_Food_History",
        "Pilcher_Food_History",
    ]
    assert len(_plan(library, select="2").to_process) == 1
    assert len(_plan(library, select="1-2").to_process) == 2
    assert _names(_plan(library, select="food_history")) == [
        "Mennell_Food_History",
        "Pilcher_Food_History",
    ]


def test_ignored_select_parts_become_plan_warnings(library: Path) -> None:
    run_plan = _plan(library, select="1,99")

    assert len(run_plan.to_process) == 1
    assert run_plan.warnings == (
        "Ignored out-of-range/invalid --select part(s): 99 (valid: 1-3)",
    )


def test_name_search_ignores_the_parent_path(library: Path) -> None:
    with pytest.raises(core.SelectionError):
        _plan(library, select="Shared")


def test_complete_items_are_skipped_unless_forced(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf")
    out = tmp_path / "out"
    out.mkdir()
    ItemPaths.for_item("doc", out).transcription.write_text("x", encoding="utf-8")

    skipping = _plan(pdf, output=out, summarize=False)
    forcing = _plan(pdf, output=out, summarize=False, force=True)

    assert [item.name for item in skipping.skipped] == ["doc"]
    assert skipping.to_process == ()
    assert [item.state for item in forcing.to_process] == ["none"]
    assert forcing.to_process[0].completed_pages == 0


def test_outputs_go_beside_the_input_without_an_output_folder(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf")

    (planned,) = _plan(pdf).to_process

    assert planned.output_dir == pdf.parent
    assert planned.paths == ItemPaths.for_item("doc", pdf.parent)


def test_paths_are_made_absolute(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_pdf("doc.pdf")
    monkeypatch.chdir(tmp_path)

    run_plan = _plan(Path("doc.pdf"), output=Path("out"))

    assert run_plan.input_path == tmp_path / "doc.pdf"
    assert run_plan.output_dir == tmp_path / "out"


def test_plan_writes_nothing(tmp_path: Path, make_pdf: MakePdf) -> None:
    pdf = make_pdf("doc.pdf")
    out = tmp_path / "out" / "sub"

    run_plan = _plan(pdf, output=out, dry_run=True)

    assert run_plan.dry_run
    assert not out.exists()


def test_context_is_resolved_per_item(tmp_path: Path, make_pdf: MakePdf) -> None:
    pdf = make_pdf("doc.pdf")
    (tmp_path / "doc_summary_context.txt").write_text(
        "Grain prices\nWages\n", encoding="utf-8"
    )
    topics = tmp_path / "topics.txt"
    topics.write_text("Bread\nBeer\n", encoding="utf-8")

    def context(**flags: Any) -> str | None:
        return _plan(pdf, **flags).to_process[0].context

    assert context() == "Grain prices, Wages"
    assert context(context="Focus on bread.") == "Focus on bread."
    assert context(context=str(topics)) == "Bread, Beer"
    assert context(summarize=False) is None


def test_context_precedence_over_sidecars_and_the_fallback(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf")

    def context(**flags: Any) -> str | None:
        return _plan(pdf, fallback_context="Fallback", **flags).to_process[0].context

    assert context() == "Fallback"
    assert context(context="none") is None
    (tmp_path / "doc_summary_context.txt").write_text(
        "Grain prices\nWages\n", encoding="utf-8"
    )
    assert context() == "Grain prices, Wages"
    assert context(context="Focus on bread.") == "Focus on bread."
    assert context(context="none") is None


def _plan_with_defaults(input_path: Path, **defaults: Any) -> core.RunPlan:
    settings = make_settings(defaults=defaults)
    resolution = resolve_settings(settings, {"input": input_path})
    return core.plan(resolution.spec, settings, sources=resolution.sources)


def test_a_settings_default_context_acts_like_the_flag(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("doc.pdf")
    (tmp_path / "doc_summary_context.txt").write_text("Sidecar", encoding="utf-8")

    def context(**defaults: Any) -> str | None:
        return _plan_with_defaults(pdf, **defaults).to_process[0].context

    assert context(context="Default topics") == "Default topics"
    assert context(context="none") is None
    assert context(fallback_context="Fallback") == "Sidecar"
    (tmp_path / "doc_summary_context.txt").unlink()
    assert context(fallback_context="Fallback") == "Fallback"


def test_the_job_holds_one_openalex_client_unless_disabled(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    settings = make_settings(state_dir=tmp_path / "state")
    pdf = make_pdf("doc.pdf")

    job = job_from_spec(make_spec(pdf), settings)
    other = job_from_spec(make_spec(pdf), settings)

    assert isinstance(job.openalex, OpenAlexClient)
    assert job.openalex.cache.state_dir == tmp_path / "state"
    assert other.openalex is not job.openalex
    assert job_from_spec(make_spec(pdf, openalex=False), settings).openalex is None
    assert not (tmp_path / "state").exists()


class _ExhaustedResponse:
    """A 429 whose retryAfter is longer than a run can wait."""

    status_code = 429
    url = "https://api.openalex.org/works"
    headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return {"retryAfter": 3600}


_CITATIONS = (
    "Smith, John. 1999. A History of Bread in Early Modern Europe. "
    "Journal of Food History 12: 1-20.",
    "Jones, Mary. 2001. Spice Merchants and the Dutch Republic. "
    "Economic History Review 54: 33-60.",
)


def test_an_exhausted_openalex_request_limit_ends_lookups_for_the_run(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    transcriber: MagicMock,
) -> None:
    sent: list[dict[str, Any]] = []

    def fake_get(url: str, params: Any = None, **_kwargs: Any) -> Any:
        sent.append(dict(params or {}))
        return _ExhaustedResponse()

    class FakeSession:
        get = staticmethod(fake_get)

        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "Session", FakeSession)

    citations = iter(_CITATIONS)

    def summary_manager(**_kwargs: Any) -> MagicMock:
        citation = next(citations)

        def summarize(transcription: str, page_num: int) -> dict[str, Any]:
            return {
                "page": page_num,
                "page_information": {
                    "page_number_integer": page_num,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["A point."],
                "references": [{"citation": citation, "is_partial": False}],
                "processing_time": 0.01,
                "provider": "openai",
            }

        manager = MagicMock()
        manager.generate_summary = AsyncMock(side_effect=summarize)
        return manager

    monkeypatch.setattr(managers_module, "SummaryManager", summary_manager)
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")
    make_pdf("in/beta.pdf")
    run_plan = core.plan(
        make_spec(tmp_path / "in", all=True, output=tmp_path / "out"),
        make_settings(state_dir=tmp_path / "state"),
    )

    result = asyncio.run(core.run(run_plan))

    assert len(result.items) == 2
    assert len(sent) == 1
    assert "Bread" in str(sent[0])
    assert run_plan.job.openalex is not None
    assert run_plan.job.openalex.budget_exhausted


def test_run_reports_events_in_order(
    tmp_path: Path, make_pdf: MakePdf, transcriber: MagicMock
) -> None:
    pdf = make_pdf("doc.pdf", num_pages=2)
    run_plan = _plan(pdf, output=tmp_path / "out", summarize=False)
    events = RecordingEvents()

    result = asyncio.run(core.run(run_plan, events))

    assert events.kinds == ["item_started", "page_done", "page_done", "item_finished"]
    assert events.of("item_started") == [ItemStarted("doc", "pdf", 1, 1)]
    pages: list[PageDone] = events.of("page_done")
    assert sorted(page.page for page in pages) == [0, 1]
    assert [page.done for page in pages] == [1, 2]
    assert {(page.status, page.total) for page in pages} == {("ok", 2)}
    (finished,) = events.of("item_finished")
    assert isinstance(finished, ItemFinished)
    assert finished.success
    assert finished.outputs == result.items[0].outputs

    paths = ItemPaths.for_item("doc", tmp_path / "out")
    assert result.complete == 1
    assert result.failed == 0
    assert result.items[0].paths == paths
    assert result.items[0].outputs == (str(paths.transcription.resolve()),)
    assert result.outputs == [
        str(paths.transcription.resolve()),
        str((tmp_path / "out" / DATABASE_NAME).resolve()),
    ]
    assert result.items[0].report is not None
    assert result.items[0].report["pages_ok"] == 2


def test_items_beside_one_another_share_one_database(
    tmp_path: Path, make_pdf: MakePdf, transcriber: MagicMock
) -> None:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf", num_pages=2)
    make_pdf("in/beta.pdf")
    run_plan = _plan(tmp_path / "in", all=True, summarize=False)

    result = asyncio.run(core.run(run_plan))

    database = tmp_path / "in" / DATABASE_NAME
    assert result.complete == 2
    documents = read_rows(database, "documents", order_by=["name"])
    assert [(row["name"], row["complete"]) for row in documents] == [
        ("alpha", 1),
        ("beta", 1),
    ]
    pages = read_rows(database, "pages", order_by=["document", "page_index"])
    assert [(row["document"], row["page_index"]) for row in pages] == [
        ("alpha", 0),
        ("alpha", 1),
        ("beta", 0),
    ]
    assert read_rows(database, "summaries") == []
    assert sorted(path.name for path in (tmp_path / "in").iterdir()) == sorted(
        [
            "alpha.pdf",
            "beta.pdf",
            DATABASE_NAME,
            "alpha_transcription.md",
            "beta_transcription.md",
            "alpha_autoexcerpter",
            "beta_autoexcerpter",
        ]
    )


def test_without_the_sqlite_format_no_database_is_written(
    tmp_path: Path, make_pdf: MakePdf, transcriber: MagicMock
) -> None:
    out = tmp_path / "out"
    run_plan = _plan(
        make_pdf("doc.pdf"), output=out, summarize=False, formats=("summary-md",)
    )

    result = asyncio.run(core.run(run_plan))

    assert result.complete == 1
    assert not (out / DATABASE_NAME).exists()


def test_a_failed_database_write_fails_the_item(
    tmp_path: Path,
    make_pdf: MakePdf,
    transcriber: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def locked(*_args: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(core, "write_document", locked)
    run_plan = _plan(make_pdf("doc.pdf"), output=tmp_path / "out", summarize=False)

    result = asyncio.run(core.run(run_plan))

    assert result.failed == 1
    assert result.items[0].outputs
    assert result.outputs == list(result.items[0].outputs)


def test_failed_page_fails_the_item_and_keeps_its_logs(
    tmp_path: Path, make_pdf: MakePdf, transcriber: MagicMock
) -> None:
    def fail_second(payload: Any, **_kwargs: Any) -> dict[str, Any]:
        result = _transcribe(payload)
        if payload.number == 2:
            result["error"] = "refused"
        return result

    transcriber.transcribe_payload.side_effect = fail_second
    pdf = make_pdf("doc.pdf", num_pages=2)
    run_plan = _plan(pdf, output=tmp_path / "out", summarize=False, force=True)
    events = RecordingEvents()

    result = asyncio.run(core.run(run_plan, events))

    assert sorted(page.status for page in events.of("page_done")) == ["failed", "ok"]
    assert result.failed == 1
    assert not result.items[0].success
    assert result.items[0].paths.working_dir.exists()


@pytest.mark.parametrize(
    ("force", "success", "removed"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_working_dir_cleanup_rule(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    force: bool,
    success: bool,
    removed: bool,
) -> None:
    working_dir = tmp_path / "wd"
    working_dir.mkdir()
    monkeypatch.setattr(
        core,
        "ItemProcessor",
        MagicMock(return_value=_fake_processor(success, working_dir)),
    )
    run_plan = _plan(make_pdf("doc.pdf"), force=force)

    asyncio.run(core.run(run_plan))

    assert working_dir.exists() is not removed


def test_kept_working_files_survive_a_forced_run(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    working_dir = tmp_path / "wd"
    working_dir.mkdir()
    monkeypatch.setattr(
        core,
        "ItemProcessor",
        MagicMock(return_value=_fake_processor(True, working_dir)),
    )
    run_plan = _plan(make_pdf("doc.pdf"), force=True, keep_working_files=True)

    asyncio.run(core.run(run_plan))

    assert working_dir.exists()


def test_item_verdict_and_outputs_reach_the_result(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = [tmp_path / "A.txt", tmp_path / "A.docx"]
    monkeypatch.setattr(
        core,
        "ItemProcessor",
        MagicMock(return_value=_fake_processor(False, tmp_path / "wd", written)),
    )
    run_plan = _plan(make_pdf("A.pdf"))

    result = asyncio.run(core.run(run_plan))

    assert result.failed == 1
    assert result.outputs == [str(path) for path in written]
    assert result.items[0].report == {"pages_ok": 1}


def test_a_crashing_item_does_not_stop_the_run(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")
    make_pdf("in/beta.pdf")
    healthy = _fake_processor(True, tmp_path / "wd")
    monkeypatch.setattr(
        core, "ItemProcessor", MagicMock(side_effect=[RuntimeError("boom"), healthy])
    )
    events = RecordingEvents()

    result = asyncio.run(core.run(_plan(tmp_path / "in", all=True), events))

    assert [item.success for item in result.items] == [False, True]
    assert [event.success for event in events.of("item_finished")] == [False, True]
    assert result.items[0].report is None


def test_a_run_of_skipped_items_returns_them(tmp_path: Path, make_pdf: MakePdf) -> None:
    pdf = make_pdf("doc.pdf")
    out = tmp_path / "out"
    out.mkdir()
    ItemPaths.for_item("doc", out).transcription.write_text("x", encoding="utf-8")
    run_plan = _plan(pdf, output=out, summarize=False)

    result = asyncio.run(core.run(run_plan))

    assert result.items == ()
    assert result.skipped == ("doc",)
    assert result.usage == {}


def test_cancellation_propagates_and_stops_the_run(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "in").mkdir()
    make_pdf("in/alpha.pdf")
    make_pdf("in/beta.pdf")
    processor = _fake_processor(False, tmp_path / "wd")
    processor.process = AsyncMock(side_effect=asyncio.CancelledError)
    factory = MagicMock(return_value=processor)
    monkeypatch.setattr(core, "ItemProcessor", factory)
    run_plan = _plan(tmp_path / "in", all=True)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(core.run(run_plan))

    assert factory.call_count == 1


@pytest.mark.parametrize(
    ("unset", "flags", "role"),
    [
        ("OPENAI_API_KEY", {"summarize": False}, "transcription"),
        ("ANTHROPIC_API_KEY", {"summary_provider": "anthropic"}, "summary"),
    ],
    ids=["transcription", "summary"],
)
def test_a_missing_key_stops_the_run_before_any_work(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    unset: str,
    flags: dict[str, Any],
    role: str,
) -> None:
    monkeypatch.delenv(unset)
    factory = MagicMock()
    monkeypatch.setattr(core, "ItemProcessor", factory)
    out = tmp_path / "out"
    run_plan = _plan(make_pdf("doc.pdf"), output=out, **flags)
    events = RecordingEvents()

    with pytest.raises(core.ConfigurationError, match=unset) as caught:
        asyncio.run(core.run(run_plan, events))

    assert role in str(caught.value)
    assert caught.value.exit_code == ExitCode.USAGE
    assert caught.value.items_total == 1
    assert not out.exists()
    assert factory.call_count == 0
    assert events.kinds == []


def test_the_checked_callers_serve_the_first_item(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    def build(phase: Any, timeouts: Any, **kwargs: Any) -> MagicMock:
        built.append(phase.name)
        return MagicMock()

    monkeypatch.setattr(caller_module, "build_chat_model", build)
    builds_seen: list[int] = []

    def processor(*args: Any, env: Any, **kwargs: Any) -> MagicMock:
        job = args[2]
        builds_seen.append(len(built))
        env.caller("transcription", job.transcription)
        builds_seen.append(len(built))
        env.caller("transcription", job.transcription)
        builds_seen.append(len(built))
        return _fake_processor(True, tmp_path / "wd")

    monkeypatch.setattr(core, "ItemProcessor", processor)
    run_plan = _plan(make_pdf("doc.pdf"), output=tmp_path / "out", summarize=False)

    asyncio.run(core.run(run_plan))

    # Built once before the item; the first caller is the prepared one.
    assert builds_seen == [1, 1, 2]


def test_a_dry_run_plan_is_not_run(tmp_path: Path, make_pdf: MakePdf) -> None:
    run_plan = _plan(make_pdf("doc.pdf"), dry_run=True)
    with pytest.raises(ValueError, match="dry-run"):
        asyncio.run(core.run(run_plan))


def test_remove_working_dir_retries_after_clearing_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "locked.txt"
    target.write_text("x", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_rmtree(path: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)
    core.remove_working_dir(tmp_path)

    removed: list[Any] = []
    captured["onexc"](removed.append, str(target), PermissionError("denied"))
    assert removed == [str(target)]
