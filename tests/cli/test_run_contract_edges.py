"""Run command edges through the full run subcommand.

- ext.startup runs once before the first model call and once for a dry run,
  but not for a real run whose items are all skipped.
- Cut-short runs report the tokens already spent.
- Runs fully handled by the resume check still write the JSON line.
- A summary-only resume reuses the logged transcription and reports its
  outputs.
- Deferred pages fail the item and withhold the partial outputs; the next
  run finishes them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import autoexcerpter.pipeline.managers as managers_module
import autoexcerpter.pipeline.outputs as outputs_module
import autoexcerpter.run as core
from autoexcerpter import __main__ as entry
from autoexcerpter import ext
from autoexcerpter.common.structured import StructuredRequest
from autoexcerpter.common.testing import FakeRequest
from autoexcerpter.common.usage import Usage
from autoexcerpter.constants import LOG_FORMAT_VERSION
from autoexcerpter.llm import ModelCaller
from autoexcerpter.pipeline.pages import PageRunner
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.rendering.sqlite import DATABASE_NAME
from autoexcerpter.settings import Settings
from autoexcerpter.spec import RunSpec
from tests.llm import helpers
from tests.llm.helpers import SUMMARY_ANSWER, TRANSCRIPTION_ANSWER

MakePdf = Callable[..., Path]
CALL = Usage(input_tokens=100, output_tokens=20, total_tokens=120)
MESSAGES = [SystemMessage(content="Transcribe."), HumanMessage(content="page")]
REQUEST = StructuredRequest(
    provider="openai",
    response_format="json",
    schema={"name": "s", "schema": {"type": "object", "required": ["a"]}},
)


def _run(*argv: str) -> int:
    return entry.main(["run", *argv])


def _summary(stdout: str) -> dict[str, Any]:
    (line,) = [line for line in stdout.splitlines() if line.startswith("{")]
    payload: dict[str, Any] = json.loads(line)
    return payload


def _is_transcription(request: FakeRequest) -> bool:
    content = request.messages[-1].content
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in content
    )


class _Timeline:
    """Records ext.startup calls and model calls in the order they happen."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entries: list[str] = []
        monkeypatch.setattr(ext, "startup", self.startup)

    def startup(
        self, spec: RunSpec, *, settings: Settings, args: argparse.Namespace
    ) -> None:
        self.entries.append("startup")

    def answer(self, request: FakeRequest) -> Any:
        self.entries.append("call")
        return TRANSCRIPTION_ANSWER if _is_transcription(request) else SUMMARY_ANSWER


def test_startup_runs_once_for_a_dry_run_without_model_calls(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _Timeline(monkeypatch)
    helpers.install(monkeypatch, [timeline.answer])

    assert _run("--input", str(make_pdf("doc.pdf")), "--dry-run") == 0
    assert timeline.entries == ["startup"]


def _all_complete(tmp_path: Path, make_pdf: MakePdf) -> list[str]:
    """Return run arguments for an input folder whose one item is complete."""
    (tmp_path / "in").mkdir()
    make_pdf("in/Done.pdf")
    out = tmp_path / "out"
    out.mkdir()
    ItemPaths.for_item("Done", out).transcription.write_text("x", encoding="utf-8")
    return ["--input", str(tmp_path / "in"), "--output", str(out), "--all"]


def test_startup_is_skipped_when_every_item_is_skipped(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _Timeline(monkeypatch)

    code = _run(*_all_complete(tmp_path, make_pdf), "--no-summarize")

    assert code == 0
    assert timeline.entries == []


def test_startup_runs_for_a_dry_run_whose_items_are_all_skipped(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _Timeline(monkeypatch)

    code = _run(*_all_complete(tmp_path, make_pdf), "--no-summarize", "--dry-run")

    assert code == 0
    assert timeline.entries == ["startup"]


@pytest.mark.usefixtures("mock_image_processing")
def test_startup_runs_once_before_the_first_model_call(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _Timeline(monkeypatch)
    helpers.install(monkeypatch, [timeline.answer])
    pdf = make_pdf("doc.pdf", num_pages=2)

    code = _run(
        "--input", str(pdf), "--output", str(tmp_path / "out"), "--no-summarize"
    )

    assert code == 0
    assert timeline.entries == ["startup", "call", "call"]


def _calls_then(failure: BaseException) -> Callable[..., Any]:
    """Return a stand-in for core.run: two model calls, then *failure*."""

    async def run(*_args: Any, **_kwargs: Any) -> Any:
        env = helpers.env()
        transcription = ModelCaller("transcription", helpers.phase(), env)
        summary = ModelCaller("summary", helpers.phase(), env)
        await transcription.call(MESSAGES, REQUEST, label="p1")
        await asyncio.create_task(summary.call(MESSAGES, REQUEST, label="p1"))
        raise failure

    return run


def _fail_at_once(failure: BaseException) -> Callable[..., Any]:
    async def run(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    return run


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_a_cut_short_run_reports_the_tokens_already_spent(
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    helpers.install(monkeypatch, [{"a": 1}])
    monkeypatch.setattr(core, "run", _calls_then(failure))

    assert _run("--input", str(make_pdf("doc.pdf")), "--json") == code

    payload = _summary(capsys.readouterr().out)
    assert payload["usage"] == {
        "transcription": CALL.as_dict(),
        "summary": CALL.as_dict(),
        "total": (CALL + CALL).as_dict(),
    }
    assert payload["items_total"] == 0


@pytest.mark.parametrize(
    ("failure", "code"), [(KeyboardInterrupt(), 130), (RuntimeError("boom"), 1)]
)
def test_a_run_cut_short_before_any_call_reports_zero_tokens(
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
) -> None:
    monkeypatch.setattr(core, "run", _fail_at_once(failure))

    assert _run("--input", str(make_pdf("doc.pdf")), "--json") == code

    assert _summary(capsys.readouterr().out)["usage"] == {"total": Usage().as_dict()}


@pytest.mark.usefixtures("mock_image_processing")
def test_an_interrupted_real_run_reports_the_page_calls(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roles: list[str] = []

    def answer(request: FakeRequest) -> Any:
        if len(roles) == 2:
            raise KeyboardInterrupt
        if _is_transcription(request):
            roles.append("transcription")
            return TRANSCRIPTION_ANSWER
        roles.append("summary")
        return SUMMARY_ANSWER

    helpers.install(monkeypatch, [answer])
    pdf = make_pdf("doc.pdf", num_pages=2)

    code = _run(
        "--input",
        str(pdf),
        "--output",
        str(tmp_path / "out"),
        "--concurrency",
        "1",
        "--no-openalex",
        "--json",
    )

    assert code == 130
    expected: dict[str, Usage] = {}
    for role in roles:
        expected[role] = expected.get(role, Usage()) + CALL
    assert _summary(capsys.readouterr().out)["usage"] == {
        **{role: usage.as_dict() for role, usage in expected.items()},
        "total": (CALL + CALL).as_dict(),
    }


def _ok_transcribe(payload: Any) -> dict[str, Any]:
    """Successful stand-in for TranscriptionManager.transcribe_payload."""
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Transcribed text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _ok_summary(transcription: str, page_num: int) -> dict[str, Any]:
    """Successful stand-in for SummaryManager.generate_summary."""
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": ["A point."],
        "references": None,
        "processing_time": 0.01,
        "provider": "openai",
    }


@pytest.fixture
def pipeline_env(
    monkeypatch: pytest.MonkeyPatch, mock_image_processing: dict[str, Any]
) -> MagicMock:
    """Patch the item processor's dependencies for offline runs (no API calls)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_manager = MagicMock()
    mock_manager.transcribe_payload = AsyncMock(side_effect=_ok_transcribe)
    monkeypatch.setattr(
        managers_module,
        "TranscriptionManager",
        MagicMock(return_value=mock_manager),
    )
    return mock_manager


@pytest.fixture
def summary_env(monkeypatch: pytest.MonkeyPatch, pipeline_env: MagicMock) -> MagicMock:
    """Enable summarization on top of pipeline_env with a mocked manager
    and mocked summary-file writers (no docx/md is actually rendered)."""
    mock_summary_manager = MagicMock()
    mock_summary_manager.generate_summary = AsyncMock(side_effect=_ok_summary)
    monkeypatch.setattr(
        managers_module,
        "SummaryManager",
        MagicMock(return_value=mock_summary_manager),
    )
    monkeypatch.setattr(
        outputs_module,
        "build_render_context",
        lambda *a, **k: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(outputs_module, "enrich_if_enabled", lambda cm: None)
    monkeypatch.setattr(outputs_module, "create_docx_summary", MagicMock())
    monkeypatch.setattr(outputs_module, "create_markdown_summary", MagicMock())
    return mock_summary_manager


def _seed_complete_log(
    output_dir: Path, item_name: str, input_path: Path, model_name: str
) -> None:
    """Write a versioned transcription log marking page 0 as completed."""
    paths = ItemPaths.for_item(item_name, output_dir)
    paths.working_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": item_name,
        "input_item_path": str(input_path),
        "input_type": "PDF",
        "total_images": 1,
        "model_name": model_name,
    }
    page = {
        "image": "page_0001.jpg",
        "original_input_order_index": 0,
        "transcription": "prior page text",
    }
    paths.transcription_log.write_text(
        json.dumps(header) + "\n" + json.dumps(page) + "\n", encoding="utf-8"
    )


def _run_main(in_dir: Path, out_dir: Path, *extra: str, summarize: bool = False) -> int:
    """Run the run subcommand with --json (resume is the default)."""
    return entry.main(
        [
            "run",
            "--input",
            str(in_dir),
            "--output",
            str(out_dir),
            "--all",
            "--json",
            "--summarize" if summarize else "--no-summarize",
            *extra,
        ]
    )


class TestResumeJsonEmission:
    """Runs handled by the resume check still write the JSON line."""

    def test_all_skipped_emits_json_line(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An all-skipped run writes the JSON line with items_skipped set."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Item.pdf", 1)
        # Without summaries only the transcription is required for the item
        # to be classified COMPLETE.
        paths = ItemPaths.for_item("Item", out_dir)
        paths.transcription.write_text("done", encoding="utf-8")

        exit_code = _run_main(in_dir, out_dir)

        out = capsys.readouterr().out.strip()
        assert out, "run must emit the --json summary line, not zero bytes"
        payload = json.loads(out.splitlines()[-1])
        assert exit_code == 0
        assert payload["items_skipped"] == 1
        assert payload["items_complete"] == 0
        assert payload["items_failed"] == 0
        assert payload["outputs"] == []
        # Nothing was processed: the transcription manager never ran.
        assert pipeline_env.transcribe_payload.call_count == 0

    def test_summary_only_resume_emits_json_with_outputs(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        summary_env: MagicMock,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A summary-only resume reuses the logged transcription.

        The transcription and its full log exist, the summaries do not; the run
        writes the summaries and reports the item complete with its outputs.
        """
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        pdf_path = make_pdf("in/Item.pdf", 1)
        paths = ItemPaths.for_item("Item", out_dir)
        paths.transcription.write_text("prior transcription", encoding="utf-8")
        _seed_complete_log(out_dir, "Item", pdf_path, "gpt-5-mini")

        exit_code = _run_main(in_dir, out_dir, summarize=True)

        out = capsys.readouterr().out.strip()
        assert out, "run must emit the --json summary line, not zero bytes"
        payload = json.loads(out.splitlines()[-1])
        assert exit_code == 0
        assert payload["items_complete"] == 1
        assert payload["items_failed"] == 0
        assert payload["items_skipped"] == 0
        expected_outputs = [
            str(paths.transcription.resolve()),
            str(paths.summary_docx.resolve()),
            str(paths.summary_md.resolve()),
            str((out_dir / DATABASE_NAME).resolve()),
        ]
        assert payload["outputs"] == expected_outputs
        # The logged transcription was reused (no fresh transcription call)
        # while the summaries were regenerated.
        assert pipeline_env.transcribe_payload.call_count == 0
        assert summary_env.generate_summary.call_count == 1


class TestDeferralWithholding:
    """Deferred pages fail the item and withhold its transcription file.

    The completed pages stay in the working log, so a later run resumes and
    finishes the deferred ones.
    """

    def test_deferral_roundtrip_via_main(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A run with deferred pages exits 1; the resumed run exits 0."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Item.pdf", 3)
        transcript = ItemPaths.for_item("Item", out_dir).transcription

        # Patch the class, because the run builds its own runner.
        real_run = PageRunner.run
        defer = {"on": True}

        async def maybe_defer(
            self: PageRunner,
            idx: int,
            source: Any,
            t_results: list[dict[str, Any]],
            s_results: list[dict[str, Any]],
            progress: PageProgress,
        ) -> dict[str, Any] | None:
            if defer["on"] and idx >= 1:
                return None
            return await real_run(self, idx, source, t_results, s_results, progress)

        monkeypatch.setattr(PageRunner, "run", maybe_defer)

        # First run: pages 1 and 2 deferred -> partial, withheld, exit 1.
        exit_code_1 = _run_main(in_dir, out_dir)
        payload_1 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert exit_code_1 == 1
        assert payload_1["items_failed"] >= 1
        assert payload_1["items_complete"] == 0
        # Only the database holds the item, as an incomplete row.
        assert payload_1["outputs"] == [str((out_dir / DATABASE_NAME).resolve())]
        assert not transcript.exists()

        # Second run: nothing deferred; resume finishes all pages, exit 0.
        defer["on"] = False
        exit_code_2 = _run_main(in_dir, out_dir)
        payload_2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert exit_code_2 == 0
        assert payload_2["items_failed"] == 0
        assert payload_2["items_complete"] == 1
        assert transcript.exists()
        # The resumed run reused page 0 and transcribed only the two that were
        # deferred (3 total minus the 1 already logged).
        txt = transcript.read_text(encoding="utf-8")
        assert "# Total images processed: 3" in txt
