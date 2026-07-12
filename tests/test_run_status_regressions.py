"""Regression tests for run-status and --json output propagation.

Covers two live-testing bugs:

- AE-2: page-level failures (transcription or summary API errors after
  retries) must propagate to the item verdict so the run summary counts the
  item failed and the process exits non-zero. Previously an item whose pages
  ALL failed was still reported complete with exit code 0.
- AE-1: the --json run summary reported ``"outputs": []`` even when .txt and
  .docx/.md files were written. ``ItemTranscriber.written_outputs`` now
  records every output file, and the CLI loop propagates it to
  ``_emit_json_summary``.
- AE-5: runs fully handled by the resume system emitted NOTHING on stdout —
  not even the --json summary line — and exited 0 silently, because
  ``_apply_resume_filtering`` called ``sys.exit(0)`` on the all-skipped path
  before ``_emit_json_summary`` was reached.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import pipeline.transcriber as transcriber_module
from config import app as app_config
from imaging.payload import PagePayload
from pipeline.transcriber import ItemTranscriber
from pipeline.types import ItemSpec


def _ok_transcribe(payload: PagePayload, max_schema_retries: int = 3) -> dict[str, Any]:
    """Successful stand-in for TranscriptionManager.transcribe_payload."""
    return {
        "image": payload.image_name,
        "sequence_number": payload.sequence_number,
        "transcription": f"Transcribed text of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _failed_transcribe(
    payload: PagePayload, max_schema_retries: int = 3
) -> dict[str, Any]:
    """API-failure stand-in mirroring TranscriptionManager's error shape."""
    return {
        "image": payload.image_name,
        "sequence_number": payload.sequence_number,
        "transcription": "[transcription error: API error after retries]",
        "processing_time": 0.01,
        "error": "API error after retries",
        "error_type": "api_failure",
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


def _failed_summary(transcription: str, page_num: int) -> dict[str, Any]:
    """API-failure stand-in mirroring SummaryManager's placeholder shape."""
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["other"],
        },
        "bullet_points": ["[Error generating summary: API error]"],
        "references": None,
        "error": "Summary API error after retries",
        "error_type": "api_failure",
        "provider": "openai",
    }


@pytest.fixture
def pipeline_env(
    monkeypatch: pytest.MonkeyPatch, mock_config_loader: MagicMock
) -> MagicMock:
    """Patch ItemTranscriber dependencies for offline runs (no API calls)."""
    mock_manager = MagicMock()
    mock_manager.transcribe_payload.side_effect = _ok_transcribe
    monkeypatch.setattr(
        transcriber_module,
        "TranscriptionManager",
        MagicMock(return_value=mock_manager),
    )
    monkeypatch.setattr(
        transcriber_module, "get_config_loader", lambda: mock_config_loader
    )
    monkeypatch.setattr("imaging.payload.get_config_loader", lambda: mock_config_loader)
    monkeypatch.setattr(app_config, "SUMMARIZE", False)
    return mock_manager


@pytest.fixture
def summary_env(monkeypatch: pytest.MonkeyPatch, pipeline_env: MagicMock) -> MagicMock:
    """Enable summarization on top of pipeline_env with a mocked manager
    and mocked summary-file writers (no docx/md is actually rendered)."""
    mock_summary_manager = MagicMock()
    mock_summary_manager.generate_summary.side_effect = _ok_summary
    monkeypatch.setattr(
        transcriber_module,
        "SummaryManager",
        MagicMock(return_value=mock_summary_manager),
    )
    monkeypatch.setattr(
        transcriber_module,
        "resolve_summary_context",
        lambda **kwargs: (None, None),
    )
    monkeypatch.setattr(
        transcriber_module,
        "build_render_context",
        lambda *a, **k: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(transcriber_module, "enrich_if_enabled", lambda cm: None)
    monkeypatch.setattr(transcriber_module, "create_docx_summary", MagicMock())
    monkeypatch.setattr(transcriber_module, "create_markdown_summary", MagicMock())
    monkeypatch.setattr(app_config, "SUMMARIZE", True)
    monkeypatch.setattr(app_config, "OUTPUT_DOCX", True)
    monkeypatch.setattr(app_config, "OUTPUT_MARKDOWN", True)
    return mock_summary_manager


def _make_transcriber(pdf_path: Path, output_dir: Path) -> ItemTranscriber:
    output_dir.mkdir(parents=True, exist_ok=True)
    return ItemTranscriber(
        input_path=pdf_path,
        input_type="pdf",
        base_output_dir=output_dir,
    )


# ============================================================================
# AE-2: page-level failure propagation to the item verdict
# ============================================================================
class TestItemFailurePropagation:
    """process_item must return False when any page fails."""

    def test_all_pages_succeed_returns_true(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        transcriber = _make_transcriber(make_pdf("OK.pdf", 2), tmp_path / "out")
        assert transcriber.process_item() is True

    def test_all_pages_fail_transcription_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        """Live-run regression: 3/3 failed pages still counted complete."""
        pipeline_env.transcribe_payload.side_effect = _failed_transcribe
        transcriber = _make_transcriber(make_pdf("AllFail.pdf", 3), tmp_path / "out")
        assert transcriber.process_item() is False

    def test_single_failed_page_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        calls: list[int] = []

        def flaky(payload: PagePayload, max_schema_retries: int = 3) -> dict[str, Any]:
            calls.append(payload.sequence_number)
            if payload.sequence_number == 2:
                return _failed_transcribe(payload)
            return _ok_transcribe(payload)

        pipeline_env.transcribe_payload.side_effect = flaky
        transcriber = _make_transcriber(make_pdf("OneFail.pdf", 3), tmp_path / "out")
        assert transcriber.process_item() is False
        assert sorted(calls) == [1, 2, 3]

    def test_failed_summary_page_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        summary_env: MagicMock,
    ) -> None:
        """Live-run regression: page 2's summary failed after retries but the
        run still reported full success and exited 0."""

        def flaky_summary(transcription: str, page_num: int) -> dict[str, Any]:
            if page_num == 2:
                return _failed_summary(transcription, page_num)
            return _ok_summary(transcription, page_num)

        summary_env.generate_summary.side_effect = flaky_summary
        transcriber = _make_transcriber(
            make_pdf("SummaryFail.pdf", 3), tmp_path / "out"
        )
        assert transcriber.process_item() is False

    def test_successful_summaries_return_true(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        summary_env: MagicMock,
    ) -> None:
        transcriber = _make_transcriber(make_pdf("SummaryOK.pdf", 2), tmp_path / "out")
        assert transcriber.process_item() is True

    def test_summary_render_failure_returns_false(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        summary_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed summary-file render means the item is not complete."""
        monkeypatch.setattr(
            transcriber_module,
            "create_docx_summary",
            MagicMock(side_effect=RuntimeError("docx exploded")),
        )
        transcriber = _make_transcriber(make_pdf("RenderFail.pdf", 1), tmp_path / "out")
        assert transcriber.process_item() is False

    def test_process_single_item_propagates_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cli.loop._process_single_item returns the item verdict, not a
        blanket True, so main counts the item failed and exits 1."""
        import cli.loop as loop_module

        instance = MagicMock()
        instance.process_item.return_value = False
        instance.written_outputs = [tmp_path / "Item.txt"]
        instance.working_dir = tmp_path / "wd"
        monkeypatch.setattr(
            loop_module, "ItemTranscriber", MagicMock(return_value=instance)
        )
        monkeypatch.setattr(app_config, "CLI_MODE", True)

        item = ItemSpec(kind="pdf", path=tmp_path / "Item.pdf")
        success, outputs = loop_module._process_single_item(item, 1, 1, tmp_path)
        assert success is False
        assert outputs == [str(tmp_path / "Item.txt")]


# ============================================================================
# AE-1: --json outputs population
# ============================================================================
class TestJsonOutputs:
    """written_outputs must flow from the transcriber to the JSON summary."""

    def test_written_outputs_contains_txt(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
    ) -> None:
        transcriber = _make_transcriber(make_pdf("Outs.pdf", 1), tmp_path / "out")
        assert transcriber.process_item() is True
        assert transcriber.written_outputs == [transcriber.output_txt_path.resolve()]
        assert transcriber.output_txt_path.exists()

    def test_written_outputs_include_summary_files(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        summary_env: MagicMock,
    ) -> None:
        transcriber = _make_transcriber(make_pdf("Summ.pdf", 1), tmp_path / "out")
        assert transcriber.process_item() is True
        assert transcriber.written_outputs == [
            transcriber.output_txt_path.resolve(),
            transcriber.output_summary_docx_path.resolve(),
            transcriber.output_summary_md_path.resolve(),
        ]

    def test_process_single_item_returns_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.loop as loop_module

        written = [tmp_path / "A.txt", tmp_path / "A.docx"]
        instance = MagicMock()
        instance.process_item.return_value = True
        instance.written_outputs = written
        instance.working_dir = tmp_path / "wd"
        monkeypatch.setattr(
            loop_module, "ItemTranscriber", MagicMock(return_value=instance)
        )
        monkeypatch.setattr(app_config, "CLI_MODE", True)

        item = ItemSpec(kind="pdf", path=tmp_path / "A.pdf")
        success, outputs = loop_module._process_single_item(item, 1, 1, tmp_path)
        assert success is True
        assert outputs == [str(p) for p in written]

    def test_emit_json_summary_includes_outputs(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: the emitted JSON line reported "outputs": [] even on
        successful runs that wrote output files."""
        import main as main_module

        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        outputs = ["C:/out/A.txt", "C:/out/A.docx"]
        main_module._emit_json_summary(1, 0, 0, 1, outputs)

        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["outputs"] == outputs
        # Backward-compatible keys stay present.
        assert payload["items_total"] == 1
        assert payload["items_complete"] == 1
        assert payload["items_failed"] == 0
        assert payload["items_skipped"] == 0


# ============================================================================
# AE-5: resume paths must still emit the --json summary line
# ============================================================================
def _seed_complete_log(
    output_dir: Path, item_name: str, input_path: Path, model_name: str
) -> None:
    """Write a versioned transcription log marking page 0 as completed."""
    from config.constants import LOG_FORMAT_VERSION
    from pipeline.paths import create_safe_directory_name, create_safe_log_filename

    working_dir = output_dir / create_safe_directory_name(item_name, "_working_files")
    working_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": item_name,
        "input_item_path": str(input_path),
        "input_type": "PDF",
        "total_images": 1,
        "model_name": model_name,
    }
    entry = {
        "image": "page_0001.jpg",
        "original_input_order_index": 0,
        "transcription": "prior page text",
    }
    log_path = working_dir / create_safe_log_filename(item_name, "transcription")
    log_path.write_text(
        json.dumps(header) + "\n" + json.dumps(entry) + "\n", encoding="utf-8"
    )


def _run_main(monkeypatch: pytest.MonkeyPatch, in_dir: Path, out_dir: Path) -> int:
    """Invoke main.main() in CLI mode with --resume --json."""
    import sys

    import main as main_module

    monkeypatch.setattr(app_config, "CLI_MODE", True)
    monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
    monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--cli", str(in_dir), str(out_dir), "--resume", "--json"],
    )
    return main_module.main()


class TestResumeJsonEmission:
    """Live-run regressions: resume-handled runs printed zero bytes."""

    def test_all_skipped_emits_json_line(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An all-complete (all-skipped) run must still emit the JSON
        summary with items_skipped populated, not sys.exit(0) silently."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Item.pdf", 1)
        # SUMMARIZE is False under pipeline_env: only the .txt is required
        # for the item to be classified COMPLETE.
        (out_dir / "Item.txt").write_text("done", encoding="utf-8")

        exit_code = _run_main(monkeypatch, in_dir, out_dir)

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
        """A summary-only resume run (TRANSCRIPTION_ONLY: .txt and full log
        exist, .docx/.md missing) must reuse the logged transcription,
        regenerate the summaries, and emit the JSON summary with
        items_complete and outputs populated."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        pdf_path = make_pdf("in/Item.pdf", 1)
        (out_dir / "Item.txt").write_text("prior transcription", encoding="utf-8")
        _seed_complete_log(out_dir, "Item", pdf_path, "gpt-5-mini")

        exit_code = _run_main(monkeypatch, in_dir, out_dir)

        out = capsys.readouterr().out.strip()
        assert out, "run must emit the --json summary line, not zero bytes"
        payload = json.loads(out.splitlines()[-1])
        assert exit_code == 0
        assert payload["items_complete"] == 1
        assert payload["items_failed"] == 0
        assert payload["items_skipped"] == 0
        expected_outputs = [
            str((out_dir / "Item.txt").resolve()),
            str((out_dir / "Item.docx").resolve()),
            str((out_dir / "Item.md").resolve()),
        ]
        assert payload["outputs"] == expected_outputs
        # The logged transcription was reused (no fresh transcription call)
        # while the summaries were regenerated.
        assert pipeline_env.transcribe_payload.call_count == 0
        assert summary_env.generate_summary.call_count == 1


# ============================================================================
# AE-6: budget-deferred pages must fail the item and withhold truncated output
# ============================================================================
class TestBudgetDeferralWithholding:
    """A run cut short by the daily token budget must NOT emit a truncated .txt
    that self-reports as complete, must fail the item (exit 1, items_failed),
    and must retain the completed pages in the working log so a later run
    resumes and finishes them."""

    def test_partial_run_withholds_txt_and_fails(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """process_item() is False, the final .txt is NOT written, nothing is
        registered in written_outputs, and the working log holds exactly the
        one page that completed before the budget stalled."""
        transcriber = _make_transcriber(make_pdf("Partial.pdf", 3), tmp_path / "out")

        # Defer every page after index 0 (budget exhausted); page 0 transcribes
        # for real via the mocked manager.
        real_process = transcriber._process_single_page

        def deferring(
            idx: int,
            source: Any,
            t_results: list[dict[str, Any]],
            s_results: list[dict[str, Any]],
            total: int,
            count_ref: list[int],
        ) -> dict[str, Any] | None:
            if idx >= 1:
                transcriber._budget_exhausted.set()
                return None
            return real_process(idx, source, t_results, s_results, total, count_ref)

        monkeypatch.setattr(transcriber, "_process_single_page", deferring)
        # The daily reset never comes (user cancels / limit unreachable).
        monkeypatch.setattr(
            transcriber_module, "wait_for_token_reset", lambda **_k: False
        )

        assert transcriber.process_item() is False
        # Leg 2: the truncated .txt must not exist and must not be advertised.
        assert not transcriber.output_txt_path.exists()
        assert transcriber.written_outputs == []
        # Leg 1 preservation: the working log retains exactly the completed page
        # so the resume completeness gate can finish the rest later.
        from pipeline.resume import load_completed_pages

        assert load_completed_pages(transcriber.log_path) == {0}

    def test_budget_deferral_roundtrip_via_main(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Main-level: a partial (budget-stalled) run exits 1 with
        items_failed >= 1 and writes no .txt; a subsequent resume run finishes
        every page and exits 0."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Item.pdf", 3)

        # Class-level patch so the instance main() builds internally defers.
        real_process = ItemTranscriber._process_single_page
        defer = {"on": True}

        def maybe_defer(
            self: ItemTranscriber,
            idx: int,
            source: Any,
            t_results: list[dict[str, Any]],
            s_results: list[dict[str, Any]],
            total: int,
            count_ref: list[int],
        ) -> dict[str, Any] | None:
            if defer["on"] and idx >= 1:
                self._budget_exhausted.set()
                return None
            return real_process(
                self, idx, source, t_results, s_results, total, count_ref
            )

        monkeypatch.setattr(ItemTranscriber, "_process_single_page", maybe_defer)
        monkeypatch.setattr(
            transcriber_module, "wait_for_token_reset", lambda **_k: False
        )

        # --- First run: pages 1,2 deferred -> partial, withheld, exit 1 ---
        exit_code_1 = _run_main(monkeypatch, in_dir, out_dir)
        payload_1 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert exit_code_1 == 1
        assert payload_1["items_failed"] >= 1
        assert payload_1["items_complete"] == 0
        assert payload_1["outputs"] == []
        assert not (out_dir / "Item.txt").exists()

        # --- Second run: budget healthy; resume finishes all pages, exit 0 ---
        defer["on"] = False
        exit_code_2 = _run_main(monkeypatch, in_dir, out_dir)
        payload_2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert exit_code_2 == 0
        assert payload_2["items_failed"] == 0
        assert payload_2["items_complete"] == 1
        assert (out_dir / "Item.txt").exists()
        # The resumed run reused page 0 and transcribed only the two that were
        # deferred (3 total minus the 1 already logged).
        txt = (out_dir / "Item.txt").read_text(encoding="utf-8")
        assert "# Total images processed: 3" in txt


def _run_main_argv(monkeypatch: pytest.MonkeyPatch, extra_argv: list[str]) -> int:
    """Invoke main.main() in CLI mode with an arbitrary argv tail."""
    import sys

    import main as main_module

    monkeypatch.setattr(app_config, "CLI_MODE", True)
    monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
    monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
    monkeypatch.setattr(sys, "argv", ["main.py", "--cli", *extra_argv])
    return main_module.main()


# ============================================================================
# AE-4 / AE-5: controlled selection exits must still emit the JSON summary
# ============================================================================
class TestSelectionExitJsonEmission:
    def test_no_match_select_exits_nonzero_with_json(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A --select pattern matching nothing is a user error: exit non-zero
        (not a silent success) while still emitting the JSON summary (AE-5)."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Alpha.pdf", 1)

        with pytest.raises(SystemExit) as exc:
            _run_main_argv(
                monkeypatch,
                [str(in_dir), str(out_dir), "--select", "NoSuchName", "--json"],
            )
        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["items_total"] == 0
        assert payload["items_complete"] == 0

    def test_multiple_items_no_all_exits_with_json(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Multiple items and neither --all nor --select is ambiguous: exit 2
        but still emit the JSON summary (AE-4)."""
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Alpha.pdf", 1)
        make_pdf("in/Beta.pdf", 1)

        with pytest.raises(SystemExit) as exc:
            _run_main_argv(monkeypatch, [str(in_dir), str(out_dir), "--json"])
        assert exc.value.code == 2
        out = capsys.readouterr().out.strip()
        assert out, "controlled exit must still emit the --json summary line"
        payload = json.loads(out.splitlines()[-1])
        assert payload["items_total"] == 0


# ============================================================================
# AE-7: items never reached (cancelled token wait) are not counted as failures
# ============================================================================
class TestUnattemptedNotFailed:
    def test_cancelled_wait_reports_unattempted_not_failed(
        self,
        make_pdf: Callable[..., Path],
        tmp_path: Path,
        pipeline_env: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When the user cancels the token-limit wait before a later item, that
        item is reported as not attempted (not a failure); the run still exits
        non-zero because the requested work did not finish."""
        import main as main_module

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Alpha.pdf", 1)
        make_pdf("in/Beta.pdf", 1)

        # Allow the first item through, then "cancel" before the second.
        calls = {"n": 0}

        def fake_wait() -> bool:
            calls["n"] += 1
            return calls["n"] == 1

        monkeypatch.setattr(main_module, "_check_and_wait_for_token_limit", fake_wait)

        exit_code = _run_main_argv(
            monkeypatch, [str(in_dir), str(out_dir), "--all", "--json"]
        )

        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert exit_code == 1
        assert payload["items_total"] == 2
        assert payload["items_complete"] == 1
        # The never-attempted item is NOT a failure ...
        assert payload["items_failed"] == 0
        # ... it is reported under the skipped/pending count instead.
        assert payload["items_skipped"] == 1
