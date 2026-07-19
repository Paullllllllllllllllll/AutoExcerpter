"""Round-3 CLI/main regression tests.

One test group per fix in the round-3 CLI hardening pass:

1.  EOF/interrupt robustness in the shared prompt helpers.
2.  The "select all" (N+1) menu entry is now displayed, not hidden.
3.  Model/behavior override flags take effect in interactive mode.
4.  --json is emitted when an interactive user declines processing.
5.  The detailed completion overview and per-item aggregation.
6.  Duplicate-output-target guard.
7.  Token-limit wait-loop throttling.
8.  Selection search is restricted to the item name (no parent-path matches).
9.  The token-limit banner is not duplicated in interactive mode.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from config import app as app_config
from pipeline.types import ItemSpec


# ============================================================================
# Fix 1: EOF / KeyboardInterrupt robustness
# ============================================================================
class TestPromptInterruptRobustness:
    def test_prompt_yes_no_eof_exits_code_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.interaction as inter

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        with pytest.raises(SystemExit) as exc:
            inter.prompt_yes_no("Proceed?")
        assert exc.value.code == 1

    def test_prompt_continue_keyboard_interrupt_exits_130(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.interaction as inter

        def _raise_kbi(_prompt: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_kbi)
        with pytest.raises(SystemExit) as exc:
            inter.prompt_continue()
        assert exc.value.code == 130

    def test_prompt_selection_eof_exits_not_loops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A closed stdin must exit, not loop forever re-calling input()."""
        import cli.interaction as inter

        calls = {"n": 0}

        def _raise_eof(_prompt: str) -> str:
            calls["n"] += 1
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        with pytest.raises(SystemExit) as exc:
            inter.prompt_selection(["a", "b", "c"], display_func=str)
        assert exc.value.code == 1
        assert calls["n"] == 1

    def test_bad_range_prints_single_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An out-of-bounds range must print exactly one error, then re-prompt."""
        import cli.interaction as inter

        responses = iter(["1-100", "1"])
        monkeypatch.setattr("builtins.input", lambda _p: next(responses))
        result = inter.prompt_selection(["a", "b", "c"], display_func=str)
        assert result == ["a"]
        out = capsys.readouterr().out
        assert out.count("[ERROR]") == 1


# ============================================================================
# Fix 2: the N+1 "select all" entry is displayed
# ============================================================================
class TestSelectAllVisible:
    def test_process_all_label_is_shown(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import cli.interaction as inter

        monkeypatch.setattr("builtins.input", lambda _p: "1")
        inter.prompt_selection(
            ["a", "b", "c"],
            display_func=str,
            allow_all=True,
            process_all_label="Process ALL listed items",
        )
        out = capsys.readouterr().out
        assert "Process ALL listed items" in out
        # N+1 numbered entry (4 for three items).
        assert "4." in out

    def test_n_plus_one_still_selects_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.interaction as inter

        monkeypatch.setattr("builtins.input", lambda _p: "4")
        result = inter.prompt_selection(["a", "b", "c"], display_func=str)
        assert result == ["a", "b", "c"]


# ============================================================================
# Fix 3: interactive-mode override flags
# ============================================================================
class TestInteractiveOverrides:
    def test_parse_execution_mode_uses_output_and_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argparse

        from cli.args import _parse_execution_mode

        monkeypatch.setattr(app_config, "CLI_MODE", False)
        args = argparse.Namespace(
            input="C:/abs/in",
            output="C:/abs/out",
            context="Food History",
            resume=None,
            force=None,
        )
        input_path, output_path, process_all, select, context, _mode = (
            _parse_execution_mode(args)
        )
        assert output_path == Path("C:/abs/out")
        assert context == "Food History"
        assert process_all is False
        assert select is None

    def test_parse_execution_mode_resolves_relative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import argparse

        from cli.args import _parse_execution_mode

        monkeypatch.setattr(app_config, "CLI_MODE", False)
        args = argparse.Namespace(
            input="rel/in", output="rel/out", context=None, resume=None, force=None
        )
        input_path, output_path, *_ = _parse_execution_mode(args)
        assert input_path.is_absolute()
        assert output_path.is_absolute()


# ============================================================================
# Fix 4: --json on interactive decline
# ============================================================================
class TestJsonOnDecline:
    def test_decline_emits_json_summary(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import sys

        import main as main_module

        item = ItemSpec(kind="pdf", path=Path("/tmp/A.pdf"))

        monkeypatch.setattr(app_config, "CLI_MODE", False)
        monkeypatch.setattr(app_config, "SUMMARIZE", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys, "argv", ["main.py", "--json"])
        monkeypatch.setattr(
            main_module,
            "_setup_and_scan",
            lambda args: ([item], Path("/tmp/out"), None, "skip"),
        )
        monkeypatch.setattr(
            main_module,
            "_apply_resume_filtering",
            lambda *a, **k: ([item], {}, "skip", []),
        )
        monkeypatch.setattr(
            main_module, "_display_processing_summary", lambda *a, **k: False
        )

        rc = main_module.main()
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["items_total"] == 1
        assert payload["items_complete"] == 0
        assert payload["items_failed"] == 0
        assert payload["items_skipped"] == 0
        assert payload["outputs"] == []


# ============================================================================
# Fix 5: detailed completion overview + aggregation
# ============================================================================
def _report(**kwargs: Any) -> dict[str, Any]:
    base = {
        "pages_total": 2,
        "pages_attempted": 2,
        "pages_ok": 2,
        "pages_failed": 0,
        "pages_deferred": 0,
        "summary_failures": 0,
        "elapsed_s": 1.0,
        "avg_api_s": 0.5,
        "outputs": [],
    }
    base.update(kwargs)
    return base


class TestCompletionOverview:
    def test_summarize_reports_aggregates(self) -> None:
        from cli.display import _summarize_reports

        reports = [
            ("A", _report(pages_ok=2, pages_failed=1)),
            ("B", _report(pages_ok=1, pages_deferred=3, summary_failures=2)),
            ("C", None),
        ]
        ok, failed, deferred, summary_failures = _summarize_reports(reports)
        assert (ok, failed, deferred, summary_failures) == (3, 1, 3, 2)

    def test_cli_overview_prints_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cli.display import _print_cli_completion_overview

        _print_cli_completion_overview(
            2,
            3,
            skipped_count=1,
            unattempted_count=0,
            reports=[("A", _report()), ("B", _report(pages_failed=2))],
            run_seconds=12.5,
            tokens_used_run=1234,
            run_outputs=["/out/A.txt", "/out/B.txt"],
        )
        captured = capsys.readouterr()
        assert captured.out == ""  # nothing on stdout
        err = captured.err
        assert "Run complete" in err
        assert "tokens this run: 1,234" in err
        assert "run time: 12.5s" in err

    def test_cli_overview_caps_outputs_at_20(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cli.display import _print_cli_completion_overview

        outputs = [f"/out/f{i}.txt" for i in range(25)]
        _print_cli_completion_overview(
            1,
            1,
            skipped_count=0,
            unattempted_count=0,
            reports=[("A", _report())],
            run_seconds=1.0,
            tokens_used_run=None,
            run_outputs=outputs,
        )
        err = capsys.readouterr().err
        assert "... and 5 more" in err

    def test_completion_summary_relabels_not_attempted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cli.display import _display_completion_summary

        monkeypatch.setattr(app_config, "SUMMARIZE", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        _display_completion_summary(
            1,
            3,
            Path("/out"),
            skipped_count=0,
            unattempted_count=1,
            reports=[("A", _report())],
            run_seconds=2.0,
            tokens_used_run=None,
            run_outputs=[],
        )
        out = capsys.readouterr().out
        assert "not attempted" in out
        assert "were skipped" not in out


class TestWarnIncompleteWithReports:
    def test_warning_includes_page_counts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from main import _warn_incomplete_items

        reports: dict[str, dict[str, Any] | None] = {
            "Book_A": _report(pages_failed=2, pages_deferred=1)
        }
        _warn_incomplete_items(["Book_A", "Book_B"], reports)
        out = capsys.readouterr().out
        # Book_A has a report: exact counts shown.
        assert "2 failed, 1 deferred" in out
        # Book_B has no report: fallback line.
        assert "Book_B (one or more pages failed or were deferred)" in out


# ============================================================================
# Fix 6: duplicate-output guard
# ============================================================================
class TestDuplicateOutputGuard:
    def test_cli_collision_exits_2_with_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main as main_module

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        items = [
            ItemSpec(kind="pdf", path=Path("/a/Book.pdf")),
            ItemSpec(kind="pdf", path=Path("/b/Book.pdf")),
        ]
        with pytest.raises(SystemExit) as exc:
            main_module._guard_duplicate_outputs(items, Path("/out"), emit_json=True)
        assert exc.value.code == 2
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["items_total"] == 2

    def test_interactive_collision_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import main as main_module

        monkeypatch.setattr(app_config, "CLI_MODE", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        items = [
            ItemSpec(kind="pdf", path=Path("/a/Book.pdf")),
            ItemSpec(kind="pdf", path=Path("/b/Book.pdf")),
        ]
        with pytest.raises(SystemExit) as exc:
            main_module._guard_duplicate_outputs(items, Path("/out"), emit_json=False)
        assert exc.value.code != 0

    def test_no_collision_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import main as main_module

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        items = [
            ItemSpec(kind="pdf", path=Path("/a/Book.pdf")),
            ItemSpec(kind="pdf", path=Path("/a/Other.pdf")),
        ]
        # Must not raise.
        main_module._guard_duplicate_outputs(items, Path("/out"), emit_json=False)


# ============================================================================
# Fix 7: token-limit wait-loop throttling
# ============================================================================
class TestWaitLoopThrottling:
    def test_sync_and_reload_are_throttled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        import cli.loop as loop_module

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(loop_module, "_user_requested_cancel", lambda: False)
        monkeypatch.setattr(time, "sleep", lambda _s: None)

        reload_calls = {"limit": 0, "pool": 0}
        monkeypatch.setattr(
            app_config,
            "reload_daily_token_limit",
            lambda: reload_calls.__setitem__("limit", reload_calls["limit"] + 1),
        )
        monkeypatch.setattr(
            app_config,
            "reload_pool_settings",
            lambda: reload_calls.__setitem__("pool", reload_calls["pool"] + 1),
        )

        tracker = MagicMock()
        tracker._shared_enabled = True
        tracker.is_limit_reached.return_value = True

        result = loop_module._wait_for_token_reset(tracker, seconds_until_reset=35)
        assert result is True
        # 35 s wait: ledger sync fires early then every ~10 s (elapsed 1/11/21/31)
        # -> 4 times, not 35. Throttled well below one-per-second.
        assert tracker.sync_ledger_now.call_count == 4
        # Config re-read fires early then every ~30 s (elapsed 1/31) -> 2 times.
        assert reload_calls["limit"] == 2
        assert reload_calls["pool"] == 2


# ============================================================================
# Fix 8: selection search restricted to item name
# ============================================================================
class TestSelectionSearchDomain:
    def test_parent_path_term_does_not_match_all(self) -> None:
        from cli.interaction import _match_items_by_name

        items = [
            ItemSpec(kind="pdf", path=Path(r"C:/Shared/Alpha.pdf")),
            ItemSpec(kind="pdf", path=Path(r"C:/Shared/Beta.pdf")),
        ]
        # "Shared" is only in the common parent path -> matches nothing now.
        matched = _match_items_by_name("Shared", items, lambda it: it.display_label())
        assert matched == set()

    def test_filename_term_still_matches(self) -> None:
        from cli.interaction import _match_items_by_name

        items = [
            ItemSpec(kind="pdf", path=Path(r"C:/Shared/Alpha.pdf")),
            ItemSpec(kind="pdf", path=Path(r"C:/Shared/Beta.pdf")),
        ]
        matched = _match_items_by_name("Alpha", items, lambda it: it.display_label())
        assert matched == {0}


# ============================================================================
# Fix 9: token banner not duplicated in interactive mode
# ============================================================================
class TestTokenBannerGating:
    def _stats(self) -> dict[str, Any]:
        return {"tokens_used_today": 100, "daily_limit": 100}

    def test_interactive_no_logger_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import cli.display as display_module

        mock_logger = MagicMock()
        monkeypatch.setattr(display_module, "logger", mock_logger)
        monkeypatch.setattr(app_config, "CLI_MODE", False)

        reset = datetime(2026, 7, 20, tzinfo=UTC)
        display_module._log_token_limit_reached(self._stats(), reset, 1, 30)

        # Interactive: pretty block on stdout, no duplicated logger.warning.
        assert mock_logger.warning.call_count == 0
        assert "Daily Token Limit Reached" in capsys.readouterr().out

    def test_cli_uses_logger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cli.display as display_module

        mock_logger = MagicMock()
        monkeypatch.setattr(display_module, "logger", mock_logger)
        monkeypatch.setattr(app_config, "CLI_MODE", True)

        reset = datetime(2026, 7, 20, tzinfo=UTC)
        display_module._log_token_limit_reached(self._stats(), reset, 1, 30)

        assert mock_logger.warning.call_count == 1
