"""Round-5 CLI/config regression tests.

One test group per behavioral fix in this pass:

1.  ``run_exit_hook`` fires the registered hook once, and the ``__main__``
    KeyboardInterrupt / unhandled-exception handlers use it so a ``--json``
    run emits its summary line on those exit paths too.
2.  The interactive branch of the duplicate-output guard emits the JSON
    summary before exiting, like its CLI-mode sibling.
3.  Every emitted JSON summary carries a ``dry_run`` boolean.
4.  The CLI token-limit wait lines are logged at WARNING, the only human
    channel in CLI mode.
5.  The wait loop no longer claims a reset it never observed.
6.  The working-directory cleanup uses ``shutil.rmtree(onexc=...)``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from config import app as app_config
from config.loader import PROJECT_ROOT
from pipeline.types import ItemSpec


@pytest.fixture(autouse=True)
def _clear_exit_hook() -> Any:
    """Keep the module-level exit hook from leaking across tests."""
    import cli.interaction as inter

    inter.set_exit_hook(None)
    yield
    inter.set_exit_hook(None)


# ============================================================================
# Fix 1: run_exit_hook and the __main__ exit handlers
# ============================================================================
class TestRunExitHook:
    """Unit-level behavior of the extracted one-shot hook runner."""

    def test_fires_registered_hook(self) -> None:
        import cli.interaction as inter

        calls: list[str] = []
        inter.set_exit_hook(lambda: calls.append("fired"))

        inter.run_exit_hook()

        assert calls == ["fired"]

    def test_is_one_shot(self) -> None:
        import cli.interaction as inter

        calls: list[int] = []
        inter.set_exit_hook(lambda: calls.append(1))

        inter.run_exit_hook()
        inter.run_exit_hook()

        assert calls == [1]
        assert inter._exit_hook is None

    def test_clears_before_invoking(self) -> None:
        """The hook is cleared BEFORE it runs, so a re-entrant call is a no-op."""
        import cli.interaction as inter

        seen: list[Any] = []

        def reentrant() -> None:
            seen.append(inter._exit_hook)
            inter.run_exit_hook()  # must not recurse

        inter.set_exit_hook(reentrant)
        inter.run_exit_hook()

        assert seen == [None]

    def test_noop_when_unset(self) -> None:
        import cli.interaction as inter

        inter.set_exit_hook(None)
        inter.run_exit_hook()  # must not raise

    def test_hook_exception_is_suppressed(self) -> None:
        import cli.interaction as inter

        def boom() -> None:
            raise RuntimeError("nope")

        inter.set_exit_hook(boom)
        inter.run_exit_hook()  # must not propagate

    def test_emit_json_summary_disarms_the_hook(self) -> None:
        """Single-emission semantics: emitting clears the hook."""
        import cli.interaction as inter
        import main.excerpt as main_module

        inter.set_exit_hook(lambda: main_module._emit_json_summary(0, 0, 0, 0, []))
        main_module._emit_json_summary(0, 0, 0, 0, [])

        assert inter._exit_hook is None


def _run_main_as_script(
    tmp_path: Path, failure: str
) -> subprocess.CompletedProcess[str]:
    """Execute main/excerpt.py as ``__main__`` with a mid-run failure injected.

    ``cli.args._parse_execution_mode`` is replaced with a raiser, which fires
    inside ``_setup_and_scan`` — i.e. AFTER ``main()`` has armed the exit hook
    but before any work is done. That is exactly the window the ``__main__``
    handlers cover.
    """
    script = textwrap.dedent(
        f"""
        import runpy
        import sys

        import cli.args
        from config import app as app_config

        # Keep the token tracker (and any shared ledger) out of this run.
        app_config.DAILY_TOKEN_LIMIT_ENABLED = False

        def _raise(args):
            raise {failure}

        cli.args._parse_execution_mode = _raise
        sys.argv = [
            "main/excerpt.py", "--cli", r"{tmp_path / "in"}", r"{tmp_path / "out"}",
            "--json"
        ]
        runpy.run_path("main/excerpt.py", run_name="__main__")
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


class TestMainExitHandlersEmitJson:
    """Regression: the __main__ handlers exited without emitting the summary.

    The exit hook registered in ``main()`` only ever fired through
    ``cli.interaction.exit_program``, so Ctrl+C and unhandled exceptions
    produced zero bytes on stdout — silently breaking the documented
    "summary line emitted on every exit path" contract.
    """

    def test_keyboard_interrupt_emits_json_once(self, tmp_path: Path) -> None:
        result = _run_main_as_script(tmp_path, "KeyboardInterrupt")

        assert result.returncode == 130
        lines = [ln for ln in result.stdout.splitlines() if ln.startswith("{")]
        assert len(lines) == 1, result.stdout
        payload = json.loads(lines[0])
        assert payload["items_total"] == 0
        assert payload["items_complete"] == 0
        assert payload["outputs"] == []

    def test_unhandled_exception_emits_json_once(self, tmp_path: Path) -> None:
        result = _run_main_as_script(tmp_path, "RuntimeError('boom')")

        assert result.returncode == 1
        lines = [ln for ln in result.stdout.splitlines() if ln.startswith("{")]
        assert len(lines) == 1, result.stdout
        assert json.loads(lines[0])["items_total"] == 0


# ============================================================================
# Fix 2: interactive duplicate-output guard honors --json
# ============================================================================
class TestInteractiveDuplicateGuardJson:
    def test_interactive_collision_emits_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Regression: only the CLI-mode branch emitted; interactive was silent."""
        import main.excerpt as main_module

        monkeypatch.setattr(app_config, "CLI_MODE", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        items = [
            ItemSpec(kind="pdf", path=Path("/a/Book.pdf")),
            ItemSpec(kind="pdf", path=Path("/b/Book.pdf")),
        ]

        with pytest.raises(SystemExit) as exc:
            main_module._guard_duplicate_outputs(items, Path("/out"), emit_json=True)

        assert exc.value.code == 1
        lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        # Same shape the CLI-mode branch emits: the colliding items counted
        # as the attempted total, nothing completed.
        assert payload["items_total"] == 2
        assert payload["items_complete"] == 0
        assert payload["items_failed"] == 0


# ============================================================================
# Fix 3: every JSON summary stamps dry_run
# ============================================================================
class TestDryRunKey:
    def test_default_summary_is_not_dry_run(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main.excerpt as main_module

        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        main_module._emit_json_summary(1, 0, 0, 1, ["/out/A.txt"])

        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["dry_run"] is False

    def test_flag_is_threaded_through(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main.excerpt as main_module

        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        main_module._emit_json_summary(0, 0, 0, 0, [], dry_run=True)

        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["dry_run"] is True

    def test_early_exit_under_dry_run_is_unambiguous(
        self,
        make_pdf: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Regression: a --dry-run early exit emitted a shape with no dry_run key.

        A consumer could not tell it apart from a real run that did nothing.
        """
        import main.excerpt as main_module

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Alpha.pdf", 1)

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "main/excerpt.py",
                "--cli",
                str(in_dir),
                str(out_dir),
                "--select",
                "NoSuchName",
                "--json",
                "--dry-run",
            ],
        )

        with pytest.raises(SystemExit) as exc:
            main_module.main()

        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["dry_run"] is True

    def test_dry_run_plan_line_keeps_its_dry_run_key(
        self,
        make_pdf: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The plan-shaped dry-run line is unchanged and still marked."""
        import main.excerpt as main_module

        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        make_pdf("in/Alpha.pdf", 1)

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "main/excerpt.py",
                "--cli",
                str(in_dir),
                str(out_dir),
                "--json",
                "--dry-run",
            ],
        )

        assert main_module.main() == 0
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["dry_run"] is True
        assert [entry["name"] for entry in payload["to_process"]] == ["Alpha"]


# ============================================================================
# Fix 4: CLI token-limit wait lines are visible
# ============================================================================
class TestTokenLimitWaitVisibility:
    def test_wait_and_cancel_lines_are_warnings(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression: both lines were INFO, below the WARNING console cutoff.

        In CLI mode the logger is the only human channel, so a run that
        waited for up to 24 h printed nothing and looked hung.
        """
        import logging

        import cli.display as display

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        stats = {"tokens_used_today": 10, "daily_limit": 100}

        with caplog.at_level(logging.INFO, logger=display.logger.name):
            display.logger.propagate = True
            try:
                display._log_token_limit_reached(
                    stats, datetime(2026, 7, 31, 0, 1), 3, 30
                )
            finally:
                display.logger.propagate = False

        waiting = [r for r in caplog.records if "Waiting until" in r.message]
        cancel = [r for r in caplog.records if "to cancel and exit" in r.message]
        assert waiting and waiting[0].levelno == logging.WARNING
        assert cancel and cancel[0].levelno == logging.WARNING


# ============================================================================
# Fix 5: the wait loop does not claim an unobserved reset
# ============================================================================
class _FakeTracker:
    """Minimal DailyTokenTracker stand-in for the wait loop."""

    _shared_enabled = False

    def __init__(self, still_limited: bool) -> None:
        self._still_limited = still_limited

    def is_limit_reached(self) -> bool:
        return self._still_limited

    def set_daily_limit(self, value: int) -> None:
        pass

    def set_pool_settings(self, **kwargs: Any) -> None:
        pass


class TestWaitDeadlineHonesty:
    def _run(self, monkeypatch: pytest.MonkeyPatch, still_limited: bool) -> bool:
        import cli.loop as loop_module

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(loop_module, "_user_requested_cancel", lambda: False)
        monkeypatch.setattr("cli.loop.time.sleep", lambda _s: None)
        monkeypatch.setattr(app_config, "reload_daily_token_limit", lambda: None)
        monkeypatch.setattr(app_config, "reload_pool_settings", lambda: None)

        tracker = _FakeTracker(still_limited)
        return loop_module._wait_for_token_reset(tracker, 2)  # type: ignore[arg-type]

    def test_deadline_with_limit_still_active_says_so(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression: the loop asserted "Token limit has been reset" blindly."""
        import logging

        import cli.loop as loop_module

        with caplog.at_level(logging.INFO, logger=loop_module.logger.name):
            loop_module.logger.propagate = True
            try:
                assert self._run(monkeypatch, still_limited=True) is True
            finally:
                loop_module.logger.propagate = False

        messages = [r.message for r in caplog.records]
        assert not any("Token limit has been reset" in m for m in messages)
        assert any("may still be in effect" in m for m in messages)

    def test_deadline_after_real_reset_keeps_the_reset_message(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        import cli.loop as loop_module

        with caplog.at_level(logging.INFO, logger=loop_module.logger.name):
            loop_module.logger.propagate = True
            try:
                assert self._run(monkeypatch, still_limited=False) is True
            finally:
                loop_module.logger.propagate = False

        assert any("Token limit has been reset" in r.message for r in caplog.records)


# ============================================================================
# Fix 6: rmtree uses onexc, not the deprecated onerror
# ============================================================================
class TestCleanupWorkingDirectory:
    def test_uses_onexc_callback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``onerror=`` has been deprecated since Python 3.12."""
        import cli.loop as loop_module

        captured: dict[str, Any] = {}

        def fake_rmtree(path: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr("cli.loop.shutil.rmtree", fake_rmtree)
        loop_module._cleanup_working_directory(Path("/tmp/wd"))

        assert "onexc" in captured
        assert "onerror" not in captured

    def test_callback_chmods_and_retries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The onexc callback still clears the read-only bit and retries."""
        import cli.loop as loop_module

        target = tmp_path / "locked.txt"
        target.write_text("x", encoding="utf-8")
        captured: dict[str, Any] = {}

        def fake_rmtree(path: Any, **kwargs: Any) -> None:
            captured["onexc"] = kwargs["onexc"]

        monkeypatch.setattr("cli.loop.shutil.rmtree", fake_rmtree)
        loop_module._cleanup_working_directory(tmp_path)

        removed: list[Any] = []
        # Invoked with the (func, path, exception) triple shutil.rmtree passes.
        captured["onexc"](removed.append, target, PermissionError("denied"))

        assert removed == [target]
