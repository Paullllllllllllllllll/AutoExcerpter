"""Round-4 CLI/main regression tests.

One test group per fix in the round-4 hardening pass:

1.  A registered exit hook fires on exit_program, guards hook exceptions, and
    is a one-shot no-op when unregistered; the non-TTY --json guard emits JSON.
2.  concurrency_limit is coerced to int, with a safe fallback to the default.
3.  A string max_output_tokens does not crash the interactive summary display.
4.  --dry-run is side-effect-free (creates no output tree).
5.  _safe_print survives a None stdout encoding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from config import app as app_config
from pipeline.types import ItemSpec


@pytest.fixture(autouse=True)
def _clear_exit_hook() -> Any:
    """Keep the module-level exit hook from leaking across tests."""
    import cli.interaction as inter

    inter.set_exit_hook(None)
    yield
    inter.set_exit_hook(None)


# ============================================================================
# Fix 1: exit hook + non-TTY --json emission
# ============================================================================
class TestExitHook:
    def test_registered_hook_fires_on_exit(self) -> None:
        import cli.interaction as inter

        calls: list[str] = []
        inter.set_exit_hook(lambda: calls.append("fired"))
        with pytest.raises(SystemExit) as exc:
            inter.exit_program("bye", exit_code=3)
        assert exc.value.code == 3
        assert calls == ["fired"]

    def test_hook_exception_is_suppressed(self) -> None:
        import cli.interaction as inter

        def boom() -> None:
            raise RuntimeError("nope")

        inter.set_exit_hook(boom)
        # The failing hook must not mask the SystemExit or leak RuntimeError.
        with pytest.raises(SystemExit) as exc:
            inter.exit_program(exit_code=0)
        assert exc.value.code == 0

    def test_unregistered_hook_is_noop(self) -> None:
        import cli.interaction as inter

        inter.set_exit_hook(None)
        with pytest.raises(SystemExit) as exc:
            inter.exit_program(exit_code=7)
        assert exc.value.code == 7

    def test_hook_is_one_shot(self) -> None:
        import cli.interaction as inter

        calls: list[int] = []
        inter.set_exit_hook(lambda: calls.append(1))
        with pytest.raises(SystemExit):
            inter.exit_program()
        # Cleared after firing so a later exit cannot double-emit.
        assert inter._exit_hook is None
        assert calls == [1]

    def test_non_tty_guard_emits_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import main as main_module

        monkeypatch.setattr(app_config, "CLI_MODE", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(sys, "argv", ["main.py", "--json"])

        rc = main_module.main()
        assert rc == 2
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["items_total"] == 0
        assert payload["items_complete"] == 0
        assert payload["items_failed"] == 0
        assert payload["items_skipped"] == 0
        assert payload["outputs"] == []

    def test_exit_program_via_hook_emits_json_once(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Typing exit at an interactive prompt still emits the JSON summary."""
        import cli.interaction as inter
        import main as main_module

        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        inter.set_exit_hook(
            lambda: main_module._emit_json_summary(0, 0, 0, 0, [])
        )

        with pytest.raises(SystemExit) as exc:
            inter.exit_program("Exiting.", exit_code=0)
        assert exc.value.code == 0
        lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["items_total"] == 0


# ============================================================================
# Fix 2: concurrency_limit int coercion
# ============================================================================
class TestConcurrencyLimitCoercion:
    def _loader(self, value: Any) -> MagicMock:
        loader = MagicMock()
        loader.get_concurrency_config.return_value = {
            "api_requests": {"transcription": {"concurrency_limit": value}}
        }
        return loader

    def test_string_value_coerced_to_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config.accessors as acc

        monkeypatch.setattr(acc, "get_config_loader", lambda: self._loader("80"))
        workers, _delay = acc.get_api_concurrency("transcription")
        assert workers == 80
        assert isinstance(workers, int)

    def test_garbage_value_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import config.accessors as acc
        from config.constants import DEFAULT_CONCURRENT_REQUESTS

        monkeypatch.setattr(acc, "get_config_loader", lambda: self._loader("abc"))
        workers, _delay = acc.get_api_concurrency("transcription")
        assert workers == DEFAULT_CONCURRENT_REQUESTS


# ============================================================================
# Fix 3: string max_output_tokens does not crash the summary display
# ============================================================================
class TestSummaryTokenFormatting:
    def test_fmt_int_coerces_and_falls_back(self) -> None:
        from cli.display import _fmt_int

        assert _fmt_int(128000) == "128,000"
        assert _fmt_int("128000") == "128,000"
        assert _fmt_int("garbage") == "garbage"
        assert _fmt_int(None) == "None"

    def test_display_survives_string_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import cli.display as display

        monkeypatch.setattr(app_config, "SUMMARIZE", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)

        loader = MagicMock()
        loader.get_model_config.return_value = {
            "transcription_model": {
                "provider": "openai",
                "name": "gpt-5-mini",
                "max_output_tokens": "128000",  # hand-edited string
            }
        }
        loader.get_concurrency_config.return_value = {"api_requests": {}, "retry": {}}
        monkeypatch.setattr("config.loader.get_config_loader", lambda: loader)
        # The confirmation prompt would block on input; short-circuit it.
        monkeypatch.setattr(display, "prompt_yes_no", lambda *a, **k: True)

        result = display._display_processing_summary(
            [ItemSpec(kind="pdf", path=Path("/tmp/A.pdf"))], Path("/out"), None
        )
        assert result is True
        # Coerced and comma-formatted rather than raising ValueError.
        assert "128,000" in capsys.readouterr().out


# ============================================================================
# Fix 4: --dry-run is side-effect-free
# ============================================================================
class TestDryRunNoSideEffects:
    def test_dry_run_creates_no_output_tree(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        make_pdf: Any,
    ) -> None:
        import main as main_module

        pdf = make_pdf("Doc.pdf", num_pages=1)
        out_dir = tmp_path / "outtree" / "sub"  # does not exist yet

        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(app_config, "SUMMARIZE", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(
            sys, "argv", ["main.py", str(pdf), str(out_dir), "--dry-run"]
        )

        rc = main_module.main()
        assert rc == 0
        # The dry run must not have created the output directory tree.
        assert not out_dir.exists()


# ============================================================================
# Fix 5: _safe_print survives a None stdout encoding
# ============================================================================
class TestSafePrintNoneEncoding:
    def test_none_encoding_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.interaction as inter

        class FakeStdout:
            encoding = None

            def __init__(self) -> None:
                self.writes: list[str] = []
                self._first = True

            def write(self, s: str) -> int:
                if self._first and s.strip("\n"):
                    self._first = False
                    raise UnicodeEncodeError("utf-8", s, 0, 1, "boom")
                self.writes.append(s)
                return len(s)

            def flush(self) -> None:
                pass

        fake = FakeStdout()
        monkeypatch.setattr(sys, "stdout", fake)
        # Without the fix, the fallback path would call text.encode(None) and
        # raise TypeError; the encoding-or-utf-8 guard prevents that.
        inter._safe_print("héllo")
        assert any("h" in w for w in fake.writes)
