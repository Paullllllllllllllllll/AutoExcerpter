"""Round 5: configuration and CLI-path maintenance regressions.

One focused regression per verified fix:

1. ``~`` and drive-relative input/output paths are expanded and absolutized in
   both execution branches (``Path.cwd() / p`` turned ``~/pdfs`` into a literal
   ``~`` directory under the CWD and left ``D:books`` drive-relative).
2. A blank interactive input or output path raises instead of silently
   scanning (and writing into) the whole current working directory, which is
   what the shipped ``app.example.yaml`` (``input_folder_path: ''``) produced.
3. The config accessors repair out-of-range values visibly: a non-positive
   ``concurrency_limit`` / ``api_timeout`` / ``target_dpi`` is clamped with a
   once-per-process WARNING, an absent key stays silent, and an all-invalid
   ``rate_limits`` list says so instead of quietly restoring the most
   permissive built-in throttle.
4. The pre-run overview survives a partially-nulled ``model.yaml`` (empty
   section bodies, ``provider: null``, ``reasoning: null``) instead of raising
   AttributeError on ``None.get`` / ``None.upper``.
5. ``run_repair --limit 0`` is rejected at parse time rather than silently
   repairing nothing while still writing reports and exiting 0.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import cli.display as display
import config.accessors as accessors
from cli.args import _parse_execution_mode
from config import app as app_config
from config.constants import (
    DEFAULT_CONCURRENT_REQUESTS,
    DEFAULT_OPENAI_TIMEOUT,
    DEFAULT_RATE_LIMITS,
    DEFAULT_TARGET_DPI,
)
from pipeline.types import ItemSpec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _make_args(**kwargs: Any) -> argparse.Namespace:
    """Build a Namespace with the fields ``_parse_execution_mode`` reads."""
    defaults: dict[str, Any] = {
        "input": "/tmp/in",
        "output": "/tmp/out",
        "input_path": None,
        "output_path": None,
        "all": False,
        "select": None,
        "context": None,
        "force": None,
        "resume": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _propagate(monkeypatch: pytest.MonkeyPatch, logger: logging.Logger) -> None:
    """Route *logger*'s records to caplog exactly once.

    ``setup_logger`` sets ``propagate = False``, so pytest's logging plugin
    attaches its capture handlers to this logger as well as to the root. Left
    alone, enabling propagation would hand each record to the same capture
    handler twice and defeat the warn-once record count, so the local handlers
    are dropped and capture happens at the root only.
    """
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)


def _loader_with(
    concurrency: dict[str, Any] | None = None,
    image_processing: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
) -> MagicMock:
    """Return a mock ConfigLoader serving the given config sections."""
    loader = MagicMock()
    loader.get_concurrency_config.return_value = concurrency or {}
    loader.get_image_processing_config.return_value = image_processing or {}
    loader.get_model_config.return_value = model or {}
    return loader


@pytest.fixture(autouse=True)
def _reset_warn_once() -> Any:
    """Clear the accessors' warn-once ledger around every test."""
    accessors._warned.clear()
    yield
    accessors._warned.clear()


# ---------------------------------------------------------------------------
# Fix 1: "~" and drive-relative paths are expanded, not concatenated
# ---------------------------------------------------------------------------
class TestPathExpansion:
    """Both branches must expand "~" and absolutize drive-relative paths."""

    @staticmethod
    def _fake_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
        def _expand(path: str) -> str:
            if path.startswith("~"):
                return str(home) + path[1:]
            return path

        monkeypatch.setattr(os.path, "expanduser", _expand)

    def test_cli_branch_expands_tilde(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._fake_home(monkeypatch, tmp_path)
        monkeypatch.setattr(app_config, "CLI_MODE", True)

        args = _make_args(input="~/pdfs", output="~/out")
        input_path, output_path, *_ = _parse_execution_mode(args)

        assert input_path == tmp_path / "pdfs"
        assert output_path == tmp_path / "out"
        assert "~" not in str(input_path)

    def test_interactive_branch_expands_tilde(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._fake_home(monkeypatch, tmp_path)
        monkeypatch.setattr(app_config, "CLI_MODE", False)

        args = _make_args(input="~/pdfs", output="~/out")
        input_path, output_path, *_ = _parse_execution_mode(args)

        assert input_path == tmp_path / "pdfs"
        assert output_path == tmp_path / "out"

    @pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are Windows-only")
    @pytest.mark.parametrize("cli_mode", [True, False])
    def test_drive_relative_path_becomes_absolute(
        self, monkeypatch: pytest.MonkeyPatch, cli_mode: bool
    ) -> None:
        """ "Q:books" stayed drive-relative under the old ``Path.cwd() / p``."""
        monkeypatch.setattr(app_config, "CLI_MODE", cli_mode)

        args = _make_args(input="Q:books", output="Q:out")
        input_path, output_path, *_ = _parse_execution_mode(args)

        assert input_path.is_absolute()
        assert output_path.is_absolute()
        assert input_path.drive.upper() == "Q:"

    @pytest.mark.parametrize("cli_mode", [True, False])
    def test_relative_path_still_resolves_against_cwd(
        self, monkeypatch: pytest.MonkeyPatch, cli_mode: bool
    ) -> None:
        """Control: an ordinary relative path keeps its old meaning."""
        monkeypatch.setattr(app_config, "CLI_MODE", cli_mode)

        args = _make_args(input="in", output="out")
        input_path, output_path, *_ = _parse_execution_mode(args)

        assert input_path == Path.cwd() / "in"
        assert output_path == Path.cwd() / "out"


# ---------------------------------------------------------------------------
# Fix 2: a blank interactive path is an error, not a CWD-wide scan
# ---------------------------------------------------------------------------
class TestBlankInteractivePaths:
    """``Path("")`` is ``Path(".")``; the shipped example config ships blanks."""

    def test_blank_input_raises_with_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_config, "CLI_MODE", False)

        with pytest.raises(ValueError, match="input_folder_path"):
            _parse_execution_mode(_make_args(input="", output="/tmp/out"))

    def test_whitespace_input_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app_config, "CLI_MODE", False)

        with pytest.raises(ValueError, match="--input"):
            _parse_execution_mode(_make_args(input="   ", output="/tmp/out"))

    def test_blank_output_raises_with_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_config, "CLI_MODE", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)

        with pytest.raises(ValueError, match="output_folder_path"):
            _parse_execution_mode(_make_args(input="/tmp/in", output=""))

    def test_blank_output_falls_back_beside_inputs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With input_paths_is_output_path, a blank output is the documented
        "write next to each input" configuration, not an error; the input
        folder becomes the nominal base directory."""
        monkeypatch.setattr(app_config, "CLI_MODE", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", True)

        args = _make_args(input=str(tmp_path / "in"), output="")
        input_path, output_path, *_ = _parse_execution_mode(args)

        assert input_path == tmp_path / "in"
        assert output_path == tmp_path / "in"

    def test_populated_paths_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Control: a configured interactive path still parses."""
        monkeypatch.setattr(app_config, "CLI_MODE", False)

        args = _make_args(input=str(tmp_path / "in"), output=str(tmp_path / "out"))
        input_path, output_path, *_ = _parse_execution_mode(args)

        assert input_path == tmp_path / "in"
        assert output_path == tmp_path / "out"


# ---------------------------------------------------------------------------
# Fix 3: silent config fallbacks become visible, once per process
# ---------------------------------------------------------------------------
class TestConcurrencyLimitClamped:
    """A non-positive worker count is clamped where the run log can see it."""

    @staticmethod
    def _cfg(value: Any) -> dict[str, Any]:
        return {"api_requests": {"transcription": {"concurrency_limit": value}}}

    def test_zero_clamped_to_one_and_warns_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            accessors, "get_config_loader", lambda: _loader_with(self._cfg(0))
        )
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            first = accessors.get_api_concurrency("transcription")
            second = accessors.get_api_concurrency("transcription")

        assert first == 1
        assert second == 1
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "concurrency_limit" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert (
            "api_requests.transcription.concurrency_limit" in warnings[0].getMessage()
        )

    def test_negative_clamped_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            accessors, "get_config_loader", lambda: _loader_with(self._cfg(-4))
        )

        assert accessors.get_api_concurrency("transcription") == 1

    def test_absent_key_uses_default_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unset key is the normal case and must stay quiet."""
        cfg: dict[str, Any] = {"api_requests": {"transcription": {}}}
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with(cfg))
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            workers = accessors.get_api_concurrency("transcription")

        assert workers == DEFAULT_CONCURRENT_REQUESTS
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_valid_value_is_silent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            accessors, "get_config_loader", lambda: _loader_with(self._cfg(12))
        )
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            workers = accessors.get_api_concurrency("transcription")

        assert workers == 12
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_garbage_value_warns_and_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            accessors, "get_config_loader", lambda: _loader_with(self._cfg("abc"))
        )
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            workers = accessors.get_api_concurrency("transcription")

        assert workers == DEFAULT_CONCURRENT_REQUESTS
        assert any(
            "is not a number" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )


class TestApiTimeoutClamped:
    """``llm/base.py`` hands the timeout to the SDK verbatim."""

    def test_zero_clamped_to_one_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = {"api_requests": {"api_timeout": 0}}
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with(cfg))
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            timeout = accessors.get_api_timeout()

        assert timeout == 1
        assert any(
            "api_requests.api_timeout" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_absent_timeout_uses_default_silently(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with({}))
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            timeout = accessors.get_api_timeout()

        assert timeout == DEFAULT_OPENAI_TIMEOUT
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


class TestTargetDpiClamped:
    """A zero DPI would render an unusable page image."""

    def test_zero_clamped_to_one_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        img_cfg = {"api_image_processing": {"target_dpi": 0}}
        monkeypatch.setattr(
            accessors,
            "get_config_loader",
            lambda: _loader_with(image_processing=img_cfg),
        )
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            dpi = accessors.get_target_dpi()

        assert dpi == 1
        assert any(
            "api_image_processing.target_dpi" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_absent_dpi_uses_default_silently(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with())
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            dpi = accessors.get_target_dpi()

        assert dpi == DEFAULT_TARGET_DPI
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


class TestRateLimitFallbackIsAnnounced:
    """The built-in limits are the most permissive setting the tool has."""

    def test_all_invalid_entries_warn_and_fall_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = {"api_requests": {"rate_limits": [[0, 1], [10, -1], "junk"]}}
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with(cfg))
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            limits = accessors.get_rate_limits()

        assert limits == list(DEFAULT_RATE_LIMITS)
        messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("No usable api_requests.rate_limits" in m for m in messages)
        assert any("non-positive rate_limits entry" in m for m in messages)

    def test_empty_list_falls_back_without_the_notice(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An explicitly empty list is a deliberate "no override", not a typo."""
        cfg: dict[str, Any] = {"api_requests": {"rate_limits": []}}
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with(cfg))
        _propagate(monkeypatch, accessors.logger)

        with caplog.at_level(logging.WARNING, logger=accessors.logger.name):
            limits = accessors.get_rate_limits()

        assert limits == list(DEFAULT_RATE_LIMITS)
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_partially_valid_list_keeps_the_good_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = {"api_requests": {"rate_limits": [[0, 1], [60, 60]]}}
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with(cfg))

        assert accessors.get_rate_limits() == [(60, 60)]


# ---------------------------------------------------------------------------
# Fix 4: the pre-run overview survives a partially-nulled model.yaml
# ---------------------------------------------------------------------------
_NULLED_SECTIONS: dict[str, Any] = {
    "transcription_model": None,
    "summary_model": None,
}
_NULLED_PROVIDER: dict[str, Any] = {
    "transcription_model": {"provider": None, "name": "gpt-5-mini"},
    "summary_model": {"provider": None, "name": "gpt-5-mini"},
}
_NULLED_SUBSECTIONS: dict[str, Any] = {
    "transcription_model": {"name": "gpt-5-mini", "reasoning": None, "text": None},
    "summary_model": {"name": "gpt-5-mini", "reasoning": None, "text": None},
}


class TestProcessingSummaryTolerantOfNulls:
    """YAML turns an empty section body into ``None``, not ``{}``."""

    @pytest.mark.parametrize(
        "model_cfg",
        [_NULLED_SECTIONS, _NULLED_PROVIDER, _NULLED_SUBSECTIONS],
        ids=["empty-sections", "null-provider", "null-subsections"],
    )
    def test_summary_renders_without_attribute_error(
        self,
        model_cfg: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(app_config, "SUMMARIZE", True)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)

        loader = _loader_with(
            concurrency={"api_requests": {}, "retry": {}}, model=model_cfg
        )
        monkeypatch.setattr("config.loader.get_config_loader", lambda: loader)
        monkeypatch.setattr(display, "prompt_yes_no", lambda *a, **k: True)

        result = display._display_processing_summary(
            [ItemSpec(kind="pdf", path=Path("/tmp/A.pdf"))], Path("/out"), None
        )

        assert result is True
        out = capsys.readouterr().out
        assert "Transcription Provider: OPENAI" in out
        assert "Summary Provider: OPENAI" in out

    def test_as_dict_helper(self) -> None:
        assert display._as_dict({"a": 1}) == {"a": 1}
        assert display._as_dict(None) == {}
        assert display._as_dict("not a mapping") == {}


# ---------------------------------------------------------------------------
# Fix 5: run_repair --limit 0 is rejected at parse time
# ---------------------------------------------------------------------------
class TestRepairLimitValidation:
    """A zero limit sliced the target list empty but still reported success."""

    def test_positive_int_accepts_positive(self) -> None:
        from scripts.repair_layout.run_repair import _positive_int

        assert _positive_int("3") == 3

    @pytest.mark.parametrize("raw", ["0", "-1"])
    def test_positive_int_rejects_non_positive(self, raw: str) -> None:
        from scripts.repair_layout.run_repair import _positive_int

        with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
            _positive_int(raw)

    def test_positive_int_rejects_non_integer(self) -> None:
        from scripts.repair_layout.run_repair import _positive_int

        with pytest.raises(argparse.ArgumentTypeError, match="must be an integer"):
            _positive_int("2.5")

    def test_cli_rejects_zero_limit_before_writing_reports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scripts.repair_layout import run_repair

        monkeypatch.setattr(
            sys,
            "argv",
            ["run_repair", "--root", str(tmp_path), "--limit", "0"],
        )

        with pytest.raises(SystemExit) as exc:
            run_repair.main()

        assert exc.value.code == 2
        assert not (tmp_path / "backup").exists()
