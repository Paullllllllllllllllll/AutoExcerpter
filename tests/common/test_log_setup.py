"""Tests for common/log_setup.py."""

from __future__ import annotations

import io
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from autoexcerpter.common.log_setup import configure_logging

_TEST_LOGGER = "log_setup_probe"


def _own_handlers(name: str = "") -> list[logging.Handler]:
    """Return the handlers that ``configure_logging`` installed on a logger."""
    return [
        handler
        for handler in logging.getLogger(name).handlers
        if getattr(handler, "_log_setup_handler", False)
    ]


def _remove_own_handlers(name: str) -> None:
    target = logging.getLogger(name)
    for handler in _own_handlers(name):
        target.removeHandler(handler)
        handler.close()


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    probe = logging.getLogger(_TEST_LOGGER)
    levels = (root.level, probe.level)
    yield
    _remove_own_handlers("")
    _remove_own_handlers(_TEST_LOGGER)
    root.setLevel(levels[0])
    probe.setLevel(levels[1])


class TestConfigureLogging:
    def test_is_idempotent(self) -> None:
        configure_logging()
        configure_logging()
        handlers = _own_handlers()
        assert len(handlers) == 1
        assert handlers[0].level == logging.WARNING

    def test_writes_to_current_stderr_after_swap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure_logging(logger_name=_TEST_LOGGER)
        swapped = io.StringIO()
        monkeypatch.setattr(sys, "stderr", swapped)

        logging.getLogger(f"{_TEST_LOGGER}.pipeline").warning("after swap")

        assert swapped.getvalue() == "[WARNING] after swap\n"

    def test_info_hidden_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        configure_logging(logger_name=_TEST_LOGGER)
        swapped = io.StringIO()
        monkeypatch.setattr(sys, "stderr", swapped)

        log = logging.getLogger(f"{_TEST_LOGGER}.cli")
        log.info("quiet")
        log.error("loud")

        assert swapped.getvalue() == "[ERROR] loud\n"

    def test_handlers_go_on_the_top_level_logger(self) -> None:
        child = logging.getLogger(f"{_TEST_LOGGER}.rendering.docx")

        configure_logging(logger_name=f"{_TEST_LOGGER}.rendering")

        assert len(_own_handlers(_TEST_LOGGER)) == 1
        assert _own_handlers() == []
        assert child.handlers == []
        assert child.level == logging.NOTSET
        assert child.propagate is True

    def test_level_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        configure_logging(logger_name=_TEST_LOGGER, level=logging.INFO)
        swapped = io.StringIO()
        monkeypatch.setattr(sys, "stderr", swapped)
        logging.getLogger(_TEST_LOGGER).info("shown")
        assert swapped.getvalue() == "[INFO] shown\n"

    def test_file_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        log_file = tmp_path / "logs" / "run.log"

        configure_logging(logger_name=_TEST_LOGGER, log_file=log_file)
        logging.getLogger(f"{_TEST_LOGGER}.core").info("into the file")
        configure_logging(logger_name=_TEST_LOGGER)

        text = log_file.read_text(encoding="utf-8")
        assert "INFO" in text
        assert f"{_TEST_LOGGER}.core - into the file" in text
        assert len(_own_handlers(_TEST_LOGGER)) == 1


_IMPORT_PROBE = """
import logging
for name in ("log_setup", "rate_limit", "responses", "retry", "usage"):
    __import__(f"autoexcerpter.common.{name}")
loggers = [logging.getLogger()] + [
    entry
    for entry in logging.Logger.manager.loggerDict.values()
    if isinstance(entry, logging.Logger)
]
print(sum(len(entry.handlers) for entry in loggers))
"""


def test_importing_common_modules_configures_nothing(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        check=True,
    )
    assert result.stdout.strip() == "0"
