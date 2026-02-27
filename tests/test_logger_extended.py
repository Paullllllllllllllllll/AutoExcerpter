"""Extended tests for modules/logger.py - Logging configuration and setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from modules.logger import (
    setup_logger,
    setup_console_handler,
    setup_file_handler,
    set_log_level,
    DEFAULT_LOG_LEVEL,
    USER_LOG_LEVEL,
    DETAILED_FORMAT,
    SIMPLE_FORMAT,
)


@pytest.fixture(autouse=True)
def _cleanup_test_loggers():
    """Remove test loggers after each test to avoid handler accumulation."""
    yield
    # Clean up any loggers created during tests
    logger_manager = logging.Logger.manager
    loggers_to_remove = [
        name for name in logger_manager.loggerDict if name.startswith("test_logger_")
    ]
    for name in loggers_to_remove:
        log = logging.getLogger(name)
        log.handlers.clear()


# ============================================================================
# setup_logger
# ============================================================================
class TestSetupLogger:
    """Tests for setup_logger()."""

    def test_creates_logger_with_handler(self):
        """setup_logger returns a logger with at least one handler."""
        logger = setup_logger("test_logger_basic")

        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) >= 1

    def test_handler_is_stream_handler(self):
        """The default handler is a StreamHandler to stderr."""
        logger = setup_logger("test_logger_stream")

        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream == sys.stderr

    def test_does_not_duplicate_handlers(self):
        """Calling setup_logger twice with same name does not add extra handlers."""
        logger1 = setup_logger("test_logger_dedup")
        handler_count_first = len(logger1.handlers)

        logger2 = setup_logger("test_logger_dedup")
        handler_count_second = len(logger2.handlers)

        assert logger1 is logger2
        assert handler_count_first == handler_count_second

    def test_default_log_level(self):
        """Logger is set to DEFAULT_LOG_LEVEL by default."""
        logger = setup_logger("test_logger_level")

        assert logger.level == DEFAULT_LOG_LEVEL

    def test_custom_log_level(self):
        """Logger respects a custom log level."""
        logger = setup_logger("test_logger_custom_level", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_verbose_mode_sets_console_to_full_level(self):
        """Verbose mode sets console handler to the logger's level."""
        logger = setup_logger("test_logger_verbose", level=logging.DEBUG, verbose=True)

        handler = logger.handlers[0]
        assert handler.level == logging.DEBUG

    def test_non_verbose_mode_sets_console_to_warning(self):
        """Non-verbose mode restricts console handler to USER_LOG_LEVEL."""
        logger = setup_logger(
            "test_logger_nonverbose", level=logging.DEBUG, verbose=False
        )

        handler = logger.handlers[0]
        assert handler.level == USER_LOG_LEVEL

    def test_propagation_disabled(self):
        """Logger propagation is disabled to avoid duplicate logs."""
        logger = setup_logger("test_logger_propagation")

        assert logger.propagate is False

    def test_uses_simple_format(self):
        """Default console handler uses SIMPLE_FORMAT."""
        logger = setup_logger("test_logger_format")

        handler = logger.handlers[0]
        assert handler.formatter._fmt == SIMPLE_FORMAT


# ============================================================================
# setup_console_handler
# ============================================================================
class TestSetupConsoleHandler:
    """Tests for setup_console_handler()."""

    def test_adds_handler(self):
        """Adds a StreamHandler to the logger."""
        logger = logging.getLogger("test_logger_console_add")
        logger.handlers.clear()

        setup_console_handler(logger)

        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1

    def test_removes_existing_stderr_handlers(self):
        """Existing stderr StreamHandlers are removed before adding new one."""
        logger = logging.getLogger("test_logger_console_replace")
        logger.handlers.clear()

        # Add two stderr handlers manually
        h1 = logging.StreamHandler(sys.stderr)
        h2 = logging.StreamHandler(sys.stderr)
        logger.addHandler(h1)
        logger.addHandler(h2)
        assert len(logger.handlers) == 2

        setup_console_handler(logger)

        # Only the new one should remain
        stderr_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
        ]
        assert len(stderr_handlers) == 1

    def test_simple_format_true(self):
        """When simple_format is True, SIMPLE_FORMAT is used."""
        logger = logging.getLogger("test_logger_console_simple")
        logger.handlers.clear()

        setup_console_handler(logger, simple_format=True)

        handler = logger.handlers[0]
        assert handler.formatter._fmt == SIMPLE_FORMAT

    def test_simple_format_false_uses_detailed(self):
        """When simple_format is False, DETAILED_FORMAT is used."""
        logger = logging.getLogger("test_logger_console_detailed")
        logger.handlers.clear()

        setup_console_handler(logger, simple_format=False)

        handler = logger.handlers[0]
        assert handler.formatter._fmt == DETAILED_FORMAT

    def test_custom_level(self):
        """Console handler respects a custom log level."""
        logger = logging.getLogger("test_logger_console_level")
        logger.handlers.clear()

        setup_console_handler(logger, level=logging.ERROR)

        handler = logger.handlers[0]
        assert handler.level == logging.ERROR

    def test_preserves_non_stderr_handlers(self):
        """Non-stderr handlers are not removed."""
        logger = logging.getLogger("test_logger_console_preserve")
        logger.handlers.clear()

        # Add a file-like handler (stdout instead of stderr)
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        setup_console_handler(logger)

        # stdout handler should still be there
        assert stdout_handler in logger.handlers
        assert len(logger.handlers) == 2  # stdout + new stderr


# ============================================================================
# setup_file_handler
# ============================================================================
class TestSetupFileHandler:
    """Tests for setup_file_handler()."""

    def test_creates_file_handler(self, tmp_path: Path):
        """Adds a FileHandler to the logger."""
        logger = logging.getLogger("test_logger_file_create")
        logger.handlers.clear()

        log_file = tmp_path / "test.log"
        setup_file_handler(logger, str(log_file))

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

    def test_writes_to_file(self, tmp_path: Path):
        """Logger writes messages to the log file."""
        logger = logging.getLogger("test_logger_file_write")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)

        log_file = tmp_path / "output.log"
        setup_file_handler(logger, str(log_file), level=logging.DEBUG)

        logger.debug("test message for file")

        # Flush handler
        for handler in logger.handlers:
            handler.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "test message for file" in content

    def test_default_level_is_debug(self, tmp_path: Path):
        """Default file handler level is DEBUG."""
        logger = logging.getLogger("test_logger_file_level")
        logger.handlers.clear()

        log_file = tmp_path / "debug.log"
        setup_file_handler(logger, str(log_file))

        file_handler = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ][0]
        assert file_handler.level == logging.DEBUG

    def test_custom_level(self, tmp_path: Path):
        """File handler respects a custom log level."""
        logger = logging.getLogger("test_logger_file_custom_level")
        logger.handlers.clear()

        log_file = tmp_path / "custom.log"
        setup_file_handler(logger, str(log_file), level=logging.WARNING)

        file_handler = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ][0]
        assert file_handler.level == logging.WARNING

    def test_uses_detailed_format(self, tmp_path: Path):
        """File handler uses DETAILED_FORMAT."""
        logger = logging.getLogger("test_logger_file_format")
        logger.handlers.clear()

        log_file = tmp_path / "fmt.log"
        setup_file_handler(logger, str(log_file))

        file_handler = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ][0]
        assert file_handler.formatter._fmt == DETAILED_FORMAT

    def test_append_mode(self, tmp_path: Path):
        """File handler opens in append mode."""
        logger = logging.getLogger("test_logger_file_append")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)

        log_file = tmp_path / "append.log"
        log_file.write_text("existing content\n", encoding="utf-8")

        setup_file_handler(logger, str(log_file), level=logging.DEBUG)
        logger.debug("new line")

        for handler in logger.handlers:
            handler.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "existing content" in content
        assert "new line" in content


# ============================================================================
# set_log_level
# ============================================================================
class TestSetLogLevel:
    """Tests for set_log_level()."""

    def test_changes_logger_level(self):
        """Logger level is updated."""
        logger = setup_logger("test_logger_set_level")
        original_level = logger.level

        set_log_level(logger, logging.CRITICAL)

        assert logger.level == logging.CRITICAL
        assert logger.level != original_level

    def test_changes_handler_levels(self):
        """All handler levels are updated."""
        logger = setup_logger("test_logger_set_handler_level")

        set_log_level(logger, logging.ERROR)

        for handler in logger.handlers:
            assert handler.level == logging.ERROR

    def test_debug_level(self):
        """Setting DEBUG level propagates to logger and handlers."""
        logger = setup_logger("test_logger_debug_level")

        set_log_level(logger, logging.DEBUG)

        assert logger.level == logging.DEBUG
        for handler in logger.handlers:
            assert handler.level == logging.DEBUG

    def test_multiple_handlers_all_updated(self, tmp_path: Path):
        """All handlers (console and file) are updated."""
        logger = logging.getLogger("test_logger_multi_handler")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)

        # Add console and file handlers
        setup_console_handler(logger, level=logging.WARNING)
        log_file = tmp_path / "multi.log"
        setup_file_handler(logger, str(log_file), level=logging.DEBUG)

        assert len(logger.handlers) == 2

        set_log_level(logger, logging.CRITICAL)

        assert logger.level == logging.CRITICAL
        for handler in logger.handlers:
            assert handler.level == logging.CRITICAL
