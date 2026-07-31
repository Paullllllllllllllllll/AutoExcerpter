"""Extended tests for config/logger.py - Logging configuration and setup."""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

from config.logger import (
    DEFAULT_LOG_LEVEL,
    SIMPLE_FORMAT,
    USER_LOG_LEVEL,
    set_log_level,
    setup_logger,
)


@pytest.fixture(autouse=True)
def _cleanup_test_loggers() -> Generator[None]:
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

    def test_creates_logger_with_handler(self) -> None:
        """setup_logger returns a logger with at least one handler."""
        logger = setup_logger("test_logger_basic")

        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) >= 1

    def test_handler_is_stream_handler(self) -> None:
        """The default handler is a StreamHandler to stderr."""
        logger = setup_logger("test_logger_stream")

        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream == sys.stderr

    def test_does_not_duplicate_handlers(self) -> None:
        """Calling setup_logger twice with same name does not add extra handlers."""
        logger1 = setup_logger("test_logger_dedup")
        handler_count_first = len(logger1.handlers)

        logger2 = setup_logger("test_logger_dedup")
        handler_count_second = len(logger2.handlers)

        assert logger1 is logger2
        assert handler_count_first == handler_count_second

    def test_default_log_level(self) -> None:
        """Logger is set to DEFAULT_LOG_LEVEL by default."""
        logger = setup_logger("test_logger_level")

        assert logger.level == DEFAULT_LOG_LEVEL

    def test_custom_log_level(self) -> None:
        """Logger respects a custom log level."""
        logger = setup_logger("test_logger_custom_level", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_verbose_mode_sets_console_to_full_level(self) -> None:
        """Verbose mode sets console handler to the logger's level."""
        logger = setup_logger("test_logger_verbose", level=logging.DEBUG, verbose=True)

        handler = logger.handlers[0]
        assert handler.level == logging.DEBUG

    def test_non_verbose_mode_sets_console_to_warning(self) -> None:
        """Non-verbose mode restricts console handler to USER_LOG_LEVEL."""
        logger = setup_logger(
            "test_logger_nonverbose", level=logging.DEBUG, verbose=False
        )

        handler = logger.handlers[0]
        assert handler.level == USER_LOG_LEVEL

    def test_propagation_disabled(self) -> None:
        """Logger propagation is disabled to avoid duplicate logs."""
        logger = setup_logger("test_logger_propagation")

        assert logger.propagate is False

    def test_uses_simple_format(self) -> None:
        """Default console handler uses SIMPLE_FORMAT."""
        logger = setup_logger("test_logger_format")

        handler = logger.handlers[0]
        assert handler.formatter is not None
        assert handler.formatter._fmt == SIMPLE_FORMAT


# ============================================================================
# set_log_level
# ============================================================================
class TestSetLogLevel:
    """Tests for set_log_level()."""

    def test_changes_logger_level(self) -> None:
        """Logger level is updated."""
        logger = setup_logger("test_logger_set_level")
        original_level = logger.level

        set_log_level(logger, logging.CRITICAL)

        assert logger.level == logging.CRITICAL
        assert logger.level != original_level

    def test_changes_handler_levels(self) -> None:
        """All handler levels are updated."""
        logger = setup_logger("test_logger_set_handler_level")

        set_log_level(logger, logging.ERROR)

        for handler in logger.handlers:
            assert handler.level == logging.ERROR

    def test_debug_level(self) -> None:
        """Setting DEBUG level propagates to logger and handlers."""
        logger = setup_logger("test_logger_debug_level")

        set_log_level(logger, logging.DEBUG)

        assert logger.level == logging.DEBUG
        for handler in logger.handlers:
            assert handler.level == logging.DEBUG

    def test_multiple_handlers_all_updated(self, tmp_path: Path) -> None:
        """Every attached handler is updated, not just the first."""
        logger = logging.getLogger("test_logger_multi_handler")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(
            str(tmp_path / "multi.log"), mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        assert len(logger.handlers) == 2

        set_log_level(logger, logging.CRITICAL)

        assert logger.level == logging.CRITICAL
        for handler in logger.handlers:
            assert handler.level == logging.CRITICAL

        file_handler.close()
        logger.handlers.clear()


# ============================================================================
# Removed handler-management helpers
# ============================================================================
class TestRemovedHandlerHelpers:
    """setup_console_handler / setup_file_handler were dead code.

    Neither was called outside its own tests, and both carried latent defects
    (an unclosed FileHandler; a stale-stream identity comparison that appended
    duplicate console handlers instead of replacing them). They are gone.
    """

    def test_helpers_no_longer_exist(self) -> None:
        import config.logger as logger_module

        assert not hasattr(logger_module, "setup_console_handler")
        assert not hasattr(logger_module, "setup_file_handler")
        assert logger_module.__all__ == ["setup_logger", "set_log_level"]
