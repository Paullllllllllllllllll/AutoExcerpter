"""Logging configuration and setup utilities.

This module provides a standardized logger setup for the application with
separation between user-facing messages and detailed technical logs.
All modules should use setup_logger(__name__) to get a properly configured logger.
"""

from __future__ import annotations

import logging
import sys

# Public API
__all__ = [
    "setup_logger",
    "set_log_level",
]

# ============================================================================
# Constants
# ============================================================================
DEFAULT_LOG_LEVEL = logging.INFO
USER_LOG_LEVEL = logging.WARNING  # Only show warnings and errors to users by default
SIMPLE_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# Logger Setup Functions
# ============================================================================
def setup_logger(
    name: str,
    level: int = DEFAULT_LOG_LEVEL,
    date_format: str | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    Set up a standardized logger for the application.

    This function creates or retrieves a logger with the specified name and
    configures it with a StreamHandler if it doesn't already have handlers.

    Args:
        name: Name for the logger (typically __name__ from the calling module)
        level: Logging level for file/detailed logging (default: INFO)
        date_format: Custom date format for timestamps
        verbose: If True, show all logs on console; if False, only warnings/errors

    Returns:
        Configured Logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Discarded in production: no file handler exists")
        >>> logger.warning("This shows on the console (stderr)")

    Note:
        - Only adds a handler if the logger doesn't already have one
        - Uses StreamHandler to output to stderr
        - No file handler is installed, so records below the console level
          (WARNING unless *verbose*) are discarded rather than archived
        - All loggers share the same format by design for consistency
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        # Console handler - only shows warnings and errors by default
        console_handler = logging.StreamHandler(sys.stderr)
        console_level = level if verbose else USER_LOG_LEVEL
        console_handler.setLevel(console_level)

        # Use simple format for console, detailed for files
        console_formatter = logging.Formatter(
            fmt=SIMPLE_FORMAT,
            datefmt=date_format or DEFAULT_DATE_FORMAT,
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)
        logger.setLevel(level)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger


# ============================================================================
# Logger Configuration Functions
# ============================================================================
def set_log_level(logger: logging.Logger, level: int) -> None:
    """
    Change the log level of an existing logger.

    Args:
        logger: The logger instance to modify
        level: New logging level (e.g., logging.DEBUG, logging.INFO)

    Example:
        >>> logger = setup_logger(__name__)
        >>> set_log_level(logger, logging.DEBUG)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
