"""Logging configuration and setup utilities.

This module provides a standardized logger setup for the application.
All modules should use setup_logger(__name__) to get a properly configured logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# Public API
__all__ = ["setup_logger", "set_log_level"]

# Constants
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = DEFAULT_LOG_LEVEL,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a standardized logger for the application.

    This function creates or retrieves a logger with the specified name and
    configures it with a StreamHandler if it doesn't already have handlers.

    Args:
        name: Name for the logger (typically __name__ from the calling module)
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        date_format: Custom date format for timestamps

    Returns:
        Configured Logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started")

    Note:
        - Only adds a handler if the logger doesn't already have one
        - Uses StreamHandler to output to stderr
        - All loggers share the same format by design for consistency
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        
        formatter = logging.Formatter(
            fmt=format_string or DEFAULT_FORMAT,
            datefmt=date_format or DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(level)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger


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
