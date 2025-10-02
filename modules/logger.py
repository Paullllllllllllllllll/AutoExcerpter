"""Logging configuration and setup utilities.

This module provides a standardized logger setup for the application with
separation between user-facing messages and detailed technical logs.
All modules should use setup_logger(__name__) to get a properly configured logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# Public API
__all__ = ["setup_logger", "set_log_level", "setup_console_handler", "setup_file_handler"]

# Constants
DEFAULT_LOG_LEVEL = logging.INFO
USER_LOG_LEVEL = logging.WARNING  # Only show warnings and errors to users by default
DETAILED_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
SIMPLE_FORMAT = "[%(levelname)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: int = DEFAULT_LOG_LEVEL,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    Set up a standardized logger for the application.

    This function creates or retrieves a logger with the specified name and
    configures it with a StreamHandler if it doesn't already have handlers.

    Args:
        name: Name for the logger (typically __name__ from the calling module)
        level: Logging level for file/detailed logging (default: INFO)
        format_string: Custom format string for log messages
        date_format: Custom date format for timestamps
        verbose: If True, show all logs on console; if False, only warnings/errors

    Returns:
        Configured Logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("This goes to detailed logs only")
        >>> logger.warning("This shows on console and in detailed logs")

    Note:
        - Only adds a handler if the logger doesn't already have one
        - Uses StreamHandler to output to stderr
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


def setup_console_handler(
    logger: logging.Logger,
    level: int = USER_LOG_LEVEL,
    simple_format: bool = True,
) -> None:
    """
    Add or update console handler for a logger.
    
    Args:
        logger: Logger instance to modify
        level: Logging level for console output
        simple_format: If True, use simple format; otherwise detailed format
    """
    # Remove existing StreamHandlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
            logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    
    format_str = SIMPLE_FORMAT if simple_format else DETAILED_FORMAT
    formatter = logging.Formatter(fmt=format_str, datefmt=DEFAULT_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)


def setup_file_handler(
    logger: logging.Logger,
    log_file_path: str,
    level: int = logging.DEBUG,
) -> None:
    """
    Add file handler to logger for detailed logging.
    
    Args:
        logger: Logger instance to modify
        log_file_path: Path to log file
        level: Logging level for file output (default: DEBUG for full details)
    """
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt=DETAILED_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


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
