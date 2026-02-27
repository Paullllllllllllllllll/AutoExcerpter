"""Centralized error handling and reporting utilities.

This module provides consistent error handling patterns across the application,
including error classification, logging, and user-friendly error messages.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, TypeVar
from collections.abc import Callable

from modules.logger import setup_logger
from modules.user_prompts import print_error, print_warning

logger = setup_logger(__name__)

# Type variable for generic error handler
T = TypeVar("T")


# ============================================================================
# Error Classification
# ============================================================================
class ProcessingError(Exception):
    """Base exception for processing errors."""


class ConfigurationError(ProcessingError):
    """Exception for configuration-related errors."""


class APIError(ProcessingError):
    """Exception for API-related errors."""


class FileProcessingError(ProcessingError):
    """Exception for file processing errors."""


# ============================================================================
# Error Handlers
# ============================================================================
def handle_critical_error(
    error: Exception,
    context: str,
    exit_on_error: bool = False,
    show_user_message: bool = True,
) -> None:
    """Handle critical errors with consistent logging and user feedback."""
    error_msg = f"Critical error in {context}: {error}"
    logger.exception(error_msg)

    if show_user_message:
        print_error(f"Critical error: {context} failed. Check logs for details.")

    if exit_on_error:
        sys.exit(1)


def handle_recoverable_error(
    error: Exception,
    context: str,
    show_user_message: bool = True,
) -> None:
    """Handle recoverable errors with logging and optional user feedback."""
    error_msg = f"Recoverable error in {context}: {error}"
    logger.warning(error_msg)
    logger.debug(traceback.format_exc())

    if show_user_message:
        print_warning(
            f"Warning: {context} encountered an issue but processing continues."
        )


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: T | None = None,
    context: str = "operation",
    log_errors: bool = True,
    **kwargs: Any,
) -> T | None:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"Error during {context}: {e}")
            logger.debug(traceback.format_exc())
        return default


# ============================================================================
# Validation Helpers
# ============================================================================
def validate_file_exists(file_path: Any, context: str = "file") -> None:
    """Validate that a file exists, raising FileProcessingError if not."""
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileProcessingError(f"{context} not found: {path}")
    if not path.is_file():
        raise FileProcessingError(f"{context} is not a file: {path}")


def validate_directory_exists(dir_path: Any, context: str = "directory") -> None:
    """Validate that a directory exists, raising FileProcessingError if not."""
    from pathlib import Path

    path = Path(dir_path)
    if not path.exists():
        raise FileProcessingError(f"{context} not found: {path}")
    if not path.is_dir():
        raise FileProcessingError(f"{context} is not a directory: {path}")


def validate_config_value(
    value: Any,
    expected_type: type,
    name: str,
    allow_none: bool = False,
) -> None:
    """Validate a configuration value."""
    if value is None and allow_none:
        return

    if not isinstance(value, expected_type):
        raise ConfigurationError(
            f"Invalid configuration for '{name}': expected {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "ProcessingError",
    "ConfigurationError",
    "APIError",
    "FileProcessingError",
    "handle_critical_error",
    "handle_recoverable_error",
    "safe_execute",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_config_value",
]
