"""Centralized error handling and reporting utilities.

This module provides consistent error handling patterns across the application,
including error classification, logging, and user-friendly error messages.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Callable, Optional, TypeVar

from modules.logger import setup_logger
from modules.user_prompts import print_error, print_warning

logger = setup_logger(__name__)

# Type variable for generic error handler
T = TypeVar('T')

# ============================================================================
# Error Classification
# ============================================================================
class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class ConfigurationError(ProcessingError):
    """Exception for configuration-related errors."""
    pass


class APIError(ProcessingError):
    """Exception for API-related errors."""
    pass


class FileProcessingError(ProcessingError):
    """Exception for file processing errors."""
    pass


# ============================================================================
# Error Handlers
# ============================================================================
def handle_critical_error(
    error: Exception,
    context: str,
    exit_on_error: bool = False,
    show_user_message: bool = True,
) -> None:
    """
    Handle critical errors with consistent logging and user feedback.
    
    Args:
        error: The exception that occurred.
        context: Description of what was being done when error occurred.
        exit_on_error: Whether to exit the program after handling.
        show_user_message: Whether to show user-friendly error message.
    """
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
    """
    Handle recoverable errors with logging and optional user feedback.
    
    Args:
        error: The exception that occurred.
        context: Description of what was being done when error occurred.
        show_user_message: Whether to show user-friendly warning message.
    """
    error_msg = f"Recoverable error in {context}: {error}"
    logger.warning(error_msg)
    logger.debug(traceback.format_exc())
    
    if show_user_message:
        print_warning(f"Warning: {context} encountered an issue but processing continues.")


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    context: str = "operation",
    log_errors: bool = True,
    **kwargs: Any,
) -> Optional[T]:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute.
        *args: Positional arguments for the function.
        default: Default value to return on error.
        context: Description of the operation for logging.
        log_errors: Whether to log errors.
        **kwargs: Keyword arguments for the function.
    
    Returns:
        Function result or default value on error.
    """
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
    """
    Validate that a file exists, raising FileProcessingError if not.
    
    Args:
        file_path: Path to validate.
        context: Description for error message.
    
    Raises:
        FileProcessingError: If file doesn't exist.
    """
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        raise FileProcessingError(f"{context} not found: {path}")
    if not path.is_file():
        raise FileProcessingError(f"{context} is not a file: {path}")


def validate_directory_exists(dir_path: Any, context: str = "directory") -> None:
    """
    Validate that a directory exists, raising FileProcessingError if not.
    
    Args:
        dir_path: Path to validate.
        context: Description for error message.
    
    Raises:
        FileProcessingError: If directory doesn't exist.
    """
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
    """
    Validate a configuration value.
    
    Args:
        value: Value to validate.
        expected_type: Expected type of the value.
        name: Name of the configuration parameter.
        allow_none: Whether None is an acceptable value.
    
    Raises:
        ConfigurationError: If validation fails.
    """
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
