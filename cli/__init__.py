"""Command-line interface for AutoExcerpter.

Public interface:

- ``setup_argparse`` — build and parse the argparse schema (``cli.args``).
- ``prompt_for_item_selection``, ``display_processing_summary``,
  ``display_completion_summary``, ``display_resume_info`` — interactive
  presentation (``cli.display``).
- ``process_single_item``, ``check_and_wait_for_token_limit`` — per-item
  execution loop (``cli.loop``).
- Domain exceptions: ``ProcessingError``, ``ConfigurationError``,
  ``APIError``, ``FileProcessingError`` (``cli.errors``).
- ``handle_critical_error``, ``handle_recoverable_error`` — error
  handlers (``cli.errors``).
- Interaction helpers: ``print_header``, ``print_section``,
  ``print_success``, ``print_warning``, ``print_error``, ``print_info``,
  ``prompt_yes_no``, ``exit_program``, ``Colors`` (``cli.interaction``).
"""

from cli.args import setup_argparse
from cli.errors import (
    APIError,
    ConfigurationError,
    FileProcessingError,
    ProcessingError,
    handle_critical_error,
    handle_recoverable_error,
    safe_execute,
    validate_config_value,
    validate_directory_exists,
    validate_file_exists,
)
from cli.interaction import (
    Colors,
    exit_program,
    print_dim,
    print_error,
    print_header,
    print_highlight,
    print_info,
    print_section,
    print_separator,
    print_success,
    print_warning,
    prompt_continue,
    prompt_selection,
    prompt_yes_no,
)

__all__ = [
    "setup_argparse",
    "APIError",
    "ConfigurationError",
    "FileProcessingError",
    "ProcessingError",
    "handle_critical_error",
    "handle_recoverable_error",
    "safe_execute",
    "validate_config_value",
    "validate_directory_exists",
    "validate_file_exists",
    "Colors",
    "exit_program",
    "print_dim",
    "print_error",
    "print_header",
    "print_highlight",
    "print_info",
    "print_section",
    "print_separator",
    "print_success",
    "print_warning",
    "prompt_continue",
    "prompt_selection",
    "prompt_yes_no",
]
