"""Command-line interface for AutoExcerpter.

Public interface:

- ``setup_argparse`` — build and parse the argparse schema (``cli.args``).
- Domain exceptions: ``ProcessingError``, ``ConfigurationError``,
  ``APIError``, ``FileProcessingError`` (``cli.errors``).
- ``handle_critical_error``, ``handle_recoverable_error``,
  ``safe_execute``, ``validate_config_value``,
  ``validate_directory_exists``, ``validate_file_exists`` (``cli.errors``).
- Console output: ``print_header``, ``print_section``, ``print_success``,
  ``print_warning``, ``print_error``, ``print_info``, ``print_separator``,
  ``print_dim``, ``print_highlight``, ``Colors`` (``cli.interaction``).
- Prompts and program control: ``prompt_selection``, ``prompt_yes_no``,
  ``prompt_continue``, ``exit_program`` (``cli.interaction``).

The presentation helpers in ``cli.display`` and the per-item execution
helpers in ``cli.loop`` are private to ``main.py`` and are imported from
their own modules rather than re-exported here.
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
