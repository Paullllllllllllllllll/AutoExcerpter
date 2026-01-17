"""Centralized user prompt utilities for consistent CLI interactions.

This module provides a standardized way to interact with users through the CLI,
including options to exit, go back, and make selections with clear formatting.

Features:
- Color-coded output (success, warning, error, info)
- Interactive selection prompts with multiple options
- Yes/No prompts with defaults
- Exit and back navigation support
- Consistent formatting with dividers and headers
- Windows color support via colorama
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Sequence, TypeVar


def _safe_print(text: str) -> None:
    """Print text with fallback for encoding errors on Windows consoles."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

# Initialize colorama for Windows color support
try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass  # colorama not available, colors may still work on Unix-like systems

from modules.constants import (
    EXIT_COMMANDS,
    BACK_COMMANDS,
    ALL_COMMANDS,
    DIVIDER_CHAR,
    DIVIDER_LENGTH,
)

T = TypeVar('T')


# ============================================================================
# ANSI Color Codes
# ============================================================================
class Colors:
    """ANSI color codes for terminal output formatting."""
    # Original colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Additional semantic colors (ChronoTranscriber style)
    INFO = '\033[0;36m'        # Cyan (informational messages)
    SUCCESS = '\033[1;32m'     # Green bold (success messages)
    ERROR = '\033[1;31m'       # Red bold (errors)
    PROMPT = '\033[1;37m'      # White bold (user prompts)
    DIM = '\033[2;37m'         # Dimmed white (secondary text)
    HIGHLIGHT = '\033[1;35m'   # Magenta bold (highlights)
    RESET = '\033[0m'          # Reset to default


# ============================================================================
# Output Functions (Print Messages)
# ============================================================================
def print_header(message: str, subtitle: str = "") -> None:
    """Print a prominent header message with optional subtitle.
    
    Args:
        message: Main header text
        subtitle: Optional subtitle text
    """
    _safe_print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * DIVIDER_LENGTH}{Colors.ENDC}")
    _safe_print(f"{Colors.BOLD}{Colors.HEADER}  {message}{Colors.ENDC}")
    if subtitle:
        _safe_print(f"{Colors.OKCYAN}  {subtitle}{Colors.ENDC}")
    _safe_print(f"{Colors.BOLD}{Colors.HEADER}{'=' * DIVIDER_LENGTH}{Colors.ENDC}\n")


def print_section(message: str) -> None:
    """Print a section divider with message."""
    _safe_print(f"\n{Colors.BOLD}{Colors.OKBLUE}{DIVIDER_CHAR * DIVIDER_LENGTH}{Colors.ENDC}")
    _safe_print(f"{Colors.BOLD}{Colors.OKBLUE}{message}{Colors.ENDC}")
    _safe_print(f"{Colors.BOLD}{Colors.OKBLUE}{DIVIDER_CHAR * DIVIDER_LENGTH}{Colors.ENDC}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    _safe_print(f"{Colors.OKGREEN}[OK] {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    _safe_print(f"{Colors.WARNING}[WARN] {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    _safe_print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """Print an info message."""
    _safe_print(f"{Colors.OKCYAN}[INFO] {message}{Colors.ENDC}")


def print_separator(char: str = "-", length: int = 80, color: str = "") -> None:
    """Print a separator line.
    
    Args:
        char: Character to use for separator
        length: Length of the separator line
        color: Optional color code (defaults to dim)
    """
    if not color:
        color = Colors.DIM
    _safe_print(f"{color}{char * length}{Colors.ENDC}")


def print_dim(message: str) -> None:
    """Print a dimmed/secondary message."""
    _safe_print(f"{Colors.DIM}{message}{Colors.ENDC}")


def print_highlight(message: str) -> None:
    """Print a highlighted message."""
    _safe_print(f"{Colors.BOLD}{Colors.OKCYAN}{message}{Colors.ENDC}")


# ============================================================================
# Program Control Functions
# ============================================================================
def exit_program(message: str = "Exiting program. Goodbye!", exit_code: int = 0) -> None:
    """Exit the program gracefully with a message."""
    _safe_print(f"\n{Colors.OKCYAN}{message}{Colors.ENDC}\n")
    sys.exit(exit_code)


# ============================================================================
# Interactive Prompt Functions
# ============================================================================
def prompt_yes_no(
    question: str,
    default: Optional[bool] = None,
    allow_exit: bool = True,
) -> bool:
    """
    Prompt user for yes/no response.
    
    Args:
        question: The question to ask
        default: Default response if user just presses Enter (None means no default)
        allow_exit: Whether to allow 'exit' or 'quit' commands
        
    Returns:
        True for yes, False for no
    """
    if default is True:
        prompt_suffix = " [Y/n]"
    elif default is False:
        prompt_suffix = " [y/N]"
    else:
        prompt_suffix = " [y/n]"
    
    exit_hint = " (or 'exit' to quit)" if allow_exit else ""
    
    while True:
        response = input(f"{question}{prompt_suffix}{exit_hint}: ").lower().strip()
        
        if allow_exit and response in EXIT_COMMANDS:
            exit_program()
        
        if not response and default is not None:
            return default
        
        if response in {'y', 'yes'}:
            return True
        if response in {'n', 'no'}:
            return False
        
        print_warning("Please answer 'yes' or 'no' (or 'y'/'n').")


def prompt_continue(message: str = "Press Enter to continue", allow_exit: bool = True) -> None:
    """Prompt user to press Enter to continue."""
    exit_hint = " (or 'exit' to quit)" if allow_exit else ""
    response = input(f"\n{Colors.OKCYAN}{message}{exit_hint}...{Colors.ENDC} ").lower().strip()
    
    if allow_exit and response in EXIT_COMMANDS:
        exit_program()


def _match_items_by_name(
    search_term: str,
    items: Sequence[T],
    display_func: Callable[[T], str],
) -> set[int]:
    """
    Match items by filename/name search.
    
    Args:
        search_term: The search string to match against item names
        items: List of items to search through
        display_func: Function to convert item to display string
        
    Returns:
        Set of indices of matching items (0-based)
    """
    search_lower = search_term.lower().strip()
    matched_indices: set[int] = set()
    
    for idx, item in enumerate(items):
        display_text = display_func(item).lower()
        # Also try to get just the filename if item has a path attribute
        item_name = ""
        if hasattr(item, 'path'):
            item_name = item.path.name.lower()
        elif hasattr(item, 'name'):
            item_name = item.name.lower()
        
        # Match against display text or item name
        if search_lower in display_text or search_lower in item_name:
            matched_indices.add(idx)
    
    return matched_indices


def prompt_selection(
    items: Sequence[T],
    display_func: Callable[[T], str],
    prompt_message: str = "Enter your choice",
    allow_multiple: bool = True,
    allow_all: bool = True,
    allow_back: bool = False,
    allow_exit: bool = True,
    process_all_label: Optional[str] = None,
) -> Optional[list[T]]:
    """
    Prompt user to select one or more items from a list.
    
    Args:
        items: List of items to select from
        display_func: Function to convert item to display string
        prompt_message: Custom prompt message
        allow_multiple: Allow selecting multiple items (ranges, semicolon-separated)
        allow_all: Allow selecting all items at once
        allow_back: Allow going back (returns None)
        allow_exit: Allow exiting the program
        process_all_label: Custom label for "process all" option
        
    Returns:
        List of selected items, or None if user chose to go back
    """
    if not items:
        print_warning("No items available to select.")
        return []
    
    # Display items with improved formatting
    for index, item in enumerate(items, start=1):
        description = display_func(item)
        # Truncate very long descriptions
        if len(description) > 75:
            description = description[:72] + "..."
        _safe_print(f"  {Colors.BOLD}{index}.{Colors.ENDC} {description}")
    
    # Selection instructions
    _safe_print(f"\n  {Colors.INFO}Selection options:{Colors.ENDC}")
    if allow_multiple:
        _safe_print(f"    {Colors.DIM}- Enter numbers separated by commas (e.g., '1,3,5'){Colors.ENDC}")
        _safe_print(f"    {Colors.DIM}- Enter a range with a dash (e.g., '1-5'){Colors.ENDC}")
    _safe_print(f"    {Colors.DIM}- Enter a filename or part of it to search{Colors.ENDC}")
    if allow_all:
        _safe_print(f"    {Colors.DIM}- Enter 'all' to select everything{Colors.ENDC}")
    
    # Navigation hints
    nav_hints = []
    if allow_back:
        nav_hints.append("'back' to go back")
    if allow_exit:
        nav_hints.append("'exit' to quit")
    
    if nav_hints:
        _safe_print(f"\n  {Colors.DIM}{' | '.join(nav_hints)}{Colors.ENDC}")
    
    hint_text = ""
    
    while True:
        try:
            choice_str = input(f"\n{Colors.PROMPT}{prompt_message}: {Colors.ENDC}").lower().strip()
            
            if not choice_str:
                print_warning("No selection made. Please make a choice.")
                continue
            
            # Handle special commands
            if allow_exit and choice_str in EXIT_COMMANDS:
                exit_program()
            
            if allow_back and choice_str in BACK_COMMANDS:
                return None
            
            if allow_all and (choice_str in ALL_COMMANDS or choice_str == str(len(items) + 1)):
                print_success(f"Selected all {len(items)} items.")
                return list(items)
            
            # Parse selection
            selected_indices: set[int] = set()
            
            # First, try to interpret as filename search (if not purely numeric/range pattern)
            # This allows filenames with spaces, commas, etc. to be matched
            stripped_input = choice_str.strip()
            
            # Check if input looks like a numeric selection (numbers, ranges, commas)
            numeric_pattern = stripped_input.replace(" ", "").replace(";", ",")
            is_numeric_selection = all(
                c.isdigit() or c in ",-" for c in numeric_pattern
            ) and any(c.isdigit() for c in numeric_pattern)
            
            if not is_numeric_selection:
                # Try filename matching first for non-numeric input
                matched = _match_items_by_name(stripped_input, items, display_func)
                if matched:
                    selected_indices.update(matched)
                else:
                    raise ValueError(
                        f"No items found matching '{stripped_input}'. Use numbers, ranges (e.g., 1-3), 'all', or a filename."
                    )
            else:
                # Numeric selection: split by semicolon or comma
                normalized_input = numeric_pattern
                parts = normalized_input.split(",") if allow_multiple else [stripped_input]
                
                for part in parts:
                    if not part:
                        continue
                    
                    # Handle ranges (e.g., "1-3")
                    if "-" in part and allow_multiple:
                        try:
                            start_str, end_str = part.split("-", 1)
                            start = int(start_str)
                            end = int(end_str)
                            
                            if not (1 <= start <= end <= len(items)):
                                raise ValueError(
                                    f"Range {part} is invalid. Must be between 1 and {len(items)}."
                                )
                            
                            selected_indices.update(range(start - 1, end))
                        except ValueError as e:
                            print_error(f"Invalid range '{part}': {e}")
                            raise
                    
                    # Handle single numbers
                    elif part.isdigit():
                        index = int(part) - 1
                        if not (0 <= index < len(items)):
                            raise ValueError(
                                f"Selection {part} is out of range. Must be between 1 and {len(items)}."
                            )
                        selected_indices.add(index)
                        if not allow_multiple:
                            break
                    
                    else:
                        raise ValueError(
                            f"Invalid input: '{part}'. Use numbers, ranges (e.g., 1-3), 'all', or a filename."
                        )
            
            if not selected_indices:
                print_warning("No valid items selected. Please try again.")
                continue
            
            selected_items = [items[i] for i in sorted(selected_indices)]
            
            # Confirm selection
            if len(selected_items) == 1:
                print_success(f"Selected: {display_func(selected_items[0])}")
            else:
                print_success(f"Selected {len(selected_items)} item(s).")
            
            return selected_items
            
        except ValueError as e:
            print_error(f"Invalid selection: {e}")
        except KeyboardInterrupt:
            if allow_exit:
                exit_program("\n\nInterrupted by user.")
            raise
        except Exception as e:
            print_error(f"Unexpected error: {e}")


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "print_header",
    "print_section",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "print_separator",
    "print_dim",
    "print_highlight",
    "prompt_selection",
    "prompt_yes_no",
    "prompt_continue",
    "exit_program",
    "Colors",
]
