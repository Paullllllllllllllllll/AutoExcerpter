"""Centralized user prompt utilities for consistent CLI interactions.

This module provides a standardized way to interact with users through the CLI,
including options to exit, go back, and make selections with clear formatting.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence, Set, Callable, TypeVar, Any

# Public API
__all__ = [
    "print_header",
    "print_section",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "prompt_selection",
    "prompt_yes_no",
    "prompt_continue",
    "exit_program",
]

# ANSI color codes for prettier output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

T = TypeVar('T')

# Constants
EXIT_COMMANDS = {'exit', 'quit', 'q'}
BACK_COMMANDS = {'back', 'b'}
ALL_COMMANDS = {'all', 'a'}
DIVIDER_CHAR = '─'
DIVIDER_LENGTH = 70


def print_header(message: str) -> None:
    """Print a prominent header message."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * DIVIDER_LENGTH}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{message.center(DIVIDER_LENGTH)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * DIVIDER_LENGTH}{Colors.ENDC}\n")


def print_section(message: str) -> None:
    """Print a section divider with message."""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{DIVIDER_CHAR * DIVIDER_LENGTH}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{DIVIDER_CHAR * DIVIDER_LENGTH}{Colors.ENDC}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def exit_program(message: str = "Exiting program. Goodbye!", exit_code: int = 0) -> None:
    """Exit the program gracefully with a message."""
    print(f"\n{Colors.OKCYAN}{message}{Colors.ENDC}\n")
    sys.exit(exit_code)


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


def prompt_selection(
    items: Sequence[T],
    display_func: Callable[[T], str],
    prompt_message: str = "Enter your choice",
    allow_multiple: bool = True,
    allow_all: bool = True,
    allow_back: bool = False,
    allow_exit: bool = True,
    process_all_label: Optional[str] = None,
) -> Optional[List[T]]:
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
    
    # Display items
    for index, item in enumerate(items, start=1):
        print(f"  {Colors.BOLD}[{index}]{Colors.ENDC} {display_func(item)}")
    
    # Add "process all" option if enabled
    if allow_all:
        all_label = process_all_label or "Process ALL listed items"
        print(f"  {Colors.BOLD}[{len(items) + 1}]{Colors.ENDC} {all_label}")
        print(f"  {Colors.BOLD}[all]{Colors.ENDC} {all_label}")
    
    # Build hint text
    hints = []
    if allow_multiple:
        hints.append("e.g., 1; 3-5")
    if allow_all:
        hints.append("'all' for all items")
    if allow_back:
        hints.append("'back' to go back")
    if allow_exit:
        hints.append("'exit' to quit")
    
    hint_text = f" ({'; '.join(hints)})" if hints else ""
    
    while True:
        try:
            choice_str = input(f"\n{prompt_message}{hint_text}: ").lower().strip()
            
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
            selected_indices: Set[int] = set()
            
            # Split by semicolon for multiple selections
            parts = choice_str.replace(" ", "").split(";") if allow_multiple else [choice_str]
            
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
                        f"Invalid input: '{part}'. Use numbers, ranges (e.g., 1-3), or 'all'."
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
