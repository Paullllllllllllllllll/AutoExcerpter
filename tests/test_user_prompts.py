"""Tests for user_prompts module, particularly filename selection functionality.

This module tests the interactive selection prompts and filename matching
capabilities added to support file selection by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from unittest.mock import patch

import pytest

from modules.user_prompts import _match_items_by_name


# ============================================================================
# Test Fixtures
# ============================================================================
@dataclass
class MockItem:
    """Mock item class for testing selection functionality."""
    path: Path
    kind: str = "pdf"
    
    def display_label(self) -> str:
        return f"PDF: {self.path.name} (from: {self.path.parent})"


@pytest.fixture
def sample_items() -> list[MockItem]:
    """Create sample items for testing selection."""
    return [
        MockItem(Path(r"C:\Documents\Abbott A History Marriage.pdf")),
        MockItem(Path(r"C:\Documents\Abbott A History Mistresses.pdf")),
        MockItem(Path(r"C:\Documents\Mennell 1996 All manners of food eating and taste in England and France.pdf")),
        MockItem(Path(r"C:\Documents\Albala 2003 Food in Early Modern Europe.pdf")),
        MockItem(Path(r"C:\Documents\Albala 2013 Routledge International Handbook of Food Studies.pdf")),
        MockItem(Path(r"C:\Documents\Ziegler The Black Death.pdf")),
    ]


def display_func(item: MockItem) -> str:
    """Display function for mock items."""
    return item.display_label()


# ============================================================================
# Tests for _match_items_by_name
# ============================================================================
class TestMatchItemsByName:
    """Test suite for the _match_items_by_name helper function."""
    
    def test_exact_filename_match(self, sample_items: list[MockItem]) -> None:
        """Test matching an exact filename."""
        matched = _match_items_by_name(
            "Ziegler The Black Death.pdf",
            sample_items,
            display_func
        )
        assert matched == {5}
    
    def test_partial_filename_match(self, sample_items: list[MockItem]) -> None:
        """Test matching a partial filename."""
        matched = _match_items_by_name(
            "Mennell",
            sample_items,
            display_func
        )
        assert matched == {2}
    
    def test_case_insensitive_match(self, sample_items: list[MockItem]) -> None:
        """Test that matching is case-insensitive."""
        matched = _match_items_by_name(
            "ALBALA",
            sample_items,
            display_func
        )
        assert matched == {3, 4}
    
    def test_multiple_matches(self, sample_items: list[MockItem]) -> None:
        """Test matching multiple items with same pattern."""
        matched = _match_items_by_name(
            "Abbott",
            sample_items,
            display_func
        )
        assert matched == {0, 1}
    
    def test_no_match(self, sample_items: list[MockItem]) -> None:
        """Test when no items match."""
        matched = _match_items_by_name(
            "NonexistentBook",
            sample_items,
            display_func
        )
        assert matched == set()
    
    def test_match_with_spaces(self, sample_items: list[MockItem]) -> None:
        """Test matching filenames with spaces."""
        matched = _match_items_by_name(
            "food eating and taste",
            sample_items,
            display_func
        )
        assert matched == {2}
    
    def test_match_year_pattern(self, sample_items: list[MockItem]) -> None:
        """Test matching by year in filename."""
        matched = _match_items_by_name(
            "2003",
            sample_items,
            display_func
        )
        assert matched == {3}
    
    def test_match_in_display_text(self, sample_items: list[MockItem]) -> None:
        """Test matching against display text (includes parent path)."""
        matched = _match_items_by_name(
            "Documents",
            sample_items,
            display_func
        )
        # All items have Documents in their path
        assert matched == {0, 1, 2, 3, 4, 5}
    
    def test_empty_search_term(self, sample_items: list[MockItem]) -> None:
        """Test with empty search term."""
        matched = _match_items_by_name(
            "",
            sample_items,
            display_func
        )
        # Empty string matches everything
        assert matched == {0, 1, 2, 3, 4, 5}
    
    def test_whitespace_only_search(self, sample_items: list[MockItem]) -> None:
        """Test with whitespace-only search term."""
        matched = _match_items_by_name(
            "   ",
            sample_items,
            display_func
        )
        # Stripped empty string matches everything
        assert matched == {0, 1, 2, 3, 4, 5}


# ============================================================================
# Tests for CLI Selection Parsing
# ============================================================================
class TestCLISelectionParsing:
    """Test suite for CLI --select argument parsing."""
    
    def test_parse_single_number(self, sample_items: list[MockItem]) -> None:
        """Test parsing a single number selection."""
        from main import _parse_cli_selection
        
        selected = _parse_cli_selection(sample_items, "3")  # type: ignore[arg-type]
        assert len(selected) == 1
        assert selected[0].path.name == "Mennell 1996 All manners of food eating and taste in England and France.pdf"
    
    def test_parse_comma_separated_numbers(self, sample_items: list[MockItem]) -> None:
        """Test parsing comma-separated numbers."""
        from main import _parse_cli_selection
        
        selected = _parse_cli_selection(sample_items, "1,3,6")  # type: ignore[arg-type]
        assert len(selected) == 3
        assert selected[0].path.name == "Abbott A History Marriage.pdf"
        assert selected[1].path.name == "Mennell 1996 All manners of food eating and taste in England and France.pdf"
        assert selected[2].path.name == "Ziegler The Black Death.pdf"
    
    def test_parse_range(self, sample_items: list[MockItem]) -> None:
        """Test parsing a range selection."""
        from main import _parse_cli_selection
        
        selected = _parse_cli_selection(sample_items, "1-3")  # type: ignore[arg-type]
        assert len(selected) == 3
        assert selected[0].path.name == "Abbott A History Marriage.pdf"
        assert selected[1].path.name == "Abbott A History Mistresses.pdf"
        assert selected[2].path.name == "Mennell 1996 All manners of food eating and taste in England and France.pdf"
    
    def test_parse_filename_pattern(self, sample_items: list[MockItem]) -> None:
        """Test parsing a filename pattern."""
        from main import _parse_cli_selection
        
        selected = _parse_cli_selection(sample_items, "Albala")  # type: ignore[arg-type]
        assert len(selected) == 2
        assert "Albala 2003" in selected[0].path.name
        assert "Albala 2013" in selected[1].path.name
    
    def test_parse_filename_with_spaces(self, sample_items: list[MockItem]) -> None:
        """Test parsing a filename pattern with spaces."""
        from main import _parse_cli_selection
        
        selected = _parse_cli_selection(sample_items, "Black Death")  # type: ignore[arg-type]
        assert len(selected) == 1
        assert "Black Death" in selected[0].path.name
    
    def test_parse_invalid_range(self, sample_items: list[MockItem]) -> None:
        """Test parsing an invalid range (out of bounds)."""
        from main import _parse_cli_selection
        
        selected = _parse_cli_selection(sample_items, "1-100")  # type: ignore[arg-type]
        assert len(selected) == 0  # Invalid range should return empty
    
    def test_parse_no_match(self, sample_items: list[MockItem]) -> None:
        """Test parsing with no matching items."""
        from main import _parse_cli_selection
        
        selected = _parse_cli_selection(sample_items, "NonexistentAuthor")  # type: ignore[arg-type]
        assert len(selected) == 0


# ============================================================================
# Integration Tests for Selection Flow
# ============================================================================
class TestSelectionIntegration:
    """Integration tests for the full selection flow."""
    
    def test_numeric_detection_simple_number(self) -> None:
        """Test that simple numbers are detected as numeric."""
        pattern = "123"
        numeric_pattern = pattern.replace(" ", "").replace(";", ",")
        is_numeric = all(
            c.isdigit() or c in ",-" for c in numeric_pattern
        ) and any(c.isdigit() for c in numeric_pattern)
        assert is_numeric is True
    
    def test_numeric_detection_range(self) -> None:
        """Test that ranges are detected as numeric."""
        pattern = "1-5"
        numeric_pattern = pattern.replace(" ", "").replace(";", ",")
        is_numeric = all(
            c.isdigit() or c in ",-" for c in numeric_pattern
        ) and any(c.isdigit() for c in numeric_pattern)
        assert is_numeric is True
    
    def test_numeric_detection_comma_list(self) -> None:
        """Test that comma-separated lists are detected as numeric."""
        pattern = "1,3,5"
        numeric_pattern = pattern.replace(" ", "").replace(";", ",")
        is_numeric = all(
            c.isdigit() or c in ",-" for c in numeric_pattern
        ) and any(c.isdigit() for c in numeric_pattern)
        assert is_numeric is True
    
    def test_numeric_detection_filename(self) -> None:
        """Test that filenames are NOT detected as numeric."""
        pattern = "Mennell 1996 All manners of food"
        numeric_pattern = pattern.replace(" ", "").replace(";", ",")
        is_numeric = all(
            c.isdigit() or c in ",-" for c in numeric_pattern
        ) and any(c.isdigit() for c in numeric_pattern)
        assert is_numeric is False
    
    def test_numeric_detection_author_name(self) -> None:
        """Test that author names are NOT detected as numeric."""
        pattern = "Abbott"
        numeric_pattern = pattern.replace(" ", "").replace(";", ",")
        is_numeric = all(
            c.isdigit() or c in ",-" for c in numeric_pattern
        ) and any(c.isdigit() for c in numeric_pattern)
        assert is_numeric is False
