"""Page-range grammar and item selection."""

from __future__ import annotations

import pytest

from autoexcerpter.common.selection import (
    ItemSelection,
    PageRange,
    is_numeric_selection,
    match_names,
    parse_page_range,
    select_items,
)

# Item names of the characterization tree: alpha.pdf, sub/beta.pdf, scans/.
TREE_NAMES = ["alpha.pdf", "beta.pdf", "scans"]


@pytest.mark.parametrize(
    ("spec", "total", "expected"),
    [
        ("5", 10, [0, 1, 2, 3, 4]),
        ("first:5", 10, [0, 1, 2, 3, 4]),
        ("FIRST : 2", 10, [0, 1]),
        ("last:3", 10, [7, 8, 9]),
        ("3-7", 10, [2, 3, 4, 5, 6]),
        ("3-", 5, [2, 3, 4]),
        ("-3", 10, [0, 1, 2]),
        ("1,3,5-8", 10, [0, 2, 4, 5, 6, 7]),
        ("5-8,1,3", 10, [0, 2, 4, 5, 6, 7]),
        ("2-4,3-6", 10, [1, 2, 3, 4, 5]),
        ("20", 3, [0, 1, 2]),
        ("last:20", 3, [0, 1, 2]),
        ("8-12", 9, [7, 8]),
        ("12-15", 9, []),
    ],
)
def test_page_range_resolves(spec: str, total: int, expected: list[int]) -> None:
    assert parse_page_range(spec).resolve(total) == expected


@pytest.mark.parametrize(
    "spec", ["", "  ", "0", "first:0", "last:0", "0-3", "5-3", "-", "a", "1,x", ","]
)
def test_page_range_rejects_malformed(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_page_range(spec)


def test_page_range_select_slices_first_and_last() -> None:
    units = ["a", "b", "c", "d"]
    assert parse_page_range("first:2").select(units) == ["a", "b"]
    assert parse_page_range("last:2").select(units) == ["c", "d"]
    assert parse_page_range("2,4").select(units) == ["b", "d"]


def test_page_range_resolve_of_empty_document() -> None:
    assert parse_page_range("1-3").resolve(0) == []


def test_page_range_describe() -> None:
    assert parse_page_range("4").describe() == "first 4 page(s)"
    assert parse_page_range("last:2").describe() == "last 2 page(s)"
    assert parse_page_range("1,3-5,9-").describe() == "pages 1,3-5,9-"
    assert PageRange().describe() == "all pages"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        pytest.param("2", ["beta.pdf"], id="directory_select_number"),
        pytest.param("2-3", ["beta.pdf", "scans"], id="directory_select_range"),
        pytest.param("alph", ["alpha.pdf"], id="directory_select_name"),
    ],
)
def test_golden_selections(expr: str, expected: list[str]) -> None:
    assert select_items(TREE_NAMES, expr).pick(TREE_NAMES) == expected


def test_name_search_matching_nothing_is_empty() -> None:
    selection = select_items(TREE_NAMES, "gamma")
    assert selection == ItemSelection(indices=(), by_name=True)


def test_name_search_ignores_case_and_outer_spaces() -> None:
    assert select_items(TREE_NAMES, "  BETA ").indices == (1,)
    assert select_items(TREE_NAMES, "a").indices == (0, 1, 2)


def test_numeric_lists_accept_semicolons_and_spaces() -> None:
    assert select_items(TREE_NAMES, "1; 3").indices == (0, 2)
    assert select_items(TREE_NAMES, "3, 1, 1").indices == (0, 2)


def test_out_of_range_and_malformed_parts_are_reported() -> None:
    selection = select_items(TREE_NAMES, "0,2,4,2-5,-1,1-2-3,3-")
    assert selection.indices == (1,)
    assert selection.unmatched == ("0", "4", "2-5", "-1", "1-2-3", "3-")
    assert not selection.by_name


def test_reversed_range_is_unmatched() -> None:
    selection = select_items(TREE_NAMES, "3-1")
    assert selection.indices == ()
    assert selection.unmatched == ("3-1",)


@pytest.mark.parametrize(
    ("expr", "numeric"),
    [
        ("2", True),
        ("1-3", True),
        ("1, 3; 5", True),
        ("-", False),
        (",", False),
        ("vol2", False),
        ("2 a", False),
        ("", False),
    ],
)
def test_numeric_detection(expr: str, numeric: bool) -> None:
    assert is_numeric_selection(expr) is numeric


def test_names_with_digits_use_name_search() -> None:
    names = ["vol1.pdf", "vol2.pdf", "vol10.pdf"]
    assert select_items(names, "vol1").indices == (0, 2)
    assert match_names("VOL2", names) == {1}
