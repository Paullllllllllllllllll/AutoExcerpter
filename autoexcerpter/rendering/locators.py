"""Page locators of citations and Roman numeral conversion.

A locator is ``(namespace, number, inferred)``. The namespace says what the
number counts: a printed arabic or roman page number, or the 1-based position
of the scan in the PDF or image folder (for pages without a printed number).
``inferred`` marks a printed-style number inferred from the neighboring pages
rather than read from the page; it renders in brackets.
"""

from __future__ import annotations

from collections.abc import Iterable

Locator = tuple[str, int, bool]

PRINTED_ARABIC = "printed-arabic"
PRINTED_ROMAN = "printed-roman"
PDF_POSITION = "pdf"
IMAGE_POSITION = "image"
# Output order of the locator groups: roman front matter before the arabic
# body, then the positional namespaces for pages without a printed number.
LOCATOR_NAMESPACES = (PRINTED_ROMAN, PRINTED_ARABIC, PDF_POSITION, IMAGE_POSITION)
# Singular and plural label per namespace.
_NAMESPACE_LABELS = {
    PRINTED_ROMAN: ("p.", "pp."),
    PRINTED_ARABIC: ("p.", "pp."),
    PDF_POSITION: ("PDF p.", "PDF pp."),
    IMAGE_POSITION: ("Image", "Images"),
}

_ROMAN_NUMERAL_VALUES = [
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
]
_ROMAN_DIGITS = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def int_to_roman(num: int) -> str:
    """Return *num* as a lowercase Roman numeral ("" for zero or less)."""
    if num <= 0:
        return ""
    result = []
    for value, numeral in _ROMAN_NUMERAL_VALUES:
        while num >= value:
            result.append(numeral)
            num -= value
    return "".join(result)


def roman_to_int(text: str) -> int | None:
    """Convert a Roman numeral to an int (case-insensitive), or None if invalid."""
    s = text.lower()
    if not s or any(ch not in _ROMAN_DIGITS for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_DIGITS[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total or None


def as_locator(locator: Locator | int | None) -> Locator | None:
    """Normalize a page argument; a bare int is a printed arabic page."""
    if locator is None:
        return None
    if isinstance(locator, int):
        return (PRINTED_ARABIC, locator, False)
    return locator


def _namespace_rank(namespace: str) -> int:
    """Return the output position of *namespace* (unknown ones go last)."""
    try:
        return LOCATOR_NAMESPACES.index(namespace)
    except ValueError:
        return len(LOCATOR_NAMESPACES)


def _sort_locators(locators: Iterable[Locator]) -> list[Locator]:
    """Return *locators* deduplicated, grouped by namespace, then by number."""
    return sorted(
        set(locators),
        key=lambda loc: (_namespace_rank(loc[0]), loc[0], loc[1], loc[2]),
    )


def _format_locator_number(namespace: str, number: int, inferred: bool) -> str:
    """Render one number in its namespace's numerals, bracketed if inferred."""
    if namespace == PRINTED_ROMAN:
        text = int_to_roman(number) or str(number)
    else:
        text = str(number)
    return f"[{text}]" if inferred else text


def format_page_range(locators: Iterable[Locator]) -> str:
    """Format page locators as compact ranges, one group per namespace.

    Consecutive numbers form a range only within one namespace and with the
    same inferred status; groups are joined with "; " in the order of
    ``LOCATOR_NAMESPACES``.

    Examples:
        printed 1, 2, 3, 5 -> "pp. 1-3, 5"
        printed 5 -> "p. 5"
        roman 10-12 -> "pp. x-xii"
        pdf 12-14 -> "PDF pp. 12-14"
        inferred 6 -> "p. [6]"
        printed 5, inferred 6, printed 7 -> "pp. 5, [6], 7"
        printed 5-7 and pdf 12 -> "pp. 5-7; PDF p. 12"
    """
    ordered = _sort_locators(locators)
    if not ordered:
        return ""

    groups: list[str] = []
    for namespace in dict.fromkeys(loc[0] for loc in ordered):
        members = [loc for loc in ordered if loc[0] == namespace]
        # One page can be recorded both as inferred and as read; the read
        # number wins.
        read = {loc[1] for loc in members if not loc[2]}
        members = [loc for loc in members if not (loc[2] and loc[1] in read)]

        runs: list[tuple[int, int, bool]] = []
        for _ns, number, inferred in members:
            if runs and runs[-1][2] == inferred and number == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], number, inferred)
            else:
                runs.append((number, number, inferred))

        parts = []
        for start, end, inferred in runs:
            first = _format_locator_number(namespace, start, inferred)
            if start == end:
                parts.append(first)
            else:
                last = _format_locator_number(namespace, end, inferred)
                parts.append(f"{first}-{last}")

        singular, plural = _NAMESPACE_LABELS.get(namespace, ("p.", "pp."))
        label = singular if len(members) == 1 else plural
        groups.append(f"{label} {', '.join(parts)}")

    return "; ".join(groups)


__all__ = [
    "IMAGE_POSITION",
    "LOCATOR_NAMESPACES",
    "PDF_POSITION",
    "PRINTED_ARABIC",
    "PRINTED_ROMAN",
    "Locator",
    "as_locator",
    "format_page_range",
    "int_to_roman",
    "roman_to_int",
]
