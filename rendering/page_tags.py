"""Read printed page numbers from ``<page_number>`` tags in a transcription."""

from __future__ import annotations

import re

from rendering.citations import _roman_to_int

_PAGE_TAG_RE = re.compile(r"<page_number>(.*?)</page_number>", re.DOTALL)
_ARABIC_RE = re.compile(r"\d+")
_ROMAN_RE = re.compile(r"\b[ivxlcdm]+\b", re.IGNORECASE)


def printed_page_numbers(text: str | None) -> list[int]:
    """Return the page numbers tagged in *text*, in order of appearance.

    Each ``<page_number>…</page_number>`` tag yields at most one integer: the
    first arabic number in the tag, else its first Roman numeral converted to
    an integer. Tags with neither are skipped.

    Args:
        text: Transcription text of one page.

    Returns:
        The tagged page numbers; empty when the page has no usable tag.
    """
    if not text:
        return []
    numbers: list[int] = []
    for inner in _PAGE_TAG_RE.findall(text):
        arabic = _ARABIC_RE.search(inner)
        if arabic:
            numbers.append(int(arabic.group()))
            continue
        roman = _ROMAN_RE.search(inner)
        if roman:
            value = _roman_to_int(roman.group())
            if value is not None:
                numbers.append(value)
    return numbers


__all__ = ["printed_page_numbers"]
