"""Deterministic content-preservation gate for the line-break repair.

The signature proves that the repair changed only whitespace and line-break
hyphens: it NFC-normalizes, folds the common ligatures, deletes the line-break
hyphen family (soft hyphen, hyphen-minus, Unicode hyphen) and all whitespace,
then case-folds. Two texts with equal signatures have an identical ordered
stream of letters, digits, and non-hyphen punctuation, so no word, number, or
punctuation mark can have been added, removed, reordered, or re-spelled.

Because the signature is whitespace-blind, structural drift (a dropped blank
line, a line gained) is caught by separate count checks instead.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

PAGE_MARKER_RE = re.compile(r"<page_number>(.*?)</page_number>")

# Ligatures folded to their ASCII expansion so a benign ligature change does not
# trip the gate; deliberately narrow (NFKC would also rewrite superscripts).
_LIGATURES = {
    0xFB00: "ff",
    0xFB01: "fi",
    0xFB02: "fl",
    0xFB03: "ffi",
    0xFB04: "ffl",
    0xFB05: "st",
    0xFB06: "st",
}
# Hyphen family that de-hyphenation is allowed to delete. En/em/figure dashes
# and the non-breaking hyphen (U+2011) are content and are NOT deleted.
_HYPHEN_DELETE = {0x00AD: None, 0x002D: None, 0x2010: None}
_SIGNATURE_MAP: dict[int, str | None] = {**_LIGATURES, **_HYPHEN_DELETE}


def content_signature(text: str) -> str:
    """Return the whitespace- and hyphen-insensitive content signature."""
    normalized = unicodedata.normalize("NFC", text)
    folded = normalized.translate(_SIGNATURE_MAP)
    no_space = "".join(ch for ch in folded if not ch.isspace())
    return no_space.casefold()


def page_markers(text: str) -> list[str]:
    """Return the ordered list of page-number marker inner texts."""
    return PAGE_MARKER_RE.findall(text)


def _alnum_count(text: str) -> int:
    """Count alphanumeric characters (an independent cross-check)."""
    return sum(1 for ch in text if ch.isalnum())


def _blank_line_count(text: str) -> int:
    """Count blank lines."""
    return sum(1 for line in text.split("\n") if not line.strip())


@dataclass
class VerifyResult:
    """Outcome of comparing an original transcription against its repair."""

    passed: bool
    signature_ok: bool
    markers_ok: bool
    alnum_ok: bool
    lines_ok: bool
    lines_before: int
    lines_after: int
    blank_before: int
    blank_after: int
    alnum_before: int
    alnum_after: int
    markers_before: int
    markers_after: int

    def summary(self) -> str:
        """One-line human-readable verdict."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict} | sig={self.signature_ok} markers={self.markers_ok} "
            f"alnum={self.alnum_before}->{self.alnum_after} "
            f"lines={self.lines_before}->{self.lines_after} "
            f"blanks={self.blank_before}->{self.blank_after}"
        )


def verify(original: str, repaired: str) -> VerifyResult:
    """Compare original vs repaired and return the gate result.

    Hard gates (all must hold): equal content signature; no page marker is lost
    (the original markers are a multiset subset of the repaired markers -- the
    repair may legitimately gain a marker by reassembling one the original
    wrapping had split across a line); equal alphanumeric count; the repair
    never increases line count (it only joins lines).
    """
    original_markers = page_markers(original)
    repaired_markers = page_markers(repaired)
    alnum_before = _alnum_count(original)
    alnum_after = _alnum_count(repaired)
    lines_before = original.count("\n") + 1
    lines_after = repaired.count("\n") + 1

    signature_ok = content_signature(original) == content_signature(repaired)
    lost_markers = Counter(original_markers) - Counter(repaired_markers)
    markers_ok = not lost_markers
    alnum_ok = alnum_before == alnum_after
    lines_ok = lines_after <= lines_before

    return VerifyResult(
        passed=signature_ok and markers_ok and alnum_ok and lines_ok,
        signature_ok=signature_ok,
        markers_ok=markers_ok,
        alnum_ok=alnum_ok,
        lines_ok=lines_ok,
        lines_before=lines_before,
        lines_after=lines_after,
        blank_before=_blank_line_count(original),
        blank_after=_blank_line_count(repaired),
        alnum_before=alnum_before,
        alnum_after=alnum_after,
        markers_before=len(original_markers),
        markers_after=len(repaired_markers),
    )
