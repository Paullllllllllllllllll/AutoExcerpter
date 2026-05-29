"""Deterministic line-break repair for transcription text.

The transcription prompt preserves the source's printed line breaks, so the raw
model output has one line per printed source line. ``wrap_long_lines`` then
re-wrapped that text greedily at a per-page width, splitting long printed lines
into a near-width segment plus a short "orphan" remainder, and leaving the
source book's line-end hyphenation (``Manage-`` / ``ment``) unmerged.

This module reverses both effects, targeting the "preserve page layout" goal:

1. Wrap-rejoin: rejoin the orphan remainders back onto their printed line, the
   exact inverse of greedy wrapping. Because greedy wrapping fills each piece to
   just under the width, an internal break sits after a line A whenever
   ``len(A) + 1 + len(first_word(B)) > width``. Short structural lines (the
   table of contents, headings, page markers, catalog blocks) fall well below
   the width and are therefore left untouched automatically.
2. De-hyphenation: for a line ending in a word-hyphen, pull the continuation
   syllable up from the next line and keep the line break (so the printed-line
   structure is preserved), merging the word unless the conservative
   ``should_keep_hyphen`` guard says the hyphen is a genuine compound.

The repair only ever joins lines and merges hyphens; it never alters words,
numbers, punctuation, case, or any structural element. The wrap width is
estimated per page segment, because ``clean_transcription`` ran per page and so
the original auto-width varied page to page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pipeline.text_cleaner import should_keep_hyphen

PAGE_MARKER_RE = re.compile(r"<page_number>(.*?)</page_number>")
_HYPHEN_END_RE = re.compile(r"([A-Za-z]{2,})-$")
_LEADING_ALPHA_RE = re.compile(r"[A-Za-z]+")

# Minimum plausible wrap width; below this we assume no wrapping was applied.
_MIN_WRAP_WIDTH = 40
# Percentile of wrappable-line lengths used to estimate the wrap ceiling. Near
# the maximum, since greedy wrapping never emits a line longer than the width;
# slightly below 1.0 only to shrug off a single anomalous long line.
_WIDTH_PERCENTILE = 0.98
# A page counts as wrapped only if enough lines cluster just below the ceiling
# (the greedy-fill signature); an un-wrapped page has more dispersed lengths.
_CLUSTER_WINDOW = 6
_CLUSTER_MIN_COUNT = 3
_CLUSTER_MIN_RATIO = 0.08
# A wrap remainder (the short tail of a split printed line) is much shorter than
# a full line. Empirically the lengths are sharply bimodal: true remainders sit
# below ~40 chars, full next-source-lines above ~60. The cap keeps the repair
# from merging two genuine printed lines that merely happen to be near-width.
_REMAINDER_MIN = 35
_REMAINDER_FRACTION = 0.5


@dataclass
class HyphenDecision:
    """One line-break hyphenation decision, recorded for audit."""

    left: str
    right: str
    kept: bool
    context: str


@dataclass
class RepairAudit:
    """Diagnostics about a single file's repair, for the read-only LLM audit."""

    hyphen_decisions: list[HyphenDecision] = field(default_factory=list)
    long_lines: list[str] = field(default_factory=list)
    page_width_estimates: list[int] = field(default_factory=list)
    rejoin_applied: bool = False


def is_passthrough_line(line: str) -> bool:
    """Return True for structural lines that must pass through unchanged.

    These act as hard boundaries: prose is never merged into or out of them.
    Covers the metadata header and markdown headings (``#``), page markers,
    image descriptions, markdown table rows, and display math.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.startswith("<page") or stripped.startswith("[Page"):
        return True
    if stripped.startswith("!["):
        return True
    if stripped.startswith("[") and "Image:" in stripped:
        return True
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return stripped.startswith("$$") or stripped.endswith("$$")


def _opens_image_block(line: str) -> bool:
    """Return True if a line begins an image description block."""
    stripped = line.strip()
    if stripped.startswith("!["):
        return True
    return stripped.startswith("[") and "Image:" in stripped


def _is_wrappable(line: str) -> bool:
    """Return True for ordinary prose lines eligible for rejoin."""
    return bool(line.strip()) and not is_passthrough_line(line)


def _first_word_len(line: str) -> int:
    """Length of the first whitespace-delimited token of a line."""
    parts = line.split(None, 1)
    return len(parts[0]) if parts else 0


def _split_first_token(line: str) -> tuple[str, str]:
    """Split a line into its first token and the remainder after whitespace."""
    match = re.match(r"(\S+)\s*(.*)$", line.lstrip(), re.DOTALL)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def _percentile(sorted_values: list[int], fraction: float) -> int:
    """Return the value at a fractional rank within a sorted list."""
    if not sorted_values:
        return 0
    index = int(round(fraction * (len(sorted_values) - 1)))
    index = max(0, min(len(sorted_values) - 1, index))
    return sorted_values[index]


def _page_regions(lines: list[str]) -> list[tuple[int, int]]:
    """Return (start, end) spans of lines between page-number markers."""
    regions: list[tuple[int, int]] = []
    start = 0
    for index, line in enumerate(lines):
        if PAGE_MARKER_RE.search(line):
            if index > start:
                regions.append((start, index))
            start = index + 1
    if start < len(lines):
        regions.append((start, len(lines)))
    return regions


def _estimate_region_width(region_lines: list[str]) -> int | None:
    """Estimate a page's wrap width, or None if wrapping was not applied.

    Uses the high percentile of wrappable-line lengths as the ceiling, and
    requires a tight cluster of lines near that ceiling as evidence that greedy
    wrapping actually occurred (an un-wrapped page has more dispersed lengths).
    """
    lengths = sorted(len(line) for line in region_lines if _is_wrappable(line))
    if len(lengths) < 4:
        return None
    width = _percentile(lengths, _WIDTH_PERCENTILE)
    if width < _MIN_WRAP_WIDTH:
        return None
    near_ceiling = sum(
        1 for length in lengths if width - _CLUSTER_WINDOW <= length <= width
    )
    if (
        near_ceiling < _CLUSTER_MIN_COUNT
        or near_ceiling / len(lengths) < _CLUSTER_MIN_RATIO
    ):
        return None
    return width


def _line_widths(lines: list[str]) -> list[int | None]:
    """Map each line index to its page's estimated wrap width (or None)."""
    widths: list[int | None] = [None] * len(lines)
    for start, end in _page_regions(lines):
        width = _estimate_region_width(lines[start:end])
        for index in range(start, end):
            widths[index] = width
    return widths


def _line_is_full(lines: list[str], idx: int, b_len: int, width: int) -> bool:
    """Return True if line ``idx`` is itself a full wrap piece, not a remainder.

    A wrap piece is filled until the next word overflows the width. So line B is
    "full" when appending the first word of the following prose line would
    exceed the width. A short wrap remainder (the tail of a wrapped source line)
    is therefore not full, which lets the caller tell a remainder (rejoin) from
    the start of the next printed source line (keep the break).
    """
    nxt = lines[idx + 1] if idx + 1 < len(lines) else ""
    if not nxt.strip() or is_passthrough_line(nxt) or _opens_image_block(nxt):
        return False
    return b_len + 1 + _first_word_len(nxt) > width


def _rejoin_wrapped_lines(lines: list[str]) -> tuple[list[str], bool]:
    """Reverse greedy wrapping by rejoining orphan continuation lines."""
    widths = _line_widths(lines)
    out: list[str] = []
    buffer: str | None = None
    prev_len = 0
    applied = False
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        if _opens_image_block(line):
            if buffer is not None:
                out.append(buffer)
                buffer = None
            block = [line]
            while not line.rstrip().endswith("]") and index + 1 < total:
                index += 1
                line = lines[index]
                block.append(line)
            out.extend(block)
            index += 1
            continue

        if not line.strip() or is_passthrough_line(line):
            if buffer is not None:
                out.append(buffer)
                buffer = None
            out.append(line)
            index += 1
            continue

        width = widths[index]
        if buffer is None:
            buffer = line.rstrip()
            prev_len = len(buffer)
            index += 1
            continue
        if width is None:
            out.append(buffer)
            buffer = line.rstrip()
            prev_len = len(buffer)
            index += 1
            continue

        if _HYPHEN_END_RE.search(buffer.rstrip()):
            # The buffer ends in source hyphenation (e.g. "func-"); that is a
            # printed-line end, not a wrap piece. Flush it so de-hyphenation can
            # pull only the continuation syllable up, instead of space-joining.
            out.append(buffer)
            buffer = line.rstrip()
            prev_len = len(buffer)
            index += 1
            continue

        continuation = line.strip()
        a_is_full = prev_len + 1 + _first_word_len(line) > width
        remainder_cap = max(_REMAINDER_MIN, int(round(width * _REMAINDER_FRACTION)))
        b_is_remainder = len(continuation) <= remainder_cap and not _line_is_full(
            lines, index, len(continuation), width
        )
        if a_is_full and b_is_remainder:
            separator = "" if " " not in buffer.strip() else " "
            buffer = buffer.rstrip() + separator + continuation
            prev_len = len(continuation)
            applied = True
        else:
            out.append(buffer)
            buffer = line.rstrip()
            prev_len = len(buffer)
        index += 1

    if buffer is not None:
        out.append(buffer)
    return out, applied


def _dehyphenate_lines(lines: list[str]) -> tuple[list[str], list[HyphenDecision]]:
    """Pull line-break hyphenation syllables up, keeping the line break."""
    result = list(lines)
    decisions: list[HyphenDecision] = []
    index = 0

    while index < len(result):
        line = result[index]
        if not line.strip() or is_passthrough_line(line) or _opens_image_block(line):
            index += 1
            continue

        match = _HYPHEN_END_RE.search(line.rstrip())
        if match and index + 1 < len(result):
            nxt = result[index + 1]
            if (
                nxt.strip()
                and not is_passthrough_line(nxt)
                and not _opens_image_block(nxt)
            ):
                first_token, rest = _split_first_token(nxt)
                leading = _LEADING_ALPHA_RE.match(first_token)
                if leading:
                    left = match.group(1)
                    right = leading.group(0)
                    keep = should_keep_hyphen(left, right)
                    base = line.rstrip()
                    merged = base + first_token if keep else base[:-1] + first_token
                    result[index] = merged
                    rest = rest.lstrip()
                    if rest:
                        result[index + 1] = rest
                    else:
                        del result[index + 1]
                    decisions.append(
                        HyphenDecision(
                            left=left,
                            right=right,
                            kept=keep,
                            context=merged[-60:],
                        )
                    )
        index += 1

    return result, decisions


def repair_text(text: str) -> tuple[str, RepairAudit]:
    """Repair spurious line-breaks in one transcription, preserving layout.

    Args:
        text: The full transcription file content (LF line endings).

    Returns:
        A tuple of (repaired_text, audit). The repaired text differs from the
        input only in whitespace and merged line-break hyphens.
    """
    lines = text.split("\n")
    widths = [w for w in _line_widths(lines) if w is not None]

    rejoined, rejoin_applied = _rejoin_wrapped_lines(lines)
    final_lines, decisions = _dehyphenate_lines(rejoined)

    audit = RepairAudit(
        hyphen_decisions=decisions,
        page_width_estimates=sorted(set(widths)),
        rejoin_applied=rejoin_applied,
    )
    audit.long_lines = [
        line for line in final_lines if _is_wrappable(line) and len(line) > 100
    ]
    return "\n".join(final_lines), audit
