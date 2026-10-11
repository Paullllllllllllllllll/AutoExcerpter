"""Version 1.0 outputs as ``migrate`` reads them.

Version 1.0 kept two working logs per item in ``<stem>_autoexcerpter/``:
``transcription.jsonl`` and ``summary.jsonl``. The first line of each is a
header carrying ``_format_version`` 2; every following line is one page
result keyed by ``original_input_order_index``. :func:`read_working_log`
reads one such log tolerantly: a truncated last line (a crash mid-append)
is dropped, unparsable interior lines and records without a page index are
skipped, and a log without the version 2 header is refused.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "FORMAT_VERSION",
    "PAGE_KEY",
    "VERSION_KEY",
    "WorkingLog",
    "WorkingLogError",
    "read_working_log",
]

logger = logging.getLogger(__name__)

VERSION_KEY: Final = "_format_version"
FORMAT_VERSION: Final = 2
PAGE_KEY: Final = "original_input_order_index"


class WorkingLogError(ValueError):
    """A working log lacks the version 2 header."""


@dataclass(frozen=True)
class WorkingLog:
    """A version 1.0 working log: its header and its page results in file order.

    *dropped_tail* says whether a truncated last line was dropped;
    *skipped_lines* holds the 1-based numbers of unparsable interior lines.
    """

    header: dict[str, Any]
    results: list[dict[str, Any]]
    dropped_tail: bool = False
    skipped_lines: tuple[int, ...] = field(default=())


def read_working_log(path: Path) -> WorkingLog | None:
    """Read the working log at *path*; None when it is missing or blank.

    Raises:
        WorkingLogError: The first line is not a version 2 header.
        OSError: The file cannot be read.
        UnicodeDecodeError: The file is not UTF-8.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").removeprefix("\N{BYTE ORDER MARK}")
    lines = [
        (number, line.strip())
        for number, line in enumerate(text.split("\n"), start=1)
        if line.strip()
    ]
    if not lines:
        return None
    try:
        header = json.loads(lines[0][1])
    except json.JSONDecodeError as exc:
        raise WorkingLogError(f"{path}: the first line is not a header: {exc}") from exc
    if not isinstance(header, dict) or header.get(VERSION_KEY) != FORMAT_VERSION:
        found = header.get(VERSION_KEY) if isinstance(header, dict) else None
        raise WorkingLogError(
            f"{path}: expected {VERSION_KEY}={FORMAT_VERSION} in the header, "
            f"found {found!r}."
        )
    results: list[dict[str, Any]] = []
    skipped: list[int] = []
    dropped_tail = False
    last = len(lines) - 1
    for position, (number, line) in enumerate(lines[1:], start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if position == last:
                dropped_tail = True
            else:
                skipped.append(number)
            continue
        if isinstance(record, dict) and isinstance(record.get(PAGE_KEY), int):
            results.append(record)
    if skipped:
        logger.warning("%s: skipped unparsable line(s) %s", path, skipped)
    return WorkingLog(header, results, dropped_tail, tuple(skipped))
