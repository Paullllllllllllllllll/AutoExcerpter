"""Output and working paths of one item.

For an item ``X`` written into a folder, the outputs are
``X_transcription.md`` (or ``.txt``), ``X_summary.md`` and ``X_summary.docx``;
the working logs ``transcription.jsonl`` and ``summary.jsonl`` live in
``X_autoexcerpter/``. When the longest of these paths would exceed the Windows
path limit, the item name is shortened and suffixed with a hash of the full
name, the same way for every file of the item.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

TRANSCRIPTION_SUFFIX = "_transcription"
SUMMARY_SUFFIX = "_summary"
WORKING_SUFFIX = "_autoexcerpter"
TRANSCRIPTION_LOG = "transcription.jsonl"
SUMMARY_LOG = "summary.jsonl"
TRANSCRIPTION_FORMATS = ("md", "txt")

# MAX_PATH (260) less the terminating null character.
WINDOWS_MAX_PATH = 259
# Room for the temp-file suffixes written beside a file while it is replaced.
TEMP_SUFFIX_RESERVE = 32
HASH_LENGTH = 8

_LONGEST_SUFFIX = max(
    *(len(f"{TRANSCRIPTION_SUFFIX}.{fmt}") for fmt in TRANSCRIPTION_FORMATS),
    len(f"{SUMMARY_SUFFIX}.docx"),
    len(WORKING_SUFFIX) + 1 + max(len(TRANSCRIPTION_LOG), len(SUMMARY_LOG)),
)


def name_hash(name: str) -> str:
    """Return the short hash that keeps a shortened item name unique."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def item_base(name: str, output_dir: Path) -> str:
    """Return the file-name base of item *name* in *output_dir*.

    The base is *name* unless the item's longest path would exceed the
    Windows path limit; then it is the name cut to fit, plus ``-`` and
    :func:`name_hash`.
    """
    limit = WINDOWS_MAX_PATH - TEMP_SUFFIX_RESERVE
    prefix = len(str(output_dir)) + 1
    if prefix + len(name) + _LONGEST_SUFFIX <= limit:
        return name
    digest = name_hash(name)
    room = limit - prefix - _LONGEST_SUFFIX - HASH_LENGTH - 1
    truncated = name[:room].rstrip(".-_ ") if room > 0 else ""
    return f"{truncated}-{digest}" if truncated else digest


@dataclass(frozen=True)
class ItemPaths:
    """Every output and working path of one item, derived in one place."""

    output_dir: Path
    working_dir: Path
    transcription: Path
    summary_docx: Path
    summary_md: Path
    transcription_log: Path
    summary_log: Path

    @classmethod
    def for_item(
        cls, name: str, output_dir: Path, transcription_format: str = "md"
    ) -> ItemPaths:
        """Return the paths of the item *name* written into *output_dir*.

        *transcription_format* is ``md`` or ``txt``, the transcription
        file's extension.
        """
        if transcription_format not in TRANSCRIPTION_FORMATS:
            raise ValueError(f"unknown transcription format {transcription_format!r}")
        base = item_base(name, output_dir)
        working_dir = output_dir / f"{base}{WORKING_SUFFIX}"
        return cls(
            output_dir=output_dir,
            working_dir=working_dir,
            transcription=output_dir
            / f"{base}{TRANSCRIPTION_SUFFIX}.{transcription_format}",
            summary_docx=output_dir / f"{base}{SUMMARY_SUFFIX}.docx",
            summary_md=output_dir / f"{base}{SUMMARY_SUFFIX}.md",
            transcription_log=working_dir / TRANSCRIPTION_LOG,
            summary_log=working_dir / SUMMARY_LOG,
        )


__all__ = [
    "HASH_LENGTH",
    "SUMMARY_LOG",
    "TRANSCRIPTION_FORMATS",
    "TRANSCRIPTION_LOG",
    "WINDOWS_MAX_PATH",
    "ItemPaths",
    "item_base",
    "name_hash",
]
