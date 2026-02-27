"""Resume-aware processing utilities for AutoExcerpter.

This module provides the ResumeChecker class that determines whether a given
input item (PDF or image folder) has already been processed, enabling skip/resume
behavior when re-running the tool on partially or fully processed directories.

Output existence is detected by replicating the deterministic output-path logic
used by ItemTranscriber to locate expected output files without invoking the
full processing pipeline.

Processing states:
    COMPLETE: All expected output files exist (transcription + summaries if enabled).
    TRANSCRIPTION_ONLY: The .txt transcription exists but summary outputs are missing.
    PARTIAL: A transcription JSONL log exists with entries (mid-document crash recovery).
    NONE: No output exists; the item must be processed from scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from modules.logger import setup_logger
from modules.path_utils import create_safe_directory_name, create_safe_log_filename

logger = setup_logger(__name__)


class ProcessingState(Enum):
    """Represents the processing state of an input item."""

    COMPLETE = "complete"
    TRANSCRIPTION_ONLY = "transcription_only"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class ResumeResult:
    """Result of a resume check for a single item."""

    item_name: str
    state: ProcessingState
    output_dir: Path | None = None
    existing_outputs: list[Path] = field(default_factory=list)
    missing_outputs: list[Path] = field(default_factory=list)
    reason: str = ""
    completed_page_indices: set[int] | None = None


class ResumeChecker:
    """Determine whether input items have already been processed.

    The checker replicates the deterministic output-path logic from
    ItemTranscriber to locate potential output files without invoking
    the full processing pipeline.

    Args:
        resume_mode: One of ``"skip"`` or ``"overwrite"``.
        summarize: Whether summarization is enabled.
        output_docx: Whether DOCX summary output is enabled.
        output_markdown: Whether Markdown summary output is enabled.
    """

    def __init__(
        self,
        resume_mode: str,
        summarize: bool = True,
        output_docx: bool = True,
        output_markdown: bool = True,
    ) -> None:
        self.resume_mode = resume_mode
        self.summarize = summarize
        self.output_docx = output_docx
        self.output_markdown = output_markdown

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_skip(self, item_name: str, output_dir: Path) -> ResumeResult:
        """Check if an item should be skipped based on existing output.

        Args:
            item_name: The stem name of the item (e.g., PDF stem or folder name).
            output_dir: The resolved output directory for this item.

        Returns:
            A :class:`ResumeResult` describing the item's state.
        """
        if self.resume_mode == "overwrite":
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.NONE,
                output_dir=output_dir,
                reason="overwrite mode",
            )

        return self._check_item(item_name, output_dir)

    def filter_items(
        self,
        items: list[Any],
        output_dir_func: Callable[[Any], Path],
        name_func: Callable[[Any], str],
    ) -> tuple[list[Any], list[ResumeResult]]:
        """Partition items into those that need processing and those to skip.

        Args:
            items: Input item objects.
            output_dir_func: Callable that returns the output directory for an item.
            name_func: Callable that returns the stem name for an item.

        Returns:
            Tuple of ``(to_process, skipped)`` where *skipped* contains
            :class:`ResumeResult` entries for items that were filtered out.
        """
        if self.resume_mode == "overwrite":
            return list(items), []

        to_process: list[Any] = []
        skipped: list[ResumeResult] = []

        for item in items:
            result = self.should_skip(name_func(item), output_dir_func(item))
            if result.state == ProcessingState.COMPLETE:
                skipped.append(result)
            else:
                to_process.append(item)

        return to_process, skipped

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_item(self, item_name: str, output_dir: Path) -> ResumeResult:
        """Check the processing state of an item by examining output files."""
        existing: list[Path] = []
        missing: list[Path] = []

        # Primary output: transcription .txt
        txt_path = output_dir / f"{item_name}.txt"
        if txt_path.exists() and txt_path.stat().st_size > 0:
            existing.append(txt_path)
        else:
            missing.append(txt_path)

        # Summary outputs (only checked if summarization is enabled)
        if self.summarize:
            if self.output_docx:
                docx_path = output_dir / f"{item_name}.docx"
                if docx_path.exists() and docx_path.stat().st_size > 0:
                    existing.append(docx_path)
                else:
                    missing.append(docx_path)

            if self.output_markdown:
                md_path = output_dir / f"{item_name}.md"
                if md_path.exists() and md_path.stat().st_size > 0:
                    existing.append(md_path)
                else:
                    missing.append(md_path)

        # Determine state
        if not missing:
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.COMPLETE,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=f"all outputs exist: {', '.join(p.name for p in existing)}",
            )

        # Check if transcription exists but summaries are missing
        txt_exists = any(p.suffix == ".txt" for p in existing)
        if txt_exists and self.summarize and missing:
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.TRANSCRIPTION_ONLY,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=f"transcription exists, missing: {', '.join(p.name for p in missing)}",
            )

        # Check for partial processing (JSONL log exists in working directory)
        completed_pages = self._check_partial_log(item_name, output_dir)
        if completed_pages is not None:
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.PARTIAL,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=f"partial log with {len(completed_pages)} completed page(s)",
                completed_page_indices=completed_pages,
            )

        return ResumeResult(
            item_name=item_name,
            state=ProcessingState.NONE,
            output_dir=output_dir,
            existing_outputs=existing,
            missing_outputs=missing,
            reason="no output",
        )

    def _check_partial_log(self, item_name: str, output_dir: Path) -> set[int] | None:
        """Check if a partial transcription log exists with completed pages.

        Returns:
            Set of completed page indices if a partial log exists, None otherwise.
        """
        safe_working_dir_name = create_safe_directory_name(item_name, "_working_files")
        working_dir = output_dir / safe_working_dir_name

        if not working_dir.exists():
            return None

        safe_log_name = create_safe_log_filename(item_name, "transcription")
        log_path = working_dir / safe_log_name

        if not log_path.exists() or log_path.stat().st_size == 0:
            return None

        return load_completed_pages(log_path)


def load_completed_pages(log_path: Path) -> set[int] | None:
    """Parse a transcription JSONL log and return indices of successfully completed pages.

    The log format is a JSON array where the first element is the header/metadata
    and subsequent elements are per-page transcription results.

    Args:
        log_path: Path to the transcription log file.

    Returns:
        Set of ``original_input_order_index`` values for successful pages,
        or None if the log cannot be parsed.
    """
    try:
        raw = log_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None

        # Handle potentially incomplete JSON arrays (crash before finalize)
        # Try parsing as-is first, then try fixing incomplete arrays
        entries = _parse_log_entries(raw)
        if entries is None:
            return None

        completed: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Skip the header entry (has "input_item_name" but no "original_input_order_index")
            if "original_input_order_index" not in entry:
                continue
            # Only count entries without errors
            if "error" not in entry:
                idx = entry.get("original_input_order_index")
                if isinstance(idx, int):
                    completed.add(idx)

        return completed if completed else None

    except Exception as exc:
        logger.warning("Could not parse transcription log %s: %s", log_path, exc)
        return None


def _parse_log_entries(raw: str) -> list[dict] | None:
    """Parse log file content, handling both complete and incomplete JSON arrays.

    Args:
        raw: Raw file content.

    Returns:
        List of parsed entries, or None if parsing fails.
    """
    # Try standard JSON parse first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        pass

    # Try fixing incomplete array (crash before finalize_log_file added closing bracket)
    # Common cases: "[header,entry1,entry2" or "[header,entry1,entry2,"
    cleaned = raw.rstrip().rstrip(",")
    if not cleaned.endswith("]"):
        cleaned += "\n]"
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            logger.debug("Recovered incomplete JSON log array")
            return data
    except json.JSONDecodeError:
        pass

    return None


def load_transcription_results_from_log(log_path: Path) -> list[dict[str, Any]] | None:
    """Load full transcription results from a log file for summary-only reprocessing.

    Args:
        log_path: Path to the transcription log file.

    Returns:
        List of transcription result dictionaries (excluding the header),
        or None if the log cannot be parsed.
    """
    try:
        raw = log_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None

        entries = _parse_log_entries(raw)
        if entries is None:
            return None

        results: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Skip header entries
            if "original_input_order_index" not in entry:
                continue
            results.append(entry)

        return results if results else None

    except Exception as exc:
        logger.warning(
            "Could not load transcription results from %s: %s", log_path, exc
        )
        return None


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "ProcessingState",
    "ResumeResult",
    "ResumeChecker",
    "load_completed_pages",
    "load_transcription_results_from_log",
]
