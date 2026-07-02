"""Resume-aware processing utilities for AutoExcerpter.

This module provides the ResumeChecker class that determines whether a given
input item (PDF or image folder) has already been processed, enabling skip/resume
behavior when re-running the tool on partially or fully processed directories.

Output existence is detected by replicating the deterministic output-path logic
used by pipeline.transcriber.ItemTranscriber to locate expected output files
without invoking the full processing pipeline.

Processing states:
    COMPLETE: All expected output files exist (transcription + summaries if enabled).
    TRANSCRIPTION_ONLY: The .txt transcription exists but summary outputs are missing.
    PARTIAL: A transcription JSONL log exists with entries
        (mid-document crash recovery).
    NONE: No output exists; the item must be processed from scratch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from config.constants import LOG_FORMAT_VERSION
from config.logger import setup_logger
from pipeline.paths import create_safe_directory_name, create_safe_log_filename

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
    logged_model_name: str | None = None


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
        retranscribe: bool = False,
    ) -> None:
        self.resume_mode = resume_mode
        self.summarize = summarize
        self.output_docx = output_docx
        self.output_markdown = output_markdown
        # When True, logged transcriptions are NOT reused: resumable items are
        # re-transcribed from scratch (``--retranscribe``).
        self.retranscribe = retranscribe

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

        # Read the working-log state once: completed page indices, the logged
        # model name, and the expected page count from the header. Used both for
        # the completeness contract (never mark COMPLETE with pages missing) and
        # for summary-only reuse.
        log_path = self._transcription_log_path(item_name, output_dir)
        completed_pages = load_completed_pages(log_path) if log_path else None
        header = load_log_header(log_path) if log_path else None
        logged_model = header.get("model_name") if isinstance(header, dict) else None
        expected_total = (
            header.get("total_images") if isinstance(header, dict) else None
        )
        # Count ALL logged page entries (successful and errored). An error page
        # was attempted and logged, so it counts as accounted-for; only pages
        # that never reached the log at all (e.g. deferred by a token-budget
        # stall) constitute a shortfall.
        logged_results = (
            load_transcription_results_from_log(log_path) if log_path else None
        )
        logged_count = len(logged_results) if logged_results else 0

        # Completeness gate: outputs may exist yet be missing pages (a stalled
        # token budget can write complete-looking outputs). If the log proves a
        # shortfall, refuse COMPLETE and resume the missing pages. When no log
        # is available we cannot prove a shortfall, so outputs are trusted.
        log_shortfall = (
            log_path is not None
            and isinstance(expected_total, int)
            and logged_count < expected_total
        )

        # Determine state
        if not missing and not log_shortfall:
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.COMPLETE,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=f"all outputs exist: {', '.join(p.name for p in existing)}",
                logged_model_name=logged_model,
            )

        reuse_pages = None if self.retranscribe else completed_pages

        # Transcription exists but summaries are missing (or pages are short of
        # the expected total): reuse the logged transcriptions and (re)generate
        # summaries only, unless --retranscribe forces a fresh pass.
        txt_exists = any(p.suffix == ".txt" for p in existing)
        if txt_exists and self.summarize and (missing or log_shortfall):
            reason = "transcription exists, missing: " + ", ".join(
                p.name for p in missing
            )
            if log_shortfall:
                reason += (
                    f" (log shows {len(completed_pages or [])}/{expected_total} "
                    "pages; resuming missing pages)"
                )
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.TRANSCRIPTION_ONLY,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=reason,
                completed_page_indices=reuse_pages,
                logged_model_name=logged_model,
            )

        # Check for partial processing (JSONL log exists in working directory)
        if completed_pages is not None:
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.PARTIAL,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=f"partial log with {len(completed_pages)} completed page(s)",
                completed_page_indices=reuse_pages,
                logged_model_name=logged_model,
            )

        return ResumeResult(
            item_name=item_name,
            state=ProcessingState.NONE,
            output_dir=output_dir,
            existing_outputs=existing,
            missing_outputs=missing,
            reason="no output",
        )

    def _transcription_log_path(self, item_name: str, output_dir: Path) -> Path | None:
        """Return the transcription log path for an item, or None if absent."""
        safe_working_dir_name = create_safe_directory_name(item_name, "_working_files")
        working_dir = output_dir / safe_working_dir_name

        if not working_dir.exists():
            return None

        safe_log_name = create_safe_log_filename(item_name, "transcription")
        log_path = working_dir / safe_log_name

        if not log_path.exists() or log_path.stat().st_size == 0:
            return None

        return log_path


def load_completed_pages(log_path: Path) -> set[int] | None:
    """Parse a transcription JSONL log and return indices of completed pages.

    The log is JSONL: line 1 is the versioned header, each later line is a
    per-page result. Logs lacking the current format marker are refused.

    Args:
        log_path: Path to the transcription log file.

    Returns:
        Set of ``original_input_order_index`` values for successful pages,
        or None if the log cannot be parsed or is unversioned.
    """
    try:
        raw = log_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None

        entries = _parse_log_entries(raw)
        if entries is None:
            return None

        completed: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Skip the header entry (no per-page order index).
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


def _parse_log_entries(raw: str) -> list[dict[str, Any]] | None:
    """Parse JSONL working-log content with version enforcement.

    The first non-blank line must be a header carrying the current
    ``_format_version`` marker; otherwise the log is refused (no migration).
    A truncated final line (crash mid-write) is dropped.

    Args:
        raw: Raw file content.

    Returns:
        List of parsed objects (header first), or None on refusal/parse
        failure.
    """
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return None

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError:
        return None

    if (
        not isinstance(header, dict)
        or header.get("_format_version") != LOG_FORMAT_VERSION
    ):
        logger.warning(
            "Refusing resume of working log with an unrecognized format "
            "(missing or outdated _format_version marker). Re-run from scratch "
            "or finish it with the version that wrote it."
        )
        return None

    entries: list[dict[str, Any]] = [header]
    last_index = len(lines) - 1
    for i, line in enumerate(lines[1:], start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if i == last_index:
                # Expected: a crash truncated the final line. Drop it.
                logger.debug("Dropped trailing truncated log line")
            else:
                logger.warning("Skipping unparsable interior log line %d", i)
            continue
        if isinstance(obj, dict):
            entries.append(obj)

    return entries


def load_log_header(log_path: Path) -> dict[str, Any] | None:
    """Return the versioned header object of a JSONL working log, or None."""
    try:
        raw = log_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        entries = _parse_log_entries(raw)
        if not entries:
            return None
        header = entries[0]
        if isinstance(header, dict) and "_format_version" in header:
            return header
        return None
    except Exception as exc:
        logger.warning("Could not read log header %s: %s", log_path, exc)
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
    "load_log_header",
    "load_transcription_results_from_log",
]
