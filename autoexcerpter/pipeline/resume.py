"""Resume classification: what an earlier run left for an item.

:class:`ResumeChecker` finds an item's outputs through
:class:`~autoexcerpter.pipeline.paths.ItemPaths` and reads its working logs
through :mod:`autoexcerpter.pipeline.log`, without running the pipeline.

Processing states:
    COMPLETE: Every expected output exists and the log shows no gap or failure.
    TRANSCRIPTION_ONLY: The transcription exists but summary outputs are
        missing, or the log shows missing or failed pages.
    PARTIAL: A transcription log holds completed pages (crash recovery).
    NONE: Nothing to reuse; the item is processed from scratch.

With the database check on, an item whose files are complete but whose
database has no complete ``documents`` row is finished from its working logs,
so the run rewrites its outputs and its rows without a model call.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from autoexcerpter.common.images import IMAGE_EXTENSIONS
from autoexcerpter.common.jsonl import input_changed, read_header
from autoexcerpter.common.native import image_settings_fingerprint
from autoexcerpter.pipeline.log import LOG_FORMAT, PAGE_KEY, WorkingLog, read_log
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.rendering.sqlite import DATABASE_NAME, complete_documents

logger = logging.getLogger(__name__)


class ProcessingState(Enum):
    """Represents the processing state of an input item."""

    COMPLETE = "complete"
    TRANSCRIPTION_ONLY = "transcription_only"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class ResumeResult:
    """Result of a resume check for a single item.

    The parsed working-log data is kept so the item processor need not read
    the logs again; a field left None falls back to a disk read. *warning* is
    a note for the user about this item, or None.
    """

    item_name: str
    state: ProcessingState
    output_dir: Path | None = None
    existing_outputs: list[Path] = field(default_factory=list)
    missing_outputs: list[Path] = field(default_factory=list)
    reason: str = ""
    completed_page_indices: set[int] | None = None
    transcription_results: list[dict[str, Any]] | None = None
    summary_results: list[dict[str, Any]] | None = None
    log_header: dict[str, Any] | None = None
    warning: str | None = None


class InputMismatchError(ValueError):
    """An item's working log belongs to another input with the same name."""


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _normalized(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def input_changed_since_log(header: dict[str, Any] | None) -> bool:
    """Whether the input changed on disk since the log's header was written.

    Compares the header's ``file_provenance`` with the input through the
    cheap size and image-set identity of
    :func:`~autoexcerpter.common.jsonl.input_changed`; False when no change
    can be proven.
    """
    if not isinstance(header, dict):
        return False
    return input_changed(header.get("file_provenance"), extensions=IMAGE_EXTENSIONS)


class ResumeChecker:
    """Determine whether input items have already been processed.

    Args:
        resume_mode: One of ``"skip"`` or ``"overwrite"``.
        summarize: Whether summarization is enabled.
        output_docx: Whether DOCX summary output is enabled.
        output_markdown: Whether Markdown summary output is enabled.
        retranscribe: Whether logged transcriptions are transcribed again
            instead of being reused.
        transcription_format: The transcription file's extension.
        check_database: Whether a complete item also needs a complete
            ``documents`` row in its output folder's database.
    """

    def __init__(
        self,
        resume_mode: str,
        summarize: bool = True,
        output_docx: bool = True,
        output_markdown: bool = True,
        retranscribe: bool = False,
        transcription_format: str = "md",
        check_database: bool = False,
    ) -> None:
        self.resume_mode = resume_mode
        self.summarize = summarize
        self.output_docx = output_docx
        self.output_markdown = output_markdown
        self.retranscribe = retranscribe
        self.transcription_format = transcription_format
        self.check_database = check_database
        self._complete_rows: dict[Path, frozenset[str]] = {}

    def should_skip(
        self, item_name: str, output_dir: Path, input_path: Path | None = None
    ) -> ResumeResult:
        """Return the state of item *item_name* written into *output_dir*.

        With *input_path*, raises :class:`InputMismatchError` when the item's
        working log was written for another input that still exists.
        """
        if input_path is not None:
            self._check_same_input(item_name, output_dir, input_path)
        if self.resume_mode == "overwrite":
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.NONE,
                output_dir=output_dir,
                reason="overwrite mode",
            )
        return self._check_item(item_name, output_dir)

    def _check_same_input(
        self, item_name: str, output_dir: Path, input_path: Path
    ) -> None:
        """Raise when the log names another input that still exists on disk.

        A logged input that no longer exists (moved or renamed) is not
        compared, so the item keeps its usual resume.
        """
        paths = ItemPaths.for_item(item_name, output_dir, self.transcription_format)
        header = read_header(paths.transcription_log, LOG_FORMAT)
        logged = header.get("input_item_path") if header is not None else None
        if not isinstance(logged, str) or not logged:
            return
        logged_path = Path(logged)
        if _normalized(logged_path) == _normalized(input_path):
            return
        if logged_path.exists():
            raise InputMismatchError(
                f"'{item_name}' in {output_dir} belongs to {logged_path}, but this "
                f"run would write {input_path} there under the same name. Choose "
                "another --output for one of them."
            )

    def _expected_outputs(self, paths: ItemPaths) -> list[Path]:
        expected = [paths.transcription]
        if self.summarize:
            if self.output_docx:
                expected.append(paths.summary_docx)
            if self.output_markdown:
                expected.append(paths.summary_md)
        return expected

    def _check_item(self, item_name: str, output_dir: Path) -> ResumeResult:
        """Classify an item from its output files and working logs."""
        paths = ItemPaths.for_item(item_name, output_dir, self.transcription_format)
        expected = self._expected_outputs(paths)
        existing = [path for path in expected if _nonempty(path)]
        missing = [path for path in expected if path not in existing]

        log = read_log(paths.transcription_log)
        header = log.header if log is not None else None
        completed_pages = log.completed_pages() if log is not None else None

        # A replaced input under the same name makes its logged pages describe
        # another document: drop page-level reuse and block COMPLETE.
        changed = input_changed_since_log(header)
        if completed_pages is not None and changed:
            logger.warning(
                "Input file changed since the working log was written "
                "(cheap provenance mismatch); refusing page-level resume for "
                "'%s' and reprocessing it from scratch.",
                item_name,
            )
            completed_pages = None

        expected_total = header.get("total_images") if header is not None else None
        # Error pages count as attempted; only pages that never reached the
        # log form a shortfall. Unique indices keep a duplicate entry from
        # masking a missing page.
        logged_results = log.results if log is not None and log.results else None
        logged_count = len(
            {r[PAGE_KEY] for r in logged_results or [] if isinstance(r[PAGE_KEY], int)}
        )
        log_shortfall = (
            log is not None
            and isinstance(expected_total, int)
            and logged_count < expected_total
        )

        # A logged failure must never classify COMPLETE, or a failed page
        # could not be repaired by running again.
        transcription_errors = sum(1 for r in logged_results or [] if "error" in r)
        summary_results = self._load_summary_results(paths.summary_log)
        summary_errors = sum(1 for r in summary_results or [] if "error" in r)
        # Every page, blank and failed ones included, gets a summary entry,
        # so an existing summary log with fewer pages is unfinished.
        summary_count = len(
            {r[PAGE_KEY] for r in summary_results or [] if isinstance(r[PAGE_KEY], int)}
        )
        summary_shortfall = (
            self.summarize
            and paths.summary_log.exists()
            and isinstance(expected_total, int)
            and summary_count < expected_total
        )
        log_has_failures = transcription_errors > 0 or summary_errors > 0
        log_incomplete = (
            log_shortfall or summary_shortfall or log_has_failures or changed
        )

        if not missing and not log_incomplete:
            complete = ResumeResult(
                item_name=item_name,
                state=ProcessingState.COMPLETE,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=f"all outputs exist: {', '.join(p.name for p in existing)}",
            )
            if not self.check_database or self._has_complete_row(item_name, output_dir):
                return complete
            if completed_pages is None:
                complete.warning = (
                    f"'{item_name}' has no complete row in "
                    f"{output_dir / DATABASE_NAME} and no working logs to rebuild "
                    "it from; --force rebuilds it."
                )
                return complete
            # Every page and summary is logged: the run rewrites the outputs
            # and the rows from the logs without a model call.
            return dataclasses.replace(
                complete,
                state=(
                    ProcessingState.TRANSCRIPTION_ONLY
                    if self.summarize
                    else ProcessingState.PARTIAL
                ),
                reason=(
                    f"no complete row for '{item_name}' in {DATABASE_NAME}; "
                    "finishing it from the working logs"
                ),
                completed_page_indices=completed_pages,
                transcription_results=logged_results,
                summary_results=summary_results,
                log_header=header,
            )

        reuse_pages = None if self.retranscribe else completed_pages

        if paths.transcription in existing and self.summarize:
            reason = "transcription exists, missing: " + ", ".join(
                p.name for p in missing
            )
            if log_shortfall:
                reason += (
                    f" (log shows {len(completed_pages or [])}/{expected_total} "
                    "pages; resuming missing pages)"
                )
            if log_has_failures:
                reason += (
                    f" ({transcription_errors} failed transcription page(s), "
                    f"{summary_errors} failed summary page(s); retrying them)"
                )
            if changed:
                reason += " (input changed since log; reprocessing from scratch)"
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.TRANSCRIPTION_ONLY,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=reason,
                completed_page_indices=reuse_pages,
                transcription_results=logged_results,
                summary_results=summary_results,
                log_header=header,
            )

        if completed_pages is not None:
            return ResumeResult(
                item_name=item_name,
                state=ProcessingState.PARTIAL,
                output_dir=output_dir,
                existing_outputs=existing,
                missing_outputs=missing,
                reason=f"partial log with {len(completed_pages)} completed page(s)",
                completed_page_indices=reuse_pages,
                transcription_results=logged_results,
                summary_results=summary_results,
                log_header=header,
            )

        return ResumeResult(
            item_name=item_name,
            state=ProcessingState.NONE,
            output_dir=output_dir,
            existing_outputs=existing,
            missing_outputs=missing,
            reason="no output",
        )

    def _has_complete_row(self, item_name: str, output_dir: Path) -> bool:
        """Whether *output_dir*'s database marks *item_name* complete.

        The database of each folder is read once per checker.
        """
        key = output_dir.resolve()
        names = self._complete_rows.get(key)
        if names is None:
            names = complete_documents(output_dir)
            self._complete_rows[key] = names
        return item_name in names

    def _load_summary_results(self, log_path: Path) -> list[dict[str, Any]] | None:
        """Return the summary log's page results; None when not summarizing."""
        if not self.summarize:
            return None
        log = read_log(log_path)
        return log.results if log is not None and log.results else None


def verify_image_settings(header: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject changed or unrecorded preprocessing before logged pages are reused.

    *current* is the resolved image settings of this run.
    """
    recorded = header.get("file_provenance")
    if not isinstance(recorded, dict) or not recorded.get("image_settings_fingerprint"):
        message = (
            "Working log has no file_provenance.image_settings_fingerprint, so "
            "its image settings cannot be verified. Skipping file; use --force "
            "to replace the existing run."
        )
        logger.error(message)
        raise ValueError(message)
    if recorded["image_settings_fingerprint"] == image_settings_fingerprint(current):
        return
    previous = recorded.get("image_config", {})
    changed = sorted(
        key
        for key in previous.keys() | current.keys()
        if previous.get(key) != current.get(key)
    )
    message = (
        f"Image settings changed: {', '.join(changed)}. "
        "Skipping file; use --force to replace the existing run."
    )
    logger.error(message)
    raise ValueError(message)


def load_results(log_path: Path) -> list[dict[str, Any]] | None:
    """Return the page results of a working log; None when it has none."""
    log = read_log(log_path)
    return log.results if log is not None and log.results else None


__all__ = [
    "InputMismatchError",
    "ProcessingState",
    "ResumeChecker",
    "ResumeResult",
    "WorkingLog",
    "input_changed_since_log",
    "load_results",
    "verify_image_settings",
]
