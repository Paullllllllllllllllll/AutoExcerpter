"""The working logs of an item: versioned JSONL written through common/jsonl.

The first line of a log is a header carrying ``_format_version``; every
following line is one page result. A crash mid-write can at most truncate the
last line, which readers drop. Logs without the current version are refused
on resume (no migration); see :data:`autoexcerpter.constants.LOG_FORMAT_VERSION`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from autoexcerpter.common.jsonl import (
    JsonlFormat,
    JsonlFormatError,
    JsonlWriter,
    read_jsonl,
    write_jsonl,
)
from autoexcerpter.constants import LOG_FORMAT_VERSION, OPENAI_MODEL_PREFIXES

logger = logging.getLogger(__name__)

LOG_FORMAT = JsonlFormat("_format_version", LOG_FORMAT_VERSION)
PAGE_KEY = "original_input_order_index"

# Header values recorded when the caller passes none (the code defaults).
DEFAULT_LOG_CONCURRENCY = 16
DEFAULT_LOG_TIMEOUT = 900
DEFAULT_LOG_SERVICE_TIER = "flex"

_WRITER = JsonlWriter()


@dataclass(frozen=True)
class WorkingLog:
    """A parsed working log: its header and its page results in file order."""

    header: dict[str, Any]
    results: list[dict[str, Any]]

    def completed_pages(self) -> set[int] | None:
        """Return the indices of error-free pages; None when there are none."""
        completed = {
            entry[PAGE_KEY]
            for entry in self.results
            if "error" not in entry and isinstance(entry.get(PAGE_KEY), int)
        }
        return completed or None


def read_log(log_path: Path) -> WorkingLog | None:
    """Read a working log; None when it is missing, empty or refused.

    Lines without a page index (a second header, stray objects) are not
    page results and are left out.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size == 0:
            return None
        content = read_jsonl(log_path, LOG_FORMAT)
    except JsonlFormatError as exc:
        logger.warning(
            "Refusing resume of working log %s with an unrecognized format (%s). "
            "Re-run from scratch or finish it with the version that wrote it.",
            log_path,
            exc,
        )
        return None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read working log %s: %s", log_path, exc)
        return None
    if content is None:
        return None
    results = [record for record in content.records if PAGE_KEY in record]
    return WorkingLog(content.header, results)


@dataclass(frozen=True)
class LogHeader:
    """The run values recorded in a working log's header.

    A None value records the code default (``"N/A"`` for *extraction_dpi*);
    *service_tier* is recorded for OpenAI models only.
    """

    item_name: str
    input_path: str
    input_type: str
    total_images: int
    model_name: str
    extraction_dpi: int | str | None = None
    concurrency_limit: int | None = None
    file_provenance: dict[str, Any] | None = None
    log_type: str = "transcription"
    api_timeout: float | None = None
    service_tier: str | None = None


def log_header(fields: LogHeader) -> dict[str, Any]:
    """Return the header of a working log (the version is added on write)."""
    is_openai_model = fields.model_name.startswith(OPENAI_MODEL_PREFIXES)
    configuration = {
        "concurrent_requests": (
            fields.concurrency_limit
            if fields.concurrency_limit is not None
            else DEFAULT_LOG_CONCURRENCY
        ),
        "api_timeout_seconds": (
            fields.api_timeout
            if fields.api_timeout is not None
            else DEFAULT_LOG_TIMEOUT
        ),
        "model_name": fields.model_name,
        "extraction_dpi": (
            fields.extraction_dpi if fields.extraction_dpi is not None else "N/A"
        ),
        "service_tier": (
            (fields.service_tier or DEFAULT_LOG_SERVICE_TIER)
            if is_openai_model
            else "N/A"
        ),
    }
    header: dict[str, Any] = {
        "log_type": fields.log_type,
        "input_item_name": fields.item_name,
        "input_item_path": fields.input_path,
        "input_type": fields.input_type,
        "processing_start_time": datetime.now().isoformat(),
        "total_images": fields.total_images,
        "model_name": fields.model_name,
        "configuration": configuration,
    }
    if fields.file_provenance is not None:
        header["file_provenance"] = fields.file_provenance
    return header


def initialize_log_file(
    log_path: Path,
    fields: LogHeader,
    entries: Sequence[dict[str, Any]] = (),
) -> bool:
    """Replace the log with its header and *entries*; False when the write failed.

    Header and entries are written in one atomic file replace, so a failure
    keeps the previous log and its completed pages, and a crash leaves either
    the old log or the new one. Later appends to a log written with entries
    open the file per append instead of caching a handle.
    """
    if not entries:
        return _WRITER.start(log_path, LOG_FORMAT, log_header(fields))
    _WRITER.close(log_path)
    try:
        write_jsonl(log_path, LOG_FORMAT, log_header(fields), entries)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to rewrite working log %s: %s", log_path, exc)
        return False
    return True


def initialize_log_or_raise(
    log_path: Path,
    fields: LogHeader,
    entries: Sequence[dict[str, Any]] = (),
) -> None:
    """Write the log header and *entries*, retrying once; raise if it fails.

    A headerless log is refused wholesale by a later resume, which would make
    the item's completed pages unrecoverable; raising lets the caller fail the
    item cleanly while the previous working log is preserved.
    """
    for _attempt in range(2):
        if initialize_log_file(log_path, fields, entries):
            return
    raise RuntimeError(
        f"Could not initialize {fields.log_type} log header at {log_path} "
        "after a retry; aborting this item to protect the working log."
    )


def append_to_log(log_path: Path, entry: dict[str, Any]) -> bool:
    """Append one page result as one line; False when the write failed."""
    return _WRITER.append(log_path, entry)


def finalize_log_file(log_path: Path) -> bool:
    """Release the log's cached handle; a later append opens the file anew.

    Never creates a file, so it is a no-op for a log never initialized.
    """
    _WRITER.close(log_path)
    return True


def release_logs() -> None:
    """Close every cached log handle; later appends open their files anew."""
    _WRITER.close_all()


__all__ = [
    "LOG_FORMAT",
    "LogHeader",
    "WorkingLog",
    "append_to_log",
    "finalize_log_file",
    "initialize_log_file",
    "initialize_log_or_raise",
    "log_header",
    "read_log",
    "release_logs",
]
