"""JSONL log file management for AutoExcerpter processing runs.

Working logs are true JSONL: the first line is a header object carrying a
``_format_version`` marker, and each subsequent line is one complete JSON
object (a per-page result). Because every record is a self-contained line, a
crash mid-write can at most truncate the final line, which the resume parser
drops. Logs lacking the current format marker are refused on resume (no
migration); see :data:`config.constants.LOG_FORMAT_VERSION`.
"""

from __future__ import annotations

import contextlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from config.accessors import (
    get_api_concurrency,
    get_api_timeout,
    get_service_tier,
)
from config.constants import LOG_FORMAT_VERSION, OPENAI_MODEL_PREFIXES
from config.logger import setup_logger

logger = setup_logger(__name__)

_LOG_HANDLES: dict[Path, tuple[Any, threading.Lock]] = {}
_LOG_HANDLES_GUARD = threading.Lock()


def _get_log_handle(log_path: Path) -> tuple[Any, threading.Lock]:
    key = log_path
    with _LOG_HANDLES_GUARD:
        existing = _LOG_HANDLES.get(key)
        if existing is not None:
            return existing
        handle = key.open("a", encoding="utf-8")
        lock = threading.Lock()
        _LOG_HANDLES[key] = (handle, lock)
        return handle, lock


def _close_log_handle(log_path: Path) -> None:
    key = log_path
    with _LOG_HANDLES_GUARD:
        existing = _LOG_HANDLES.pop(key, None)
    if existing is None:
        return
    handle, _lock = existing
    with contextlib.suppress(OSError):
        handle.close()


def initialize_log_file(
    log_path: Path,
    item_name: str,
    input_path: str,
    input_type: str,
    total_images: int,
    model_name: str,
    extraction_dpi: int | None = None,
    concurrency_limit: int | None = None,
    file_provenance: dict[str, Any] | None = None,
    log_type: str = "transcription",
) -> bool:
    """Create the per-item log file header as the first JSONL line."""
    # Determine if this is an OpenAI model for flex processing metadata
    is_openai_model = model_name.startswith(OPENAI_MODEL_PREFIXES)
    default_concurrency, _ = get_api_concurrency()
    service_tier = get_service_tier() if is_openai_model else "N/A"
    configuration = {
        "concurrent_requests": (
            concurrency_limit if concurrency_limit is not None else default_concurrency
        ),
        "api_timeout_seconds": get_api_timeout(),
        "model_name": model_name,
        "extraction_dpi": extraction_dpi if extraction_dpi is not None else "N/A",
        "service_tier": service_tier,
    }

    payload = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": log_type,
        "input_item_name": item_name,
        "input_item_path": input_path,
        "input_type": input_type,
        "processing_start_time": datetime.now().isoformat(),
        "total_images": total_images,
        "model_name": model_name,
        "configuration": configuration,
    }
    if file_provenance is not None:
        payload["file_provenance"] = file_provenance

    try:
        _close_log_handle(log_path)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False))
            log_file.write("\n")
        return True
    except OSError as exc:
        logger.warning("Failed to initialize log file %s: %s", log_path, exc)
        return False


def append_to_log(log_path: Path, entry: dict[str, Any]) -> bool:
    """Append a single JSON object as one JSONL line."""
    try:
        log_file, lock = _get_log_handle(log_path)
        with lock:
            log_file.write(json.dumps(entry, ensure_ascii=False))
            log_file.write("\n")
            # Flush per line: the cached handle is block-buffered, so without
            # this a hard crash (power loss, kill) could drop kilobytes of
            # completed page records — not just the final line the resume
            # parser is designed to tolerate — and resume would re-buy those
            # pages. One flush per page is negligible next to the API call.
            log_file.flush()
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to write to log file %s: %s", log_path, exc)
        return False


def finalize_log_file(log_path: Path) -> bool:
    """Flush and close the log handle.

    With JSONL there is no array to close, so this only releases the cached
    file handle. It never creates a file, so calling it on a log that was
    never initialized is a no-op (no stray artifact is written).
    """
    _close_log_handle(log_path)
    return True
