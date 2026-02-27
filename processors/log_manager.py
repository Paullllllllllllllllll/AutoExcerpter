"""JSON log file management for AutoExcerpter processing runs."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.concurrency_helper import (
    get_api_concurrency,
    get_api_timeout,
    get_service_tier,
)
from modules.logger import setup_logger

logger = setup_logger(__name__)

_LOG_HANDLES: dict[Path, tuple[Any, threading.Lock]] = {}
_LOG_HANDLES_GUARD = threading.Lock()


def _get_log_handle(log_path: Path):
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
    try:
        handle.close()
    except OSError:
        pass


def initialize_log_file(
    log_path: Path,
    item_name: str,
    input_path: str,
    input_type: str,
    total_images: int,
    model_name: str,
    extraction_dpi: int | None = None,
    concurrency_limit: int | None = None,
) -> bool:
    """Create the per-item log file header as the start of a JSON array."""
    # Determine if this is an OpenAI model for flex processing metadata
    is_openai_model = model_name.startswith(("gpt-", "o1", "o3", "o4"))
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
        "input_item_name": item_name,
        "input_item_path": input_path,
        "input_type": input_type,
        "processing_start_time": datetime.now().isoformat(),
        "total_images": total_images,
        "configuration": configuration,
    }

    try:
        _close_log_handle(log_path)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("[\n")  # Start JSON array
            json.dump(payload, log_file)
        return True
    except OSError as exc:
        logger.warning("Failed to initialize log file %s: %s", log_path, exc)
        return False


def append_to_log(log_path: Path, entry: dict[str, Any]) -> bool:
    """Append a JSON entry to the log file array (comma-separated)."""
    try:
        log_file, lock = _get_log_handle(log_path)
        with lock:
            log_file.write(",\n")  # Add comma separator
            json.dump(entry, log_file)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to write to log file %s: %s", log_path, exc)
        return False


def finalize_log_file(log_path: Path) -> bool:
    """Finalize the log file by closing the JSON array."""
    try:
        log_file, lock = _get_log_handle(log_path)
        with lock:
            log_file.write("\n]")  # Close JSON array
        _close_log_handle(log_path)
        return True
    except (OSError, ValueError) as exc:
        logger.warning("Failed to finalize log file %s: %s", log_path, exc)
        return False
