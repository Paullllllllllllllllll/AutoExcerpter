"""User-level state directory and atomic JSON persistence.

The persistent OpenAlex cache lives in the state directory: the settings'
``state_dir`` when given, else ``.autoexcerpter`` in the home folder.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DIR_NAME = ".autoexcerpter"

# Windows antivirus and search indexers briefly lock the temp or destination
# file, so the write or the os.replace can fail transiently.
_ATOMIC_WRITE_ATTEMPTS = 4
_ATOMIC_WRITE_RETRY_SLEEP_S = 0.05


def get_state_dir(state_dir: Path | None = None) -> Path:
    """Return the state directory, creating it if needed.

    *state_dir* None selects ``.autoexcerpter`` in the home folder.
    """
    if state_dir is not None:
        resolved = Path(state_dir).expanduser()
    else:
        resolved = Path.home() / DEFAULT_DIR_NAME
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create state dir %s: %s", resolved, exc)
    return resolved


def resolve_state_file(filename: str, state_dir: Path | None = None) -> Path:
    """Resolve *filename* inside the state directory."""
    return get_state_dir(state_dir) / filename


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from *path*, returning {} on any failure."""
    try:
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        logger.debug("Could not read state JSON %s: %s", path, exc)
        return {}


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as JSON to *path* atomically (temp file + os.replace).

    The temp file name carries the pid and a random token, so concurrent
    processes writing the same destination never share a temp path. Transient
    ``PermissionError`` and ``FileNotFoundError`` are retried before the write
    is abandoned with a warning; the temp file is always removed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create parent dir for %s: %s", path, exc)
        return

    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    last_exc: OSError | None = None
    try:
        for attempt in range(_ATOMIC_WRITE_ATTEMPTS):
            try:
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
                return
            except (PermissionError, FileNotFoundError) as exc:
                last_exc = exc
                if attempt < _ATOMIC_WRITE_ATTEMPTS - 1:
                    time.sleep(_ATOMIC_WRITE_RETRY_SLEEP_S)
            except OSError as exc:
                last_exc = exc
                break
        if last_exc is not None:
            logger.warning("Could not write state JSON %s: %s", path, last_exc)
    finally:
        # A successful os.replace already consumed the temp file; clean up any
        # remnant left by a failed write or replace.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


__all__ = [
    "DEFAULT_DIR_NAME",
    "get_state_dir",
    "read_json",
    "resolve_state_file",
    "write_json_atomic",
]
