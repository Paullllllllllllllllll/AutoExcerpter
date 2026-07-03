"""User-level state directory resolution and atomic JSON persistence.

The daily token-budget state and the persistent OpenAlex cache live under a
user-level directory (``~/.autoexcerpter`` by default), overridable via
``paths.state_dir`` in ``app.yaml``. A legacy state file in the working
directory is adopted once when the user-level file is absent, so existing
setups keep their accumulated state on first upgrade.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from config.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_DIR_NAME = ".autoexcerpter"

# Number of write attempts before giving up. Windows antivirus / search
# indexers briefly lock the temp or destination file, surfacing transient
# PermissionError / FileNotFoundError on the write or the os.replace; a short
# retry loop rides those out (pattern adopted from ChronoMiner v1.21.0).
_ATOMIC_WRITE_ATTEMPTS = 4
_ATOMIC_WRITE_RETRY_SLEEP_S = 0.05


def get_state_dir() -> Path:
    """Return the resolved state directory, creating it if needed."""
    from config import app as config

    override = getattr(config, "STATE_DIR", "") or ""
    if override:
        state_dir = Path(override).expanduser()
    else:
        state_dir = Path.home() / _DEFAULT_DIR_NAME
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create state dir %s: %s", state_dir, exc)
    return state_dir


def resolve_state_file(filename: str, legacy_path: Path | None = None) -> Path:
    """Resolve *filename* inside the state dir, adopting a legacy file once.

    If the user-level file is absent but *legacy_path* exists, the legacy file
    is copied into the state dir (one-time adoption) so prior state survives the
    move out of the working directory.
    """
    target = get_state_dir() / filename
    if not target.exists() and legacy_path is not None and legacy_path.exists():
        try:
            shutil.copy2(legacy_path, target)
            logger.info("Adopted legacy state file %s -> %s", legacy_path, target)
        except OSError as exc:
            logger.warning("Could not adopt legacy state file %s: %s", legacy_path, exc)
    return target


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

    The temp file carries a per-process-unique suffix (pid + random token) so
    concurrent processes writing the same destination never collide on the temp
    path. Transient ``PermissionError`` / ``FileNotFoundError`` races (Windows
    AV / indexer holding the file briefly) are retried a few times before the
    write is abandoned with a warning. The temp file is always cleaned up.
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
                # Transient on Windows; back off briefly and retry.
                last_exc = exc
                if attempt < _ATOMIC_WRITE_ATTEMPTS - 1:
                    time.sleep(_ATOMIC_WRITE_RETRY_SLEEP_S)
            except OSError as exc:
                # Non-transient (e.g. bad path): stop retrying.
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
    "get_state_dir",
    "resolve_state_file",
    "read_json",
    "write_json_atomic",
]
