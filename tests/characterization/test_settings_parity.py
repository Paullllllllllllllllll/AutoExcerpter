"""Settings resolve the same through the shared settings schema.

``settings_before_schema`` is ``settings.py`` as it stood before the shared
blocks moved into ``common/settings_schema.py``. Every settings text below
and each file named in ``CHRONO_SETTINGS_PARITY`` (paths separated by
``os.pathsep``) load through both modules with the extension's blocks; the
resolved values, or the error messages, must agree. A difference is reported
by key path only, never by value, so files with private values can be
checked.

Version 3.0 accepts the blocks of :data:`NEW_BLOCKS`, which the earlier
module refused; the texts that set them, the packaged example (which shows
them) and the new fields of the code defaults are left out of the
comparison and tested in ``tests/core/test_settings.py``.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from autoexcerpter import ext
from autoexcerpter import settings as current
from autoexcerpter.common.settings_loader import SettingsError
from tests.characterization import settings_before_schema as before

PARITY_ENV = "CHRONO_SETTINGS_PARITY"

TEXTS = (
    "",
    "api_keys:\n  openai: MY_KEY\n  google: ''\n",
    "api_keys:\n  mistral: K\n",
    "api_keys: [openai]\n",
    "endpoints:\n  local:\n    base_url: http://127.0.0.1:8000/v1/\n"
    "    api_key_env: LOCAL_KEY\n    supports_vision: false\n"
    "    supports_schema: true\n",
    "endpoints:\n  local:\n    base_url: http://x/\n",
    "endpoints:\n  local:\n    base_url: http://x/\n    api_key_env: K\n    model: m\n",
    "endpoints:\n  local: 5\n",
    "endpoints:\n  local:\n    base_url: http://x/\n    api_key_env: K\n"
    "    supports_schema: maybe\n",
    "timeouts:\n  request: 300\n  write: 240\n",
    "timeouts:\n  request: fast\n",
    "timeouts:\n  connect: -1\n",
    "timeouts:\n  read: 5\n",
    "rate_limits:\n  - [120, 1]\n  - [15000, 60]\n",
    "rate_limits: []\n",
    "rate_limits:\n  - [10, 0]\n",
    "rate_limits:\n  - [10, true]\n",
    "retry:\n  max_attempts: 5\n",
    "retry:\n  max_attempts: 0\n",
    "retry:\n  max_attempts: -2\n",
    "retry:\n  timeout_attempts: 2\n  validation_attempts: 3\n"
    "  min_input_tokens: 50\n  page_timeout: 120\n",
    "retry:\n  page_timeout: 'off'\n",
    "retry:\n  page_timeout: false\n",
    "retry:\n  page_timeout: soon\n",
    "retry:\n  validation_attempts: 0\n",
    "retry:\n  backoff_base: 1.0\n  backoff_cap: 60\n"
    "  backoff_multipliers:\n    timeout: 3\n  jitter:\n    min: 0\n    max: 2\n",
    "retry:\n  backoff_multipliers:\n    sideways: 1\n",
    "retry:\n  jitter:\n    mid: 1\n",
    "retry:\n  schema_retries:\n    summary:\n      validation_failure:\n"
    "        enabled: false\n        max_attempts: 1\n",
    "retry:\n  schema_retries:\n    summary:\n      validation_failure:\n"
    "        enabled: maybe\n",
    "retry:\n  schema_retries:\n    review: {}\n",
    "retry:\n  attempts: 3\n",
    "concurrency:\n  image_processing: 2\n  delay_between_tasks: 0.05\n",
    "concurrency:\n  audio: 3\n",
    "concurrency:\n  audio: 0\n",
    "concurrency:\n  image_processing: 0\n  delay_between_tasks: -1\n",
    "concurrency:\n  audio: 0\n  delay_between_tasks: -1\n",
    "concurrency:\n  transcription: 3\n",
    "images:\n  original_max_side_px: 2500\n",
    "images:\n  original_max_side_px: null\n",
    "images: {}\n",
    "images:\n  original_max_side_px: 0\n",
    "images:\n  original_max_side_px: wide\n",
    "images:\n  max_side_px: 3000\n",
    "prompt_caching:\n  enabled: true\n  anthropic:\n    ttl: 1h\n"
    "  openai:\n    prompt_cache_retention: 24h\n"
    "  openrouter:\n    anthropic_cache_control: false\n",
    "prompt_caching:\n  openai:\n    prompt_cache_retention: null\n",
    "prompt_caching:\n  openai:\n    prompt_cache_retention: ''\n",
    "prompt_caching:\n  anthropic:\n    ttl: 2h\n",
    "prompt_caching:\n  google: {}\n",
    "state_dir: ~/state\n",
    "state_dir: ''\n",
    "state_dir: 5\n",
    "logs_dir: logs\n",
    "logs_dir: run.log\n",
    "logs_dir: 5\n",
    "openalex:\n  email: someone@example.org\n  max_requests: 10\n",
    "openalex:\n  max_requests: -1\n",
    "tools:\n  tesseract_cmd: tesseract\n  ffmpeg_cmd: bin/ffmpeg\n",
    "whisper:\n  device: tpu\n",
    "audio_chunking:\n  chunk_format: wav16\n",
    "audio_api:\n  languages: [en, 5]\n",
    "defaults:\n  concurrency: 4\n",
    "unknown: 1\n",
    "[1, 2]\n",
)


NEW_BLOCKS = ("concurrency", "logs_dir", "prompt_caching")
"""Top-level blocks that version 3.0 added to the accepted settings."""

COMPARED = tuple(
    text for text in TEXTS if not text.startswith(tuple(f"{b}:" for b in NEW_BLOCKS))
)


def _extra_files() -> list[Path]:
    raw = os.environ.get(PARITY_ENV, "")
    return [Path(item) for item in raw.split(os.pathsep) if item.strip()]


def _normalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            f.name: _normalize(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
        return {"__type__": type(value).__name__, **fields}
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _outcome(module: ModuleType, path: Path) -> tuple[str, Any]:
    try:
        loaded = module.load(path, blocks=ext.settings_blocks())
    except SettingsError as exc:
        return "error", str(exc)
    return "ok", _without_new_blocks(_normalize(loaded))


def _without_new_blocks(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__type__") == "Settings":
        return {key: item for key, item in value.items() if key not in NEW_BLOCKS}
    return value


def _differences(old: Any, new: Any, path: str = "") -> list[str]:
    """Return the key paths at which *old* and *new* differ; no values."""
    if isinstance(old, dict) and isinstance(new, dict):
        found: list[str] = []
        for key in sorted(set(old) | set(new)):
            where = f"{path}.{key}" if path else key
            if key not in old or key not in new:
                found.append(f"{where} (missing on one side)")
            else:
                found.extend(_differences(old[key], new[key], where))
        return found
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        found = []
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            found.extend(_differences(left, right, f"{path}[{index}]"))
        return found
    return [] if old == new else [path or "(top level)"]


def _compare(path: Path) -> list[str]:
    old_kind, old_value = _outcome(before, path)
    new_kind, new_value = _outcome(current, path)
    if old_kind != new_kind:
        return [f"old {old_kind}, new {new_kind}"]
    if old_kind == "error":
        return [] if old_value == new_value else ["error message differs"]
    return _differences(old_value, new_value)


@pytest.mark.parametrize("index", range(len(COMPARED)))
def test_settings_texts_resolve_alike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "settings.yaml"
    path.write_text(COMPARED[index], encoding="utf-8")
    assert _compare(path) == []


@pytest.mark.parametrize(
    "path", _extra_files(), ids=lambda path: f"file-{Path(path).name}"
)
def test_named_files_resolve_alike(path: Path) -> None:
    assert path.is_file()
    assert _compare(path) == []


def test_code_defaults_alike() -> None:
    old = _normalize(before.Settings())
    new = _without_new_blocks(_normalize(current.Settings()))
    assert _differences(old, new) == []
