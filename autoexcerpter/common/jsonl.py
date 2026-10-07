"""Versioned JSONL files for resumable runs.

A file starts with a header object that carries a format version under a
caller-chosen key; every following line is one record. Header and record
fields belong to the caller. Readers tolerate a truncated last line (a crash
mid-write) and refuse files whose header lacks the expected version. The
module also holds input-identity helpers that detect an input changed since
its log was written.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

logger = logging.getLogger(__name__)


class JsonlFormatError(ValueError):
    """A JSONL file lacks a header with the expected format version."""


@dataclass(frozen=True)
class JsonlFormat:
    """The header key that holds the format version, and the current version."""

    version_key: str
    version: int

    def matches(self, header: object) -> bool:
        """Whether *header* is a dict carrying the current version."""
        return isinstance(header, dict) and header.get(self.version_key) == (
            self.version
        )

    def header(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Return a header record: the version first, then *fields*."""
        stored = fields.get(self.version_key, self.version)
        if stored != self.version:
            raise ValueError(
                f"Header field {self.version_key!r} is {stored!r}, "
                f"expected {self.version!r}."
            )
        return {self.version_key: self.version, **fields}


@dataclass(frozen=True)
class JsonlContent:
    """A parsed versioned JSONL file.

    ``dropped_tail`` is True when an unparsable last line was dropped;
    ``skipped_lines`` holds the 1-based numbers of unparsable interior lines.
    """

    header: dict[str, Any]
    records: list[dict[str, Any]]
    dropped_tail: bool = False
    skipped_lines: tuple[int, ...] = ()


def _dumps(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def parse_jsonl(text: str, fmt: JsonlFormat) -> JsonlContent | None:
    """Parse versioned JSONL *text*; None when it holds no lines.

    Raises :class:`JsonlFormatError` when the first line is not a header with
    the current version. Lines that are not JSON objects are skipped.
    """
    lines = _lines(text.removeprefix("\N{BYTE ORDER MARK}"))
    if not lines:
        return None
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise JsonlFormatError(f"First line is not a JSON header: {exc}") from exc
    if not fmt.matches(header):
        found = header.get(fmt.version_key) if isinstance(header, dict) else None
        raise JsonlFormatError(
            f"Expected {fmt.version_key}={fmt.version} in the header, found {found!r}."
        )

    records: list[dict[str, Any]] = []
    skipped: list[int] = []
    dropped_tail = False
    last = len(lines) - 1
    for number, line in enumerate(lines[1:], start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if number == last:
                dropped_tail = True
            else:
                skipped.append(number + 1)
            continue
        if isinstance(obj, dict):
            records.append(obj)
    if skipped:
        logger.warning("Skipped unparsable JSONL line(s) %s", skipped)
    return JsonlContent(header, records, dropped_tail, tuple(skipped))


def read_jsonl(path: Path, fmt: JsonlFormat) -> JsonlContent | None:
    """Read a versioned JSONL file; None when it is missing or blank.

    Raises :class:`JsonlFormatError` as :func:`parse_jsonl` does, and
    ``OSError`` or ``UnicodeDecodeError`` when the file cannot be read.
    """
    if not path.exists():
        return None
    return parse_jsonl(path.read_text(encoding="utf-8"), fmt)


def read_header(path: Path, fmt: JsonlFormat) -> dict[str, Any] | None:
    """Return the header of *path* when it carries the current version.

    Reads only the first non-blank line; None when the file is missing,
    unreadable, or has no current header.
    """
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    header = json.loads(line)
                    return header if fmt.matches(header) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return None


def read_records(path: Path) -> list[dict[str, Any]]:
    """Return every JSON object line of *path*, skipping unparsable lines.

    For unversioned JSONL; a missing file gives an empty list.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in _lines(path.read_text(encoding="utf-8-sig")):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def repair_trailing_newline(path: Path) -> bool:
    """End a non-empty file with a newline; return True if one was added.

    A crash mid-append can leave a last line without its newline; the next
    append would otherwise fuse a new record onto it.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return False
            handle.write(b"\n")
            return True
    except OSError:
        return False


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* through a sibling temp file and ``os.replace``.

    A failure while writing leaves the previous file intact.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def write_jsonl(
    path: Path,
    fmt: JsonlFormat,
    header: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]] = (),
) -> None:
    """Write a complete versioned JSONL file atomically."""
    lines = [_dumps(fmt.header(header))]
    lines.extend(_dumps(record) for record in records)
    atomic_write_text(path, "\n".join(lines) + "\n")


class JsonlWriter:
    """Appends records to JSONL files, one cached handle and lock per path.

    Every append writes one line and flushes it. Appends to the same path
    from several threads are serialized. After :meth:`close`, a late append
    opens, writes and closes the file without caching a handle. Write
    failures are logged and reported as False, so a log failure never stops
    the work being logged.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._handles: dict[Path, tuple[TextIO, threading.Lock]] = {}
        self._closed: set[Path] = set()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close_all()

    def start(self, path: Path, fmt: JsonlFormat, header: Mapping[str, Any]) -> bool:
        """Replace *path* with a file holding only the header.

        The header is written atomically, so a failure keeps the previous
        file and its records.
        """
        self._release(path)
        with self._guard:
            self._closed.discard(path)
        try:
            atomic_write_text(path, _dumps(fmt.header(header)) + "\n")
        except OSError as exc:
            logger.warning("Failed to start JSONL file %s: %s", path, exc)
            return False
        return True

    def append(self, path: Path, record: Mapping[str, Any]) -> bool:
        """Append *record* as one line; return False when the write failed."""
        try:
            line = _dumps(record) + "\n"
        except (TypeError, ValueError) as exc:
            logger.warning("Cannot serialize JSONL record for %s: %s", path, exc)
            return False
        try:
            with self._guard:
                acquired = self._acquire(path)
        except OSError as exc:
            logger.warning("Failed to open JSONL file %s: %s", path, exc)
            return False
        if acquired is None:
            return self._append_once(path, line)
        handle, lock = acquired
        try:
            with lock:
                handle.write(line)
                handle.flush()
        except ValueError:
            return self._append_once(path, line)
        except OSError as exc:
            logger.warning("Failed to write JSONL file %s: %s", path, exc)
            return False
        return True

    def update_header(
        self, path: Path, fmt: JsonlFormat, fields: Mapping[str, Any]
    ) -> bool:
        """Merge *fields* into the header of *path* and rewrite it atomically.

        Returns False when the file is missing or has no current header.
        """
        self._release(path)
        try:
            content = read_jsonl(path, fmt)
        except (OSError, UnicodeDecodeError, JsonlFormatError) as exc:
            logger.warning("Cannot update JSONL header of %s: %s", path, exc)
            return False
        if content is None:
            return False
        header = {**content.header, **fields}
        try:
            write_jsonl(path, fmt, header, content.records)
        except OSError as exc:
            logger.warning("Failed to rewrite JSONL file %s: %s", path, exc)
            return False
        return True

    def close(self, path: Path) -> None:
        """Flush and release the handle of *path*; later appends open it anew."""
        self._release(path)
        with self._guard:
            self._closed.add(path)

    def close_all(self) -> None:
        """Release every cached handle."""
        with self._guard:
            paths = list(self._handles)
        for path in paths:
            self.close(path)

    def _acquire(self, path: Path) -> tuple[TextIO, threading.Lock] | None:
        """Return the cached handle of *path*; the caller holds the guard."""
        if path in self._closed:
            return None
        cached = self._handles.get(path)
        if cached is not None:
            return cached
        repair_trailing_newline(path)
        handle = path.open("a", encoding="utf-8", newline="\n")
        cached = (handle, threading.Lock())
        self._handles[path] = cached
        return cached

    def _release(self, path: Path) -> None:
        with self._guard:
            cached = self._handles.pop(path, None)
        if cached is None:
            return
        handle, lock = cached
        with lock, contextlib.suppress(OSError):
            handle.close()

    def _append_once(self, path: Path, line: str) -> bool:
        try:
            with self._guard:
                repair_trailing_newline(path)
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
        except OSError as exc:
            logger.warning("Failed to write JSONL file %s: %s", path, exc)
            return False
        return True


def file_identity(path: Path) -> dict[str, Any]:
    """Return the cheap identity of an input file: its path and byte size."""
    try:
        size: int | None = path.stat().st_size
    except OSError:
        size = None
    return {"source_file": str(path), "size": size}


def folder_identity(folder: Path, files: Sequence[Path]) -> dict[str, Any]:
    """Return the cheap identity of an input folder made of *files*.

    Records the file count, a hash of the sorted file names (which pins a
    name-derived page order) and the combined byte size, or None for the size
    when a file cannot be read.
    """
    return {
        "source_file": str(folder),
        "image_count": len(files),
        "image_names_sha256": _names_hash(file.name for file in files),
        "total_image_bytes": _total_size(files),
    }


def input_changed(
    identity: Mapping[str, Any] | None, *, extensions: Collection[str]
) -> bool:
    """Whether the input recorded in *identity* changed on disk.

    *identity* comes from :func:`file_identity` or :func:`folder_identity`;
    *extensions* (lowercase, with dot) select the files of a folder input. A
    change is reported only when it can be proven: a missing identity, a
    missing field or an input that cannot be read gives False.
    """
    if not isinstance(identity, Mapping):
        return False
    source = identity.get("source_file")
    if not isinstance(source, str):
        return False
    if "image_count" in identity or "total_image_bytes" in identity:
        return _folder_changed(Path(source), identity, extensions)
    try:
        size = Path(source).stat().st_size
    except OSError:
        return False
    stored = identity.get("size")
    return isinstance(stored, int) and stored != size


def _folder_changed(
    folder: Path, identity: Mapping[str, Any], extensions: Collection[str]
) -> bool:
    try:
        files = [p for p in folder.glob("*") if p.suffix.lower() in extensions]
    except OSError:
        return False
    stored_count = identity.get("image_count")
    if isinstance(stored_count, int) and stored_count != len(files):
        return True
    stored_names = identity.get("image_names_sha256")
    if isinstance(stored_names, str) and stored_names != _names_hash(
        p.name for p in files
    ):
        return True
    stored_bytes = identity.get("total_image_bytes")
    if not isinstance(stored_bytes, int):
        return False
    total = _total_size(files)
    return total is not None and total != stored_bytes


def _total_size(files: Iterable[Path]) -> int | None:
    total = 0
    for file in files:
        try:
            total += file.stat().st_size
        except OSError:
            return None
    return total


def _names_hash(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()
