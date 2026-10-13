"""Move the absolute paths a run stored in a fixture folder to another root.

A run records the absolute paths of its input in the database and in some
output files. A fixture stores them under :data:`ROOT_PLACEHOLDER`, so it
holds no machine path; :func:`relocate` writes the placeholder when the
fixture is made and the test's root when it is used. Paths are replaced in
their raw, JSON-escaped and forward-slash forms; nothing else changes.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

ROOT_PLACEHOLDER = "<fixture-root>"
TEXT_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt"})


def _forms(old: str, new: str) -> list[tuple[str, str]]:
    return [
        (json.dumps(old)[1:-1], json.dumps(new)[1:-1]),
        (old, new),
        (old.replace("\\", "/"), new.replace("\\", "/")),
    ]


def _replaced(text: str, forms: list[tuple[str, str]]) -> str:
    for old, new in forms:
        text = text.replace(old, new)
    return text


def _relocate_database(path: Path, forms: list[tuple[str, str]]) -> None:
    with contextlib.closing(sqlite3.connect(path)) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cursor = connection.execute(f'SELECT rowid, * FROM "{table}"')
            columns = [description[0] for description in cursor.description][1:]
            for rowid, *values in cursor.fetchall():
                for column, value in zip(columns, values, strict=True):
                    if not isinstance(value, str):
                        continue
                    changed = _replaced(value, forms)
                    if changed != value:
                        connection.execute(
                            f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                            (changed, rowid),
                        )
        connection.commit()


def relocate(folder: Path, old: str, new: str) -> None:
    """Replace the root *old* with *new* in every database and text file."""
    forms = _forms(old, new)
    for path in sorted(folder.rglob("*")):
        if path.suffix == ".sqlite":
            _relocate_database(path, forms)
        elif path.suffix in TEXT_SUFFIXES:
            text = path.read_bytes().decode("utf-8")
            path.write_bytes(_replaced(text, forms).encode("utf-8"))
