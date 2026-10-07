"""SQLite output written by a single owner of the connection.

One database file per output folder. The writer opens it in WAL mode with a
busy timeout, so two processes can write the same file. Within a process, one
dedicated thread owns the connection and runs every write, which serializes
writes submitted from parallel tasks. Each document's rows are committed in one
transaction and rolled back as a whole on error. Rows are upserted on the key
columns the caller declares, so resumed and repeated runs never duplicate them.
The caller supplies the table definitions; flat values are stored as they are,
and mappings, lists and tuples as JSON text.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sqlite3
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import TracebackType
from typing import Any

DEFAULT_BUSY_TIMEOUT = 30.0
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class TableSpec:
    """A table: its name, its DDL and the key columns that identify a row.

    The DDL must declare a primary key or unique constraint on exactly the key
    columns, and should use ``IF NOT EXISTS`` so that reopening a database
    works.
    """

    name: str
    ddl: str
    key: tuple[str, ...]

    def __post_init__(self) -> None:
        _quote(self.name)
        if not self.key:
            raise ValueError(f"Table {self.name!r} declares no key columns.")
        for column in self.key:
            _quote(column)


def _quote(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def to_sql_value(value: Any) -> Any:
    """Convert a Python value to a column value.

    Mappings, lists and tuples become JSON text; paths become strings; other
    values pass through to the driver.
    """
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, PurePath):
        return str(value)
    return value


@dataclass(frozen=True)
class _Statement:
    sql: str
    params: tuple[Any, ...]


def _upsert_statement(spec: TableSpec, row: Mapping[str, Any]) -> _Statement:
    missing = [column for column in spec.key if column not in row]
    if missing:
        raise ValueError(f"Row for {spec.name!r} lacks key column(s) {missing}.")
    columns = list(row)
    names = ", ".join(_quote(column) for column in columns)
    marks = ", ".join("?" for _ in columns)
    key = ", ".join(_quote(column) for column in spec.key)
    updates = [
        f"{_quote(column)} = excluded.{_quote(column)}"
        for column in columns
        if column not in spec.key
    ]
    action = f"DO UPDATE SET {', '.join(updates)}" if updates else "DO NOTHING"
    sql = (
        f"INSERT INTO {_quote(spec.name)} ({names}) VALUES ({marks}) "
        f"ON CONFLICT ({key}) {action}"
    )
    return _Statement(sql, tuple(to_sql_value(row[column]) for column in columns))


def _delete_statement(
    spec: TableSpec,
    match: Mapping[str, Any],
    keep: Mapping[str, Collection[Any]] | None = None,
) -> _Statement:
    if not match:
        raise ValueError(f"A delete from {spec.name!r} needs at least one column.")
    where = " AND ".join(f"{_quote(column)} = ?" for column in match)
    params = [to_sql_value(value) for value in match.values()]
    kept = {column: list(values) for column, values in (keep or {}).items()}
    if kept and all(kept.values()):
        clauses = []
        for column, values in kept.items():
            marks = ", ".join("?" for _ in values)
            clauses.append(f"{_quote(column)} IN ({marks})")
            params.extend(to_sql_value(value) for value in values)
        where += f" AND NOT ({' AND '.join(clauses)})"
    sql = f"DELETE FROM {_quote(spec.name)} WHERE {where}"
    return _Statement(sql, tuple(params))


class DocumentTransaction:
    """Collects the writes of one document and commits them together.

    Use it as an async context manager from :meth:`SqliteWriter.document`.
    Writes are validated when added and sent to the writer on a clean exit;
    an exception inside the block discards them, so no partial rows land.
    """

    def __init__(self, writer: SqliteWriter) -> None:
        self._writer = writer
        self._statements: list[_Statement] = []

    def upsert(self, table: str, row: Mapping[str, Any]) -> None:
        """Insert *row* or update the existing row with the same key."""
        spec = self._writer.table(table)
        self._statements.append(_upsert_statement(spec, row))

    def delete(
        self,
        table: str,
        match: Mapping[str, Any],
        *,
        keep: Mapping[str, Collection[Any]] | None = None,
    ) -> None:
        """Delete the rows whose columns equal the values in *match*.

        With *keep*, rows whose every *keep* column holds one of its listed
        values stay; an empty list keeps nothing.
        """
        spec = self._writer.table(table)
        self._statements.append(_delete_statement(spec, match, keep))

    async def __aenter__(self) -> DocumentTransaction:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        statements, self._statements = self._statements, []
        if exc_type is None and statements:
            await self._writer._execute(statements)


class SqliteWriter:
    """Owns one connection to a database file and serializes all writes.

    Open it with ``async with SqliteWriter(path, tables) as writer``. Write a
    document with ``async with writer.document() as tx`` or
    :meth:`upsert`. A write that is still running when its awaiting task is
    cancelled finishes in the writer thread, committed or rolled back as a
    whole.
    """

    def __init__(
        self,
        path: Path,
        tables: Sequence[TableSpec],
        *,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
    ) -> None:
        self.path = Path(path)
        self._tables = {spec.name: spec for spec in tables}
        self._busy_timeout = busy_timeout
        self._executor: ThreadPoolExecutor | None = None
        self._connection: sqlite3.Connection | None = None

    def table(self, name: str) -> TableSpec:
        """Return the declared table *name*; ``KeyError`` when undeclared."""
        try:
            return self._tables[name]
        except KeyError:
            raise KeyError(f"Table {name!r} is not declared for {self.path}") from None

    async def open(self) -> None:
        """Open the database, enable WAL mode and create the tables."""
        if self._executor is not None:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sqlite-writer"
        )
        try:
            await self._run(self._connect)
        except BaseException:
            self._executor.shutdown(wait=False)
            self._executor = None
            raise

    async def close(self) -> None:
        """Close the connection and stop the writer thread."""
        executor = self._executor
        if executor is None:
            return
        try:
            await self._run(self._disconnect)
        finally:
            self._executor = None
            executor.shutdown(wait=False)

    async def __aenter__(self) -> SqliteWriter:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def document(self) -> DocumentTransaction:
        """Return a transaction that commits one document's writes together."""
        return DocumentTransaction(self)

    async def upsert(self, table: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """Upsert *rows* into *table* in one transaction."""
        spec = self.table(table)
        await self._execute([_upsert_statement(spec, row) for row in rows])

    async def _execute(self, statements: Sequence[_Statement]) -> None:
        await self._run(self._transaction, statements)

    async def _run(self, func: Callable[..., None], *args: Any) -> None:
        if self._executor is None:
            raise RuntimeError(f"SQLite writer for {self.path} is not open.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, func, *args)

    def _connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path, timeout=self._busy_timeout, isolation_level=None
        )
        try:
            connection.execute(
                f"PRAGMA busy_timeout = {int(self._busy_timeout * 1000)}"
            )
            connection.execute("PRAGMA journal_mode = WAL")
            for spec in self._tables.values():
                connection.executescript(spec.ddl)
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def _disconnect(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _transaction(self, statements: Sequence[_Statement]) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError(f"SQLite writer for {self.path} is not open.")
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                connection.execute(statement.sql, statement.params)
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")


def table_names(path: Path) -> list[str]:
    """Return the names of the tables in the database at *path*, sorted."""
    with _read_only(path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def read_rows(
    path: Path, table: str, *, order_by: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Return the rows of *table* as dicts, ordered by *order_by* or rowid."""
    order = ", ".join(_quote(column) for column in order_by) if order_by else "rowid"
    with _read_only(path) as connection:
        cursor = connection.execute(f"SELECT * FROM {_quote(table)} ORDER BY {order}")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


@contextlib.contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection]:
    """Open an existing database for queries only.

    The connection may write the file, unlike a ``mode=ro`` one, so that as
    the last connection it checkpoints the WAL and removes the ``-wal`` and
    ``-shm`` files instead of leaving them beside the database.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"No database at {path}")
    connection = sqlite3.connect(Path(path))
    try:
        connection.execute("PRAGMA query_only = ON")
        yield connection
    finally:
        connection.close()
