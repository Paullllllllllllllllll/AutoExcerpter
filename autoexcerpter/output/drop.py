"""Drop stored documents: the core of the ``drop`` subcommand.

:func:`find_database` names the database of an input path,
:func:`select_stored` the stored documents a selection chooses, and
:func:`drop_documents` deletes them with every row keyed to them and the
rendered files that still hold the tool's rendering
(:meth:`~autoexcerpter.common.store.Store.drop`). A real drop is recorded as a
run with the command ``drop``; a dry run writes nothing. The next run
processes a dropped document from scratch, unless a rendered file of it was
kept (:data:`KEPT_FILE_HINT`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from autoexcerpter.common.contract import ExitCode
from autoexcerpter.common.store import (
    DocumentRow,
    DocumentSelection,
    DocumentSelectionError,
    DropReport,
    select_documents,
)
from autoexcerpter.output.store import (
    DATABASE_NAME,
    database_path,
    open_store,
    read_store,
)

__all__ = [
    "KEPT_FILE_HINT",
    "DropInputError",
    "drop_documents",
    "drop_exit_code",
    "find_database",
    "select_stored",
]


KEPT_FILE_HINT = (
    "A run skips a dropped document as unstored while a kept file of it remains: "
    "move the file away, or run with --force, which replaces it."
)
"""What a kept file means for the next run (``run`` classifies an item with
files but no stored units as unstored)."""


class DropInputError(ValueError):
    """The input names no store, or the selection chooses no stored document."""


def find_database(path: Path) -> Path:
    """Return the database that *path* names: the file itself or the one inside.

    Raises:
        DropInputError: *path* does not exist, is a file other than
            ``autoexcerpter.sqlite``, or is a folder without one.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise DropInputError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.name != DATABASE_NAME:
            raise DropInputError(f"{path} is not a {DATABASE_NAME} file")
        return path
    database = database_path(path)
    if not database.is_file():
        raise DropInputError(f"No {DATABASE_NAME} in {path}")
    return database


def select_stored(
    database: Path,
    *,
    select: str | None = None,
    files: Sequence[str] = (),
    select_all: bool = False,
) -> DocumentSelection:
    """Return the stored documents of *database* that the selection chooses.

    Raises:
        DropInputError: The database holds no store, the selection is
            malformed or chooses no document, or a ``--files`` entry names
            several.
        StoreError: The database is versionless, newer or unreadable.
    """
    store = read_store(database.parent)
    if store is None:
        raise DropInputError(f"{database} holds no stored document")
    try:
        selection = select_documents(
            store.documents(), select=select, files=files, select_all=select_all
        )
    except DocumentSelectionError as exc:
        raise DropInputError(str(exc)) from exc
    if not selection.documents:
        raise DropInputError(f"{database} holds no stored document")
    return selection


async def drop_documents(
    database: Path, documents: Sequence[DocumentRow], *, dry_run: bool = False
) -> DropReport:
    """Drop *documents* from *database*, or report what a drop would do.

    A real drop opens the store's writer and records a ``drop`` run with the
    names of the documents asked for and its exit code.

    Raises:
        StoreError: The database is versionless, newer or unreadable.
    """
    ids = [document.doc_id for document in documents]
    root = database.parent
    if dry_run:
        reader = read_store(root)
        if reader is None:
            raise DropInputError(f"{database} holds no stored document")
        return await reader.drop(ids, dry_run=True)
    names = sorted(document.name for document in documents)
    async with open_store(root) as store:
        await store.begin_run(
            "drop", spec={"documents": {"value": names, "source": "flag"}}
        )
        code: int = ExitCode.FAILURE
        try:
            report = await store.drop(ids)
            code = drop_exit_code(report)
        finally:
            await store.end_run(code)
    return report


def drop_exit_code(report: DropReport) -> ExitCode:
    """Return 1 when the drop refused a document, else 0."""
    return ExitCode.FAILURE if report.refused else ExitCode.OK
