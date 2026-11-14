"""The transcription and summary schemas of a run: lookup, shape check, hash.

AutoExcerpter reads two schemas by file name, :data:`TRANSCRIPTION_SCHEMA_FILE`
and :data:`SUMMARY_SCHEMA_FILE`, from ``resources/schemas/``. A file of the
same name in a user folder (the ``--schemas-dir`` folders, then the settings'
``schemas_dirs``) replaces the packaged one; the same file name in two user
folders raises :class:`AmbiguousSchemaError`. :func:`resolve_schemas` finds
both files, reads them and checks that each keeps the fields the code reads
(:func:`check_schema`). Every reader takes the :class:`PhaseSchema` it
returns, so the hash a run records, the schema text it stores and the
requests it sends come from one text.

The check follows the readers: the page reader (``llm.transcription``), the
summary reader (``llm.summary``), batch finalize, which validates the same
required keys, and the summary renderer (``rendering.summary``), which drops
bullet points that are not strings and citations that are malformed, and
renders only the page types it knows. A replacement may add fields, change
descriptions and change enums the code does not read; it may not remove,
rename or retype a field the code reads, nor make a field nullable where the
code reads no null. Each field the code reads declares its JSON type with
``type`` (a name or a list of names).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final

from autoexcerpter.common.classifiers import AmbiguousNameError, search_folders
from autoexcerpter.llm.prompts import SCHEMAS_DIR
from autoexcerpter.rendering.summary import PAGE_TYPE_LABELS
from autoexcerpter.spec import SpecError

__all__ = [
    "PAGE_NUMBER_TYPES",
    "SCHEMA_FILES",
    "SUMMARY_SCHEMA_FILE",
    "TRANSCRIPTION_SCHEMA_FILE",
    "AmbiguousSchemaError",
    "PhaseSchema",
    "RunSchemas",
    "SchemaFileError",
    "check_schema",
    "load_schema",
    "packaged_schemas",
    "resolve_schemas",
    "schema_path",
]

TRANSCRIPTION_SCHEMA_FILE: Final = "transcription_schema.json"
SUMMARY_SCHEMA_FILE: Final = "summary_schema.json"
SCHEMA_FILES: Final[Mapping[str, str]] = {
    "transcription": TRANSCRIPTION_SCHEMA_FILE,
    "summary": SUMMARY_SCHEMA_FILE,
}
"""The schema file of each role."""

PAGE_NUMBER_TYPES: Final = frozenset({"roman", "arabic", "none"})
"""The ``page_number_type`` values the renderer reads."""


class SchemaFileError(SpecError):
    """A schema file cannot be read or lacks a shape the code reads."""


class AmbiguousSchemaError(SchemaFileError, AmbiguousNameError):
    """A schema file name is found in two user folders."""


@dataclass(frozen=True)
class PhaseSchema:
    """The schema of one role as the run uses it.

    *text* is the schema JSON as the run stores and sends it (two-space
    indent, non-ASCII kept); *sha256* is its hash, the ``artifacts.sha256``
    under which the run stores it. *path* is the file it was read from.
    """

    role: str
    path: Path
    text: str

    @property
    def sha256(self) -> str:
        """The SHA-256 of :attr:`text` in UTF-8."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def spec(self) -> dict[str, Any]:
        """Return a fresh copy of the schema object."""
        data: dict[str, Any] = json.loads(self.text)
        return data


@dataclass(frozen=True)
class RunSchemas:
    """The transcription and summary schemas of one run."""

    transcription: PhaseSchema
    summary: PhaseSchema

    def of(self, role: str) -> PhaseSchema:
        """Return the schema of *role* (transcription or summary)."""
        schema: PhaseSchema = getattr(self, role)
        return schema


def schema_path(file_name: str, folders: Iterable[Path | str] = ()) -> Path:
    """Return the file named *file_name* in the user *folders*, else the packaged one.

    A folder named twice is searched once; a folder that does not exist is
    passed over.

    Raises:
        AmbiguousSchemaError: Two user folders hold a file of that name.
    """
    found = [
        folder / file_name
        for folder in search_folders(folders)
        if (folder / file_name).is_file()
    ]
    if len(found) > 1:
        raise AmbiguousSchemaError("schema", file_name, found)
    return found[0] if found else SCHEMAS_DIR / file_name


def load_schema(role: str, folders: Iterable[Path | str] = ()) -> PhaseSchema:
    """Return the checked schema of *role*, found by :func:`schema_path`.

    Raises:
        AmbiguousSchemaError: Two user folders hold the role's file.
        SchemaFileError: The file cannot be read, is not a JSON object, or
            lacks a shape the code reads; the message names the file and the
            field.
    """
    path = schema_path(SCHEMA_FILES[role], folders)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchemaFileError(f"cannot read the {role} schema {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaFileError(f"the {role} schema {path} is not a JSON object")
    check_schema(role, data, path)
    return PhaseSchema(role, path, json.dumps(data, ensure_ascii=False, indent=2))


def resolve_schemas(folders: Iterable[Path | str] = ()) -> RunSchemas:
    """Return both schemas of a run, each found and checked by :func:`load_schema`.

    Raises:
        SchemaFileError: A schema is ambiguous, unreadable or of another shape.
    """
    folders = tuple(folders)
    return RunSchemas(
        transcription=load_schema("transcription", folders),
        summary=load_schema("summary", folders),
    )


@cache
def packaged_schemas() -> RunSchemas:
    """Return the packaged schemas, read once."""
    return resolve_schemas()


# =============================================================================
# Shape check
# =============================================================================


@dataclass(frozen=True)
class _Field:
    """What the code reads of one schema field.

    The field's ``type`` must allow *need* and may allow only *allowed*.
    *required* puts the field in its parent's ``required`` list.
    """

    need: str
    allowed: frozenset[str]
    required: bool = True


def _field(need: str, *extra: str, required: bool = True) -> _Field:
    return _Field(need, frozenset({need, *extra}), required)


_TRANSCRIPTION_FIELDS: Final[Mapping[str, _Field]] = {
    "transcription": _field("string", "null"),
    "no_transcribable_text": _field("boolean", "null"),
    "transcription_not_possible": _field("boolean", "null"),
}
_SUMMARY_FIELDS: Final[Mapping[str, _Field]] = {
    "page_information": _field("object", "null"),
    "bullet_points": _field("array", "null"),
    "references": _field("array", "null"),
}
_PAGE_FIELDS: Final[Mapping[str, _Field]] = {
    "page_number_integer": _field("integer", "null", required=False),
    "is_two_page_spread": _field("boolean", "null", required=False),
    "page_number_integer_end": _field("integer", "null", required=False),
    "page_number_type": _field("string", "null", required=False),
    "page_types": _field("array", "null", required=False),
}
_REFERENCE_FIELDS: Final[Mapping[str, _Field]] = {
    "citation": _field("string"),
    "is_partial": _field("boolean", "null", required=False),
}


class _Checker:
    """Checks one schema file, naming the file and the field in every error."""

    def __init__(self, role: str, path: Path) -> None:
        self.role = role
        self.path = path

    def fail(self, where: str, message: str) -> SchemaFileError:
        return SchemaFileError(
            f"the {self.role} schema {self.path} cannot replace the packaged one: "
            f"{where} {message}"
        )

    def types(self, node: Any, where: str) -> frozenset[str]:
        if not isinstance(node, Mapping):
            raise self.fail(where, "is not a schema object")
        declared = node.get("type")
        if isinstance(declared, str):
            return frozenset({declared})
        if (
            isinstance(declared, list)
            and declared
            and all(isinstance(name, str) for name in declared)
        ):
            return frozenset(declared)
        raise self.fail(where, "declares no type")

    def typed(self, node: Any, where: str, want: _Field) -> Mapping[str, Any]:
        types = self.types(node, where)
        if want.need not in types:
            raise self.fail(where, f"must allow type {want.need!r}")
        extra = sorted(types - want.allowed)
        if extra:
            raise self.fail(
                where, f"allows type {extra[0]!r}, which the code cannot read"
            )
        assert isinstance(node, Mapping)
        return node

    def properties(
        self, node: Mapping[str, Any], where: str, fields: Mapping[str, _Field]
    ) -> dict[str, Mapping[str, Any]]:
        """Check the listed fields of an object node; return them by name."""
        properties = node.get("properties")
        if not isinstance(properties, Mapping):
            raise self.fail(where, "has no properties")
        required = node.get("required")
        names = set(required) if isinstance(required, list) else set()
        found: dict[str, Mapping[str, Any]] = {}
        for name, want in fields.items():
            path = f"{where}.{name}" if where != "schema" else name
            if name not in properties:
                raise self.fail(path, "is missing")
            if want.required and name not in names:
                owner = "the schema" if where == "schema" else where
                raise self.fail(path, f"is not in the required list of {owner}")
            found[name] = self.typed(properties[name], path, want)
        return found

    def items(
        self, node: Mapping[str, Any], where: str, want: _Field
    ) -> Mapping[str, Any]:
        return self.typed(node.get("items"), f"{where}.items", want)

    def strings(self, node: Mapping[str, Any], where: str) -> frozenset[str] | None:
        """Return the node's enum values; None when it has no enum."""
        enum = node.get("enum")
        if enum is None:
            return None
        if not isinstance(enum, list) or not all(isinstance(v, str) for v in enum):
            raise self.fail(where, "has an enum that is not a list of strings")
        return frozenset(enum)


def _body(checker: _Checker, data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the schema object: the wrapper's ``schema`` or the file itself."""
    body = data.get("schema", data)
    return checker.typed(body, "schema", _field("object"))


def _check_transcription(checker: _Checker, data: Mapping[str, Any]) -> None:
    checker.properties(_body(checker, data), "schema", _TRANSCRIPTION_FIELDS)


def _check_summary(checker: _Checker, data: Mapping[str, Any]) -> None:
    top = checker.properties(_body(checker, data), "schema", _SUMMARY_FIELDS)
    page = checker.properties(top["page_information"], "page_information", _PAGE_FIELDS)
    numbering = checker.strings(
        page["page_number_type"], "page_information.page_number_type"
    )
    if numbering is not None and numbering - PAGE_NUMBER_TYPES:
        unknown_type = sorted(numbering - PAGE_NUMBER_TYPES)[0]
        raise checker.fail(
            "page_information.page_number_type",
            f"has the enum value {unknown_type!r}, which the renderer does not read",
        )
    where = "page_information.page_types"
    types = checker.items(page["page_types"], where, _field("string"))
    values = checker.strings(types, f"{where}.items")
    known = frozenset(PAGE_TYPE_LABELS)
    if values is None:
        raise checker.fail(f"{where}.items", "needs an enum of the page types")
    unknown = sorted(values - known)
    if unknown:
        raise checker.fail(
            f"{where}.items", f"has the page type {unknown[0]!r}, which no page renders"
        )
    if "content" not in values:
        raise checker.fail(f"{where}.items", "has no enum value 'content'")
    checker.items(top["bullet_points"], "bullet_points", _field("string"))
    reference = checker.items(top["references"], "references", _field("object"))
    checker.properties(reference, "references.items", _REFERENCE_FIELDS)


def check_schema(role: str, data: Mapping[str, Any], path: Path) -> None:
    """Check that schema *data* of *role* keeps every shape the code reads.

    *data* is the file's JSON object, a wrapper with the schema under
    ``schema`` or the schema itself; *path* names the file in the error.

    Raises:
        SchemaFileError: A field the code reads is missing, of another type,
            nullable where the code reads no null, not required where the
            reader needs it, or (for the summary page types and page number
            type) with an enum the renderer cannot read.
    """
    checker = _Checker(role, path)
    if role == "transcription":
        _check_transcription(checker, data)
    elif role == "summary":
        _check_summary(checker, data)
    else:
        raise ValueError(f"unknown schema role {role!r}")
