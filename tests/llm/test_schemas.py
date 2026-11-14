"""The schema lookup by file name and the shape check of a replacement schema."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.common.classifiers import AmbiguousNameError
from autoexcerpter.llm.prompts import SCHEMAS_DIR
from autoexcerpter.llm.schemas import (
    SCHEMA_FILES,
    SUMMARY_SCHEMA_FILE,
    TRANSCRIPTION_SCHEMA_FILE,
    AmbiguousSchemaError,
    SchemaFileError,
    check_schema,
    packaged_schemas,
    resolve_schemas,
    schema_path,
)
from autoexcerpter.spec import SpecError

Edit = Callable[[dict[str, Any]], None]


def _packaged(file_name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (SCHEMAS_DIR / file_name).read_text(encoding="utf-8")
    )
    return data


def _write(folder: Path, file_name: str, data: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _reworded(file_name: str) -> dict[str, Any]:
    data = _packaged(file_name)
    data["schema"]["description"] = "A project's own instructions."
    return data


@pytest.mark.parametrize("role", sorted(SCHEMA_FILES))
def test_the_packaged_schemas_pass_the_check(role: str) -> None:
    path = SCHEMAS_DIR / SCHEMA_FILES[role]
    check_schema(role, _packaged(SCHEMA_FILES[role]), path)
    schemas = packaged_schemas()
    assert schemas.of(role).path == path
    assert schemas.of(role).spec() == _packaged(SCHEMA_FILES[role])


def test_a_user_file_replaces_the_packaged_one_of_its_name(tmp_path: Path) -> None:
    own = _write(
        tmp_path / "project", SUMMARY_SCHEMA_FILE, _reworded(SUMMARY_SCHEMA_FILE)
    )

    schemas = resolve_schemas([tmp_path / "project"])

    assert schemas.summary.path == own
    assert schemas.summary.spec()["schema"]["description"].startswith("A project")
    assert schemas.transcription.path == SCHEMAS_DIR / TRANSCRIPTION_SCHEMA_FILE
    assert schemas.summary.sha256 != packaged_schemas().summary.sha256
    assert schemas.transcription.sha256 == packaged_schemas().transcription.sha256


def test_the_text_is_the_two_space_dump_the_store_keeps(tmp_path: Path) -> None:
    data = _reworded(TRANSCRIPTION_SCHEMA_FILE)
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / TRANSCRIPTION_SCHEMA_FILE).write_text(json.dumps(data), encoding="utf-8")

    schema = resolve_schemas([folder]).transcription

    assert schema.text == json.dumps(data, ensure_ascii=False, indent=2)


def test_a_missing_folder_is_passed_over_and_a_repeated_one_searched_once(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "project"
    own = _write(
        folder, TRANSCRIPTION_SCHEMA_FILE, _reworded(TRANSCRIPTION_SCHEMA_FILE)
    )

    found = schema_path(
        TRANSCRIPTION_SCHEMA_FILE, [tmp_path / "missing", folder, folder / "."]
    )

    assert found == own


def test_the_same_file_in_two_user_folders_is_ambiguous(tmp_path: Path) -> None:
    first = _write(
        tmp_path / "one", SUMMARY_SCHEMA_FILE, _reworded(SUMMARY_SCHEMA_FILE)
    )
    second = _write(
        tmp_path / "two", SUMMARY_SCHEMA_FILE, _packaged(SUMMARY_SCHEMA_FILE)
    )

    with pytest.raises(AmbiguousSchemaError) as caught:
        resolve_schemas([first.parent, second.parent])

    assert isinstance(caught.value, AmbiguousNameError)
    assert isinstance(caught.value, SchemaFileError)
    assert isinstance(caught.value, SpecError)
    assert caught.value.paths == (first, second)
    assert str(first) in str(caught.value) and str(second) in str(caught.value)


def test_an_unreadable_file_names_the_file(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / SUMMARY_SCHEMA_FILE).write_text("{not json", encoding="utf-8")

    with pytest.raises(SchemaFileError, match="summary schema"):
        resolve_schemas([folder])


def _set(path: str, value: Any) -> Edit:
    """Return an edit that sets the dotted *path* of the schema body."""

    def edit(body: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node = body
        for part in parents:
            node = node[part]
        node[last] = value

    return edit


def _drop(path: str) -> Edit:
    def edit(body: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node = body
        for part in parents:
            node = node[part]
        del node[last]

    return edit


def _unrequire(path: str, name: str) -> Edit:
    def edit(body: dict[str, Any]) -> None:
        node = body
        for part in path.split(".") if path else []:
            node = node[part]
        node["required"] = [item for item in node["required"] if item != name]

    return edit


TX = "properties"
SM_PAGE = "properties.page_information.properties"
SM_REF = "properties.references.items"

REFUSED: list[Any] = [
    pytest.param(
        "transcription",
        _drop(f"{TX}.transcription"),
        "transcription is missing",
        id="transcription-removed",
    ),
    pytest.param(
        "transcription",
        _set(f"{TX}.transcription.type", "array"),
        "transcription must allow type 'string'",
        id="transcription-retyped",
    ),
    pytest.param(
        "transcription",
        _set(f"{TX}.no_transcribable_text.type", ["boolean", "string"]),
        "no_transcribable_text allows type 'string'",
        id="flag-widened",
    ),
    pytest.param(
        "transcription",
        _unrequire("", "transcription_not_possible"),
        "transcription_not_possible is not in the required list",
        id="flag-not-required",
    ),
    pytest.param(
        "summary",
        _set("properties.bullet_points.items.type", ["string", "null"]),
        "bullet_points.items allows type 'null'",
        id="bullet-items-nullable",
    ),
    pytest.param(
        "summary",
        _drop(f"{SM_PAGE}.page_number_integer"),
        "page_information.page_number_integer is missing",
        id="page-number-removed",
    ),
    pytest.param(
        "summary",
        _set(f"{SM_PAGE}.page_number_type.enum", ["arabic", "none", "greek"]),
        "page_number_type has the enum value 'greek', which the renderer does not",
        id="unknown-numbering-added",
    ),
    pytest.param(
        "summary",
        _set(f"{SM_PAGE}.page_types.items.enum", ["content", "recipe"]),
        "page type 'recipe', which no page renders",
        id="unknown-page-type",
    ),
    pytest.param(
        "summary",
        _set(f"{SM_PAGE}.page_types.items.enum", ["bibliography", "index"]),
        "has no enum value 'content'",
        id="content-page-type-removed",
    ),
    pytest.param(
        "summary",
        _drop(f"{SM_PAGE}.page_types.items.enum"),
        "needs an enum of the page types",
        id="page-types-free-text",
    ),
    pytest.param(
        "summary",
        _unrequire(SM_REF, "citation"),
        "references.items.citation is not in the required list",
        id="citation-not-required",
    ),
    pytest.param(
        "summary",
        _set(f"{SM_REF}.properties.citation", {"anyOf": [{"type": "string"}]}),
        "references.items.citation declares no type",
        id="citation-anyof",
    ),
    pytest.param(
        "summary",
        _unrequire("", "references"),
        "references is not in the required list of the schema",
        id="references-not-required",
    ),
]


@pytest.mark.parametrize(("role", "edit", "message"), REFUSED)
def test_a_replacement_that_breaks_a_read_shape_is_refused(
    tmp_path: Path, role: str, edit: Edit, message: str
) -> None:
    data = copy.deepcopy(_packaged(SCHEMA_FILES[role]))
    edit(data["schema"])
    path = _write(tmp_path / "project", SCHEMA_FILES[role], data)

    with pytest.raises(SchemaFileError) as caught:
        resolve_schemas([path.parent])

    text = str(caught.value)
    assert message in text
    assert str(path) in text


ALLOWED: list[Any] = [
    pytest.param(
        "transcription",
        _set(f"{TX}.notes", {"type": "string", "description": "Extra notes."}),
        id="field-added",
    ),
    pytest.param(
        "transcription", _set(f"{TX}.transcription.type", "string"), id="null-removed"
    ),
    pytest.param(
        "summary",
        _set(f"{SM_PAGE}.page_types.items.enum", ["content", "bibliography"]),
        id="page-types-narrowed",
    ),
    pytest.param(
        "summary",
        _set(f"{SM_PAGE}.page_types.maxItems", 1),
        id="limit-changed",
    ),
    pytest.param(
        "summary",
        _set("properties.bullet_points.description", "One bullet per claim."),
        id="description-changed",
    ),
]


@pytest.mark.parametrize(("role", "edit"), ALLOWED)
def test_a_replacement_may_change_what_the_code_does_not_read(
    tmp_path: Path, role: str, edit: Edit
) -> None:
    data = copy.deepcopy(_packaged(SCHEMA_FILES[role]))
    edit(data["schema"])
    path = _write(tmp_path / "project", SCHEMA_FILES[role], data)

    assert resolve_schemas([path.parent]).of(role).path == path


def test_a_bare_schema_without_the_wrapper_is_read(tmp_path: Path) -> None:
    body = _packaged(TRANSCRIPTION_SCHEMA_FILE)["schema"]
    path = _write(tmp_path / "project", TRANSCRIPTION_SCHEMA_FILE, body)

    assert resolve_schemas([path.parent]).transcription.spec() == body
