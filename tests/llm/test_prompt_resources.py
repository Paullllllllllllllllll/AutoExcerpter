"""The shipped schemas and prompts keep their placeholders and rules."""

from __future__ import annotations

import json
from typing import Any

import pytest

from autoexcerpter.llm.prompts import (
    PROMPTS_DIR,
    SCHEMAS_DIR,
    render_prompt_with_schema,
)
from autoexcerpter.llm.transcription import SENTINELS


def _schema(name: str = "summary_schema.json") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((SCHEMAS_DIR / name).read_text(encoding="utf-8"))
    return data


def _prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def test_bullet_points_carry_the_length_rules() -> None:
    description = _schema()["schema"]["properties"]["bullet_points"]["description"]
    assert "condense" in description
    assert "2-6 per page" in description
    assert "about a quarter of the length of the page's own text" in description
    assert "LaTeX" in description
    assert "table_of_contents" in description


def test_transcription_field_carries_the_rules() -> None:
    schema = _schema("transcription_schema.json")["schema"]
    assert "image_analysis" not in schema["properties"]
    assert schema["required"] == [
        "transcription",
        "no_transcribable_text",
        "transcription_not_possible",
    ]
    description = schema["properties"]["transcription"]["description"]
    for rule in ("verbatim", "[illegible]", "[?]", "<page_number>", "\\left."):
        assert rule in description


@pytest.mark.parametrize(
    "name",
    [
        "transcription_system_prompt.txt",
        "summary_system_prompt.txt",
        "summary_plain_text_prompt.txt",
    ],
)
def test_prompts_open_with_the_schema_reference(name: str) -> None:
    text = _prompt(name)
    assert text.splitlines()[0].startswith("For ")
    assert "field descriptions of the JSON schema below" in text.splitlines()[0]
    assert text.rstrip().endswith("{{SCHEMA}}")


def test_plain_transcription_prompt_matches_the_schema_rules() -> None:
    text = _prompt("transcription_plain_text_prompt.txt")
    for marker in (*SENTINELS, "verbatim", "[illegible]", "[?]", "\\left."):
        assert marker in text
    assert "{{SCHEMA}}" not in text


@pytest.mark.parametrize(
    ("name", "markdown"),
    [("summary_system_prompt.txt", True), ("summary_plain_text_prompt.txt", False)],
)
def test_summary_prompts_keep_context_and_language_rules(
    name: str, markdown: bool
) -> None:
    text = _prompt(name)
    assert "{{CONTEXT}}" in text
    assert "takes priority" in text
    assert "language of the document itself" in text
    assert ("![Image: description]" in text) is markdown


def test_schema_renders_into_the_prompt() -> None:
    rendered = render_prompt_with_schema(
        _prompt("summary_system_prompt.txt"), _schema()["schema"], context="X"
    )
    assert "{{SCHEMA}}" not in rendered
    assert "{{CONTEXT}}" not in rendered
    assert "bullet_points" in rendered
