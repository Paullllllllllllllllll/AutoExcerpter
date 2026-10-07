"""The shipped summary schema and prompts keep their placeholders and rules."""

from __future__ import annotations

import json
from typing import Any

from autoexcerpter.llm.prompts import (
    PROMPTS_DIR,
    SCHEMAS_DIR,
    render_prompt_with_schema,
)


def _schema() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (SCHEMAS_DIR / "summary_schema.json").read_text(encoding="utf-8")
    )
    return data


def _prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def test_bullet_points_ask_for_condensed_complete_coverage() -> None:
    description = _schema()["schema"]["properties"]["bullet_points"]["description"]
    assert "condense" in description
    assert "LaTeX" in description
    assert "table_of_contents" in description


def test_system_prompt_keeps_placeholders_and_coverage_rules() -> None:
    text = _prompt("summary_system_prompt.txt")
    assert "{{SCHEMA}}" in text
    assert "{{CONTEXT}}" in text
    lowered = text.lower()
    assert "completeness matters" in lowered
    assert "every substantive claim" in lowered
    assert "never omitted entirely" in lowered
    assert 'page_number_type to "none"' in text
    assert "![Image:" in text


def test_plain_text_prompt_is_aligned() -> None:
    text = _prompt("summary_plain_text_prompt.txt")
    assert "{{SCHEMA}}" in text
    assert "condense" in text.lower()
    assert 'page_number_type to "none"' in text


def test_schema_renders_into_the_prompt() -> None:
    rendered = render_prompt_with_schema(
        _prompt("summary_system_prompt.txt"), _schema()["schema"], context="X"
    )
    assert "{{SCHEMA}}" not in rendered
    assert "{{CONTEXT}}" not in rendered
    assert "bullet_points" in rendered
