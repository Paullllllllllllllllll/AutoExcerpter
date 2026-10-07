"""Render prompt templates with an embedded JSON schema and optional context.

The schema goes in by the first strategy that applies: replace the
``{{SCHEMA}}`` token; replace the JSON after the ``The JSON schema:`` marker;
append the marker and the schema at the end.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PROMPTS_DIR",
    "RESOURCES_DIR",
    "SCHEMAS_DIR",
    "render_prompt_with_schema",
    "strip_markdown_code_block",
]

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"
PROMPTS_DIR = RESOURCES_DIR / "prompts"
SCHEMAS_DIR = RESOURCES_DIR / "schemas"

SCHEMA_TOKEN_GENERIC = "{{SCHEMA}}"
SCHEMA_MARKER = "The JSON schema:"
DEFAULT_INDENT = 2


def strip_markdown_code_block(text: str) -> str:
    """Strip the markdown code fence around an answer, with any label."""
    stripped = text.strip()
    if stripped.startswith("```"):
        newline_idx = stripped.find("\n")
        if newline_idx != -1:
            stripped = stripped[newline_idx + 1 :]
        elif stripped[:7].lower() == "```json":
            stripped = stripped[7:]
        else:
            stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def render_prompt_with_schema(
    prompt_text: str,
    schema_obj: dict[str, Any],
    context: str | None = None,
) -> str:
    """Inject a JSON schema and optional context into a prompt.

    A ``{{CONTEXT}}`` placeholder takes *context*; without context its whole
    line is removed.

    Raises:
        ValueError: *prompt_text* is empty or *schema_obj* is not a dict.
    """
    if not prompt_text:
        logger.warning("Empty prompt text provided to render_prompt_with_schema")
        raise ValueError("prompt_text cannot be empty")

    if not isinstance(schema_obj, dict):
        logger.warning(f"Invalid schema_obj type: {type(schema_obj)}. Expected dict.")
        raise ValueError("schema_obj must be a dictionary")

    prompt_text = _inject_context(prompt_text, context)

    try:
        schema_str = json.dumps(schema_obj, indent=DEFAULT_INDENT, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize schema to JSON: {e}")
        schema_str = str(schema_obj)

    if SCHEMA_TOKEN_GENERIC in prompt_text:
        return prompt_text.replace(SCHEMA_TOKEN_GENERIC, schema_str)

    if SCHEMA_MARKER in prompt_text:
        return _replace_schema_at_marker(prompt_text, schema_str)

    return prompt_text + f"\n\n{SCHEMA_MARKER}\n" + schema_str


def _replace_schema_at_marker(prompt_text: str, schema_str: str) -> str:
    """Replace the brace block after the schema marker, or append the schema."""
    marker_idx = prompt_text.find(SCHEMA_MARKER)
    if marker_idx == -1:
        return prompt_text + f"\n\n{SCHEMA_MARKER}\n" + schema_str

    start_brace = prompt_text.find("{", marker_idx)

    if start_brace == -1:
        return prompt_text + "\n" + schema_str

    end_brace = prompt_text.rfind("}")

    if end_brace == -1 or end_brace <= start_brace:
        return prompt_text + "\n" + schema_str

    return prompt_text[:start_brace] + schema_str + prompt_text[end_brace + 1 :]


def _inject_context(prompt_text: str, context: str | None) -> str:
    """Replace ``{{CONTEXT}}`` with *context*, or drop its line when empty."""
    context_placeholder = "{{CONTEXT}}"

    if context_placeholder not in prompt_text:
        return prompt_text

    if context and context.strip():
        return prompt_text.replace(context_placeholder, context.strip())
    else:
        prompt_text = re.sub(
            r"^.*\{\{CONTEXT\}\}.*\n?", "", prompt_text, flags=re.MULTILINE
        )
        return prompt_text
