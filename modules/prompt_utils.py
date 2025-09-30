"""Utilities for rendering prompts with embedded JSON schemas."""

from __future__ import annotations

import json
from typing import Any, Dict

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Public API
__all__ = ["render_prompt_with_schema"]

# Constants
SCHEMA_TOKEN = "{{TRANSCRIPTION_SCHEMA}}"
SCHEMA_MARKER = "The JSON schema:"
DEFAULT_INDENT = 2


def render_prompt_with_schema(prompt_text: str, schema_obj: Dict[str, Any]) -> str:
    """
    Inject a JSON schema into a prompt using one of three strategies.

    This function supports three ways to inject a schema:
    1. Token replacement: If "{{TRANSCRIPTION_SCHEMA}}" exists, replace it
    2. Marker replacement: If "The JSON schema:" exists, replace JSON after it
    3. Append: If neither exists, append schema at the end

    Args:
        prompt_text: The prompt template text
        schema_obj: The JSON schema object to inject

    Returns:
        Prompt text with schema injected

    Raises:
        ValueError: If prompt_text is empty or schema_obj is invalid

    Example:
        >>> schema = {"type": "object", "properties": {...}}
        >>> prompt = "Transcribe this. {{TRANSCRIPTION_SCHEMA}}"
        >>> result = render_prompt_with_schema(prompt, schema)
    """
    # Validation
    if not prompt_text:
        logger.warning("Empty prompt text provided to render_prompt_with_schema")
        raise ValueError("prompt_text cannot be empty")

    if not isinstance(schema_obj, dict):
        logger.warning(f"Invalid schema_obj type: {type(schema_obj)}. Expected dict.")
        raise ValueError("schema_obj must be a dictionary")

    # Convert schema to pretty-printed JSON
    try:
        schema_str = json.dumps(schema_obj, indent=DEFAULT_INDENT, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize schema to JSON: {e}")
        # Fallback to string representation
        schema_str = str(schema_obj)

    # Strategy 1: Token replacement
    if SCHEMA_TOKEN in prompt_text:
        return prompt_text.replace(SCHEMA_TOKEN, schema_str)

    # Strategy 2: Marker replacement
    if SCHEMA_MARKER in prompt_text:
        return _replace_schema_at_marker(prompt_text, schema_str)

    # Strategy 3: Append to end
    return prompt_text + f"\n\n{SCHEMA_MARKER}\n" + schema_str


def _replace_schema_at_marker(prompt_text: str, schema_str: str) -> str:
    """
    Replace or append schema after the schema marker.

    Args:
        prompt_text: The prompt text containing the marker
        schema_str: The schema string to inject

    Returns:
        Updated prompt text with schema
    """
    marker_idx = prompt_text.find(SCHEMA_MARKER)
    if marker_idx == -1:
        # Should not happen, but handle gracefully
        return prompt_text + f"\n\n{SCHEMA_MARKER}\n" + schema_str

    # Locate the first opening brace after the marker
    start_brace = prompt_text.find("{", marker_idx)
    
    if start_brace == -1:
        # No existing schema, append after marker
        return prompt_text + "\n" + schema_str

    # Find the last closing brace to identify the schema block
    end_brace = prompt_text.rfind("}")
    
    if end_brace == -1 or end_brace <= start_brace:
        # No valid schema block found, append after marker
        return prompt_text + "\n" + schema_str

    # Replace the existing schema block
    return (
        prompt_text[:start_brace]
        + schema_str
        + prompt_text[end_brace + 1:]
    )
