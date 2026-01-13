"""Utilities for rendering prompts with embedded JSON schemas.

This module provides flexible prompt template rendering with JSON schema injection.
It supports multiple injection strategies to accommodate different prompt formats.

Injection Strategies (applied in order):
1. **Generic Token Replacement**: Replaces {{SCHEMA}} with formatted JSON
2. **Legacy Token Replacement**: Replaces {{TRANSCRIPTION_SCHEMA}} for backward compatibility
3. **Marker Replacement**: Finds "The JSON schema:" and replaces/appends JSON after it
4. **Append Strategy**: Appends schema at end if no token/marker found

Features:
- Pretty-prints JSON with configurable indentation
- Validates input parameters (prompt text, schema object)
- Handles malformed JSON gracefully with fallback strategies
- Supports both token-based and marker-based templates

Usage:
    >>> schema = {"type": "object", "properties": {...}}
    >>> prompt = "Transcribe this image. {{SCHEMA}}"
    >>> rendered = render_prompt_with_schema(prompt, schema)
    # Returns prompt with {{SCHEMA}} replaced by formatted JSON

This ensures prompts include structured output specifications for the OpenAI API
while maintaining flexibility in template design.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Public API
__all__ = ["render_prompt_with_schema"]

# ============================================================================
# Constants
# ============================================================================
SCHEMA_TOKEN = "{{TRANSCRIPTION_SCHEMA}}"  # Legacy token for backward compatibility
SCHEMA_TOKEN_GENERIC = "{{SCHEMA}}"  # Generic token for any schema
SCHEMA_MARKER = "The JSON schema:"
DEFAULT_INDENT = 2


# ============================================================================
# Prompt Rendering Functions
# ============================================================================
def render_prompt_with_schema(
    prompt_text: str,
    schema_obj: Dict[str, Any],
    context: Optional[str] = None,
) -> str:
    """
    Inject a JSON schema and optional context into a prompt.

    This function supports multiple injection strategies:
    1. Token replacement: If "{{SCHEMA}}" exists, replace it
    2. Legacy token replacement: If "{{TRANSCRIPTION_SCHEMA}}" exists, replace it
    3. Marker replacement: If "The JSON schema:" exists, replace JSON after it
    4. Append: If none exist, append schema at the end
    
    For context injection:
    - If "{{CONTEXT}}" placeholder exists and context is provided, it's replaced
    - If context is None or empty, the entire context line is removed

    Args:
        prompt_text: The prompt template text
        schema_obj: The JSON schema object to inject
        context: Optional context string to inject (topics to focus on)

    Returns:
        Prompt text with schema and context injected

    Raises:
        ValueError: If prompt_text is empty or schema_obj is invalid

    Example:
        >>> schema = {"type": "object", "properties": {...}}
        >>> prompt = "Transcribe this. {{SCHEMA}}"
        >>> result = render_prompt_with_schema(prompt, schema)
    """
    # Validation
    if not prompt_text:
        logger.warning("Empty prompt text provided to render_prompt_with_schema")
        raise ValueError("prompt_text cannot be empty")

    if not isinstance(schema_obj, dict):
        logger.warning(f"Invalid schema_obj type: {type(schema_obj)}. Expected dict.")
        raise ValueError("schema_obj must be a dictionary")

    # Handle context placeholder
    prompt_text = _inject_context(prompt_text, context)

    # Convert schema to pretty-printed JSON
    try:
        schema_str = json.dumps(schema_obj, indent=DEFAULT_INDENT, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize schema to JSON: {e}")
        # Fallback to string representation
        schema_str = str(schema_obj)

    # Strategy 1: Generic token replacement (preferred)
    if SCHEMA_TOKEN_GENERIC in prompt_text:
        return prompt_text.replace(SCHEMA_TOKEN_GENERIC, schema_str)
    
    # Strategy 2: Legacy token replacement (for backward compatibility)
    if SCHEMA_TOKEN in prompt_text:
        return prompt_text.replace(SCHEMA_TOKEN, schema_str)

    # Strategy 3: Marker replacement
    if SCHEMA_MARKER in prompt_text:
        return _replace_schema_at_marker(prompt_text, schema_str)

    # Strategy 4: Append to end
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


def _inject_context(prompt_text: str, context: Optional[str]) -> str:
    """
    Inject context into a prompt or remove the context placeholder.

    If context is provided and non-empty, replaces {{CONTEXT}} with the context.
    If context is None or empty, removes the entire line containing {{CONTEXT}}.

    Args:
        prompt_text: The prompt text potentially containing {{CONTEXT}}
        context: Optional context string to inject

    Returns:
        Updated prompt text with context injected or placeholder removed
    """
    context_placeholder = "{{CONTEXT}}"
    
    if context_placeholder not in prompt_text:
        return prompt_text
    
    if context and context.strip():
        # Replace placeholder with actual context
        return prompt_text.replace(context_placeholder, context.strip())
    else:
        # Remove entire line containing the placeholder to save tokens
        # Pattern matches lines containing {{CONTEXT}} including the line break
        prompt_text = re.sub(r"^.*\{\{CONTEXT\}\}.*\n?", "", prompt_text, flags=re.MULTILINE)
        return prompt_text
