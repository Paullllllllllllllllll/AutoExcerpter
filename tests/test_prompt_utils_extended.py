"""Extended tests for modules/prompt_utils.py - Untested strategies and helpers."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from modules.prompt_utils import (
    render_prompt_with_schema,
    _replace_schema_at_marker,
    _inject_context,
    SCHEMA_MARKER,
    SCHEMA_TOKEN,
    SCHEMA_TOKEN_GENERIC,
)


# ============================================================================
# render_prompt_with_schema - Strategy 3: Marker Replacement
# ============================================================================
class TestRenderPromptStrategy3MarkerReplacement:
    """Tests for render_prompt_with_schema() using marker-based replacement."""

    def test_marker_present_replaces_existing_schema(self):
        """When marker and existing JSON block are present, replaces the block."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        prompt = (
            'Transcribe this document.\n'
            'The JSON schema:\n'
            '{"old": "schema"}'
        )

        result = render_prompt_with_schema(prompt, schema)

        assert '"old"' not in result
        assert '"type": "object"' in result
        assert SCHEMA_MARKER in result

    def test_marker_present_no_existing_json(self):
        """When marker is present but no JSON follows, schema is appended."""
        schema = {"key": "value"}
        prompt = "Process this.\nThe JSON schema:\nPlease follow it."

        result = render_prompt_with_schema(prompt, schema)

        assert '"key": "value"' in result

    def test_marker_not_confused_with_token(self):
        """Marker strategy is used only when no token is present."""
        schema = {"a": 1}
        prompt = f"Do something.\n{SCHEMA_MARKER}\nold json here {{}}"

        result = render_prompt_with_schema(prompt, schema)

        assert '"a": 1' in result


# ============================================================================
# render_prompt_with_schema - Strategy 4: Append to End
# ============================================================================
class TestRenderPromptStrategy4Append:
    """Tests for render_prompt_with_schema() append strategy."""

    def test_no_token_or_marker_appends(self):
        """Schema is appended when neither token nor marker exists."""
        schema = {"type": "string"}
        prompt = "Transcribe the following image."

        result = render_prompt_with_schema(prompt, schema)

        assert result.startswith("Transcribe the following image.")
        assert SCHEMA_MARKER in result
        assert '"type": "string"' in result

    def test_appended_schema_is_valid_json(self):
        """The appended schema portion is valid JSON."""
        schema = {"items": [1, 2, 3]}
        prompt = "Simple prompt."

        result = render_prompt_with_schema(prompt, schema)

        # Extract JSON from the result after the marker
        marker_idx = result.find(SCHEMA_MARKER)
        json_part = result[marker_idx + len(SCHEMA_MARKER):].strip()
        parsed = json.loads(json_part)
        assert parsed == schema


# ============================================================================
# render_prompt_with_schema - Validation
# ============================================================================
class TestRenderPromptValidation:
    """Tests for render_prompt_with_schema() input validation."""

    def test_empty_prompt_raises_value_error(self):
        """Empty prompt text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            render_prompt_with_schema("", {"type": "object"})

    def test_none_prompt_raises_value_error(self):
        """None prompt text (falsy) raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            render_prompt_with_schema(None, {"type": "object"})  # type: ignore

    def test_invalid_schema_type_raises_value_error(self):
        """Non-dict schema raises ValueError."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            render_prompt_with_schema("prompt text", "not a dict")  # type: ignore

    def test_list_schema_raises_value_error(self):
        """List schema raises ValueError."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            render_prompt_with_schema("prompt text", [1, 2, 3])  # type: ignore

    def test_json_serialization_failure_falls_back_to_str(self):
        """When JSON serialization fails, falls back to str() representation."""
        # Create a dict with a non-serializable value
        class NonSerializable:
            def __repr__(self):
                return "NON_SERIALIZABLE_REPR"

        schema = {"key": NonSerializable()}

        # Patch json.dumps to raise for this specific call
        original_dumps = json.dumps

        def patched_dumps(obj, **kwargs):
            if any(isinstance(v, NonSerializable) for v in obj.values()):
                raise TypeError("Object of type NonSerializable is not serializable")
            return original_dumps(obj, **kwargs)

        with patch("modules.prompt_utils.json.dumps", side_effect=patched_dumps):
            result = render_prompt_with_schema(
                f"Prompt with {SCHEMA_TOKEN_GENERIC}", schema
            )

        # The fallback str() representation should appear
        assert "NON_SERIALIZABLE_REPR" in result


# ============================================================================
# render_prompt_with_schema - Context Injection
# ============================================================================
class TestRenderPromptContextInjection:
    """Tests for context parameter in render_prompt_with_schema()."""

    def test_context_replaces_placeholder(self):
        """Context string replaces {{CONTEXT}} placeholder."""
        schema = {"type": "object"}
        prompt = "Focus on: {{CONTEXT}}\n{{SCHEMA}}"

        result = render_prompt_with_schema(prompt, schema, context="medieval recipes")

        assert "medieval recipes" in result
        assert "{{CONTEXT}}" not in result

    def test_none_context_removes_placeholder_line(self):
        """None context removes the line containing {{CONTEXT}}."""
        schema = {"type": "object"}
        prompt = "Line one.\nFocus on: {{CONTEXT}}\nLine three.\n{{SCHEMA}}"

        result = render_prompt_with_schema(prompt, schema, context=None)

        assert "{{CONTEXT}}" not in result
        assert "Focus on:" not in result
        assert "Line one." in result
        assert "Line three." in result

    def test_empty_context_removes_placeholder_line(self):
        """Empty string context removes the line containing {{CONTEXT}}."""
        schema = {"type": "object"}
        prompt = "Header.\nTopics: {{CONTEXT}}\nBody.\n{{SCHEMA}}"

        result = render_prompt_with_schema(prompt, schema, context="")

        assert "{{CONTEXT}}" not in result
        assert "Topics:" not in result

    def test_no_context_placeholder_prompt_unchanged(self):
        """Prompt without {{CONTEXT}} is unchanged regardless of context value."""
        schema = {"type": "object"}
        prompt = "Simple prompt.\n{{SCHEMA}}"

        result = render_prompt_with_schema(
            prompt, schema, context="this should be ignored"
        )

        assert "this should be ignored" not in result
        assert "Simple prompt." in result


# ============================================================================
# _replace_schema_at_marker
# ============================================================================
class TestReplaceSchemaAtMarker:
    """Tests for _replace_schema_at_marker()."""

    def test_marker_with_existing_schema_replaces(self):
        """Existing schema block after marker is replaced."""
        prompt = (
            'Instructions here.\n'
            'The JSON schema:\n'
            '{"old": "data", "nested": {"a": 1}}\n'
            'End.'
        )
        new_schema = '{"new": "schema"}'

        result = _replace_schema_at_marker(prompt, new_schema)

        assert '"old"' not in result
        assert '"new": "schema"' in result
        assert "End." in result

    def test_marker_no_opening_brace_appends(self):
        """Schema is appended when no opening brace follows marker."""
        prompt = "Instructions.\nThe JSON schema:\nNo braces here."

        new_schema = '{"appended": true}'
        result = _replace_schema_at_marker(prompt, new_schema)

        assert '{"appended": true}' in result

    def test_marker_no_closing_brace_appends(self):
        """Schema is appended when opening brace exists but no closing brace after it."""
        # This scenario: rfind("}") returns -1 or <= start_brace
        # We need a case where there's an opening brace but the only closing
        # brace is before it (rfind returns something <= start_brace).
        # Actually, rfind("}") == -1 means no } at all.
        prompt = "The JSON schema:\n{incomplete schema without closing"

        new_schema = '{"complete": true}'
        result = _replace_schema_at_marker(prompt, new_schema)

        assert '{"complete": true}' in result

    def test_marker_not_found_appends_gracefully(self):
        """If marker is somehow not found, gracefully appends."""
        # This should not normally happen since the caller checks,
        # but test the defensive code path.
        prompt = "No marker here."
        new_schema = '{"safe": true}'

        result = _replace_schema_at_marker(prompt, new_schema)

        assert '{"safe": true}' in result
        assert SCHEMA_MARKER in result


# ============================================================================
# _inject_context
# ============================================================================
class TestInjectContext:
    """Tests for _inject_context()."""

    def test_placeholder_with_valid_context(self):
        """{{CONTEXT}} is replaced with the provided context string."""
        prompt = "Focus on the following: {{CONTEXT}}\nContinue."
        result = _inject_context(prompt, "culinary history")

        assert result == "Focus on the following: culinary history\nContinue."

    def test_placeholder_with_none_context_removes_line(self):
        """Line containing {{CONTEXT}} is removed when context is None."""
        prompt = "Line 1.\nTopics: {{CONTEXT}}\nLine 3."
        result = _inject_context(prompt, None)

        assert "{{CONTEXT}}" not in result
        assert "Topics:" not in result
        assert "Line 1." in result
        assert "Line 3." in result

    def test_placeholder_with_empty_context_removes_line(self):
        """Line containing {{CONTEXT}} is removed when context is empty."""
        prompt = "Header.\nContext: {{CONTEXT}}\nBody."
        result = _inject_context(prompt, "")

        assert "{{CONTEXT}}" not in result
        assert "Context:" not in result

    def test_placeholder_with_whitespace_only_context_removes_line(self):
        """Whitespace-only context is treated as empty and line is removed."""
        prompt = "Start.\nFocus: {{CONTEXT}}\nEnd."
        result = _inject_context(prompt, "   ")

        assert "{{CONTEXT}}" not in result
        assert "Focus:" not in result

    def test_no_placeholder_prompt_unchanged(self):
        """Prompt without {{CONTEXT}} is returned unchanged."""
        prompt = "A simple prompt with no placeholders."
        result = _inject_context(prompt, "some context")

        assert result == prompt

    def test_no_placeholder_none_context_unchanged(self):
        """Prompt without {{CONTEXT}} is unchanged even with None context."""
        prompt = "Unchanged prompt."
        result = _inject_context(prompt, None)

        assert result == "Unchanged prompt."

    def test_context_is_stripped(self):
        """Leading/trailing whitespace in context is stripped."""
        prompt = "Focus: {{CONTEXT}}\nDone."
        result = _inject_context(prompt, "  spice trade  ")

        assert "spice trade" in result
        assert "  spice trade  " not in result

    def test_multiple_context_placeholders(self):
        """All {{CONTEXT}} placeholders on different lines are handled."""
        prompt = "A: {{CONTEXT}}\nB: {{CONTEXT}}\nC."
        result = _inject_context(prompt, "recipes")

        # Both should be replaced
        assert result.count("recipes") == 2
        assert "{{CONTEXT}}" not in result
