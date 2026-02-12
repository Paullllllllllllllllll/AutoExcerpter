"""Tests for the context resolver module.

This module tests hierarchical context resolution for summarization,
including file-specific, folder-specific, and general context files.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.context_resolver import (
    resolve_summary_context,
    format_context_for_prompt,
    _read_and_validate_context,
    DEFAULT_CONTEXT_SIZE_THRESHOLD,
    CONTEXT_SUFFIX,
)


class TestResolveSummaryContext:
    """Tests for resolve_summary_context function."""

    def test_returns_none_when_no_context_files_exist(self, tmp_path):
        """Should return (None, None) when no context files exist."""
        input_file = tmp_path / "test.pdf"
        input_file.touch()
        
        content, path = resolve_summary_context(
            input_file=input_file,
            global_context_dir=tmp_path / "context"
        )
        
        assert content is None
        assert path is None

    def test_file_specific_context_takes_priority(self, tmp_path):
        """File-specific context should take priority over folder and general."""
        # Create input file
        input_file = tmp_path / "document.pdf"
        input_file.touch()
        
        # Create file-specific context
        file_context = tmp_path / "document_summary_context.txt"
        file_context.write_text("File-specific topics")
        
        # Create folder-specific context (should be ignored)
        folder_context = tmp_path.parent / f"{tmp_path.name}_summary_context.txt"
        
        # Create general context (should be ignored)
        context_dir = tmp_path / "context" / "summary"
        context_dir.mkdir(parents=True)
        general_context = context_dir / "general.txt"
        general_context.write_text("General topics")
        
        content, path = resolve_summary_context(
            input_file=input_file,
            global_context_dir=tmp_path / "context"
        )
        
        assert content == "File-specific topics"
        assert path == file_context

    def test_folder_specific_context_fallback(self, tmp_path):
        """Folder-specific context should be used when file-specific doesn't exist."""
        # Create subdirectory structure
        subdir = tmp_path / "project_folder"
        subdir.mkdir()
        input_file = subdir / "document.pdf"
        input_file.touch()
        
        # Create folder-specific context in parent directory
        folder_context = tmp_path / "project_folder_summary_context.txt"
        folder_context.write_text("Folder-specific topics")
        
        # Create general context (should be ignored)
        context_dir = tmp_path / "context" / "summary"
        context_dir.mkdir(parents=True)
        general_context = context_dir / "general.txt"
        general_context.write_text("General topics")
        
        content, path = resolve_summary_context(
            input_file=input_file,
            global_context_dir=tmp_path / "context"
        )
        
        assert content == "Folder-specific topics"
        assert path == folder_context

    def test_general_context_fallback(self, tmp_path):
        """General context should be used when no specific context exists."""
        input_file = tmp_path / "document.pdf"
        input_file.touch()
        
        # Create only general context
        context_dir = tmp_path / "context" / "summary"
        context_dir.mkdir(parents=True)
        general_context = context_dir / "general.txt"
        general_context.write_text("General topics")
        
        content, path = resolve_summary_context(
            input_file=input_file,
            global_context_dir=tmp_path / "context"
        )
        
        assert content == "General topics"
        assert path == general_context

    def test_no_input_file_uses_general_context(self, tmp_path):
        """When no input file is provided, should fall back to general context."""
        context_dir = tmp_path / "context" / "summary"
        context_dir.mkdir(parents=True)
        general_context = context_dir / "general.txt"
        general_context.write_text("General topics")
        
        content, path = resolve_summary_context(
            input_file=None,
            global_context_dir=tmp_path / "context"
        )
        
        assert content == "General topics"
        assert path == general_context

    def test_empty_context_file_returns_none(self, tmp_path):
        """Empty context files should be treated as non-existent."""
        input_file = tmp_path / "document.pdf"
        input_file.touch()
        
        # Create empty file-specific context
        file_context = tmp_path / "document_summary_context.txt"
        file_context.write_text("")
        
        # Create general context as fallback
        context_dir = tmp_path / "context" / "summary"
        context_dir.mkdir(parents=True)
        general_context = context_dir / "general.txt"
        general_context.write_text("General topics")
        
        content, path = resolve_summary_context(
            input_file=input_file,
            global_context_dir=tmp_path / "context"
        )
        
        # Should skip empty file and use general
        assert content == "General topics"
        assert path == general_context

    def test_whitespace_only_context_file_returns_none(self, tmp_path):
        """Whitespace-only context files should be treated as non-existent."""
        input_file = tmp_path / "document.pdf"
        input_file.touch()
        
        # Create whitespace-only file-specific context
        file_context = tmp_path / "document_summary_context.txt"
        file_context.write_text("   \n\t  \n  ")
        
        content, path = resolve_summary_context(
            input_file=input_file,
            global_context_dir=tmp_path / "context"
        )
        
        assert content is None
        assert path is None


class TestReadAndValidateContext:
    """Tests for _read_and_validate_context function."""

    def test_reads_valid_context_file(self, tmp_path):
        """Should read and return content from valid context file."""
        context_file = tmp_path / "context.txt"
        context_file.write_text("Food History, Wages, Early Modern")
        
        content = _read_and_validate_context(context_file)
        
        assert content == "Food History, Wages, Early Modern"

    def test_strips_whitespace(self, tmp_path):
        """Should strip leading and trailing whitespace."""
        context_file = tmp_path / "context.txt"
        context_file.write_text("  Food History  \n")
        
        content = _read_and_validate_context(context_file)
        
        assert content == "Food History"

    def test_returns_none_for_empty_file(self, tmp_path):
        """Should return None for empty files."""
        context_file = tmp_path / "context.txt"
        context_file.write_text("")
        
        content = _read_and_validate_context(context_file)
        
        assert content is None

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        """Should return None for non-existent files."""
        context_file = tmp_path / "nonexistent.txt"
        
        content = _read_and_validate_context(context_file)
        
        assert content is None

    def test_warns_for_large_context_file(self, tmp_path, caplog):
        """Should log warning for context files exceeding threshold."""
        import logging
        context_logger = logging.getLogger("modules.context_resolver")
        original_propagate = context_logger.propagate
        context_logger.propagate = True
        try:
            context_file = tmp_path / "context.txt"
            large_content = "x" * (DEFAULT_CONTEXT_SIZE_THRESHOLD + 100)
            context_file.write_text(large_content)
            
            with caplog.at_level(logging.WARNING, logger="modules.context_resolver"):
                content = _read_and_validate_context(context_file, size_threshold=DEFAULT_CONTEXT_SIZE_THRESHOLD)
            
            assert content == large_content
            assert "large" in caplog.text.lower()
        finally:
            context_logger.propagate = original_propagate

    def test_custom_size_threshold(self, tmp_path):
        """Should respect custom size threshold."""
        context_file = tmp_path / "context.txt"
        context_file.write_text("x" * 100)
        
        # Small threshold should trigger warning
        content = _read_and_validate_context(context_file, size_threshold=50)
        
        assert content == "x" * 100


class TestFormatContextForPrompt:
    """Tests for format_context_for_prompt function."""

    def test_single_line_context(self):
        """Should handle single line context."""
        context = "Food History"
        
        result = format_context_for_prompt(context)
        
        assert result == "Food History"

    def test_multiline_context_joined_with_commas(self):
        """Should join multiple lines with commas."""
        context = "Food History\nWages\nEarly Modern History"
        
        result = format_context_for_prompt(context)
        
        assert result == "Food History, Wages, Early Modern History"

    def test_strips_whitespace_from_lines(self):
        """Should strip whitespace from each line."""
        context = "  Food History  \n  Wages  \n  Early Modern  "
        
        result = format_context_for_prompt(context)
        
        assert result == "Food History, Wages, Early Modern"

    def test_skips_empty_lines(self):
        """Should skip empty lines."""
        context = "Food History\n\nWages\n\n\nEarly Modern"
        
        result = format_context_for_prompt(context)
        
        assert result == "Food History, Wages, Early Modern"

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        result = format_context_for_prompt("")
        
        assert result == ""

    def test_handles_whitespace_only_string(self):
        """Should handle whitespace-only string."""
        result = format_context_for_prompt("   \n\t  \n  ")
        
        assert result == ""


class TestContextSuffix:
    """Tests for context file naming convention."""

    def test_context_suffix_value(self):
        """CONTEXT_SUFFIX should match expected value."""
        assert CONTEXT_SUFFIX == "_summary_context.txt"

    def test_file_specific_naming(self, tmp_path):
        """File-specific context should use correct naming pattern."""
        input_file = tmp_path / "my_document.pdf"
        input_file.touch()
        
        # Create context with correct naming
        expected_context_name = "my_document_summary_context.txt"
        context_file = tmp_path / expected_context_name
        context_file.write_text("Test context")
        
        content, path = resolve_summary_context(
            input_file=input_file,
            global_context_dir=tmp_path / "context"
        )
        
        assert content == "Test context"
        assert path.name == expected_context_name


class TestIntegrationWithPromptUtils:
    """Integration tests with prompt_utils module."""

    def test_context_injection_in_prompt(self):
        """Context should be properly injected into prompts."""
        from modules.prompt_utils import render_prompt_with_schema
        
        prompt = "Instructions:\n- Rule 1\n- Pay attention to: {{CONTEXT}}\n\nThe JSON schema:\n{{SCHEMA}}"
        schema = {"type": "object"}
        context = "Food History, Wages"
        
        result = render_prompt_with_schema(prompt, schema, context=context)
        
        assert "Food History, Wages" in result
        assert "{{CONTEXT}}" not in result

    def test_context_line_removed_when_no_context(self):
        """Context placeholder line should be removed when no context provided."""
        from modules.prompt_utils import render_prompt_with_schema
        
        prompt = "Instructions:\n- Rule 1\n- Pay attention to: {{CONTEXT}}\n\nThe JSON schema:\n{{SCHEMA}}"
        schema = {"type": "object"}
        
        result = render_prompt_with_schema(prompt, schema, context=None)
        
        assert "{{CONTEXT}}" not in result
        assert "Pay attention to:" not in result

    def test_context_line_removed_when_empty_context(self):
        """Context placeholder line should be removed when context is empty string."""
        from modules.prompt_utils import render_prompt_with_schema
        
        prompt = "Instructions:\n- Rule 1\n- Pay attention to: {{CONTEXT}}\n\nThe JSON schema:\n{{SCHEMA}}"
        schema = {"type": "object"}
        
        result = render_prompt_with_schema(prompt, schema, context="")
        
        assert "{{CONTEXT}}" not in result
        assert "Pay attention to:" not in result
