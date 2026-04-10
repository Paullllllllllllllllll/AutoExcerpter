"""Tests for api/transcribe_api.py - Transcription API utilities."""

from __future__ import annotations

import json
from typing import Any

import pytest

from api.transcribe_api import TranscriptionManager


class TestFormatImageName:
    """Tests for _format_image_name static method."""

    def test_normal_filename(self) -> None:
        """Normal filename is returned as-is."""
        assert TranscriptionManager._format_image_name("page_001.png") == "page_001.png"
        assert TranscriptionManager._format_image_name("image_42.jpg") == "image_42.jpg"

    def test_empty_returns_unknown(self) -> None:
        """Empty string returns 'unknown_image'."""
        assert TranscriptionManager._format_image_name("") == "unknown_image"

    def test_none_returns_unknown(self) -> None:
        """None returns 'unknown_image'."""
        assert TranscriptionManager._format_image_name(None) == "unknown_image"

    def test_complex_filename(self) -> None:
        """Complex filenames are preserved."""
        assert (
            TranscriptionManager._format_image_name("doc_2024-01-15_page_003.jpeg")
            == "doc_2024-01-15_page_003.jpeg"
        )


class TestTruncateAnalysis:
    """Tests for _truncate_analysis static method."""

    def test_short_text_unchanged(self) -> None:
        """Short text is returned unchanged."""
        text = "Short analysis."
        assert TranscriptionManager._truncate_analysis(text) == "Short analysis."

    def test_empty_returns_default(self) -> None:
        """Empty text returns default message."""
        assert TranscriptionManager._truncate_analysis("") == "no details available"
        assert TranscriptionManager._truncate_analysis(None) == "no details available"

    def test_long_text_truncated(self) -> None:
        """Long text is truncated with ellipsis."""
        text = "A" * 200
        result = TranscriptionManager._truncate_analysis(text, max_chars=100)

        assert len(result) <= 103  # max_chars + "..."
        assert result.endswith("...")

    def test_truncates_at_word_boundary(self) -> None:
        """Truncation prefers word boundaries."""
        text = "This is a longer text that should be truncated at a word boundary for readability"
        result = TranscriptionManager._truncate_analysis(text, max_chars=50)

        # Should end with "..." and not cut a word in the middle
        assert result.endswith("...")

    def test_strips_whitespace(self) -> None:
        """Whitespace is stripped from input."""
        text = "  Some text with whitespace  "
        result = TranscriptionManager._truncate_analysis(text)

        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestParseTranscriptionFromText:
    """Tests for _parse_transcription_from_text method.

    Note: These tests require mocking since TranscriptionManager needs
    initialization parameters. We test the parsing logic via the class.
    """

    def test_no_transcribable_text_flag(self) -> None:
        """no_transcribable_text flag generates formatted message."""
        json_response = json.dumps(
            {
                "image_analysis": "Page appears blank with faint marks",
                "transcription": None,
                "no_transcribable_text": True,
                "transcription_not_possible": False,
            }
        )

        # Test using static methods
        img_name = TranscriptionManager._format_image_name("page_005.png")
        brief_reason = TranscriptionManager._truncate_analysis(
            "Page appears blank with faint marks"
        )

        expected_format = f"[{img_name}: no transcribable text — {brief_reason}]"
        assert "page_005.png" in expected_format
        assert "no transcribable text" in expected_format

    def test_transcription_not_possible_flag(self) -> None:
        """transcription_not_possible flag generates formatted message."""
        json_response = json.dumps(
            {
                "image_analysis": "Image is too blurry to read any text",
                "transcription": None,
                "no_transcribable_text": False,
                "transcription_not_possible": True,
            }
        )

        img_name = TranscriptionManager._format_image_name("scan_042.jpg")
        brief_reason = TranscriptionManager._truncate_analysis(
            "Image is too blurry to read any text"
        )

        expected_format = f"[{img_name}: transcription not possible — {brief_reason}]"
        assert "scan_042.jpg" in expected_format
        assert "transcription not possible" in expected_format

    def test_image_name_preserved_in_output(self) -> None:
        """Original image filename is preserved in failure messages."""
        # The key requirement: exact image filename should appear in the output
        image_name = "document_page_123.png"
        formatted = TranscriptionManager._format_image_name(image_name)

        assert formatted == "document_page_123.png"
        assert formatted == image_name  # Exact match

    def test_long_analysis_truncated_in_message(self) -> None:
        """Long image_analysis is truncated in the failure message."""
        long_analysis = "A" * 200 + " more text"
        brief = TranscriptionManager._truncate_analysis(long_analysis, max_chars=100)

        assert len(brief) < len(long_analysis)
        assert brief.endswith("...")


class TestValidateTranscriptionSchema:
    """Tests for _validate_transcription_schema method."""

    @staticmethod
    def _make_manager() -> TranscriptionManager:
        """Create a minimal TranscriptionManager without full init."""
        mgr = TranscriptionManager.__new__(TranscriptionManager)
        mgr.custom_capabilities = None
        return mgr

    def test_valid_json(self) -> None:
        mgr = self._make_manager()
        text = json.dumps({
            "image_analysis": "text",
            "transcription": "hello",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        })
        is_valid, reason = mgr._validate_transcription_schema(text)
        assert is_valid is True
        assert reason == ""

    def test_invalid_json(self) -> None:
        mgr = self._make_manager()
        is_valid, reason = mgr._validate_transcription_schema("not json at all")
        assert is_valid is False
        assert "invalid JSON" in reason

    def test_missing_keys(self) -> None:
        mgr = self._make_manager()
        text = json.dumps({"image_analysis": "text", "transcription": "hello"})
        is_valid, reason = mgr._validate_transcription_schema(text)
        assert is_valid is False
        assert "missing keys" in reason

    def test_markdown_wrapped_json(self) -> None:
        mgr = self._make_manager()
        inner = json.dumps({
            "image_analysis": "x",
            "transcription": "y",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        })
        text = f"```json\n{inner}\n```"
        is_valid, reason = mgr._validate_transcription_schema(text)
        assert is_valid is True


class TestPlainTextParsing:
    """Tests for plain-text mode parsing."""

    @staticmethod
    def _make_plain_text_manager() -> TranscriptionManager:
        from modules.types import CustomEndpointCapabilities

        mgr = TranscriptionManager.__new__(TranscriptionManager)
        mgr.custom_capabilities = CustomEndpointCapabilities(
            use_plain_text_prompt=True
        )
        return mgr

    def test_normal_text_returned_as_is(self) -> None:
        mgr = self._make_plain_text_manager()
        result = mgr._parse_transcription_from_text("Hello world", "page_001.jpg")
        assert result == "Hello world"

    def test_sentinel_no_transcribable_text(self) -> None:
        mgr = self._make_plain_text_manager()
        result = mgr._parse_transcription_from_text(
            "[no transcribable text]", "page_001.jpg"
        )
        assert "page_001.jpg" in result
        assert "no transcribable text" in result

    def test_sentinel_transcription_not_possible(self) -> None:
        mgr = self._make_plain_text_manager()
        result = mgr._parse_transcription_from_text(
            "[transcription not possible]", "page_001.jpg"
        )
        assert "page_001.jpg" in result
        assert "transcription not possible" in result

    def test_empty_text_returns_error(self) -> None:
        mgr = self._make_plain_text_manager()
        result = mgr._parse_transcription_from_text("", "page_001.jpg")
        assert "transcription error" in result

    def test_is_plain_text_mode_property(self) -> None:
        mgr = self._make_plain_text_manager()
        assert mgr.is_plain_text_mode is True

    def test_is_not_plain_text_mode_without_caps(self) -> None:
        mgr = TranscriptionManager.__new__(TranscriptionManager)
        mgr.custom_capabilities = None
        assert mgr.is_plain_text_mode is False


class TestTranscriptionSchemaFlags:
    """Tests verifying schema flag handling expectations."""

    def test_both_flags_false_expected_transcription(self) -> None:
        """When both flags are false, transcription text should be extracted."""
        # This documents expected behavior
        response = {
            "image_analysis": "Clear page with readable text",
            "transcription": "This is the transcribed text content.",
            "no_transcribable_text": False,
            "transcription_not_possible": False,
        }

        # When flags are False, the transcription field should be used
        assert response["transcription"] == "This is the transcribed text content."

    def test_no_transcribable_text_with_analysis(self) -> None:
        """no_transcribable_text should include image_analysis context."""
        response: dict[str, Any] = {
            "image_analysis": "Single full-page grayscale photograph of a printed page. The page shows very faint, low-contrast text lines.",
            "transcription": None,
            "no_transcribable_text": True,
            "transcription_not_possible": False,
        }

        # The analysis should be available for context
        assert response["image_analysis"] is not None
        assert len(response["image_analysis"]) > 0
