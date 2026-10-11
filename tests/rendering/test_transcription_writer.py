"""The transcription file built by rendering/transcription.py.

Covers the metadata header with its success and failure counts and the given
processing time, page bodies, the placeholder for a result without a
transcription, non-string transcriptions, verbatim CRLF and the single final
newline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoexcerpter.rendering.transcription import transcription_text

PROCESSED = datetime(2026, 10, 11, 9, 30, 5, tzinfo=UTC)


def _text(results: list[dict[str, Any]], name: str = "Doc", **kwargs: Any) -> str:
    return transcription_text(
        results, name, "pdf", Path("/path/to/source.pdf"), PROCESSED, **kwargs
    )


class TestTranscriptionText:
    """Tests for transcription_text."""

    def test_header_and_every_page_body(self) -> None:
        content = _text(
            [
                {"transcription": "Page 1 content here."},
                {"transcription": "Page 2 content here."},
            ],
            "Test Document",
        )

        assert "# Transcription of: Test Document" in content
        assert "# Type: pdf" in content
        assert "# Processed on: 2026-10-11 09:30:05" in content
        assert "# Total images processed: 2" in content
        assert "# Successfully transcribed: 2" in content
        assert "# Failed items: 0" in content
        assert "Total processing time" not in content
        assert "Page 1 content here." in content
        assert "Page 2 content here." in content

    def test_counts_errors_in_results(self) -> None:
        content = _text(
            [
                {"transcription": "Good page."},
                {"error": "API timeout"},
                {"transcription": "Another good page."},
            ]
        )

        assert "# Successfully transcribed: 2" in content
        assert "# Failed items: 1" in content

    def test_notes_follow_the_counts(self) -> None:
        content = _text([{"transcription": "x"}], metadata_notes=["A note."])

        assert "# Failed items: 0\n# Note: A note.\n\n" in content

    def test_missing_transcription_uses_error_placeholder(self) -> None:
        content = _text([{"no_transcription_key": True}])

        assert "[ERROR] Transcription data missing" in content

    def test_empty_results(self) -> None:
        content = _text([])

        assert "# Total images processed: 0" in content
        assert content.endswith("# Failed items: 0\n")

    def test_none_transcription_is_coerced(self) -> None:
        content = _text([{"transcription": None}, {"transcription": "real text"}])

        assert "real text" in content

    def test_crlf_in_transcription_survives_verbatim(self) -> None:
        content = _text([{"transcription": "line1\r\nline2"}])

        assert "\r\r\n" not in content
        assert "line1\r\nline2" in content

    def test_the_text_ends_with_exactly_one_newline(self) -> None:
        for last in ("text", "text\n", "text\n\n\n"):
            content = _text([{"transcription": last}])
            assert content.endswith("text\n")
            assert not content.endswith("\n\n")
