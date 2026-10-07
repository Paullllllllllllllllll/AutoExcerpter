"""The plain-text transcription writer in rendering/transcription.py.

Covers the metadata header with its success and failure counts, page bodies,
the placeholder for a result without a transcription, non-string
transcriptions, verbatim CRLF, the atomic write and write failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.rendering.transcription import write_transcription_to_text


# ============================================================================
# Header and body
# ============================================================================
class TestWriteTranscriptionToText:
    """Tests for write_transcription_to_text."""

    def test_writes_transcription_file(self, tmp_path: Path) -> None:
        """The file carries the metadata header and every page body."""
        output_path = tmp_path / "output.txt"
        results = [
            {"transcription": "Page 1 content here."},
            {"transcription": "Page 2 content here."},
        ]

        success = write_transcription_to_text(
            transcription_results=results,
            output_path=output_path,
            document_name="Test Document",
            item_type="pdf",
            total_elapsed_time=120.5,
            source_path=Path("/path/to/source.pdf"),
        )

        assert success is True
        content = output_path.read_text(encoding="utf-8")
        assert "# Transcription of: Test Document" in content
        assert "# Type: pdf" in content
        assert "# Total images processed: 2" in content
        assert "# Successfully transcribed: 2" in content
        assert "# Failed items: 0" in content
        assert "Page 1 content here." in content
        assert "Page 2 content here." in content

    def test_counts_errors_in_results(self, tmp_path: Path) -> None:
        """Error results are counted as failures."""
        output_path = tmp_path / "output.txt"
        results = [
            {"transcription": "Good page."},
            {"error": "API timeout"},
            {"transcription": "Another good page."},
        ]

        success = write_transcription_to_text(
            transcription_results=results,
            output_path=output_path,
            document_name="Partial",
            item_type="pdf",
            total_elapsed_time=60.0,
            source_path=Path("/path/to/source.pdf"),
        )

        assert success is True
        content = output_path.read_text(encoding="utf-8")
        assert "# Successfully transcribed: 2" in content
        assert "# Failed items: 1" in content

    def test_missing_transcription_uses_error_placeholder(self, tmp_path: Path) -> None:
        output_path = tmp_path / "output.txt"

        write_transcription_to_text(
            transcription_results=[{"no_transcription_key": True}],
            output_path=output_path,
            document_name="Missing",
            item_type="pdf",
            total_elapsed_time=10.0,
            source_path=Path("/path/to/source.pdf"),
        )

        content = output_path.read_text(encoding="utf-8")
        assert "[ERROR] Transcription data missing" in content

    def test_empty_results(self, tmp_path: Path) -> None:
        """No results still give a valid file with zero counts."""
        output_path = tmp_path / "output.txt"

        success = write_transcription_to_text(
            transcription_results=[],
            output_path=output_path,
            document_name="Empty",
            item_type="pdf",
            total_elapsed_time=0.0,
            source_path=Path("/path"),
        )

        assert success is True
        content = output_path.read_text(encoding="utf-8")
        assert "# Total images processed: 0" in content

    def test_none_transcription_is_coerced(self, tmp_path: Path) -> None:
        output_path = tmp_path / "out.txt"
        results: list[dict[str, Any]] = [
            {"transcription": None},
            {"transcription": "real text"},
        ]

        ok = write_transcription_to_text(
            results,
            output_path=output_path,
            document_name="Doc",
            item_type="pdf",
            total_elapsed_time=1.0,
            source_path=tmp_path / "src.pdf",
        )

        assert ok is True
        assert "real text" in output_path.read_text(encoding="utf-8")
        assert not output_path.with_name(output_path.name + ".tmp").exists()

    def test_crlf_in_transcription_survives_verbatim(self, tmp_path: Path) -> None:
        """The writer does not translate newlines, so CRLF is not doubled."""
        out = tmp_path / "book.txt"

        ok = write_transcription_to_text(
            transcription_results=[{"transcription": "line1\r\nline2"}],
            output_path=out,
            document_name="book",
            item_type="PDF",
            total_elapsed_time=1.0,
            source_path=tmp_path / "book.pdf",
        )

        assert ok is True
        data = out.read_bytes()
        assert b"\r\r\n" not in data
        assert b"line1\r\nline2" in data


# ============================================================================
# Atomic write and write failures
# ============================================================================
class TestAtomicWrite:
    """The file is written to a temp sibling and swapped in with os.replace."""

    def test_successful_write_leaves_no_temp_file(self, tmp_path: Path) -> None:
        out = tmp_path / "doc.txt"
        results = [
            {"transcription": "Page one text."},
            {"transcription": "Page two text."},
        ]

        ok = write_transcription_to_text(
            results, out, "doc", "PDF", 1.0, tmp_path / "doc.pdf"
        )

        assert ok is True
        content = out.read_text(encoding="utf-8")
        assert "Page one text." in content
        assert "Page two text." in content
        assert not out.with_name(out.name + ".tmp").exists()

    def test_failed_replace_keeps_previous_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "doc.txt"
        out.write_text("PRIOR COMPLETE OUTPUT", encoding="utf-8")

        def _boom(_src: Any, _dst: Any) -> None:
            raise OSError("replace failed mid-swap")

        monkeypatch.setattr("autoexcerpter.rendering.transcription.os.replace", _boom)

        ok = write_transcription_to_text(
            [{"transcription": "new text"}],
            out,
            "doc",
            "PDF",
            1.0,
            tmp_path / "doc.pdf",
        )

        assert ok is False
        assert out.read_text(encoding="utf-8") == "PRIOR COMPLETE OUTPUT"
        assert not out.with_name(out.name + ".tmp").exists()


class TestWriteFailure:
    """A file that cannot be written returns False."""

    def test_returns_false_on_write_error(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "no_dir" / "sub" / "output.txt"

        success = write_transcription_to_text(
            transcription_results=[{"transcription": "data"}],
            output_path=bad_path,
            document_name="Bad",
            item_type="pdf",
            total_elapsed_time=0.0,
            source_path=Path("/path"),
        )

        assert success is False
