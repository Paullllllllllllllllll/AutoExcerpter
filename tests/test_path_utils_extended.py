"""Extended tests for modules/path_utils.py - Untested path utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from modules.path_utils import (
    create_safe_log_filename,
    ensure_path_safe,
    MAX_SAFE_NAME_LENGTH,
    HASH_LENGTH,
)


# ============================================================================
# create_safe_log_filename
# ============================================================================
class TestCreateSafeLogFilename:
    """Tests for create_safe_log_filename()."""

    def test_short_name_preserved(self):
        """Short base name is included in full without truncation."""
        result = create_safe_log_filename("short_doc", "transcription")

        assert result.startswith("short_doc-")
        assert result.endswith("_transcription_log.json")
        assert len(result) <= MAX_SAFE_NAME_LENGTH

    def test_long_name_gets_truncated(self):
        """Long base name is truncated to fit within MAX_SAFE_NAME_LENGTH."""
        long_name = "A" * 200
        result = create_safe_log_filename(long_name, "summary")

        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert result.endswith("_summary_log.json")
        # The full 200-char name should not appear
        assert long_name not in result

    def test_hash_present_for_uniqueness(self):
        """Result includes a hash segment for uniqueness."""
        result = create_safe_log_filename("my_document", "transcription")

        # The hash is between the dash after the name and the suffix
        # Format: name-HASH_suffix_log.json
        parts = result.split("-")
        assert len(parts) >= 2
        hash_and_suffix = parts[-1]
        # The hash is the first HASH_LENGTH characters of the suffix portion
        assert len(hash_and_suffix) >= HASH_LENGTH

    def test_different_names_produce_different_hashes(self):
        """Different base names produce different filenames."""
        result_a = create_safe_log_filename("document_alpha", "transcription")
        result_b = create_safe_log_filename("document_beta", "transcription")

        assert result_a != result_b

    def test_same_name_produces_same_hash(self):
        """Same base name always produces the same filename (deterministic)."""
        result_1 = create_safe_log_filename("consistent_name", "summary")
        result_2 = create_safe_log_filename("consistent_name", "summary")

        assert result_1 == result_2

    def test_different_log_types_produce_different_filenames(self):
        """Same base name with different log types produces different filenames."""
        result_trans = create_safe_log_filename("doc", "transcription")
        result_summ = create_safe_log_filename("doc", "summary")

        assert result_trans != result_summ
        assert "_transcription_log.json" in result_trans
        assert "_summary_log.json" in result_summ

    def test_truncation_strips_trailing_punctuation(self):
        """Truncated name has trailing punctuation stripped."""
        # Build a name where the truncation point falls on a period or dash
        suffix = "_transcription_log.json"
        reserved = len(suffix) + HASH_LENGTH + 1
        available = MAX_SAFE_NAME_LENGTH - reserved

        # Make the name end with punctuation right at the truncation boundary
        base = "A" * (available - 1) + "."
        long_name = base + "B" * 50  # Ensure it exceeds available length

        result = create_safe_log_filename(long_name, "transcription")

        # The truncated portion should not end with trailing punctuation
        # before the dash-hash segment
        name_part = result.split("-")[0]
        assert not name_part.endswith(".")
        assert not name_part.endswith("-")
        assert not name_part.endswith("_")

    def test_result_always_within_limit(self):
        """Result length never exceeds MAX_SAFE_NAME_LENGTH, regardless of input."""
        names = [
            "x",
            "normal_document_name",
            "Z" * 500,
            "Beukers et al 2025 Grape (Vitis vinifera) use in the early modern period",
        ]
        for name in names:
            result = create_safe_log_filename(name, "transcription")
            assert (
                len(result) <= MAX_SAFE_NAME_LENGTH
            ), f"Filename too long ({len(result)} chars) for input: {name[:40]}..."


# ============================================================================
# ensure_path_safe
# ============================================================================
class TestEnsurePathSafe:
    """Tests for ensure_path_safe().

    The ``platform`` module is imported locally inside ``ensure_path_safe``,
    so we patch ``platform.system`` at the ``platform`` module level rather
    than at ``modules.path_utils.platform``.
    """

    @patch("platform.system", return_value="Windows")
    def test_windows_returns_resolved_path(self, mock_system, tmp_path: Path):
        """On Windows, returns a resolved (absolute) path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = ensure_path_safe(test_file)

        assert result.is_absolute()
        assert result == test_file.resolve()

    @patch("platform.system", return_value="Windows")
    def test_windows_resolve_failure_returns_original(self, mock_system):
        """On Windows, if resolve() fails, returns the original path."""
        bad_path = Path("\\\\?\\impossible\\path\\that\\wont\\resolve")

        with patch.object(Path, "resolve", side_effect=OSError("resolve failed")):
            result = ensure_path_safe(bad_path)

        assert result == bad_path

    @patch("platform.system", return_value="Linux")
    def test_linux_returns_path_unchanged(self, mock_system, tmp_path: Path):
        """On Linux, the path is returned without modification."""
        test_path = tmp_path / "document.pdf"

        result = ensure_path_safe(test_path)

        assert result == test_path

    @patch("platform.system", return_value="Darwin")
    def test_macos_returns_path_unchanged(self, mock_system, tmp_path: Path):
        """On macOS, the path is returned without modification."""
        test_path = tmp_path / "document.pdf"

        result = ensure_path_safe(test_path)

        assert result == test_path

    @patch("platform.system", return_value="Windows")
    def test_windows_directory_path(self, mock_system, tmp_path: Path):
        """On Windows, directory paths are also resolved."""
        result = ensure_path_safe(tmp_path)

        assert result.is_absolute()
        assert result == tmp_path.resolve()

    def test_returns_path_object(self, tmp_path: Path):
        """Return value is always a Path object."""
        result = ensure_path_safe(tmp_path / "something")

        assert isinstance(result, Path)
