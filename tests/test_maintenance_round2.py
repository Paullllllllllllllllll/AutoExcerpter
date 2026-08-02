"""Regression tests for the round-2 maintenance sweep (citations + pipeline).

Each test pins a defect fixed in this sweep:

- OpenAlex polite delay applies to every attempted request, not only matches.
- Abbreviated year ranges after an author initial ("Sraffa, P. 1951-73") are
  no longer stripped as page markers.
- ``consolidate()`` is deterministic under mention order (longest variant is
  the survivor).
- An in-text partial citing the reprint year merges into the full reference
  that canonicalized to the original year.
- English "Band"/large Arabic tome numbers no longer parse as volumes when
  they are plausible publication years.
- Non-decomposable Latin letters transliterate instead of mangling tokens.
- Asterisks are stripped from citation comparison text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import pipeline.transcriber as transcriber_module
from config import app as app_config
from config.constants import LOG_FORMAT_VERSION
from llm.base import abort_requested, clear_abort, request_abort
from pipeline.paths import create_safe_directory_name, create_safe_log_filename
from pipeline.resume import (
    ProcessingState,
    ResumeChecker,
    _folder_changed_since_log,
)
from pipeline.transcriber import ItemTranscriber
from rendering.citations import (
    CitationManager,
    _extract_volume,
    _extract_year,
    _fold,
    _token_set,
)


class TestPoliteDelayOnMiss:
    """The polite delay must throttle misses, not only successful matches."""

    def test_sleep_fires_for_unmatched_citations(self) -> None:
        manager = CitationManager()
        manager.add_citations(["Aaa, A. (2001). First Unmatchable. Press."], 1)
        manager.add_citations(["Bbb, B. (2002). Second Unmatchable. Press."], 2)

        with (
            patch.object(
                manager, "_fetch_metadata_from_openalex", return_value=None
            ) as mock_fetch,
            patch("rendering.citations.time.sleep") as mock_sleep,
            patch("rendering.citations._save_persistent_openalex_cache"),
        ):
            manager.enrich_with_metadata(max_requests=10)

        assert mock_fetch.call_count == 2
        assert mock_sleep.call_count == 2


class TestAbbreviatedYearRange:
    """ "Sraffa, P. 1951-73" must keep its year (previously stripped)."""

    def test_abbreviated_range_year_survives(self) -> None:
        assert _extract_year("Sraffa, P. 1951-73. Works. Cambridge.") == 1951

    def test_full_range_year_survives(self) -> None:
        assert _extract_year("Sraffa, P. 1951-1973. Works. Cambridge.") == 1951

    def test_ordinary_page_range_still_stripped(self) -> None:
        assert _extract_year("Some Title, S. 42-58, o.J.") is None


class TestConsolidateOrderIndependence:
    """The merge survivor must not depend on document mention order."""

    _LONG = "Smith, John. (1990). The Complete Economic History of Trade. Press."
    _SHORT = "Smith, J. (1990). The Complete Economic History of Trade."

    def _consolidated_raw(self, first: str, second: str) -> list[str]:
        manager = CitationManager()
        manager.add_citations([first], 1)
        manager.add_citations([second], 2)
        manager.consolidate()
        return sorted(c.raw_text for c in manager.citations.values())

    def test_survivor_is_longest_regardless_of_order(self) -> None:
        forward = self._consolidated_raw(self._LONG, self._SHORT)
        backward = self._consolidated_raw(self._SHORT, self._LONG)
        assert forward == backward
        assert forward == [self._LONG]


class TestReprintPartialResolution:
    """A partial citing the reprint year merges into the canonicalized full."""

    def test_partial_with_reprint_year_is_merged_not_dropped(self) -> None:
        manager = CitationManager()
        manager.add_citations(
            ["Smith, John. 1990 [1890]. *Great Work of Smith*. Somewhere."], 3
        )
        manager.add_citations([("(Smith 1990)", True)], 17)
        manager.consolidate()

        assert len(manager.citations) == 1
        survivor = next(iter(manager.citations.values()))
        assert survivor.year == 1890
        assert {3, 17} <= survivor.pages


class TestVolumeYearGuard:
    """Year-like Arabic numbers must not parse as volume designators."""

    def test_english_band_year_not_a_volume(self) -> None:
        assert _extract_volume("The Band 1968 Story") is None

    def test_large_journal_volume_still_parses(self) -> None:
        assert _extract_volume("Nature, Vol. 585, pp. 10-12") == 585

    def test_german_band_still_parses(self) -> None:
        assert _extract_volume("Gesammelte Werke, Bd. 2, Berlin") == 2


class TestNonDecomposableLetters:
    """ø/æ/ł and friends transliterate instead of breaking token sets."""

    def test_fold_transliterates_o_slash(self) -> None:
        assert _fold("Møller") == "moller"

    def test_token_set_keeps_whole_surname(self) -> None:
        assert _token_set(_fold("Møller")) == {"moller"}

    def test_ligatures(self) -> None:
        assert _fold("Ægir Œuvres Łódź") == "aegir oeuvres lodz"


class TestAsteriskStripped:
    """Markdown italics asterisks are stripped from comparison text."""

    def test_comparison_text_has_no_asterisk(self) -> None:
        manager = CitationManager()
        manager.add_citations(["Doe, J. (2000). *A Starred Title*. Press."], 1)
        citation = next(iter(manager.citations.values()))
        assert "*" not in citation.comparison_text


# ============================================================================
# Pipeline fixes
# ============================================================================


@pytest.fixture
def pipeline_env(
    monkeypatch: pytest.MonkeyPatch, mock_config_loader: MagicMock
) -> MagicMock:
    """Offline ItemTranscriber environment (mocked manager, no summaries)."""
    mock_manager = MagicMock()
    mock_manager_cls = MagicMock(return_value=mock_manager)
    monkeypatch.setattr(transcriber_module, "TranscriptionManager", mock_manager_cls)
    monkeypatch.setattr(
        transcriber_module, "get_config_loader", lambda: mock_config_loader
    )
    monkeypatch.setattr(app_config, "SUMMARIZE", False)
    return mock_manager


class TestItemNameIdentity:
    """ItemTranscriber.name must mirror ItemSpec.output_stem semantics."""

    def test_dotted_image_folder_keeps_full_name(
        self, tmp_path: Path, pipeline_env: MagicMock
    ) -> None:
        folder = tmp_path / "photos.2023"
        folder.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        transcriber = ItemTranscriber(folder, "image_folder", out)
        assert transcriber.name == "photos.2023"
        assert transcriber.output_txt_path.name == "photos.2023.txt"

    def test_pdf_still_drops_extension(
        self, tmp_path: Path, pipeline_env: MagicMock
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        transcriber = ItemTranscriber(tmp_path / "paper.v2.pdf", "pdf", out)
        assert transcriber.name == "paper.v2"


def _write_complete_outputs(
    output_dir: Path, item_name: str, provenance: dict[str, Any]
) -> None:
    """Create a complete-looking output set plus a versioned working log."""
    for suffix in (".txt", ".docx", ".md"):
        (output_dir / f"{item_name}{suffix}").write_text("x", encoding="utf-8")
    working = output_dir / create_safe_directory_name(item_name, "_working_files")
    working.mkdir(parents=True)
    log_path = working / create_safe_log_filename(item_name, "transcription")
    header = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": item_name,
        "total_images": 1,
        "model_name": "test-model",
        "file_provenance": provenance,
    }
    entry = {
        "original_input_order_index": 0,
        "image": "page_001.png",
        "transcription": "text",
    }
    with log_path.open("w", encoding="utf-8", newline="\n") as f:
        for obj in (header, entry):
            f.write(json.dumps(obj) + "\n")


class TestInputChangedBlocksComplete:
    """A changed input must not let existing outputs classify as COMPLETE."""

    def test_changed_size_forces_reprocess(self, tmp_path: Path) -> None:
        source = tmp_path / "book.pdf"
        source.write_bytes(b"new content of a different length")
        out = tmp_path / "out"
        out.mkdir()
        _write_complete_outputs(
            out,
            "book",
            {"source_file": str(source), "size": 5},  # stale size
        )
        result = ResumeChecker(resume_mode="skip").should_skip("book", out)
        assert result.state != ProcessingState.COMPLETE
        assert result.completed_page_indices is None

    def test_unchanged_size_still_complete(self, tmp_path: Path) -> None:
        source = tmp_path / "book.pdf"
        source.write_bytes(b"12345")
        out = tmp_path / "out"
        out.mkdir()
        _write_complete_outputs(
            out,
            "book",
            {"source_file": str(source), "size": 5},
        )
        result = ResumeChecker(resume_mode="skip").should_skip("book", out)
        assert result.state == ProcessingState.COMPLETE


class TestFolderRenameDetected:
    """A rename preserving count and bytes must invalidate folder resume."""

    @staticmethod
    def _names_hash(names: list[str]) -> str:
        return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()

    def test_rename_flagged_as_changed(self, tmp_path: Path) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "b.jpg").write_bytes(b"bb")
        (folder / "z.jpg").write_bytes(b"aa")  # was a.jpg when logged
        provenance = {
            "source_file": str(folder),
            "image_count": 2,
            "total_image_bytes": 4,
            "image_names_sha256": self._names_hash(["a.jpg", "b.jpg"]),
        }
        assert _folder_changed_since_log(folder, provenance) is True

    def test_unchanged_names_pass(self, tmp_path: Path) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(b"aa")
        (folder / "b.jpg").write_bytes(b"bb")
        provenance = {
            "source_file": str(folder),
            "image_count": 2,
            "total_image_bytes": 4,
            "image_names_sha256": self._names_hash(["a.jpg", "b.jpg"]),
        }
        assert _folder_changed_since_log(folder, provenance) is False


class TestAbortEvent:
    """The cooperative abort defers new pages and is resettable."""

    def test_abort_defers_new_pages(
        self, tmp_path: Path, pipeline_env: MagicMock
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        transcriber = ItemTranscriber(tmp_path / "doc.pdf", "pdf", out)
        source = MagicMock()
        request_abort()
        try:
            assert abort_requested() is True
            result = transcriber._process_single_page(0, source, [], [], 1, [0])
        finally:
            clear_abort()
        assert result is None
        source.build_payload.assert_not_called()
        assert abort_requested() is False
