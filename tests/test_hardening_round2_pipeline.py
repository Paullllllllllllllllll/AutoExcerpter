"""Regression tests for hardening round 2.

Each test targets one fix applied in this round:

- fix 1: a budget-deferred completed page is re-logged (so resume works) yet
  withheld from the in-memory results.
- fix 3: a lone currency "$" followed by a digit is not treated as unclosed
  inline math.
- fix 5: promoting a longer variant to canonical re-derives the survivor's
  comparison_text while leaving its normalized_key (the dict key) untouched.
- fix 6: sanitize_for_xml strips XML-1.0-illegal codepoints beyond C0/DEL.
- fix 8: a 0/negative target_dpi config is clamped to >= 1.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import pipeline.transcriber as transcriber
from config import app as app_config
from imaging.payload import PdfPayloadSource
from pipeline.text_cleaner import balance_dollar_signs
from pipeline.transcriber import ItemTranscriber
from rendering.citations import Citation, CitationManager
from rendering.summary import sanitize_for_xml


# ---------------------------------------------------------------------------
# fix 3: currency-dollar guard in balance_dollar_signs
# ---------------------------------------------------------------------------
class TestBalanceDollarSignsCurrencyGuard:
    def test_single_currency_dollar_left_untouched(self) -> None:
        text = "The book cost $3 in 1850."
        assert balance_dollar_signs(text) == text

    def test_single_currency_dollar_no_spurious_close(self) -> None:
        # Neither a closing "$" is inserted nor the "$" removed.
        text = "It sold for $5 at auction."
        assert balance_dollar_signs(text) == text

    def test_genuine_unclosed_math_still_closed(self) -> None:
        # A lone "$" NOT followed by a digit is still treated as open math.
        result = balance_dollar_signs("$x + y")
        assert result == "$x + y$"

    def test_even_count_unchanged(self) -> None:
        text = "Compute $x + y$ then stop."
        assert balance_dollar_signs(text) == text


# ---------------------------------------------------------------------------
# fix 5: survivor derived fields refreshed on canonical promotion
# ---------------------------------------------------------------------------
class TestMergeIntoRederivesFields:
    def test_longer_variant_refreshes_comparison_text_not_key(self) -> None:
        survivor = Citation(raw_text="Smith, Cooking.")
        other = Citation(
            raw_text="Smith, John. The Art of Cooking. London: Test Press, 1850."
        )

        original_key = survivor.normalized_key

        CitationManager._merge_into(survivor, other)

        # raw_text promoted to the longer variant.
        assert survivor.raw_text == other.raw_text
        # comparison_text now reflects the promoted raw_text (recomputed).
        assert survivor.comparison_text == other.comparison_text
        assert "art" in survivor.comparison_text
        # Derived discriminators re-extracted from the new text.
        assert survivor.year == 1850
        # The stored dict key must NOT change.
        assert survivor.normalized_key == original_key


# ---------------------------------------------------------------------------
# fix 6: sanitize_for_xml removes XML-1.0-illegal codepoints
# ---------------------------------------------------------------------------
class TestSanitizeForXml:
    def test_removes_noncharacter_fffe(self) -> None:
        assert sanitize_for_xml("ab￾cd") == "abcd"

    def test_removes_surrogate(self) -> None:
        assert sanitize_for_xml("ab\ud800cd") == "abcd"

    def test_removes_fdd0_noncharacter_block(self) -> None:
        assert sanitize_for_xml("x﷐y﷯z") == "xyz"

    def test_preserves_normal_text(self) -> None:
        text = "Normal text with accents: café — dash."
        assert sanitize_for_xml(text) == text

    def test_still_strips_c0_controls(self) -> None:
        assert sanitize_for_xml("a\x00b\x07c") == "abc"


# ---------------------------------------------------------------------------
# fix 8: target_dpi clamped to >= 1
# ---------------------------------------------------------------------------
class TestTargetDpiClamp:
    def test_zero_dpi_clamped_to_one(
        self,
        make_pdf: Any,
        mock_config_loader: MagicMock,
        mock_image_processing_config: dict[str, Any],
    ) -> None:
        mock_image_processing_config["api_image_processing"]["target_dpi"] = 0
        pdf_path = make_pdf("one.pdf", num_pages=1)
        with patch(
            "imaging.payload.get_config_loader", return_value=mock_config_loader
        ):
            source = PdfPayloadSource(pdf_path)
            with source:
                assert source.target_dpi == 1

    def test_negative_dpi_clamped_to_one(
        self,
        make_pdf: Any,
        mock_config_loader: MagicMock,
        mock_image_processing_config: dict[str, Any],
    ) -> None:
        mock_image_processing_config["api_image_processing"]["target_dpi"] = -50
        pdf_path = make_pdf("one.pdf", num_pages=1)
        with patch(
            "imaging.payload.get_config_loader", return_value=mock_config_loader
        ):
            source = PdfPayloadSource(pdf_path)
            with source:
                assert source.target_dpi == 1


# ---------------------------------------------------------------------------
# fix 1: budget-deferred completed page is re-logged yet withheld
# ---------------------------------------------------------------------------
def _bare_transcriber() -> ItemTranscriber:
    """Construct an ItemTranscriber without running __init__."""
    return ItemTranscriber.__new__(ItemTranscriber)


class TestReloadCompletedPagesDeferredRelog:
    def test_deferred_page_relogged_but_withheld(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj.completed_page_indices = {0}
        obj.total_items_to_transcribe = 1
        obj.log_path = tmp_path / "trans.log"
        obj.summary_log_path = tmp_path / "sum.log"
        obj._prior_transcription_results = [
            {"original_input_order_index": 0, "image": "p0", "transcription": "text"}
        ]
        obj._prior_summary_results = []  # no logged summary -> needs generation
        obj.summary_manager = MagicMock()
        # Budget already exhausted: the completed page must defer.
        obj._budget_exhausted = threading.Event()
        obj._budget_exhausted.set()

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        logged: list[tuple[Path, dict[str, Any]]] = []
        monkeypatch.setattr(
            transcriber,
            "append_to_log",
            lambda path, entry: logged.append((path, entry)),
        )

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        obj._reload_completed_pages(t_results, s_results)

        # Withheld from in-memory results (page still counts as deferred).
        assert t_results == []
        assert s_results == []
        # But the completed transcription was re-logged for resume.
        assert len(logged) == 1
        path, entry = logged[0]
        assert path == obj.log_path
        assert entry["original_input_order_index"] == 0

    def test_phantom_index_beyond_page_count_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stale log entry whose idx exceeds the current page count must be
        # skipped entirely (neither logged nor counted).
        obj = _bare_transcriber()
        obj.completed_page_indices = {0, 5}
        obj.total_items_to_transcribe = 1  # input swapped for a shorter file
        obj.log_path = tmp_path / "trans.log"
        obj.summary_log_path = tmp_path / "sum.log"
        obj._prior_transcription_results = [
            {"original_input_order_index": 5, "image": "p5", "transcription": "stale"}
        ]
        obj._prior_summary_results = []
        obj.summary_manager = None
        obj._budget_exhausted = threading.Event()

        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)

        logged: list[tuple[Path, dict[str, Any]]] = []
        monkeypatch.setattr(
            transcriber,
            "append_to_log",
            lambda path, entry: logged.append((path, entry)),
        )

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        obj._reload_completed_pages(t_results, s_results)

        assert t_results == []
        assert logged == []
