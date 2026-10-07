"""Tests that no page position is recorded as a printed page number.

Covers the placeholder sites in ``SummaryManager`` and ``PageRunner``, an
adversarial unnumbered page (footnote, figure and section numbers but no
``<page_number>`` tag), the tag parser ``printed_page_numbers``, and the
warning logged when a summary's page number disagrees with the page's tags.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.llm.summary import (
    SummaryManager,
    ensure_page_information,
    placeholder_summary,
)
from autoexcerpter.pipeline import pages as pages_module
from autoexcerpter.pipeline.pages import PageRunner
from autoexcerpter.rendering.page_tags import printed_page_numbers
from tests.llm import helpers as llm
from tests.pipeline.helpers import make_runner, summarizer


def _summary_manager(
    monkeypatch: pytest.MonkeyPatch, answer: str, **phase_fields: Any
) -> SummaryManager:
    """Return a summary manager whose model always answers *answer*."""
    llm.install(monkeypatch, [answer])
    retry = llm.rules("summary", validation_failure=(True, 0, 0.0, 1.0))
    return SummaryManager(llm.phase(**phase_fields), llm.env(retry))


def _runner(tmp_path: Path, summary_manager: Any) -> PageRunner:
    return make_runner(tmp_path, summary_manager=summary_manager)


def _assert_no_printed_number(page_info: dict[str, Any]) -> None:
    assert page_info["page_number_integer"] is None
    assert page_info["page_number_type"] == "none"


# Footnote markers, a figure number, a section number and a year, but no
# <page_number> tag: nothing here is a printed page number.
_UNNUMBERED_PAGE = (
    "## 2.4 Prices and wages\n\n"
    "Grain prices rose sharply after 1709.[^3] As Figure 3 shows, the wage "
    "series of Section 2.4 lags behind by about 12 months.\n\n"
    "[^3]: See chapter 7, note 41.\n"
)


# ============================================================================
# Placeholder sites
# ============================================================================
class TestSummaryManagerPlaceholders:
    def test_placeholder_summary(self) -> None:
        result = placeholder_summary(9, "boom")
        assert result["page"] == 9
        _assert_no_printed_number(result["page_information"])

    def test_missing_page_information(self) -> None:
        summary: dict[str, Any] = {"bullet_points": ["x"], "references": None}
        ensure_page_information(summary)
        _assert_no_printed_number(summary["page_information"])

    def test_partial_page_information(self) -> None:
        summary: dict[str, Any] = {"page_information": {"page_types": ["content"]}}
        ensure_page_information(summary)
        _assert_no_printed_number(summary["page_information"])

    def test_model_page_number_left_alone(self) -> None:
        summary: dict[str, Any] = {
            "page_information": {"page_number_integer": 41, "page_types": ["content"]}
        }
        ensure_page_information(summary)
        assert summary["page_information"]["page_number_integer"] == 41
        assert summary["page_information"]["page_number_type"] == "arabic"

    def test_text_answer_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sm = _summary_manager(
            monkeypatch, "Plain prose summary, not JSON.", response_format="text"
        )

        result = asyncio.run(sm.generate_summary("text", page_num=9))

        assert result["page"] == 9
        assert result["bullet_points"] == ["Plain prose summary, not JSON."]
        assert "error" not in result
        _assert_no_printed_number(result["page_information"])


class TestPageRunnerPlaceholders:
    def test_blank_page_placeholder(self, tmp_path: Path) -> None:
        summary_manager = summarizer()
        runner = _runner(tmp_path, summary_manager)
        entry = {
            "transcription": "[no transcribable text]",
            "page": 5,
            "original_input_order_index": 4,
        }

        result = asyncio.run(runner.summarize(entry, 4, "p5.png"))

        assert result is not None
        summary_manager.generate_summary.assert_not_called()
        assert result["page"] == 5
        assert result["original_input_order_index"] == 4
        _assert_no_printed_number(result["page_information"])

    def test_failed_transcription_placeholder_with_position_page(
        self, tmp_path: Path
    ) -> None:
        # The critical-error entry stores the position as an int "page"; it
        # must not become an arabic printed number.
        runner = _runner(tmp_path, summarizer())
        entry = {
            "transcription": "[critical error]",
            "page": 3,
            "error": "boom",
            "original_input_order_index": 2,
        }

        result = asyncio.run(runner.summarize(entry, 2, "p3.png"))

        assert result is not None
        assert result["page"] == 3
        _assert_no_printed_number(result["page_information"])


# ============================================================================
# Adversarial unnumbered page
# ============================================================================
class TestAdversarialUnnumberedPage:
    def test_no_tag_numbers_found(self) -> None:
        assert printed_page_numbers(_UNNUMBERED_PAGE) == []

    def test_null_summary_stays_null(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sm = _summary_manager(
            monkeypatch,
            json.dumps(
                {
                    "page_information": {
                        "page_number_integer": None,
                        "page_number_type": "none",
                        "page_types": ["content"],
                    },
                    "bullet_points": ["Grain prices rose after 1709."],
                    "references": None,
                }
            ),
        )
        runner = _runner(tmp_path, sm)
        entry = {"transcription": _UNNUMBERED_PAGE, "original_input_order_index": 6}

        with caplog.at_level(logging.WARNING):
            result = asyncio.run(runner.summarize(entry, 6, "p7.png"))

        assert result is not None
        assert result["page"] == 7
        _assert_no_printed_number(result["page_information"])
        messages = [r.getMessage() for r in caplog.records]
        assert not any("keeping the returned value" in m for m in messages)


class TestPageNumberMismatchWarning:
    @staticmethod
    def _summary(value: int | None) -> dict[str, Any]:
        return {
            "page_information": {
                "page_number_integer": value,
                "page_number_type": "arabic" if value is not None else "none",
            }
        }

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (41, "Text <page_number>14</page_number>"),
            (None, "Text <page_number>14</page_number>"),
            (14, "Text without any tag"),
        ],
    )
    def test_mismatch_warns_and_keeps_value(
        self,
        value: int | None,
        text: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        summary = self._summary(value)

        with caplog.at_level(logging.WARNING):
            warned = pages_module.warn_on_page_number_mismatch(summary, text, "p.png")

        assert warned is True
        assert summary["page_information"]["page_number_integer"] == value
        messages = [r.getMessage() for r in caplog.records]
        assert any("keeping the returned value" in m for m in messages)

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (14, "Text <page_number>14</page_number>"),
            (12, "<page_number>xii</page_number>"),
            (None, "Text without any tag"),
        ],
    )
    def test_match_is_silent(self, value: int | None, text: str) -> None:
        summary = self._summary(value)
        assert (
            pages_module.warn_on_page_number_mismatch(summary, text, "p.png") is False
        )

    def test_warning_fires_after_generate_summary(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sm = summarizer(
            return_value={
                "page": 1,
                "page_information": {
                    "page_number_integer": 41,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["x"],
                "references": None,
            }
        )
        runner = _runner(tmp_path, sm)
        entry = {
            "transcription": "Body <page_number>14</page_number>",
            "original_input_order_index": 0,
        }

        with caplog.at_level(logging.WARNING):
            result = asyncio.run(runner.summarize(entry, 0, "p1.png"))

        assert result is not None
        assert result["page_information"]["page_number_integer"] == 41
        assert any("returned page 41" in r.getMessage() for r in caplog.records)
