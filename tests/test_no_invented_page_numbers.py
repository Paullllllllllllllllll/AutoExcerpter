"""Tests that no page position is recorded as a printed page number.

Covers the placeholder sites in ``SummaryManager`` and ``ItemTranscriber``, an
adversarial unnumbered page (footnote, figure and section numbers but no
``<page_number>`` tag), the tag parser ``printed_page_numbers``, and the
warning logged when a summary's page number disagrees with the page's tags.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from config import app as app_config
from llm.summary import SummaryManager
from llm.types import CustomEndpointCapabilities
from pipeline import transcriber as transcriber_module
from pipeline.transcriber import ItemTranscriber
from rendering.page_tags import printed_page_numbers


def _make_sm(**overrides: Any) -> SummaryManager:
    """Bare SummaryManager bypassing __init__."""
    sm = SummaryManager.__new__(SummaryManager)
    defaults: dict[str, Any] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "custom_capabilities": None,
        "transcription_was_plain_text": False,
        "_stats_lock": threading.Lock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "rate_limiter": None,
        "model_config": {},
        "summary_context": None,
        "_output_schema": None,
        "schema_retry_config": {},
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(sm, attr, val)
    return sm


def _wire_sm(sm: SummaryManager, text: str) -> None:
    """Stub the API boundary so every call yields *text*."""
    sm._build_model_inputs = MagicMock(return_value=([], {}))  # type: ignore[method-assign]
    sm._get_structured_chat_model = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    sm._invoke_with_retry = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    sm._report_token_usage = MagicMock()  # type: ignore[method-assign]
    sm._extract_output_text = MagicMock(return_value=text)  # type: ignore[method-assign]


def _bare_transcriber(summary_manager: Any) -> ItemTranscriber:
    obj = ItemTranscriber.__new__(ItemTranscriber)
    obj.summary_manager = summary_manager
    return obj


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
        result = _make_sm()._create_placeholder_summary(9, "boom")
        assert result["page"] == 9
        _assert_no_printed_number(result["page_information"])

    def test_missing_page_information(self) -> None:
        summary: dict[str, Any] = {"bullet_points": ["x"], "references": None}
        _make_sm()._ensure_page_information_structure(summary, 9)
        _assert_no_printed_number(summary["page_information"])

    def test_partial_page_information(self) -> None:
        summary: dict[str, Any] = {"page_information": {"page_types": ["content"]}}
        _make_sm()._ensure_page_information_structure(summary, 9)
        _assert_no_printed_number(summary["page_information"])

    def test_model_page_number_left_alone(self) -> None:
        summary: dict[str, Any] = {
            "page_information": {"page_number_integer": 41, "page_types": ["content"]}
        }
        _make_sm()._ensure_page_information_structure(summary, 9)
        assert summary["page_information"]["page_number_integer"] == 41
        assert summary["page_information"]["page_number_type"] == "arabic"

    def test_non_json_wrapper(self) -> None:
        cfg = {
            "validation_failure": {
                "enabled": True,
                "max_attempts": 0,
                "backoff_base": 0.0,
                "backoff_multiplier": 1.0,
            }
        }
        caps = CustomEndpointCapabilities(supports_structured_output=False)
        sm = _make_sm(
            provider="custom", custom_capabilities=caps, schema_retry_config=cfg
        )
        _wire_sm(sm, "Plain prose summary, not JSON.")

        with patch("llm.summary.time.sleep"):
            result = sm.generate_summary("text", page_num=9)

        assert result["page"] == 9
        assert result["bullet_points"] == ["Plain prose summary, not JSON."]
        _assert_no_printed_number(result["page_information"])


class TestItemTranscriberPlaceholders:
    @pytest.fixture(autouse=True)
    def _summarize(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

    def test_blank_page_placeholder(self) -> None:
        summary_manager = MagicMock()
        obj = _bare_transcriber(summary_manager)
        entry = {
            "transcription": "[no transcribable text]",
            "page": 5,
            "original_input_order_index": 4,
        }

        result = obj._summarize_transcription(entry, 4, "p5.png")

        assert result is not None
        summary_manager.generate_summary.assert_not_called()
        assert result["page"] == 5
        assert result["original_input_order_index"] == 4
        _assert_no_printed_number(result["page_information"])

    def test_failed_transcription_placeholder_with_position_page(self) -> None:
        # The critical-error entry stores the position as an int "page"; it
        # must not become an arabic printed number.
        obj = _bare_transcriber(MagicMock())
        entry = {
            "transcription": "[critical error]",
            "page": 3,
            "error": "boom",
            "original_input_order_index": 2,
        }

        result = obj._summarize_transcription(entry, 2, "p3.png")

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
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)
        sm = _make_sm()
        _wire_sm(
            sm,
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
        obj = _bare_transcriber(sm)
        entry = {"transcription": _UNNUMBERED_PAGE, "original_input_order_index": 6}

        with caplog.at_level(logging.WARNING):
            result = obj._summarize_transcription(entry, 6, "p7.png")

        assert result is not None
        assert result["page"] == 7
        _assert_no_printed_number(result["page_information"])
        messages = [r.getMessage() for r in caplog.records]
        assert not any("keeping the returned value" in m for m in messages)


# ============================================================================
# Tag parser and mismatch warning
# ============================================================================
class TestPrintedPageNumbers:
    def test_arabic_and_roman(self) -> None:
        text = "<page_number>xii</page_number> body <page_number>13</page_number>"
        assert printed_page_numbers(text) == [12, 13]

    def test_decorated_tag(self) -> None:
        assert printed_page_numbers("<page_number>- 42 -</page_number>") == [42]

    def test_empty_and_unusable(self) -> None:
        assert printed_page_numbers(None) == []
        assert printed_page_numbers("<page_number>?</page_number>") == []


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
            warned = transcriber_module._warn_on_page_number_mismatch(
                summary, text, "p.png"
            )

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
            transcriber_module._warn_on_page_number_mismatch(summary, text, "p.png")
            is False
        )

    def test_warning_fires_after_generate_summary(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)
        sm = MagicMock()
        sm.generate_summary.return_value = {
            "page": 1,
            "page_information": {
                "page_number_integer": 41,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["x"],
            "references": None,
        }
        obj = _bare_transcriber(sm)
        entry = {
            "transcription": "Body <page_number>14</page_number>",
            "original_input_order_index": 0,
        }

        with caplog.at_level(logging.WARNING):
            result = obj._summarize_transcription(entry, 0, "p1.png")

        assert result is not None
        assert result["page_information"]["page_number_integer"] == 41
        assert any("returned page 41" in r.getMessage() for r in caplog.records)
