"""Tests for pipeline.page_numbering: page number inference logic."""

from typing import Any

import pytest

from pipeline.page_numbering import PageNumberProcessor
from rendering.summary import int_to_roman


class TestIntToRoman:
    """Tests for the int_to_roman utility function."""

    def test_basic_numbers(self) -> None:
        """Test basic Roman numeral conversions."""
        assert int_to_roman(1) == "i"
        assert int_to_roman(5) == "v"
        assert int_to_roman(10) == "x"
        assert int_to_roman(50) == "l"
        assert int_to_roman(100) == "c"

    def test_compound_numbers(self) -> None:
        """Test compound Roman numeral conversions."""
        assert int_to_roman(4) == "iv"
        assert int_to_roman(9) == "ix"
        assert int_to_roman(12) == "xii"
        assert int_to_roman(14) == "xiv"

    def test_zero_returns_empty(self) -> None:
        """Zero should return empty string."""
        assert int_to_roman(0) == ""

    def test_negative_returns_empty(self) -> None:
        """Negative numbers should return empty string."""
        assert int_to_roman(-5) == ""


class TestInferUnnumberedPageNumbers:
    """Tests for the infer_unnumbered_page_numbers method."""

    @pytest.fixture
    def processor(self) -> PageNumberProcessor:
        """Create a PageNumberProcessor instance."""
        return PageNumberProcessor()

    def test_empty_list_returns_zero(self, processor) -> None:
        """Empty parsed_summaries should return 0 inferred pages."""
        result = processor.infer_unnumbered_page_numbers([])
        assert result == 0

    def test_no_inference_at_type_boundary(self, processor) -> None:
        """Unnumbered page between Roman and Arabic should NOT be inferred."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 12,
                "page_number_type": "roman",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 2,
                "model_page_number_int": 2,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference at type boundaries - page stays unnumbered
        assert result == 0
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is True

    def test_no_inference_when_page_would_be_zero(self, processor) -> None:
        """Should not infer page 0 (invalid page number)."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # Page 1 - 1 = 0, which is invalid, so no inference
        assert result == 0
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True

    def test_no_inference_when_page_already_claimed(self, processor) -> None:
        """Should not infer if the page number is already used by another page."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 2,
                "model_page_number_int": 2,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # Page 2 - 1 = 1, but page 1 is already claimed
        assert result == 0
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is True

    def test_no_inference_for_non_consecutive_positions(self, processor) -> None:
        """Should not infer if document positions are not consecutive."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 5,  # Not consecutive with index 0
                "model_page_number_int": 2,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        assert result == 0
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True

    def test_no_inference_at_sequence_start(self, processor) -> None:
        """Should NOT infer for unnumbered page at start of sequence (boundary)."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": 5,
                "page_number_type": "roman",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference at start of sequence - stays unnumbered
        assert result == 0
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True

    def test_no_inference_without_surrounding_numbered_pages(self, processor) -> None:
        """Unnumbered pages lacking numbered neighbors on both sides are not
        inferred."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 2,
                "model_page_number_int": 2,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference - pages need numbered pages on BOTH sides
        assert result == 0
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is True

    def test_no_inference_at_type_boundary_unordered(self, processor) -> None:
        """No inference at type boundaries even with unordered input."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 2,
                "model_page_number_int": 2,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 0,
                "model_page_number_int": 12,
                "page_number_type": "roman",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference - boundary between Roman and Arabic
        assert result == 0
        unnumbered = next(
            s for s in parsed_summaries if s["original_input_order_index"] == 1
        )
        assert unnumbered["is_genuinely_unnumbered"] is True

    def test_no_inference_when_next_is_also_unnumbered(self, processor) -> None:
        """Should not infer if the next page is also unnumbered."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        assert result == 0

    def test_no_inference_at_sequence_start_high_index(self, processor) -> None:
        """Should NOT infer at start of sequence even with high page numbers."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 10,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 11,
                "model_page_number_int": 5,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference - needs numbered pages on BOTH sides
        assert result == 0
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True

    def test_no_inference_at_sequence_end(self, processor) -> None:
        """Should NOT infer at end of sequence (needs pages on BOTH sides)."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 5,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference - needs numbered pages on BOTH sides
        assert result == 0
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is True

    def test_no_inference_for_roman_at_start(self, processor) -> None:
        """Should NOT infer for Roman page at start of sequence."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": 5,
                "page_number_type": "roman",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference at start - needs numbered pages on BOTH sides
        assert result == 0
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True

    def test_no_inference_for_roman_at_end(self, processor) -> None:
        """Should NOT infer at end of Roman sequence (needs pages on BOTH sides)."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 9,
                "page_number_type": "roman",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference at end of sequence
        assert result == 0
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is True

    def test_no_inference_at_arabic_start(self, processor) -> None:
        """Should NOT infer at start of Arabic sequence (needs pages on BOTH sides)."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": 2,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # No inference at start of sequence
        assert result == 0
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True

    def test_gap_filling_arabic(self, processor) -> None:
        """Should infer page 6 when between pages 5 and 7."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 5,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 2,
                "model_page_number_int": 7,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        assert result == 1
        assert parsed_summaries[1]["model_page_number_int"] == 6
        assert parsed_summaries[1]["page_number_type"] == "arabic"

    def test_gap_filling_roman(self, processor) -> None:
        """Should infer Roman page 8 when between Roman pages 7 and 9."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 7,
                "page_number_type": "roman",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 2,
                "model_page_number_int": 9,
                "page_number_type": "roman",
                "page_types": ["preface"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        assert result == 1
        assert parsed_summaries[1]["model_page_number_int"] == 8
        assert parsed_summaries[1]["page_number_type"] == "roman"

    def test_no_backward_inference_when_page_claimed(self, processor) -> None:
        """Should not infer backward if page number already exists."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 5,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "data": {},
            },
            {
                "original_input_order_index": 2,
                "model_page_number_int": 6,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        # Page 6 is already claimed, so no inference should happen
        assert result == 0
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is True


class TestAdjustAndSortPageNumbers:
    """Tests for the adjust_and_sort_page_numbers method."""

    @pytest.fixture
    def processor(self) -> PageNumberProcessor:
        """Create a PageNumberProcessor instance."""
        return PageNumberProcessor()

    def _create_summary_result(
        self,
        original_index: int,
        page_number: int | None,
        page_type: str = "arabic",
        page_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Helper to create a summary result dict in flat format (preferred)."""
        if page_types is None:
            page_types = ["content"]

        is_unnumbered = page_number is None or page_type == "none"

        # Flat structure - page_information at top level
        return {
            "original_input_order_index": original_index,
            "page_information": {
                "page_number_integer": page_number,
                "page_number_type": page_type if not is_unnumbered else "none",
                "page_types": page_types,
            },
            "bullet_points": ["Test bullet point"],
        }

    def test_empty_list_returns_empty(self, processor) -> None:
        """Empty input should return empty output."""
        result = processor.adjust_and_sort_page_numbers([])
        assert result == []

    def test_consecutive_arabic_sequence_preserved(self, processor) -> None:
        """A consecutive sequence of Arabic pages should be preserved correctly."""
        summary_results = [
            self._create_summary_result(0, 1, "arabic"),
            self._create_summary_result(1, 2, "arabic"),
            self._create_summary_result(2, 3, "arabic"),
            self._create_summary_result(3, 4, "arabic"),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # Pages should retain their consecutive numbering
        for i, r in enumerate(result):
            page_info = r["page_information"]
            assert page_info["page_number_integer"] == i + 1
            assert page_info["page_number_type"] == "arabic"

    def test_anchor_based_adjustment(self, processor) -> None:
        """Pages should be adjusted based on the longest consecutive sequence."""
        # Simulate a document where model detected:
        # - Page 0: detected as page 5 (wrong)
        # - Pages 1-4: detected as pages 1-4 (correct consecutive sequence)
        # - Page 5: detected as page 10 (wrong)
        summary_results = [
            self._create_summary_result(0, 5, "arabic"),  # Wrong
            self._create_summary_result(1, 1, "arabic"),  # Start of correct sequence
            self._create_summary_result(2, 2, "arabic"),
            self._create_summary_result(3, 3, "arabic"),
            self._create_summary_result(4, 4, "arabic"),  # End of correct sequence
            self._create_summary_result(5, 10, "arabic"),  # Wrong
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # Anchor should be page 1 at index 1
        # All pages should be adjusted relative to this anchor
        expected_pages = [0, 1, 2, 3, 4, 5]  # 1 + (index - 1) for each
        for i, r in enumerate(result):
            page_info = r["page_information"]
            expected = expected_pages[i]
            if expected < 1:
                assert page_info["page_number_integer"] is None
                assert page_info["page_number_type"] == "none"
            else:
                got = page_info["page_number_integer"]
                assert page_info["page_number_integer"] == expected, (
                    f"Page {i} expected {expected}, got {got}"
                )

    def test_roman_and_arabic_separate_anchors(self, processor) -> None:
        """Roman and Arabic pages should use separate anchor points."""
        summary_results = [
            self._create_summary_result(0, 10, "roman", ["preface"]),  # Roman x
            self._create_summary_result(1, 11, "roman", ["preface"]),  # Roman xi
            self._create_summary_result(2, 12, "roman", ["preface"]),  # Roman xii
            self._create_summary_result(3, 1, "arabic", ["content"]),  # Arabic 1
            self._create_summary_result(4, 2, "arabic", ["content"]),  # Arabic 2
            self._create_summary_result(5, 3, "arabic", ["content"]),  # Arabic 3
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # Roman pages should be adjusted with Roman anchor
        for i in range(3):
            page_info = result[i]["page_information"]
            assert page_info["page_number_type"] == "roman"
            assert page_info["page_number_integer"] == 10 + i

        # Arabic pages should be adjusted with Arabic anchor
        for i in range(3, 6):
            page_info = result[i]["page_information"]
            assert page_info["page_number_type"] == "arabic"
            assert page_info["page_number_integer"] == i - 2  # 1, 2, 3

    def test_unnumbered_pages_stay_unnumbered(self, processor) -> None:
        """Pages with no detected page number should remain unnumbered."""
        summary_results = [
            self._create_summary_result(0, None, "none", ["figures_tables_sources"]),
            self._create_summary_result(1, 1, "arabic", ["content"]),
            self._create_summary_result(2, 2, "arabic", ["content"]),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # First page should be unnumbered
        page_info = result[0]["page_information"]
        assert page_info["page_number_integer"] is None
        assert page_info["page_number_type"] == "none"

        # Other pages should be numbered
        assert result[1]["page_information"]["page_number_integer"] == 1
        assert result[2]["page_information"]["page_number_integer"] == 2

    def test_no_inference_at_type_boundary_in_adjustment(self, processor) -> None:
        """Unnumbered pages at type boundaries should stay unnumbered."""
        # Page at index 1 is unnumbered between Roman and Arabic — stays unnumbered.
        summary_results = [
            self._create_summary_result(0, 10, "roman", ["preface"]),
            self._create_summary_result(
                1, None, "none", ["content"]
            ),  # Stays unnumbered
            self._create_summary_result(2, 2, "arabic", ["content"]),
            self._create_summary_result(3, 3, "arabic", ["content"]),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # Page at index 1 should remain unnumbered (boundary between Roman and Arabic)
        page_info = result[1]["page_information"]
        assert page_info["page_number_integer"] is None
        assert page_info["page_number_type"] == "none"

    def test_all_pages_detected_as_same_number(self, processor) -> None:
        """When model detects all pages as the same number, use index-based fallback."""
        # Simulate model incorrectly detecting all pages as page 1
        summary_results = [
            self._create_summary_result(0, 1, "arabic"),
            self._create_summary_result(1, 1, "arabic"),
            self._create_summary_result(2, 1, "arabic"),
            self._create_summary_result(3, 1, "arabic"),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # No consecutive sequence found; fallback anchor is first page (index 0).
        # All pages adjusted relative to this: page = 1 + (index - 0).
        for i, r in enumerate(result):
            page_info = r["page_information"]
            expected = 1 + i  # 1, 2, 3, 4
            assert page_info["page_number_integer"] == expected, (
                f"Page {i} expected {expected}, got {page_info['page_number_integer']}"
            )

    def test_preserves_document_order(self, processor) -> None:
        """Output should be sorted by original_input_order_index."""
        # Input in random order
        summary_results = [
            self._create_summary_result(3, 4, "arabic"),
            self._create_summary_result(0, 1, "arabic"),
            self._create_summary_result(2, 3, "arabic"),
            self._create_summary_result(1, 2, "arabic"),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # Output should be sorted by original index
        for i, r in enumerate(result):
            assert r["original_input_order_index"] == i

    def _create_api_style_result(
        self,
        original_index: int,
        page_number: int | None,
        page_type: str = "arabic",
        page_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a result in flat format (same as _create_summary_result).

        Kept for backward compatibility with existing tests.
        """
        # Now uses the same flat structure as _create_summary_result
        return self._create_summary_result(
            original_index, page_number, page_type, page_types
        )

    def test_api_style_response_consecutive_sequence(self, processor) -> None:
        """Test with API-style responses (flat structure)."""
        summary_results = [
            self._create_api_style_result(0, 1, "arabic"),
            self._create_api_style_result(1, 2, "arabic"),
            self._create_api_style_result(2, 3, "arabic"),
            self._create_api_style_result(3, 4, "arabic"),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # Pages should retain their consecutive numbering
        for i, r in enumerate(result):
            page_info = r["page_information"]
            assert page_info["page_number_integer"] == i + 1, (
                f"Page {i} expected {i + 1}, got {page_info['page_number_integer']}"
            )
            assert page_info["page_number_type"] == "arabic"

    def test_api_style_anchor_adjustment(self, processor) -> None:
        """Test anchor-based adjustment with flat structure responses."""
        summary_results = [
            self._create_api_style_result(0, 5, "arabic"),  # Wrong detection
            self._create_api_style_result(1, 1, "arabic"),  # Start of correct sequence
            self._create_api_style_result(2, 2, "arabic"),
            self._create_api_style_result(3, 3, "arabic"),
            self._create_api_style_result(4, 4, "arabic"),  # End of correct sequence
            self._create_api_style_result(5, 10, "arabic"),  # Wrong detection
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # Anchor should be page 1 at index 1
        # All pages adjusted: page = 1 + (index - 1)
        expected_pages = [0, 1, 2, 3, 4, 5]  # 1 + (index - 1)
        for i, r in enumerate(result):
            page_info = r["page_information"]
            expected = expected_pages[i]
            if expected < 1:
                assert page_info["page_number_integer"] is None
                assert page_info["page_number_type"] == "none"
            else:
                got = page_info["page_number_integer"]
                assert page_info["page_number_integer"] == expected, (
                    f"Page {i} expected {expected}, got {got}"
                )

    def test_api_style_with_preface_and_content(self, processor) -> None:
        """Test mixed Roman (preface) and Arabic (content) pages with flat structure."""
        summary_results = [
            self._create_api_style_result(0, None, "none", ["figures_tables_sources"]),
            self._create_api_style_result(1, None, "none", ["figures_tables_sources"]),
            self._create_api_style_result(2, None, "none", ["preface"]),
            self._create_api_style_result(3, 10, "roman", ["preface"]),
            self._create_api_style_result(4, 11, "roman", ["preface"]),
            self._create_api_style_result(5, 12, "roman", ["preface"]),
            self._create_api_style_result(6, 1, "arabic", ["content"]),
            self._create_api_style_result(7, 2, "arabic", ["content"]),
            self._create_api_style_result(8, 3, "arabic", ["content"]),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        # First 3 pages should be unnumbered
        for i in range(3):
            page_info = result[i]["page_information"]
            assert page_info["page_number_integer"] is None
            assert page_info["page_number_type"] == "none"

        # Pages 3-5 should be Roman x, xi, xii
        for i in range(3, 6):
            page_info = result[i]["page_information"]
            assert page_info["page_number_type"] == "roman"
            assert page_info["page_number_integer"] == 10 + (i - 3)

        # Pages 6-8 should be Arabic 1, 2, 3
        for i in range(6, 9):
            page_info = result[i]["page_information"]
            assert page_info["page_number_type"] == "arabic"
            assert page_info["page_number_integer"] == i - 5


class TestSpreadParsing:
    """Tests for two-page-spread parsing and normalization."""

    @pytest.fixture
    def processor(self) -> PageNumberProcessor:
        return PageNumberProcessor()

    def _result(
        self,
        page_number: int | None,
        is_spread: bool,
        page_end: int | None,
        page_type: str = "arabic",
    ) -> dict[str, Any]:
        return {
            "original_input_order_index": 0,
            "page_information": {
                "page_number_integer": page_number,
                "is_two_page_spread": is_spread,
                "page_number_integer_end": page_end,
                "page_number_type": page_type,
                "page_types": ["content"],
            },
        }

    def test_spread_end_normalized_to_start_plus_one(self, processor) -> None:
        """A spread with a wrong model end is normalized to start + 1."""
        result = self._result(11, True, 99)
        page, _ptype, _ptypes, unnum, spread, end = processor.parse_page_information(
            result
        )
        assert page == 11
        assert spread is True
        assert end == 12
        assert unnum is False

    def test_spread_missing_end_derived(self, processor) -> None:
        """A spread with a null end derives start + 1."""
        result = self._result(11, True, None)
        _page, _ptype, _ptypes, _unnum, spread, end = processor.parse_page_information(
            result
        )
        assert spread is True
        assert end == 12

    def test_non_spread_forces_end_none(self, processor) -> None:
        """A non-spread page never carries an end number."""
        result = self._result(11, False, 12)
        _page, _ptype, _ptypes, _unnum, spread, end = processor.parse_page_information(
            result
        )
        assert spread is False
        assert end is None

    def test_missing_spread_flag_defaults_false(self, processor) -> None:
        """Legacy page_information without the spread flag defaults to single."""
        result = {
            "original_input_order_index": 0,
            "page_information": {
                "page_number_integer": 5,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
        }
        _page, _ptype, _ptypes, _unnum, spread, end = processor.parse_page_information(
            result
        )
        assert spread is False
        assert end is None

    def test_unnumbered_spread_end_none(self, processor) -> None:
        """An unnumbered spread has no start and no end but keeps the flag."""
        result = self._result(None, True, None, page_type="none")
        page, _ptype, _ptypes, unnum, spread, end = processor.parse_page_information(
            result
        )
        assert page is None
        assert unnum is True
        assert spread is True
        assert end is None


class TestSpreadConsecutiveSequence:
    """Tests for span-aware longest-consecutive-sequence detection."""

    @pytest.fixture
    def processor(self) -> PageNumberProcessor:
        return PageNumberProcessor()

    def test_spread_inside_sequence(self, processor) -> None:
        """A spread advances page number and virtual position by two."""
        items = [
            {
                "model_page_number_int": 10,
                "virtual_pos": 0,
                "span": 1,
                "original_input_order_index": 0,
            },
            {
                "model_page_number_int": 11,
                "virtual_pos": 1,
                "span": 2,
                "original_input_order_index": 1,
            },
            {
                "model_page_number_int": 13,
                "virtual_pos": 3,
                "span": 1,
                "original_input_order_index": 2,
            },
        ]
        seq = processor.find_longest_consecutive_sequence(items)
        assert [it["model_page_number_int"] for it in seq] == [10, 11, 13]

    def test_spread_break_when_not_advancing_by_span(self, processor) -> None:
        """A page that does not honor the previous spread's span breaks the run."""
        items = [
            {
                "model_page_number_int": 10,
                "virtual_pos": 0,
                "span": 2,
                "original_input_order_index": 0,
            },
            # Advances by 1 rather than the spread's span of 2 -> breaks.
            {
                "model_page_number_int": 11,
                "virtual_pos": 1,
                "span": 1,
                "original_input_order_index": 1,
            },
            {
                "model_page_number_int": 12,
                "virtual_pos": 2,
                "span": 1,
                "original_input_order_index": 2,
            },
        ]
        seq = processor.find_longest_consecutive_sequence(items)
        # Longest run is the trailing [11, 12] pair.
        assert [it["model_page_number_int"] for it in seq] == [11, 12]


class TestSpreadAdjustment:
    """Tests for anchor adjustment and gap inference across spreads."""

    @pytest.fixture
    def processor(self) -> PageNumberProcessor:
        return PageNumberProcessor()

    def _spread(
        self,
        original_index: int,
        page_number: int | None,
        page_end: int | None,
        is_spread: bool,
        page_type: str = "arabic",
        page_types: list[str] | None = None,
    ) -> dict[str, Any]:
        if page_types is None:
            page_types = ["content"]
        return {
            "original_input_order_index": original_index,
            "page_information": {
                "page_number_integer": page_number,
                "is_two_page_spread": is_spread,
                "page_number_integer_end": page_end,
                "page_number_type": page_type,
                "page_types": page_types,
            },
            "bullet_points": ["bp"],
        }

    def test_anchor_adjustment_across_spread(self, processor) -> None:
        """Pages 10, [11-12 spread], 13 stay consistent after adjustment."""
        summary_results = [
            self._spread(0, 10, None, False),
            self._spread(1, 11, 12, True),
            self._spread(2, 13, None, False),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        assert result[0]["page_information"]["page_number_integer"] == 10
        assert result[0]["page_information"]["page_number_integer_end"] is None
        assert result[0]["page_information"]["is_two_page_spread"] is False

        spread_info = result[1]["page_information"]
        assert spread_info["page_number_integer"] == 11
        assert spread_info["page_number_integer_end"] == 12
        assert spread_info["is_two_page_spread"] is True

        assert result[2]["page_information"]["page_number_integer"] == 13
        assert result[2]["page_information"]["page_number_integer_end"] is None

    def test_anchor_realigns_wrong_spread_start(self, processor) -> None:
        """A spread with a wrong model start is corrected by the section anchor."""
        summary_results = [
            self._spread(0, 10, None, False),
            self._spread(1, 11, None, False),
            self._spread(2, 12, None, False),
            # Model mislabeled this spread's start as 99; anchor should fix to 13.
            self._spread(3, 99, 100, True),
            self._spread(4, 15, None, False),
        ]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        spread_info = result[3]["page_information"]
        assert spread_info["page_number_integer"] == 13
        assert spread_info["page_number_integer_end"] == 14
        assert result[4]["page_information"]["page_number_integer"] == 15

    def test_gap_inference_around_spread(self, processor) -> None:
        """An unnumbered spread between 5 and 8 is inferred as 6 (occupying 6-7)."""
        parsed_summaries: list[dict[str, Any]] = [
            {
                "original_input_order_index": 0,
                "model_page_number_int": 5,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "span": 1,
                "data": {},
            },
            {
                "original_input_order_index": 1,
                "model_page_number_int": None,
                "page_number_type": "none",
                "page_types": ["content"],
                "is_genuinely_unnumbered": True,
                "span": 2,
                "data": {},
            },
            {
                "original_input_order_index": 2,
                "model_page_number_int": 8,
                "page_number_type": "arabic",
                "page_types": ["content"],
                "is_genuinely_unnumbered": False,
                "span": 1,
                "data": {},
            },
        ]

        result = processor.infer_unnumbered_page_numbers(parsed_summaries)

        assert result == 1
        assert parsed_summaries[1]["model_page_number_int"] == 6
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is False


class TestSectionMedianOrdering:
    """Tests for section-median final ordering."""

    @pytest.fixture
    def processor(self) -> PageNumberProcessor:
        return PageNumberProcessor()

    def _page(self, original_index: int, page_types: list[str]) -> dict[str, Any]:
        return {
            "original_input_order_index": original_index,
            "page_information": {
                "page_number_integer": None,
                "is_two_page_spread": False,
                "page_number_integer_end": None,
                "page_number_type": "none",
                "page_types": page_types,
            },
            "bullet_points": ["bp"],
        }

    def test_straggler_grouped_with_section(self, processor) -> None:
        """A straggler page groups with its section, keeping in-section scan order.

        Appendix pages sit mostly at indices 10-12 with a single straggler at
        index 0. The section median (10.5) keeps appendix ordered AFTER content
        (median 5), even though a min-based rank would wrongly float appendix
        first. In-section scan order is preserved.
        """
        summary_results = [self._page(0, ["appendix"])]
        summary_results += [self._page(i, ["content"]) for i in range(1, 10)]
        summary_results += [self._page(i, ["appendix"]) for i in range(10, 13)]

        result = processor.adjust_and_sort_page_numbers(summary_results)

        order = [r["original_input_order_index"] for r in result]
        assert order == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12]
