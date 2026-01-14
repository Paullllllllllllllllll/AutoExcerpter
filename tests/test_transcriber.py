"""Tests for core.page_numbering module, specifically page number inference logic."""

import pytest

from core.page_numbering import PageNumberProcessor
from modules.roman_numerals import int_to_roman


class TestIntToRoman:
    """Tests for the int_to_roman utility function."""

    def test_basic_numbers(self):
        """Test basic Roman numeral conversions."""
        assert int_to_roman(1) == "i"
        assert int_to_roman(5) == "v"
        assert int_to_roman(10) == "x"
        assert int_to_roman(50) == "l"
        assert int_to_roman(100) == "c"

    def test_compound_numbers(self):
        """Test compound Roman numeral conversions."""
        assert int_to_roman(4) == "iv"
        assert int_to_roman(9) == "ix"
        assert int_to_roman(12) == "xii"
        assert int_to_roman(14) == "xiv"

    def test_zero_returns_empty(self):
        """Zero should return empty string."""
        assert int_to_roman(0) == ""

    def test_negative_returns_empty(self):
        """Negative numbers should return empty string."""
        assert int_to_roman(-5) == ""


class TestInferUnnumberedPageNumbers:
    """Tests for the infer_unnumbered_page_numbers method."""

    @pytest.fixture
    def processor(self):
        """Create a PageNumberProcessor instance."""
        return PageNumberProcessor()

    def test_empty_list_returns_zero(self, processor):
        """Empty parsed_summaries should return 0 inferred pages."""
        result = processor.infer_unnumbered_page_numbers([])
        assert result == 0

    def test_no_inference_at_type_boundary(self, processor):
        """Unnumbered page between Roman and Arabic should NOT be inferred (boundary)."""
        parsed_summaries = [
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

    def test_no_inference_when_page_would_be_zero(self, processor):
        """Should not infer page 0 (invalid page number)."""
        parsed_summaries = [
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

    def test_no_inference_when_page_already_claimed(self, processor):
        """Should not infer if the page number is already used by another page."""
        parsed_summaries = [
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

    def test_no_inference_for_non_consecutive_positions(self, processor):
        """Should not infer if document positions are not consecutive."""
        parsed_summaries = [
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

    def test_no_inference_at_sequence_start(self, processor):
        """Should NOT infer for unnumbered page at start of sequence (boundary)."""
        parsed_summaries = [
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

    def test_no_inference_without_surrounding_numbered_pages(self, processor):
        """Unnumbered pages without numbered pages on BOTH sides should not be inferred."""
        parsed_summaries = [
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

    def test_no_inference_at_type_boundary_unordered(self, processor):
        """No inference at type boundaries even with unordered input."""
        parsed_summaries = [
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
        unnumbered = next(s for s in parsed_summaries if s["original_input_order_index"] == 1)
        assert unnumbered["is_genuinely_unnumbered"] is True

    def test_no_inference_when_next_is_also_unnumbered(self, processor):
        """Should not infer if the next page is also unnumbered."""
        parsed_summaries = [
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

    def test_no_inference_at_sequence_start_high_index(self, processor):
        """Should NOT infer at start of sequence even with high page numbers."""
        parsed_summaries = [
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

    def test_no_inference_at_sequence_end(self, processor):
        """Should NOT infer at end of sequence (needs pages on BOTH sides)."""
        parsed_summaries = [
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

    def test_no_inference_for_roman_at_start(self, processor):
        """Should NOT infer for Roman page at start of sequence."""
        parsed_summaries = [
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

    def test_no_inference_for_roman_at_end(self, processor):
        """Should NOT infer at end of Roman sequence (needs pages on BOTH sides)."""
        parsed_summaries = [
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

    def test_no_inference_at_arabic_start(self, processor):
        """Should NOT infer at start of Arabic sequence (needs pages on BOTH sides)."""
        parsed_summaries = [
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

    def test_gap_filling_arabic(self, processor):
        """Should infer page 6 when between pages 5 and 7."""
        parsed_summaries = [
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

    def test_gap_filling_roman(self, processor):
        """Should infer Roman page 8 when between Roman pages 7 and 9."""
        parsed_summaries = [
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

    def test_no_backward_inference_when_page_claimed(self, processor):
        """Should not infer backward if page number already exists."""
        parsed_summaries = [
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
    def processor(self):
        """Create a PageNumberProcessor instance."""
        return PageNumberProcessor()

    def _create_summary_result(
        self, 
        original_index: int, 
        page_number: int | None, 
        page_type: str = "arabic",
        page_types: list[str] | None = None
    ) -> dict:
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

    def test_empty_list_returns_empty(self, processor):
        """Empty input should return empty output."""
        result = processor.adjust_and_sort_page_numbers([])
        assert result == []

    def test_consecutive_arabic_sequence_preserved(self, processor):
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

    def test_anchor_based_adjustment(self, processor):
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
                assert page_info["page_number_integer"] == expected, f"Page {i} expected {expected}, got {page_info['page_number_integer']}"

    def test_roman_and_arabic_separate_anchors(self, processor):
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

    def test_unnumbered_pages_stay_unnumbered(self, processor):
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

    def test_no_inference_at_type_boundary_in_adjustment(self, processor):
        """Unnumbered pages at type boundaries should stay unnumbered."""
        # Page at index 1 is unnumbered between Roman and Arabic - should stay unnumbered
        summary_results = [
            self._create_summary_result(0, 10, "roman", ["preface"]),
            self._create_summary_result(1, None, "none", ["content"]),  # Stays unnumbered
            self._create_summary_result(2, 2, "arabic", ["content"]),
            self._create_summary_result(3, 3, "arabic", ["content"]),
        ]
        
        result = processor.adjust_and_sort_page_numbers(summary_results)
        
        # Page at index 1 should remain unnumbered (boundary between Roman and Arabic)
        page_info = result[1]["page_information"]
        assert page_info["page_number_integer"] is None
        assert page_info["page_number_type"] == "none"

    def test_all_pages_detected_as_same_number(self, processor):
        """When model detects all pages as the same number, use index-based fallback."""
        # Simulate model incorrectly detecting all pages as page 1
        summary_results = [
            self._create_summary_result(0, 1, "arabic"),
            self._create_summary_result(1, 1, "arabic"),
            self._create_summary_result(2, 1, "arabic"),
            self._create_summary_result(3, 1, "arabic"),
        ]
        
        result = processor.adjust_and_sort_page_numbers(summary_results)
        
        # No consecutive sequence found, so fallback anchor is first page (page 1 at index 0)
        # All pages should be adjusted relative to this: page = 1 + (index - 0)
        for i, r in enumerate(result):
            page_info = r["page_information"]
            expected = 1 + i  # 1, 2, 3, 4
            assert page_info["page_number_integer"] == expected, f"Page {i} expected {expected}, got {page_info['page_number_integer']}"

    def test_preserves_document_order(self, processor):
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
        page_types: list[str] | None = None
    ) -> dict:
        """Create a result in flat format (same as _create_summary_result).
        
        Kept for backward compatibility with existing tests.
        """
        # Now uses the same flat structure as _create_summary_result
        return self._create_summary_result(original_index, page_number, page_type, page_types)

    def test_api_style_response_consecutive_sequence(self, processor):
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
            assert page_info["page_number_integer"] == i + 1, f"Page {i} expected {i+1}, got {page_info['page_number_integer']}"
            assert page_info["page_number_type"] == "arabic"

    def test_api_style_anchor_adjustment(self, processor):
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
                assert page_info["page_number_integer"] == expected, \
                    f"Page {i} expected {expected}, got {page_info['page_number_integer']}"

    def test_api_style_with_preface_and_content(self, processor):
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
