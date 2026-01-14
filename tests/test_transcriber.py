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

    def test_basic_inference_roman_to_arabic(self, processor):
        """Unnumbered page between Roman xii and Arabic 2 should become Arabic 1."""
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
        
        assert result == 1
        # Check the unnumbered page was updated
        inferred_page = parsed_summaries[1]
        assert inferred_page["model_page_number_int"] == 1
        assert inferred_page["page_number_type"] == "arabic"
        assert inferred_page["is_genuinely_unnumbered"] is False

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

    def test_inference_from_roman_now_supported(self, processor):
        """Should now infer from Roman-numbered pages as well as Arabic."""
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
        
        # Now we DO infer from Roman pages
        assert result == 1
        assert parsed_summaries[0]["model_page_number_int"] == 4
        assert parsed_summaries[0]["page_number_type"] == "roman"

    def test_multiple_unnumbered_pages_only_first_inferred(self, processor):
        """When multiple unnumbered pages exist, only one before Arabic can be inferred."""
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
        
        # Only the page at index 1 (immediately before Arabic 2) should be inferred as 1
        assert result == 1
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is True
        assert parsed_summaries[1]["model_page_number_int"] == 1
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is False

    def test_inference_with_unordered_input(self, processor):
        """Inference should work correctly even if input is not sorted by index."""
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
        
        assert result == 1
        # Find the unnumbered page (index 1) and verify it was updated
        unnumbered = next(s for s in parsed_summaries if s["original_input_order_index"] == 1)
        assert unnumbered["model_page_number_int"] == 1
        assert unnumbered["page_number_type"] == "arabic"
        assert unnumbered["is_genuinely_unnumbered"] is False

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

    def test_inference_higher_page_number(self, processor):
        """Should correctly infer page 4 when followed by page 5."""
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
        
        assert result == 1
        assert parsed_summaries[0]["model_page_number_int"] == 4
        assert parsed_summaries[0]["page_number_type"] == "arabic"

    def test_backward_inference_from_preceding_arabic(self, processor):
        """Should infer page 6 when preceded by Arabic page 5."""
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
        
        assert result == 1
        assert parsed_summaries[1]["model_page_number_int"] == 6
        assert parsed_summaries[1]["page_number_type"] == "arabic"
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is False

    def test_forward_inference_from_following_roman(self, processor):
        """Should infer Roman page 4 when followed by Roman page 5."""
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
        
        assert result == 1
        assert parsed_summaries[0]["model_page_number_int"] == 4
        assert parsed_summaries[0]["page_number_type"] == "roman"
        assert parsed_summaries[0]["is_genuinely_unnumbered"] is False

    def test_backward_inference_from_preceding_roman(self, processor):
        """Should infer Roman page 10 when preceded by Roman page 9."""
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
        
        assert result == 1
        assert parsed_summaries[1]["model_page_number_int"] == 10
        assert parsed_summaries[1]["page_number_type"] == "roman"
        assert parsed_summaries[1]["is_genuinely_unnumbered"] is False

    def test_arabic_priority_over_roman(self, processor):
        """Arabic inference should run before Roman, claiming the page first."""
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
        
        # Should be inferred as Arabic 1, not Roman
        assert result == 1
        assert parsed_summaries[0]["model_page_number_int"] == 1
        assert parsed_summaries[0]["page_number_type"] == "arabic"

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
