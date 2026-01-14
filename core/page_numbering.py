"""Page numbering logic for document processing.

This module handles page number extraction, inference, and adjustment for
documents with mixed Roman numeral (preface) and Arabic (main text) numbering.
"""

from typing import Any, Dict, List, Optional, Tuple

from modules.logger import setup_logger
from modules.roman_numerals import int_to_roman


logger = setup_logger(__name__)

# Constants for page number adjustment
MIN_SEQUENCE_LENGTH_FOR_ANCHOR = 2


class PageNumberProcessor:
	"""Process and adjust page numbers for document summaries.
	
	Handles extraction of page information from API results, inference of
	page numbers for unnumbered pages, and adjustment based on anchor points.
	"""

	def parse_page_information(
		self, summary_result: Dict[str, Any]
	) -> Tuple[Optional[int], str, List[str], bool]:
		"""
		Extract page information from a summary result.
		
		Supports both flat structure (preferred) and legacy nested structure.
		
		Args:
			summary_result: Summary result dictionary.
			
		Returns:
			Tuple of (model_page_number, page_number_type, page_types, is_genuinely_unnumbered).
			page_number_type is one of: 'roman', 'arabic', 'none'.
			page_types is a list of page type classifications.
		"""
		# Try flat structure first (page_information at top level)
		page_info_obj = summary_result.get('page_information')
		
		# Fall back to legacy nested structure if needed
		if not isinstance(page_info_obj, dict) or not page_info_obj:
			summary_container = summary_result.get("summary", {})
			inner_summary = summary_container.get("summary") if isinstance(
				summary_container, dict
			) else None
			
			if isinstance(inner_summary, dict):
				page_info_obj = inner_summary.get('page_information', {})
			elif isinstance(summary_container, dict):
				page_info_obj = summary_container.get('page_information', {})
			else:
				page_info_obj = {}
		
		model_page_num = None
		page_number_type = "none"
		page_types = ["content"]
		is_genuinely_unnumbered = True
		
		if isinstance(page_info_obj, dict) and page_info_obj:
			# New schema format with page_information object
			model_page_num = page_info_obj.get('page_number_integer')
			page_number_type = page_info_obj.get('page_number_type', 'none')
			
			# Handle both page_types (array) and legacy page_type (string)
			raw_page_types = page_info_obj.get('page_types')
			if raw_page_types is None:
				# Fallback to legacy page_type field
				legacy_type = page_info_obj.get('page_type', 'content')
				page_types = [legacy_type] if legacy_type else ["content"]
			elif isinstance(raw_page_types, str):
				page_types = [raw_page_types]
			elif isinstance(raw_page_types, list) and raw_page_types:
				page_types = raw_page_types
			else:
				page_types = ["content"]
			
			# Derive unnumbered status from page_number_type or null page_number_integer
			is_genuinely_unnumbered = (
				page_number_type == "none" or model_page_num is None
			)
			if is_genuinely_unnumbered:
				page_number_type = "none"
		
		return model_page_num, page_number_type, page_types, is_genuinely_unnumbered

	def find_longest_consecutive_sequence(
		self, summaries_with_pages: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""
		Find the longest consecutive sequence of page numbers in document order.
		
		Args:
			summaries_with_pages: List of summary wrappers with page number info.
			
		Returns:
			Longest consecutive sequence of summaries.
		"""
		if not summaries_with_pages:
			return []
		
		longest_sequence = []
		current_sequence = []
		
		for item in summaries_with_pages:
			# Start a new sequence or add to current if consecutive
			if not current_sequence:
				current_sequence = [item]
			elif (
				item["model_page_number_int"] == current_sequence[-1]["model_page_number_int"] + 1
				and item["original_input_order_index"] == current_sequence[-1]["original_input_order_index"] + 1
			):
				# Both page number and document position are consecutive
				current_sequence.append(item)
			else:
				# End of sequence, check if it's longer than our longest
				if len(current_sequence) > len(longest_sequence):
					longest_sequence = current_sequence.copy()
				# Start a new sequence with the current item
				current_sequence = [item]
		
		# Check if the last sequence is the longest
		if len(current_sequence) > len(longest_sequence):
			longest_sequence = current_sequence
		
		return longest_sequence

	def calculate_adjusted_page_number(
		self, 
		original_index: int, 
		anchor_model_page: int, 
		anchor_original_index: int
	) -> int:
		"""
		Calculate adjusted page number based on anchor point.
		
		Args:
			original_index: Original position in document.
			anchor_model_page: Model-detected page number at anchor.
			anchor_original_index: Original position of anchor.
			
		Returns:
			Adjusted page number.
		"""
		offset = original_index - anchor_original_index
		adjusted_page = anchor_model_page + offset
		return max(1, adjusted_page)  # Ensure page number is at least 1

	def _infer_from_following_page(
		self,
		sorted_summaries: List[Dict[str, Any]],
		page_type: str,
		claimed_pages: set
	) -> int:
		"""
		Infer page numbers for gaps between numbered pages.
		
		Only infers if:
		- Current page is unnumbered
		- Previous page is numbered with the same type (creating a gap to fill)
		- Next page is numbered with the same type
		- The gap is exactly one page (prev+1 == next-1)
		
		This conservative approach only fills gaps in sequences, not boundaries.
		
		Args:
			sorted_summaries: Summaries sorted by document order.
			page_type: 'arabic' or 'roman'.
			claimed_pages: Set of already-claimed page numbers (modified in-place).
			
		Returns:
			Number of pages inferred.
		"""
		inferred_count = 0
		
		for i, current in enumerate(sorted_summaries):
			if not current["is_genuinely_unnumbered"]:
				continue
			
			# Need both previous and next pages to exist
			if i == 0 or i + 1 >= len(sorted_summaries):
				continue
			
			prev_page = sorted_summaries[i - 1]
			next_page = sorted_summaries[i + 1]
			
			# Both surrounding pages must be numbered with the same type
			if (
				prev_page["page_number_type"] == page_type
				and prev_page["model_page_number_int"] is not None
				and not prev_page["is_genuinely_unnumbered"]
				and next_page["page_number_type"] == page_type
				and next_page["model_page_number_int"] is not None
				and not next_page["is_genuinely_unnumbered"]
				# Document positions must be consecutive
				and prev_page["original_input_order_index"] == current["original_input_order_index"] - 1
				and next_page["original_input_order_index"] == current["original_input_order_index"] + 1
				# Page numbers must indicate a single-page gap
				and next_page["model_page_number_int"] == prev_page["model_page_number_int"] + 2
			):
				inferred_page = prev_page["model_page_number_int"] + 1
				
				if inferred_page >= 1 and inferred_page not in claimed_pages:
					current["model_page_number_int"] = inferred_page
					current["page_number_type"] = page_type
					current["is_genuinely_unnumbered"] = False
					claimed_pages.add(inferred_page)
					inferred_count += 1
					logger.info(
						f"Inferred {page_type} page {inferred_page} for unnumbered page at "
						f"document position {current['original_input_order_index']} "
						f"(gap between pages {prev_page['model_page_number_int']} and {next_page['model_page_number_int']})"
					)
		
		return inferred_count

	def _infer_from_preceding_page(
		self,
		sorted_summaries: List[Dict[str, Any]],
		page_type: str,
		claimed_pages: set
	) -> int:
		"""
		Infer page numbers for gaps (handled by _infer_from_following_page).
		
		This method is now a no-op since gap filling is handled by the forward pass.
		Kept for API compatibility.
		
		Args:
			sorted_summaries: Summaries sorted by document order.
			page_type: 'arabic' or 'roman'.
			claimed_pages: Set of already-claimed page numbers (modified in-place).
			
		Returns:
			Number of pages inferred (always 0, kept for compatibility).
		"""
		# Gap filling is now handled entirely by _infer_from_following_page
		# which requires both preceding and following pages to be numbered.
		# This conservative approach prevents converting genuinely unnumbered
		# pages at sequence boundaries.
		return 0

	def infer_unnumbered_page_numbers(
		self, parsed_summaries: List[Dict[str, Any]]
	) -> int:
		"""
		Infer page numbers for unnumbered pages based on surrounding context.
		
		Uses multiple passes to maximize inference:
		1. Forward inference from following Arabic page (N-1)
		2. Backward inference from preceding Arabic page (N+1)
		3. Forward inference from following Roman page (N-1)
		4. Backward inference from preceding Roman page (N+1)
		
		Examples:
		- [Preface] Page xii -> [Unnumbered] -> Page 2  =>  infer Arabic page 1
		- Page 5 -> [Unnumbered] -> Page 7  =>  infer Arabic page 6
		- Page viii -> [Unnumbered] -> Page x  =>  infer Roman page ix
		
		Args:
			parsed_summaries: List of parsed summary wrappers.
			
		Returns:
			Number of pages that had their page numbers inferred.
		"""
		if not parsed_summaries:
			return 0
		
		# Sort by document order for sequential analysis
		sorted_summaries = sorted(
			parsed_summaries, key=lambda x: x["original_input_order_index"]
		)
		
		# Build sets of already-claimed page numbers to avoid conflicts
		claimed_arabic_pages = {
			s["model_page_number_int"]
			for s in sorted_summaries
			if s["page_number_type"] == "arabic"
			and s["model_page_number_int"] is not None
			and not s["is_genuinely_unnumbered"]
		}
		claimed_roman_pages = {
			s["model_page_number_int"]
			for s in sorted_summaries
			if s["page_number_type"] == "roman"
			and s["model_page_number_int"] is not None
			and not s["is_genuinely_unnumbered"]
		}
		
		inferred_count = 0
		
		# Pass 1: Forward inference from following Arabic page
		inferred_count += self._infer_from_following_page(
			sorted_summaries, "arabic", claimed_arabic_pages
		)
		
		# Pass 2: Backward inference from preceding Arabic page
		inferred_count += self._infer_from_preceding_page(
			sorted_summaries, "arabic", claimed_arabic_pages
		)
		
		# Pass 3: Forward inference from following Roman page
		inferred_count += self._infer_from_following_page(
			sorted_summaries, "roman", claimed_roman_pages
		)
		
		# Pass 4: Backward inference from preceding Roman page
		inferred_count += self._infer_from_preceding_page(
			sorted_summaries, "roman", claimed_roman_pages
		)
		
		return inferred_count

	def adjust_and_sort_page_numbers(
		self, summary_results: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""
		Adjust page numbers based on model-detected sequences and sort results.
		
		Handles both Roman numeral (preface) and Arabic (main text) page numbering
		systems separately, preserving the original document order.
		
		Args:
			summary_results: List of summary results with page information.
			
		Returns:
			Sorted list of summaries with adjusted page numbers.
		"""
		if not summary_results:
			return []
		
		logger.info("Adjusting page numbers based on model-detected page sequences...")
		
		# Parse page information from all summaries
		parsed_summaries = []
		for r in summary_results:
			model_page_num, page_num_type, page_types, is_unnumbered = self.parse_page_information(r)
			parsed_summaries.append({
				"original_input_order_index": r["original_input_order_index"],
				"model_page_number_int": model_page_num,
				"page_number_type": page_num_type,
				"page_types": page_types,
				"data": r,
				"is_genuinely_unnumbered": is_unnumbered
			})

		# Infer page numbers for unnumbered pages based on surrounding context
		# This must happen before separating pages by type, as it may convert
		# unnumbered pages to Arabic-numbered pages
		inferred_count = self.infer_unnumbered_page_numbers(parsed_summaries)
		if inferred_count > 0:
			logger.info(f"Inferred page numbers for {inferred_count} previously unnumbered page(s)")

		# Separate Roman and Arabic numbered pages
		roman_pages = [
			s for s in parsed_summaries
			if s["model_page_number_int"] is not None 
			and s["page_number_type"] == "roman"
			and not s["is_genuinely_unnumbered"]
		]
		arabic_pages = [
			s for s in parsed_summaries
			if s["model_page_number_int"] is not None 
			and s["page_number_type"] == "arabic"
			and not s["is_genuinely_unnumbered"]
		]
		
		logger.info(
			f"Found {len(roman_pages)} Roman numeral pages, "
			f"{len(arabic_pages)} Arabic numeral pages"
		)
		
		# Determine anchor points for each numbering system
		roman_anchor_page = 1
		roman_anchor_index = 0
		arabic_anchor_page = 1
		arabic_anchor_index = 0
		
		# Process Roman numeral sequence
		if roman_pages:
			roman_pages.sort(key=lambda x: x["original_input_order_index"])
			detected_roman = [s["model_page_number_int"] for s in roman_pages]
			detected_roman_pos = [s["original_input_order_index"] for s in roman_pages]
			logger.info(f"Roman page numbers (in document order): {detected_roman}")
			logger.info(f"At document positions: {detected_roman_pos}")
			
			longest_roman_seq = self.find_longest_consecutive_sequence(roman_pages)
			if longest_roman_seq and len(longest_roman_seq) > MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
				anchor_item = longest_roman_seq[0]
				roman_anchor_page = anchor_item["model_page_number_int"]
				roman_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Roman anchor: page {roman_anchor_page} at index {roman_anchor_index}")
			elif roman_pages:
				anchor_item = roman_pages[0]
				roman_anchor_page = anchor_item["model_page_number_int"]
				roman_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Roman fallback anchor: page {roman_anchor_page} at index {roman_anchor_index}")
		
		# Process Arabic numeral sequence
		if arabic_pages:
			arabic_pages.sort(key=lambda x: x["original_input_order_index"])
			detected_arabic = [s["model_page_number_int"] for s in arabic_pages]
			detected_arabic_pos = [s["original_input_order_index"] for s in arabic_pages]
			logger.info(f"Arabic page numbers (in document order): {detected_arabic}")
			logger.info(f"At document positions: {detected_arabic_pos}")
			
			longest_arabic_seq = self.find_longest_consecutive_sequence(arabic_pages)
			if longest_arabic_seq and len(longest_arabic_seq) > MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
				anchor_item = longest_arabic_seq[0]
				arabic_anchor_page = anchor_item["model_page_number_int"]
				arabic_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Arabic anchor: page {arabic_anchor_page} at index {arabic_anchor_index}")
			elif arabic_pages:
				anchor_item = arabic_pages[0]
				arabic_anchor_page = anchor_item["model_page_number_int"]
				arabic_anchor_index = anchor_item["original_input_order_index"]
				logger.info(f"Arabic fallback anchor: page {arabic_anchor_page} at index {arabic_anchor_index}")
		
		if not roman_pages and not arabic_pages:
			logger.info("No valid model page numbers detected. Using default numbering starting from 1.")

		# Apply page number adjustment to all summaries
		final_ordered_summaries = []
		for s_wrapper in parsed_summaries:
			original_index = s_wrapper["original_input_order_index"]
			page_num_type = s_wrapper["page_number_type"]
			content_page_types = s_wrapper["page_types"]
			data = s_wrapper["data"]

			# Ensure page_information exists at top level (flat structure)
			if "page_information" not in data or not isinstance(
					data["page_information"], dict):
				data["page_information"] = {
					"page_number_integer": None,
					"page_number_type": "none",
					"page_types": content_page_types,
				}

			if s_wrapper["is_genuinely_unnumbered"]:
				data["page_information"]["page_number_integer"] = None
				data["page_information"]["page_number_type"] = "none"
			elif page_num_type == "roman" and roman_pages:
				# Use Roman anchor for adjustment
				adjusted_page = self.calculate_adjusted_page_number(
					original_index, roman_anchor_page, roman_anchor_index)
				adjusted_page = max(1, adjusted_page)
				data["page_information"]["page_number_integer"] = adjusted_page
				data["page_information"]["page_number_type"] = "roman"
			elif page_num_type == "arabic" and arabic_pages:
				# Use Arabic anchor for adjustment
				adjusted_page = self.calculate_adjusted_page_number(
					original_index, arabic_anchor_page, arabic_anchor_index)
				adjusted_page = max(1, adjusted_page)
				data["page_information"]["page_number_integer"] = adjusted_page
				data["page_information"]["page_number_type"] = "arabic"
			else:
				# No valid page type or no anchors - default to Arabic with index-based numbering
				adjusted_page = original_index + 1
				# Try to use Arabic anchor if available
				if arabic_pages:
					adjusted_page = self.calculate_adjusted_page_number(
						original_index, arabic_anchor_page, arabic_anchor_index)
				adjusted_page = max(1, adjusted_page)
				data["page_information"]["page_number_integer"] = adjusted_page
				data["page_information"]["page_number_type"] = page_num_type if page_num_type != "none" else "arabic"
			
			# Preserve page_types from the model
			data["page_information"]["page_types"] = content_page_types

			final_ordered_summaries.append(data)

		# Sort by original input order to preserve document sequence, regardless of page_number
		final_ordered_summaries.sort(
			key=lambda x: x.get("original_input_order_index", float('inf'))
		)
		return final_ordered_summaries
