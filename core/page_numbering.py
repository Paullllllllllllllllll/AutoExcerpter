"""Page numbering logic for document processing.

This module handles page number extraction, inference, and adjustment for
documents with mixed Roman numeral (preface) and Arabic (main text) numbering.

The key algorithm is per-section anchor-based adjustment:
1. Group pages by section type (content, preface, abstract, appendix, etc.)
2. For each section, find the longest consecutive sequence of model-detected page numbers
3. Use that sequence as the anchor and adjust all pages in that section accordingly
4. Conservatively infer page numbers for isolated unnumbered pages between numbered pages
"""

from typing import Any, Dict, List, Optional, Tuple

from modules.logger import setup_logger
from modules.roman_numerals import int_to_roman


logger = setup_logger(__name__)

# Constants for page number adjustment
MIN_SEQUENCE_LENGTH_FOR_ANCHOR = 2

# Section types that get full bullet-point summaries (and need per-section anchoring)
SUMMARY_SECTION_TYPES = {"content", "preface", "abstract", "appendix", "figures_tables_sources"}


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
        
        longest_sequence: list[dict[str, Any]] = []
        current_sequence: list[dict[str, Any]] = []
        
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
        return adjusted_page

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

    def infer_unnumbered_page_numbers(
        self, parsed_summaries: List[Dict[str, Any]]
    ) -> int:
        """
        Infer page numbers for unnumbered pages based on surrounding context.
        
        Uses multiple passes to maximize inference:
        1. Forward inference from following Arabic page (N-1)
        2. Forward inference from following Roman page (N-1)
        
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
        
        # Pass 2: Forward inference from following Roman page
        inferred_count += self._infer_from_following_page(
            sorted_summaries, "roman", claimed_roman_pages
        )
        
        return inferred_count

    def _get_primary_section_type(self, page_types: List[str]) -> str:
        """
        Get the primary section type for a page.
        
        Any page containing "content" in its page_types is treated as a content page
        for page numbering and section ordering purposes. This ensures content pages
        share the same anchor even if they have multiple type classifications.
        
        Priority order: content > preface > abstract > appendix > figures_tables_sources
        
        Args:
            page_types: List of page type classifications.
            
        Returns:
            Primary section type string.
        """
        # Content takes absolute priority - any page with "content" is a content page
        if "content" in page_types:
            return "content"
        
        # For non-content pages, use priority order
        priority_order = ["preface", "abstract", "appendix", "figures_tables_sources"]
        for section in priority_order:
            if section in page_types:
                return section
        return page_types[0] if page_types else "content"

    def _find_section_anchor(
        self, 
        section_pages: List[Dict[str, Any]]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find the anchor point for a section based on longest consecutive sequence.
        
        Args:
            section_pages: List of parsed summaries for this section (sorted by doc order).
            
        Returns:
            Tuple of (anchor_page_number, anchor_document_index) or (None, None) if no anchor.
        """
        if not section_pages:
            return None, None
        
        # Get only pages with valid page numbers (not unnumbered)
        numbered_pages = [
            p for p in section_pages
            if p["model_page_number_int"] is not None
            and not p["is_genuinely_unnumbered"]
        ]
        
        if not numbered_pages:
            return None, None
        
        # Find longest consecutive sequence
        longest_seq = self.find_longest_consecutive_sequence(numbered_pages)
        
        if longest_seq and len(longest_seq) >= MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
            anchor_item = longest_seq[0]
            return anchor_item["model_page_number_int"], anchor_item["original_input_order_index"]
        elif numbered_pages:
            # Fallback to first numbered page
            anchor_item = numbered_pages[0]
            return anchor_item["model_page_number_int"], anchor_item["original_input_order_index"]
        
        return None, None

    def adjust_and_sort_page_numbers(
        self, summary_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Adjust page numbers based on per-section anchor logic.
        
        For each section type (content, preface, abstract, appendix, figures_tables_sources),
        finds the longest consecutive sequence of model-detected page numbers and uses that
        as the anchor for adjusting all pages in that section.
        
        Args:
            summary_results: List of summary results with page information.
            
        Returns:
            Sorted list of summaries with adjusted page numbers.
        """
        if not summary_results:
            return []
        
        logger.info("Adjusting page numbers using per-section anchor logic...")
        
        # Parse page information from all summaries
        parsed_summaries = []
        for r in summary_results:
            model_page_num, page_num_type, page_types, is_unnumbered = self.parse_page_information(r)
            primary_section = self._get_primary_section_type(page_types)
            parsed_summaries.append({
                "original_input_order_index": r["original_input_order_index"],
                "model_page_number_int": model_page_num,
                "page_number_type": page_num_type,
                "page_types": page_types,
                "primary_section": primary_section,
                "data": r,
                "is_genuinely_unnumbered": is_unnumbered
            })

        # Sort by document order for sequential processing
        parsed_summaries.sort(key=lambda x: x["original_input_order_index"])

        # Group pages by section type
        section_groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in parsed_summaries:
            section = p["primary_section"]
            if section not in section_groups:
                section_groups[section] = []
            section_groups[section].append(p)
        
        logger.info(f"Section distribution: {', '.join(f'{k}: {len(v)} pages' for k, v in section_groups.items())}")

        # Find anchor for each section
        section_anchors: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        for section, pages in section_groups.items():
            # Sort section pages by document order
            pages.sort(key=lambda x: x["original_input_order_index"])
            anchor_page, anchor_index = self._find_section_anchor(pages)
            section_anchors[section] = (anchor_page, anchor_index)
            if anchor_page is not None:
                logger.info(f"Section '{section}' anchor: page {anchor_page} at document index {anchor_index}")
            else:
                logger.info(f"Section '{section}': no valid anchor found")

        # Infer page numbers for isolated unnumbered pages between numbered pages
        # Do this AFTER we have section anchors, but BEFORE final adjustment
        inferred_count = self.infer_unnumbered_page_numbers(parsed_summaries)
        if inferred_count > 0:
            logger.info(f"Inferred page numbers for {inferred_count} isolated unnumbered page(s)")

        # Apply per-section page number adjustment
        final_ordered_summaries = []
        for p in parsed_summaries:
            original_index = p["original_input_order_index"]
            page_num_type = p["page_number_type"]
            content_page_types = p["page_types"]
            primary_section = p["primary_section"]
            data = p["data"]

            # Ensure page_information exists at top level (flat structure)
            if "page_information" not in data or not isinstance(data["page_information"], dict):
                data["page_information"] = {
                    "page_number_integer": None,
                    "page_number_type": "none",
                    "page_types": content_page_types,
                }

            # Get section anchor
            anchor_page, anchor_index = section_anchors.get(primary_section, (None, None))

            if p["is_genuinely_unnumbered"]:
                # Genuinely unnumbered page - keep as unnumbered
                data["page_information"]["page_number_integer"] = None
                data["page_information"]["page_number_type"] = "none"
            elif anchor_page is not None and anchor_index is not None:
                # Calculate adjusted page number using section anchor
                adjusted_page = self.calculate_adjusted_page_number(
                    original_index, anchor_page, anchor_index
                )
                if adjusted_page < 1:
                    # Invalid page number - mark as unnumbered
                    data["page_information"]["page_number_integer"] = None
                    data["page_information"]["page_number_type"] = "none"
                else:
                    data["page_information"]["page_number_integer"] = adjusted_page
                    # Preserve original number type (roman/arabic) from model
                    data["page_information"]["page_number_type"] = page_num_type if page_num_type != "none" else "arabic"
            else:
                # No anchor available - use index-based fallback (1-indexed)
                adjusted_page = original_index + 1
                data["page_information"]["page_number_integer"] = adjusted_page
                data["page_information"]["page_number_type"] = page_num_type if page_num_type != "none" else "arabic"
            
            # Preserve page_types from the model
            data["page_information"]["page_types"] = content_page_types

            final_ordered_summaries.append(data)

        # Sort by original input order to preserve document sequence
        final_ordered_summaries.sort(
            key=lambda x: x.get("original_input_order_index", float('inf'))
        )
        
        # Log final page number distribution for debugging
        page_nums = [
            f.get("page_information", {}).get("page_number_integer") 
            for f in final_ordered_summaries
        ]
        logger.info(f"Final page numbers: {page_nums}")
        
        return final_ordered_summaries
