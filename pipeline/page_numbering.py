"""Page numbering logic for document processing.

This module handles page number extraction, inference, and adjustment for
documents with mixed Roman numeral (preface) and Arabic (main text) numbering.

The key algorithm is per-section anchor-based adjustment:
1. Group pages by section type (content, preface, abstract, appendix, etc.)

2. For each section, find the longest consecutive sequence of model-detected page
   numbers
3. Use that sequence as the anchor and adjust all pages in that section accordingly
4. Conservatively infer page numbers for isolated unnumbered pages between numbered
   pages
"""

from __future__ import annotations

from statistics import median
from typing import Any

from config.logger import setup_logger

logger = setup_logger(__name__)

# Constants for page number adjustment
MIN_SEQUENCE_LENGTH_FOR_ANCHOR = 2

# Section types that get full bullet-point summaries (and need per-section anchoring)
SUMMARY_SECTION_TYPES = {
    "content",
    "preface",
    "abstract",
    "appendix",
    "figures_tables_sources",
}


class PageNumberProcessor:
    """Process and adjust page numbers for document summaries.

    Handles extraction of page information from API results, inference of
    page numbers for unnumbered pages, and adjustment based on anchor points.
    """

    def parse_page_information(
        self, summary_result: dict[str, Any]
    ) -> tuple[int | None, str, list[str], bool, bool, int | None]:
        """
        Extract page information from a summary result.

        Args:
            summary_result: Summary result dictionary.

        Returns:
            Tuple of (model_page_number, page_number_type, page_types,
            is_genuinely_unnumbered, is_two_page_spread, page_number_integer_end).
            page_number_type is one of: 'roman', 'arabic', 'none'.
            page_types is a list of page type classifications.

        Two-page spreads are normalized: when the page is a spread and the start
        number is known, the end number is forced to ``start + 1`` (a debug
        message is logged when the model reported a different end). When the page
        is not a spread, the end number is ``None``.
        """
        page_info_obj = summary_result.get("page_information")
        if not isinstance(page_info_obj, dict) or not page_info_obj:
            page_info_obj = {}

        model_page_num = None
        page_number_type = "none"
        page_types = ["content"]
        is_genuinely_unnumbered = True
        is_two_page_spread = False
        page_number_integer_end: int | None = None

        if isinstance(page_info_obj, dict) and page_info_obj:
            # New schema format with page_information object
            model_page_num = page_info_obj.get("page_number_integer")
            page_number_type = page_info_obj.get("page_number_type", "none")

            raw_page_types = page_info_obj.get("page_types")
            if raw_page_types is None:
                page_types = ["content"]
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

            is_two_page_spread = bool(page_info_obj.get("is_two_page_spread", False))
            page_number_integer_end = page_info_obj.get("page_number_integer_end")

        # Normalize the end page number against the spread flag.
        if is_two_page_spread and isinstance(model_page_num, int):
            expected_end = model_page_num + 1
            if (
                isinstance(page_number_integer_end, int)
                and page_number_integer_end != expected_end
            ):
                logger.debug(
                    "Spread end page %s != start+1 (%s); normalizing to %s",
                    page_number_integer_end,
                    expected_end,
                    expected_end,
                )
            page_number_integer_end = expected_end
        elif not is_two_page_spread:
            page_number_integer_end = None

        return (
            model_page_num,
            page_number_type,
            page_types,
            is_genuinely_unnumbered,
            is_two_page_spread,
            page_number_integer_end,
        )

    def find_longest_consecutive_sequence(
        self, summaries_with_pages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                continue

            prev = current_sequence[-1]
            prev_span = prev.get("span", 1)
            is_consecutive = (
                item["model_page_number_int"]
                == prev["model_page_number_int"] + prev_span
                and item["virtual_pos"] == prev["virtual_pos"] + prev_span
            )
            if is_consecutive:
                # Page number and virtual document position both advance by the
                # previous item's span (2 for a spread, 1 for a single page).
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
        self, virtual_pos: int, anchor_model_page: int, anchor_virtual_pos: int
    ) -> int:
        """
        Calculate adjusted page number based on anchor point.

        Uses document-wide virtual page positions (which account for two-page
        spreads occupying two page slots) rather than raw input indices.

        Args:
            virtual_pos: Virtual page position of the page being adjusted.
            anchor_model_page: Model-detected page number at anchor.
            anchor_virtual_pos: Virtual page position of the anchor.

        Returns:
            Adjusted page number.
        """
        offset = virtual_pos - anchor_virtual_pos
        adjusted_page = anchor_model_page + offset
        return adjusted_page

    def _infer_from_following_page(
        self,
        sorted_summaries: list[dict[str, Any]],
        page_type: str,
        claimed_pages: set[int],
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

            prev_span = prev_page.get("span", 1)
            current_span = current.get("span", 1)

            # Both surrounding pages must be numbered with the same type
            if (
                prev_page["page_number_type"] == page_type
                and prev_page["model_page_number_int"] is not None
                and not prev_page["is_genuinely_unnumbered"]
                and next_page["page_number_type"] == page_type
                and next_page["model_page_number_int"] is not None
                and not next_page["is_genuinely_unnumbered"]
                # Document positions must be consecutive
                and prev_page["original_input_order_index"]
                == current["original_input_order_index"] - 1
                and next_page["original_input_order_index"]
                == current["original_input_order_index"] + 1
                # Page numbers must indicate a single-image gap. The next page's
                # number equals the previous page's number plus the spans of the
                # previous image and the current (possibly spread) image.
                and next_page["model_page_number_int"]
                == prev_page["model_page_number_int"] + prev_span + current_span
            ):
                inferred_page = prev_page["model_page_number_int"] + prev_span

                if inferred_page >= 1 and inferred_page not in claimed_pages:
                    current["model_page_number_int"] = inferred_page
                    current["page_number_type"] = page_type
                    current["is_genuinely_unnumbered"] = False
                    claimed_pages.add(inferred_page)
                    if current_span == 2:
                        # A spread also occupies the following page slot.
                        claimed_pages.add(inferred_page + 1)
                    inferred_count += 1
                    logger.info(
                        f"Inferred {page_type} page {inferred_page} for unnumbered "
                        f"page at document position "
                        f"{current['original_input_order_index']} "
                        f"(gap between pages {prev_page['model_page_number_int']}"
                        f" and {next_page['model_page_number_int']})"
                    )

        return inferred_count

    def infer_unnumbered_page_numbers(
        self, parsed_summaries: list[dict[str, Any]]
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

        # Build sets of already-claimed page numbers to avoid conflicts. A spread
        # claims both the start slot and the following (start + 1) slot.
        def _claimed_for_type(page_type: str) -> set[int]:
            claimed: set[int] = set()
            for s in sorted_summaries:
                if (
                    s["page_number_type"] == page_type
                    and s["model_page_number_int"] is not None
                    and not s["is_genuinely_unnumbered"]
                ):
                    start = s["model_page_number_int"]
                    claimed.add(start)
                    if s.get("span", 1) == 2:
                        claimed.add(start + 1)
            return claimed

        claimed_arabic_pages = _claimed_for_type("arabic")
        claimed_roman_pages = _claimed_for_type("roman")

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

    def _get_primary_section_type(self, page_types: list[str]) -> str:
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
        self, section_pages: list[dict[str, Any]]
    ) -> tuple[int | None, int | None]:
        """
        Find the anchor point for a section based on longest consecutive sequence.

        Args:
            section_pages: List of parsed summaries for this section
                (sorted by doc order).

        Returns:
            Tuple of (anchor_page_number, anchor_virtual_pos)
            or (None, None) if no anchor.
        """
        if not section_pages:
            return None, None

        # Get only pages with valid page numbers (not unnumbered)
        numbered_pages = [
            p
            for p in section_pages
            if p["model_page_number_int"] is not None
            and not p["is_genuinely_unnumbered"]
        ]

        if not numbered_pages:
            return None, None

        # Find longest consecutive sequence
        longest_seq = self.find_longest_consecutive_sequence(numbered_pages)

        if longest_seq and len(longest_seq) >= MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
            anchor_item = longest_seq[0]
            return (
                anchor_item["model_page_number_int"],
                anchor_item["virtual_pos"],
            )
        elif numbered_pages:
            # Fallback to first numbered page
            anchor_item = numbered_pages[0]
            return (
                anchor_item["model_page_number_int"],
                anchor_item["virtual_pos"],
            )

        return None, None

    def adjust_and_sort_page_numbers(
        self, summary_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Adjust page numbers based on per-section anchor logic.

        For each section type (content, preface, abstract, appendix,
        figures_tables_sources), finds the longest consecutive sequence of
        model-detected page numbers and uses that as the anchor for adjusting
        all pages in that section.

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
            (
                model_page_num,
                page_num_type,
                page_types,
                is_unnumbered,
                is_spread,
                page_num_end,
            ) = self.parse_page_information(r)
            primary_section = self._get_primary_section_type(page_types)
            parsed_summaries.append(
                {
                    "original_input_order_index": r["original_input_order_index"],
                    "model_page_number_int": model_page_num,
                    "page_number_type": page_num_type,
                    "page_types": page_types,
                    "primary_section": primary_section,
                    "data": r,
                    "is_genuinely_unnumbered": is_unnumbered,
                    "is_two_page_spread": is_spread,
                    "page_number_integer_end": page_num_end,
                    # A two-page spread occupies two page slots.
                    "span": 2 if is_spread else 1,
                }
            )

        # Sort by document order for sequential processing
        parsed_summaries.sort(key=lambda x: x["original_input_order_index"])

        # Compute a document-wide virtual page position per summary: the sum of
        # the spans of all earlier images in document order. Spreads advance the
        # position by two so anchor offsets stay aligned across them.
        cumulative = 0
        for p in parsed_summaries:
            p["virtual_pos"] = cumulative
            cumulative += p["span"]

        # Group pages by section type
        section_groups: dict[str, list[dict[str, Any]]] = {}
        for p in parsed_summaries:
            section = p["primary_section"]
            if section not in section_groups:
                section_groups[section] = []
            section_groups[section].append(p)

        logger.info(
            "Section distribution: "
            + ", ".join(f"{k}: {len(v)} pages" for k, v in section_groups.items())
        )

        # Find anchor for each section
        section_anchors: dict[str, tuple[int | None, int | None]] = {}
        for section, pages in section_groups.items():
            # Sort section pages by document order
            pages.sort(key=lambda x: x["original_input_order_index"])
            anchor_page, anchor_virtual_pos = self._find_section_anchor(pages)
            section_anchors[section] = (anchor_page, anchor_virtual_pos)
            if anchor_page is not None:
                logger.info(
                    f"Section '{section}' anchor: page {anchor_page} "
                    f"at virtual position {anchor_virtual_pos}"
                )
            else:
                logger.info(f"Section '{section}': no valid anchor found")

        # Infer page numbers for isolated unnumbered pages between numbered pages
        # Do this AFTER we have section anchors, but BEFORE final adjustment
        inferred_count = self.infer_unnumbered_page_numbers(parsed_summaries)
        if inferred_count > 0:
            logger.info(
                f"Inferred page numbers for {inferred_count} "
                "isolated unnumbered page(s)"
            )

        # Apply per-section page number adjustment
        for p in parsed_summaries:
            page_num_type = p["page_number_type"]
            content_page_types = p["page_types"]
            primary_section = p["primary_section"]
            is_spread = p["is_two_page_spread"]
            data = p["data"]

            # Ensure page_information exists at top level (flat structure)
            if "page_information" not in data or not isinstance(
                data["page_information"], dict
            ):
                data["page_information"] = {
                    "page_number_integer": None,
                    "page_number_type": "none",
                    "page_types": content_page_types,
                }

            page_info = data["page_information"]
            # Spread-ness is a physical property of the scan; preserve it even
            # when the page ends up unnumbered so rendering can label it.
            page_info["is_two_page_spread"] = is_spread

            # Get section anchor
            anchor_page, anchor_virtual_pos = section_anchors.get(
                primary_section, (None, None)
            )

            resolved_page: int | None
            if p["is_genuinely_unnumbered"]:
                # Genuinely unnumbered page - keep as unnumbered
                resolved_page = None
                page_info["page_number_integer"] = None
                page_info["page_number_type"] = "none"
            elif anchor_page is not None and anchor_virtual_pos is not None:
                # Calculate adjusted page number using section anchor
                adjusted_page = self.calculate_adjusted_page_number(
                    p["virtual_pos"], anchor_page, anchor_virtual_pos
                )
                if adjusted_page < 1:
                    # Invalid page number - mark as unnumbered
                    resolved_page = None
                    page_info["page_number_integer"] = None
                    page_info["page_number_type"] = "none"
                else:
                    resolved_page = adjusted_page
                    page_info["page_number_integer"] = adjusted_page
                    # Preserve original number type (roman/arabic) from model
                    page_info["page_number_type"] = (
                        page_num_type if page_num_type != "none" else "arabic"
                    )
            else:
                # No anchor available - use virtual-position fallback (1-indexed)
                resolved_page = p["virtual_pos"] + 1
                page_info["page_number_integer"] = resolved_page
                page_info["page_number_type"] = (
                    page_num_type if page_num_type != "none" else "arabic"
                )

            # Record the spread end page (right page) when numbered.
            if is_spread and resolved_page is not None:
                page_info["page_number_integer_end"] = resolved_page + 1
            else:
                page_info["page_number_integer_end"] = None

            # Preserve page_types from the model
            page_info["page_types"] = content_page_types

        # Order sections by the MEDIAN original input index of their pages so a
        # single misclassified straggler cannot reorder a whole section; break
        # ties by the section's minimum index. Within a section, keep physical
        # scan order (original_input_order_index).
        section_rank: dict[str, tuple[float, int]] = {}
        for section, pages in section_groups.items():
            indices = [p["original_input_order_index"] for p in pages]
            section_rank[section] = (median(indices), min(indices))

        parsed_summaries.sort(
            key=lambda p: (
                section_rank[p["primary_section"]],
                p["original_input_order_index"],
            )
        )
        final_ordered_summaries = [p["data"] for p in parsed_summaries]

        # Log final page number distribution for debugging
        page_nums = [
            f.get("page_information", {}).get("page_number_integer")
            for f in final_ordered_summaries
        ]
        logger.info(f"Final page numbers: {page_nums}")

        return final_ordered_summaries
