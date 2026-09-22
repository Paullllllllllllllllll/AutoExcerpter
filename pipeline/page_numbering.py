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

Sections scope page-number anchors only. Pages are always emitted in physical
scan order: tables, per-chapter bibliographies and part titles are interleaved
with the running text in real books, so relocating a section tears the document
apart.
"""

from __future__ import annotations

from typing import Any

from config.logger import setup_logger

logger = setup_logger(__name__)

# Constants for page number adjustment
MIN_SEQUENCE_LENGTH_FOR_ANCHOR = 2


def _coerce_page_number(value: Any) -> int | None:
    """Coerce a model-reported page number to ``int`` or ``None``.

    Structured output occasionally yields a string ("12") or a bool where an
    integer is expected; both would otherwise reach the arithmetic in
    ``adjust_and_sort_page_numbers`` and raise ``TypeError``. Anything that
    cannot be read as an integer is treated as "unnumbered".
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("Ignoring uninterpretable page number %r", value)
        return None


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

        model_page_num: int | None = None
        page_number_type = "none"
        page_types = ["content"]
        is_genuinely_unnumbered = True
        is_two_page_spread = False
        page_number_integer_end: int | None = None

        if isinstance(page_info_obj, dict) and page_info_obj:
            # New schema format with page_information object
            model_page_num = _coerce_page_number(
                page_info_obj.get("page_number_integer")
            )
            page_number_type = page_info_obj.get("page_number_type", "none")

            raw_page_types = page_info_obj.get("page_types")
            if raw_page_types is None:
                page_types = ["content"]
            elif isinstance(raw_page_types, str):
                page_types = [raw_page_types]
            elif isinstance(raw_page_types, list) and raw_page_types:
                # Drop non-string debris (e.g. dicts from a corrupt or reused
                # log): such an entry becomes the primary section type and then
                # raises TypeError (unhashable) as a dict key in the sort below.
                page_types = [pt for pt in raw_page_types if isinstance(pt, str)] or [
                    "content"
                ]
            else:
                page_types = ["content"]

            # Derive unnumbered status from page_number_type or null page_number_integer
            is_genuinely_unnumbered = (
                page_number_type == "none" or model_page_num is None
            )
            if is_genuinely_unnumbered:
                page_number_type = "none"

            is_two_page_spread = bool(page_info_obj.get("is_two_page_spread", False))
            # Coerce like the start number: structured output can yield a
            # string here too, and this value is part of the public return.
            page_number_integer_end = _coerce_page_number(
                page_info_obj.get("page_number_integer_end")
            )

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
                    # No number is printed on this page; rendering brackets it.
                    current["number_inferred"] = True
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

        Runs one pass per numbering type. In each pass an unnumbered page is
        filled only when both immediate neighbors are numbered with that same
        type and their numbers leave exactly this page's slot free; a gap
        spanning a change of numbering type is never bridged.

        Examples:
        - Page 5 -> [Unnumbered] -> Page 7  =>  infer Arabic page 6
        - Page viii -> [Unnumbered] -> Page x  =>  infer Roman page ix
        - [Preface] Page xii -> [Unnumbered] -> Page 2  =>  no inference
          (the neighbors use different numbering types)

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

        # Pure table/figure/source pages carry the printed page number of the
        # text they sit in -- a plate between pages 31 and 33 is page 32 -- and
        # they are scattered through the body rather than forming a block. Give
        # them the content anchor instead of a pseudo-section of their own,
        # whose anchor would renumber them into a sequence of its own invention.
        if page_types == ["figures_tables_sources"]:
            return "content"

        # For non-content pages, use priority order
        priority_order = ["preface", "abstract", "appendix", "figures_tables_sources"]
        for section in priority_order:
            if section in page_types:
                return section
        # Failed ("other") and blank pages are unnumbered placeholders; fold
        # them into the content flow instead of minting singleton
        # pseudo-sections, which would each get their own (absent) anchor and
        # fall back to a physical-position number. Real unknown sections
        # (e.g. a numbered "toc") keep their own pseudo-section so the content
        # anchor cannot renumber them.
        if page_types and all(pt in ("other", "blank") for pt in page_types):
            return "content"
        return page_types[0] if page_types else "content"

    def _find_section_anchor(
        self, section_pages: list[dict[str, Any]]
    ) -> tuple[int | None, int | None, str | None]:
        """
        Find the anchor point for a section based on longest consecutive sequence.

        Args:
            section_pages: List of parsed summaries for this section
                (sorted by doc order).

        Returns:
            Tuple of (anchor_page_number, anchor_virtual_pos, anchor_number_type)
            or (None, None, None) if no anchor.
        """
        if not section_pages:
            return None, None, None

        # Get only pages with valid page numbers (not unnumbered)
        numbered_pages = [
            p
            for p in section_pages
            if p["model_page_number_int"] is not None
            and not p["is_genuinely_unnumbered"]
        ]

        if not numbered_pages:
            return None, None, None

        # Find longest consecutive sequence
        longest_seq = self.find_longest_consecutive_sequence(numbered_pages)

        if longest_seq and len(longest_seq) >= MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
            anchor_item = longest_seq[0]
        else:
            # Fallback to first numbered page
            anchor_item = numbered_pages[0]
        return (
            anchor_item["model_page_number_int"],
            anchor_item["virtual_pos"],
            anchor_item["page_number_type"],
        )

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
        section_anchors: dict[str, tuple[int | None, int | None, str | None]] = {}
        for section, pages in section_groups.items():
            # Sort section pages by document order
            pages.sort(key=lambda x: x["original_input_order_index"])
            anchor_page, anchor_virtual_pos, anchor_type = self._find_section_anchor(
                pages
            )
            section_anchors[section] = (anchor_page, anchor_virtual_pos, anchor_type)
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
            anchor_page, anchor_virtual_pos, anchor_type = section_anchors.get(
                primary_section, (None, None, None)
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
                    # Invalid page number - mark as unnumbered. Worth a trace:
                    # an anchor sitting far downstream of a leading section
                    # blanks that whole section's numbering this way.
                    logger.debug(
                        "Page at index %s in section '%s' resolved to %s "
                        "(< 1) from anchor %s; marking unnumbered",
                        p["original_input_order_index"],
                        primary_section,
                        adjusted_page,
                        anchor_page,
                    )
                    resolved_page = None
                    page_info["page_number_integer"] = None
                    page_info["page_number_type"] = "none"
                else:
                    resolved_page = adjusted_page
                    page_info["page_number_integer"] = adjusted_page
                    # The number was derived from the section anchor, so it is
                    # expressed in the anchor's numbering system. Keeping the
                    # model's own type would render an arabic-derived number as
                    # a roman numeral ("Page xciii" between 92 and 94).
                    resolved_type = anchor_type or page_num_type
                    page_info["page_number_type"] = (
                        resolved_type if resolved_type != "none" else "arabic"
                    )
            else:
                # No anchor available - keep a model- or inference-set page
                # number (infer_unnumbered_page_numbers writes it into
                # model_page_number_int) so an inferred page is not discarded.
                # Without one the page stays unnumbered: its physical position
                # is kept in original_input_order_index, never passed off as a
                # printed number.
                model_page = p["model_page_number_int"]
                if isinstance(model_page, int) and model_page >= 1:
                    resolved_page = model_page
                    page_info["page_number_integer"] = resolved_page
                    page_info["page_number_type"] = (
                        page_num_type if page_num_type != "none" else "arabic"
                    )
                else:
                    resolved_page = None
                    page_info["page_number_integer"] = None
                    page_info["page_number_type"] = "none"

            # An inferred number was not printed on the page; the flag lets
            # rendering bracket it. It is meaningless once the page is
            # unnumbered. A flag already present comes from an earlier pass over
            # the same (resumed) results and still holds.
            inferred = p.get("number_inferred") or page_info.get("number_inferred")
            if resolved_page is not None and inferred:
                page_info["number_inferred"] = True
            else:
                page_info.pop("number_inferred", None)

            # Record the spread end page (right page) when numbered.
            if is_spread and resolved_page is not None:
                page_info["page_number_integer_end"] = resolved_page + 1
            else:
                page_info["page_number_integer_end"] = None

            # Preserve page_types from the model
            page_info["page_types"] = content_page_types

        # Emit strictly in physical scan order. Sections exist to scope page
        # number anchors, not to reorder the document: apparatus pages (tables,
        # per-chapter bibliographies, part titles) are interleaved with the
        # running text in real books, so grouping them displaces the prose.
        parsed_summaries.sort(key=lambda p: p["original_input_order_index"])
        final_ordered_summaries = [p["data"] for p in parsed_summaries]

        # Summarize the final numbering rather than dumping every page: a
        # 600-page monograph would otherwise emit one enormous INFO record.
        page_nums = [
            f.get("page_information", {}).get("page_number_integer")
            for f in final_ordered_summaries
        ]
        numbered = [n for n in page_nums if isinstance(n, int)]
        logger.info(
            "Final page numbers: %d page(s), %d numbered (%s-%s), %d unnumbered",
            len(page_nums),
            len(numbered),
            min(numbered) if numbered else "-",
            max(numbered) if numbered else "-",
            len(page_nums) - len(numbered),
        )
        logger.debug("Final page number sequence: %s", page_nums)

        return final_ordered_summaries
