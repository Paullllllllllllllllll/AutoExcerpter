"""Page numbering logic for document processing.

This module handles page number extraction, inference, and adjustment for
documents with mixed Roman numeral (preface) and Arabic (main text) numbering.

The key algorithm is per-section anchor-based adjustment:
1. Group pages by section type (content, preface, abstract, appendix, etc.)
2. For each section, find the longest consecutive sequence of model-detected page
   numbers
3. Use that sequence as the anchor and adjust all pages in that section accordingly;
   a section with several anchor runs (runs of at least two pages with one offset
   and one number type, as in an excerpt that skips pages) resolves each page
   against its own run, else the nearest preceding run
4. Conservatively infer page numbers for isolated unnumbered pages between numbered
   pages

Sections scope page-number anchors only. Pages are always emitted in physical
scan order: tables, per-chapter bibliographies and part titles are interleaved
with the running text in real books, so relocating a section tears the document
apart.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MIN_SEQUENCE_LENGTH_FOR_ANCHOR = 2

_Anchor = tuple[int | None, int | None, str | None]
_NO_ANCHOR: _Anchor = (None, None, None)


def _coerce_page_number(value: Any) -> int | None:
    """Coerce a model-reported page number to ``int`` or ``None``.

    Structured output occasionally yields a string ("12") or a bool where an
    integer is expected; both would otherwise raise ``TypeError`` in the anchor
    arithmetic. Anything that cannot be read as an integer is "unnumbered".
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


def _coerce_page_types(raw_page_types: Any) -> list[str]:
    """Return the model's page types as a non-empty list of strings."""
    if isinstance(raw_page_types, str):
        return [raw_page_types]
    if isinstance(raw_page_types, list) and raw_page_types:
        # Non-string debris (e.g. dicts from a corrupt log) would become the
        # primary section type and fail as a dict key.
        return [pt for pt in raw_page_types if isinstance(pt, str)] or ["content"]
    return ["content"]


def _log_final_numbering(final_ordered_summaries: list[dict[str, Any]]) -> None:
    """Log a summary of the final numbering instead of every page at INFO."""
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
            model_page_num = _coerce_page_number(
                page_info_obj.get("page_number_integer")
            )
            page_number_type = page_info_obj.get("page_number_type", "none")
            page_types = _coerce_page_types(page_info_obj.get("page_types"))

            is_genuinely_unnumbered = (
                page_number_type == "none" or model_page_num is None
            )
            if is_genuinely_unnumbered:
                page_number_type = "none"

            is_two_page_spread = bool(page_info_obj.get("is_two_page_spread", False))
            # Part of the public return, so coerced like the start number.
            page_number_integer_end = _coerce_page_number(
                page_info_obj.get("page_number_integer_end")
            )

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
            if not current_sequence:
                current_sequence = [item]
                continue

            # Page number and virtual position both advance by the previous
            # item's span (2 for a spread, 1 for a single page).
            prev = current_sequence[-1]
            prev_span = prev.get("span", 1)
            is_consecutive = (
                item["model_page_number_int"]
                == prev["model_page_number_int"] + prev_span
                and item["virtual_pos"] == prev["virtual_pos"] + prev_span
            )
            if is_consecutive:
                current_sequence.append(item)
            else:
                if len(current_sequence) > len(longest_sequence):
                    longest_sequence = current_sequence.copy()
                current_sequence = [item]

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

            if i == 0 or i + 1 >= len(sorted_summaries):
                continue

            prev_page = sorted_summaries[i - 1]
            next_page = sorted_summaries[i + 1]

            prev_span = prev_page.get("span", 1)
            current_span = current.get("span", 1)

            if (
                prev_page["page_number_type"] == page_type
                and prev_page["model_page_number_int"] is not None
                and not prev_page["is_genuinely_unnumbered"]
                and next_page["page_number_type"] == page_type
                and next_page["model_page_number_int"] is not None
                and not next_page["is_genuinely_unnumbered"]
                and prev_page["original_input_order_index"]
                == current["original_input_order_index"] - 1
                and next_page["original_input_order_index"]
                == current["original_input_order_index"] + 1
                # A single-image gap: the next number is the previous one plus
                # the spans of the previous and the current (possibly spread)
                # image.
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

        sorted_summaries = sorted(
            parsed_summaries, key=lambda x: x["original_input_order_index"]
        )

        # A spread claims both its start slot and the following slot.
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
        inferred_count += self._infer_from_following_page(
            sorted_summaries, "arabic", claimed_arabic_pages
        )
        inferred_count += self._infer_from_following_page(
            sorted_summaries, "roman", claimed_roman_pages
        )

        return inferred_count

    def _get_primary_section_type(self, page_types: list[str]) -> str:
        """
        Get the primary section type for a page.

        Any page containing "content" in its page_types is treated as a content
        page, so content pages share one anchor even with several type labels.

        Priority order: content > preface > abstract > appendix > figures_tables_sources

        Args:
            page_types: List of page type classifications.

        Returns:
            Primary section type string.
        """
        if "content" in page_types:
            return "content"

        # Pure table/figure/source pages carry the printed page number of the
        # text they sit in -- a plate between pages 31 and 33 is page 32 -- and
        # they are scattered through the body rather than forming a block. Give
        # them the content anchor instead of a pseudo-section of their own,
        # whose anchor would renumber them into a sequence of its own invention.
        if page_types == ["figures_tables_sources"]:
            return "content"

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

    def _find_section_anchor(self, section_pages: list[dict[str, Any]]) -> _Anchor:
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
            return _NO_ANCHOR

        numbered_pages = [
            p
            for p in section_pages
            if p["model_page_number_int"] is not None
            and not p["is_genuinely_unnumbered"]
        ]

        if not numbered_pages:
            return _NO_ANCHOR

        longest_seq = self.find_longest_consecutive_sequence(numbered_pages)

        if longest_seq and len(longest_seq) >= MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
            anchor_item = longest_seq[0]
        else:
            anchor_item = numbered_pages[0]
        return (
            anchor_item["model_page_number_int"],
            anchor_item["virtual_pos"],
            anchor_item["page_number_type"],
        )

    @staticmethod
    def _anchor_runs(
        numbered_pages: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Split numbered pages (doc order) into anchor runs.

        A run is a stretch of pages with one offset (model number minus
        virtual position) and one number type; runs shorter than
        MIN_SEQUENCE_LENGTH_FOR_ANCHOR are dropped, and a run that returns to
        an earlier run's offset and type merges with it, dropping the runs
        between.
        """
        runs: list[list[dict[str, Any]]] = []
        for p in numbered_pages:
            key = (p["model_page_number_int"] - p["virtual_pos"], p["page_number_type"])
            last = runs[-1] if runs else None
            if last is not None and key == (
                last[0]["model_page_number_int"] - last[0]["virtual_pos"],
                last[0]["page_number_type"],
            ):
                last.append(p)
            else:
                runs.append([p])
        # An excerpt never returns to an earlier offset: a run that does
        # merges with that earlier run, and the runs between were misreads.
        anchor_runs: list[list[dict[str, Any]]] = []
        for run in runs:
            if len(run) < MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
                continue
            key = (
                run[0]["model_page_number_int"] - run[0]["virtual_pos"],
                run[0]["page_number_type"],
            )
            earlier = next(
                (
                    i
                    for i, kept in enumerate(anchor_runs)
                    if key
                    == (
                        kept[0]["model_page_number_int"] - kept[0]["virtual_pos"],
                        kept[0]["page_number_type"],
                    )
                ),
                None,
            )
            if earlier is None:
                anchor_runs.append(run)
            else:
                del anchor_runs[earlier + 1 :]
                anchor_runs[earlier].extend(run)
        return anchor_runs

    def _page_anchors(self, section_pages: list[dict[str, Any]]) -> dict[int, _Anchor]:
        """Per-page anchors of a section with several anchor runs, keyed by id.

        Empty when the section has at most one anchor run; its single section
        anchor then applies. Otherwise a page takes the anchor of its own run,
        else of the nearest preceding run (pages before the first take the
        first), so a real discontinuity in an excerpt is kept.
        """
        numbered = [
            p
            for p in section_pages
            if p["model_page_number_int"] is not None
            and not p["is_genuinely_unnumbered"]
        ]
        runs = self._anchor_runs(numbered)
        if len(runs) <= 1:
            return {}
        anchors = [
            (
                run[0]["model_page_number_int"],
                run[0]["virtual_pos"],
                run[0]["page_number_type"],
            )
            for run in runs
        ]
        member = {id(p): anchors[i] for i, run in enumerate(runs) for p in run}
        page_anchors: dict[int, _Anchor] = {}
        for p in section_pages:
            anchor = member.get(id(p))
            if anchor is None:
                anchor = anchors[0]
                for candidate in anchors:
                    if candidate[1] is not None and candidate[1] < p["virtual_pos"]:
                        anchor = candidate
            page_anchors[id(p)] = anchor
        return page_anchors

    def _parse_summaries(
        self, summary_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Wrap each result with its parsed page fields, in document order."""
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
                    "span": 2 if is_spread else 1,
                }
            )

        parsed_summaries.sort(key=lambda x: x["original_input_order_index"])

        # The virtual position is the summed span of all earlier images, so
        # anchor offsets stay aligned across spreads.
        cumulative = 0
        for p in parsed_summaries:
            p["virtual_pos"] = cumulative
            cumulative += p["span"]
        return parsed_summaries

    def _find_section_anchors(
        self, parsed_summaries: list[dict[str, Any]]
    ) -> dict[str, _Anchor]:
        """Group pages by primary section and find each section's anchor."""
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

        section_anchors: dict[str, _Anchor] = {}
        for section, pages in section_groups.items():
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
        return section_anchors

    def _resolve_page_number(
        self, p: dict[str, Any], anchor: _Anchor, page_info: dict[str, Any]
    ) -> int | None:
        """Write the resolved number and type into ``page_info``; return the number."""
        anchor_page, anchor_virtual_pos, anchor_type = anchor
        page_num_type = p["page_number_type"]

        if p["is_genuinely_unnumbered"]:
            pass
        elif anchor_page is not None and anchor_virtual_pos is not None:
            adjusted_page = self.calculate_adjusted_page_number(
                p["virtual_pos"], anchor_page, anchor_virtual_pos
            )
            if adjusted_page >= 1:
                page_info["page_number_integer"] = adjusted_page
                # The number derives from the section anchor, so it takes the
                # anchor's numbering system; the model's own type would render
                # an arabic-derived number as a roman numeral.
                resolved_type = anchor_type or page_num_type
                page_info["page_number_type"] = (
                    resolved_type if resolved_type != "none" else "arabic"
                )
                return adjusted_page
            # An anchor far downstream of a leading section blanks that whole
            # section's numbering this way.
            logger.debug(
                "Page at index %s in section '%s' resolved to %s "
                "(< 1) from anchor %s; marking unnumbered",
                p["original_input_order_index"],
                p["primary_section"],
                adjusted_page,
                anchor_page,
            )
        else:
            # Without an anchor, keep a model- or inference-set number; the
            # physical position is never passed off as a printed number.
            model_page = p["model_page_number_int"]
            if isinstance(model_page, int) and model_page >= 1:
                page_info["page_number_integer"] = model_page
                page_info["page_number_type"] = (
                    page_num_type if page_num_type != "none" else "arabic"
                )
                return model_page

        page_info["page_number_integer"] = None
        page_info["page_number_type"] = "none"
        return None

    def _apply_page_numbering(self, p: dict[str, Any], anchor: _Anchor) -> None:
        """Write the final page fields into the wrapped result's page_information."""
        data = p["data"]
        if "page_information" not in data or not isinstance(
            data["page_information"], dict
        ):
            data["page_information"] = {
                "page_number_integer": None,
                "page_number_type": "none",
                "page_types": p["page_types"],
            }

        page_info = data["page_information"]
        # Spread-ness is a physical property of the scan; it is kept even on
        # unnumbered pages so rendering can label them.
        page_info["is_two_page_spread"] = p["is_two_page_spread"]

        resolved_page = self._resolve_page_number(p, anchor, page_info)

        # An inferred number was not printed on the page; the flag lets
        # rendering bracket it. A flag already present comes from an earlier
        # pass over the same (resumed) results and still holds.
        inferred = p.get("number_inferred") or page_info.get("number_inferred")
        if resolved_page is not None and inferred:
            page_info["number_inferred"] = True
        else:
            page_info.pop("number_inferred", None)

        if p["is_two_page_spread"] and resolved_page is not None:
            page_info["page_number_integer_end"] = resolved_page + 1
        else:
            page_info["page_number_integer_end"] = None

        page_info["page_types"] = p["page_types"]

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
            Summaries in physical scan order with adjusted page numbers.
        """
        if not summary_results:
            return []

        logger.info("Adjusting page numbers using per-section anchor logic...")

        parsed_summaries = self._parse_summaries(summary_results)
        section_anchors = self._find_section_anchors(parsed_summaries)
        page_anchors: dict[int, _Anchor] = {}
        for section in section_anchors:
            page_anchors.update(
                self._page_anchors(
                    [p for p in parsed_summaries if p["primary_section"] == section]
                )
            )

        # Inference runs after the anchors are fixed, so inferred numbers never
        # move an anchor.
        inferred_count = self.infer_unnumbered_page_numbers(parsed_summaries)
        if inferred_count > 0:
            logger.info(
                f"Inferred page numbers for {inferred_count} "
                "isolated unnumbered page(s)"
            )

        for p in parsed_summaries:
            anchor = page_anchors.get(id(p)) or section_anchors.get(
                p["primary_section"], _NO_ANCHOR
            )
            self._apply_page_numbering(p, anchor)

        # Sections scope the anchors but never reorder the document: apparatus
        # pages are interleaved with the running text in real books.
        parsed_summaries.sort(key=lambda p: p["original_input_order_index"])
        final_ordered_summaries = [p["data"] for p in parsed_summaries]
        _log_final_numbering(final_ordered_summaries)
        return final_ordered_summaries
