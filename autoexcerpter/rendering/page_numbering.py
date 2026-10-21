"""Page numbering logic for document processing.

This module handles page number extraction, inference, and adjustment for
documents with mixed Roman numeral (preface) and Arabic (main text) numbering.

A model-flagged two-page spread is first checked against the printed numbers
of its physical neighbors and becomes a single page when they do not support
it; such a page takes the model's second number instead of its first when only
the second continues the previous page. The virtual positions, which count a
spread as two pages, follow from the checked flags.

The key algorithm is per-section anchor-based adjustment:
1. Group pages by section type (content, preface, abstract, appendix, etc.)
2. For each section, find the longest consecutive sequence of model-detected page
   numbers of one number type
3. Use that sequence as the anchor and adjust all pages in that section accordingly;
   a section with several anchor runs (runs of at least two pages with one offset
   and one number type, as in an excerpt or a PDF that skips or repeats pages)
   resolves each page against its own run, else the nearest preceding run. A
   run that returns to an earlier run's offset drops the short runs between
   as misreads. A short run whose offset occurs nowhere else in the block,
   between two pages of another numbering (section numbers between folios),
   is a misread too: it anchors nothing and its pages are numbered from the
   surrounding numbering
4. Conservatively infer page numbers for isolated unnumbered pages between numbered
   pages

Steps 1-4 run on each block of consecutive scans on its own: a document
stored with gaps (``--pages``) carries no correction across a gap.

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
# A run of this many pages is real numbering even inside another numbering
# (a PDF that skips or repeats pages), not a stretch of misreads.
MIN_ENCLOSED_RUN_TO_KEEP = 4

_Anchor = tuple[int | None, int | None, str | None]
_NO_ANCHOR: _Anchor = (None, None, None)
_Key = tuple[int, str]


def _key(page: dict[str, Any]) -> _Key:
    """Return a numbered page's offset (number minus virtual position) and type."""
    offset: int = page["model_page_number_int"] - page["virtual_pos"]
    return offset, page["page_number_type"]


def _is_numbered(page: dict[str, Any]) -> bool:
    return (
        page["model_page_number_int"] is not None
        and not page["is_genuinely_unnumbered"]
    )


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


def _source_blocks(parsed: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split pages in source order into runs of consecutive source indices.

    A ``--pages`` run stores some scans only; how many printed pages an
    omitted scan holds is unknown, so no anchor reaches across the gap.
    """
    blocks: list[list[dict[str, Any]]] = []
    for p in parsed:
        index = p["original_input_order_index"]
        if blocks and index == blocks[-1][-1]["original_input_order_index"] + 1:
            blocks[-1].append(p)
        else:
            blocks.append([p])
    return blocks


def _adjacent_numbered(
    pages: list[dict[str, Any]], i: int, step: int
) -> dict[str, Any] | None:
    """Return the physically adjacent scan of ``pages[i]`` when it is numbered."""
    j = i + step
    if not 0 <= j < len(pages):
        return None
    other = pages[j]
    if (
        other["original_input_order_index"]
        != pages[i]["original_input_order_index"] + step
        or other["is_genuinely_unnumbered"]
    ):
        return None
    return other


def _span_from_neighbors(pages: list[dict[str, Any]], i: int) -> int | None:
    """Return the span of scan ``i`` that its immediate neighbors establish.

    The next scan's number is compared with this scan's start and with the
    previous scan's number plus that scan's span: an advance of one makes the
    scan a single page, an advance of two a spread. None when the neighbors
    are missing, of another number type or discontinuous.
    """
    page = pages[i]
    nxt = _adjacent_numbered(pages, i, 1)
    if nxt is None:
        return None
    prev = _adjacent_numbered(pages, i, -1)
    steps: set[int] = set()
    if (
        not page["is_genuinely_unnumbered"]
        and page["page_number_type"] == nxt["page_number_type"]
    ):
        steps.add(nxt["model_page_number_int"] - page["model_page_number_int"])
    if prev is not None and prev["page_number_type"] == nxt["page_number_type"]:
        steps.add(
            nxt["model_page_number_int"] - prev["model_page_number_int"] - prev["span"]
        )
    if 1 in steps:
        return 1
    if 2 in steps:
        return 2
    return None


def _end_and_next_agree(pages: list[dict[str, Any]], i: int) -> bool:
    """Whether a spread's own end and its next numbered neighbor support it.

    The model's end must be the start plus one, and the next numbered scan
    of the block, if there is one, must not hold start + 1. A next number at
    or before the start (a repeated or misread scan) says nothing against it.
    """
    page = pages[i]
    start: int = page["model_page_number_int"]
    if page["is_genuinely_unnumbered"] or page["raw_page_number_end"] != start + 1:
        return False
    index = page["original_input_order_index"]
    for distance, later in enumerate(pages[i + 1 :], 1):
        if later["original_input_order_index"] != index + distance:
            break
        if later["is_genuinely_unnumbered"]:
            continue
        if later["page_number_type"] != page["page_number_type"]:
            break
        number: int = later["model_page_number_int"]
        return number != start + 1
    return True


def _take_continuing_end(pages: list[dict[str, Any]], i: int) -> None:
    """Give a rejected spread the model's end number when only it continues.

    A page that holds a section or signature number beside its folio comes
    back as a spread from the one number to the other. When the end, and not
    the start, follows the previous scan's number, the end is the folio.
    """
    page = pages[i]
    end = page["raw_page_number_end"]
    prev = _adjacent_numbered(pages, i, -1)
    if page["is_genuinely_unnumbered"] or end is None or prev is None:
        return
    follows = prev["model_page_number_int"] + prev["span"]
    if end == follows and page["model_page_number_int"] != follows:
        page["model_page_number_int"] = end
        page["page_number_type"] = prev["page_number_type"]


def _validate_spreads(pages: list[dict[str, Any]]) -> None:
    """Keep a model-flagged spread only where the page numbers support it.

    *pages* are in scan order. The immediate neighbors decide when they
    establish the scan's span (:func:`_span_from_neighbors`); otherwise the
    flag stands only when the model's end and the next numbered neighbor
    agree (:func:`_end_and_next_agree`). A rejected spread, such as a page
    with a running header number beside its folio, becomes a single page
    (numbered by :func:`_take_continuing_end`).
    """
    for i, page in enumerate(pages):
        if not page["is_two_page_spread"]:
            continue
        span = _span_from_neighbors(pages, i)
        keep = _end_and_next_agree(pages, i) if span is None else span == 2
        if not keep:
            logger.debug(
                "Treating the spread flag of the page at index %s as a single page",
                page["original_input_order_index"],
            )
            page["is_two_page_spread"] = False
            page["page_number_integer_end"] = None
            page["span"] = 1
            _take_continuing_end(pages, i)


def _mark_enclosed_runs(block: list[dict[str, Any]]) -> None:
    """Mark short runs that lie inside another numbering of *block* as misreads.

    A run (consecutive numbered pages of one offset and type) shorter than
    MIN_ENCLOSED_RUN_TO_KEEP is enclosed when its offset and type occur
    nowhere else in the block, while the numbered pages just before and just
    after it share an offset and type that more pages hold: section numbers
    or a second pagination between folios. Each of its pages gets
    ``enclosed_by``, the anchor of that numbering, and no run of misreads
    anchors a section. A real skip or repeat never returns to the offset
    before it, so its runs stay.
    """
    numbered = [p for p in block if _is_numbered(p)]
    support: dict[_Key, int] = {}
    for p in numbered:
        support[_key(p)] = support.get(_key(p), 0) + 1
    runs: list[list[dict[str, Any]]] = []
    for p in numbered:
        if runs and _key(runs[-1][0]) == _key(p):
            runs[-1].append(p)
        else:
            runs.append([p])
    position = 0
    for run in runs:
        start, position = position, position + len(run)
        if not MIN_SEQUENCE_LENGTH_FOR_ANCHOR <= len(run) < MIN_ENCLOSED_RUN_TO_KEEP:
            continue
        if start == 0 or position == len(numbered):
            continue
        before, after = numbered[start - 1], numbered[position]
        key = _key(before)
        own = support[_key(run[0])]
        if key != _key(after) or own > len(run) or support[key] <= own:
            continue
        anchor: _Anchor = (
            before["model_page_number_int"],
            before["virtual_pos"],
            before["page_number_type"],
        )
        for p in run:
            p["enclosed_by"] = anchor


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

        A sequence holds one number type, so the anchor it yields carries the
        type of the numbering it spans.

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
            # item's span (2 for a spread, 1 for a single page), within one
            # number type: a roman i before arabic 2 starts no arabic run.
            prev = current_sequence[-1]
            prev_span = prev.get("span", 1)
            is_consecutive = (
                item["model_page_number_int"]
                == prev["model_page_number_int"] + prev_span
                and item["virtual_pos"] == prev["virtual_pos"] + prev_span
                and item.get("page_number_type") == prev.get("page_number_type")
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

        A section whose longest run of consecutive numbers is shorter than
        MIN_SEQUENCE_LENGTH_FOR_ANCHOR (a foliated volume numbered on rectos
        only, or a single read number) has no anchor, so its pages keep the
        model's numbers instead of being renumbered by position.

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
            p for p in section_pages if _is_numbered(p) and "enclosed_by" not in p
        ]

        if not numbered_pages:
            return _NO_ANCHOR

        longest_seq = self.find_longest_consecutive_sequence(numbered_pages)

        if len(longest_seq) < MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
            return _NO_ANCHOR
        anchor_item = longest_seq[0]
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
        MIN_SEQUENCE_LENGTH_FOR_ANCHOR are dropped. A run that returns to the
        latest earlier run's offset and type merges with it, dropping the
        runs between as misreads, unless one of them holds at least
        MIN_ENCLOSED_RUN_TO_KEEP pages: a PDF that skips or repeats pages can
        return to an earlier offset after a long correct run, which keeps its
        own anchor. Misread runs that the neighboring pages enclose are left
        out by the caller (``enclosed_by``).
        """
        runs: list[list[dict[str, Any]]] = []
        for p in numbered_pages:
            if runs and _key(runs[-1][0]) == _key(p):
                runs[-1].append(p)
            else:
                runs.append([p])
        anchor_runs: list[list[dict[str, Any]]] = []
        # The longest unbroken stretch of each anchor run; merged misreads
        # do not add up to a long run.
        longest: list[int] = []
        for run in runs:
            if len(run) < MIN_SEQUENCE_LENGTH_FOR_ANCHOR:
                continue
            earlier = next(
                (
                    i
                    for i in range(len(anchor_runs) - 1, -1, -1)
                    if _key(anchor_runs[i][0]) == _key(run[0])
                ),
                None,
            )
            if earlier is None or any(
                length >= MIN_ENCLOSED_RUN_TO_KEEP for length in longest[earlier + 1 :]
            ):
                anchor_runs.append(run)
                longest.append(len(run))
            else:
                del anchor_runs[earlier + 1 :], longest[earlier + 1 :]
                anchor_runs[earlier].extend(run)
                longest[earlier] = max(longest[earlier], len(run))
        return anchor_runs

    def _page_anchors(self, section_pages: list[dict[str, Any]]) -> dict[int, _Anchor]:
        """Per-page anchors of a section with several anchor runs, keyed by id.

        Empty when the section has at most one anchor run; its single section
        anchor then applies. Otherwise a page takes the anchor of its own run,
        else of the nearest preceding run (pages before the first take the
        first), so a real discontinuity in an excerpt is kept.
        """
        numbered = [
            p for p in section_pages if _is_numbered(p) and "enclosed_by" not in p
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
            # The model's own end, before the normalization above, is
            # evidence when the spread flag is validated.
            info = r.get("page_information")
            raw_end = _coerce_page_number(
                info.get("page_number_integer_end") if isinstance(info, dict) else None
            )
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
                    "raw_page_number_end": raw_end,
                    "span": 2 if is_spread else 1,
                }
            )

        parsed_summaries.sort(key=lambda x: x["original_input_order_index"])
        _validate_spreads(parsed_summaries)

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

    def _number_block(self, parsed_summaries: list[dict[str, Any]]) -> None:
        """Anchor, infer and write the page numbers of one source block."""
        _mark_enclosed_runs(parsed_summaries)
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
            anchor = (
                p.get("enclosed_by")
                or page_anchors.get(id(p))
                or section_anchors.get(p["primary_section"], _NO_ANCHOR)
            )
            self._apply_page_numbering(p, anchor)

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
        for block in _source_blocks(parsed_summaries):
            self._number_block(block)

        # Sections scope the anchors but never reorder the document: apparatus
        # pages are interleaved with the running text in real books.
        parsed_summaries.sort(key=lambda p: p["original_input_order_index"])
        final_ordered_summaries = [p["data"] for p in parsed_summaries]
        _log_final_numbering(final_ordered_summaries)
        return final_ordered_summaries
