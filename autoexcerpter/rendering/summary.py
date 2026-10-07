"""Summary data preparation and page-rendering helpers for AutoExcerpter.

Shared by the DOCX and Markdown writers. Contains XML sanitization, page
information extraction, content filtering, and the shared
:func:`prepare_summary_data` pipeline.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from autoexcerpter.constants import ERROR_MARKERS
from autoexcerpter.rendering.citations import (
    DEFAULT_MAX_API_REQUESTS,
    CitationManager,
    OpenAlexClient,
)
from autoexcerpter.rendering.locators import (
    PRINTED_ARABIC,
    PRINTED_ROMAN,
    int_to_roman,
)

logger = logging.getLogger(__name__)


def sanitize_for_xml(text: str | None) -> str:
    """Return XML-safe text for DOCX output by removing control characters."""
    if not text:
        return ""
    # Strip all XML-1.0-illegal codepoints: C0 controls/DEL plus surrogates
    # and the noncharacters python-docx rejects on save.
    sanitized = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uD800-\uDFFF﷐-﷯￾￿]",
        "",
        text,
    )
    return sanitized


def _page_information(summary_data: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized page information from summary payload.

    Returns dict with:
        - page_number_integer: The numeric page number (or "?" if null/missing)
        - page_number_end: The right-page number for a two-page spread (or None)
        - page_number_type: 'roman', 'arabic', or 'none'
        - page_types: List of content classifications (content, bibliography, etc.)
        - is_unnumbered: Boolean flag derived from page_number_type == 'none' or
          null integer
        - is_spread: Boolean flag for two-page-spread scans
        - inferred: Boolean flag for a number inferred from the neighbouring
          pages rather than printed on the page (never set when unnumbered)

    A payload without page information is treated as unnumbered: its
    ``page`` field is a scan position, not a printed number.
    """
    page_info = summary_data.get("page_information", {})

    if isinstance(page_info, dict) and page_info:
        page_int = page_info.get("page_number_integer")
        page_num_type = page_info.get("page_number_type", "arabic")

        page_types = page_info.get("page_types")
        if isinstance(page_types, str):
            # Prompted JSON (no enforced schema) may return a single type.
            page_types = [page_types]
        if not isinstance(page_types, list) or not page_types:
            page_types = ["content"]
        else:
            # Drop non-string debris (e.g. dicts from a corrupt or reused
            # log): downstream set(page_types) membership checks would raise
            # TypeError on unhashable entries and kill the whole render.
            page_types = [pt for pt in page_types if isinstance(pt, str)] or ["content"]

        is_spread = bool(page_info.get("is_two_page_spread", False))
        page_end = page_info.get("page_number_integer_end")
        if not is_spread:
            page_end = None

        is_unnumbered = page_num_type == "none" or page_int is None
        if is_unnumbered:
            page_num_type = "none"
            page_int = "?"
            page_end = None
        elif is_spread and not isinstance(page_end, int) and isinstance(page_int, int):
            page_end = page_int + 1

        return {
            "page_number_integer": page_int,
            "page_number_end": page_end,
            "page_number_type": page_num_type,
            "page_types": page_types,
            "is_unnumbered": is_unnumbered,
            "is_spread": is_spread,
            "inferred": bool(page_info.get("number_inferred")) and not is_unnumbered,
        }

    return {
        "page_number_integer": "?",
        "page_number_end": None,
        "page_number_type": "none",
        "page_types": ["content"],
        "is_unnumbered": True,
        "is_spread": False,
        "inferred": False,
    }


# Page types that should have bullet points extracted (summarizable prose)
PAGE_TYPES_WITH_BULLETS = {
    "content",
    "abstract",
    "preface",
    "appendix",
    "figures_tables_sources",
}

# Page types shown in Document Structure section (ordered by typical document position)
STRUCTURE_PAGE_TYPE_ORDER = [
    "title_page",
    "copyright",
    "abstract",
    "table_of_contents",
    "preface",
    "figures_tables_sources",
    "appendix",
    "bibliography",
    "index",
    "other",
]

# Human-readable labels for page types
PAGE_TYPE_LABELS = {
    "content": "Content",
    "preface": "Preface",
    "appendix": "Appendix",
    "figures_tables_sources": "Figures, Tables & Sources",
    "table_of_contents": "Table of Contents",
    "bibliography": "Bibliography",
    "title_page": "Title Page",
    "index": "Index",
    "blank": "Blank Pages",
    "abstract": "Abstract",
    "copyright": "Copyright",
    "other": "Other",
}


def _normalize_references(references: Any) -> list[tuple[str, bool]]:
    """Normalize a page's ``references`` array into ``(text, is_partial)`` tuples.

    Items follow the schema form ``{"citation", "is_partial"}``. A bare string,
    which prompted JSON without an enforced schema may return, is a complete
    reference. Malformed items (other types, non-string citation, empty text)
    are skipped.
    """
    normalized: list[tuple[str, bool]] = []
    if not isinstance(references, list):
        return normalized
    for item in references:
        raw: Any
        if isinstance(item, str):
            raw, is_partial = item, False
        elif isinstance(item, dict):
            raw, is_partial = item.get("citation"), bool(item.get("is_partial", False))
        else:
            continue
        if not isinstance(raw, str) or not raw.strip():
            continue
        normalized.append((raw.strip(), is_partial))
    return normalized


def _should_render_bullets(page_types: list[str]) -> bool:
    """Check if page should have bullet points rendered based on its types."""
    return bool(set(page_types) & PAGE_TYPES_WITH_BULLETS)


def _get_structure_types(page_types: list[str]) -> list[str]:
    """Return page types that should appear in Document Structure section."""
    return [pt for pt in page_types if pt in STRUCTURE_PAGE_TYPE_ORDER]


# A single bullet starting with one of these bracketed prefixes is a
# summary-layer failure marker that must stay visible in the output, unlike the
# blank-page sentinels, which drop the page.
_ERROR_PLACEHOLDER_MARKERS: tuple[str, ...] = tuple(
    m for m in ERROR_MARKERS if m.startswith("[")
)


def collapse_internal_newlines(text: str) -> str:
    """Collapse internal whitespace-newline runs to a single space.

    Internal newlines in a bullet or citation string break Markdown list and
    numbered-reference structure and land raw inside DOCX runs; flattening them
    at render time keeps each item a single logical line.
    """
    return re.sub(r"\s*[\r\n]+\s*", " ", text)


def _string_bullets(bullet_points: Any) -> list[str]:
    """Return only the string entries of a bullet-point list.

    Non-string items (None, ints, dicts from a corrupt or reused log) would
    crash the writers on ``.strip()``/rendering, so they are dropped here. A
    non-list input yields an empty list.
    """
    if not isinstance(bullet_points, list):
        return []
    return [b for b in bullet_points if isinstance(b, str)]


def _is_error_placeholder_bullet(bullet: str) -> bool:
    """Return True if *bullet* is a bracketed summary-layer error placeholder."""
    text = bullet.strip().lower()
    return any(text.startswith(marker) for marker in _ERROR_PLACEHOLDER_MARKERS)


def _is_error_result(result: dict[str, Any], bullet_points: list[str]) -> bool:
    """Return True if *result* is a failed-page placeholder that must stay visible.

    A failure carries either an explicit ``error`` key (set by
    ``llm.summary.placeholder_summary``) or a single bracketed
    error-placeholder bullet. Blank-page sentinels ("[empty page]") are
    deliberately excluded so genuinely blank pages are still dropped.
    """
    if result.get("error"):
        return True
    return len(bullet_points) == 1 and _is_error_placeholder_bullet(bullet_points[0])


def _is_meaningful_summary(summary_data: dict[str, Any]) -> bool:
    """Check if a summary is meaningful based on page_types and content."""
    # A page carrying references is meaningful even without bullet content:
    # dropping it here would discard its citations before they reach the
    # CitationManager.
    if _normalize_references(summary_data.get("references")):
        return True

    bullet_points = _string_bullets(summary_data.get("bullet_points"))

    # A failed-page placeholder is kept regardless of its page_types so the
    # coverage gap stays visible; blank-page sentinels fall through and drop.
    if _is_error_result(summary_data, bullet_points):
        return True

    page_info = _page_information(summary_data)
    page_types = page_info.get("page_types", ["content"])

    if page_types == ["blank"]:
        return False

    has_bullet_types = _should_render_bullets(page_types)

    if has_bullet_types:
        if bullet_points:
            if len(bullet_points) == 1:
                marker_text = bullet_points[0].strip().lower()
                if marker_text.startswith("[") and any(
                    marker in marker_text for marker in ERROR_MARKERS
                ):
                    return bool(_get_structure_types(page_types))
            return True
        return bool(_get_structure_types(page_types))

    return bool(_get_structure_types(page_types))


@dataclass
class PageRenderData:
    """Pre-computed rendering data for a single summary page."""

    page_number: int | str
    page_number_type: str
    page_types: list[str]
    is_unnumbered: bool
    bullet_points: list[str]
    heading_text: str
    page_number_end: int | None = None
    is_spread: bool = False
    # 1-based position of the scan in the PDF or image folder.
    position: int | None = None
    # The printed-style number was inferred, not read from the page.
    inferred: bool = False


@dataclass
class SummaryData:
    """Aggregated data produced by :func:`prepare_summary_data`.

    Attributes:
        filtered_results: Summary results after filtering empty pages.
        page_type_pages: Mapping from structure page type to a list of
            ``(page_number, numbering_type)`` tuples, where numbering_type is
            'roman' or 'arabic'. Two-page spreads contribute both page numbers.
        content_page_count: Number of pages with bullet-renderable content.
        page_render_items: Per-page rendering data (only pages with bullets).
        source_page_count: Number of source pages BEFORE empty-page filtering
            (the true total page count reported in the document metadata).
    """

    filtered_results: list[dict[str, Any]]
    page_type_pages: dict[str, list[tuple[int, str]]]
    content_page_count: int
    page_render_items: list[PageRenderData] = field(default_factory=list)
    source_page_count: int = 0


def _scan_position(result: dict[str, Any], fallback_positions: dict[int, int]) -> int:
    """Return the 1-based scan position of *result* in the PDF/image folder.

    Read from ``original_input_order_index``; a result without it falls back to
    its place in the unfiltered result list.
    """
    index = result.get("original_input_order_index")
    if isinstance(index, int) and not isinstance(index, bool):
        return index + 1
    return fallback_positions[id(result)]


def prepare_summary_data(
    summary_results: list[dict[str, Any]],
    citation_manager: CitationManager,
    position_label: str = "pdf",
) -> SummaryData:
    """Shared preparation logic for DOCX and Markdown summary writers.

    Filters empty pages, collects citations, builds page-type mapping,
    and pre-computes per-page rendering data.

    Args:
        summary_results: Raw summary results from the API.
        citation_manager: A :class:`CitationManager` instance (passed in for
            testability).
        position_label: ``"pdf"`` or ``"image"``: how a page without a printed
            number is located (``PDF p. N`` or ``Image N``).

    Returns:
        A :class:`SummaryData` instance with all prepared data.
    """
    fallback_positions = {
        id(result): position for position, result in enumerate(summary_results, 1)
    }
    filtered_results = filter_empty_pages(summary_results)
    if len(filtered_results) < len(summary_results):
        logger.info(
            "Filtered out %s pages with no useful content",
            len(summary_results) - len(filtered_results),
        )

    page_type_pages: dict[str, list[tuple[int, str]]] = {
        pt: [] for pt in STRUCTURE_PAGE_TYPE_ORDER
    }
    page_render_items: list[PageRenderData] = []
    content_page_count = 0

    for result in filtered_results:
        page_info = _page_information(result)
        page_number = page_info["page_number_integer"]
        page_end = page_info["page_number_end"]
        page_num_type = page_info["page_number_type"]
        is_spread = page_info["is_spread"]
        page_types = page_info["page_types"]
        inferred = page_info["inferred"]
        position = _scan_position(result, fallback_positions)
        references = _normalize_references(result.get("references"))

        numbered_pages: list[int] = []
        if isinstance(page_number, int):
            numbered_pages.append(page_number)
            if is_spread and isinstance(page_end, int):
                numbered_pages.append(page_end)

        if references:
            # A numbered page (both pages of a spread) is cited by its printed
            # number, keeping roman numerals roman and inferred numbers marked;
            # an unnumbered page by its scan position. The citation manager
            # deduplicates per locator.
            if numbered_pages:
                namespace = (
                    PRINTED_ROMAN if page_num_type == "roman" else PRINTED_ARABIC
                )
                for pg in numbered_pages:
                    citation_manager.add_citations(
                        references, (namespace, pg, inferred)
                    )
            else:
                citation_manager.add_citations(
                    references, (position_label, position, False)
                )

        for pg in numbered_pages:
            for pt in _get_structure_types(page_types):
                page_type_pages[pt].append((pg, page_num_type))

        bullet_points = _string_bullets(result.get("bullet_points"))

        # A failed page (page_types typically ["other"]) is rendered with its
        # placeholder bullet so the reader sees the coverage gap in place.
        renders_bullets = _should_render_bullets(page_types)
        if renders_bullets:
            content_page_count += 1
        if bullet_points and (
            renders_bullets or _is_error_result(result, bullet_points)
        ):
            heading = format_page_heading(
                page_number,
                page_num_type,
                page_types,
                page_info["is_unnumbered"],
                page_number_end=page_end,
                is_spread=is_spread,
                inferred=inferred,
                scan=(position_label, position),
            )
            page_render_items.append(
                PageRenderData(
                    page_number=page_number,
                    page_number_type=page_num_type,
                    page_types=page_types,
                    is_unnumbered=page_info["is_unnumbered"],
                    bullet_points=bullet_points,
                    heading_text=heading,
                    page_number_end=page_end,
                    is_spread=is_spread,
                    position=position,
                    inferred=inferred,
                )
            )

    return SummaryData(
        filtered_results=filtered_results,
        page_type_pages=page_type_pages,
        content_page_count=content_page_count,
        page_render_items=page_render_items,
        source_page_count=len(summary_results),
    )


def build_render_context(
    summary_results: list[dict[str, Any]],
    position_label: str = "pdf",
    *,
    openalex_enabled: bool = True,
    max_requests: int | None = None,
    openalex: OpenAlexClient | None = None,
    stop: threading.Event | None = None,
) -> tuple[CitationManager, SummaryData]:
    """Build one enriched-once render context shared by both writers.

    Collects citations, runs the conservative :meth:`CitationManager.consolidate`
    merge, and returns the (not-yet-enriched) manager plus the prepared
    :class:`SummaryData`. The caller runs OpenAlex enrichment exactly once
    (``enrich_if_enabled``) so both the DOCX and Markdown writers render from
    identical, already-enriched citations. ``position_label`` is ``"pdf"`` or
    ``"image"`` (see :func:`prepare_summary_data`). The OpenAlex options and
    the *stop* signal are stored on the manager; *openalex* is the run's
    shared client.
    """
    citation_manager = CitationManager(
        openalex_enabled=openalex_enabled,
        max_requests=(
            max_requests if max_requests is not None else DEFAULT_MAX_API_REQUESTS
        ),
        openalex=openalex,
        stop=stop,
    )
    data = prepare_summary_data(
        summary_results, citation_manager, position_label=position_label
    )
    citation_manager.consolidate()
    return citation_manager, data


def format_page_heading(
    page_number: int | str,
    page_number_type: str,
    page_types: list[str],
    is_unnumbered: bool,
    page_number_end: int | None = None,
    is_spread: bool = False,
    inferred: bool = False,
    scan: tuple[str, int] | None = None,
) -> str:
    """Format page heading text based on page_types and numbering.

    Returns bare heading text (no markdown prefix). Writers add their own
    format-specific prefix (e.g. ``## `` for Markdown). Two-page spreads render
    a page range ("Pages 12-13" / "Pages xii-xiii").

    A page without a printed number is located by its *scan* locator
    ``(position_label, position)``, the 1-based position in the PDF or image
    folder: "[No printed number; PDF p. 7]", or "Image 7" when the label is
    ``"image"``. A spread is one scan, so an unnumbered spread gets one
    position: "[No printed numbers; PDF p. 7]". An *inferred* number is
    bracketed: "Page [6]", "Page [vi]", "Pages [6]-[7]".
    """
    type_prefix = ""
    if "abstract" in page_types and "content" not in page_types:
        type_prefix = "[Abstract] "
    elif "preface" in page_types:
        type_prefix = "[Preface] "
    elif "appendix" in page_types:
        type_prefix = "[Appendix] "
    elif "figures_tables_sources" in page_types:
        type_prefix = "[Figures/Tables] "

    if scan is None:
        locator = ""
    elif scan[0] == "image":
        locator = f"; Image {scan[1]}"
    else:
        locator = f"; PDF p. {scan[1]}"

    def number(value: int | str) -> str:
        if page_number_type == "roman" and isinstance(value, int):
            text = int_to_roman(value)
        else:
            text = str(value)
        return f"[{text}]" if inferred else text

    if is_spread:
        if (
            page_number_type == "none"
            or is_unnumbered
            or not isinstance(page_number, int)
        ):
            if not locator:
                return f"{type_prefix}[Unnumbered spread]"
            return f"{type_prefix}[No printed numbers{locator}]"
        end = page_number_end if isinstance(page_number_end, int) else page_number + 1
        return f"{type_prefix}Pages {number(page_number)}-{number(end)}"

    if page_number_type == "none" or is_unnumbered:
        if not locator:
            return f"{type_prefix}[Unnumbered page]"
        return f"{type_prefix}[No printed number{locator}]"
    return f"{type_prefix}Page {number(page_number)}"


def _compact_int_ranges(nums: list[int]) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` ranges of consecutive integers."""
    ordered = sorted(set(nums))
    ranges: list[tuple[int, int]] = []
    start = end = ordered[0]
    for n in ordered[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append((start, end))
            start = end = n
    ranges.append((start, end))
    return ranges


def format_structure_page_range(entries: list[tuple[int, str]]) -> str:
    """Format Document Structure page entries as a compact range string.

    Each entry is ``(page_number, numbering_type)`` where numbering_type is
    'roman' or 'arabic'. Roman (front-matter) pages are rendered first as
    lowercase Roman numerals, then Arabic pages, e.g. ``"pp. iii-xii, 100-105"``.
    Consecutive integers are compacted within each numbering system.
    """
    if not entries:
        return ""

    roman_nums = sorted({n for n, t in entries if t == "roman"})
    arabic_nums = sorted({n for n, t in entries if t != "roman"})

    parts: list[str] = []
    if roman_nums:
        for start, end in _compact_int_ranges(roman_nums):
            if start == end:
                parts.append(int_to_roman(start))
            else:
                parts.append(f"{int_to_roman(start)}-{int_to_roman(end)}")
    if arabic_nums:
        for start, end in _compact_int_ranges(arabic_nums):
            if start == end:
                parts.append(str(start))
            else:
                parts.append(f"{start}-{end}")

    total = len(roman_nums) + len(arabic_nums)
    prefix = "p." if total == 1 else "pp."
    return f"{prefix} {', '.join(parts)}"


def filter_empty_pages(
    summary_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove entries without meaningful summary content."""
    return [result for result in summary_results if _is_meaningful_summary(result)]
