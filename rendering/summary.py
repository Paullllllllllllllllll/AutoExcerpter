"""Summary data preparation and page-rendering helpers for AutoExcerpter.

Shared by the DOCX and Markdown writers. Contains XML sanitization, page
information extraction, content filtering, and the shared
:func:`prepare_summary_data` pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from config.constants import ERROR_MARKERS
from config.logger import setup_logger

if TYPE_CHECKING:
    from rendering.citations import CitationManager

logger = setup_logger(__name__)


# ============================================================================
# Roman Numeral Helper (inlined from modules/roman_numerals.py)
# ============================================================================

# Mapping of integer values to lowercase Roman numeral symbols.
_ROMAN_NUMERAL_VALUES = [
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
]


def int_to_roman(num: int) -> str:
    """Convert integer to lowercase Roman numeral string.

    Args:
        num: Positive integer to convert.

    Returns:
        Lowercase Roman numeral string.
    """
    if num <= 0:
        return ""
    result = []
    for value, numeral in _ROMAN_NUMERAL_VALUES:
        while num >= value:
            result.append(numeral)
            num -= value
    return "".join(result)


# ============================================================================
# Text Sanitization
# ============================================================================


def sanitize_for_xml(text: str | None) -> str:
    """Return XML-safe text for DOCX output by removing control characters."""
    if not text:
        return ""
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return sanitized


# ============================================================================
# Summary Data Extraction (shared by DOCX and Markdown writers)
# ============================================================================


def _extract_summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return the summary payload dict from flat structure.

    Expects page_information, bullet_points, references at top level.
    Falls back to summary dict if page_information is not at top level.
    """
    if "page_information" in result and isinstance(
        result.get("page_information"), dict
    ):
        return result

    summary = result.get("summary", {})
    return summary if isinstance(summary, dict) else {}


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
    """
    page_info = summary_data.get("page_information", {})

    if isinstance(page_info, dict) and page_info:
        page_int = page_info.get("page_number_integer")
        page_num_type = page_info.get("page_number_type", "arabic")

        page_types = page_info.get("page_types")
        if page_types is None:
            page_types = ["content"]
        elif isinstance(page_types, str):
            page_types = [page_types]
        elif not isinstance(page_types, list) or not page_types:
            page_types = ["content"]

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
            # Numbered spread missing an explicit end: derive it.
            page_end = page_int + 1

        return {
            "page_number_integer": page_int,
            "page_number_end": page_end,
            "page_number_type": page_num_type,
            "page_types": page_types,
            "is_unnumbered": is_unnumbered,
            "is_spread": is_spread,
        }

    page_val = summary_data.get("page", "?")
    return {
        "page_number_integer": page_val,
        "page_number_end": None,
        "page_number_type": "arabic",
        "page_types": ["content"],
        "is_unnumbered": False,
        "is_spread": False,
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

    Accepts both the current schema form (objects ``{"citation", "is_partial"}``)
    and the legacy string form (older cached/resumed JSON), which is treated as a
    complete, non-partial reference. Malformed items (non-string citation, empty
    text) are skipped.
    """
    normalized: list[tuple[str, bool]] = []
    if not isinstance(references, list):
        return normalized
    for item in references:
        if isinstance(item, str):
            text = item
            is_partial = False
        elif isinstance(item, dict):
            raw = item.get("citation")
            if not isinstance(raw, str):
                continue
            text = raw
            is_partial = bool(item.get("is_partial", False))
        else:
            continue
        if not text.strip():
            continue
        normalized.append((text.strip(), is_partial))
    return normalized


def _should_render_bullets(page_types: list[str]) -> bool:
    """Check if page should have bullet points rendered based on its types."""
    return bool(set(page_types) & PAGE_TYPES_WITH_BULLETS)


def _get_structure_types(page_types: list[str]) -> list[str]:
    """Return page types that should appear in Document Structure section."""
    return [pt for pt in page_types if pt in STRUCTURE_PAGE_TYPE_ORDER]


def _is_meaningful_summary(summary_data: dict[str, Any]) -> bool:
    """Check if a summary is meaningful based on page_types and content."""
    page_info = _page_information(summary_data)
    page_types = page_info.get("page_types", ["content"])

    if page_types == ["blank"]:
        return False

    has_bullet_types = _should_render_bullets(page_types)

    if has_bullet_types:
        bullet_points = summary_data.get("bullet_points") or []
        if bullet_points:
            if len(bullet_points) == 1:
                marker_text = bullet_points[0].strip().lower()
                if any(marker in marker_text for marker in ERROR_MARKERS):
                    return bool(_get_structure_types(page_types))
            return True
        return bool(_get_structure_types(page_types))

    return bool(_get_structure_types(page_types))


# ============================================================================
# Shared Summary Preparation (used by DOCX and Markdown writers)
# ============================================================================


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
    """

    filtered_results: list[dict[str, Any]]
    page_type_pages: dict[str, list[tuple[int, str]]]
    content_page_count: int
    page_render_items: list[PageRenderData] = field(default_factory=list)


def prepare_summary_data(
    summary_results: list[dict[str, Any]],
    citation_manager: CitationManager,
) -> SummaryData:
    """Shared preparation logic for DOCX and Markdown summary writers.

    Filters empty pages, collects citations, builds page-type mapping,
    and pre-computes per-page rendering data.

    Args:
        summary_results: Raw summary results from the API.
        citation_manager: A :class:`CitationManager` instance (passed in for
            testability).

    Returns:
        A :class:`SummaryData` instance with all prepared data.
    """
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
        summary_payload = _extract_summary_payload(result)
        page_info = _page_information(summary_payload)
        page_number = page_info["page_number_integer"]
        page_end = page_info["page_number_end"]
        page_num_type = page_info["page_number_type"]
        is_spread = page_info["is_spread"]
        page_types = page_info["page_types"]
        references = _normalize_references(summary_payload.get("references"))

        # Collect the integer page number(s) this scan covers (both for spreads).
        numbered_pages: list[int] = []
        if isinstance(page_number, int):
            numbered_pages.append(page_number)
            if is_spread and isinstance(page_end, int):
                numbered_pages.append(page_end)

        if references:
            # Include citations on unnumbered pages (recorded with page=None and
            # rendered as "unnumbered") rather than silently discarding them. For
            # a numbered spread, record the references against both pages; the
            # citation manager deduplicates per page.
            if numbered_pages:
                for pg in numbered_pages:
                    citation_manager.add_citations(references, pg)
            else:
                citation_manager.add_citations(references, None)

        for pg in numbered_pages:
            for pt in _get_structure_types(page_types):
                page_type_pages[pt].append((pg, page_num_type))

        # Track content pages and build render data
        if _should_render_bullets(page_types):
            content_page_count += 1
            bullet_points = summary_payload.get("bullet_points") or []
            if bullet_points:
                heading = format_page_heading(
                    page_number,
                    page_num_type,
                    page_types,
                    page_info["is_unnumbered"],
                    page_number_end=page_end,
                    is_spread=is_spread,
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
                    )
                )

    return SummaryData(
        filtered_results=filtered_results,
        page_type_pages=page_type_pages,
        content_page_count=content_page_count,
        page_render_items=page_render_items,
    )


def build_render_context(
    summary_results: list[dict[str, Any]],
    polite_pool_email: str | None = None,
) -> tuple[CitationManager, SummaryData]:
    """Build one enriched-once render context shared by both writers.

    Collects citations, runs the conservative :meth:`CitationManager.consolidate`
    merge, and returns the (not-yet-enriched) manager plus the prepared
    :class:`SummaryData`. The caller runs OpenAlex enrichment exactly once
    (``enrich_if_enabled``) so both the DOCX and Markdown writers render from
    identical, already-enriched citations.
    """
    from rendering.citations import CitationManager

    citation_manager = CitationManager(polite_pool_email=polite_pool_email)
    data = prepare_summary_data(summary_results, citation_manager)
    citation_manager.consolidate()
    return citation_manager, data


# ============================================================================
# Page Heading Formatting (shared by DOCX and Markdown writers)
# ============================================================================


def format_page_heading(
    page_number: int | str,
    page_number_type: str,
    page_types: list[str],
    is_unnumbered: bool,
    page_number_end: int | None = None,
    is_spread: bool = False,
) -> str:
    """Format page heading text based on page_types and numbering.

    Returns bare heading text (no markdown prefix). Writers add their own
    format-specific prefix (e.g. ``## `` for Markdown). Two-page spreads render
    a page range ("Pages 12-13" / "Pages xii-xiii") or "[Unnumbered spread]".
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

    if is_spread:
        if (
            page_number_type == "none"
            or is_unnumbered
            or not isinstance(page_number, int)
        ):
            return f"{type_prefix}[Unnumbered spread]"
        end = page_number_end if isinstance(page_number_end, int) else page_number + 1
        if page_number_type == "roman":
            return f"{type_prefix}Pages {int_to_roman(page_number)}-{int_to_roman(end)}"
        return f"{type_prefix}Pages {page_number}-{end}"

    if page_number_type == "roman" and isinstance(page_number, int):
        roman_str = int_to_roman(page_number)
        return f"{type_prefix}Page {roman_str}"
    elif page_number_type == "none" or is_unnumbered:
        return f"{type_prefix}[Unnumbered page]"
    else:
        return f"{type_prefix}Page {page_number}"


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


# ============================================================================
# Content Filtering
# ============================================================================


def filter_empty_pages(
    summary_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove entries without meaningful summary content."""
    filtered: list[dict[str, Any]] = []
    for result in summary_results:
        summary_payload = _extract_summary_payload(result)
        if summary_payload and _is_meaningful_summary(summary_payload):
            filtered.append(result)
    return filtered
