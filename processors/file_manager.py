"""File management utilities for transcription and summary outputs.

This module provides shared helpers for content filtering, XML sanitization,
and transcription text output. DOCX, Markdown, and log writing are handled
by dedicated submodules (docx_writer, markdown_writer, log_manager).

Re-exports from submodules are provided for backward compatibility.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from modules.constants import ERROR_MARKERS
from modules.logger import setup_logger

logger = setup_logger(__name__)


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
    """Return the summary payload dict, handling both flat and nested formats.

    Flat structure (preferred): page_information, bullet_points, references at top level.
    Legacy nested: summary.summary containing those fields.
    """
    if "page_information" in result and isinstance(
        result.get("page_information"), dict
    ):
        return result

    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return {}

    nested_summary = summary.get("summary")
    if isinstance(nested_summary, dict):
        return nested_summary
    return summary


def _page_information(summary_data: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized page information from summary payload.

    Returns dict with:
        - page_number_integer: The numeric page number (or "?" if null/missing)
        - page_number_type: 'roman', 'arabic', or 'none'
        - page_types: List of content classifications (content, bibliography, etc.)
        - is_unnumbered: Boolean flag derived from page_number_type == 'none' or null integer
    """
    page_info = summary_data.get("page_information", {})

    if isinstance(page_info, dict) and page_info:
        page_int = page_info.get("page_number_integer")
        page_num_type = page_info.get("page_number_type", "arabic")

        page_types = page_info.get("page_types")
        if page_types is None:
            legacy_type = page_info.get("page_type", "content")
            page_types = [legacy_type] if legacy_type else ["content"]
        elif isinstance(page_types, str):
            page_types = [page_types]
        elif not isinstance(page_types, list) or not page_types:
            page_types = ["content"]

        is_unnumbered = page_num_type == "none" or page_int is None
        if is_unnumbered:
            page_num_type = "none"
            page_int = "?"

        return {
            "page_number_integer": page_int,
            "page_number_type": page_num_type,
            "page_types": page_types,
            "is_unnumbered": is_unnumbered,
        }

    page_val = summary_data.get("page", "?")
    return {
        "page_number_integer": page_val,
        "page_number_type": "arabic",
        "page_types": ["content"],
        "is_unnumbered": False,
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


def _format_page_range(pages: list[int]) -> str:
    """Format a list of page numbers as a compact range string.

    Examples:
        [1, 2, 3, 5, 7, 8, 9] -> "pp. 1-3, 5, 7-9"
        [5] -> "p. 5"
    """
    if not pages:
        return ""

    pages = sorted(set(pages))
    if len(pages) == 1:
        return f"p. {pages[0]}"

    ranges = []
    start = pages[0]
    end = pages[0]

    for page in pages[1:]:
        if page == end + 1:
            end = page
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = page

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return f"pp. {', '.join(ranges)}"


# ============================================================================
# Transcription Text Output
# ============================================================================
def write_transcription_to_text(
    transcription_results: list[dict[str, Any]],
    output_path: Path,
    document_name: str,
    item_type: str,
    total_elapsed_time: float,
    source_path: Path,
) -> bool:
    """Persist transcription output as a text file alongside basic metadata."""
    elapsed_str = str(timedelta(seconds=int(total_elapsed_time)))
    successes = sum(1 for result in transcription_results if "error" not in result)
    failures = len(transcription_results) - successes

    try:
        with output_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(f"# Transcription of: {document_name}\n")
            file_handle.write(f"# Source Path: {source_path}\n")
            file_handle.write(f"# Type: {item_type}\n")
            file_handle.write(
                f"# Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            file_handle.write(
                f"# Total images processed: {len(transcription_results)}\n"
            )
            file_handle.write(f"# Successfully transcribed: {successes}\n")
            file_handle.write(f"# Failed items: {failures}\n")
            file_handle.write(
                f"# Total processing time for this item: {elapsed_str}\n\n"
            )

            for index, result in enumerate(transcription_results):
                file_handle.write(
                    result.get("transcription", "[ERROR] Transcription data missing")
                )
                if index < len(transcription_results) - 1:
                    file_handle.write("\n")

        logger.info("Transcription text file saved: %s", output_path)
        return True
    except OSError as exc:
        logger.error(
            "Error writing transcription to text file %s: %s", output_path, exc
        )
        return False


# ============================================================================
# Content Filtering
# ============================================================================
def filter_empty_pages(summary_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove entries without meaningful summary content."""
    filtered: list[dict[str, Any]] = []
    for result in summary_results:
        summary_payload = _extract_summary_payload(result)
        if summary_payload and _is_meaningful_summary(summary_payload):
            filtered.append(result)
    return filtered
