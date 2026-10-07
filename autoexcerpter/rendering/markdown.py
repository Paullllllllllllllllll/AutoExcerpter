"""Markdown summary document creation for AutoExcerpter."""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from autoexcerpter.rendering.citations import (
    Citation,
    CitationManager,
    enrich_if_enabled,
)
from autoexcerpter.rendering.summary import (
    PAGE_TYPE_LABELS,
    STRUCTURE_PAGE_TYPE_ORDER,
    PageRenderData,
    SummaryData,
    collapse_internal_newlines,
    format_structure_page_range,
    prepare_summary_data,
    sanitize_for_xml,
)

logger = logging.getLogger(__name__)


def create_markdown_summary(
    summary_results: list[dict[str, Any]],
    output_path: Path,
    document_name: str,
    citation_manager: CitationManager | None = None,
    data: Any = None,
) -> None:
    """Create a Markdown summary document from structured summary results.

    Output order:
    1. Title + Metadata
    2. Document Structure (page type overview)
    3. Content Summaries (in document order)
    4. Consolidated References

    Args:
        summary_results: List of summary result dictionaries from the API.
        output_path: Path where the markdown file will be written.
        document_name: Name of the source document for the title.
        citation_manager: Optional pre-built, consolidated, and enriched citation
            manager shared with the DOCX writer. Built (with default OpenAlex
            options) and enriched here when omitted.
        data: Optional pre-computed :class:`SummaryData` paired with
            *citation_manager*.
    """
    if citation_manager is None or data is None:
        citation_manager = CitationManager()
        data = prepare_summary_data(summary_results, citation_manager)
        citation_manager.consolidate()
        enrich_if_enabled(citation_manager)

    lines = [
        *_title_lines(document_name, data),
        *_structure_lines(data.page_type_pages),
        *_page_summary_lines(data.page_render_items),
        *_reference_lines(citation_manager),
    ]
    _write_atomically("\n".join(lines) + "\n", output_path)
    logger.info("Markdown summary saved to %s", output_path)


def _title_lines(document_name: str, data: SummaryData) -> list[str]:
    """Return the H1 title and the italic metadata line."""
    return [
        f"# {sanitize_for_xml(document_name)}",
        "",
        f"*Summary generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"· {data.content_page_count} content pages "
        f"· {data.source_page_count} total pages*",
        "",
    ]


def _structure_lines(page_type_pages: dict[str, list[tuple[int, str]]]) -> list[str]:
    """Return the Document Structure section, empty when no type has pages."""
    if not any(pages for pages in page_type_pages.values()):
        return []
    lines = ["## Document Structure", ""]
    for pt in STRUCTURE_PAGE_TYPE_ORDER:
        pages = page_type_pages.get(pt, [])
        if pages:
            label = PAGE_TYPE_LABELS.get(pt, pt.replace("_", " ").title())
            page_range = format_structure_page_range(pages)
            lines.append(f"- **{label}**: {page_range}")
    lines.append("")
    return lines


def _page_summary_lines(page_render_items: list[PageRenderData]) -> list[str]:
    """Return the page summaries: one H2 grouping an H3 per page."""
    if not page_render_items:
        return []
    lines = ["## Page Summaries", ""]
    for page_item in page_render_items:
        # Sanitized for the same reason as in the DOCX writer: a page number
        # from a lenient endpoint may carry XML-illegal control characters.
        lines.append(f"### {sanitize_for_xml(page_item.heading_text)}")
        lines.append("")
        for point in page_item.bullet_points:
            sanitized_point = sanitize_for_xml(collapse_internal_newlines(point))
            lines.append(f"- {sanitized_point}")
        lines.append("")
    return lines


def _reference_lines(citation_manager: CitationManager) -> list[str]:
    """Return the Consolidated References section, empty without citations."""
    if not citation_manager.citations:
        return []
    logger.info(
        "Processing %d unique citations for consolidated references section",
        len(citation_manager.citations),
    )
    lines = [
        "---",
        "",
        "## Consolidated References",
        "",
        "*The following references were extracted from the document and "
        "consolidated. Duplicate citations have been merged, showing all "
        "pages where each citation appears. Where available, hyperlinks "
        "provide access to extended metadata via OpenAlex.*",
        "",
    ]
    citations_with_pages = citation_manager.get_citations_with_pages()
    for idx, (citation, page_range_str) in enumerate(citations_with_pages, start=1):
        lines.append(_reference_line(idx, citation, page_range_str))
    return lines


def _reference_line(idx: int, citation: Citation, page_range_str: str) -> str:
    """Return one numbered reference with its pages and OpenAlex metadata."""
    citation_text = sanitize_for_xml(collapse_internal_newlines(citation.raw_text))

    if citation.url:
        # Escape link-breaking characters: brackets in the text and
        # parentheses/spaces/angle brackets in the URL (DOIs like
        # S0140-6736(00)… or ones carrying <...> segments).
        link_text = citation_text.replace("[", "\\[").replace("]", "\\]")
        link_url = (
            citation.url.replace("(", "%28")
            .replace(")", "%29")
            .replace(" ", "%20")
            .replace("<", "%3C")
            .replace(">", "%3E")
        )
        line = f"{idx}. [{link_text}]({link_url})"
    else:
        line = f"{idx}. {citation_text}"

    if page_range_str:
        line += f" *({page_range_str})*"

    if citation.metadata:
        meta_parts = []
        if citation.doi:
            meta_parts.append(f"DOI: {citation.doi}")
        if citation.metadata.get("publication_year"):
            meta_parts.append(f"Year: {citation.metadata['publication_year']}")
        if meta_parts:
            line += f" *[{', '.join(meta_parts)}]*"

    return line


def _write_atomically(text: str, output_path: Path) -> None:
    """Write to a sibling ``.tmp`` file and replace the target atomically.

    A crash mid-write never leaves a truncated .md that resume would trust as
    complete; the temp file is removed on any failure.
    """
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp_path, output_path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
