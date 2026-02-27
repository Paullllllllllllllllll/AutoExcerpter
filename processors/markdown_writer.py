"""Markdown summary document creation for AutoExcerpter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from modules import app_config as config
from modules.logger import setup_logger
from modules.roman_numerals import int_to_roman
from processors.citation_manager import CitationManager
from processors.file_manager import (
    sanitize_for_xml,
    _extract_summary_payload,
    _page_information,
    _should_render_bullets,
    _get_structure_types,
    _format_page_range,
    filter_empty_pages,
    STRUCTURE_PAGE_TYPE_ORDER,
    PAGE_TYPE_LABELS,
)

logger = setup_logger(__name__)


def _format_page_heading_md(
    page_number: Any, page_number_type: str, page_types: list[str], is_unnumbered: bool
) -> str:
    """Format page heading for markdown output based on page_types and numbering."""
    type_prefix = ""
    if "abstract" in page_types and "content" not in page_types:
        type_prefix = "[Abstract] "
    elif "preface" in page_types:
        type_prefix = "[Preface] "
    elif "appendix" in page_types:
        type_prefix = "[Appendix] "
    elif "figures_tables_sources" in page_types:
        type_prefix = "[Figures/Tables] "

    if page_number_type == "roman" and isinstance(page_number, int):
        roman_str = int_to_roman(page_number)
        return f"## {type_prefix}Page {roman_str}"
    elif page_number_type == "none" or is_unnumbered:
        return f"## {type_prefix}[Unnumbered page]"
    else:
        return f"## {type_prefix}Page {page_number}"


def create_markdown_summary(
    summary_results: list[dict[str, Any]], output_path: Path, document_name: str
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
    """
    filtered_results = filter_empty_pages(summary_results)
    if len(filtered_results) < len(summary_results):
        logger.info(
            "Filtered out %s pages with no useful content",
            len(summary_results) - len(filtered_results),
        )

    citation_manager = CitationManager(polite_pool_email=config.CITATION_OPENALEX_EMAIL)

    page_type_pages: dict[str, list[int]] = {pt: [] for pt in STRUCTURE_PAGE_TYPE_ORDER}

    for result in filtered_results:
        summary_payload = _extract_summary_payload(result)
        page_info = _page_information(summary_payload)
        page_number = page_info["page_number_integer"]
        page_types = page_info["page_types"]
        references = summary_payload.get("references") or []

        if references and isinstance(page_number, int):
            citation_manager.add_citations(references, page_number)

        if isinstance(page_number, int):
            for pt in _get_structure_types(page_types):
                page_type_pages[pt].append(page_number)

    lines: list[str] = []

    # === SECTION 1: Title ===
    lines.append(f"# Summary of {sanitize_for_xml(document_name)}")
    lines.append("")

    content_pages = sum(
        1
        for r in filtered_results
        if _should_render_bullets(
            _page_information(_extract_summary_payload(r)).get(
                "page_types", ["content"]
            )
        )
    )
    lines.append(
        f"*Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Content pages: {content_pages} | Total pages: {len(filtered_results)}*"
    )
    lines.append("")

    # === SECTION 2: Document Structure ===
    has_structure_info = any(pages for pages in page_type_pages.values())
    if has_structure_info:
        lines.append("## Document Structure")
        lines.append("")

        for pt in STRUCTURE_PAGE_TYPE_ORDER:
            pages = page_type_pages.get(pt, [])
            if pages:
                label = PAGE_TYPE_LABELS.get(pt, pt.replace("_", " ").title())
                page_range = _format_page_range(pages)
                lines.append(f"- **{label}**: {page_range}")

        lines.append("")
        lines.append("---")
        lines.append("")

    # === SECTION 3: Content Summaries ===
    for result in filtered_results:
        summary_payload = _extract_summary_payload(result)
        page_info = _page_information(summary_payload)
        page_number = page_info["page_number_integer"]
        page_number_type = page_info["page_number_type"]
        page_types = page_info["page_types"]
        bullet_points = summary_payload.get("bullet_points") or []

        if _should_render_bullets(page_types) and bullet_points:
            lines.append(
                _format_page_heading_md(
                    page_number,
                    page_number_type,
                    page_types,
                    page_info["is_unnumbered"],
                )
            )
            lines.append("")

            for point in bullet_points:
                sanitized_point = sanitize_for_xml(point)
                lines.append(f"- {sanitized_point}")

            lines.append("")

    # === Consolidated References ===
    if citation_manager.citations:
        logger.info(
            "Processing %d unique citations for consolidated references section",
            len(citation_manager.citations),
        )

        if config.CITATION_ENABLE_OPENALEX:
            citation_manager.enrich_with_metadata(
                max_requests=config.CITATION_MAX_API_REQUESTS
            )
        else:
            logger.info("OpenAlex enrichment disabled - skipping metadata lookup")

        lines.append("---")
        lines.append("")
        lines.append("## Consolidated References")
        lines.append("")
        lines.append(
            "*The following references were extracted from the document and consolidated. "
            "Duplicate citations have been merged, showing all pages where each citation appears. "
            "Where available, hyperlinks provide access to extended metadata via OpenAlex.*"
        )
        lines.append("")

        citations_with_pages = citation_manager.get_citations_with_pages()

        for idx, (citation, page_range_str) in enumerate(citations_with_pages, start=1):
            citation_text = sanitize_for_xml(citation.raw_text)

            if citation.url:
                line = f"{idx}. [{citation_text}]({citation.url})"
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

            lines.append(line)

    # === Write file ===
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown summary saved to %s", output_path)
