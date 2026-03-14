"""Markdown summary document creation for AutoExcerpter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from modules import app_config as config
from modules.logger import setup_logger
from processors.citation_manager import CitationManager
from processors.file_manager import (
    prepare_summary_data,
    sanitize_for_xml,
    _format_page_range,
    STRUCTURE_PAGE_TYPE_ORDER,
    PAGE_TYPE_LABELS,
)

logger = setup_logger(__name__)


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
    citation_manager = CitationManager(polite_pool_email=config.CITATION_OPENALEX_EMAIL)
    data = prepare_summary_data(summary_results, citation_manager)
    filtered_results = data.filtered_results
    page_type_pages = data.page_type_pages

    lines: list[str] = []

    # === SECTION 1: Title ===
    lines.append(f"# Summary of {sanitize_for_xml(document_name)}")
    lines.append("")

    lines.append(
        f"*Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Content pages: {data.content_page_count} | Total pages: {len(filtered_results)}*"
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
    for page_item in data.page_render_items:
        lines.append(f"## {page_item.heading_text}")
        lines.append("")

        for point in page_item.bullet_points:
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
