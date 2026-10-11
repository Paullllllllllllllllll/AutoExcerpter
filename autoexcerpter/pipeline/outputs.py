"""Final outputs of one item: its citations, its formats and the closing log.

The citations are consolidated and OpenAlex-enriched once per item by
:func:`collect_citations`; the item stores them, and
:func:`~autoexcerpter.output.render.render_item` writes the files from the
store. The enrichment blocks on OpenAlex lookups, so the item processor runs
it in a worker thread and stops it through a ``threading.Event``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.output.render import (
    SUMMARY_DOCX,
    SUMMARY_MD,
    TRANSCRIPTION_MD,
    TRANSCRIPTION_TXT,
)
from autoexcerpter.output.store import CitationRow
from autoexcerpter.pipeline.job import ItemJob
from autoexcerpter.rendering.citations import enrich_if_enabled
from autoexcerpter.rendering.page_numbering import PageNumberProcessor
from autoexcerpter.rendering.summary import build_render_context

logger = logging.getLogger(__name__)


@dataclass
class OutputStatus:
    """Whether the transcription and the summary files hold their rendering."""

    transcription_ok: bool = True
    summaries_ok: bool = True


def _position_label(item_type_label: str) -> str:
    return "pdf" if item_type_label == "PDF" else "image"


def render_formats(job: ItemJob) -> list[str]:
    """Return the formats an item of *job* is rendered in."""
    formats = [
        TRANSCRIPTION_TXT if job.transcription_format == "txt" else TRANSCRIPTION_MD
    ]
    if job.summarize and job.output_docx:
        formats.append(SUMMARY_DOCX)
    if job.summarize and job.output_markdown:
        formats.append(SUMMARY_MD)
    return formats


def collect_citations(
    summaries: list[dict[str, Any]],
    *,
    item_type_label: str,
    job: ItemJob,
    stop: threading.Event | None = None,
) -> list[CitationRow] | None:
    """Return the item's citations, consolidated and OpenAlex-enriched.

    None when *stop* was set during the enrichment.
    """
    adjusted = PageNumberProcessor().adjust_and_sort_page_numbers(summaries)
    manager, _ = build_render_context(
        adjusted,
        position_label=_position_label(item_type_label),
        openalex_enabled=job.openalex_enabled and job.openalex is not None,
        max_requests=job.openalex_max_requests,
        openalex=job.openalex,
        stop=stop,
    )
    enrich_if_enabled(manager)
    if stop is not None and stop.is_set():
        return None
    return [
        CitationRow(
            text=citation.raw_text,
            pages=pages,
            partial=citation.partial,
            doi=citation.doi,
            url=citation.url,
            metadata=citation.metadata,
        )
        for citation, pages in manager.get_citations_with_pages()
    ]


def log_item_summary(
    *,
    name: str,
    paths: ItemPaths,
    job: ItemJob,
    pages_logged: int,
    pages_ok: int,
    pages_failed: int,
    elapsed_s: float,
    avg_api_s: float | None,
) -> None:
    """Log the closing statistics of one item."""
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_s))
    logger.info(f"PROCESSING COMPLETE for item: {name}")
    logger.info(f"  Total images for this item: {pages_logged}")
    logger.info(f"  Successfully transcribed: {pages_ok}")
    logger.info(f"  Failed items: {pages_failed}")
    logger.info(f"  Total time for this item: {elapsed_str}")
    if avg_api_s is not None:
        logger.info(
            f"  Average API processing time per successful image: {avg_api_s:.2f}s"
        )
    if elapsed_s > 0 and pages_ok > 0:
        throughput_iph = (pages_ok / elapsed_s) * 3600
        logger.info(
            f"  Overall throughput for this item: "
            f"{throughput_iph:.1f} successful images/hour"
        )
    logger.info(f"  Final transcription output: {paths.transcription}")
    if job.summarize:
        summary_outputs = []
        if job.output_docx:
            summary_outputs.append(str(paths.summary_docx))
        if job.output_markdown:
            summary_outputs.append(str(paths.summary_md))
        if summary_outputs:
            logger.info(f"  Final summary outputs: {', '.join(summary_outputs)}")


__all__ = [
    "OutputStatus",
    "collect_citations",
    "log_item_summary",
    "render_formats",
]
