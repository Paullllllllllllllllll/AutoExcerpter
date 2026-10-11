"""Final outputs of one item: its citations, the transcription and summary files.

The citations are consolidated and OpenAlex-enriched once per item by
:func:`collect_citations`; the item stores them, and :func:`write_outputs`
writes the files from what the store holds. Both block (OpenAlex lookups,
file I/O), so the item processor runs them in a worker thread and stops
them through a ``threading.Event``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoexcerpter.output.store import CitationRow
from autoexcerpter.pipeline.job import ItemJob
from autoexcerpter.pipeline.page_numbering import PageNumberProcessor
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.rendering import (
    create_docx_summary,
    create_markdown_summary,
    write_transcription_to_text,
)
from autoexcerpter.rendering.citations import CitationManager, enrich_if_enabled
from autoexcerpter.rendering.summary import (
    StoredCitations,
    build_render_context,
    prepare_summary_data,
)

logger = logging.getLogger(__name__)


@dataclass
class OutputStatus:
    """Whether the transcription and the summary files were written."""

    transcription_ok: bool = True
    summaries_ok: bool = True


@dataclass(frozen=True)
class ItemResults:
    """What the store holds for one item, for its final files.

    *transcriptions* and *summaries* are in page order; *citations* in
    rendered order; *metadata_notes* are extra metadata lines for the
    transcription file.
    """

    transcriptions: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    citations: Sequence[CitationRow]
    elapsed_s: float
    metadata_notes: list[str]


def _stopped(stop: threading.Event | None, name: str, what: str) -> bool:
    """Return whether *stop* is set, logging the withheld output if so."""
    if stop is None or not stop.is_set():
        return False
    logger.warning("Item %s stopped: withholding the %s.", name, what)
    return True


def _position_label(item_type_label: str) -> str:
    return "pdf" if item_type_label == "PDF" else "image"


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


def write_outputs(
    *,
    name: str,
    item_type_label: str,
    input_path: Path,
    paths: ItemPaths,
    job: ItemJob,
    results: ItemResults,
    written: list[Path],
    stop: threading.Event | None = None,
) -> OutputStatus:
    """Write the item's final files and append each written path to *written*.

    A writer that fails marks the status without stopping the other writers.
    Once *stop* is set, no further file is written; the outputs not written
    count as failed.
    """
    if _stopped(stop, name, "final outputs"):
        return OutputStatus(transcription_ok=False, summaries_ok=False)
    status = OutputStatus()
    logger.info(f"Writing final transcription output to: {paths.transcription}")
    status.transcription_ok = write_transcription_to_text(
        results.transcriptions,
        paths.transcription,
        name,
        item_type_label,
        results.elapsed_s,
        input_path,
        metadata_notes=results.metadata_notes or None,
    )
    if status.transcription_ok:
        written.append(paths.transcription.resolve())
    else:
        logger.error(
            "Item %s: the transcription file could not be written; marking the "
            "item failed (its pages stay in the store).",
            name,
        )

    if job.summarize and results.summaries:
        status.summaries_ok = _write_summaries(
            name=name,
            item_type_label=item_type_label,
            paths=paths,
            job=job,
            results=results,
            written=written,
            stop=stop,
        )
    return status


def _write_summaries(
    *,
    name: str,
    item_type_label: str,
    paths: ItemPaths,
    job: ItemJob,
    results: ItemResults,
    written: list[Path],
    stop: threading.Event | None,
) -> bool:
    """Render the DOCX and Markdown summaries with the stored citations.

    Returns whether every writer succeeded.
    """
    adjusted = PageNumberProcessor().adjust_and_sort_page_numbers(results.summaries)
    data = prepare_summary_data(
        adjusted,
        CitationManager(openalex_enabled=False),
        position_label=_position_label(item_type_label),
    )
    citations = StoredCitations(results.citations)
    ok = True
    if job.output_docx and _stopped(stop, name, "DOCX summary"):
        ok = False
    elif job.output_docx:
        try:
            create_docx_summary(
                adjusted,
                paths.summary_docx,
                name,
                citation_manager=citations,
                data=data,
            )
            written.append(paths.summary_docx.resolve())
        except Exception as e:  # noqa: BLE001 - one optional output fails alone
            ok = False
            logger.error(f"Error creating DOCX summary: {e}")

    if job.output_markdown and _stopped(stop, name, "Markdown summary"):
        ok = False
    elif job.output_markdown:
        try:
            create_markdown_summary(
                adjusted,
                paths.summary_md,
                name,
                citation_manager=citations,
                data=data,
            )
            written.append(paths.summary_md.resolve())
        except Exception as e:  # noqa: BLE001 - one optional output fails alone
            ok = False
            logger.error(f"Error creating Markdown summary: {e}")
    return ok


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
    "ItemResults",
    "OutputStatus",
    "collect_citations",
    "log_item_summary",
    "write_outputs",
]
