"""Final outputs of one item: the transcription file and the summary files.

The writers block (OpenAlex lookups, file I/O), so the item processor runs
:func:`write_outputs` in a worker thread and stops it through a
``threading.Event``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoexcerpter.pipeline.job import ItemJob
from autoexcerpter.pipeline.page_numbering import PageNumberProcessor
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.rendering import (
    create_docx_summary,
    create_markdown_summary,
    write_transcription_to_text,
)
from autoexcerpter.rendering.citations import Citation, enrich_if_enabled
from autoexcerpter.rendering.summary import build_render_context

logger = logging.getLogger(__name__)


@dataclass
class OutputStatus:
    """Whether the transcription and the summary files were written.

    *citations* are the citations the summaries were rendered with, None
    when no summary was rendered.
    """

    transcription_ok: bool = True
    summaries_ok: bool = True
    citations: list[Citation] | None = None


@dataclass(frozen=True)
class ItemResults:
    """What one item's run produced, for its final files.

    *transcriptions* must be in input order; *metadata_notes* are extra
    metadata lines for the transcription file.
    """

    transcriptions: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    elapsed_s: float
    metadata_notes: list[str]


def _stopped(stop: threading.Event | None, name: str, what: str) -> bool:
    """Return whether *stop* is set, logging the withheld output if so."""
    if stop is None or not stop.is_set():
        return False
    logger.warning(
        "Item %s stopped: withholding the %s (working logs retained).", name, what
    )
    return True


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
    Once *stop* is set, OpenAlex enrichment ends and no further file is
    written; the outputs not written count as failed.
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
        # The file does not exist, so it is neither advertised nor allowed to
        # count as complete; the working log is retained for a re-run.
        logger.error(
            "Item %s: the transcription file could not be written; marking the "
            "item failed (working log retained for a re-run).",
            name,
        )

    if job.summarize and results.summaries:
        status.summaries_ok, status.citations = _write_summaries(
            name=name,
            item_type_label=item_type_label,
            paths=paths,
            job=job,
            summary_results=results.summaries,
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
    summary_results: list[dict[str, Any]],
    written: list[Path],
    stop: threading.Event | None,
) -> tuple[bool, list[Citation] | None]:
    """Render the DOCX and Markdown summaries from one citation manager.

    Citations are collected, consolidated and OpenAlex-enriched once, and both
    writers render from the same instance. A failure while building it, or a
    stop during the enrichment, blocks both writers. Returns whether every
    writer succeeded and the citations.
    """
    adjusted = PageNumberProcessor().adjust_and_sort_page_numbers(summary_results)
    ok = True
    try:
        citation_manager, render_data = build_render_context(
            adjusted,
            position_label="pdf" if item_type_label == "PDF" else "image",
            openalex_enabled=job.openalex_enabled and job.openalex is not None,
            max_requests=job.openalex_max_requests,
            openalex=job.openalex,
            stop=stop,
        )
        enrich_if_enabled(citation_manager)
    except Exception as e:  # noqa: BLE001 - the summary files fail alone
        logger.error(f"Error building summary render context: {e}")
        return False, None
    if _stopped(stop, name, "summary files"):
        return False, None

    if job.output_docx:
        try:
            create_docx_summary(
                adjusted,
                paths.summary_docx,
                name,
                citation_manager=citation_manager,
                data=render_data,
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
                citation_manager=citation_manager,
                data=render_data,
            )
            written.append(paths.summary_md.resolve())
        except Exception as e:  # noqa: BLE001 - one optional output fails alone
            ok = False
            logger.error(f"Error creating Markdown summary: {e}")
    return ok, citation_manager.get_sorted_citations()


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
    detail_suffix = " and " + str(paths.summary_log) if job.summarize else ""
    logger.info(f"  Detailed logs: {paths.transcription_log}{detail_suffix}")


__all__ = ["ItemResults", "OutputStatus", "log_item_summary", "write_outputs"]
