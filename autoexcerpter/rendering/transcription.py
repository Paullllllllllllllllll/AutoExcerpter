"""The transcription file of an item: a metadata header and the pages in order."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from autoexcerpter.rendering.page_tags import printed_page_numbers


def page_break_marker(
    result: dict[str, Any], item_type: str, fallback_index: int
) -> str:
    """Return the ``<page_break .../>`` line for a page without a printed number.

    The attribute is ``pdf`` only when the page has a real PDF page index;
    image-folder pages (and pages whose PDF index is unknown) use ``image``.
    Both count from 1 in input order.
    """
    index = result.get("original_input_order_index")
    if not isinstance(index, int) or isinstance(index, bool):
        index = fallback_index
    has_pdf_index = item_type == "PDF" and result.get("page_index") is not None
    attribute = "pdf" if has_pdf_index else "image"
    return f'<page_break {attribute}="{index + 1}"/>\n'


def transcription_text(
    transcription_results: list[dict[str, Any]],
    document_name: str,
    item_type: str,
    source_path: Path | str,
    processed: datetime,
    metadata_notes: list[str] | None = None,
) -> str:
    """Return the transcription file: the header, then the pages in order.

    *processed* is the time the header records; *metadata_notes* are extra
    provenance lines of the header (a model change on resume, for example).
    The text ends with exactly one newline.
    """
    successes = sum(1 for result in transcription_results if "error" not in result)
    failures = len(transcription_results) - successes
    parts = [
        f"# Transcription of: {document_name}\n",
        f"# Source Path: {source_path}\n",
        f"# Type: {item_type}\n",
        f"# Processed on: {processed.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"# Total images processed: {len(transcription_results)}\n",
        f"# Successfully transcribed: {successes}\n",
        f"# Failed items: {failures}\n",
        *(f"# Note: {note}\n" for note in metadata_notes or []),
        "\n",
    ]
    for index, result in enumerate(transcription_results):
        transcription = result.get(
            "transcription", "[ERROR] Transcription data missing"
        )
        if not isinstance(transcription, str):
            transcription = str(transcription)
        # A page without a printed number gets a positional marker so it
        # stays locatable; numbered pages are written unchanged.
        if not printed_page_numbers(transcription):
            parts.append(page_break_marker(result, item_type, index))
        parts.append(transcription)
        if index < len(transcription_results) - 1:
            parts.append("\n")
    return "".join(parts).rstrip("\n") + "\n"
