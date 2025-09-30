"""File management utilities for transcription and summary outputs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Pt

from modules import app_config as config
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Public API
__all__ = [
    "create_docx_summary",
    "write_transcription_to_text",
    "initialize_log_file",
    "append_to_log",
    "filter_empty_pages",
    "sanitize_for_xml",
]

# Constants for document formatting
TITLE_HEADING_LEVEL = 0
PAGE_HEADING_LEVEL = 1
REFERENCES_HEADING_LEVEL = 2
NORMAL_SPACE_BEFORE_PT = 0
NORMAL_SPACE_AFTER_PT = 4
HEADING1_SPACE_BEFORE_PT = 12
HEADING_OTHER_SPACE_BEFORE_PT = 8
HEADING_SPACE_AFTER_PT = 4
TITLE_SPACE_AFTER_PT = 8
METADATA_SPACE_AFTER_PT = 8
BULLET_INDENT_PT = 12
REF_INDENT_PT = 12
BULLET_SPACE_BEFORE_PT = 0
BULLET_SPACE_AFTER_PT = 2

# Constants for error markers
ERROR_MARKERS = ["[empty page", "no transcription possible", "empty page", "error"]


def sanitize_for_xml(text: Optional[str]) -> str:
    """
    Return XML-safe text for DOCX output.

    Args:
    text (Optional[str]): The text to be sanitized.

    Returns:
    str: The sanitized text.
    """
    if not text:
        return ""

    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    sanitized = sanitized.replace("&", "&amp;")
    sanitized = sanitized.replace("<", "&lt;")
    sanitized = sanitized.replace(">", "&gt;")
    sanitized = sanitized.replace('"', "&quot;")
    sanitized = sanitized.replace("'", "&apos;")
    return sanitized


def _extract_summary_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the innermost summary payload dict, handling nested formats.

    Args:
    result (Dict[str, Any]): The result dictionary.

    Returns:
    Dict[str, Any]: The innermost summary payload dictionary.
    """
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return {}

    nested_summary = summary.get("summary")
    if isinstance(nested_summary, dict):
        return nested_summary
    return summary


def _page_number_and_flags(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize page number information from summary payload.

    Args:
    summary_data (Dict[str, Any]): The summary data dictionary.

    Returns:
    Dict[str, Any]: A dictionary containing page number information.
    """
    page_number = summary_data.get("page_number", {})
    if isinstance(page_number, dict):
        return {
            "page_number_integer": page_number.get("page_number_integer", "?"),
            "contains_no_page_number": bool(page_number.get("contains_no_page_number", False)),
        }
    if isinstance(page_number, int):
        return {
            "page_number_integer": page_number,
            "contains_no_page_number": page_number == 0,
        }
    return {
        "page_number_integer": summary_data.get("page", "?"),
        "contains_no_page_number": False,
    }


def _is_meaningful_summary(summary_data: Dict[str, Any]) -> bool:
    """
    Check if a summary is meaningful.

    Args:
    summary_data (Dict[str, Any]): The summary data dictionary.

    Returns:
    bool: True if the summary is meaningful, False otherwise.
    """
    page_info = _page_number_and_flags(summary_data)
    if page_info["contains_no_page_number"]:
        page_int = page_info["page_number_integer"]
        if isinstance(page_int, int) and page_int == 0:
            return False

    bullet_points = summary_data.get("bullet_points", [])
    if not bullet_points:
        return False

    if len(bullet_points) == 1:
        marker_text = bullet_points[0].strip().lower()
        if any(marker in marker_text for marker in ERROR_MARKERS):
            return False

    if summary_data.get("contains_no_semantic_content", False):
        return False

    return True


def create_docx_summary(
    summary_results: List[Dict[str, Any]], output_path: Path, document_name: str
) -> None:
    """
    Create a compact DOCX summary document from structured summary results.

    Args:
    summary_results (List[Dict[str, Any]]): A list of summary results.
    output_path (Path): The output path for the DOCX file.
    document_name (str): The name of the document.
    """
    filtered_results = filter_empty_pages(summary_results)
    if len(filtered_results) < len(summary_results):
        logger.info(
            "Filtered out %s pages with no useful content",
            len(summary_results) - len(filtered_results),
        )

    document = Document()

    normal_style = document.styles["Normal"]
    normal_style.paragraph_format.space_before = Pt(NORMAL_SPACE_BEFORE_PT)
    normal_style.paragraph_format.space_after = Pt(NORMAL_SPACE_AFTER_PT)

    for level in range(1, 4):
        heading_style = document.styles[f"Heading {level}"]
        heading_style.paragraph_format.space_before = Pt(
            HEADING1_SPACE_BEFORE_PT if level == 1 else HEADING_OTHER_SPACE_BEFORE_PT
        )
        heading_style.paragraph_format.space_after = Pt(HEADING_SPACE_AFTER_PT)

    title = document.add_heading(f"Summary of {sanitize_for_xml(document_name)}", TITLE_HEADING_LEVEL)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    title.paragraph_format.space_after = Pt(TITLE_SPACE_AFTER_PT)

    metadata = "Processed: %s | Pages: %s" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        len(filtered_results),
    )
    meta_paragraph = document.add_paragraph(metadata)
    meta_paragraph.paragraph_format.space_after = Pt(METADATA_SPACE_AFTER_PT)

    for index, result in enumerate(filtered_results):
        summary_payload = _extract_summary_payload(result)
        page_info = _page_number_and_flags(summary_payload)
        page_number = page_info["page_number_integer"]
        bullet_points = summary_payload.get("bullet_points", [])
        references = summary_payload.get("references", [])

        page_heading = document.add_heading(f"Page {page_number}", PAGE_HEADING_LEVEL)
        page_heading.paragraph_format.space_before = Pt(HEADING_OTHER_SPACE_BEFORE_PT if index == 0 else HEADING_OTHER_SPACE_BEFORE_PT)
        page_heading.paragraph_format.space_after = Pt(HEADING_SPACE_AFTER_PT)

        if bullet_points:
            for point in bullet_points:
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.space_before = Pt(BULLET_SPACE_BEFORE_PT)
                paragraph.paragraph_format.space_after = Pt(BULLET_SPACE_AFTER_PT)
                paragraph.paragraph_format.left_indent = Pt(BULLET_INDENT_PT)
                bullet_run = paragraph.add_run("• ")
                bullet_run.bold = True
                paragraph.add_run(sanitize_for_xml(point))
        else:
            no_points_paragraph = document.add_paragraph(
                "No bullet points available for this page."
            )
            no_points_paragraph.paragraph_format.space_after = Pt(NORMAL_SPACE_AFTER_PT)

        if references:
            ref_heading = document.add_heading("References", REFERENCES_HEADING_LEVEL)
            ref_heading.paragraph_format.space_before = Pt(HEADING_OTHER_SPACE_BEFORE_PT)
            ref_heading.paragraph_format.space_after = Pt(HEADING_SPACE_AFTER_PT)

            for reference in references:
                ref_paragraph = document.add_paragraph(sanitize_for_xml(reference))
                ref_paragraph.paragraph_format.space_before = Pt(BULLET_SPACE_BEFORE_PT)
                ref_paragraph.paragraph_format.space_after = Pt(BULLET_SPACE_AFTER_PT)
                ref_paragraph.paragraph_format.left_indent = Pt(REF_INDENT_PT)

        if index < len(filtered_results) - 1:
            separator = document.add_paragraph()
            separator.paragraph_format.space_before = Pt(HEADING_OTHER_SPACE_BEFORE_PT)
            separator.paragraph_format.space_after = Pt(HEADING_OTHER_SPACE_BEFORE_PT)
            separator.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    document.save(output_path)
    logger.info("Compact summary DOCX file saved: %s", output_path)


def write_transcription_to_text(
    transcription_results: List[Dict[str, Any]],
    output_path: Path,
    document_name: str,
    item_type: str,
    total_elapsed_time: float,
    source_path: Path,
) -> bool:
    """
    Persist transcription output as a text file alongside basic metadata.

    Args:
    transcription_results (List[Dict[str, Any]]): A list of transcription results.
    output_path (Path): The output path for the text file.
    document_name (str): The name of the document.
    item_type (str): The type of the item.
    total_elapsed_time (float): The total elapsed time.
    source_path (Path): The source path.

    Returns:
    bool: True if the transcription was written successfully, False otherwise.
    """
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
            file_handle.write(f"# Total images processed: {len(transcription_results)}\n")
            file_handle.write(f"# Successfully transcribed: {successes}\n")
            file_handle.write(f"# Failed items: {failures}\n")
            file_handle.write(
                f"# Total processing time for this item: {elapsed_str}\n\n---\n\n"
            )

            for index, result in enumerate(transcription_results):
                file_handle.write(
                    result.get("transcription", "[ERROR] Transcription data missing")
                )
                if index < len(transcription_results) - 1:
                    file_handle.write("\n\n---\n\n")

        logger.info("Transcription text file saved: %s", output_path)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error writing transcription to text file %s: %s", output_path, exc)
        return False


def initialize_log_file(
    log_path: Path,
    item_name: str,
    input_path: str,
    input_type: str,
    total_images: int,
    model_name: str,
    extraction_dpi: Optional[int] = None,
) -> bool:
    """
    Create the per-item log file header.

    Args:
    log_path (Path): The log file path.
    item_name (str): The name of the item.
    input_path (str): The input path.
    input_type (str): The input type.
    total_images (int): The total number of images.
    model_name (str): The name of the model.
    extraction_dpi (Optional[int]): The extraction DPI. Defaults to None.

    Returns:
    bool: True if the log file was initialized successfully, False otherwise.
    """
    configuration = {
        "concurrent_requests": config.CONCURRENT_REQUESTS,
        "api_timeout_seconds": config.API_TIMEOUT,
        "model_name": model_name,
        "extraction_dpi": extraction_dpi if extraction_dpi is not None else "N/A",
        "openai_flex_processing": config.OPENAI_USE_FLEX if model_name == config.OPENAI_MODEL else "N/A",
    }

    payload = {
        "input_item_name": item_name,
        "input_item_path": input_path,
        "input_type": input_type,
        "processing_start_time": datetime.now().isoformat(),
        "total_images": total_images,
        "configuration": configuration,
    }

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            json.dump(payload, log_file)
            log_file.write("\n")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to initialize log file %s: %s", log_path, exc)
        return False


def append_to_log(log_path: Path, entry: Dict[str, Any]) -> bool:
    """
    Append a JSON line to an existing log file.

    Args:
    log_path (Path): The log file path.
    entry (Dict[str, Any]): The log entry.

    Returns:
    bool: True if the log entry was appended successfully, False otherwise.
    """
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to write to log file %s: %s", log_path, exc)
        return False


def filter_empty_pages(summary_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove entries without meaningful summary content.

    Args:
    summary_results (List[Dict[str, Any]]): A list of summary results.

    Returns:
    List[Dict[str, Any]]: A list of filtered summary results.
    """
    filtered: List[Dict[str, Any]] = []
    for result in summary_results:
        summary_payload = _extract_summary_payload(result)
        if summary_payload and _is_meaningful_summary(summary_payload):
            filtered.append(result)
    return filtered