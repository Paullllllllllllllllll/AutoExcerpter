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


def sanitize_for_xml(text: Optional[str]) -> str:
    """Return XML-safe text for DOCX output."""
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
    """Return the innermost summary payload dict, handling nested formats."""
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return {}

    nested_summary = summary.get("summary")
    if isinstance(nested_summary, dict):
        return nested_summary
    return summary


def _page_number_and_flags(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize page number information from summary payload."""
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
        if any(marker in marker_text for marker in [
            "[empty page",
            "no transcription possible",
            "empty page",
            "error",
        ]):
            return False

    if summary_data.get("contains_no_semantic_content", False):
        return False

    return True


def create_docx_summary(
    summary_results: List[Dict[str, Any]], output_path: Path, document_name: str
) -> None:
    """Create a compact DOCX summary document from structured summary results."""
    filtered_results = filter_empty_pages(summary_results)
    if len(filtered_results) < len(summary_results):
        logger.info(
            "Filtered out %s pages with no useful content",
            len(summary_results) - len(filtered_results),
        )

    document = Document()

    normal_style = document.styles["Normal"]
    normal_style.paragraph_format.space_before = Pt(0)
    normal_style.paragraph_format.space_after = Pt(4)

    for level in range(1, 4):
        heading_style = document.styles[f"Heading {level}"]
        heading_style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        heading_style.paragraph_format.space_after = Pt(4)

    title = document.add_heading(f"Summary of {sanitize_for_xml(document_name)}", 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    title.paragraph_format.space_after = Pt(8)

    metadata = "Processed: %s | Pages: %s" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        len(filtered_results),
    )
    meta_paragraph = document.add_paragraph(metadata)
    meta_paragraph.paragraph_format.space_after = Pt(8)

    for index, result in enumerate(filtered_results):
        summary_payload = _extract_summary_payload(result)
        page_info = _page_number_and_flags(summary_payload)
        page_number = page_info["page_number_integer"]
        bullet_points = summary_payload.get("bullet_points", [])
        references = summary_payload.get("references", [])

        page_heading = document.add_heading(f"Page {page_number}", 1)
        page_heading.paragraph_format.space_before = Pt(4 if index == 0 else 8)
        page_heading.paragraph_format.space_after = Pt(4)

        if bullet_points:
            for point in bullet_points:
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.left_indent = Pt(12)
                bullet_run = paragraph.add_run("• ")
                bullet_run.bold = True
                paragraph.add_run(sanitize_for_xml(point))
        else:
            no_points_paragraph = document.add_paragraph(
                "No bullet points available for this page."
            )
            no_points_paragraph.paragraph_format.space_after = Pt(4)

        if references:
            ref_heading = document.add_heading("References", 2)
            ref_heading.paragraph_format.space_before = Pt(6)
            ref_heading.paragraph_format.space_after = Pt(2)

            for reference in references:
                ref_paragraph = document.add_paragraph(sanitize_for_xml(reference))
                ref_paragraph.paragraph_format.space_before = Pt(0)
                ref_paragraph.paragraph_format.space_after = Pt(2)
                ref_paragraph.paragraph_format.left_indent = Pt(12)

        if index < len(filtered_results) - 1:
            separator = document.add_paragraph()
            separator.paragraph_format.space_before = Pt(4)
            separator.paragraph_format.space_after = Pt(4)
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
    """Create the per-item log file header."""

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
    """Append a JSON line to an existing log file."""

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to write to log file %s: %s", log_path, exc)
        return False


def filter_empty_pages(summary_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove entries without meaningful summary content."""

    filtered: List[Dict[str, Any]] = []
    for result in summary_results:
        summary_payload = _extract_summary_payload(result)
        if summary_payload and _is_meaningful_summary(summary_payload):
            filtered.append(result)
    return filtered