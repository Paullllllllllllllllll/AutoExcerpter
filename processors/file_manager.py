"""File management utilities for transcription and summary outputs.

This module handles all file I/O operations for the AutoExcerpter application:

1. **DOCX Summary Creation**: 
   - Generates formatted Word documents with summaries
   - Manages citation deduplication and consolidation
   - Adds hyperlinks to citations with metadata
   - Applies consistent formatting (headings, bullets, spacing)

2. **Transcription Text Output**:
   - Writes raw transcriptions to text files
   - Includes metadata (timestamps, statistics, source info)

3. **JSON Logging**:
   - Initializes structured log files for API results
   - Appends entries during processing
   - Finalizes logs with proper JSON formatting

4. **Content Filtering**:
   - Filters out empty/error pages from summaries
   - Validates semantic content

5. **XML Safety**:
   - Sanitizes text for safe DOCX/XML output
   - Removes invalid control characters

All functions handle errors gracefully and provide detailed logging.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Pt
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from latex2mathml.converter import convert as latex_to_mathml
import mathml2omml

# ============================================================================
# Math Conversion Constants
# ============================================================================
MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"

from modules import app_config as config
from modules.logger import setup_logger
from processors.citation_manager import CitationManager

logger = setup_logger(__name__)

# Public API
__all__ = [
    "create_docx_summary",
    "write_transcription_to_text",
    "initialize_log_file",
    "append_to_log",
    "finalize_log_file",
    "filter_empty_pages",
    "sanitize_for_xml",
]

# ============================================================================
# Document Formatting Constants
# ============================================================================
TITLE_HEADING_LEVEL = 0
PAGE_HEADING_LEVEL = 1
REFERENCES_HEADING_LEVEL = 2
TITLE_SPACE_AFTER_PT = 6
PAGE_HEADING_SPACE_BEFORE_PT = 6
PAGE_HEADING_SPACE_AFTER_PT = 3
REF_HEADING_SPACE_BEFORE_PT = 4
BULLET_SPACE_AFTER_PT = 2
REF_INDENT_PT = 18
BULLET_INDENT_PT = 18

# ============================================================================
# Error Detection Constants
# ============================================================================
ERROR_MARKERS = ["[empty page", "no transcription possible", "empty page", "error"]


# ============================================================================
# Text Sanitization Functions
# ============================================================================
def sanitize_for_xml(text: Optional[str]) -> str:
    """
    Return XML-safe text for DOCX output by removing control characters.
    
    Note: python-docx handles XML entity encoding automatically (e.g., &, <, >, ', ")
    so we only need to remove invalid control characters.

    Args:
    text (Optional[str]): The text to be sanitized.

    Returns:
    str: The sanitized text with control characters removed.
    """
    if not text:
        return ""

    # Remove control characters (except tab \x09, newline \x0A, and carriage return \x0D)
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return sanitized


def parse_latex_in_text(text: str) -> List[Tuple[str, str]]:
    """
    Parse text and split it into segments of regular text and LaTeX formulas.
    
    Returns a list of tuples: (content, type) where type is either 'text' or 'latex'.
    
    Args:
        text: The text to parse for LaTeX formulas.
    
    Returns:
        List of tuples containing (content, type).
    """
    segments = []
    pattern = r'\$\$(.+?)\$\$'
    last_end = 0
    
    for match in re.finditer(pattern, text):
        # Add text before the formula
        if match.start() > last_end:
            segments.append((text[last_end:match.start()], 'text'))
        
        # Add the LaTeX formula (without the $$ delimiters)
        segments.append((match.group(1), 'latex'))
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        segments.append((text[last_end:], 'text'))
    
    return segments if segments else [(text, 'text')]


def add_math_to_paragraph(paragraph, latex_code: str) -> None:
    """
    Add a mathematical formula to a paragraph using OMML so Word renders it
    as a native equation.
    
    Args:
        paragraph: The paragraph object to add the formula to.
        latex_code: The LaTeX code for the formula.
    """
    try:
        # Convert LaTeX -> MathML -> OMML
        mathml = latex_to_mathml(latex_code)
        omml_markup = mathml2omml.convert(mathml)

        # Ensure the OMML output has the proper namespace declaration
        if "xmlns:m" not in omml_markup.split("\n", 1)[0]:
            omml_markup = omml_markup.replace(
                "<m:oMath",
                f'<m:oMath xmlns:m="{MATH_NAMESPACE}"',
                1,
            )

        # Wrap in oMathPara for compatibility with Word paragraphs
        if not omml_markup.strip().startswith("<m:oMath"):
            omml_markup = f'<m:oMath xmlns:m="{MATH_NAMESPACE}">{omml_markup}</m:oMath>'

        omml_para = (
            f'<m:oMathPara xmlns:m="{MATH_NAMESPACE}">{omml_markup}</m:oMathPara>'
        )
        omml_element = parse_xml(omml_para)

        # Append native equation to the paragraph
        paragraph._p.append(omml_element)
    except Exception as exc:
        # If conversion fails, add as formatted text
        logger.warning("Failed to convert LaTeX to MathML: %s. Displaying as text.", exc)
        run = paragraph.add_run(f" {latex_code} ")
        run.font.name = 'Cambria Math'
        run.italic = True


def add_formatted_text_to_paragraph(paragraph, text: str) -> None:
    """
    Add text to a paragraph, parsing and rendering LaTeX formulas.
    
    Args:
        paragraph: The paragraph object to add text to.
        text: The text that may contain LaTeX formulas in $$ ... $$ format.
    """
    segments = parse_latex_in_text(text)
    
    for content, segment_type in segments:
        if segment_type == 'latex':
            add_math_to_paragraph(paragraph, content)
        else:
            paragraph.add_run(sanitize_for_xml(content))


def add_hyperlink(paragraph, url: str, text: str) -> None:
    """
    Add a hyperlink to a paragraph in a DOCX document.
    
    Args:
        paragraph: The paragraph object to add the hyperlink to.
        url: The URL for the hyperlink.
        text: The display text for the hyperlink.
    """
    # This gets access to the document.xml.rels file and gets a new relation id value
    part = paragraph.part
    r_id = part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)

    # Create the w:hyperlink tag and add needed values
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)

    # Create a new run object (a wrapper over a text element)
    new_run = OxmlElement('w:r')
    
    # Set the run's text
    rPr = OxmlElement('w:rPr')
    
    # Add run properties for hyperlink styling
    rStyle = OxmlElement('w:rStyle')
    rStyle.set(qn('w:val'), 'Hyperlink')
    rPr.append(rStyle)
    new_run.append(rPr)
    
    # Add the actual text
    new_run.text = text
    hyperlink.append(new_run)

    # Add the hyperlink to the paragraph
    paragraph._p.append(hyperlink)


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
        summary_data: The summary data dictionary.

    Returns:
        Dictionary containing page number information.
    """
    page_number = summary_data.get("page_number", {})
    
    # Handle dict format (preferred)
    if isinstance(page_number, dict):
        return {
            "page_number_integer": page_number.get("page_number_integer", "?"),
            "contains_no_page_number": bool(page_number.get("contains_no_page_number", False)),
        }
    
    # Handle int format (legacy)
    if isinstance(page_number, int):
        return {
            "page_number_integer": page_number,
            "contains_no_page_number": page_number == 0,
        }
    
    # Fallback to page field
    return {
        "page_number_integer": summary_data.get("page", "?"),
        "contains_no_page_number": False,
    }


def _is_meaningful_summary(summary_data: Dict[str, Any]) -> bool:
    """
    Check if a summary is meaningful (not empty, error, or unnumbered).

    Args:
        summary_data: The summary data dictionary.

    Returns:
        True if the summary is meaningful, False otherwise.
    """
    # Check for unnumbered pages (page 0)
    page_info = _page_number_and_flags(summary_data)
    if page_info["contains_no_page_number"]:
        page_int = page_info["page_number_integer"]
        if isinstance(page_int, int) and page_int == 0:
            return False

    # Check for empty or error bullet points
    bullet_points = summary_data.get("bullet_points", [])
    if not bullet_points:
        return False

    if len(bullet_points) == 1:
        marker_text = bullet_points[0].strip().lower()
        if any(marker in marker_text for marker in ERROR_MARKERS):
            return False

    # Check semantic content flag
    if summary_data.get("contains_no_semantic_content", False):
        return False

    return True


def create_docx_summary(
    summary_results: List[Dict[str, Any]], output_path: Path, document_name: str
) -> None:
    """
    Create a compact DOCX summary document from structured summary results.
    Citations are collected, deduplicated, and displayed at the end with page tracking.

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

    # Initialize citation manager for the document
    citation_manager = CitationManager(polite_pool_email=config.CITATION_OPENALEX_EMAIL)
    
    document = Document()

    # Configure default Normal style for compact spacing
    normal_style = document.styles["Normal"]
    normal_style.paragraph_format.space_before = Pt(0)
    normal_style.paragraph_format.space_after = Pt(BULLET_SPACE_AFTER_PT)

    # Title section
    title = document.add_heading(f"Summary of {sanitize_for_xml(document_name)}", TITLE_HEADING_LEVEL)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    title.paragraph_format.space_after = Pt(TITLE_SPACE_AFTER_PT)

    # Metadata line
    metadata = "Processed: %s | Pages: %s" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        len(filtered_results),
    )
    meta_paragraph = document.add_paragraph(metadata)
    meta_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    meta_paragraph.paragraph_format.space_after = Pt(TITLE_SPACE_AFTER_PT)

    # Process each page
    for result in filtered_results:
        summary_payload = _extract_summary_payload(result)
        page_info = _page_number_and_flags(summary_payload)
        page_number = page_info["page_number_integer"]
        bullet_points = summary_payload.get("bullet_points", [])
        references = summary_payload.get("references", [])

        # Collect citations for consolidated section
        if references:
            citation_manager.add_citations(references, page_number)

        # Page heading
        page_heading = document.add_heading(f"Page {page_number}", PAGE_HEADING_LEVEL)
        page_heading.paragraph_format.space_before = Pt(PAGE_HEADING_SPACE_BEFORE_PT)
        page_heading.paragraph_format.space_after = Pt(PAGE_HEADING_SPACE_AFTER_PT)

        # Bullet points - use Word's native bullet list style
        if bullet_points:
            for point in bullet_points:
                paragraph = document.add_paragraph(style='List Bullet')
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(BULLET_SPACE_AFTER_PT)
                # Add text with LaTeX formula support
                add_formatted_text_to_paragraph(paragraph, point)
        else:
            no_points = document.add_paragraph("No bullet points available for this page.")
            no_points.paragraph_format.left_indent = Pt(BULLET_INDENT_PT)

        # NOTE: Per-page references section removed - citations now consolidated at end

    # Add consolidated references section at the end of the document
    if citation_manager.citations:
        logger.info("Processing %d unique citations for consolidated references section", 
                   len(citation_manager.citations))
        
        # Enrich citations with metadata from OpenAlex API
        citation_manager.enrich_with_metadata(max_requests=config.CITATION_MAX_API_REQUESTS)
        
        # Add references heading
        document.add_page_break()
        ref_main_heading = document.add_heading("Consolidated References", TITLE_HEADING_LEVEL)
        ref_main_heading.paragraph_format.space_before = Pt(PAGE_HEADING_SPACE_BEFORE_PT)
        ref_main_heading.paragraph_format.space_after = Pt(PAGE_HEADING_SPACE_AFTER_PT)
        
        # Add note about the references
        note_text = ("The following references were extracted from the document and consolidated. "
                    "Duplicate citations have been merged, showing all pages where each citation appears. "
                    "Where available, hyperlinks provide access to extended metadata via OpenAlex.")
        note_paragraph = document.add_paragraph(note_text)
        note_paragraph.paragraph_format.space_after = Pt(PAGE_HEADING_SPACE_AFTER_PT)
        
        # Get sorted citations with page information
        citations_with_pages = citation_manager.get_citations_with_pages()
        
        for idx, (citation, page_range_str) in enumerate(citations_with_pages, start=1):
            # Create citation paragraph
            ref_paragraph = document.add_paragraph()
            ref_paragraph.paragraph_format.space_before = Pt(0)
            ref_paragraph.paragraph_format.space_after = Pt(BULLET_SPACE_AFTER_PT)
            ref_paragraph.paragraph_format.left_indent = Pt(REF_INDENT_PT)
            ref_paragraph.paragraph_format.first_line_indent = Pt(-REF_INDENT_PT)
            
            # Add citation number
            num_run = ref_paragraph.add_run(f"[{idx}] ")
            num_run.bold = True
            
            # Add citation text
            citation_text = sanitize_for_xml(citation.raw_text)
            
            # If we have metadata with a URL, add as hyperlink
            if citation.url:
                # Add citation text as hyperlink
                add_hyperlink(ref_paragraph, citation.url, citation_text)
            else:
                ref_paragraph.add_run(citation_text)
            
            # Add page information
            if page_range_str:
                page_run = ref_paragraph.add_run(f" ({page_range_str})")
                page_run.italic = True
            
            # Add metadata note if available
            if citation.metadata:
                meta_info_parts = []
                if citation.doi:
                    meta_info_parts.append(f"DOI: {citation.doi}")
                if citation.metadata.get('publication_year'):
                    meta_info_parts.append(f"Year: {citation.metadata['publication_year']}")
                
                if meta_info_parts:
                    meta_run = ref_paragraph.add_run(f" [{', '.join(meta_info_parts)}]")
                    meta_run.font.size = Pt(9)
                    meta_run.italic = True

    document.save(output_path)
    logger.info("Summary document saved to %s", output_path)


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
    concurrency_limit: Optional[int] = None,
) -> bool:
    """
    Create the per-item log file header as the start of a JSON array.

    Args:
    log_path (Path): The log file path.
    item_name (str): The name of the item.
    input_path (str): The input path.
    input_type (str): The input type.
    total_images (int): The total number of images.
    model_name (str): The name of the model.
    extraction_dpi (Optional[int]): The extraction DPI. Defaults to None.
    concurrency_limit (Optional[int]): The actual concurrency limit being used. Defaults to None.

    Returns:
    bool: True if the log file was initialized successfully, False otherwise.
    """
    configuration = {
        "concurrent_requests": concurrency_limit if concurrency_limit is not None else config.CONCURRENT_REQUESTS,
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
            log_file.write("[\n")  # Start JSON array
            json.dump(payload, log_file)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to initialize log file %s: %s", log_path, exc)
        return False


def append_to_log(log_path: Path, entry: Dict[str, Any]) -> bool:
    """
    Append a JSON entry to the log file array (comma-separated).

    Args:
    log_path (Path): The log file path.
    entry (Dict[str, Any]): The log entry.

    Returns:
    bool: True if the log entry was appended successfully, False otherwise.
    """
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(",\n")  # Add comma separator
            json.dump(entry, log_file)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to write to log file %s: %s", log_path, exc)
        return False


def finalize_log_file(log_path: Path) -> bool:
    """
    Finalize the log file by closing the JSON array.

    Args:
    log_path (Path): The log file path.

    Returns:
    bool: True if the log file was finalized successfully, False otherwise.
    """
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n]")  # Close JSON array
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to finalize log file %s: %s", log_path, exc)
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