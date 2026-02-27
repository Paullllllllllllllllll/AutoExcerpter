"""Processors package for AutoExcerpter.

This package provides file processing utilities:

- **pdf_processor**: PDF page extraction to images
- **file_manager**: Content filtering, XML sanitization, transcription output
- **docx_writer**: DOCX summary document creation with LaTeX equation rendering
- **markdown_writer**: Markdown summary document creation
- **log_manager**: JSON log file management for processing runs
- **citation_manager**: Citation deduplication and OpenAlex enrichment
"""

from processors.pdf_processor import (
    extract_pdf_pages_to_images,
    get_image_paths_from_folder,
)
from processors.docx_writer import create_docx_summary
from processors.markdown_writer import create_markdown_summary
from processors.file_manager import (
    write_transcription_to_text,
    filter_empty_pages,
    sanitize_for_xml,
)
from processors.log_manager import (
    initialize_log_file,
    append_to_log,
    finalize_log_file,
)
from processors.citation_manager import CitationManager

__all__ = [
    # PDF Processing
    "extract_pdf_pages_to_images",
    "get_image_paths_from_folder",
    # File Management
    "create_docx_summary",
    "create_markdown_summary",
    "write_transcription_to_text",
    "initialize_log_file",
    "append_to_log",
    "finalize_log_file",
    "filter_empty_pages",
    "sanitize_for_xml",
    # Citation Management
    "CitationManager",
]
