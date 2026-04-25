"""Output rendering for AutoExcerpter.

Public interface:

- ``write_transcription_to_text`` — write the full transcription `.txt` file.
- ``create_docx_summary`` — generate the `.docx` summary document.
- ``create_markdown_summary`` — generate the `.md` summary document.

Citation deduplication and OpenAlex enrichment live in
``rendering.citations`` (``CitationManager``); the writers call it
internally.
"""

from rendering.docx import create_docx_summary
from rendering.markdown import create_markdown_summary
from rendering.text import write_transcription_to_text

__all__ = [
    "create_docx_summary",
    "create_markdown_summary",
    "write_transcription_to_text",
]
