"""Output rendering for AutoExcerpter.

Public interface:

- ``write_transcription_to_text``: write the transcription file.
- ``create_docx_summary``: generate the `.docx` summary document.
- ``create_markdown_summary``: generate the `.md` summary document.

Citation deduplication and OpenAlex enrichment live in the
``rendering.citations`` package (``CitationManager``); the writers call it
internally. ``rendering.equations`` converts LaTeX formulas to Word equations,
``rendering.locators`` formats page locators and Roman numerals, and
``rendering.sqlite`` defines the tables of ``autoexcerpter.sqlite``.
"""

from autoexcerpter.rendering.docx import create_docx_summary
from autoexcerpter.rendering.markdown import create_markdown_summary
from autoexcerpter.rendering.transcription import write_transcription_to_text

__all__ = [
    "create_docx_summary",
    "create_markdown_summary",
    "write_transcription_to_text",
]
