"""Output rendering for AutoExcerpter.

Public interface, each returning a file's content:

- ``transcription_text``: the transcription file.
- ``docx_summary``: the `.docx` summary document, as bytes.
- ``markdown_summary``: the `.md` summary document.

``autoexcerpter.output.render`` feeds them from the store and writes the
files. Citation deduplication and OpenAlex enrichment live in the
``rendering.citations`` package (``CitationManager``); the writers call it
internally. ``rendering.page_numbering`` settles the page numbers of the
summaries, ``rendering.equations`` converts LaTeX formulas to Word equations
and ``rendering.locators`` formats page locators and Roman numerals.
"""

from autoexcerpter.rendering.docx import docx_summary
from autoexcerpter.rendering.markdown import markdown_summary
from autoexcerpter.rendering.transcription import transcription_text

__all__ = [
    "docx_summary",
    "markdown_summary",
    "transcription_text",
]
