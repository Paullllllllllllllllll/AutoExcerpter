"""Core package for AutoExcerpter.

This package provides the main transcription orchestration:

- **transcriber**: ItemTranscriber class for processing PDFs and image folders
"""

from core.transcriber import ItemTranscriber

__all__ = [
    "ItemTranscriber",
]
