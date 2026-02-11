"""Core package for AutoExcerpter.

This package provides the main transcription orchestration:

- **transcriber**: ItemTranscriber class for processing PDFs and image folders
- **resume**: Resume-aware processing utilities (ResumeChecker, ProcessingState)
"""

from core.resume import ProcessingState, ResumeChecker, ResumeResult
from core.transcriber import ItemTranscriber

__all__ = [
    "ItemTranscriber",
    "ProcessingState",
    "ResumeChecker",
    "ResumeResult",
]
