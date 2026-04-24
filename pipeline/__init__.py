"""Pipeline orchestration for AutoExcerpter.

Public interface:

- ``ItemTranscriber`` — processes a single PDF / image folder end-to-end.
- ``ResumeChecker``, ``ProcessingState``, ``ResumeResult`` — resume-from-interrupt state machine.
- ``scan_input_path``, ``is_pdf_file``, ``is_supported_image`` — input discovery.
- ``ItemSpec`` — descriptor for a PDF or image folder to process.

Implementation details (``pipeline.context``, ``pipeline.paths``,
``pipeline.page_numbering``, ``pipeline.log``) are importable directly
for testing but not re-exported here.
"""

from pipeline.resume import ProcessingState, ResumeChecker, ResumeResult
from pipeline.scanner import is_pdf_file, is_supported_image, scan_input_path
from pipeline.transcriber import ItemTranscriber
from pipeline.types import ItemSpec

__all__ = [
    "ItemTranscriber",
    "ResumeChecker",
    "ProcessingState",
    "ResumeResult",
    "scan_input_path",
    "is_pdf_file",
    "is_supported_image",
    "ItemSpec",
]
