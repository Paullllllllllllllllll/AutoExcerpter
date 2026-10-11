"""Process the items of a run: discovery, one item end to end, into the store.

The package exports the item processor and its job, the input scanner and
the item descriptor; the other modules are imported directly where needed.
"""

from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.job import ItemJob, PhaseModel
from autoexcerpter.pipeline.scanner import (
    is_pdf_file,
    is_supported_image,
    scan_input_path,
)
from autoexcerpter.pipeline.types import ItemSpec

__all__ = [
    "ItemJob",
    "ItemProcessor",
    "PhaseModel",
    "scan_input_path",
    "is_pdf_file",
    "is_supported_image",
    "ItemSpec",
]
