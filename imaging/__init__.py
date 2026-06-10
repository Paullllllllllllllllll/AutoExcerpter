"""Image and PDF preprocessing for AutoExcerpter.

Public interface:

- ``PagePayload`` — immutable in-memory unit of work (base64 JPEG + provenance).
- ``PdfPayloadSource`` / ``FolderPayloadSource`` — lazy streaming payload
  producers for PDFs and image folders.
- ``get_image_paths_from_folder(folder_path)`` — return the sorted list of
  supported images in a folder.
- ``ImageProcessor`` — static in-memory image preprocessing core.

Model-type helpers live in ``imaging._provider`` and are package-private.
"""

from imaging.payload import FolderPayloadSource, PagePayload, PdfPayloadSource
from imaging.pdf import get_image_paths_from_folder
from imaging.preprocessing import ImageProcessor

__all__ = [
    "PagePayload",
    "PdfPayloadSource",
    "FolderPayloadSource",
    "get_image_paths_from_folder",
    "ImageProcessor",
]
