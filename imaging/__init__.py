"""Image and PDF preprocessing for AutoExcerpter.

Public interface:

- ``extract_pdf_pages_to_images(pdf_path, output_dir, provider, model_name)`` — render a PDF to JPEG pages with provider-specific preprocessing.
- ``get_image_paths_from_folder(folder_path)`` — return the sorted list of supported images in a folder.
- ``ImageProcessor`` — in-memory image preprocessing engine used by the LLM transcription path.

Model-type helpers live in ``imaging._provider`` and are package-private.
"""

from imaging.pdf import extract_pdf_pages_to_images, get_image_paths_from_folder
from imaging.preprocessing import ImageProcessor

__all__ = [
    "extract_pdf_pages_to_images",
    "get_image_paths_from_folder",
    "ImageProcessor",
]
