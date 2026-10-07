"""Page images of AutoExcerpter's input items.

Public interface:

- ``PdfPayloadSource`` / ``FolderPayloadSource``: lazy payload sources for a
  PDF or an image folder, with the image settings of the transcription model.
- ``get_image_paths_from_folder(folder_path)``: the supported images of a
  folder in natural filename order.

Rendering, preprocessing, encoding and the ``PagePayload`` type live in
``common.images``; the recommended settings and their resolution in
``imaging.settings``.
"""

from autoexcerpter.imaging.payload import (
    FolderPayloadSource,
    PdfPayloadSource,
    get_image_paths_from_folder,
)

__all__ = [
    "PdfPayloadSource",
    "FolderPayloadSource",
    "get_image_paths_from_folder",
]
