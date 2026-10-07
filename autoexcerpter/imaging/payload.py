"""Payload sources of one input item, on the shared page sources.

``PdfPayloadSource`` and ``FolderPayloadSource`` resolve the image settings of
the transcription model and hand them to ``common.images``. Both expose
``__len__``, ``image_name(index)``, ``build_payload(index)``,
``file_provenance()`` and ``close()``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from autoexcerpter.common.images import (
    IMAGE_EXTENSIONS,
    FolderPageSource,
    PdfPageSource,
    list_folder_images,
)
from autoexcerpter.imaging.settings import resolve_image_settings
from autoexcerpter.spec import ImageSpec

logger = logging.getLogger(__name__)


def get_image_paths_from_folder(folder_path: Path) -> list[Path]:
    """Return the supported images of *folder_path* in natural filename order."""
    logger.info(f"Scanning image folder: {folder_path.name}...")
    image_paths = list_folder_images(folder_path, IMAGE_EXTENSIONS)
    logger.info(f"Found {len(image_paths)} images in folder.")
    return image_paths


class PdfPayloadSource(PdfPageSource):
    """Render the pages of one PDF for a transcription model."""

    def __init__(
        self,
        pdf_path: Path,
        provider: str | None = "openai",
        model_name: str | None = "",
        images: ImageSpec | None = None,
    ) -> None:
        img_cfg, model_type, section = resolve_image_settings(
            provider, model_name, images
        )
        super().__init__(pdf_path, img_cfg, model_type, section=section)


class FolderPayloadSource(FolderPageSource):
    """Load the images of one folder for a transcription model."""

    def __init__(
        self,
        folder_path: Path,
        provider: str | None = "openai",
        model_name: str | None = "",
        images: ImageSpec | None = None,
    ) -> None:
        img_cfg, model_type, section = resolve_image_settings(
            provider, model_name, images
        )
        super().__init__(
            folder_path,
            img_cfg,
            model_type,
            image_paths=get_image_paths_from_folder(folder_path),
            section=section,
        )


__all__ = [
    "FolderPayloadSource",
    "PdfPayloadSource",
    "get_image_paths_from_folder",
]
