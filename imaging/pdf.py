"""Image folder scanning utilities.

PDF page rendering moved to ``imaging.payload`` (streaming, in-memory);
this module retains the folder-scanning helper used by the image-folder
payload source.
"""

from __future__ import annotations

from pathlib import Path

from config.constants import SUPPORTED_IMAGE_EXTENSIONS
from config.logger import setup_logger

logger = setup_logger(__name__)


def get_image_paths_from_folder(folder_path: Path) -> list[Path]:
    """
    Get a sorted list of image paths from a folder.

    Args:
        folder_path: Path to folder containing images

    Returns:
        List of image paths, sorted by filename
    """
    logger.info(f"Scanning image folder: {folder_path.name}...")
    image_paths = sorted(
        [
            p
            for p in folder_path.glob("*")
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name,
    )
    logger.info(f"Found {len(image_paths)} images in folder.")
    return image_paths


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "get_image_paths_from_folder",
]
