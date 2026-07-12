"""Image folder scanning utilities.

PDF page rendering moved to ``imaging.payload`` (streaming, in-memory);
this module retains the folder-scanning helper used by the image-folder
payload source.
"""

from __future__ import annotations

import re
from pathlib import Path

from config.constants import SUPPORTED_IMAGE_EXTENSIONS
from config.logger import setup_logger

logger = setup_logger(__name__)


def _natural_sort_key(name: str) -> tuple[str | int, ...]:
    """Sort key comparing digit runs numerically (page_2 before page_10).

    Plain lexicographic sorting misorders unpadded numeric filenames
    (page_1, page_10, page_11, ..., page_2), which would drive the .txt
    concatenation, log, and summary order. Splitting on digit runs keeps
    str/int positions aligned (even indices are always the non-digit parts).
    """
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    )


def get_image_paths_from_folder(folder_path: Path) -> list[Path]:
    """
    Get a sorted list of image paths from a folder.

    Args:
        folder_path: Path to folder containing images

    Returns:
        List of image paths, sorted by filename (natural sort)
    """
    logger.info(f"Scanning image folder: {folder_path.name}...")
    image_paths = sorted(
        [
            p
            for p in folder_path.glob("*")
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda p: _natural_sort_key(p.name),
    )
    logger.info(f"Found {len(image_paths)} images in folder.")
    return image_paths


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "get_image_paths_from_folder",
]
