"""Item scanning utilities for discovering PDFs and image folders.

This module provides functions to scan directories and identify processable items
(PDF files and image folders) for the AutoExcerpter pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterable

from modules.constants import SUPPORTED_IMAGE_EXTENSIONS
from modules.logger import setup_logger
from modules.types import ItemSpec

logger = setup_logger(__name__)


# ============================================================================
# File Type Detection
# ============================================================================
def is_pdf_file(path: Path) -> bool:
    """Check if a path points to a PDF file."""
    return path.suffix.lower() == ".pdf"


def is_supported_image(path: Path) -> bool:
    """Check if a path points to a supported image file."""
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


# ============================================================================
# Item Scanning
# ============================================================================
def scan_input_path(path_to_scan: Path) -> list[ItemSpec]:
    """Gather items from a file or directory path."""
    logger.debug("Scanning input: %s", path_to_scan)
    collected: list[ItemSpec] = []

    if path_to_scan.is_file():
        if is_pdf_file(path_to_scan):
            collected.append(_build_pdf_item(path_to_scan))
        else:
            logger.warning(
                "Input path %s is not a PDF file. Skipping.",
                path_to_scan,
            )
    elif path_to_scan.is_dir():
        collected.extend(_collect_items_from_directory(path_to_scan))
    else:
        logger.warning(
            "Input path %s is not a PDF file or a directory. Skipping.",
            path_to_scan,
        )

    logger.debug("Found %s potential items from %s.", len(collected), path_to_scan)
    return collected


# ============================================================================
# Private Helper Functions
# ============================================================================
def _collect_items_from_directory(path_to_scan: Path) -> Iterable[ItemSpec]:
    """Recursively collect PDF files and image folders from a directory."""
    image_folders: dict[Path, list[Path]] = {}
    items: list[ItemSpec] = []

    for root, dirs, files in os.walk(path_to_scan):
        current_dir = Path(root)

        # Collect PDF files
        for file_name in files:
            file_path = current_dir / file_name
            if is_pdf_file(file_path):
                items.append(_build_pdf_item(file_path))

        # Collect images for image folder detection
        for file_name in files:
            file_path = current_dir / file_name
            if is_supported_image(file_path):
                image_folders.setdefault(current_dir, []).append(file_path)

        # Skip descending into directories that contain images
        dirs[:] = [name for name in dirs if (current_dir / name) not in image_folders]

    items.extend(_build_image_folder_items(image_folders))
    return items


def _build_pdf_item(pdf_path: Path) -> ItemSpec:
    """Create an ItemSpec for a PDF file."""
    return ItemSpec(kind="pdf", path=pdf_path)


def _build_image_folder_items(image_folders: dict[Path, list[Path]]) -> list[ItemSpec]:
    """Create ItemSpec objects for image folders."""
    image_items: list[ItemSpec] = []
    for folder_path, images in image_folders.items():
        if not images:
            continue
        sorted_images = sorted(images, key=lambda target: target.name)
        image_items.append(
            ItemSpec(
                kind="image_folder",
                path=folder_path,
                image_count=len(sorted_images),
            )
        )
    return image_items


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "scan_input_path",
    "is_pdf_file",
    "is_supported_image",
]
