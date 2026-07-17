"""Item scanning utilities for discovering PDFs and image folders.

This module provides functions to scan directories and identify processable items
(PDF files and image folders) for the AutoExcerpter pipeline.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from config.constants import SUPPORTED_IMAGE_EXTENSIONS
from config.logger import setup_logger
from pipeline.types import ItemSpec

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

    for root, _dirs, files in os.walk(path_to_scan):
        current_dir = Path(root)

        # Collect PDF files and images in a single pass (a file is at most one
        # of the two)
        for file_name in files:
            file_path = current_dir / file_name
            if is_pdf_file(file_path):
                items.append(_build_pdf_item(file_path))
            elif is_supported_image(file_path):
                image_folders.setdefault(current_dir, []).append(file_path)

    # Descend the whole tree so nested PDFs and image folders are never lost:
    # pruning the os.walk at the first image-bearing directory silently dropped
    # everything below it (e.g. a stray cover.jpg at the root hid all PDFs
    # underneath). Instead, only suppress image folders nested under another
    # image folder (e.g. Book1/thumbnails under an image-book Book1). Because
    # the parent's image glob is non-recursive, a suppressed subfolder's images
    # are excluded rather than merged into the parent item, so every suppressed
    # folder is logged (with a hint to point the input at it directly) so
    # nothing processable is silently discarded. PDFs anywhere below are always
    # kept.
    folder_paths = set(image_folders)
    kept: dict[Path, list[Path]] = {}
    for folder, images in image_folders.items():
        suppressing = next(
            (parent for parent in folder.parents if parent in folder_paths), None
        )
        if suppressing is None:
            kept[folder] = images
        else:
            logger.warning(
                "Skipping image folder %s (%d image(s)): nested under image "
                "folder %s. Point the input path at it directly to process it.",
                folder,
                len(images),
                suppressing,
            )
    image_folders = kept

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
