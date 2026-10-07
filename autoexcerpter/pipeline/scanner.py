"""Find the processable items (PDF files and image folders) under an input path."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from autoexcerpter.common.images import IMAGE_EXTENSIONS
from autoexcerpter.pipeline.types import ItemSpec

logger = logging.getLogger(__name__)


def is_pdf_file(path: Path) -> bool:
    """Check if a path points to a PDF file."""
    return path.suffix.lower() == ".pdf"


def is_supported_image(path: Path) -> bool:
    """Check if a path points to a supported image file."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


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


def _collect_items_from_directory(path_to_scan: Path) -> Iterable[ItemSpec]:
    """Recursively collect PDF files and image folders from a directory."""
    image_folders: dict[Path, list[Path]] = {}
    items: list[ItemSpec] = []

    for root, _dirs, files in os.walk(path_to_scan):
        current_dir = Path(root)
        for file_name in files:
            file_path = current_dir / file_name
            if is_pdf_file(file_path):
                items.append(_build_pdf_item(file_path))
            elif is_supported_image(file_path):
                image_folders.setdefault(current_dir, []).append(file_path)

    # The walk covers the whole tree, so PDFs below an image folder are kept.
    # An image folder nested under another image folder (Book1/thumbnails
    # under Book1) is skipped with a warning that names it, since the
    # parent's non-recursive image glob does not include its images.
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
        image_items.append(
            ItemSpec(
                kind="image_folder",
                path=folder_path,
                image_count=len(images),
            )
        )
    return image_items


__all__ = [
    "scan_input_path",
    "is_pdf_file",
    "is_supported_image",
]
