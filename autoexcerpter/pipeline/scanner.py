"""Find the processable items (PDF files and image folders) under an input path."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from autoexcerpter.common.discovery import image_folders, ordered, scan
from autoexcerpter.common.images import IMAGE_EXTENSIONS
from autoexcerpter.pipeline.types import ItemSpec

logger = logging.getLogger(__name__)


def is_pdf_file(path: Path) -> bool:
    """Check if a path points to a PDF file."""
    return path.suffix.lower() == ".pdf"


def is_supported_image(path: Path) -> bool:
    """Check if a path points to a supported image file."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


@dataclass(frozen=True)
class Discovered:
    """The items found at an input path.

    *unentered* counts the items in the subfolders that a search without
    ``--recursive`` did not enter.
    """

    items: list[ItemSpec] = field(default_factory=list)
    unentered: int = 0


def scan_input_path(path_to_scan: Path, *, recursive: bool = False) -> Discovered:
    """Gather items from a file or directory path.

    A folder gives its PDFs and the image folders that are the folder itself
    or one of its subfolders; with *recursive*, the PDFs and image folders
    at every level.
    """
    logger.debug("Scanning input: %s", path_to_scan)
    found = Discovered()

    if path_to_scan.is_file():
        if is_pdf_file(path_to_scan):
            found = Discovered([_build_pdf_item(path_to_scan)])
        else:
            logger.warning(
                "Input path %s is not a PDF file. Skipping.",
                path_to_scan,
            )
    elif path_to_scan.is_dir():
        found = _collect_items_from_directory(path_to_scan, recursive=recursive)
    else:
        logger.warning(
            "Input path %s is not a PDF file or a directory. Skipping.",
            path_to_scan,
        )

    logger.debug("Found %s potential items from %s.", len(found.items), path_to_scan)
    return found


def _depth(path: Path, root: Path) -> int:
    return len(path.relative_to(root).parts)


def _entered(item: ItemSpec, root: Path) -> bool:
    """Whether a search without ``--recursive`` takes *item*.

    It takes the PDFs in *root* and the image folders that are *root* or one
    of its subfolders.
    """
    if item.kind == "pdf":
        return item.path.parent == root
    return _depth(item.path, root) <= 1


def _collect_items_from_directory(
    path_to_scan: Path, *, recursive: bool = False
) -> Discovered:
    """Collect PDF files and image folders from a directory.

    The items come in the natural order of their paths below the folder
    (:func:`~autoexcerpter.common.discovery.ordered`), PDFs and image
    folders together. Without *recursive* the tree is still listed whole,
    to count the items in the subfolders left out.
    """
    try:
        files = scan(path_to_scan, depth=None)
    except OSError as exc:
        logger.warning("Cannot list %s: %s", path_to_scan, exc)
        return Discovered()
    found = image_folders(files, IMAGE_EXTENSIONS)
    items = [_build_pdf_item(file) for file in files if is_pdf_file(file)]
    items.extend(_build_image_folder_items(found.folders))
    kept = items if recursive else [i for i in items if _entered(i, path_to_scan)]
    # A folder nested in an image folder is no item at any depth: warn about
    # each one below an image folder this search takes.
    taken = {item.path for item in kept}
    images = Counter(file.parent for file in files if is_supported_image(file))
    for folder, above in found.nested:
        if any(parent in taken for parent in folder.parents):
            logger.warning(
                "Skipping image folder %s (%d image(s)): nested under image "
                "folder %s. Point the input path at it directly to process it.",
                folder,
                images[folder],
                above,
            )
    return Discovered(
        ordered(kept, path_to_scan, lambda item: item.path), len(items) - len(kept)
    )


def _build_pdf_item(pdf_path: Path) -> ItemSpec:
    """Create an ItemSpec for a PDF file."""
    return ItemSpec(kind="pdf", path=pdf_path)


def _build_image_folder_items(
    image_folders: Mapping[Path, Sequence[Path]],
) -> list[ItemSpec]:
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
    "Discovered",
    "scan_input_path",
    "is_pdf_file",
    "is_supported_image",
]
