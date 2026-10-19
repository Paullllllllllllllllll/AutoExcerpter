"""Deterministic PDF and image-folder inputs carrying a decodable page marker.

Every page is filled with one gray level from ``PALETTE``; the level encodes
the page index. The marker survives JPEG and PNG encoding, grayscale
conversion, resizing and white box-fit padding, because ``decode_marker``
takes the median of the non-white pixels and picks the nearest palette level.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

PALETTE: tuple[int, ...] = tuple(24 + 18 * step for step in range(12))
"""Gray levels for page indices 0-11; padding white (255) stays out of range."""

_PADDING_CUTOFF = 245

_PIL_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".webp": "WEBP",
}


def marker_gray(index: int) -> int:
    """Return the gray level that marks page *index* (0-based)."""
    if not 0 <= index < len(PALETTE):
        raise ValueError(f"page index {index} outside the marker palette")
    return PALETTE[index]


def decode_marker(image: Image.Image) -> int | None:
    """Return the page index encoded in *image*, or None for an unmarked image."""
    gray = image.convert("L")
    gray.thumbnail((256, 256))
    histogram = gray.histogram()[:_PADDING_CUTOFF]
    total = sum(histogram)
    if total == 0:
        return None
    running = 0
    median = 0
    for level, count in enumerate(histogram):
        running += count
        if running * 2 >= total:
            median = level
            break
    return min(range(len(PALETTE)), key=lambda idx: abs(PALETTE[idx] - median))


def _page_image(index: int, size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    level = marker_gray(index)
    color: int | tuple[int, ...] = level if mode == "L" else (level, level, level)
    return Image.new(mode, size, color=color)


def make_pdf(
    path: Path,
    pages: int = 3,
    *,
    page_size: tuple[float, float] = (200.0, 300.0),
    scan_size: tuple[int, int] | None = None,
) -> Path:
    """Write a PDF whose pages carry the page markers.

    By default each page is a vector fill, which native DPI resolution treats
    as a page without a scan. With *scan_size* each page instead holds one
    page-spanning grayscale image of that pixel size, so native DPI resolves
    from the scan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        width, height = page_size
        for index in range(pages):
            page = doc.new_page(width=width, height=height)
            if scan_size is None:
                shade = marker_gray(index) / 255
                page.draw_rect(page.rect, color=None, fill=(shade, shade, shade))
            else:
                buffer = io.BytesIO()
                _page_image(index, scan_size, mode="L").save(buffer, format="PNG")
                page.insert_image(page.rect, stream=buffer.getvalue())
        doc.save(str(path), garbage=4, deflate=True, no_new_id=True)
    finally:
        doc.close()
    return path


def make_image_folder(
    folder: Path,
    pages: int = 3,
    *,
    extensions: Sequence[str] = (".jpg",),
    size: tuple[int, int] = (400, 600),
    stem: str = "page",
) -> Path:
    """Write an image folder with one marked image per page.

    Page *i* is named ``<stem>_<i+1:03d><ext>``, cycling through *extensions*,
    so one folder can mix every supported image type.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(pages):
        extension = extensions[index % len(extensions)].lower()
        image = _page_image(index, size)
        target = folder / f"{stem}_{index + 1:03d}{extension}"
        pil_format = _PIL_FORMATS[extension]
        if pil_format == "JPEG":
            image.save(target, format=pil_format, quality=95)
        elif pil_format == "WEBP":
            image.save(target, format=pil_format, lossless=True)
        else:
            image.save(target, format=pil_format)
    return folder


def make_named_images(
    folder: Path,
    names: Sequence[str],
    *,
    size: tuple[int, int] = (400, 600),
) -> Path:
    """Write one marked image per name; ``names[i]`` carries page marker *i*.

    The format follows each name's extension, so a caller fixes both the file
    names (for example unpadded numbers) and the page each file holds.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        _save_image(_page_image(index, size), folder / name)
    return folder


def _save_image(image: Image.Image, target: Path) -> None:
    pil_format = _PIL_FORMATS[target.suffix.lower()]
    if pil_format == "JPEG":
        image.save(target, format=pil_format, quality=95)
    elif pil_format == "WEBP":
        image.save(target, format=pil_format, lossless=True)
    else:
        image.save(target, format=pil_format)
