"""In-memory page payload production for the streaming transcription pipeline.

This module replaces the former disk-based image round-trip: instead of
rendering every PDF page (or copying every folder image) to
``working_dir/images/`` and re-reading the JPEGs at request time, pages are
rendered/loaded, preprocessed, JPEG-encoded, and base64-encoded fully in
memory, one page per transcription worker.

Key abstractions:

- ``PagePayload``: immutable per-page unit of work carrying the base64 JPEG,
  naming/ordering metadata, and an image-provenance record (SHA-256 of the
  exact bytes sent to the API, dimensions, byte size, effective DPI).
- ``PdfPayloadSource``: lazy page renderer over a single open PyMuPDF
  document. PyMuPDF documents are not thread-safe, so all document access is
  serialized behind a lock; preprocessing and encoding run outside the lock.
- ``FolderPayloadSource``: lazy loader over a sorted image folder.

Both sources expose ``__len__``, ``image_name(index)``,
``build_payload(index)``, ``file_provenance()``, and ``close()`` so the
pipeline can treat PDFs and image folders uniformly.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import fitz  # PyMuPDF
import PIL
from PIL import Image, ImageOps

from config.constants import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_TARGET_DPI,
    PDF_DPI_CONVERSION_FACTOR,
)
from config.loader import get_config_loader
from config.logger import setup_logger
from imaging._provider import (
    detect_model_type,
    get_image_config_section_name,
)
from imaging.pdf import get_image_paths_from_folder
from imaging.preprocessing import ImageProcessor

logger = setup_logger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024


def _openai_detail_is_original(model_name: str) -> bool:
    """Whether the transcription model wants full-resolution ("original") detail.

    True only when the transcription model config sets ``image_size: original``
    AND the model actually accepts it (GPT-5.6 family). Used to skip the local
    box-fit resize so the bytes sent to the API keep their native resolution.
    Any config/registry hiccup yields False (keep the default resize) and never
    raises.
    """
    try:
        model_cfg = (
            get_config_loader().get_model_config().get("transcription_model", {})
        )
        image_size = str(model_cfg.get("image_size", "") or "").strip().lower()
        if image_size != "original":
            return False
        from llm.capabilities import detect_capabilities

        return detect_capabilities(model_name).supports_original_image_detail
    except Exception:
        return False


@dataclass(frozen=True)
class PagePayload:
    """A single page, preprocessed and base64-encoded, ready for the API."""

    base64: str
    image_name: str
    sequence_number: int
    original_input_order_index: int
    provenance: dict[str, Any]
    source_file: str
    page_index: int | None = None


def _sha256_of_file(path: Path) -> str:
    """Stream-hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_jpeg(img: Image.Image, jpeg_quality: int) -> tuple[bytes, int, int]:
    """JPEG-encode a preprocessed PIL image, returning (bytes, width, height)."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue(), img.width, img.height


def _extract_sequence_number(image_path: Path) -> int:
    """Extract a page/sequence number from an image filename.

    Pattern ``page_0001`` yields 1; otherwise the last number found in the
    stem is used; 0 when no number is present.
    """
    try:
        last = image_path.stem.split("_")[-1]
        if last.isdigit():
            return int(last)
    except (ValueError, IndexError):
        pass
    try:
        nums = [int(s) for s in re.findall(r"\d+", image_path.stem)]
        return nums[-1] if nums else 0
    except (ValueError, IndexError):
        return 0


class _PayloadSourceBase:
    """Shared provider-config resolution and provenance plumbing."""

    def __init__(self, source_path: Path, provider: str, model_name: str) -> None:
        self.source_path = source_path
        self.model_type = detect_model_type(provider.lower(), model_name.lower())
        section_name = get_image_config_section_name(self.model_type)
        full_img_cfg = get_config_loader().get_image_processing_config()
        self.img_cfg: dict[str, Any] = full_img_cfg.get(section_name, {})
        # Full-resolution ("original") OpenAI detail: route through the capped
        # 'original' resize path instead of the 768x1536 box-fit, keeping native
        # resolution but bounding the longest side / pixel budget so a folio
        # scan cannot exceed the API's input limits. Grayscale/transparency
        # handling is untouched.
        if self.model_type == "openai" and _openai_detail_is_original(model_name):
            self.img_cfg = {**self.img_cfg, "llm_detail": "original"}
        self.jpeg_quality = int(self.img_cfg.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
        # Render strategy: 'direct' (default) derives the PDF render DPI from the
        # active resize profile so pages rasterize straight to their final size;
        # 'supersample' restores the legacy render-at-target_dpi-then-downscale.
        # Honored per provider section, then top-level, then the 'direct' default.
        render_strategy = (
            str(
                self.img_cfg.get("render_strategy")
                or full_img_cfg.get("render_strategy")
                or "direct"
            )
            .strip()
            .lower()
        )
        self.render_strategy = (
            render_strategy
            if render_strategy in ("direct", "supersample")
            else "direct"
        )
        self._config_section_name = section_name

    def _base_file_provenance(self) -> dict[str, Any]:
        return {
            "source_file": str(self.source_path),
            "pymupdf_version": getattr(fitz, "pymupdf_version", None)
            or getattr(fitz, "VersionBind", "unknown"),
            "pillow_version": PIL.__version__,
            "image_config_section": self._config_section_name,
            "image_config": dict(self.img_cfg),
        }

    def close(self) -> None:  # pragma: no cover - overridden where needed
        """Release any resources held by the source."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PdfPayloadSource(_PayloadSourceBase):
    """Lazily render PDF pages into in-memory payloads.

    The fitz document is opened once and shared across transcription
    workers; ALL document access (page lookup, ``get_pixmap``, pixel-buffer
    copy) is serialized behind ``_render_lock``. Preprocessing, JPEG
    encoding, and hashing run outside the lock.
    """

    def __init__(
        self,
        pdf_path: Path,
        provider: str = "openai",
        model_name: str = "",
    ) -> None:
        super().__init__(pdf_path, provider, model_name)
        # Clamp to >= 1 so a 0/negative config cannot produce a degenerate
        # render matrix.
        self.target_dpi = max(
            1, int(self.img_cfg.get("target_dpi", DEFAULT_TARGET_DPI))
        )
        self._render_lock = threading.Lock()
        self._doc: fitz.Document | None = fitz.open(pdf_path)
        # A password-protected PDF opens and reports a page count, but every
        # page render raises, which would otherwise fill a full .txt with error
        # placeholders. Abort construction so the item fails cleanly via the
        # payload-source setup-failure path instead.
        if self._doc.needs_pass:
            self._doc.close()
            self._doc = None
            raise ValueError(f"Cannot open password-protected PDF: {pdf_path.name}")
        logger.debug(
            f"PdfPayloadSource opened {pdf_path.name}: {len(self)} page(s), "
            f"model_type={self.model_type}, dpi={self.target_dpi}"
        )

    def __len__(self) -> int:
        return len(self._doc) if self._doc is not None else 0

    @staticmethod
    def image_name(index: int) -> str:
        """Virtual page name; matches the legacy on-disk naming contract."""
        return f"page_{index + 1:04d}.jpg"

    def _render_zoom(self, page: fitz.Page) -> float:
        """Zoom factor for rasterizing *page*, honoring the render strategy.

        Under ``supersample`` this is always ``target_dpi/72`` (render large,
        downscale later). Under ``direct`` the zoom is derived from the active
        resize profile so the page rasterizes straight to (approximately) its
        final content size; it is clamped at ``target_dpi/72`` so a page is
        never render-upscaled beyond the quality ceiling (box-fit padding and
        any upscale still happen in the post-render pipeline).
        """
        base_zoom = self.target_dpi / PDF_DPI_CONVERSION_FACTOR
        if self.render_strategy != "direct":
            return base_zoom
        rect = page.rect
        pt_w = float(rect.width)
        pt_h = float(rect.height)
        if pt_w <= 0.0 or pt_h <= 0.0:
            return base_zoom
        scale = ImageProcessor.content_scale_factor(
            (pt_w * base_zoom, pt_h * base_zoom), self.img_cfg, self.model_type
        )
        return base_zoom * min(scale, 1.0)

    def build_payload(self, index: int) -> PagePayload:
        """Render, preprocess, and encode page *index* (0-based).

        Raises on render/preprocess failure; the caller converts the
        exception into a per-page error entry.
        """
        if self._doc is None:
            raise RuntimeError("PdfPayloadSource is closed")

        grayscale_enabled = bool(self.img_cfg.get("grayscale_conversion", True))

        with self._render_lock:
            if self._doc is None:
                raise RuntimeError("PdfPayloadSource is closed")
            page = self._doc[index]
            zoom = self._render_zoom(page)
            matrix = fitz.Matrix(zoom, zoom)
            if grayscale_enabled:
                pix = page.get_pixmap(
                    matrix=matrix, alpha=False, colorspace=fitz.csGRAY
                )
                pil_img = Image.frombytes("L", (pix.width, pix.height), pix.samples_mv)
            else:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pil_img = Image.frombytes(
                    "RGB", (pix.width, pix.height), pix.samples_mv
                )
            del pix, page

        pil_img = ImageProcessor.preprocess_pil_image(
            pil_img, self.img_cfg, self.model_type
        )
        jpeg_bytes, width, height = _encode_jpeg(pil_img, self.jpeg_quality)

        return PagePayload(
            base64=base64.b64encode(jpeg_bytes).decode("utf-8"),
            image_name=self.image_name(index),
            sequence_number=index + 1,
            original_input_order_index=index,
            provenance={
                "sha256": hashlib.sha256(jpeg_bytes).hexdigest(),
                "width": width,
                "height": height,
                "byte_size": len(jpeg_bytes),
                "effective_dpi": round(zoom * PDF_DPI_CONVERSION_FACTOR, 2),
            },
            source_file=str(self.source_path),
            page_index=index,
        )

    def file_provenance(self) -> dict[str, Any]:
        """File-level reproducibility record for the run's log header."""
        provenance = self._base_file_provenance()
        try:
            provenance["source_sha256"] = _sha256_of_file(self.source_path)
        except OSError as exc:
            logger.warning(f"Could not hash {self.source_path.name}: {exc}")
            provenance["source_sha256"] = None
        # Cheap identity field consumed by the resume input-change guard
        # (pipeline.resume._input_changed_since_log). Size only: mtime changes
        # on copies of an identical file and would force needless reprocessing.
        try:
            provenance["size"] = self.source_path.stat().st_size
        except OSError:
            provenance["size"] = None
        provenance["target_dpi"] = self.target_dpi
        return provenance

    def close(self) -> None:
        # Acquire the render lock so we never close the document out from
        # under an in-flight page.get_pixmap() render on a worker thread.
        with self._render_lock:
            if self._doc is not None:
                self._doc.close()
                self._doc = None


class FolderPayloadSource(_PayloadSourceBase):
    """Lazily load and preprocess images from a folder into payloads."""

    def __init__(
        self,
        folder_path: Path,
        provider: str = "openai",
        model_name: str = "",
    ) -> None:
        super().__init__(folder_path, provider, model_name)
        self.image_paths: list[Path] = get_image_paths_from_folder(folder_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def image_name(self, index: int) -> str:
        return self.image_paths[index].name

    def build_payload(self, index: int) -> PagePayload:
        """Load, preprocess, and encode the folder image at *index*."""
        image_path = self.image_paths[index]
        source_bytes = image_path.read_bytes()

        with Image.open(io.BytesIO(source_bytes)) as img_file:
            img_file.load()
            # Honor EXIF orientation so phone-photographed pages are not sent
            # sideways. exif_transpose returns an independent copy (or None).
            transposed = ImageOps.exif_transpose(img_file)
            oriented = transposed if transposed is not None else img_file
            pil_img = ImageProcessor.preprocess_pil_image(
                oriented, self.img_cfg, self.model_type
            )
            # Encode inside the with-block: preprocessing may return the same
            # (unmodified) image object, which is closed when the block exits.
            jpeg_bytes, width, height = _encode_jpeg(pil_img, self.jpeg_quality)

        return PagePayload(
            base64=base64.b64encode(jpeg_bytes).decode("utf-8"),
            image_name=image_path.name,
            sequence_number=_extract_sequence_number(image_path),
            original_input_order_index=index,
            provenance={
                "sha256": hashlib.sha256(jpeg_bytes).hexdigest(),
                "width": width,
                "height": height,
                "byte_size": len(jpeg_bytes),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            },
            source_file=str(image_path),
            page_index=None,
        )

    def file_provenance(self) -> dict[str, Any]:
        """Folder-level reproducibility record (per-image hashes live on pages)."""
        provenance = self._base_file_provenance()
        # Cheap identity fields consumed by the resume input-change guard
        # (pipeline.resume._input_changed_since_log): the image count and their
        # combined byte size. A folder whose image set changed under the same
        # name would otherwise let page-level reuse splice two documents into
        # one output. total_image_bytes is None when any image cannot be stat'd
        # (a mismatch cannot then be proven).
        provenance["image_count"] = len(self.image_paths)
        total_bytes = 0
        total_ok = True
        for path in self.image_paths:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                total_ok = False
                break
        provenance["total_image_bytes"] = total_bytes if total_ok else None
        return provenance


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "PagePayload",
    "PdfPayloadSource",
    "FolderPayloadSource",
]
