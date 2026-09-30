"""In-memory page payload production for the streaming transcription pipeline.

This module replaces the former disk-based image round-trip: instead of
rendering every PDF page (or copying every folder image) to
``working_dir/images/`` and re-reading the JPEGs at request time, pages are
rendered/loaded, preprocessed, JPEG/PNG-encoded, and base64-encoded fully in
memory, one page per transcription worker.

Key abstractions:

- ``PagePayload``: immutable per-page unit of work carrying the base64 image,
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
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import fitz  # PyMuPDF
import PIL
from PIL import Image, ImageOps

from config.constants import PDF_DPI_CONVERSION_FACTOR
from config.loader import get_config_loader
from config.logger import setup_logger
from imaging.native import (
    TargetDpi,
    encode_payload,
    format_downscale_log,
    guarded_payload,
    image_settings_fingerprint,
    model_image_cap,
    native_page_dpi,
    resolve_target_size,
)
from imaging.pdf import get_image_paths_from_folder
from imaging.preprocessing import ImageProcessor
from imaging.settings import resolve_image_settings

logger = setup_logger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024


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
    mime_type: str = "image/jpeg"


def _sha256_of_file(path: Path) -> str:
    """Stream-hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_jpeg(img: Image.Image, jpeg_quality: int) -> tuple[bytes, int, int]:
    """JPEG-encode a preprocessed PIL image, returning (bytes, width, height)."""
    data, _ = encode_payload(img, "jpeg", jpeg_quality)
    return data, img.width, img.height


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

    _native_profile_warned = False

    def __init__(
        self, source_path: Path, provider: str | None, model_name: str | None
    ) -> None:
        self.source_path = source_path
        self.img_cfg, self.model_type, self._config_section_name = (
            resolve_image_settings(get_config_loader(), provider, model_name)
        )
        self.jpeg_quality = int(self.img_cfg["jpeg_quality"])
        self.render_strategy = str(self.img_cfg["render_strategy"])
        self.target_dpi: TargetDpi = self.img_cfg["target_dpi"]
        self.max_pixels = int(self.img_cfg["max_pixels_per_page"])
        if (
            self.target_dpi == "native"
            and self.img_cfg["cap_policy"] == "profile-v1"
            and not _PayloadSourceBase._native_profile_warned
        ):
            logger.warning(
                "Native resolution is bounded by a resize profile; "
                "the profile discards native resolution."
            )
            _PayloadSourceBase._native_profile_warned = True

    def _base_file_provenance(self) -> dict[str, Any]:
        return {
            "source_file": str(self.source_path),
            "pymupdf_version": getattr(fitz, "pymupdf_version", None)
            or getattr(fitz, "VersionBind", "unknown"),
            "pillow_version": PIL.__version__,
            "image_config_section": self._config_section_name,
            "image_config": dict(self.img_cfg),
            "model_type": self.model_type,
            "model_name": self.img_cfg["model_name"],
            "detail": self.img_cfg["request_detail"],
            "cap_policy": self.img_cfg["cap_policy"],
            "image_settings_fingerprint": image_settings_fingerprint(self.img_cfg),
        }

    def _payload_from_pil(
        self,
        image: Image.Image,
        index: int,
        image_name: str,
        sequence_number: int,
        source_file: str,
        page_index: int | None,
        provenance: dict[str, Any],
        img_cfg: dict[str, Any],
        *,
        effective_dpi: float | None = None,
    ) -> PagePayload:
        """Preprocess, encode and record the exact transmitted payload."""
        detail = str(img_cfg["resolved_detail"])
        if "downscale_reason" not in provenance:
            size, provenance["downscale_reason"] = resolve_target_size(
                image.width,
                image.height,
                self.model_type,
                img_cfg["model_name"],
                detail,
                img_cfg,
            )
            img_cfg = {**img_cfg, "_target_size": size}
        processed = ImageProcessor.preprocess_pil_image(image, img_cfg, self.model_type)
        data, mime, fallback = guarded_payload(
            processed,
            img_cfg["payload_format"],
            self.jpeg_quality,
            int(img_cfg["max_image_bytes"]),
            log=logger,
        )
        padded = (
            model_image_cap(self.model_type, img_cfg["model_name"], detail, img_cfg)
            is None
            and self.model_type != "anthropic"
            and detail != "low"
            and img_cfg["resize_profile"] != "none"
        )
        provenance.update(
            sha256=hashlib.sha256(data).hexdigest(),
            width=processed.width,
            height=processed.height,
            byte_size=len(data),
            sent_dpi=(
                None
                if padded or provenance["source_dpi_x"] is None
                else max(provenance["source_dpi_x"], provenance["source_dpi_y"])
                * processed.width
                / provenance["source_width"]
            ),
            cap_policy=img_cfg["cap_policy"],
            payload_format=mime.split("/")[1],
            mime_type=mime,
            format_fallback=fallback,
        )
        if effective_dpi is not None:
            provenance["effective_dpi"] = effective_dpi
        if provenance["downscale_reason"] != "none":
            logger.info(
                format_downscale_log(index + 1, provenance, img_cfg["model_name"])
            )
        return PagePayload(
            base64=base64.b64encode(data).decode("utf-8"),
            image_name=image_name,
            sequence_number=sequence_number,
            original_input_order_index=index,
            provenance=provenance,
            source_file=source_file,
            page_index=page_index,
            mime_type=mime,
        )

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
        provider: str | None = "openai",
        model_name: str | None = "",
    ) -> None:
        super().__init__(pdf_path, provider, model_name)
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

    def _render_zoom(
        self, page: fitz.Page, img_cfg: dict[str, Any] | None = None
    ) -> float:
        """Zoom factor for rasterizing *page*, honoring the render strategy.

        Under ``supersample`` this starts at ``target_dpi/72`` (render large,
        downscale later). The numeric memory guard can reduce either strategy.
        Under ``direct`` the zoom is derived from the active
        resize profile so the page rasterizes straight to (approximately) its
        final content size; it is clamped at ``target_dpi/72`` so a page is
        never render-upscaled beyond the quality ceiling (box-fit padding and
        any upscale still happen in the post-render pipeline).
        """
        if self.target_dpi == "native":
            raise ValueError("Native zoom is resolved per page inside the render lock")
        base_zoom = self.target_dpi / PDF_DPI_CONVERSION_FACTOR
        rect = page.rect
        pt_w, pt_h = float(rect.width), float(rect.height)
        zoom = base_zoom
        if self.render_strategy == "direct" and pt_w > 0 and pt_h > 0:
            scale = ImageProcessor.content_scale_factor(
                (pt_w * base_zoom, pt_h * base_zoom),
                img_cfg if img_cfg is not None else self.img_cfg,
                self.model_type,
            )
            zoom *= min(scale, 1.0)
        # Numeric renders only, matching the reference's float memory guard.
        pixels = (pt_w * zoom) * (pt_h * zoom)
        if self.max_pixels > 0 and pixels > self.max_pixels:
            zoom *= math.sqrt(self.max_pixels / pixels)
        return zoom

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
            density = (
                native_page_dpi(page, int(self.img_cfg["native_fallback_dpi"]))
                if self.target_dpi == "native"
                else None
            )
            source_dpi = density.dpi if density else float(self.target_dpi)
            source_width = math.ceil(page.rect.width * source_dpi / 72 - 1e-7)
            source_height = math.ceil(page.rect.height * source_dpi / 72 - 1e-7)
            size, reason = resolve_target_size(
                source_width,
                source_height,
                self.model_type,
                self.img_cfg["model_name"],
                self.img_cfg["resolved_detail"],
                self.img_cfg,
            )
            img_cfg = {**self.img_cfg, "_target_size": size}
            if density:
                zoom = (
                    source_dpi
                    / 72
                    * min(size[0] / source_width, size[1] / source_height)
                )
            else:
                zoom = self._render_zoom(page, img_cfg)
                if self.max_pixels > 0 and self.max_pixels < size[0] * size[1]:
                    reason = "memory_guard"
            provenance = {
                "source_dpi_x": density.dpi_x if density else source_dpi,
                "source_dpi_y": density.dpi_y if density else source_dpi,
                "dpi_source": density.source if density else "numeric",
                "source_width": source_width,
                "source_height": source_height,
                "render_dpi": zoom * 72,
                "downscale_reason": reason,
            }
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

        return self._payload_from_pil(
            pil_img,
            index,
            self.image_name(index),
            index + 1,
            str(self.source_path),
            index,
            provenance,
            img_cfg,
            effective_dpi=round(zoom * PDF_DPI_CONVERSION_FACTOR, 2),
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
        provider: str | None = "openai",
        model_name: str | None = "",
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
            metadata = img_file.info.get("dpi")
            provenance = {
                "source_dpi_x": None,
                "source_dpi_y": None,
                "dpi_source": "file",
                "source_width": oriented.width,
                "source_height": oriented.height,
                "render_dpi": None,
                "file_dpi_metadata": metadata,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
            # Encoding stays inside the with-block; processing can return the input.
            return self._payload_from_pil(
                oriented,
                index,
                image_path.name,
                _extract_sequence_number(image_path),
                str(image_path),
                None,
                provenance,
                self.img_cfg,
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
        # Name-set identity: page order is derived deterministically from the
        # filenames (natural sort), so a hash over the sorted names pins the
        # order too. Count + bytes alone missed a RENAME that reshuffles the
        # order while preserving both (e.g. a.jpg -> z.jpg), which let resume
        # splice logged pages into the wrong slots.
        provenance["image_names_sha256"] = hashlib.sha256(
            "\n".join(sorted(p.name for p in self.image_paths)).encode("utf-8")
        ).hexdigest()
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
