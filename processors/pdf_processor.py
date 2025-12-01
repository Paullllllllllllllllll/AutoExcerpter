"""PDF page extraction and image folder processing utilities.

This module provides functions for extracting pages from PDF files and
processing image folders for the AutoExcerpter transcription pipeline.

Key Features:
1. **Parallel PDF Extraction**: Uses ThreadPoolExecutor to extract multiple
   PDF pages concurrently for improved performance
   
2. **Integrated Preprocessing**: Applies image preprocessing (grayscale, resize,
   transparency handling) during extraction to eliminate redundant operations
   
3. **Configuration-Driven**: Loads target DPI, JPEG quality, and preprocessing
   settings from image_processing.yaml
   
4. **Error Resilient**: Continues processing even if individual pages fail,
   logging errors for troubleshooting

5. **Image Folder Support**: Scans directories for supported image formats
   and returns sorted paths for processing

The extracted/processed images are saved as JPEG files with consistent naming
(page_0001.jpg, page_0002.jpg, etc.) for predictable ordering in transcription.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from modules.config_loader import get_config_loader
from modules.constants import (
    SUPPORTED_IMAGE_EXTENSIONS,
    DEFAULT_TARGET_DPI,
    DEFAULT_JPEG_QUALITY,
    MAX_EXTRACTION_WORKERS,
    PDF_DPI_CONVERSION_FACTOR,
    WHITE_BACKGROUND_COLOR,
)
from modules.image_utils import ImageProcessor
from modules.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================
def _apply_image_preprocessing(pil_img: Image.Image, img_cfg: dict[str, any]) -> Image.Image:
    """
    Apply preprocessing steps to an image.
    
    Delegates to ImageProcessor static methods to avoid code duplication.
    """
    from PIL import ImageOps
    
    # Handle transparency
    if img_cfg.get('handle_transparency', True):
        if pil_img.mode in ('RGBA', 'LA') or (
                pil_img.mode == 'P' and 'transparency' in pil_img.info):
            background = Image.new("RGB", pil_img.size, WHITE_BACKGROUND_COLOR)
            mask = pil_img.split()[-1] if pil_img.mode in ('RGBA', 'LA') else None
            background.paste(pil_img, mask=mask)
            pil_img = background
    
    # Grayscale conversion
    if img_cfg.get('grayscale_conversion', True):
        pil_img = ImageOps.grayscale(pil_img)
    
    # Resize based on detail level (delegates to ImageProcessor)
    detail = img_cfg.get('llm_detail', 'high') or 'high'
    pil_img = ImageProcessor.resize_for_detail(pil_img, detail, img_cfg)
    
    return pil_img


# ============================================================================
# PDF Extraction Functions
# ============================================================================
def extract_pdf_pages_to_images(pdf_path: Path, output_images_dir: Path) -> list[Path]:
    """
    Extract pages from a PDF and save them as images.

    Args:
        pdf_path: Path to the PDF file
        output_images_dir: Directory to save extracted images

    Returns:
        List of paths to extracted images, ordered by page number
    """
    logger.info(f"Extracting pages from PDF: {pdf_path.name}...")
    extracted_image_paths: list[Path] = []
    pdf_document = None

    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = len(pdf_document)

        if num_pages == 0:
            logger.warning("PDF appears to be empty.")
            return []

        # Load target DPI and JPEG quality from configuration
        cfg_loader = get_config_loader()
        img_cfg = cfg_loader.get_image_processing_config().get('api_image_processing', {})
        target_dpi = int(img_cfg.get('target_dpi', DEFAULT_TARGET_DPI))
        jpeg_quality = int(img_cfg.get('jpeg_quality', DEFAULT_JPEG_QUALITY))

        page_numbers = list(range(num_pages))
        results_map = {}

        def extract_page_task(page_num: int) -> tuple[int, Path | None]:
            """
            Extract a single page from the PDF and apply preprocessing.

            Args:
                page_num: Zero-indexed page number

            Returns:
                Tuple of (page_num, image_path or None on error)
            """
            try:
                # Extract page from PDF
                page = pdf_document[page_num]
                zoom = target_dpi / PDF_DPI_CONVERSION_FACTOR
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                
                # Apply preprocessing directly (grayscale, resize, etc.)
                pil_img = _apply_image_preprocessing(pil_img, img_cfg)
                
                # Save preprocessed image
                image_path = output_images_dir / f"page_{page_num + 1:04d}.jpg"
                pil_img.save(image_path, "JPEG", quality=jpeg_quality)
                return page_num, image_path
            except Exception as e:
                logger.error(f"Error extracting page {page_num + 1}: {e}")
                return page_num, None

        # Extract pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EXTRACTION_WORKERS) as executor:
            future_to_page = {
                executor.submit(extract_page_task, pn): pn for pn in page_numbers
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_page),
                total=num_pages,
                desc="Extracting PDF pages",
            ):
                pn, image_path = future.result()
                if image_path:
                    results_map[pn] = image_path

        # Return images in order
        extracted_image_paths = [
            results_map[i] for i in sorted(results_map.keys()) if i in results_map
        ]

    except Exception as e:
        logger.exception(f"Failed to process PDF {pdf_path.name}: {e}")
        return []
    finally:
        if pdf_document:
            pdf_document.close()

    logger.info(
        f"Successfully extracted {len(extracted_image_paths)} pages to {output_images_dir}."
    )
    return extracted_image_paths


# ============================================================================
# Image Folder Processing Functions
# ============================================================================
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
    "extract_pdf_pages_to_images",
    "get_image_paths_from_folder",
]
