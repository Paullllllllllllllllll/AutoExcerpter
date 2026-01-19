"""PDF page extraction and image folder processing utilities.

This module provides functions for extracting pages from PDF files and
processing image folders for the AutoExcerpter transcription pipeline.

Key Features:
1. **Parallel PDF Extraction**: Uses ThreadPoolExecutor to extract multiple
   PDF pages concurrently for improved performance
   
2. **Provider-Specific Preprocessing**: Applies image preprocessing optimized
   for the target LLM provider (OpenAI, Google Gemini, Anthropic Claude):
   - OpenAI: Box fitting with padding (768×1536)
   - Google: Box fitting optimized for 768px tiles  
   - Anthropic: Max-side capping (1568px, no padding)
   
3. **Configuration-Driven**: Loads target DPI, JPEG quality, and preprocessing
   settings from image_processing.yaml with provider-specific sections
   
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
from typing import Any

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
from modules.model_utils import detect_model_type, get_image_config_section_name, ModelType
from modules.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================
def _apply_image_preprocessing(
    pil_img: Image.Image,
    img_cfg: dict[str, Any],
    model_type: ModelType = "openai",
) -> Image.Image:
    """
    Apply preprocessing steps to an image with provider-specific resizing.
    
    Args:
        pil_img: PIL Image to preprocess
        img_cfg: Provider-specific image configuration dict
        model_type: Model type for resize strategy ('openai', 'google', 'anthropic')
    
    Returns:
        Preprocessed PIL Image
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
        if pil_img.mode != "L":
            pil_img = ImageOps.grayscale(pil_img)
    
    # Get detail parameter based on model type
    if model_type == "google":
        detail = img_cfg.get('media_resolution', 'high') or 'high'
    elif model_type == "anthropic":
        detail = img_cfg.get('resize_profile', 'auto') or 'auto'
    else:
        detail = img_cfg.get('llm_detail', 'high') or 'high'
    
    # Resize with provider-specific strategy
    pil_img = ImageProcessor.resize_for_detail(pil_img, detail, img_cfg, model_type)
    
    return pil_img


# ============================================================================
# PDF Extraction Functions
# ============================================================================
def extract_pdf_pages_to_images(
    pdf_path: Path,
    output_images_dir: Path,
    provider: str = "openai",
    model_name: str = "",
) -> list[Path]:
    """
    Extract pages from a PDF and save them as images with provider-specific preprocessing.

    Args:
        pdf_path: Path to the PDF file
        output_images_dir: Directory to save extracted images
        provider: LLM provider name ('openai', 'google', 'anthropic', 'openrouter')
        model_name: Model name for OpenRouter passthrough detection

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

        # Detect model type and load provider-specific config
        model_type = detect_model_type(provider, model_name)
        section_name = get_image_config_section_name(model_type)
        
        cfg_loader = get_config_loader()
        full_img_cfg = cfg_loader.get_image_processing_config()
        img_cfg = full_img_cfg.get(section_name, {})
        
        target_dpi = int(img_cfg.get('target_dpi', DEFAULT_TARGET_DPI))
        jpeg_quality = int(img_cfg.get('jpeg_quality', DEFAULT_JPEG_QUALITY))
        
        logger.debug(
            f"PDF extraction using provider={provider}, model_type={model_type}, "
            f"config_section={section_name}, dpi={target_dpi}"
        )

        page_numbers = list(range(num_pages))
        results_map = {}

        def extract_page_task(page_num: int) -> tuple[int, Path | None]:
            """
            Extract a single page from the PDF and apply provider-specific preprocessing.

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
                grayscale_enabled = bool(img_cfg.get('grayscale_conversion', True))
                if grayscale_enabled:
                    pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csGRAY)
                    pil_img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                else:
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                
                # Apply provider-specific preprocessing (grayscale, resize, etc.)
                pil_img = _apply_image_preprocessing(pil_img, img_cfg, model_type)
                
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
