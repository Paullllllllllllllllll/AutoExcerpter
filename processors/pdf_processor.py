import concurrent.futures
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from modules.config_loader import ConfigLoader
from modules.image_utils import SUPPORTED_IMAGE_EXTENSIONS


def extract_pdf_pages_to_images(pdf_path: Path, output_images_dir: Path) -> List[Path]:
    """Extracts pages from a PDF and saves them as images."""
    print(f"Extracting pages from PDF: {pdf_path.name}...")
    extracted_image_paths: List[Path] = []
    pdf_document = None

    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = len(pdf_document)
        if num_pages == 0:
            print("PDF appears to be empty.")
            return []

        # Load target DPI and JPEG quality from modules/image_processing_config.yaml
        cfg_loader = ConfigLoader()
        cfg_loader.load_configs()
        img_cfg = cfg_loader.get_image_processing_config().get('api_image_processing', {})
        target_dpi = int(img_cfg.get('target_dpi', 300))
        jpeg_quality = int(img_cfg.get('jpeg_quality', 95))

        page_numbers = list(range(num_pages))
        results_map = {}

        def extract_page_task(page_num: int):
            try:
                page = pdf_document[page_num]
                zoom = target_dpi / 72.0
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                image_path = output_images_dir / f"page_{page_num + 1:04d}.jpg"
                pil_img.save(image_path, "JPEG", quality=jpeg_quality)
                return page_num, image_path
            except Exception as e:
                print(f"Error extracting page {page_num + 1}: {e}")
                return page_num, None

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=8
        ) as executor:
            future_to_page = {
                executor.submit(extract_page_task, pn): pn for pn in
                page_numbers
            }
            for future in tqdm(
                    concurrent.futures.as_completed(future_to_page),
                    total=num_pages,
                    desc="Extracting PDF pages"
            ):
                pn, image_path = future.result()
                if image_path:
                    results_map[pn] = image_path

        extracted_image_paths = [results_map[i] for i in
                                 sorted(results_map.keys()) if i in results_map]

    except Exception as e:
        print(f"Failed to process PDF {pdf_path.name}: {e}")
        return []
    finally:
        if pdf_document:
            pdf_document.close()

    print(
        f"Successfully extracted {len(extracted_image_paths)} pages to {output_images_dir}.")
    return extracted_image_paths


def get_image_paths_from_folder(folder_path: Path) -> List[Path]:
    """Get a list of image paths from a folder, sorted by name."""
    print(f"Scanning image folder: {folder_path.name}...")
    image_paths = sorted(
        [p for p in folder_path.glob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
        key=lambda p: p.name
    )
    print(f"Found {len(image_paths)} images in folder.")
    return image_paths
