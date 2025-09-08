"""
Transcription and summarization pipeline powered by OpenAI (gpt-5-mini).

This script:
1. Processes PDFs by extracting each page as an image, or processes images from a folder
2. Transcribes the images with OpenAI Responses API (gpt-5-mini) using a JSON schema
3. Optionally sends the transcribed text to OpenAI (gpt-5-mini) for structured summarization
4. Saves transcriptions to TXT and summaries to DOCX
5. Verifies page numbering coherence
"""

import argparse
import os
import shutil
import sys
import stat
from pathlib import Path
from typing import List, Dict, Any

from modules import app_config as config
from core.transcriber import ItemTranscriber
from modules.image_utils import SUPPORTED_IMAGE_EXTENSIONS
from modules.logger import setup_logger

logger = setup_logger(__name__)

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF and Image Folder Transcription and Summarization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=config.INPUT_FOLDER_PATH,
        help="Path to the folder containing PDFs and/or image folders, or path to a single PDF/image folder.",
    )
    return parser.parse_args()

def scan_input_path(path_to_scan: Path) -> List[Dict[str, Any]]:
    """Scans a given path. If it's a file, processes it. If it's a directory, walks it."""
    items_to_process: List[Dict[str, Any]] = []
    logger.info(f"Scanning input: {path_to_scan}")

    if path_to_scan.is_file() and path_to_scan.suffix.lower() == ".pdf":
        items_to_process.append(
            {"type": "pdf", "path": path_to_scan, "name": path_to_scan.name})
    elif path_to_scan.is_dir():
        image_folders_found: Dict[Path, List[Path]] = {}  # Parent dir -> list of image files

        for root, dirs, files in os.walk(path_to_scan):
            current_dir = Path(root)
            # Check for PDF files
            for file_name in files:
                file_path = current_dir / file_name
                if file_path.suffix.lower() == ".pdf":
                    items_to_process.append({"type": "pdf", "path": file_path, "name": file_path.name})

            # Group images by their parent directory to identify image folders
            for file_name in files:
                file_path = current_dir / file_name
                if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    if current_dir not in image_folders_found:
                        image_folders_found[current_dir] = []
                    image_folders_found[current_dir].append(file_path)

            # Prune directories that are already marked as image folders from further os.walk
            dirs[:] = [d_name for d_name in dirs if (current_dir / d_name) not in image_folders_found]

        # Add collected image folders to items_to_process
        for folder_path, image_files_list in image_folders_found.items():
            if image_files_list:  # Ensure the folder actually contained images
                image_files_list.sort(key=lambda p: p.name)  # Sort images within the folder
                items_to_process.append({
                    "type": "image_folder",
                    "path": folder_path,
                    "name": folder_path.name,  # Name of the folder
                    "image_count": len(image_files_list),
                })
    else:
        logger.warning(f"Input path {path_to_scan} is not a PDF file or a directory. Skipping.")

    logger.info(f"Found {len(items_to_process)} potential items from {path_to_scan}.")
    return items_to_process

def prompt_for_item_selection(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        print("No processable PDF files or image folders found in the input.")
        return []

    print("\nFound the following items to process:")
    for i, item in enumerate(items):
        item_type_label = "PDF" if item["type"] == "pdf" else "Image Folder"
        count_str = f" ({item.get('image_count', 0)} images)" if item["type"] == "image_folder" else ""
        print(f"  [{i + 1}] {item_type_label}: {item['name']}{count_str} (from: {item['path'].parent})")
    print(f"  [{len(items) + 1}] Process ALL listed items")

    while True:
        try:
            choice_str = input("\nEnter your choice(s) (e.g., 1; 3-5; all): ").lower().strip()

            if not choice_str:  # User pressed Enter without input
                print("No selection made. Please enter choices or 'all'.")
                continue
            if choice_str == "all" or choice_str == str(len(items) + 1):
                return items

            selected_indices = set()
            parts = choice_str.replace(" ", "").split(";")
            for part in parts:
                if not part: continue
                if "-" in part:
                    start_str, end_str = part.split("-")
                    start = int(start_str)
                    end = int(end_str)
                    if not (1 <= start <= end <= len(items)):
                        raise ValueError(f"Invalid range: {part}. Must be between 1 and {len(items)}.")
                    selected_indices.update(range(start - 1, end))
                elif part.isdigit():
                    index = int(part) - 1
                    if not (0 <= index < len(items)):
                        raise ValueError(f"Invalid index: {part}. Must be between 1 and {len(items)}.")
                    selected_indices.add(index)
                else:
                    raise ValueError(f"Invalid input part: {part}. Use numbers, ranges (e.g., 1-3), or 'all'.")

            if not selected_indices:
                logger.warning("No valid items selected from your input. Please try again.")
                continue
            return [items[i] for i in sorted(list(selected_indices))]
        except ValueError as e:
            logger.warning(f"Invalid selection: {e}")
        except Exception as e_outer:
            logger.exception(f"An unexpected error occurred during selection: {e_outer}")

def main():
    args = setup_argparse()
    # The input can be a folder to scan, or a direct path to a PDF/image_folder
    input_path_arg = Path(args.input)

    # Ensure base output directory exists
    base_output_dir = Path(config.OUTPUT_FOLDER_PATH)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    all_items_to_consider = scan_input_path(input_path_arg)
    if not all_items_to_consider:
        logger.info("No items found to process. Exiting.")
        sys.exit(0)

    selected_items = prompt_for_item_selection(all_items_to_consider)
    if not selected_items:
        logger.info("No items selected for processing. Exiting.")
        sys.exit(0)

    logger.info(f"Selected {len(selected_items)} item(s) for processing.")
    for i, item_spec in enumerate(selected_items):
        item_path = item_spec["path"]
        item_type = item_spec["type"]
        item_name = item_path.stem  # Used for naming output files and working dirs

        logger.info(f"--- Starting Item {i + 1} of {len(selected_items)}: {item_name} ({item_type}) ---")

        # Check if final output files already exist for this item
        expected_final_outputs = [
            base_output_dir / f"{item_name}.txt",
        ]

        if config.SUMMARIZE:
            expected_final_outputs.append(
                base_output_dir / f"{item_name}_summary.docx"
            )

        if all(path.exists() for path in expected_final_outputs):
            logger.info(f"All output files for {item_name} already exist. Skipping.")
            continue

        transcriber_instance = None
        try:
            transcriber_instance = ItemTranscriber(item_path, item_type, base_output_dir)
            transcriber_instance.process_item()
        except Exception as e:
            logger.exception(f"CRITICAL ERROR processing item: {item_name}")
            logger.info("--- Attempting to continue with the next item if any. ---")
        finally:
            if transcriber_instance:
                # Conditionally delete the temporary working directory for the processed item
                if config.DELETE_TEMP_WORKING_DIR and transcriber_instance.working_dir.exists():
                    def onerror(func, path, exc_info):
                        # Try to change the file to writable and retry
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                        except Exception as e:
                            logger.warning(f"Could not forcibly remove {path}: {e}")
                    try:
                        shutil.rmtree(transcriber_instance.working_dir, onerror=onerror)
                        logger.info(
                            f"Deleted temporary working directory: {transcriber_instance.working_dir}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove working directory {transcriber_instance.working_dir}: {e}")
                # The item's working directory (containing its log) is retained if not deleted.

    logger.info("All selected items have been processed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user (Ctrl+C). Exiting.")
        sys.exit(0)
    except Exception as e:  # Catch-all for unexpected errors in main setup
        logger.exception(
            f"An unexpected critical error occurred in the main execution flow: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
