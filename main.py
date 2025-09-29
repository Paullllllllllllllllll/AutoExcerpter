"""
Transcription and summarization pipeline powered by OpenAI (gpt-5-mini).

This script:
1. Processes PDFs by extracting each page as an image, or processes images from a folder
2. Transcribes the images with OpenAI Responses API (gpt-5-mini) using a JSON schema
3. Optionally sends the transcribed text to OpenAI (gpt-5-mini) for structured summarization
4. Saves transcriptions to TXT and summaries to DOCX
5. Verifies page numbering coherence
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from core.transcriber import ItemTranscriber
from modules import app_config as config
from modules.image_utils import SUPPORTED_IMAGE_EXTENSIONS
from modules.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class ItemSpec:
    """Descriptor for a PDF file or image folder to process."""

    kind: str
    path: Path
    image_count: Optional[int] = None

    @property
    def output_stem(self) -> str:
        return self.path.stem

    def display_label(self) -> str:
        item_type_label = "PDF" if self.kind == "pdf" else "Image Folder"
        count_str = ""
        if self.kind == "image_folder" and self.image_count is not None:
            count_str = f" ({self.image_count} images)"
        return f"{item_type_label}: {self.path.name}{count_str} (from: {self.path.parent})"


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


def scan_input_path(path_to_scan: Path) -> List[ItemSpec]:
    """Gather items from a file or directory path."""

    logger.info("Scanning input: %s", path_to_scan)
    collected: List[ItemSpec] = []

    if path_to_scan.is_file():
        if _is_pdf_file(path_to_scan):
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

    logger.info("Found %s potential items from %s.", len(collected), path_to_scan)
    return collected


def prompt_for_item_selection(items: Sequence[ItemSpec]) -> List[ItemSpec]:
    if not items:
        print("No processable PDF files or image folders found in the input.")
        return []

    print("\nFound the following items to process:")
    for index, item in enumerate(items, start=1):
        print(f"  [{index}] {item.display_label()}")
    print(f"  [{len(items) + 1}] Process ALL listed items")

    while True:
        try:
            choice_str = input("\nEnter your choice(s) (e.g., 1; 3-5; all): ").lower().strip()

            if not choice_str:
                print("No selection made. Please enter choices or 'all'.")
                continue
            if choice_str == "all" or choice_str == str(len(items) + 1):
                return list(items)

            selected_indices: Set[int] = set()
            for part in choice_str.replace(" ", "").split(";"):
                if not part:
                    continue
                if "-" in part:
                    start_str, end_str = part.split("-", 1)
                    start = int(start_str)
                    end = int(end_str)
                    if not (1 <= start <= end <= len(items)):
                        raise ValueError(
                            f"Invalid range: {part}. Must be between 1 and {len(items)}."
                        )
                    selected_indices.update(range(start - 1, end))
                elif part.isdigit():
                    index = int(part) - 1
                    if not (0 <= index < len(items)):
                        raise ValueError(
                            f"Invalid index: {part}. Must be between 1 and {len(items)}."
                        )
                    selected_indices.add(index)
                else:
                    raise ValueError(
                        f"Invalid input part: {part}. Use numbers, ranges (e.g., 1-3), or 'all'."
                    )

            if not selected_indices:
                logger.warning("No valid items selected from your input. Please try again.")
                continue
            return [items[i] for i in sorted(selected_indices)]
        except ValueError as exc:
            logger.warning("Invalid selection: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("An unexpected error occurred during selection: %s", exc)


def _collect_items_from_directory(path_to_scan: Path) -> Iterable[ItemSpec]:
    image_folders: dict[Path, List[Path]] = {}
    items: List[ItemSpec] = []

    for root, dirs, files in os.walk(path_to_scan):
        current_dir = Path(root)

        for file_name in files:
            file_path = current_dir / file_name
            if _is_pdf_file(file_path):
                items.append(_build_pdf_item(file_path))

        for file_name in files:
            file_path = current_dir / file_name
            if _is_supported_image(file_path):
                image_folders.setdefault(current_dir, []).append(file_path)

        dirs[:] = [name for name in dirs if (current_dir / name) not in image_folders]

    items.extend(_build_image_folder_items(image_folders))
    return items


def _build_pdf_item(pdf_path: Path) -> ItemSpec:
    return ItemSpec(kind="pdf", path=pdf_path)


def _build_image_folder_items(image_folders: dict[Path, List[Path]]) -> List[ItemSpec]:
    image_items: List[ItemSpec] = []
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


def _is_pdf_file(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def main() -> None:
    args = setup_argparse()
    input_path_arg = Path(args.input)

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

    logger.info("Selected %s item(s) for processing.", len(selected_items))
    for index, item_spec in enumerate(selected_items, start=1):
        logger.info(
            "--- Starting Item %s of %s: %s (%s) ---",
            index,
            len(selected_items),
            item_spec.output_stem,
            item_spec.kind,
        )

        expected_outputs = [base_output_dir / f"{item_spec.output_stem}.txt"]
        if config.SUMMARIZE:
            expected_outputs.append(base_output_dir / f"{item_spec.output_stem}_summary.docx")

        if all(path.exists() for path in expected_outputs):
            logger.info("All output files for %s already exist. Skipping.", item_spec.output_stem)
            continue

        transcriber_instance: Optional[ItemTranscriber] = None
        try:
            transcriber_instance = ItemTranscriber(item_spec.path, item_spec.kind, base_output_dir)
            transcriber_instance.process_item()
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("CRITICAL ERROR processing item: %s", item_spec.output_stem)
            logger.info("--- Attempting to continue with the next item if any. ---")
        finally:
            if not transcriber_instance:
                continue
            working_dir = transcriber_instance.working_dir
            if not (config.DELETE_TEMP_WORKING_DIR and working_dir.exists()):
                continue

            def _on_remove_error(func, path_to_fix, _exc_info):
                try:
                    os.chmod(path_to_fix, stat.S_IWRITE)
                    func(path_to_fix)
                except Exception as exc_inner:  # pylint: disable=broad-except
                    logger.warning("Could not forcibly remove %s: %s", path_to_fix, exc_inner)

            try:
                shutil.rmtree(working_dir, onerror=_on_remove_error)
                logger.info("Deleted temporary working directory: %s", working_dir)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to remove working directory %s: %s", working_dir, exc)

    logger.info("All selected items have been processed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user (Ctrl+C). Exiting.")
        sys.exit(0)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("An unexpected critical error occurred in the main execution flow: %s", exc)
        import traceback

        traceback.print_exc()
        sys.exit(1)
