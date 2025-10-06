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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from core.transcriber import ItemTranscriber
from modules import app_config as config
from modules.image_utils import SUPPORTED_IMAGE_EXTENSIONS
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker
from modules.user_prompts import (
    print_header,
    print_section,
    print_success,
    print_warning,
    print_error,
    print_info,
    prompt_selection,
    exit_program,
)

logger = setup_logger(__name__)

# Constants
MIN_VALID_CHOICE = 1
SELECTION_PROMPT = "\nEnter your choice(s) (e.g., 1; 3-5; all): "

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
    
    if config.CLI_MODE:
        # CLI mode: require input and output arguments
        parser.add_argument(
            "input",
            type=str,
            help="Path to PDF file, image folder, or directory containing PDFs/image folders (relative or absolute).",
        )
        parser.add_argument(
            "output",
            type=str,
            help="Output directory path for transcriptions and summaries (relative or absolute).",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Process all items found in input directory without prompting.",
        )
    else:
        # Interactive mode: optional input argument with default from config
        parser.add_argument(
            "--input",
            type=str,
            default=config.INPUT_FOLDER_PATH,
            help="Path to the folder containing PDFs and/or image folders, or path to a single PDF/image folder.",
        )
    
    return parser.parse_args()


def scan_input_path(path_to_scan: Path) -> List[ItemSpec]:
    """Gather items from a file or directory path."""

    logger.debug("Scanning input: %s", path_to_scan)
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

    logger.debug("Found %s potential items from %s.", len(collected), path_to_scan)
    return collected


def prompt_for_item_selection(items: Sequence[ItemSpec]) -> List[ItemSpec]:
    """Prompt user to select items to process with improved UX."""
    if not items:
        print_warning("No processable PDF files or image folders found in the input.")
        return []

    print_section("Available Items to Process")
    
    selected = prompt_selection(
        items=items,
        display_func=lambda item: item.display_label(),
        prompt_message="Select items to process",
        allow_multiple=True,
        allow_all=True,
        allow_back=False,
        allow_exit=True,
        process_all_label="Process ALL listed items",
    )
    
    return selected if selected is not None else []


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


def _process_single_item(
    item_spec: ItemSpec,
    index: int,
    total_items: int,
    base_output_dir: Path,
) -> bool:
    """
    Process a single PDF or image folder item.
    
    Args:
        item_spec: Specification of the item to process.
        index: Current item index (1-based).
        total_items: Total number of items to process.
        base_output_dir: Base directory for outputs.
        
    Returns:
        True if processing succeeded, False otherwise.
    """
    logger.info(
        "--- Starting Item %s of %s: %s (%s) ---",
        index,
        total_items,
        item_spec.output_stem,
        item_spec.kind,
    )
    
    if not config.CLI_MODE:
        print_info(f"Processing [{index}/{total_items}]: {item_spec.output_stem}")

    # Check if output files already exist
    expected_outputs = [base_output_dir / f"{item_spec.output_stem}.txt"]
    if config.SUMMARIZE:
        expected_outputs.append(base_output_dir / f"{item_spec.output_stem}_summary.docx")

    if all(path.exists() for path in expected_outputs):
        if config.CLI_MODE:
            logger.warning(f"Output files for '{item_spec.output_stem}' already exist. Skipping.")
        else:
            print_warning(f"Output files for '{item_spec.output_stem}' already exist. Skipping.")
        logger.info("All output files for %s already exist. Skipping.", item_spec.output_stem)
        return True

    # Process the item
    transcriber_instance: Optional[ItemTranscriber] = None
    try:
        transcriber_instance = ItemTranscriber(item_spec.path, item_spec.kind, base_output_dir)
        transcriber_instance.process_item()
        
        if config.CLI_MODE:
            logger.info(f"Successfully processed: {item_spec.output_stem}")
        else:
            print_success(f"Successfully processed: {item_spec.output_stem}")
        return True
        
    except Exception as exc:
        logger.exception("CRITICAL ERROR processing item: %s", item_spec.output_stem)
        if not config.CLI_MODE:
            print_error(f"Critical error processing '{item_spec.output_stem}'. Check logs for details.")
        logger.info("--- Attempting to continue with the next item if any. ---")
        return False
        
    finally:
        # Clean up temporary working directory if configured
        if transcriber_instance and config.DELETE_TEMP_WORKING_DIR:
            working_dir = transcriber_instance.working_dir
            if working_dir.exists():
                _cleanup_working_directory(working_dir)


def _cleanup_working_directory(working_dir: Path) -> None:
    """
    Clean up temporary working directory with proper error handling.
    
    Args:
        working_dir: Path to working directory to remove.
    """
    def _on_remove_error(func, path_to_fix, _exc_info):
        """Handle permission errors during directory removal."""
        try:
            os.chmod(path_to_fix, stat.S_IWRITE)
            func(path_to_fix)
        except Exception as exc_inner:
            logger.warning("Could not forcibly remove %s: %s", path_to_fix, exc_inner)

    try:
        shutil.rmtree(working_dir, onerror=_on_remove_error)
        logger.debug("Deleted temporary working directory: %s", working_dir)
    except Exception as exc:
        logger.warning("Failed to remove working directory %s: %s", working_dir, exc)


def _check_and_wait_for_token_limit() -> bool:
    """
    Check if daily token limit is reached and wait until next day if needed.
    
    Returns:
        True if processing can continue, False if user cancelled wait.
    """
    if not config.DAILY_TOKEN_LIMIT_ENABLED:
        return True
    
    token_tracker = get_token_tracker()
    
    if not token_tracker.is_limit_reached():
        return True
    
    # Token limit reached - need to wait until next day
    stats = token_tracker.get_stats()
    reset_time = token_tracker.get_reset_time()
    seconds_until_reset = token_tracker.get_seconds_until_reset()
    
    logger.warning(
        f"Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    logger.info(
        f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m) "
        "for token limit reset..."
    )
    
    cancel_prompt = "Type 'q' and press Enter to cancel and exit."
    if config.CLI_MODE:
        logger.info(cancel_prompt)
    else:
        print_warning(
            f"\n⚠ Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
        )
        print_info(
            f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} for daily reset "
            f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m remaining)"
        )
        print_info(cancel_prompt)

    try:
        # Sleep in smaller intervals to allow for interruption
        sleep_interval = 1  # Check every second for responsiveness
        elapsed = 0

        while elapsed < seconds_until_reset:
            if _user_requested_cancel():
                logger.info("Wait cancelled by user ('q').")
                if not config.CLI_MODE:
                    print_warning("\nWait cancelled by user.")
                return False

            interval = min(sleep_interval, max(0, seconds_until_reset - elapsed))
            time.sleep(interval)
            elapsed += interval

            # Re-check if it's a new day
            if not token_tracker.is_limit_reached():
                logger.info("Token limit has been reset. Resuming processing.")
                if not config.CLI_MODE:
                    print_success("Token limit has been reset. Resuming processing.")

        logger.info("Token limit has been reset. Resuming processing.")
        if not config.CLI_MODE:
            print_success("\nToken limit has been reset. Resuming processing.")
        return True
        
    except KeyboardInterrupt:
        logger.info("Wait cancelled by user (KeyboardInterrupt).")
        if not config.CLI_MODE:
            print_warning("\nWait cancelled by user.")
        return False


def _user_requested_cancel() -> bool:
    """Check if the user requested cancellation by pressing 'q'."""
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            cancelled = False
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    continue
                if ch.lower() == "q":
                    cancelled = True
                # Consume remaining characters to avoid re-processing
            if cancelled:
                # Clear any trailing newline characters
                while msvcrt.kbhit():
                    _ = msvcrt.getwch()
                return True
            return False

        import select

        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line.lower() == "q":
                return True
        return False
    except Exception:
        # If any error occurs (e.g., select not supported), fall back to KeyboardInterrupt handling.
        return False


def main() -> None:
    args = setup_argparse()
    
    # Determine input and output paths based on mode
    if config.CLI_MODE:
        # CLI mode: use command line arguments
        input_path_arg = Path(args.input)
        base_output_dir = Path(args.output)
        process_all = args.all
        
        # Resolve relative paths to absolute
        if not input_path_arg.is_absolute():
            input_path_arg = Path.cwd() / input_path_arg
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir
        
        logger.info(f"CLI Mode: Input={input_path_arg}, Output={base_output_dir}")
    else:
        # Interactive mode: use config defaults and prompts
        input_path_arg = Path(args.input)
        base_output_dir = Path(config.OUTPUT_FOLDER_PATH)
        process_all = False
        
        print_header("AutoExcerpter - PDF & Image Transcription Tool")
    
    # Create output directory
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for items to process
    all_items_to_consider = scan_input_path(input_path_arg)
    if not all_items_to_consider:
        if config.CLI_MODE:
            logger.error(f"No items found to process in: {input_path_arg}")
            sys.exit(1)
        else:
            print_info("No items found to process. Please check your input path.")
            logger.debug("No items found in: %s", input_path_arg)
            sys.exit(0)

    # Select items based on mode
    if config.CLI_MODE:
        # CLI mode: process all items or just the single item
        if process_all or len(all_items_to_consider) == 1:
            selected_items = list(all_items_to_consider)
            logger.info(f"Processing {len(selected_items)} item(s) in CLI mode")
        else:
            # If multiple items found but --all not specified, process only the first
            selected_items = [all_items_to_consider[0]]
            logger.info(f"Processing first item (use --all to process all {len(all_items_to_consider)} items)")
    else:
        # Interactive mode: prompt user for selection
        selected_items = prompt_for_item_selection(all_items_to_consider)
        if not selected_items:
            print_info("No items selected for processing. Exiting.")
            sys.exit(0)

    logger.info("Selected %s item(s) for processing.", len(selected_items))
    
    if not config.CLI_MODE:
        print_section(f"Processing {len(selected_items)} Item(s)")
    
    # Display initial token usage statistics if enabled
    if config.DAILY_TOKEN_LIMIT_ENABLED:
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%) - "
            f"{stats['tokens_remaining']:,} tokens remaining today"
        )
        if not config.CLI_MODE:
            print_info(
                f"Daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
    
    # Process selected items sequentially (file-by-file to support token limit enforcement)
    processed_count = 0
    for index, item_spec in enumerate(selected_items, start=1):
        # Check token limit before starting each new file
        if not _check_and_wait_for_token_limit():
            # User cancelled wait - stop processing
            logger.info(f"Processing stopped by user. Processed {processed_count}/{len(selected_items)} items.")
            if not config.CLI_MODE:
                print_warning(f"\nProcessing stopped. Completed {processed_count}/{len(selected_items)} items.")
            break
        
        # Process this file
        success = _process_single_item(item_spec, index, len(selected_items), base_output_dir)
        if success:
            processed_count += 1
        
        # Log token usage after each file if enabled
        if config.DAILY_TOKEN_LIMIT_ENABLED:
            token_tracker = get_token_tracker()
            stats = token_tracker.get_stats()
            logger.info(
                f"Token usage after file {index}/{len(selected_items)}: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )

    # Final summary
    if config.CLI_MODE:
        logger.info(f"{processed_count}/{len(selected_items)} selected item(s) have been processed.")
    else:
        print_success(f"\n✓ {processed_count}/{len(selected_items)} selected item(s) have been processed!")
    logger.info(f"{processed_count}/{len(selected_items)} selected items have been processed.")
    
    # Final token usage statistics
    if config.DAILY_TOKEN_LIMIT_ENABLED:
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Final token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )
        if not config.CLI_MODE:
            print_info(
                f"\nFinal daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user (Ctrl+C). Exiting.")
        exit_program("\nProcessing interrupted by user.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("An unexpected critical error occurred in the main execution flow: %s", exc)
        print_error(f"An unexpected critical error occurred: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
