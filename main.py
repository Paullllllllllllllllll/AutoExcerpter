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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.resume import ProcessingState, ResumeChecker, ResumeResult
from core.transcriber import ItemTranscriber
from modules import app_config as config
from modules.error_handler import handle_critical_error
from modules.item_scanner import scan_input_path
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker
from modules.types import ItemSpec
from modules.user_prompts import (
    print_header,
    print_section,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_separator,
    print_dim,
    print_highlight,
    prompt_selection,
    prompt_yes_no,
    exit_program,
    Colors,
)

logger = setup_logger(__name__)

# Constants
MIN_VALID_CHOICE = 1
SELECTION_PROMPT = "\nEnter your choice(s) (e.g., 1; 3-5; all): "


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF and Image Folder Transcription and Summarization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Resume / overwrite behavior (available in both modes)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Skip files whose output already exists (default behavior). Overrides resume_mode in config.",
    )
    resume_group.add_argument(
        "--force", "--overwrite",
        action="store_true",
        dest="force",
        default=None,
        help="Force reprocessing of all files, overwriting existing output.",
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
        parser.add_argument(
            "--select",
            type=str,
            default=None,
            help="Select items by number (e.g., '1,3,5'), range (e.g., '1-5'), or filename pattern (e.g., 'Mennell').",
        )
        parser.add_argument(
            "--context",
            type=str,
            default=None,
            help="Summary context: topics to focus on during summarization (e.g., 'Food History, Wages, Early Modern').",
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




def prompt_for_item_selection(items: Sequence[ItemSpec]) -> List[ItemSpec]:
    """Prompt user to select items to process with improved UX."""
    if not items:
        print_warning("No processable PDF files or image folders found in the input.")
        return []

    print_section("Item Selection")
    print_info(f"Found {len(items)} item(s) available for processing")
    print()
    
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




def _process_single_item(
    item_spec: ItemSpec,
    index: int,
    total_items: int,
    base_output_dir: Path,
    summary_context: Optional[str] = None,
    resume_mode: str = "skip",
    completed_page_indices: Optional[set] = None,
) -> bool:
    """
    Process a single PDF or image folder item.
    
    Args:
        item_spec: Specification of the item to process.
        index: Current item index (1-based).
        total_items: Total number of items to process.
        base_output_dir: Base directory for outputs.
        summary_context: Optional context for guiding summarization focus.
        resume_mode: Resume mode ("skip" or "overwrite").
        completed_page_indices: Set of page indices already completed (for page-level resume).
        
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
        print()
        print_separator(char="=")
        print_highlight(f"  Processing Item {index}/{total_items}")
        print_separator(char="=")
        print_info(f"    • Name: {item_spec.output_stem}")
        print_info(f"    • Type: {item_spec.kind.replace('_', ' ').title()}")
        if completed_page_indices:
            print_dim(f"    • Resuming: {len(completed_page_indices)} page(s) already transcribed")
        print()

    # Process the item
    transcriber_instance: Optional[ItemTranscriber] = None
    try:
        transcriber_instance = ItemTranscriber(
            item_spec.path, item_spec.kind, base_output_dir,
            summary_context=summary_context,
            resume_mode=resume_mode,
            completed_page_indices=completed_page_indices,
        )
        transcriber_instance.process_item()
        
        if config.CLI_MODE:
            logger.info(f"Successfully processed: {item_spec.output_stem}")
        else:
            print_success(f"Successfully processed: {item_spec.output_stem}")
        return True
        
    except Exception as exc:
        handle_critical_error(
            exc,
            f"processing item '{item_spec.output_stem}'",
            exit_on_error=False,
            show_user_message=not config.CLI_MODE,
        )
        logger.info("--- Attempting to continue with the next item if any. ---")
        return False
        
    finally:
        # Clean up temporary working directory if configured
        # In resume mode ("skip"), retain working dir to preserve JSONL logs for future resumes
        should_cleanup_working_dir = config.DELETE_TEMP_WORKING_DIR and resume_mode != "skip"
        if transcriber_instance and should_cleanup_working_dir:
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
    
    # Token limit reached - display information and wait
    stats = token_tracker.get_stats()
    reset_time = token_tracker.get_reset_time()
    seconds_until_reset = token_tracker.get_seconds_until_reset()
    hours = seconds_until_reset // 3600
    minutes = (seconds_until_reset % 3600) // 60
    
    _log_token_limit_reached(stats, reset_time, hours, minutes)
    
    try:
        return _wait_for_token_reset(token_tracker, seconds_until_reset)
    except KeyboardInterrupt:
        logger.info("Wait cancelled by user (KeyboardInterrupt).")
        if not config.CLI_MODE:
            print_warning("\nWait cancelled by user.")
        return False


def _log_token_limit_reached(stats: Dict[str, Any], reset_time, hours: int, minutes: int) -> None:
    """Log token limit reached message to appropriate output."""
    logger.warning(
        f"Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    logger.info(f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} ({hours}h {minutes}m) for token limit reset...")
    
    if config.CLI_MODE:
        logger.info("Type 'q' and press Enter to cancel and exit.")
    else:
        print()
        print_separator(char="=")
        print_warning(f"  Daily Token Limit Reached")
        print_separator(char="=")
        print_info(f"    • Tokens used: {stats['tokens_used_today']:,}/{stats['daily_limit']:,}")
        print_info(f"    • Reset time: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"    • Time remaining: {hours}h {minutes}m")
        print()
        print_dim("  Waiting for token limit reset...")
        print_dim("  Type 'q' and press Enter to cancel and exit.")
        print()


def _wait_for_token_reset(token_tracker, seconds_until_reset: int) -> bool:
    """Wait for token limit reset with cancellation support."""
    elapsed = 0
    sleep_interval = 1  # Check every second for responsiveness

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
            return True

    logger.info("Token limit has been reset. Resuming processing.")
    if not config.CLI_MODE:
        print_success("\nToken limit has been reset. Resuming processing.")
    return True


def _parse_execution_mode(args: argparse.Namespace) -> Tuple[Path, Path, bool, Optional[str], Optional[str], str]:
    """Parse execution mode and return input path, output path, process_all flag, select pattern, context, and resume mode."""
    # Resolve resume mode from CLI flags (default: "skip")
    if getattr(args, "force", None):
        resume_mode = "overwrite"
    else:
        resume_mode = "skip"

    if config.CLI_MODE:
        # CLI mode: use command line arguments
        input_path_arg = Path(args.input)
        base_output_dir = Path(args.output)
        process_all = args.all
        select_pattern = args.select
        summary_context = args.context
        
        # Resolve relative paths to absolute
        if not input_path_arg.is_absolute():
            input_path_arg = Path.cwd() / input_path_arg
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir
        
        logger.info(f"CLI Mode: Input={input_path_arg}, Output={base_output_dir}")
        if summary_context:
            logger.info(f"CLI Mode: Summary context={summary_context}")
    else:
        # Interactive mode: use config defaults and prompts
        input_path_arg = Path(args.input)
        base_output_dir = Path(config.OUTPUT_FOLDER_PATH)
        process_all = False
        select_pattern = None
        summary_context = None
    
    return input_path_arg, base_output_dir, process_all, select_pattern, summary_context, resume_mode


def _select_items_for_processing(
    all_items: List[ItemSpec], process_all: bool, select_pattern: Optional[str] = None
) -> List[ItemSpec]:
    """Select items for processing based on mode and user input."""
    if config.CLI_MODE:
        # CLI mode: process all items, use select pattern, or just the single item
        if process_all:
            logger.info(f"Processing all {len(all_items)} item(s) in CLI mode")
            return list(all_items)
        elif select_pattern:
            # Use select pattern to filter items
            selected = _parse_cli_selection(all_items, select_pattern)
            if selected:
                logger.info(f"Processing {len(selected)} item(s) matching '{select_pattern}'")
                return selected
            else:
                logger.error(f"No items found matching '{select_pattern}'")
                return []
        elif len(all_items) == 1:
            logger.info(f"Processing single item in CLI mode")
            return list(all_items)
        else:
            # If multiple items found but --all/--select not specified, process only the first
            logger.info(f"Processing first item (use --all or --select to process specific items)")
            return [all_items[0]]
    else:
        # Interactive mode: prompt user for selection
        return prompt_for_item_selection(all_items) or []


def _parse_cli_selection(items: List[ItemSpec], pattern: str) -> List[ItemSpec]:
    """
    Parse CLI selection pattern and return matching items.
    
    Args:
        items: List of available items
        pattern: Selection pattern (numbers, ranges, or filename search)
        
    Returns:
        List of selected items
    """
    selected_indices: set[int] = set()
    pattern = pattern.strip()
    
    # Check if pattern looks like numeric selection
    numeric_pattern = pattern.replace(" ", "").replace(";", ",")
    is_numeric = all(c.isdigit() or c in ",-" for c in numeric_pattern) and any(c.isdigit() for c in numeric_pattern)
    
    if not is_numeric:
        # Filename search
        search_lower = pattern.lower()
        for idx, item in enumerate(items):
            item_name = item.path.name.lower()
            if search_lower in item_name:
                selected_indices.add(idx)
    else:
        # Numeric selection
        parts = numeric_pattern.split(",")
        for part in parts:
            if not part:
                continue
            if "-" in part:
                try:
                    start_str, end_str = part.split("-", 1)
                    start = int(start_str)
                    end = int(end_str)
                    if 1 <= start <= end <= len(items):
                        selected_indices.update(range(start - 1, end))
                except ValueError:
                    pass
            elif part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(items):
                    selected_indices.add(idx)
    
    return [items[i] for i in sorted(selected_indices)]


def _display_processing_summary(
    selected_items: List[ItemSpec],
    base_output_dir: Path,
    summary_context: Optional[str] = None,
) -> bool:
    """Display detailed processing summary and ask for confirmation.
    
    Args:
        selected_items: List of items selected for processing
        base_output_dir: Output directory path
        summary_context: Optional summary context for focused summarization
        
    Returns:
        True if user confirms, False to cancel
    """
    from modules.config_loader import get_config_loader
    
    # Load configurations
    config_loader = get_config_loader()
    model_config = config_loader.get_model_config()
    concurrency_config = config_loader.get_concurrency_config()
    
    print_header("PROCESSING SUMMARY")
    print_info("Review your selections before processing")
    print()
    
    # Count PDFs and image folders
    pdf_count = sum(1 for item in selected_items if item.kind == "pdf")
    image_folder_count = sum(1 for item in selected_items if item.kind == "image_folder")
    
    # Display item summary
    if pdf_count > 0 and image_folder_count > 0:
        item_summary = f"item(s) ({pdf_count} PDF, {image_folder_count} image folder)"
    elif pdf_count > 0:
        item_summary = "PDF file(s)"
    else:
        item_summary = "image folder(s)"
    
    print(f"  Ready to process {Colors.BOLD}{Colors.OKCYAN}{len(selected_items)}{Colors.ENDC} {item_summary}")
    print()
    
    # === Processing Configuration ===
    print_highlight("  Processing Configuration:")
    print_separator()
    
    # Document types
    if pdf_count > 0 and image_folder_count > 0:
        print_info(f"    • Document types: PDFs and Image Folders")
    elif pdf_count > 0:
        print_info(f"    • Document type: PDFs")
    else:
        print_info(f"    • Document type: Image Folders")
    
    # Summarization
    if config.SUMMARIZE:
        print_info(f"    • Summarization: Enabled")
        if summary_context:
            # Truncate long context for display
            display_context = summary_context if len(summary_context) <= 60 else summary_context[:57] + "..."
            print_dim(f"      - Focus topics: {display_context}")
        else:
            print_dim(f"      - Focus topics: Auto-resolved from context files")
    else:
        print_info(f"    • Summarization: Disabled (transcription only)")
    
    print_separator()
    
    # === Model Configuration ===
    print()
    print_highlight("  Model Configuration:")
    print_separator()
    
    # Transcription model
    trans_model = model_config.get("transcription_model", {})
    trans_provider = trans_model.get("provider", "openai").upper()
    trans_model_name = trans_model.get("name", "gpt-5-mini")
    trans_temp = trans_model.get("temperature")
    trans_max_tokens = trans_model.get("max_output_tokens", 16000)
    
    print_info(f"    • Transcription Provider: {trans_provider}")
    print_info(f"    • Transcription Model: {trans_model_name}")
    if trans_temp is not None:
        print_dim(f"      - Temperature: {trans_temp}")
    print_dim(f"      - Max output tokens: {trans_max_tokens:,}")
    
    # Reasoning configuration
    trans_reasoning = trans_model.get("reasoning", {})
    if trans_reasoning:
        effort = trans_reasoning.get("effort", "medium")
        print_dim(f"      - Reasoning effort: {effort}")
    
    # Text verbosity (OpenAI GPT-5 specific)
    trans_text = trans_model.get("text", {})
    if trans_text:
        verbosity = trans_text.get("verbosity", "medium")
        print_dim(f"      - Text verbosity: {verbosity}")
    
    # Summary model (if enabled)
    if config.SUMMARIZE:
        print()
        sum_model = model_config.get("summary_model", {})
        sum_provider = sum_model.get("provider", "openai").upper()
        sum_model_name = sum_model.get("name", "gpt-5-mini")
        sum_temp = sum_model.get("temperature")
        sum_max_tokens = sum_model.get("max_output_tokens", 16384)
        
        print_info(f"    • Summary Provider: {sum_provider}")
        print_info(f"    • Summary Model: {sum_model_name}")
        if sum_temp is not None:
            print_dim(f"      - Temperature: {sum_temp}")
        print_dim(f"      - Max output tokens: {sum_max_tokens:,}")
        
        # Reasoning configuration
        sum_reasoning = sum_model.get("reasoning", {})
        if sum_reasoning:
            effort = sum_reasoning.get("effort", "medium")
            print_dim(f"      - Reasoning effort: {effort}")
        
        # Text verbosity
        sum_text = sum_model.get("text", {})
        if sum_text:
            verbosity = sum_text.get("verbosity", "low")
            print_dim(f"      - Text verbosity: {verbosity}")
    
    print_separator()
    
    # === Concurrency Configuration ===
    print()
    print_highlight("  Concurrency Configuration:")
    print_separator()
    
    # Image processing
    img_proc = concurrency_config.get("image_processing", {})
    img_concurrency = img_proc.get("concurrency_limit", 24)
    print_info(f"    • Image extraction: {img_concurrency} concurrent tasks")
    
    # API requests
    api_requests = concurrency_config.get("api_requests", {})
    trans_api = api_requests.get("transcription", {})
    trans_concurrency = trans_api.get("concurrency_limit", 5)
    trans_service_tier = trans_api.get("service_tier", "default")
    print_info(f"    • Transcription API: {trans_concurrency} concurrent requests")
    print_dim(f"      - Service tier: {trans_service_tier}")
    
    if config.SUMMARIZE:
        sum_api = api_requests.get("summary", {})
        sum_concurrency = sum_api.get("concurrency_limit", 5)
        sum_service_tier = sum_api.get("service_tier", "flex")
        print_info(f"    • Summary API: {sum_concurrency} concurrent requests")
        print_dim(f"      - Service tier: {sum_service_tier}")
    
    # Retry configuration
    retry_config = concurrency_config.get("retry", {})
    max_attempts = retry_config.get("max_attempts", 5)
    print_dim(f"      - Max retry attempts: {max_attempts}")
    
    print_separator()
    
    # === Output Location ===
    print()
    print_highlight("  Output Location:")
    print_separator()
    if config.INPUT_PATHS_IS_OUTPUT_PATH:
        print_info("    • Output: Same directory as input files")
    else:
        print_info(f"    • Output directory: {base_output_dir}")
    print_separator()
    
    # === Token Limit ===
    if config.DAILY_TOKEN_LIMIT_ENABLED:
        print()
        print_highlight("  Daily Token Limit:")
        print_separator()
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        print_info(f"    • Current usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} ({stats['usage_percentage']:.1f}%)")
        print_info(f"    • Remaining today: {stats['tokens_remaining']:,} tokens")
        print_separator()
    
    # === Selected Items ===
    print()
    print_highlight("  Selected Items (first 5 shown):")
    for i, item in enumerate(selected_items[:5], 1):
        print_dim(f"    {i}. {item.display_label()}")
    
    if len(selected_items) > 5:
        print_dim(f"    ... and {len(selected_items) - 5} more")
    
    print()
    
    # Prompt for confirmation
    return prompt_yes_no(
        "Proceed with processing?",
        default=True,
        allow_exit=True
    )


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


def _display_completion_summary(processed_count: int, total_count: int, output_dir: Optional[Path]) -> None:
    """Display completion summary with statistics.
    
    Args:
        processed_count: Number of successfully processed items
        total_count: Total number of selected items
        output_dir: Output directory (None if co-located with input)
    """
    print()
    print_header("PROCESSING COMPLETE")
    
    if processed_count == total_count:
        print_success(f"All {processed_count} item(s) processed successfully!")
    else:
        print_warning(f"Processed {processed_count} out of {total_count} item(s)")
        print_info(f"{total_count - processed_count} item(s) failed or were skipped")
    
    print()
    print_highlight("  Output Files:")
    print_separator()
    
    if output_dir:
        print_info(f"    • Location: {output_dir}")
    else:
        print_info(f"    • Location: Same directory as input files")
    
    print_info(f"    • Transcriptions: .txt files")
    if config.SUMMARIZE:
        print_info(f"    • Summaries: .docx files")
    
    print_separator()
    
    # Token usage statistics
    if config.DAILY_TOKEN_LIMIT_ENABLED:
        print()
        print_highlight("  Token Usage:")
        print_separator()
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        print_info(f"    • Total used today: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} ({stats['usage_percentage']:.1f}%)")
        print_info(f"    • Remaining today: {stats['tokens_remaining']:,} tokens")
        print_separator()
    
    print()
    print_highlight("  Thank you for using AutoExcerpter!")
    print()


def _prompt_for_summary_context() -> Optional[str]:
    """Prompt user for optional summary context in interactive mode.
    
    Returns:
        Summary context string or None if skipped.
    """
    if not config.SUMMARIZE:
        return None
    
    print()
    print_info("You can optionally provide topics to focus on during summarization.")
    print_dim("  Example: 'Food History, Wages, Early Modern History'")
    print_dim("  Press Enter to skip and use automatic context resolution.")
    print()
    
    try:
        context_input = input(f"{Colors.PROMPT}Summary context (or Enter to skip): {Colors.ENDC}").strip()
        if context_input:
            print_success(f"Using summary context: {context_input}")
            return context_input
        else:
            print_dim("  No context provided. Will use file/folder/general context if available.")
            return None
    except (KeyboardInterrupt, EOFError):
        return None


def _resolve_item_output_dir(item_spec: ItemSpec, base_output_dir: Path) -> Path:
    """Resolve the output directory for a single item.
    
    Args:
        item_spec: The item specification.
        base_output_dir: The base output directory.
        
    Returns:
        The resolved output directory path.
    """
    if config.INPUT_PATHS_IS_OUTPUT_PATH:
        return item_spec.path.parent
    return base_output_dir


def _display_resume_info(
    resume_mode: str,
    selected_items: List[ItemSpec],
    skipped: List[ResumeResult],
    to_process: List[ItemSpec],
) -> bool:
    """Display resume information and handle the all-skipped case.
    
    Args:
        resume_mode: Current resume mode ("skip" or "overwrite").
        selected_items: Original list of selected items before filtering.
        skipped: List of ResumeResult for skipped items.
        to_process: List of items that will be processed.
        
    Returns:
        True if processing should continue, False if user cancelled or nothing to do.
    """
    if not skipped:
        return True

    total_selected = len(selected_items)
    new_count = len(to_process)

    if config.CLI_MODE:
        logger.info(
            f"Resume: skipping {len(skipped)} already-processed item(s), "
            f"{new_count} item(s) to process"
        )
        for sr in skipped:
            logger.info("  Skipped: %s — %s", sr.item_name, sr.reason)
    else:
        print()
        print_highlight("  Resume Information:")
        print_separator()
        print_warning(
            f"    • {len(skipped)} of {total_selected} item(s) already have output and will be skipped"
        )
        print_info(f"    • {new_count} item(s) will be processed")
        print_dim(f"    • Resume mode: {resume_mode}")
        print_dim("    • Use --force / --overwrite to reprocess all items")
        print_separator()

    if new_count == 0:
        if config.CLI_MODE:
            logger.info("All items already processed. Nothing to do. Use --force to reprocess.")
            return False
        else:
            print_warning("All items already processed. Nothing to do.")
            force = prompt_yes_no(
                "Force reprocess all items?",
                default=False,
                allow_exit=True,
            )
            return force

    return True


def main() -> None:
    args = setup_argparse()
    
    # Determine input and output paths based on mode
    input_path_arg, base_output_dir, process_all, select_pattern, summary_context, resume_mode = _parse_execution_mode(args)
    
    if not config.CLI_MODE:
        print_header(
            "AUTOEXCERPTER",
            "PDF & Image Transcription and Summarization Tool"
        )
        print_info("Transform PDFs and images into structured transcriptions and summaries")
        print_info("using state-of-the-art AI models.")
        print()
    
    # Create output directory
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for items to process
    if not config.CLI_MODE:
        print_section("Scanning Input Directory")
        print_info(f"Searching for PDFs and image folders in: {input_path_arg}")
    
    all_items_to_consider = scan_input_path(input_path_arg)
    
    if not all_items_to_consider:
        if config.CLI_MODE:
            logger.error(f"No items found to process in: {input_path_arg}")
            sys.exit(1)
        else:
            print_error("No items found to process. Please check your input path.")
            logger.debug("No items found in: %s", input_path_arg)
            sys.exit(0)
    
    if not config.CLI_MODE:
        print_success(f"Found {len(all_items_to_consider)} item(s) available for processing")

    # Select items based on mode
    selected_items = _select_items_for_processing(
        all_items_to_consider, process_all, select_pattern
    )
    if not selected_items:
        if not config.CLI_MODE:
            print_info("No items selected for processing. Exiting.")
        sys.exit(0)

    logger.info("Selected %s item(s) for processing.", len(selected_items))

    # --- Resume filtering ---
    resume_checker = ResumeChecker(
        resume_mode=resume_mode,
        summarize=config.SUMMARIZE,
        output_docx=config.OUTPUT_DOCX,
        output_markdown=config.OUTPUT_MARKDOWN,
    )
    items_to_process, skipped_items = resume_checker.filter_items(
        items=selected_items,
        output_dir_func=lambda item: _resolve_item_output_dir(item, base_output_dir),
        name_func=lambda item: item.output_stem,
    )

    # Build per-item resume info map for items that will be processed
    # (includes PARTIAL and TRANSCRIPTION_ONLY states for page-level / summary-only resume)
    item_resume_map: Dict[str, ResumeResult] = {}
    for item in items_to_process:
        result = resume_checker.should_skip(
            item.output_stem,
            _resolve_item_output_dir(item, base_output_dir),
        )
        if result.state in (ProcessingState.PARTIAL, ProcessingState.TRANSCRIPTION_ONLY):
            item_resume_map[item.output_stem] = result

    # Log resume mode
    if resume_mode == "skip":
        logger.info("Resume mode: skip (use --force to reprocess all)")
    else:
        logger.info("Resume mode: overwrite (all files will be reprocessed)")

    # Display resume information and handle all-skipped case
    should_continue = _display_resume_info(resume_mode, selected_items, skipped_items, items_to_process)
    if not should_continue:
        # User declined to force-reprocess, or CLI with nothing to do
        if not config.CLI_MODE:
            print_info("No items to process. Exiting.")
        sys.exit(0)

    # If user chose to force-reprocess from the prompt, use original selection
    if not items_to_process and should_continue:
        items_to_process = list(selected_items)
        resume_mode = "overwrite"
        item_resume_map.clear()
    # --- End resume filtering ---
    
    # Prompt for summary context in interactive mode (if summarization enabled and not already set)
    if not config.CLI_MODE and config.SUMMARIZE and not summary_context:
        summary_context = _prompt_for_summary_context()
    
    # Display processing summary and get confirmation (interactive mode only)
    if not config.CLI_MODE:
        confirmed = _display_processing_summary(items_to_process, base_output_dir, summary_context)
        if not confirmed:
            print_info("Processing cancelled by user.")
            sys.exit(0)
        
        print_section(f"Processing {len(items_to_process)} Item(s)")
    
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
    total_to_process = len(items_to_process)
    for index, item_spec in enumerate(items_to_process, start=1):
        # Check token limit before starting each new file
        if not _check_and_wait_for_token_limit():
            # User cancelled wait - stop processing
            logger.info(f"Processing stopped by user. Processed {processed_count}/{total_to_process} items.")
            if not config.CLI_MODE:
                print_warning(f"\nProcessing stopped. Completed {processed_count}/{total_to_process} items.")
            break
        
        # Determine output directory for this item
        item_output_dir = _resolve_item_output_dir(item_spec, base_output_dir)
        item_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Look up per-item resume info (for page-level resume)
        item_resume = item_resume_map.get(item_spec.output_stem)
        completed_pages = item_resume.completed_page_indices if item_resume else None

        # Process this file
        success = _process_single_item(
            item_spec, index, total_to_process, item_output_dir,
            summary_context=summary_context,
            resume_mode=resume_mode,
            completed_page_indices=completed_pages,
        )
        if success:
            processed_count += 1
        
        # Log token usage after each file if enabled
        if config.DAILY_TOKEN_LIMIT_ENABLED:
            token_tracker = get_token_tracker()
            stats = token_tracker.get_stats()
            logger.info(
                f"Token usage after file {index}/{total_to_process}: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )

    # Final summary
    skipped_count = len(skipped_items)
    if config.CLI_MODE:
        msg = f"{processed_count}/{total_to_process} item(s) processed."
        if skipped_count > 0:
            msg += f" {skipped_count} item(s) skipped (already complete)."
        logger.info(msg)
    else:
        _display_completion_summary(processed_count, total_to_process, base_output_dir if not config.INPUT_PATHS_IS_OUTPUT_PATH else None)
        if skipped_count > 0:
            print_dim(f"  ({skipped_count} item(s) were skipped as already complete)")
            print()
    
    logger.info(f"{processed_count}/{total_to_process} selected items have been processed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user (Ctrl+C). Exiting.")
        exit_program("\nProcessing interrupted by user.")
    except Exception as exc:
        handle_critical_error(
            exc,
            "main execution flow",
            exit_on_error=True,
            show_user_message=True,
        )
