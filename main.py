"""Multi-provider transcription and summarization pipeline.

This script:
1. Processes PDFs by extracting each page as an image, or processes images from a folder
2. Transcribes the images via configurable LLM providers (OpenAI, Anthropic, Google, OpenRouter)
3. Optionally sends the transcribed text to an LLM for structured summarization
4. Saves transcriptions to TXT and summaries to DOCX/Markdown
5. Verifies page numbering coherence
"""

from __future__ import annotations

import sys
from pathlib import Path

from cli.argument_parser import (
    setup_argparse,
    _build_cli_model_overrides,
    _parse_cli_selection,
    _parse_execution_mode,
)
from cli.display import (
    _display_completion_summary,
    _display_processing_summary,
    _display_resume_info,
    _prompt_for_summary_context,
    prompt_for_item_selection,
)
from cli.processing import (
    _check_and_wait_for_token_limit,
    _process_single_item,
)
from core.resume import ProcessingState, ResumeChecker
from modules import app_config as config
from modules.config_loader import get_config_loader
from modules.error_handler import handle_critical_error
from modules.item_scanner import scan_input_path
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker
from modules.types import ItemSpec
from modules.user_prompts import (
    print_section,
    print_success,
    print_warning,
    print_info,
    print_dim,
    exit_program,
)

logger = setup_logger(__name__)


def _select_items_for_processing(
    all_items: list[ItemSpec], process_all: bool, select_pattern: str | None = None
) -> list[ItemSpec]:
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
                logger.info(
                    f"Processing {len(selected)} item(s) matching '{select_pattern}'"
                )
                return selected
            else:
                logger.error(f"No items found matching '{select_pattern}'")
                return []
        elif len(all_items) == 1:
            logger.info(f"Processing single item in CLI mode")
            return list(all_items)
        else:
            # If multiple items found but --all/--select not specified, process only the first
            logger.info(
                f"Processing first item (use --all or --select to process specific items)"
            )
            return [all_items[0]]
    else:
        # Interactive mode: prompt user for selection
        return prompt_for_item_selection(all_items) or []


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


def main() -> None:
    args = setup_argparse()

    # Determine input and output paths based on mode
    (
        input_path_arg,
        base_output_dir,
        process_all,
        select_pattern,
        summary_context,
        resume_mode,
    ) = _parse_execution_mode(args)

    cli_model_overrides = _build_cli_model_overrides(args)
    if cli_model_overrides:
        get_config_loader().apply_model_overrides(cli_model_overrides)
        logger.info("CLI mode model overrides applied: %s", cli_model_overrides)

    if not config.CLI_MODE:
        from modules.user_prompts import print_header

        print_header(
            "AUTOEXCERPTER", "PDF & Image Transcription and Summarization Tool"
        )
        print_info(
            "Transform PDFs and images into structured transcriptions and summaries"
        )
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
            from modules.user_prompts import print_error

            print_error("No items found to process. Please check your input path.")
            logger.debug("No items found in: %s", input_path_arg)
            sys.exit(0)

    if not config.CLI_MODE:
        print_success(
            f"Found {len(all_items_to_consider)} item(s) available for processing"
        )

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
    item_resume_map: dict[str, object] = {}
    for item in items_to_process:
        result = resume_checker.should_skip(
            item.output_stem,
            _resolve_item_output_dir(item, base_output_dir),
        )
        if result.state in (
            ProcessingState.PARTIAL,
            ProcessingState.TRANSCRIPTION_ONLY,
        ):
            item_resume_map[item.output_stem] = result

    # Log resume mode
    if resume_mode == "skip":
        logger.info("Resume mode: skip (use --force to reprocess all)")
    else:
        logger.info("Resume mode: overwrite (all files will be reprocessed)")

    # Display resume information and handle all-skipped case
    should_continue = _display_resume_info(
        resume_mode, selected_items, skipped_items, items_to_process
    )
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
        confirmed = _display_processing_summary(
            items_to_process, base_output_dir, summary_context
        )
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
            logger.info(
                f"Processing stopped by user. Processed {processed_count}/{total_to_process} items."
            )
            if not config.CLI_MODE:
                print_warning(
                    f"\nProcessing stopped. Completed {processed_count}/{total_to_process} items."
                )
            break

        # Determine output directory for this item
        item_output_dir = _resolve_item_output_dir(item_spec, base_output_dir)
        item_output_dir.mkdir(parents=True, exist_ok=True)

        # Look up per-item resume info (for page-level resume)
        item_resume = item_resume_map.get(item_spec.output_stem)
        completed_pages = item_resume.completed_page_indices if item_resume else None

        # Process this file
        success = _process_single_item(
            item_spec,
            index,
            total_to_process,
            item_output_dir,
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
        _display_completion_summary(
            processed_count,
            total_to_process,
            base_output_dir if not config.INPUT_PATHS_IS_OUTPUT_PATH else None,
        )
        if skipped_count > 0:
            print_dim(f"  ({skipped_count} item(s) were skipped as already complete)")
            print()

    logger.info(
        f"{processed_count}/{total_to_process} selected items have been processed."
    )


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
