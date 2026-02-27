"""CLI display and user interaction utilities."""

from __future__ import annotations

import collections.abc
from pathlib import Path
from typing import Any

from core.resume import ResumeResult
from modules import app_config as config
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker
from modules.types import ItemSpec
from modules.user_prompts import (
    print_header,
    print_section,
    print_success,
    print_warning,
    print_info,
    print_separator,
    print_dim,
    print_highlight,
    prompt_selection,
    prompt_yes_no,
    Colors,
)

logger = setup_logger(__name__)


def prompt_for_item_selection(
    items: collections.abc.Sequence[ItemSpec],
) -> list[ItemSpec]:
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


def _display_processing_summary(
    selected_items: list[ItemSpec],
    base_output_dir: Path,
    summary_context: str | None = None,
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
    image_folder_count = sum(
        1 for item in selected_items if item.kind == "image_folder"
    )

    # Display item summary
    if pdf_count > 0 and image_folder_count > 0:
        item_summary = f"item(s) ({pdf_count} PDF, {image_folder_count} image folder)"
    elif pdf_count > 0:
        item_summary = "PDF file(s)"
    else:
        item_summary = "image folder(s)"

    print(
        f"  Ready to process {Colors.BOLD}{Colors.OKCYAN}{len(selected_items)}{Colors.ENDC} {item_summary}"
    )
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
            display_context = (
                summary_context
                if len(summary_context) <= 60
                else summary_context[:57] + "..."
            )
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
        print_info(
            f"    • Current usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} ({stats['usage_percentage']:.1f}%)"
        )
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
    return prompt_yes_no("Proceed with processing?", default=True, allow_exit=True)


def _display_completion_summary(
    processed_count: int, total_count: int, output_dir: Path | None
) -> None:
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
        print_info(
            f"    • Total used today: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} ({stats['usage_percentage']:.1f}%)"
        )
        print_info(f"    • Remaining today: {stats['tokens_remaining']:,} tokens")
        print_separator()

    print()
    print_highlight("  Thank you for using AutoExcerpter!")
    print()


def _log_token_limit_reached(
    stats: dict[str, Any], reset_time, hours: int, minutes: int
) -> None:
    """Log token limit reached message to appropriate output."""
    logger.warning(
        f"Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    logger.info(
        f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} ({hours}h {minutes}m) for token limit reset..."
    )

    if config.CLI_MODE:
        logger.info("Type 'q' and press Enter to cancel and exit.")
    else:
        print()
        print_separator(char="=")
        print_warning(f"  Daily Token Limit Reached")
        print_separator(char="=")
        print_info(
            f"    • Tokens used: {stats['tokens_used_today']:,}/{stats['daily_limit']:,}"
        )
        print_info(f"    • Reset time: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"    • Time remaining: {hours}h {minutes}m")
        print()
        print_dim("  Waiting for token limit reset...")
        print_dim("  Type 'q' and press Enter to cancel and exit.")
        print()


def _prompt_for_summary_context() -> str | None:
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
        context_input = input(
            f"{Colors.PROMPT}Summary context (or Enter to skip): {Colors.ENDC}"
        ).strip()
        if context_input:
            print_success(f"Using summary context: {context_input}")
            return context_input
        else:
            print_dim(
                "  No context provided. Will use file/folder/general context if available."
            )
            return None
    except (KeyboardInterrupt, EOFError):
        return None


def _display_resume_info(
    resume_mode: str,
    selected_items: list[ItemSpec],
    skipped: list[ResumeResult],
    to_process: list[ItemSpec],
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
            logger.info(
                "All items already processed. Nothing to do. Use --force to reprocess."
            )
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
