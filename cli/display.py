"""CLI display and user interaction utilities."""

from __future__ import annotations

import collections.abc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from cli.interaction import (
    Colors,
    print_dim,
    print_header,
    print_highlight,
    print_info,
    print_section,
    print_separator,
    print_success,
    print_warning,
    prompt_selection,
    prompt_yes_no,
)
from config import app as config
from config.constants import DEFAULT_CONCURRENT_REQUESTS
from config.logger import setup_logger
from llm.token_tracker import _describe_reset_time, get_token_tracker
from pipeline.resume import ResumeResult
from pipeline.types import ItemSpec

logger = setup_logger(__name__)


def _fmt_int(value: Any) -> str:
    """Format a value with thousands separators, falling back to ``str()``.

    A hand-edited model.yaml can carry ``max_output_tokens`` as a quoted string;
    ``f"{value:,}"`` would then raise ValueError before processing even starts.
    Coerce to int when possible, otherwise print the raw value un-formatted.
    """
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


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
    from config.loader import get_config_loader

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
        f"  Ready to process {Colors.BOLD}{Colors.OKCYAN}"
        f"{len(selected_items)}{Colors.ENDC} {item_summary}"
    )
    print()

    # === Processing Configuration ===
    print_highlight("  Processing Configuration:")
    print_separator()

    # Document types
    if pdf_count > 0 and image_folder_count > 0:
        print_info("    • Document types: PDFs and Image Folders")
    elif pdf_count > 0:
        print_info("    • Document type: PDFs")
    else:
        print_info("    • Document type: Image Folders")

    # Summarization
    if config.SUMMARIZE:
        print_info("    • Summarization: Enabled")
        if summary_context:
            # Truncate long context for display
            display_context = (
                summary_context
                if len(summary_context) <= 60
                else summary_context[:57] + "..."
            )
            print_dim(f"      - Focus topics: {display_context}")
        else:
            print_dim("      - Focus topics: Auto-resolved from context files")
    else:
        print_info("    • Summarization: Disabled (transcription only)")

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
    trans_max_tokens = trans_model.get("max_output_tokens", 16384)

    print_info(f"    • Transcription Provider: {trans_provider}")
    print_info(f"    • Transcription Model: {trans_model_name}")
    if trans_temp is not None:
        print_dim(f"      - Temperature: {trans_temp}")
    print_dim(f"      - Max output tokens: {_fmt_int(trans_max_tokens)}")

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
        print_dim(f"      - Max output tokens: {_fmt_int(sum_max_tokens)}")

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

    # API requests. Summaries run inline within the transcription workers and
    # share their concurrency, so only the transcription phase is reported here.
    api_requests = concurrency_config.get("api_requests", {})
    trans_api = api_requests.get("transcription", {})
    trans_concurrency = trans_api.get("concurrency_limit", DEFAULT_CONCURRENT_REQUESTS)
    trans_service_tier = trans_api.get("service_tier", "flex")
    print_info(f"    • Transcription API: {trans_concurrency} concurrent requests")
    print_dim(f"      - Service tier: {trans_service_tier}")

    if config.SUMMARIZE:
        sum_api = api_requests.get("summary", {})
        sum_service_tier = sum_api.get("service_tier", trans_service_tier)
        print_dim(f"      - Summary service tier: {sum_service_tier}")

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
            f"    • Current usage: {stats['tokens_used_today']:,}"
            f"/{stats['daily_limit']:,} ({stats['usage_percentage']:.1f}%)"
        )
        print_info(f"    • Remaining today: {stats['tokens_remaining']:,} tokens")
        _print_shared_budget_breakdown(stats)
        _print_pool_budget_breakdown(stats)
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


def _summarize_reports(
    reports: list[tuple[str, dict[str, Any] | None]] | None,
) -> tuple[int, int, int, int]:
    """Sum per-item page counters across the run's reports.

    Returns ``(pages_ok, pages_failed, pages_deferred, summary_failures)``;
    items with a missing report (None) contribute nothing.
    """
    pages_ok = pages_failed = pages_deferred = summary_failures = 0
    for _name, rep in reports or []:
        if not rep:
            continue
        pages_ok += int(rep.get("pages_ok", 0) or 0)
        pages_failed += int(rep.get("pages_failed", 0) or 0)
        pages_deferred += int(rep.get("pages_deferred", 0) or 0)
        summary_failures += int(rep.get("summary_failures", 0) or 0)
    return pages_ok, pages_failed, pages_deferred, summary_failures


def _print_cli_completion_overview(
    processed_count: int,
    total_count: int,
    *,
    skipped_count: int,
    unattempted_count: int,
    reports: list[tuple[str, dict[str, Any] | None]] | None,
    run_seconds: float | None,
    tokens_used_run: int | None,
    run_outputs: list[str] | None,
) -> None:
    """Print a concise final overview to stderr for CLI mode.

    The existing logger.info summary sits below the WARNING console threshold
    and is therefore invisible; this block (a real stderr print) surfaces the
    same numbers so an operator sees the run outcome without raising log levels.
    """
    failed = total_count - processed_count - unattempted_count
    pages_ok, pages_failed, pages_deferred, summary_failures = _summarize_reports(
        reports
    )
    lines = [
        "Run complete:",
        f"  items: {processed_count} processed, {failed} failed, "
        f"{unattempted_count} not attempted, {skipped_count} skipped "
        f"(of {total_count})",
        f"  pages: {pages_ok} ok, {pages_failed} failed, {pages_deferred} deferred"
        + (f", {summary_failures} summary failure(s)" if summary_failures else ""),
    ]
    if run_seconds is not None:
        lines.append(f"  run time: {run_seconds:.1f}s")
    if tokens_used_run is not None:
        lines.append(f"  tokens this run: {tokens_used_run:,}")
    if run_outputs:
        lines.append(f"  outputs ({len(run_outputs)}):")
        for out in run_outputs[:20]:
            lines.append(f"    {out}")
        if len(run_outputs) > 20:
            lines.append(f"    ... and {len(run_outputs) - 20} more")
    for line in lines:
        print(line, file=sys.stderr)


def _display_completion_summary(
    processed_count: int,
    total_count: int,
    output_dir: Path | None,
    *,
    skipped_count: int = 0,
    unattempted_count: int = 0,
    reports: list[tuple[str, dict[str, Any] | None]] | None = None,
    run_seconds: float | None = None,
    tokens_used_run: int | None = None,
    run_outputs: list[str] | None = None,
) -> None:
    """Display completion summary with statistics.

    Args:
        processed_count: Number of successfully processed items
        total_count: Total number of items attempted (items_to_process)
        output_dir: Output directory (None if co-located with input)
        skipped_count: Items skipped as already complete
        unattempted_count: Items never reached (cancelled token wait)
        reports: Per-item ``(name, last_run_report)`` pairs for page aggregates
        run_seconds: Wall-clock time spent in the processing loop
        tokens_used_run: Tokens consumed this run (None when not tracked)
        run_outputs: Absolute paths of output files written this run
    """
    failed = total_count - processed_count - unattempted_count

    print()
    print_header("PROCESSING COMPLETE")

    if processed_count == total_count:
        print_success(f"All {processed_count} item(s) processed successfully!")
    else:
        print_warning(f"Processed {processed_count} out of {total_count} item(s)")
        detail = []
        if failed > 0:
            detail.append(f"{failed} failed")
        if unattempted_count > 0:
            detail.append(f"{unattempted_count} not attempted")
        if detail:
            # Skipped items are never part of total_count, so the old
            # "failed or were skipped" wording mislabeled this line.
            print_info("    " + "; ".join(detail))

    # === Run Overview ===
    print()
    print_highlight("  Run Overview:")
    print_separator()
    print_info(
        f"    • Items: {processed_count} processed, {failed} failed, "
        f"{unattempted_count} not attempted, {skipped_count} skipped"
    )
    pages_ok, pages_failed, pages_deferred, summary_failures = _summarize_reports(
        reports
    )
    print_info(
        f"    • Pages: {pages_ok} transcribed ok, {pages_failed} failed, "
        f"{pages_deferred} deferred"
    )
    if summary_failures:
        print_info(f"    • Summary failures: {summary_failures}")
    if run_seconds is not None:
        print_info(f"    • Total run time: {run_seconds:.1f}s")
    if tokens_used_run is not None:
        print_info(f"    • Tokens used this run: {tokens_used_run:,}")
    print_separator()

    if run_outputs:
        print()
        print_highlight("  Output Files Written This Run:")
        print_separator()
        for out in run_outputs[:20]:
            print_dim(f"    • {out}")
        if len(run_outputs) > 20:
            print_dim(f"    ... and {len(run_outputs) - 20} more")
        print_separator()

    print()
    print_highlight("  Output Files:")
    print_separator()

    if output_dir:
        print_info(f"    • Location: {output_dir}")
    else:
        print_info("    • Location: Same directory as input files")

    print_info("    • Transcriptions: .txt files")
    if config.SUMMARIZE:
        summary_formats = []
        if config.OUTPUT_DOCX:
            summary_formats.append(".docx")
        if config.OUTPUT_MARKDOWN:
            summary_formats.append(".md")
        if summary_formats:
            print_info(f"    • Summaries: {', '.join(summary_formats)} files")

    print_separator()

    # Token usage statistics
    if config.DAILY_TOKEN_LIMIT_ENABLED:
        print()
        print_highlight("  Token Usage:")
        print_separator()
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        print_info(
            f"    • Total used today: {stats['tokens_used_today']:,}"
            f"/{stats['daily_limit']:,} ({stats['usage_percentage']:.1f}%)"
        )
        print_info(f"    • Remaining today: {stats['tokens_remaining']:,} tokens")
        _print_shared_budget_breakdown(stats)
        _print_pool_budget_breakdown(stats)
        print_separator()

    print()
    print_highlight("  Thank you for using AutoExcerpter!")
    print()


def _print_pool_budget_breakdown(stats: dict[str, Any]) -> None:
    """Print per-key daily-pool caps and used/remaining, if any.

    A no-op when the per-key gate is disabled or no pooled bucket has been seen
    today, so runs without pooled keys are unchanged. A pool without a
    resolvable cap is shown as tracked but uncapped.
    """
    if not stats.get("per_key_pool_caps_enabled", False):
        return
    buckets = stats.get("pool_buckets") or []
    if not buckets:
        return
    print_dim("      - Per-key pool caps:")
    for row in buckets:
        cap = row.get("cap")
        if cap is None:
            print_dim(
                f"        · {row['key_env']} [{row['pool']}]: "
                f"{row['used']:,} used (uncapped)"
            )
        else:
            print_dim(
                f"        · {row['key_env']} [{row['pool']}]: "
                f"{row['used']:,}/{cap:,} "
                f"({row['remaining']:,} remaining)"
            )


def _print_shared_budget_breakdown(stats: dict[str, Any]) -> None:
    """Print the per-tool shared-ledger split when the shared budget is active.

    A no-op when the shared budget is disabled, so standalone runs are unchanged.
    The combined figure is already shown as the usage line above; here we surface
    this tool's own share and the cross-tool breakdown so a combined cap that is
    driven mostly by a sibling tool is legible.
    """
    if not stats.get("shared_budget_enabled"):
        return
    own = stats.get("own_tokens_used_today", 0)
    print_dim(f"      - This tool (autoexcerpter): {own:,} tokens")
    breakdown = stats.get("shared_breakdown") or {}
    if breakdown:
        parts = "; ".join(
            f"{name}: {value:,}" for name, value in sorted(breakdown.items())
        )
        print_dim(f"      - Shared ledger split: {parts}")
    if stats.get("shared_budget_degraded"):
        print_dim("      - Shared ledger degraded: using this tool's count only")


def _log_token_limit_reached(
    stats: dict[str, Any], reset_time: datetime, hours: int, minutes: int
) -> None:
    """Log token limit reached message to appropriate output."""
    if config.CLI_MODE:
        # CLI mode: the logger (stderr, WARNING+) is the only human channel.
        logger.warning(
            f"Daily token limit reached: {stats['tokens_used_today']:,}"
            f"/{stats['daily_limit']:,} tokens used"
        )
        logger.info(
            f"Waiting until {_describe_reset_time(reset_time)}"
            f" ({hours}h {minutes}m) for token limit reset..."
        )
        logger.info("Type 'q' and press Enter to cancel and exit.")
    else:
        # Interactive mode: the pretty block below is the user-facing channel;
        # the logger.warning/info would duplicate it on stderr, so it is gated
        # to CLI mode only.
        print()
        print_separator(char="=")
        print_warning("  Daily Token Limit Reached")
        print_separator(char="=")
        print_info(
            f"    • Tokens used: {stats['tokens_used_today']:,}"
            f"/{stats['daily_limit']:,}"
        )
        print_info(f"    • Reset time: {_describe_reset_time(reset_time)}")
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
                "  No context provided."
                " Will use file/folder/general context if available."
            )
            return None
    except KeyboardInterrupt:
        # Ctrl+C here must abort the (paid) run, not silently continue. Re-raise
        # so the __main__ handler exits with code 130.
        raise
    except EOFError:
        # Closed stdin on this optional prompt: skip context, keep going.
        return None


def _display_resume_info(
    resume_mode: str,
    selected_items: list[ItemSpec],
    skipped: list[ResumeResult],
    to_process: list[ItemSpec],
    dry_run: bool = False,
) -> bool:
    """Display resume information and handle the all-skipped case.

    Args:
        resume_mode: Current resume mode ("skip" or "overwrite").
        selected_items: Original list of selected items before filtering.
        skipped: List of ResumeResult for skipped items.
        to_process: List of items that will be processed.
        dry_run: When True, never prompt to force-reprocess; a dry run is
            side-effect-free and must not block on interactive input.

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
            f"    • {len(skipped)} of {total_selected} item(s)"
            " already have output and will be skipped"
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
            if dry_run:
                # A dry run reports the plan (all complete -> skip) without
                # prompting; short-circuit before the force-reprocess prompt.
                return False
            force = prompt_yes_no(
                "Force reprocess all items?",
                default=False,
                allow_exit=True,
            )
            return force

    return True
