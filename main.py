"""Multi-provider transcription and summarization pipeline.

This script:
1. Processes PDFs by extracting each page as an image, or processes images from a folder
2. Transcribes the images via configurable LLM providers
   (OpenAI, Anthropic, Google, OpenRouter)
3. Optionally sends the transcribed text to an LLM for structured summarization
4. Saves transcriptions to TXT and summaries to DOCX/Markdown
5. Verifies page numbering coherence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cli.args import (
    _apply_app_config_overrides,
    _build_cli_model_overrides,
    _parse_cli_selection,
    _parse_execution_mode,
    setup_argparse,
)
from cli.display import (
    _display_completion_summary,
    _display_processing_summary,
    _display_resume_info,
    _prompt_for_summary_context,
    prompt_for_item_selection,
)
from cli.errors import handle_critical_error
from cli.interaction import (
    print_dim,
    print_info,
    print_section,
    print_success,
    print_warning,
)
from cli.loop import (
    _check_and_wait_for_token_limit,
    _process_single_item,
)
from config import app as config
from config.loader import get_config_loader
from config.logger import setup_logger
from llm.token_tracker import get_token_tracker
from pipeline import ProcessingState, ResumeChecker, ResumeResult, scan_input_path
from pipeline.types import ItemSpec

logger = setup_logger(__name__)


def _select_items_for_processing(
    all_items: list[ItemSpec],
    process_all: bool,
    select_pattern: str | None = None,
    emit_json: bool = False,
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
                # A --select pattern matching nothing is almost always a typo,
                # not a valid "process nothing" request. Exit non-zero (like the
                # empty-input-directory case) so a driving agent sees the error
                # rather than a false success, still emitting the JSON summary
                # per the "emit JSON on all exits" contract (AE-5).
                logger.error(f"No items found matching '{select_pattern}'")
                if emit_json:
                    _emit_json_summary(0, 0, 0, 0, [])
                sys.exit(1)
        elif len(all_items) == 1:
            logger.info("Processing single item in CLI mode")
            return list(all_items)
        else:
            # Multiple items with no --all/--select is ambiguous in CLI mode:
            # error out rather than silently processing only the first (which
            # loses the rest without any signal to a driving agent). Emit the
            # JSON summary first so this controlled exit honors the same
            # "emit JSON on all exits" contract as its neighbors (AE-4).
            logger.error(
                "%d items found but neither --all nor --select was given. "
                "Refusing to silently process only the first item; pass --all "
                "to process every item or --select to choose specific ones.",
                len(all_items),
            )
            if emit_json:
                _emit_json_summary(0, 0, 0, 0, [])
            sys.exit(2)
    else:
        # Interactive mode: prompt user for selection
        return prompt_for_item_selection(all_items) or []


def _resolve_item_output_dir(item_spec: ItemSpec, base_output_dir: Path) -> Path:
    """Resolve the output directory for a single item."""
    if config.INPUT_PATHS_IS_OUTPUT_PATH:
        return item_spec.path.parent
    return base_output_dir


def _setup_and_scan(
    args: argparse.Namespace,
) -> tuple[list[ItemSpec], Path, str | None, str]:
    """Parse args, apply overrides, scan input, select items.

    Returns:
        (selected_items, base_output_dir, summary_context, resume_mode)
    """
    (
        input_path_arg,
        base_output_dir,
        process_all,
        select_pattern,
        summary_context,
        resume_mode,
    ) = _parse_execution_mode(args)

    _apply_app_config_overrides(args)

    cli_model_overrides = _build_cli_model_overrides(args)
    if cli_model_overrides:
        get_config_loader().apply_model_overrides(cli_model_overrides)
        logger.info("CLI mode model overrides applied: %s", cli_model_overrides)

    if not config.CLI_MODE:
        from cli.interaction import print_header

        print_header(
            "AUTOEXCERPTER", "PDF & Image Transcription and Summarization Tool"
        )
        print_info(
            "Transform PDFs and images into structured transcriptions and summaries"
        )
        print_info("using state-of-the-art AI models.")
        print()

    base_output_dir.mkdir(parents=True, exist_ok=True)

    if not config.CLI_MODE:
        print_section("Scanning Input Directory")
        print_info(f"Searching for PDFs and image folders in: {input_path_arg}")

    all_items_to_consider = scan_input_path(input_path_arg)

    emit_json = bool(getattr(args, "json", False))

    if not all_items_to_consider:
        _hint = (
            "Set input_folder_path in config/defaults/app.yaml "
            "(copy app.example.yaml) or pass --input."
        )
        if config.CLI_MODE:
            logger.error("No items found to process in: %s. %s", input_path_arg, _hint)
            if emit_json:
                _emit_json_summary(0, 0, 0, 0, [])
            sys.exit(1)
        else:
            from cli.interaction import print_error

            print_error(f"No items found to process in: {input_path_arg}. {_hint}")
            logger.debug("No items found in: %s", input_path_arg)
            if emit_json:
                _emit_json_summary(0, 0, 0, 0, [])
            sys.exit(0)

    if not config.CLI_MODE:
        print_success(
            f"Found {len(all_items_to_consider)} item(s) available for processing"
        )

    selected_items = _select_items_for_processing(
        all_items_to_consider, process_all, select_pattern, emit_json
    )
    if not selected_items:
        if not config.CLI_MODE:
            print_info("No items selected for processing. Exiting.")
        if emit_json:
            _emit_json_summary(0, 0, 0, 0, [])
        sys.exit(0)

    logger.info("Selected %s item(s) for processing.", len(selected_items))
    return selected_items, base_output_dir, summary_context, resume_mode


def _apply_resume_filtering(
    selected_items: list[ItemSpec],
    base_output_dir: Path,
    resume_mode: str,
    retranscribe: bool = False,
) -> tuple[list[ItemSpec], dict[Path, ResumeResult], str, list[ResumeResult]]:
    """Filter items through the resume checker.

    Returns:
        (items_to_process, item_resume_map, resume_mode, skipped_items)
    """
    resume_checker = ResumeChecker(
        resume_mode=resume_mode,
        summarize=config.SUMMARIZE,
        output_docx=config.OUTPUT_DOCX,
        output_markdown=config.OUTPUT_MARKDOWN,
        retranscribe=retranscribe,
    )
    filtered_result = resume_checker.filter_items(
        items=selected_items,
        output_dir_func=lambda item: _resolve_item_output_dir(item, base_output_dir),
        name_func=lambda item: item.output_stem,
    )
    items_to_process: list[ItemSpec] = filtered_result[0]
    skipped_items: list[ResumeResult] = filtered_result[1]

    # Keyed by the item's input path, not its output stem: two items in
    # different directories can share a stem (e.g. BookA/images and BookB/images
    # both stem "images" under input_paths_is_output_path), and a stem key would
    # let one item's resume set clobber the other's (AE-1).
    item_resume_map: dict[Path, ResumeResult] = {}
    for item in items_to_process:
        result = resume_checker.should_skip(
            item.output_stem,
            _resolve_item_output_dir(item, base_output_dir),
        )
        if result.state in (
            ProcessingState.PARTIAL,
            ProcessingState.TRANSCRIPTION_ONLY,
        ):
            item_resume_map[item.path] = result

    if resume_mode == "skip":
        logger.info("Resume mode: skip (use --force to reprocess all)")
    else:
        logger.info("Resume mode: overwrite (all files will be reprocessed)")

    should_continue = _display_resume_info(
        resume_mode, selected_items, skipped_items, items_to_process
    )
    if not should_continue:
        # Nothing to do (all items already complete, or the user declined a
        # force reprocess). Return an empty work list instead of exiting so
        # main() can still emit the --json run summary (AE-5: a silent
        # sys.exit here swallowed the JSON line on all-skipped runs).
        if not config.CLI_MODE:
            print_info("No items to process. Exiting.")
        return [], item_resume_map, resume_mode, skipped_items

    if not items_to_process:
        items_to_process = list(selected_items)
        resume_mode = "overwrite"
        item_resume_map.clear()

    return items_to_process, item_resume_map, resume_mode, skipped_items


def _log_token_usage(context: str = "") -> None:
    """Log current token usage if daily limit tracking is enabled."""
    if not config.DAILY_TOKEN_LIMIT_ENABLED:
        return
    token_tracker = get_token_tracker()
    stats = token_tracker.get_stats()
    prefix = f"{context}: " if context else "Token usage: "
    logger.info(
        f"{prefix}{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
        f"({stats['usage_percentage']:.1f}%)"
        + (
            f" - {stats['tokens_remaining']:,} tokens remaining today"
            if not context
            else ""
        )
    )
    if not context and not config.CLI_MODE:
        print_info(
            f"Daily token usage: "
            f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )


def _run_processing_loop(
    items_to_process: list[ItemSpec],
    base_output_dir: Path,
    item_resume_map: dict[Path, ResumeResult],
    summary_context: str | None,
    resume_mode: str,
) -> tuple[int, list[str], int]:
    """Process items sequentially.

    Returns:
        ``(processed_count, outputs, unattempted_count)``: the number of items
        that fully succeeded, the absolute paths of all output files written
        this run, and the number of items never reached because the user
        cancelled the token-limit wait.
    """
    processed_count = 0
    run_outputs: list[str] = []
    total_to_process = len(items_to_process)
    attempted_count = 0

    for index, item_spec in enumerate(items_to_process, start=1):
        if not _check_and_wait_for_token_limit():
            logger.info(
                f"Processing stopped by user. "
                f"Processed {processed_count}/{total_to_process} items."
            )
            if not config.CLI_MODE:
                print_warning(
                    f"\nProcessing stopped. "
                    f"Completed {processed_count}/{total_to_process} items."
                )
            break

        attempted_count += 1
        item_output_dir = _resolve_item_output_dir(item_spec, base_output_dir)
        item_output_dir.mkdir(parents=True, exist_ok=True)

        item_resume = item_resume_map.get(item_spec.path)
        completed_pages = item_resume.completed_page_indices if item_resume else None

        success, item_outputs = _process_single_item(
            item_spec,
            index,
            total_to_process,
            item_output_dir,
            summary_context=summary_context,
            resume_mode=resume_mode,
            completed_page_indices=completed_pages,
        )
        run_outputs.extend(item_outputs)
        if success:
            processed_count += 1

        _log_token_usage(f"Token usage after file {index}/{total_to_process}")

    return processed_count, run_outputs, total_to_process - attempted_count


def _emit_json_summary(
    processed: int,
    failed: int,
    skipped: int,
    total: int,
    outputs: list[str],
) -> None:
    """Print one machine-readable JSON run-summary line on stdout."""
    stats = get_token_tracker().get_stats() if config.DAILY_TOKEN_LIMIT_ENABLED else {}
    # tokens_used_today stays this tool's OWN usage; when the shared budget is
    # enabled the ledger exposes the per-tool figure as own_tokens_used_today
    # (equal to tokens_used_today in standalone mode). The combined cross-tool
    # total is surfaced separately only when the shared budget is active.
    summary: dict[str, Any] = {
        "items_total": total,
        "items_complete": processed,
        "items_failed": failed,
        "items_skipped": skipped,
        "outputs": outputs,
        "tokens_used_today": stats.get(
            "own_tokens_used_today", stats.get("tokens_used_today")
        ),
        "daily_token_limit": stats.get("daily_limit"),
    }
    if stats.get("shared_budget_enabled"):
        summary["combined_tokens_today"] = stats.get("combined_tokens_used_today")
    # Per-key pool caps and each pooled bucket's used/remaining, when any pooled
    # key was used this run (existing keys above are untouched).
    pool_buckets = stats.get("pool_buckets")
    if pool_buckets:
        summary["per_key_pool_caps_enabled"] = stats.get("per_key_pool_caps_enabled")
        summary["pool_buckets"] = pool_buckets
    print(json.dumps(summary, ensure_ascii=False))


def _run_dry_run(
    items_to_process: list[ItemSpec],
    item_resume_map: dict[Path, ResumeResult],
    skipped_items: list[ResumeResult],
    emit_json: bool,
) -> None:
    """Report the planned actions (discovery + resume classification), no work."""
    plan: list[dict[str, Any]] = []
    for item in items_to_process:
        resume = item_resume_map.get(item.path)
        state = resume.state.value if resume else "none"
        pages = (
            len(resume.completed_page_indices)
            if resume and resume.completed_page_indices
            else 0
        )
        plan.append(
            {"name": item.output_stem, "state": state, "completed_pages": pages}
        )
        logger.info(
            "DRY-RUN plan: %s -> %s (%d completed page(s))",
            item.output_stem,
            state,
            pages,
        )
    for skipped in skipped_items:
        logger.info("DRY-RUN skip: %s -> complete", skipped.item_name)

    if emit_json:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "to_process": plan,
                    "skipped": [s.item_name for s in skipped_items],
                },
                ensure_ascii=False,
            )
        )


def main() -> int:
    args = setup_argparse()
    emit_json = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))
    retranscribe = bool(getattr(args, "retranscribe", False))

    # Non-TTY guard: interactive mode would block on input() prompts. Fail fast
    # with a clear message and usage exit code instead of hanging or EOF-ing.
    if not config.CLI_MODE and not sys.stdin.isatty():
        logger.error(
            "Interactive mode requires a TTY. Re-run with --cli (and input/"
            "output paths) for non-interactive use."
        )
        return 2

    selected_items, base_output_dir, summary_context, resume_mode = _setup_and_scan(
        args
    )

    items_to_process, item_resume_map, resume_mode, skipped_items = (
        _apply_resume_filtering(
            selected_items, base_output_dir, resume_mode, retranscribe=retranscribe
        )
    )

    if dry_run:
        _run_dry_run(items_to_process, item_resume_map, skipped_items, emit_json)
        return 0

    if not items_to_process:
        # All requested items were skipped as already complete (or the user
        # declined a force reprocess). Still honor the --json contract: emit
        # one run-summary line with the skipped count before exiting (AE-5).
        skipped_count = len(skipped_items)
        logger.info(
            "Nothing to process: %d item(s) skipped as already complete.",
            skipped_count,
        )
        if emit_json:
            _emit_json_summary(0, 0, skipped_count, 0, [])
        return 0

    # Prompt for summary context in interactive mode
    if not config.CLI_MODE and config.SUMMARIZE and not summary_context:
        summary_context = _prompt_for_summary_context()

    # Confirm and display section header
    if not config.CLI_MODE:
        confirmed = _display_processing_summary(
            items_to_process, base_output_dir, summary_context
        )
        if not confirmed:
            print_info("Processing cancelled by user.")
            return 0
        print_section(f"Processing {len(items_to_process)} Item(s)")

    _log_token_usage()

    processed_count, run_outputs, unattempted_count = _run_processing_loop(
        items_to_process,
        base_output_dir,
        item_resume_map,
        summary_context,
        resume_mode,
    )

    # Final summary
    skipped_count = len(skipped_items)
    total_to_process = len(items_to_process)
    # Only items that were actually attempted can be counted as failures. Items
    # never reached because the user cancelled the token-limit wait are reported
    # as not attempted (folded into the skipped/pending count and logged), never
    # as failures (AE-7). The run still exits non-zero when any item was left
    # unattempted, since the requested work did not finish.
    attempted_count = total_to_process - unattempted_count
    failed_count = attempted_count - processed_count
    if config.CLI_MODE:
        msg = f"{processed_count}/{total_to_process} item(s) processed."
        if skipped_count > 0:
            msg += f" {skipped_count} item(s) skipped (already complete)."
        if unattempted_count > 0:
            msg += f" {unattempted_count} item(s) not attempted (stopped early)."
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
        if unattempted_count > 0:
            print_dim(f"  ({unattempted_count} item(s) were not attempted)")
            print()

    logger.info(
        f"{processed_count}/{total_to_process} selected items have been processed."
    )
    if unattempted_count > 0:
        logger.info(
            "%d item(s) were not attempted (processing stopped before reaching them).",
            unattempted_count,
        )

    if emit_json:
        _emit_json_summary(
            processed_count,
            failed_count,
            skipped_count + unattempted_count,
            total_to_process,
            run_outputs,
        )

    # Exit code contract: 0 = all requested items succeeded; 1 = one or more
    # failed/partial, or one or more items were left unattempted.
    return 1 if failed_count > 0 or unattempted_count > 0 else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user (Ctrl+C). Exiting.")
        sys.exit(130)
    except Exception as exc:
        handle_critical_error(
            exc,
            "main execution flow",
            exit_on_error=True,
            show_user_message=True,
        )
