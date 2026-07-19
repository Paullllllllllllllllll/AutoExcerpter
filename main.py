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
import time
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
    _print_cli_completion_overview,
    _prompt_for_summary_context,
    prompt_for_item_selection,
)
from cli.errors import handle_critical_error
from cli.interaction import (
    print_info,
    print_section,
    print_success,
    print_warning,
    set_exit_hook,
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

    # A --dry-run must be side-effect-free: creating the output tree here would
    # leave directories behind for a run that does no work. Defer it to the
    # actual processing loop, which mkdirs each item's output dir on demand.
    if not bool(getattr(args, "dry_run", False)):
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
    dry_run: bool = False,
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
        resume_mode, selected_items, skipped_items, items_to_process, dry_run=dry_run
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
) -> tuple[
    int,
    list[str],
    int,
    list[str],
    list[tuple[str, dict[str, Any] | None]],
    float,
    int | None,
]:
    """Process items sequentially.

    Returns:
        ``(processed_count, outputs, unattempted_count, incomplete_items,
        reports, run_seconds, tokens_used_run)``: the number of items that fully
        succeeded, the absolute paths of all output files written this run, the
        number of items never reached because the user cancelled the token-limit
        wait, the display names of items that were attempted but finished
        incomplete, the per-item ``(name, last_run_report)`` pairs for every
        attempted item, the wall-clock time spent in the loop, and the tokens
        consumed this run (None when daily-token tracking is disabled).
    """
    processed_count = 0
    run_outputs: list[str] = []
    incomplete_items: list[str] = []
    reports: list[tuple[str, dict[str, Any] | None]] = []
    total_to_process = len(items_to_process)
    attempted_count = 0

    tokens_before: int | None = None
    if config.DAILY_TOKEN_LIMIT_ENABLED:
        tokens_before = get_token_tracker().get_stats().get("tokens_used_today")

    loop_start = time.monotonic()
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

        success, item_outputs, report = _process_single_item(
            item_spec,
            index,
            total_to_process,
            item_output_dir,
            summary_context=summary_context,
            resume_mode=resume_mode,
            completed_page_indices=completed_pages,
            # Reuse working-log data already parsed by the resume check so the
            # transcriber need not re-read the same JSONL files.
            prior_transcription_results=(
                item_resume.transcription_results if item_resume else None
            ),
            prior_summary_results=(
                item_resume.summary_results if item_resume else None
            ),
            logged_log_header=item_resume.log_header if item_resume else None,
        )
        run_outputs.extend(item_outputs)
        reports.append((item_spec.output_stem, report))
        if success:
            processed_count += 1
        else:
            incomplete_items.append(item_spec.output_stem)

        _log_token_usage(f"Token usage after file {index}/{total_to_process}")

    run_seconds = time.monotonic() - loop_start

    tokens_used_run: int | None = None
    if config.DAILY_TOKEN_LIMIT_ENABLED and tokens_before is not None:
        tokens_after = get_token_tracker().get_stats().get("tokens_used_today")
        if tokens_after is not None:
            tokens_used_run = max(0, tokens_after - tokens_before)

    return (
        processed_count,
        run_outputs,
        total_to_process - attempted_count,
        incomplete_items,
        reports,
        run_seconds,
        tokens_used_run,
    )


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
    # A summary line was emitted, so disarm any interactive exit hook: a later
    # exit_program (defensive) must not print a second, contradictory summary.
    set_exit_hook(None)
    if stats.get("shared_budget_enabled"):
        summary["combined_tokens_today"] = stats.get("combined_tokens_used_today")
    # Per-key pool caps and each pooled bucket's used/remaining, when any pooled
    # key was used this run (existing keys above are untouched).
    pool_buckets = stats.get("pool_buckets")
    if pool_buckets:
        summary["per_key_pool_caps_enabled"] = stats.get("per_key_pool_caps_enabled")
        summary["pool_buckets"] = pool_buckets
    print(json.dumps(summary, ensure_ascii=False))


def _warn_incomplete_items(
    incomplete_items: list[str],
    reports: dict[str, dict[str, Any] | None] | None = None,
) -> None:
    """Print a prominent, user-facing warning naming items that finished
    incomplete (failed/partial pages or missing outputs).

    Emitted once after all items in BOTH interactive and CLI modes via the
    shared console helper (a real stdout print, not just a log record), so the
    partial failure is visible even when only log-level errors were recorded.
    When a per-item report is available, the exact failed/deferred page counts
    are shown inline; otherwise it falls back to the log pointer.
    """
    if not incomplete_items:
        return
    reports = reports or {}
    print_warning(
        f"{len(incomplete_items)} item(s) finished INCOMPLETE. Their "
        ".txt/.docx/.md outputs may contain error placeholders:"
    )
    for name in incomplete_items:
        rep = reports.get(name)
        if rep:
            failed = rep.get("pages_failed", 0)
            deferred = rep.get("pages_deferred", 0)
            summary_failures = rep.get("summary_failures", 0)
            detail = f"{failed} failed, {deferred} deferred page(s)"
            if summary_failures:
                detail += f", {summary_failures} summary failure(s)"
            print_warning(f"    - {name} ({detail})")
        else:
            print_warning(f"    - {name} (one or more pages failed or were deferred)")
    print_warning(
        "Re-run with --resume to retry only the missing pages; see each item's "
        "log for the exact failed/deferred page counts."
    )


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
        # The logger.info above sits below the CLI WARNING threshold and is
        # otherwise invisible, so also surface the plan on the human channel.
        plan_line = f"{item.output_stem} -> {state} ({pages} completed page(s))"
        if config.CLI_MODE:
            print(f"DRY-RUN: {plan_line}", file=sys.stderr)
        else:
            print_info(f"Plan: {plan_line}")
    for skipped in skipped_items:
        logger.info("DRY-RUN skip: %s -> complete", skipped.item_name)
        skip_line = f"{skipped.item_name} -> already complete (skip)"
        if config.CLI_MODE:
            print(f"DRY-RUN: {skip_line}", file=sys.stderr)
        else:
            print_info(f"Plan: {skip_line}")

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


def _guard_duplicate_outputs(
    items_to_process: list[ItemSpec],
    base_output_dir: Path,
    emit_json: bool,
) -> None:
    """Abort if two selected items resolve to the same output target.

    Two items with the same output stem in the same resolved output directory
    would silently overwrite each other's .txt/.docx/.md and interleave their
    resume logs. Detect the collision up front and stop with a clear error.
    """
    seen: dict[tuple[Path, str], ItemSpec] = {}
    for item in items_to_process:
        key = (
            _resolve_item_output_dir(item, base_output_dir).resolve(),
            item.output_stem,
        )
        prior = seen.get(key)
        if prior is not None:
            message = (
                f"Duplicate output target: '{prior.path}' and '{item.path}' both "
                f"resolve to output stem '{item.output_stem}' in the same "
                "directory; their outputs would overwrite each other. Rename one "
                "input or send them to separate output directories."
            )
            if config.CLI_MODE:
                logger.error(message)
                if emit_json:
                    _emit_json_summary(0, 0, 0, len(items_to_process), [])
                sys.exit(2)
            else:
                from cli.interaction import print_error

                print_error(message)
                sys.exit(1)
        seen[key] = item


def main() -> int:
    args = setup_argparse()
    emit_json = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))
    retranscribe = bool(getattr(args, "retranscribe", False))

    # Typing exit/quit/q (or a closed stdin / Ctrl+C) at any interactive prompt
    # terminates via exit_program, bypassing main's own JSON emit. Register a
    # hook so those exits still honor the emit-JSON-on-all-exits contract with
    # the same zero/empty summary shape used by the decline path. The hook is
    # one-shot and is cleared by _emit_json_summary, so it can never double-emit.
    set_exit_hook((lambda: _emit_json_summary(0, 0, 0, 0, [])) if emit_json else None)

    # Non-TTY guard: interactive mode would block on input() prompts. Fail fast
    # with a clear message and usage exit code instead of hanging or EOF-ing.
    if not config.CLI_MODE and not sys.stdin.isatty():
        logger.error(
            "Interactive mode requires a TTY. Re-run with --cli (and input/"
            "output paths) for non-interactive use."
        )
        # Honor the emit-JSON-on-all-exits contract even on this early guard.
        if emit_json:
            _emit_json_summary(0, 0, 0, 0, [])
        return 2

    selected_items, base_output_dir, summary_context, resume_mode = _setup_and_scan(
        args
    )

    items_to_process, item_resume_map, resume_mode, skipped_items = (
        _apply_resume_filtering(
            selected_items,
            base_output_dir,
            resume_mode,
            retranscribe=retranscribe,
            dry_run=dry_run,
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

    # Refuse to silently overwrite when two items share an output target.
    _guard_duplicate_outputs(items_to_process, base_output_dir, emit_json)

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
            # Honor the --json contract even on an interactive decline: 0
            # processed/failed, the skipped count, and the full attempted total.
            if emit_json:
                _emit_json_summary(0, 0, len(skipped_items), len(items_to_process), [])
            return 0
        print_section(f"Processing {len(items_to_process)} Item(s)")

    _log_token_usage()

    (
        processed_count,
        run_outputs,
        unattempted_count,
        incomplete_items,
        reports,
        run_seconds,
        tokens_used_run,
    ) = _run_processing_loop(
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
        # The logger.info line above is below the CLI WARNING threshold and thus
        # invisible; print a concise overview to stderr so the outcome is seen.
        _print_cli_completion_overview(
            processed_count,
            total_to_process,
            skipped_count=skipped_count,
            unattempted_count=unattempted_count,
            reports=reports,
            run_seconds=run_seconds,
            tokens_used_run=tokens_used_run,
            run_outputs=run_outputs,
        )
    else:
        _display_completion_summary(
            processed_count,
            total_to_process,
            base_output_dir if not config.INPUT_PATHS_IS_OUTPUT_PATH else None,
            skipped_count=skipped_count,
            unattempted_count=unattempted_count,
            reports=reports,
            run_seconds=run_seconds,
            tokens_used_run=tokens_used_run,
            run_outputs=run_outputs,
        )

    logger.info(
        f"{processed_count}/{total_to_process} selected items have been processed."
    )
    if unattempted_count > 0:
        logger.info(
            "%d item(s) were not attempted (processing stopped before reaching them).",
            unattempted_count,
        )

    # Prominent, once-per-run warning naming any item that finished incomplete.
    # Printed before the JSON line so a --json consumer still finds the summary
    # as the last stdout line.
    _warn_incomplete_items(incomplete_items, dict(reports))

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
