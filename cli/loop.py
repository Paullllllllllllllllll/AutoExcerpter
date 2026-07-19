"""Item processing and token-limit utilities for the CLI."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any

from cli.display import _log_token_limit_reached
from cli.errors import handle_critical_error
from cli.interaction import (
    print_dim,
    print_highlight,
    print_separator,
    print_success,
    print_warning,
)
from config import app as config
from config.logger import setup_logger
from llm.token_tracker import DailyTokenTracker, get_token_tracker
from pipeline import ItemTranscriber
from pipeline.types import ItemSpec

logger = setup_logger(__name__)


def _process_single_item(
    item_spec: ItemSpec,
    index: int,
    total_items: int,
    base_output_dir: Path,
    summary_context: str | None = None,
    resume_mode: str = "skip",
    completed_page_indices: set[int] | None = None,
    prior_transcription_results: list[dict[str, Any]] | None = None,
    prior_summary_results: list[dict[str, Any]] | None = None,
    logged_log_header: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Process a single PDF or image folder item.

    Args:
        item_spec: Specification of the item to process.
        index: Current item index (1-based).
        total_items: Total number of items to process.
        base_output_dir: Base directory for outputs.
        summary_context: Optional context for guiding summarization focus.
        resume_mode: Resume mode ("skip" or "overwrite").
        completed_page_indices: Set of page indices already completed
            (for page-level resume).
        prior_transcription_results: Transcription working-log entries already
            parsed by the resume check, reused to avoid re-reading the log.
            None falls back to a disk read inside ItemTranscriber.
        prior_summary_results: Summary working-log entries already parsed by the
            resume check. None falls back to a disk read.
        logged_log_header: Transcription-log header already parsed by the resume
            check. None falls back to a disk read.

    Returns:
        ``(success, outputs, report)`` where *success* is True only when every
        page succeeded and all configured outputs were written, *outputs* is the
        list of absolute output-file paths written for this item, and *report*
        is the transcriber's ``last_run_report`` (None if the item aborted
        before any per-item stats were computed).
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
        from cli.interaction import print_info

        print_info(f"    • Name: {item_spec.output_stem}")
        print_info(f"    • Type: {item_spec.kind.replace('_', ' ').title()}")
        if completed_page_indices:
            print_dim(
                f"    • Resuming: {len(completed_page_indices)}"
                " page(s) already transcribed"
            )
        print()

    # Process the item
    transcriber_instance: ItemTranscriber | None = None
    success = False
    try:
        transcriber_instance = ItemTranscriber(
            item_spec.path,
            item_spec.kind,
            base_output_dir,
            summary_context=summary_context,
            resume_mode=resume_mode,
            completed_page_indices=completed_page_indices,
            prior_transcription_results=prior_transcription_results,
            prior_summary_results=prior_summary_results,
            logged_log_header=logged_log_header,
        )
        success = transcriber_instance.process_item()
        outputs = [str(p) for p in transcriber_instance.written_outputs]
        report = transcriber_instance.last_run_report

        if success:
            if config.CLI_MODE:
                logger.info(f"Successfully processed: {item_spec.output_stem}")
            else:
                print_success(f"Successfully processed: {item_spec.output_stem}")
        else:
            logger.error(
                "Item finished with page-level failures or missing outputs: %s",
                item_spec.output_stem,
            )
            if not config.CLI_MODE:
                print_warning(
                    f"Completed with failures: {item_spec.output_stem}"
                    " (see log for failed pages)"
                )
        _print_item_result(item_spec.output_stem, report)
        return success, outputs, report

    except Exception as exc:
        handle_critical_error(
            exc,
            f"processing item '{item_spec.output_stem}'",
            exit_on_error=False,
            show_user_message=not config.CLI_MODE,
        )
        logger.info("--- Attempting to continue with the next item if any. ---")
        outputs = (
            [str(p) for p in transcriber_instance.written_outputs]
            if transcriber_instance is not None
            else []
        )
        report = (
            transcriber_instance.last_run_report
            if transcriber_instance is not None
            else None
        )
        _print_item_result(item_spec.output_stem, report)
        return False, outputs, report

    finally:
        # Clean up temporary working directory if configured
        # In resume mode ("skip"), retain working dir to preserve JSONL logs
        # for future resumes. An incomplete item (deferred or failed pages)
        # also keeps its working dir: process_item promises those JSONL logs
        # are retained for a later resume run.
        should_cleanup_working_dir = (
            config.DELETE_TEMP_WORKING_DIR and resume_mode != "skip" and success
        )
        if transcriber_instance and should_cleanup_working_dir:
            working_dir = transcriber_instance.working_dir
            if working_dir.exists():
                _cleanup_working_directory(working_dir)


def _print_item_result(name: str, report: dict[str, Any] | None) -> None:
    """Print a compact per-item result block after each item.

    Interactive mode uses the dimmed print helpers (stdout); CLI mode writes
    one or two plain lines to stderr (stdout is reserved for the --json line).
    A no-op when no report was produced (item aborted before stats).
    """
    if not isinstance(report, dict):
        return

    attempted = report.get("pages_attempted", 0)
    ok = report.get("pages_ok", 0)
    failed = report.get("pages_failed", 0)
    deferred = report.get("pages_deferred", 0)
    elapsed = report.get("elapsed_s")
    avg_api = report.get("avg_api_s")
    outputs = report.get("outputs") or []

    if config.CLI_MODE:
        line = (
            f"  {name}: {ok}/{attempted} page(s) ok, "
            f"{failed} failed, {deferred} deferred"
        )
        if elapsed is not None:
            line += f"; {elapsed:.1f}s"
        if avg_api is not None:
            line += f"; avg API {avg_api:.1f}s"
        print(line, file=sys.stderr)
        if outputs:
            print(f"    outputs: {', '.join(outputs)}", file=sys.stderr)
    else:
        print_dim(
            f"    • Pages: {ok}/{attempted} ok, {failed} failed, {deferred} deferred"
        )
        if elapsed is not None:
            print_dim(f"    • Elapsed: {elapsed:.1f}s")
        if avg_api is not None:
            print_dim(f"    • Avg API time: {avg_api:.1f}s")
        for out in outputs:
            print_dim(f"    • Output: {out}")


def _cleanup_working_directory(working_dir: Path) -> None:
    """Clean up temporary working directory with proper error handling.

    Args:
        working_dir: Path to working directory to remove.
    """

    def _on_remove_error(func: Any, path_to_fix: Any, _exc_info: Any) -> None:
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
    """Check if daily token limit is reached and wait until next day if needed.

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


def _wait_for_token_reset(
    token_tracker: DailyTokenTracker, seconds_until_reset: int
) -> bool:
    """Wait for token limit reset with cancellation support.

    The cancellation poll stays at 1 s for responsiveness, but the two costly
    side effects are throttled: the shared-ledger force-sync runs at most every
    ~10 s and the app.yaml / pool-settings re-reads at most every ~30 s. A wait
    can last hours, so doing both every second churned the shared ledger and
    re-parsed YAML needlessly.
    """
    ledger_sync_interval = 10
    config_reload_interval = 30

    elapsed = 0
    sleep_interval = 1  # Check every second for responsiveness
    # Seed the throttle markers one interval in the past so the first sync and
    # config re-read fire early (external changes are picked up promptly, and a
    # short wait still rereads), then settle onto the throttled cadence.
    last_ledger_sync = -ledger_sync_interval
    last_config_reload = -config_reload_interval

    while elapsed < seconds_until_reset:
        if _user_requested_cancel():
            logger.info("Wait cancelled by user ('q').")
            if not config.CLI_MODE:
                print_warning("\nWait cancelled by user.")
            return False

        interval = min(sleep_interval, max(0, seconds_until_reset - elapsed))
        time.sleep(interval)
        elapsed += interval

        # Throttled forced ledger refresh so another tool's usage or its 00:01
        # UTC reset is observed while we wait, without a sync every second. A
        # no-op when the shared budget is off, so single-tool waits are cheap.
        if (
            token_tracker._shared_enabled
            and elapsed - last_ledger_sync >= ledger_sync_interval
        ):
            last_ledger_sync = elapsed
            try:
                token_tracker.sync_ledger_now()
            except Exception:
                logger.debug("Shared ledger refresh during wait failed", exc_info=True)

        # Throttled live re-read of the configured daily limit and per-key-pool
        # settings so a mid-wait edit (raising daily_tokens, remapping pools)
        # takes effect without a restart, without re-parsing YAML every second.
        # The per-item managers re-resolve their key env vars at the next item's
        # construction, so an api_keys.yaml remap is picked up on the next item.
        if elapsed - last_config_reload >= config_reload_interval:
            last_config_reload = elapsed
            new_limit = config.reload_daily_token_limit()
            if new_limit is not None:
                token_tracker.set_daily_limit(new_limit)

            pool_cfg = config.reload_pool_settings()
            if pool_cfg is not None:
                token_tracker.set_pool_settings(
                    scope=pool_cfg.get("scope"),
                    pool_caps_enabled=pool_cfg.get("pool_caps_enabled"),
                    pool_caps=pool_cfg.get("pool_caps"),
                    pool_models=pool_cfg.get("pool_models"),
                )

        # Re-check if it's a new day (or the cap was lifted above)
        if not token_tracker.is_limit_reached():
            logger.info("Token limit has been reset. Resuming processing.")
            if not config.CLI_MODE:
                print_success("Token limit has been reset. Resuming processing.")
            return True

    logger.info("Token limit has been reset. Resuming processing.")
    if not config.CLI_MODE:
        print_success("\nToken limit has been reset. Resuming processing.")
    return True


def _user_requested_cancel() -> bool:
    """Check if the user requested cancellation by pressing 'q'."""
    try:
        if os.name == "nt":
            import msvcrt

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
        # If any error occurs (e.g., select not supported), fall back to
        # KeyboardInterrupt handling.
        return False
