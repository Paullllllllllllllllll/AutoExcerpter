"""CLI argument parsing and execution-mode resolution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from modules import app_config as config
from modules.logger import setup_logger

logger = setup_logger(__name__)

REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high")
VERBOSITY_CHOICES = ("low", "medium", "high")


def _positive_int(value: str) -> int:
    """Argparse type validator for positive integers."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


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
        "--force",
        "--overwrite",
        action="store_true",
        dest="force",
        default=None,
        help="Force reprocessing of all files, overwriting existing output.",
    )

    if config.CLI_MODE:
        # CLI mode: require input and output arguments
        parser.add_argument(
            "input",
            nargs="?",
            type=str,
            default=None,
            help="Path to PDF file, image folder, or directory containing PDFs/image folders (relative or absolute).",
        )
        parser.add_argument(
            "output",
            nargs="?",
            type=str,
            default=None,
            help="Output directory path for transcriptions and summaries (relative or absolute).",
        )
        parser.add_argument(
            "--input-path",
            type=str,
            default=None,
            help="Named input path (same as positional input). Overrides positional input if both are provided.",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=None,
            help="Named output path (same as positional output). Overrides positional output if both are provided.",
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
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Set both transcription and summary model names.",
        )
        parser.add_argument(
            "--transcription-model",
            type=str,
            default=None,
            help="Override transcription model name.",
        )
        parser.add_argument(
            "--summary-model",
            type=str,
            default=None,
            help="Override summary model name.",
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=REASONING_EFFORT_CHOICES,
            default=None,
            help="Set reasoning effort for both models (OpenAI reasoning models).",
        )
        parser.add_argument(
            "--transcription-reasoning-effort",
            choices=REASONING_EFFORT_CHOICES,
            default=None,
            help="Override transcription reasoning effort.",
        )
        parser.add_argument(
            "--summary-reasoning-effort",
            choices=REASONING_EFFORT_CHOICES,
            default=None,
            help="Override summary reasoning effort.",
        )
        parser.add_argument(
            "--verbosity",
            choices=VERBOSITY_CHOICES,
            default=None,
            help="Set output verbosity for both models (currently GPT-5 OpenAI models).",
        )
        parser.add_argument(
            "--transcription-verbosity",
            choices=VERBOSITY_CHOICES,
            default=None,
            help="Override transcription verbosity (currently GPT-5 OpenAI models).",
        )
        parser.add_argument(
            "--summary-verbosity",
            choices=VERBOSITY_CHOICES,
            default=None,
            help="Override summary verbosity (currently GPT-5 OpenAI models).",
        )
        parser.add_argument(
            "--max-output-tokens",
            type=_positive_int,
            default=None,
            help="Set max output tokens for both models.",
        )
        parser.add_argument(
            "--transcription-max-output-tokens",
            type=_positive_int,
            default=None,
            help="Override transcription max output tokens.",
        )
        parser.add_argument(
            "--summary-max-output-tokens",
            type=_positive_int,
            default=None,
            help="Override summary max output tokens.",
        )
    else:
        # Interactive mode: optional input argument with default from config
        parser.add_argument(
            "--input",
            type=str,
            default=config.INPUT_FOLDER_PATH,
            help="Path to the folder containing PDFs and/or image folders, or path to a single PDF/image folder.",
        )

    args = parser.parse_args()
    if config.CLI_MODE:
        effective_input = getattr(args, "input_path", None) or getattr(
            args, "input", None
        )
        effective_output = getattr(args, "output_path", None) or getattr(
            args, "output", None
        )
        if not effective_input or not effective_output:
            parser.error("CLI mode requires both input and output paths.")

    return args


def _parse_execution_mode(
    args: argparse.Namespace,
) -> tuple[Path, Path, bool, str | None, str | None, str]:
    """Parse execution mode and return input path, output path, process_all flag, select pattern, context, and resume mode."""
    # Resolve resume mode from CLI flags (default: "skip")
    if getattr(args, "force", None):
        resume_mode = "overwrite"
    else:
        resume_mode = "skip"

    if config.CLI_MODE:
        # CLI mode: use command line arguments
        input_value = getattr(args, "input_path", None) or getattr(args, "input", None)
        output_value = getattr(args, "output_path", None) or getattr(
            args, "output", None
        )
        if not input_value or not output_value:
            raise ValueError("CLI mode requires input and output paths")

        input_path_arg = Path(input_value)
        base_output_dir = Path(output_value)
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

    return (
        input_path_arg,
        base_output_dir,
        process_all,
        select_pattern,
        summary_context,
        resume_mode,
    )


def _set_model_override(
    overrides: dict[str, Any], model_key: str, path: list[str], value: Any
) -> None:
    """Set nested model override value for runtime config application."""
    model_overrides = overrides.setdefault(model_key, {})
    cursor = model_overrides
    for segment in path[:-1]:
        cursor = cursor.setdefault(segment, {})
    cursor[path[-1]] = value


def _build_cli_model_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build model.yaml runtime override dict from CLI args."""
    if not config.CLI_MODE:
        return {}

    overrides: dict[str, Any] = {}

    shared_model = getattr(args, "model", None)
    shared_reasoning = getattr(args, "reasoning_effort", None)
    shared_verbosity = getattr(args, "verbosity", None)
    shared_max_tokens = getattr(args, "max_output_tokens", None)

    transcription_model = getattr(args, "transcription_model", None) or shared_model
    summary_model = getattr(args, "summary_model", None) or shared_model
    if transcription_model:
        _set_model_override(
            overrides, "transcription_model", ["name"], transcription_model
        )
    if summary_model:
        _set_model_override(overrides, "summary_model", ["name"], summary_model)

    transcription_reasoning = (
        getattr(args, "transcription_reasoning_effort", None) or shared_reasoning
    )
    summary_reasoning = (
        getattr(args, "summary_reasoning_effort", None) or shared_reasoning
    )
    if transcription_reasoning:
        _set_model_override(
            overrides,
            "transcription_model",
            ["reasoning", "effort"],
            transcription_reasoning,
        )
    if summary_reasoning:
        _set_model_override(
            overrides,
            "summary_model",
            ["reasoning", "effort"],
            summary_reasoning,
        )

    transcription_verbosity = (
        getattr(args, "transcription_verbosity", None) or shared_verbosity
    )
    summary_verbosity = getattr(args, "summary_verbosity", None) or shared_verbosity
    if transcription_verbosity:
        _set_model_override(
            overrides,
            "transcription_model",
            ["text", "verbosity"],
            transcription_verbosity,
        )
    if summary_verbosity:
        _set_model_override(
            overrides,
            "summary_model",
            ["text", "verbosity"],
            summary_verbosity,
        )

    transcription_max_tokens = (
        getattr(args, "transcription_max_output_tokens", None) or shared_max_tokens
    )
    summary_max_tokens = (
        getattr(args, "summary_max_output_tokens", None) or shared_max_tokens
    )
    if transcription_max_tokens is not None:
        _set_model_override(
            overrides,
            "transcription_model",
            ["max_output_tokens"],
            int(transcription_max_tokens),
        )
    if summary_max_tokens is not None:
        _set_model_override(
            overrides,
            "summary_model",
            ["max_output_tokens"],
            int(summary_max_tokens),
        )

    return overrides


def _parse_cli_selection(items: list, pattern: str) -> list:
    """Parse CLI selection pattern and return matching items.

    Args:
        items: List of available items (ItemSpec instances)
        pattern: Selection pattern (numbers, ranges, or filename search)

    Returns:
        List of selected items
    """
    selected_indices: set[int] = set()
    pattern = pattern.strip()

    # Check if pattern looks like numeric selection
    numeric_pattern = pattern.replace(" ", "").replace(";", ",")
    is_numeric = all(c.isdigit() or c in ",-" for c in numeric_pattern) and any(
        c.isdigit() for c in numeric_pattern
    )

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
