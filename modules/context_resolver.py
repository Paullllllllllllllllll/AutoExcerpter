"""Context resolution utilities for AutoExcerpter.

This module provides hierarchical context resolution for summarization tasks,
supporting file-specific, folder-specific, and general context files.

Context Resolution Hierarchy:
1. File-specific: <filename>_summary_context.txt next to the input file
2. Folder-specific: <foldername>_summary_context.txt in the parent directory
3. General fallback: context/summary/general.txt

The resolved context is injected into the summarization system prompt to guide
the model to pay special attention to specific topics during summarization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from modules.logger import setup_logger

logger = setup_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONTEXT_DIR = _PROJECT_ROOT / "context"

DEFAULT_CONTEXT_SIZE_THRESHOLD = 4000
CONTEXT_SUFFIX = "_summary_context.txt"


def resolve_summary_context(
    input_file: Optional[Path] = None,
    global_context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve summarization context using hierarchical fallback.
    
    Searches for context in this order:
    1. File-specific: <filename>_summary_context.txt in the same directory as input_file
    2. Folder-specific: <parent_folder_name>_summary_context.txt in grandparent directory
    3. General fallback: context/summary/general.txt
    
    Parameters
    ----------
    input_file : Optional[Path]
        Path to the input file (PDF or image folder) for file/folder-specific context
    global_context_dir : Optional[Path]
        Override for the global context directory (defaults to PROJECT_ROOT/context)
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        A tuple of (context_content, resolved_path) or (None, None) if no context found
    """
    context_dir = global_context_dir or _CONTEXT_DIR
    
    # 1. File-specific context
    if input_file is not None:
        input_file = Path(input_file).resolve()
        file_specific = input_file.with_name(f"{input_file.stem}{CONTEXT_SUFFIX}")
        if file_specific.exists():
            content = _read_and_validate_context(file_specific, size_threshold)
            if content:
                logger.info(f"Using file-specific summary context: {file_specific}")
                return content, file_specific
        
        # 2. Folder-specific context
        parent_folder = input_file.parent
        if parent_folder.parent.exists():
            folder_specific = parent_folder.parent / f"{parent_folder.name}{CONTEXT_SUFFIX}"
            if folder_specific.exists():
                content = _read_and_validate_context(folder_specific, size_threshold)
                if content:
                    logger.info(f"Using folder-specific summary context: {folder_specific}")
                    return content, folder_specific
    
    # 3. General fallback
    general_fallback = context_dir / "summary" / "general.txt"
    if general_fallback.exists():
        content = _read_and_validate_context(general_fallback, size_threshold)
        if content:
            logger.info(f"Using general summary context: {general_fallback}")
            return content, general_fallback
    
    logger.debug("No summary context found")
    return None, None


def _read_and_validate_context(
    context_path: Path,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Optional[str]:
    """Read and validate a context file.
    
    Parameters
    ----------
    context_path : Path
        Path to the context file
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Optional[str]
        The context content, or None if file is empty or unreadable
    """
    try:
        content = context_path.read_text(encoding="utf-8").strip()
        
        if not content:
            logger.debug(f"Context file is empty: {context_path}")
            return None
        
        if len(content) > size_threshold:
            logger.warning(
                f"Context file '{context_path.name}' is large ({len(content):,} chars). "
                f"Consider reducing to under {size_threshold:,} chars for optimal performance."
            )
        
        return content
        
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(f"Failed to read context file {context_path}: {exc}")
        return None


def format_context_for_prompt(context: str) -> str:
    """Format context content for injection into the system prompt.
    
    Parameters
    ----------
    context : str
        Raw context content from context file
        
    Returns
    -------
    str
        Formatted context string ready for prompt injection
    """
    # Clean up the context - remove extra whitespace but preserve structure
    lines = [line.strip() for line in context.strip().split('\n') if line.strip()]
    return ', '.join(lines) if lines else context.strip()


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "resolve_summary_context",
    "format_context_for_prompt",
    "DEFAULT_CONTEXT_SIZE_THRESHOLD",
    "CONTEXT_SUFFIX",
]
