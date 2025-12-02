"""Text cleaning utilities for post-processing transcription output.

This module provides configurable text cleaning for OCR/transcription outputs,
adapted from the ChronoMiner post-processing pipeline. It performs conservative
cleanup while preserving semantic and typographic content.

Processing stages (all configurable):
1. Unicode normalization - NFC, remove control chars, soft hyphens, BOMs
2. LaTeX formula fixing - balance delimiters, close braces, fix common issues
3. Hyphenation merging - rejoin words split at line breaks
4. Whitespace normalization - collapse spaces, limit blank lines, expand tabs
5. Line wrapping (optional) - wrap long lines with smart heuristics

Configuration is loaded from image_processing.yaml under the 'text_cleaning' section.

Usage:
    >>> from modules.text_cleaner import clean_transcription, get_text_cleaning_config
    >>> config = get_text_cleaning_config()
    >>> cleaned_text = clean_transcription(raw_text, config)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from modules.config_loader import get_config_loader
from modules.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Constants
# ============================================================================

# High-plane icon glyphs sometimes seen in OCR output, map to bullet
AEGEAN_ICON_CODEPOINTS = {
    0x10101,  # AEGEAN WORD SEPARATOR DOT
    0x10102,  # AEGEAN CHECK MARK
    0x10103,
    0x10104,
    0x10105,
}

# Characters to drop during Unicode normalization
DROP_CHARS = {
    "\u00AD",  # SOFT HYPHEN
    "\u200B",  # ZERO WIDTH SPACE
    "\u200C",  # ZERO WIDTH NON-JOINER
    "\u200D",  # ZERO WIDTH JOINER
    "\uFEFF",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u2060",  # WORD JOINER
}

# Common LaTeX command typos from OCR
LATEX_COMMAND_FIXES = [
    (r"\\frac\s*\{", r"\\frac{"),  # Remove space after \frac
    (r"\\sqrt\s*\{", r"\\sqrt{"),  # Remove space after \sqrt
    (r"\\sum\s*_", r"\\sum_"),     # Remove space before subscript
    (r"\\int\s*_", r"\\int_"),     # Remove space before subscript
    (r"\\prod\s*_", r"\\prod_"),   # Remove space before subscript
    (r"\\lim\s*_", r"\\lim_"),     # Remove space before subscript
    (r"\\mathrm\s*\{", r"\\mathrm{"),
    (r"\\mathbf\s*\{", r"\\mathbf{"),
    (r"\\mathit\s*\{", r"\\mathit{"),
    (r"\\text\s*\{", r"\\text{"),
]

# Regex pattern for hyphenated line breaks
_HYPHEN_PATTERN = re.compile(r"(\w{3,})-\n(\w{2,})")


# ============================================================================
# Configuration
# ============================================================================

def get_text_cleaning_config() -> dict[str, Any]:
    """Load text cleaning configuration from image_processing.yaml."""
    config_loader = get_config_loader()
    img_config = config_loader.get_image_processing_config()
    
    # Get text_cleaning section with defaults
    cleaning_cfg = img_config.get("text_cleaning", {})
    
    # Return config with defaults for all options
    return {
        "enabled": cleaning_cfg.get("enabled", True),
        "unicode_normalization": cleaning_cfg.get("unicode_normalization", True),
        "latex_fixing": cleaning_cfg.get("latex_fixing", {
            "enabled": True,
            "balance_dollar_signs": True,
            "close_unclosed_braces": True,
            "fix_common_commands": True,
        }),
        "merge_hyphenation": cleaning_cfg.get("merge_hyphenation", False),
        "whitespace_normalization": cleaning_cfg.get("whitespace_normalization", {
            "enabled": True,
            "collapse_internal_spaces": True,
            "max_blank_lines": 2,
            "tab_size": 4,
        }),
        "line_wrapping": cleaning_cfg.get("line_wrapping", {
            "enabled": False,
            "auto_width": False,
            "fixed_width": 80,
        }),
    }


# ============================================================================
# Unicode Normalization
# ============================================================================

def normalize_unicode(text: str) -> str:
    """Normalize Unicode and remove spurious control characters.
    
    Steps:
    - NFC normalization (compose accented characters)
    - Map high-plane icon glyphs to bullet character
    - Drop soft hyphens, zero-width spaces, BOMs
    - Remove control/format/surrogate/unassigned chars (except newline/tab)
    
    Args:
        text: Input text to normalize.
        
    Returns:
        Unicode-normalized text with control characters removed.
    """
    if not text:
        return text
    
    # NFC normalization for composed accents
    text = unicodedata.normalize("NFC", text)
    
    # Map rare icon glyphs to bullet
    translation = {cp: "•" for cp in AEGEAN_ICON_CODEPOINTS}
    text = text.translate(translation)
    
    # Drop known layout artifacts
    for ch in DROP_CHARS:
        text = text.replace(ch, "")
    
    # Remove remaining control/format/unassigned chars (keep newline and tab)
    out_chars: list[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            out_chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            # Skip control/format/surrogate/unassigned characters
            continue
        out_chars.append(ch)
    
    return "".join(out_chars)


# ============================================================================
# LaTeX Formula Fixing
# ============================================================================

def _count_unescaped(text: str, char: str) -> int:
    """Count unescaped occurrences of a character."""
    count = 0
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == char:
            # Skip escaped character
            i += 2
            continue
        if text[i] == char:
            count += 1
        i += 1
    return count


def _find_unmatched_braces(text: str) -> tuple[int, int]:
    """Find counts of unmatched opening and closing braces.
    
    Returns:
        Tuple of (unmatched_open, unmatched_close) counts.
    """
    depth = 0
    unmatched_close = 0
    
    i = 0
    while i < len(text):
        # Skip escaped braces
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth > 0:
                depth -= 1
            else:
                unmatched_close += 1
        i += 1
    
    return depth, unmatched_close  # depth = unmatched_open


def balance_dollar_signs(text: str) -> str:
    """Balance unmatched dollar sign delimiters in LaTeX formulas.
    
    Handles both inline ($...$) and display ($$...$$) math modes.
    For unbalanced delimiters, attempts to close them at line end or
    remove orphan delimiters.
    
    Args:
        text: Text potentially containing LaTeX formulas.
        
    Returns:
        Text with balanced dollar sign delimiters.
    """
    if "$" not in text:
        return text
    
    lines = text.split("\n")
    result_lines = []
    
    for line in lines:
        # Skip lines without dollar signs
        if "$" not in line:
            result_lines.append(line)
            continue
        
        # Count unescaped dollar signs
        dollar_count = _count_unescaped(line, "$")
        
        if dollar_count % 2 == 0:
            # Even count - likely balanced
            result_lines.append(line)
            continue
        
        # Odd count - try to fix
        # Strategy: Find the last $ and check if it's likely opening or closing
        
        # Find positions of all unescaped $
        positions = []
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "$":
                i += 2
                continue
            if line[i] == "$":
                positions.append(i)
            i += 1
        
        if len(positions) == 1:
            # Single $ - likely unclosed inline math
            # Check if there's content after it that looks like math
            pos = positions[0]
            after = line[pos + 1:].strip()
            
            if after and not after.endswith("$"):
                # Add closing $ at end of meaningful content
                # Find end of math-like content (before punctuation/whitespace)
                end_pos = len(line)
                for j in range(len(line) - 1, pos, -1):
                    if line[j] in ".,;:!? \t":
                        end_pos = j
                    else:
                        break
                line = line[:end_pos] + "$" + line[end_pos:]
            else:
                # Orphan $ at end or no content - remove it
                line = line[:pos] + line[pos + 1:]
        
        result_lines.append(line)
    
    return "\n".join(result_lines)


def close_unclosed_braces(text: str) -> str:
    """Close unclosed braces in LaTeX commands.
    
    Scans for common LaTeX commands with unclosed braces and attempts
    to close them at reasonable positions.
    
    Args:
        text: Text with potential unclosed LaTeX braces.
        
    Returns:
        Text with braces closed where possible.
    """
    if "{" not in text:
        return text
    
    # Process line by line for better locality
    lines = text.split("\n")
    result_lines = []
    
    for line in lines:
        unmatched_open, unmatched_close = _find_unmatched_braces(line)
        
        if unmatched_open > 0:
            # Add closing braces at end of line
            line = line.rstrip() + "}" * unmatched_open
        
        if unmatched_close > 0:
            # Remove orphan closing braces (less common, be conservative)
            # Only remove if at start of content
            stripped = line.lstrip()
            leading_space = line[:len(line) - len(stripped)]
            
            removed = 0
            while stripped.startswith("}") and removed < unmatched_close:
                stripped = stripped[1:]
                removed += 1
            
            line = leading_space + stripped
        
        result_lines.append(line)
    
    return "\n".join(result_lines)


def fix_common_latex_commands(text: str) -> str:
    """Fix common OCR errors in LaTeX command syntax.
    
    Applies regex-based fixes for common issues like:
    - Spaces between command and opening brace
    - Spaces before subscripts/superscripts
    
    Args:
        text: Text with potential LaTeX command errors.
        
    Returns:
        Text with common command syntax errors fixed.
    """
    for pattern, replacement in LATEX_COMMAND_FIXES:
        text = re.sub(pattern, replacement, text)
    
    return text


def fix_latex_formulas(text: str, config: dict[str, Any]) -> str:
    """Apply all LaTeX fixing operations based on configuration.
    
    Args:
        text: Text containing LaTeX formulas.
        config: LaTeX fixing configuration dict.
        
    Returns:
        Text with LaTeX formulas fixed according to config.
    """
    if not config.get("enabled", True):
        return text
    
    if config.get("fix_common_commands", True):
        text = fix_common_latex_commands(text)
    
    if config.get("balance_dollar_signs", True):
        text = balance_dollar_signs(text)
    
    if config.get("close_unclosed_braces", True):
        text = close_unclosed_braces(text)
    
    return text


# ============================================================================
# Hyphenation Merging
# ============================================================================

def merge_hyphenation(text: str) -> str:
    """Merge words split across lines with a hyphen.
    
    Example: "politi-\\nche" -> "politiche"
    
    Uses conservative heuristics to avoid damaging genuine hyphenated
    compounds like "Jean-Baptiste" (only merges lowercase fragments).
    
    Args:
        text: Text with potential line-break hyphenation.
        
    Returns:
        Text with hyphenated line breaks merged where appropriate.
    """
    def _replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        # Only merge if both ends are lowercase letters
        if (
            left[-1].isalpha()
            and right[0].isalpha()
            and left[-1].islower()
            and right[0].islower()
        ):
            return left + right
        # Keep original hyphenation
        return left + "-\n" + right
    
    return _HYPHEN_PATTERN.sub(_replace, text)


# ============================================================================
# Whitespace Normalization
# ============================================================================

def normalize_whitespace(
    text: str,
    collapse_internal: bool = True,
    max_blank_lines: int = 2,
    tab_size: int = 4,
) -> str:
    """Normalize whitespace in text.
    
    Operations:
    - Expand tabs to spaces
    - Strip trailing spaces from lines
    - Collapse internal runs of spaces (optional)
    - Limit consecutive blank lines
    
    Args:
        text: Input text.
        collapse_internal: If True, collapse 3+ internal spaces to 2.
        max_blank_lines: Maximum consecutive blank lines to keep.
        tab_size: Number of spaces per tab.
        
    Returns:
        Whitespace-normalized text.
    """
    if not text:
        return text
    
    # Expand tabs
    text = text.expandtabs(tab_size)
    
    lines = text.splitlines()
    result_lines: list[str] = []
    blank_run = 0
    
    for line in lines:
        # Strip trailing spaces (keep leading indentation)
        line = line.rstrip(" ")
        
        # Collapse internal space runs
        if collapse_internal:
            line = re.sub(r"(?<=\S) {3,}(?=\S)", "  ", line)
        
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_blank_lines:
                result_lines.append("")
        else:
            blank_run = 0
            result_lines.append(line)
    
    # Ensure single trailing newline
    return "\n".join(result_lines) + "\n"


# ============================================================================
# Line Wrapping
# ============================================================================

def _should_wrap_line(line: str) -> bool:
    """Determine if a line should be wrapped.
    
    Skip wrapping for:
    - Empty/whitespace lines
    - Markdown headings
    - Page markers
    - Image annotations
    - Markdown tables
    - Lines with LaTeX display math
    
    Args:
        line: Line to evaluate.
        
    Returns:
        True if line should be wrapped, False otherwise.
    """
    if not line.strip():
        return False
    
    stripped = line.lstrip()
    
    # Markdown heading
    if stripped.startswith("#"):
        return False
    
    # Page markers (common in transcription output)
    if stripped.startswith("<page") or stripped.startswith("[Page"):
        return False
    
    # Image annotations
    if stripped.startswith("[") and "Image:" in stripped and stripped.endswith("]"):
        return False
    
    # Markdown table rows
    if stripped.startswith("|") and stripped.endswith("|"):
        return False
    
    # Display math ($$...$$)
    if stripped.startswith("$$") or stripped.endswith("$$"):
        return False
    
    return True


def compute_auto_wrap_width(text: str) -> int:
    """Compute automatic wrap width from text block statistics.
    
    Uses average line length of text blocks (consecutive non-empty lines)
    with at least 3 lines to avoid distortion from headings.
    
    Args:
        text: Text to analyze.
        
    Returns:
        Computed wrap width (minimum 20, default 80 if insufficient data).
    """
    lines = text.splitlines()
    
    # Group lines into blocks
    blocks: list[list[int]] = []
    current_block: list[int] = []
    
    for line in lines:
        if line.strip():
            current_block.append(len(line))
        elif current_block:
            blocks.append(current_block)
            current_block = []
    
    if current_block:
        blocks.append(current_block)
    
    # Compute mean line length for blocks with 3+ lines
    block_means: list[float] = []
    for block in blocks:
        if len(block) >= 3:
            block_means.append(sum(block) / len(block))
    
    if not block_means:
        return 80  # Default fallback
    
    avg = sum(block_means) / len(block_means)
    return max(20, int(round(avg)))


def wrap_long_lines(text: str, width: int) -> str:
    """Wrap lines longer than specified width.
    
    Uses word-based wrapping that:
    - Preserves leading indentation
    - Skips structural lines (headings, tables, etc.)
    - Hard-breaks if no space found
    
    Args:
        text: Text to wrap.
        width: Target line width.
        
    Returns:
        Text with long lines wrapped.
    """
    if width <= 0:
        return text
    
    lines = text.splitlines()
    result_lines: list[str] = []
    
    for line in lines:
        # Skip short or structural lines
        if len(line) <= width or not _should_wrap_line(line):
            result_lines.append(line)
            continue
        
        # Preserve indentation
        indent_len = len(line) - len(line.lstrip(" "))
        indent = line[:indent_len]
        content = line[indent_len:].strip()
        
        if not content:
            result_lines.append(line)
            continue
        
        max_content_width = max(1, width - indent_len)
        
        while len(content) > max_content_width:
            # Find last space before limit
            break_pos = content.rfind(" ", 0, max_content_width + 1)
            if break_pos <= 0:
                # No space - hard break
                break_pos = max_content_width
            
            segment = content[:break_pos].rstrip()
            result_lines.append(indent + segment)
            content = content[break_pos:].lstrip()
        
        result_lines.append(indent + content)
    
    return "\n".join(result_lines) + "\n"


# ============================================================================
# Main Cleaning Pipeline
# ============================================================================

def clean_transcription(text: str, config: dict[str, Any] | None = None) -> str:
    """Run the full text cleaning pipeline on transcription text.
    
    Applies cleaning stages in order based on configuration:
    1. Unicode normalization
    2. LaTeX formula fixing
    3. Hyphenation merging (optional)
    4. Whitespace normalization
    5. Line wrapping (optional)
    
    Args:
        text: Raw transcription text to clean.
        config: Cleaning configuration dict. If None, loads from config file.
        
    Returns:
        Cleaned text.
    """
    if not text:
        return text
    
    # Load config if not provided
    if config is None:
        config = get_text_cleaning_config()
    
    # Check if cleaning is globally enabled
    if not config.get("enabled", True):
        return text
    
    # 1. Unicode normalization
    if config.get("unicode_normalization", True):
        text = normalize_unicode(text)
    
    # 2. LaTeX formula fixing
    latex_config = config.get("latex_fixing", {})
    if latex_config.get("enabled", True):
        text = fix_latex_formulas(text, latex_config)
    
    # 3. Hyphenation merging (off by default - can damage compound words)
    if config.get("merge_hyphenation", False):
        text = merge_hyphenation(text)
    
    # 4. Whitespace normalization
    ws_config = config.get("whitespace_normalization", {})
    if ws_config.get("enabled", True):
        text = normalize_whitespace(
            text,
            collapse_internal=ws_config.get("collapse_internal_spaces", True),
            max_blank_lines=ws_config.get("max_blank_lines", 2),
            tab_size=ws_config.get("tab_size", 4),
        )
    
    # 5. Line wrapping (off by default)
    wrap_config = config.get("line_wrapping", {})
    if wrap_config.get("enabled", False):
        if wrap_config.get("auto_width", False):
            width = compute_auto_wrap_width(text)
        else:
            width = wrap_config.get("fixed_width", 80)
        text = wrap_long_lines(text, width)
    
    return text


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "clean_transcription",
    "get_text_cleaning_config",
    "normalize_unicode",
    "fix_latex_formulas",
    "merge_hyphenation",
    "normalize_whitespace",
    "wrap_long_lines",
]
