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
    >>> from pipeline.text_cleaner import clean_transcription, get_text_cleaning_config
    >>> config = get_text_cleaning_config()
    >>> cleaned_text = clean_transcription(raw_text, config)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from config.loader import get_config_loader
from config.logger import setup_logger

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
    "\u00ad",  # SOFT HYPHEN
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u2060",  # WORD JOINER
}

# Common LaTeX command typos from OCR
LATEX_COMMAND_FIXES = [
    (r"\\frac\s*\{", r"\\frac{"),  # Remove space after \frac
    (r"\\sqrt\s*\{", r"\\sqrt{"),  # Remove space after \sqrt
    (r"\\sum\s*_", r"\\sum_"),  # Remove space before subscript
    (r"\\int\s*_", r"\\int_"),  # Remove space before subscript
    (r"\\prod\s*_", r"\\prod_"),  # Remove space before subscript
    (r"\\lim\s*_", r"\\lim_"),  # Remove space before subscript
    (r"\\mathrm\s*\{", r"\\mathrm{"),
    (r"\\mathbf\s*\{", r"\\mathbf{"),
    (r"\\mathit\s*\{", r"\\mathit{"),
    (r"\\text\s*\{", r"\\text{"),
]

# Regex pattern for hyphenated line breaks
_HYPHEN_PATTERN = re.compile(r"(\w{3,})-\n(\w{2,})")

# `\left`/`\right` as delimiter commands (not \leftarrow, \rightarrow, etc.):
# the trailing negative lookahead excludes command names that merely start
# with "left"/"right".
_LEFT_CMD = re.compile(r"\\left(?![a-zA-Z])")
_RIGHT_CMD = re.compile(r"\\right(?![a-zA-Z])")

# Display math blocks and math spans used for scoped, in-block LaTeX repairs.
_DISPLAY_BLOCK = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
_ENV_BEGIN = re.compile(r"\\begin\{[^}]*\}")
_ENV_END = re.compile(r"\\end\{[^}]*\}")

# Single-token HTML sub/superscript, optionally wrapped in markdown emphasis
# asterisks (e.g. "*A*<sub>m</sub>", "x<sup>2</sup>"). The leading lookbehind
# keeps the base token off the tail of a longer word; the `</\2>` backreference
# requires a matching closing tag. Kept deliberately narrow: 1-3 alphanumerics.
_HTML_SUBSUP = re.compile(
    r"(?<![A-Za-z0-9])\*?([A-Za-z0-9]{1,3})\*?<(sub|sup)>([A-Za-z0-9]{1,3})</\2>"
)

# Prefixes that are almost always genuinely hyphenated. Following the
# conservative "keep the hyphen when unsure" policy, a line-break hyphen whose
# left fragment is one of these is preserved (for example "co-ordinating",
# "self-evident") rather than merged. The set is deliberately small and
# high-precision: common word-initial syllables such as "con", "pre" or "de"
# are excluded because line-break hyphenation of ordinary words (for example
# "concep-tion") far outnumbers genuine compounds starting that way.
HYPHEN_KEEP_PREFIXES = frozenset(
    {
        "co",
        "self",
        "non",
        "anti",
        "pseudo",
        "quasi",
        "semi",
        "multi",
        "cross",
        "well",
        "ill",
        "half",
        "vice",
        "all",
        "neo",
    }
)


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
        "latex_fixing": cleaning_cfg.get(
            "latex_fixing",
            {
                "enabled": True,
                "balance_dollar_signs": True,
                "close_unclosed_braces": True,
                "fix_common_commands": True,
                "normalize_math_delimiters": True,
                "balance_left_right": True,
                "convert_html_subsup": True,
            },
        ),
        "merge_hyphenation": cleaning_cfg.get("merge_hyphenation", True),
        "whitespace_normalization": cleaning_cfg.get(
            "whitespace_normalization",
            {
                "enabled": True,
                "collapse_internal_spaces": True,
                "max_blank_lines": 2,
                "tab_size": 4,
            },
        ),
        "line_wrapping": cleaning_cfg.get(
            "line_wrapping",
            {
                "enabled": False,
                "auto_width": False,
                "fixed_width": 80,
            },
        ),
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
            after = line[pos + 1 :].strip()

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
                line = line[:pos] + line[pos + 1 :]

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
            leading_space = line[: len(line) - len(stripped)]

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


def normalize_math_delimiters(text: str) -> str:
    """Normalize alternate LaTeX math delimiters to dollar-sign form.

    Maps ``\\(...\\)`` to inline ``$...$`` and ``\\[...\\]`` to display
    ``$$...$$``. This is a safe textual substitution: the delimiter tokens
    only occur as math markers, so the mapping preserves the enclosed content
    while unifying the delimiter style the downstream renderer expects.

    Args:
        text: Text potentially using ``\\(...\\)`` / ``\\[...\\]`` delimiters.

    Returns:
        Text with math delimiters normalized to dollar-sign form.
    """
    if not any(tok in text for tok in ("\\(", "\\)", "\\[", "\\]")):
        return text

    # Display first, then inline (distinct tokens, so order is not critical).
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    text = text.replace("\\(", "$").replace("\\)", "$")
    return text


def _apply_outside_math(text: str, transform: Any) -> str:
    """Apply ``transform`` only to text outside ``$...$`` / ``$$...$$`` spans.

    Math spans are passed through byte-identically so repairs meant for prose
    never touch already-delimited math.

    Args:
        text: Input text possibly containing math spans.
        transform: Callable mapping a plain-text segment to its replacement.

    Returns:
        Text with ``transform`` applied to non-math segments only.
    """
    result: list[str] = []
    last = 0
    for match in _MATH_SPAN.finditer(text):
        result.append(transform(text[last : match.start()]))
        result.append(match.group(0))  # math span left untouched
        last = match.end()
    result.append(transform(text[last:]))
    return "".join(result)


def convert_html_subsup(text: str) -> str:
    """Convert single-token HTML sub/superscripts to inline math.

    Rewrites ``X<sub>y</sub>`` -> ``$X_y$`` and ``X<sup>y</sup>`` -> ``$X^y$``
    (optionally with markdown emphasis asterisks around the base, e.g.
    ``*A*<sub>m</sub>``), but only when the pattern appears OUTSIDE existing
    math delimiters. Scope is intentionally narrow -- 1-3 alphanumeric base and
    script tokens -- to avoid touching genuine HTML or prose.

    Args:
        text: Text possibly containing HTML sub/superscript notation.

    Returns:
        Text with qualifying HTML sub/superscripts rewritten as inline math.
    """
    if "<sub>" not in text and "<sup>" not in text:
        return text

    def _repl(match: re.Match[str]) -> str:
        base, kind, script = match.group(1), match.group(2), match.group(3)
        op = "_" if kind == "sub" else "^"
        return f"${base}{op}{script}$"

    return _apply_outside_math(text, lambda seg: _HTML_SUBSUP.sub(_repl, seg))


def balance_left_right(text: str) -> str:
    """Balance ``\\left``/``\\right`` pairs within each display math block.

    Operates conservatively and only inside a single ``$$...$$`` block: if the
    block has more ``\\right`` than ``\\left``, the missing openers are added as
    ``\\left.`` at the block (or enclosing environment) start; the mirror case
    appends ``\\right.`` at the block end. Blocks whose counts already match --
    including misordered-but-balanced or nested cases -- are left unchanged, and
    inline ``$...$`` math is never touched.

    Args:
        text: Text potentially containing unbalanced ``\\left``/``\\right``.

    Returns:
        Text with per-block ``\\left``/``\\right`` counts balanced.
    """
    if "\\left" not in text and "\\right" not in text:
        return text

    def _fix_block(match: re.Match[str]) -> str:
        inner = match.group(1)
        n_left = len(_LEFT_CMD.findall(inner))
        n_right = len(_RIGHT_CMD.findall(inner))
        if n_left == n_right:
            return match.group(0)  # balanced (or ambiguous) -> leave unchanged

        if n_right > n_left:
            # Unmatched \right: prepend \left. inside the enclosing environment
            # (after \begin{...} when present) so the opener stays in scope.
            prefix = "\\left." * (n_right - n_left)
            begin = _ENV_BEGIN.search(inner)
            if begin:
                inner = inner[: begin.end()] + prefix + inner[begin.end() :]
            else:
                inner = prefix + inner
        else:
            # Unmatched \left: append \right. before the environment close.
            suffix = "\\right." * (n_left - n_right)
            end = _ENV_END.search(inner)
            if end:
                inner = inner[: end.start()] + suffix + inner[end.start() :]
            else:
                inner = inner + suffix

        return "$$" + inner + "$$"

    return _DISPLAY_BLOCK.sub(_fix_block, text)


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

    # Unify delimiters first so later steps see canonical $...$ / $$...$$ forms.
    if config.get("normalize_math_delimiters", True):
        text = normalize_math_delimiters(text)

    # Convert HTML sub/sup (outside math) before dollar balancing, since it
    # introduces balanced inline $...$ spans.
    if config.get("convert_html_subsup", True):
        text = convert_html_subsup(text)

    if config.get("fix_common_commands", True):
        text = fix_common_latex_commands(text)

    if config.get("balance_left_right", True):
        text = balance_left_right(text)

    if config.get("balance_dollar_signs", True):
        text = balance_dollar_signs(text)

    if config.get("close_unclosed_braces", True):
        text = close_unclosed_braces(text)

    return text


# ============================================================================
# Hyphenation Merging
# ============================================================================


def should_keep_hyphen(left: str, right: str) -> bool:
    """Decide whether a line-break hyphen is a genuine compound to keep.

    Implements the conservative "keep the hyphen when unsure" policy. The
    hyphen is kept (the word is treated as a real hyphenated compound) when:
    the characters adjacent to the hyphen are not both lowercase letters (for
    example "Jean-Baptiste", "page-42"); or the alphabetic fragment ending at
    the hyphen is one of ``HYPHEN_KEEP_PREFIXES``. Otherwise the break is
    treated as wrap/line-break hyphenation and the word should be merged.

    Args:
        left: The fragment before the hyphen (for example "Manage", "co").
        right: The continuation fragment after the line break (for example
            "ment", "ordinating").

    Returns:
        True to keep the hyphen, False to merge into a single word.
    """
    if not left or not right:
        return True
    if not (left[-1].isalpha() and right[0].isalpha()):
        return True
    if not (left[-1].islower() and right[0].islower()):
        # Uppercase-adjacent. An all-caps word split mid-word (e.g. "KNOWL-EDGE",
        # "MAJ-ESTY") is not a compound and should merge; a Title-case pairing
        # (e.g. "Jean-Baptiste", "18th-Century") is a genuine compound to keep.
        return not (left.isupper() and right.isupper())
    match = re.search(r"[A-Za-z]+$", left)
    prefix_word = match.group(0).lower() if match else left.lower()
    return prefix_word in HYPHEN_KEEP_PREFIXES


def merge_hyphenation(text: str) -> str:
    """Merge words split across lines with a hyphen.

    Example: "politi-\\nche" -> "politiche"

    Uses the conservative ``should_keep_hyphen`` guard to avoid damaging
    genuine hyphenated compounds like "Jean-Baptiste" or "co-ordinating",
    while still merging ordinary line-break hyphenation like "Manage-\\nment".

    Args:
        text: Text with potential line-break hyphenation.

    Returns:
        Text with hyphenated line breaks merged where appropriate.
    """

    def _replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if should_keep_hyphen(left, right):
            # Keep original hyphenation (compound or ambiguous fragment)
            return left + "-\n" + right
        return left + right

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
    return not (stripped.startswith("$$") or stripped.endswith("$$"))


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
        logger.warning(
            "Line wrapping is enabled; it re-wraps already laid-out LLM "
            "transcription text and is the known cause of spurious mid-paragraph "
            "line breaks. It is usually not needed for LLM output -- consider "
            "setting text_cleaning.line_wrapping.enabled to false."
        )
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
    "normalize_math_delimiters",
    "convert_html_subsup",
    "balance_left_right",
    "merge_hyphenation",
    "should_keep_hyphen",
    "normalize_whitespace",
    "wrap_long_lines",
]
