"""LaTeX formulas in summary text and their native Word equations (OMML).

``parse_latex_in_text`` splits text into prose and ``$...$`` / ``$$...$$``
formulas; ``add_math_to_paragraph`` converts a formula through MathML to OMML
and appends it to a python-docx paragraph, falling back to simplified LaTeX
and finally to italic text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any
from xml.etree import ElementTree as ET

import mathml2omml
from docx.oxml import parse_xml
from latex2mathml.converter import convert as latex_to_mathml

from autoexcerpter.constants import MATH_NAMESPACE
from autoexcerpter.rendering.summary import sanitize_for_xml

logger = logging.getLogger(__name__)


_MATH_INDICATORS = frozenset("\\^_={}")


def _is_currency_span(content: str) -> bool:
    """Return True when a ``$...$`` capture is prose currency, not inline math.

    "Wages rose from $3 to $5" would otherwise be parsed as inline math over
    "3 to ", destroying the sentence. A capture opening on a digit and holding
    no math indicator (``\\``, ``^``, ``_``, ``=``, ``{``, ``}``) is treated as
    a price range and left as literal text.
    """
    if not content or not content[0].isdigit():
        return False
    return not any(ch in _MATH_INDICATORS for ch in content)


def parse_latex_in_text(text: str) -> list[tuple[str, str]]:
    """Parse text and split it into segments of regular text and LaTeX formulas.

    Handles:
    - Display math: $$...$$ (processed first to avoid conflicts)
    - Inline math: $...$ (single dollar, non-greedy)
    - Escaped dollar signs: \\$ are preserved as literal $
    - Multi-line formulas (DOTALL mode)
    - Nested braces within formulas

    Returns:
        List of (content, type) tuples where type is 'text', 'latex_display', or
        'latex_inline'.
    """
    if not text:
        return [("", "text")]

    ESCAPED_DOLLAR_PLACEHOLDER = "\x00ESCAPED_DOLLAR\x00"
    protected_text = text.replace("\\$", ESCAPED_DOLLAR_PLACEHOLDER)

    segments: list[tuple[str, str]] = []

    display_pattern = r"\$\$(.+?)\$\$"
    inline_pattern = r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"

    # (content, type, start, end); display math is extracted first.
    temp_segments: list[tuple[str, str, int, int]] = []

    for match in re.finditer(display_pattern, protected_text, re.DOTALL):
        temp_segments.append(
            (match.group(1), "latex_display", match.start(), match.end())
        )

    display_ranges = [(s[2], s[3]) for s in temp_segments]

    def overlaps_display_range(start: int, end: int) -> bool:
        # True if [start, end) intersects any display range, including the case
        # where the inline match fully encloses a $$...$$ block (both endpoints
        # outside the range). Half-open interval overlap test.
        return any(start < d_end and d_start < end for d_start, d_end in display_ranges)

    inline_re = re.compile(inline_pattern, re.DOTALL)
    pos = 0
    while (inline_match := inline_re.search(protected_text, pos)) is not None:
        if overlaps_display_range(inline_match.start(), inline_match.end()):
            # Do not consume the whole rejected span: when the candidate opens
            # at a currency "$" before a $$...$$ block, its closing "$" is the
            # OPENING "$" of the first genuine formula after the block. Resume
            # just past the candidate's opening "$" (mirroring the currency
            # branch below) so later dollars keep pairing correctly.
            pos = inline_match.start() + 1
        elif _is_currency_span(
            # Judge the span on its restored text: the escaped-dollar
            # placeholder contains "_", a math indicator, so a price span
            # holding "\$" would otherwise be misread as inline math.
            inline_match.group(1).replace(ESCAPED_DOLLAR_PLACEHOLDER, "$")
        ):
            # Do not consume the whole rejected span: with an odd dollar count
            # ("$5 ... $x$") the closing "$" of the currency capture is the
            # OPENING "$" of a genuine formula. Resume just past the opening
            # currency "$" so later dollars can still pair as math.
            pos = inline_match.start() + 1
        else:
            temp_segments.append(
                (
                    inline_match.group(1),
                    "latex_inline",
                    inline_match.start(),
                    inline_match.end(),
                )
            )
            pos = inline_match.end()

    temp_segments.sort(key=lambda x: x[2])

    last_end = 0
    for content, seg_type, start, end in temp_segments:
        if start > last_end:
            text_segment = protected_text[last_end:start]
            text_segment = text_segment.replace(ESCAPED_DOLLAR_PLACEHOLDER, "$")
            if text_segment:
                segments.append((text_segment, "text"))

        restored_content = content.replace(ESCAPED_DOLLAR_PLACEHOLDER, "$")
        segments.append((restored_content, seg_type))
        last_end = end

    if last_end < len(protected_text):
        remaining = protected_text[last_end:]
        remaining = remaining.replace(ESCAPED_DOLLAR_PLACEHOLDER, "$")
        if remaining:
            segments.append((remaining, "text"))

    if not segments:
        return [(text, "text")]

    return segments


def normalize_latex_whitespace(latex_code: str) -> str:
    """Normalize whitespace in LaTeX formulas for consistent processing."""
    normalized = latex_code.strip()
    normalized = re.sub(r" +", " ", normalized)
    normalized = re.sub(r"\s*([=+\-*/^_{}])\s*", r"\1", normalized)
    return normalized


def _regex_rule(
    pattern: str, replacement: str, desc: str, flags: int = 0
) -> tuple[
    Callable[[str], bool],
    Callable[[str], str],
    str,
]:
    """Create a (should_apply, transform, description) tuple for regex-based rules."""
    compiled = re.compile(pattern, flags)

    def should_apply(text: str) -> bool:
        return bool(compiled.search(text))

    def transform(text: str) -> str:
        return compiled.sub(replacement, text)

    return should_apply, transform, desc


def _literal_rule(
    old: str, new: str, desc: str
) -> tuple[
    Callable[[str], bool],
    Callable[[str], str],
    str,
]:
    """Create a (should_apply, transform, description) tuple for literal rules."""

    def should_apply(text: str) -> bool:
        return old in text

    def transform(text: str) -> str:
        return text.replace(old, new)

    return should_apply, transform, desc


_DELIMITER_SIZES = [
    "\\Big",
    "\\big",
    "\\bigg",
    "\\Bigg",
    "\\Bigl",
    "\\Bigr",
    "\\bigl",
    "\\bigr",
    "\\biggl",
    "\\biggr",
    "\\Biggl",
    "\\Biggr",
    "\\bigm",
    "\\Bigm",
]
_SIZED_DELIMITERS = ["(", ")", "[", "]", "|", "\\{", "\\}", "\\|", "."]


def _has_delimiter_sizing(text: str) -> bool:
    """Return True when *text* holds a sized delimiter such as ``\\big(``."""
    return any(
        f"{size}{delim}" in text
        for size in _DELIMITER_SIZES
        for delim in _SIZED_DELIMITERS
    )


def _simplify_delimiter_sizing(text: str) -> str:
    """Replace all delimiter size commands with plain delimiters."""
    for size in _DELIMITER_SIZES:
        for delim in _SIZED_DELIMITERS:
            target = f"{size}{delim}"
            if target in text:
                # "\{" / "\}" stay escaped: unescaping them would turn visible
                # braces into invisible LaTeX grouping. Only "\|" collapses.
                if delim == ".":
                    plain = ""
                elif delim == "\\|":
                    plain = "|"
                else:
                    plain = delim
                text = text.replace(target, plain)
    return text


def _simplify_auto_sizing(text: str) -> str:
    """Replace \\left/\\right auto-sizing delimiters."""
    text = text.replace("\\left.", "")
    text = text.replace("\\right.", "")
    text = re.sub(r"\\left\s*([(\[|])", r"\1", text)
    text = re.sub(r"\\right\s*([)\]|])", r"\1", text)
    text = re.sub(r"\\left\s*\\([{|}])", r"\\\1", text)
    text = re.sub(r"\\right\s*\\([{|}])", r"\\\1", text)
    text = text.replace("\\middle|", "|")
    text = text.replace("\\middle", "")
    return text


type _LaTeXRule = tuple[Callable[[str], bool], Callable[[str], str], str]

_LATEX_SIMPLIFICATIONS: list[_LaTeXRule] = [
    _literal_rule("\\land", "\\wedge", "logical operator (\\land -> \\wedge)"),
    _literal_rule("\\lor", "\\vee", "logical operator (\\lor -> \\vee)"),
    _literal_rule("\\lnot", "\\neg", "logical operator (\\lnot -> \\neg)"),
    _literal_rule(
        "\\iff", "\\Leftrightarrow", "logical operator (\\iff -> \\Leftrightarrow)"
    ),
    _literal_rule(
        "\\implies", "\\Rightarrow", "logical operator (\\implies -> \\Rightarrow)"
    ),
    _regex_rule(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", "text -> mathrm"),
    _regex_rule(r"\\textrm\{([^}]*)\}", r"\\mathrm{\1}", "textrm -> mathrm"),
    _regex_rule(r"\\textit\{([^}]*)\}", r"\\mathit{\1}", "textit -> mathit"),
    _regex_rule(r"\\textbf\{([^}]*)\}", r"\\mathbf{\1}", "textbf -> mathbf"),
    _regex_rule(r"\\bar\{([^}]+)\}", r"\\overline{\1}", "accent (bar -> overline)"),
    _regex_rule(
        r"\\tilde\{([^}]+)\}", r"\\widetilde{\1}", "accent (tilde -> widetilde)"
    ),
    _regex_rule(r"\\hat\{([^}]+)\}", r"\\widehat{\1}", "accent (hat -> widehat)"),
    _regex_rule(
        r"\\vec\{([^}]+)\}", r"\\overrightarrow{\1}", "accent (vec -> overrightarrow)"
    ),
    _regex_rule(r"\\dot\{([^}]+)\}", r"\1", "accent (dot removed)"),
    _regex_rule(r"\\ddot\{([^}]+)\}", r"\1", "accent (ddot removed)"),
    (
        _has_delimiter_sizing,
        _simplify_delimiter_sizing,
        "delimiter sizing removed",
    ),
    (
        lambda text: "\\left" in text or "\\right" in text,
        _simplify_auto_sizing,
        "auto-sizing delimiters (left/right) removed",
    ),
    # Overbrace/underbrace. The optional script group is explicit and anchored
    # to the brace so it never devours following formula content ("a+b} + \frac
    # {c}{d}" must not collapse into "a+b{d}").
    _regex_rule(
        r"\\overbrace\{([^}]+)\}(?:\^(?:\{[^}]*\}|\S))?",
        r"\1",
        "overbrace simplified",
    ),
    _regex_rule(
        r"\\underbrace\{([^}]+)\}(?:_(?:\{[^}]*\}|\S))?",
        r"\1",
        "underbrace simplified",
    ),
    _regex_rule(r"\\overleftarrow\{([^}]+)\}", r"\1", "overleftarrow simplified"),
    _regex_rule(
        r"\\overrightarrow\{([^}]+)\}", r"\\vec{\1}", "overrightarrow simplified"
    ),
    _regex_rule(r"\\underleftarrow\{([^}]+)\}", r"\1", "underleftarrow simplified"),
    _regex_rule(r"\\underrightarrow\{([^}]+)\}", r"\1", "underrightarrow simplified"),
    _regex_rule(r"\\phantom\{[^}]*\}", "", "phantom removed"),
    _regex_rule(r"\\hphantom\{[^}]*\}", "", "hphantom removed"),
    _regex_rule(r"\\vphantom\{[^}]*\}", "", "vphantom removed"),
    _literal_rule("\\,", " ", "thin space normalized"),
    _literal_rule("\\;", " ", "thick space normalized"),
    _literal_rule("\\!", "", "negative thin space normalized"),
    _literal_rule("\\quad", " ", "quad normalized"),
    _literal_rule("\\qquad", "  ", "qquad normalized"),
    _regex_rule(r"\\hspace\{[^}]*\}", " ", "hspace normalized"),
    _regex_rule(r"\\vspace\{[^}]*\}", "", "vspace normalized"),
    _regex_rule(r"\\mspace\{[^}]*\}", " ", "mspace normalized"),
    _regex_rule(
        r"\\operatorname\{([^}]+)\}", r"\\mathrm{\1}", "operatorname -> mathrm"
    ),
    _regex_rule(
        r"\\begin\{align\*?\}(.+?)\\end\{align\*?\}",
        r"\1",
        "align env unwrapped",
        re.DOTALL,
    ),
    _regex_rule(
        r"\\begin\{gather\*?\}(.+?)\\end\{gather\*?\}",
        r"\1",
        "gather env unwrapped",
        re.DOTALL,
    ),
    _regex_rule(
        r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}",
        r"\1",
        "equation env unwrapped",
        re.DOTALL,
    ),
    _regex_rule(
        r"\\begin\{split\}(.+?)\\end\{split\}", r"\1", "split env unwrapped", re.DOTALL
    ),
    _regex_rule(
        r"\\begin\{cases\}(.+?)\\end\{cases\}", r"\1", "cases env unwrapped", re.DOTALL
    ),
    # Alignment markers. The row separator goes first so "\\&" degrades to a
    # bare alignment "&"; the negative lookbehind then spares an escaped "\&",
    # which is a printable ampersand rather than an alignment marker.
    (
        lambda text: "&" in text or "\\\\" in text,
        lambda text: re.sub(r"(?<!\\)&", " ", text.replace("\\\\", " ")),
        "alignment markers cleaned",
    ),
    (
        lambda text: "\\limits" in text or "\\nolimits" in text,
        lambda text: text.replace("\\limits", "").replace("\\nolimits", ""),
        "limits modifiers removed",
    ),
]


def simplify_problematic_latex(latex_code: str) -> tuple[str, list[str]]:
    """Simplify LaTeX code by removing constructs known to cause mathml2omml issues.

    Returns:
        Tuple of (simplified_latex, list_of_applied_simplifications)
    """
    simplified = latex_code
    applied: list[str] = []

    original_len = len(simplified)
    simplified = normalize_latex_whitespace(simplified)
    if len(simplified) != original_len:
        applied.append("whitespace normalization")

    for should_apply, transform, desc in _LATEX_SIMPLIFICATIONS:
        if should_apply(simplified):
            simplified = transform(simplified)
            applied.append(desc)

    simplified = re.sub(r"\s+", " ", simplified).strip()

    return simplified, applied


def sanitize_omml_xml(omml_markup: str) -> str:
    """Attempt to fix common XML issues in OMML markup generated by mathml2omml."""
    try:
        ET.fromstring(omml_markup)
        return omml_markup
    except ET.ParseError as e:
        # Gate on the markup itself, not the parser message: expat error
        # strings never name the offending tag ("mismatched tag: line 1,
        # column 158"), so matching "groupChr" against them would leave the
        # repair logic unreachable. The repairs are no-ops when no
        # groupChrPr construct exists.
        logger.debug("OMML parse failed, attempting sanitization: %s", e)

        if "<m:groupChrPr" in omml_markup:
            fixed_markup = omml_markup

            pattern = r"<m:groupChrPr[^>]*>"
            positions = [
                (m.start(), m.end()) for m in re.finditer(pattern, fixed_markup)
            ]

            for _start, end in reversed(positions):
                after_tag = fixed_markup[end:]
                close_prop = after_tag.find("</m:groupChrPr>")
                close_group = after_tag.find("</m:groupChr>")

                if close_group != -1 and (close_prop == -1 or close_prop > close_group):
                    insert_pos = end + close_group
                    fixed_markup = (
                        fixed_markup[:insert_pos]
                        + "</m:groupChrPr>"
                        + fixed_markup[insert_pos:]
                    )

            try:
                ET.fromstring(fixed_markup)
                logger.info("Successfully sanitized OMML with groupChr fix")
                return fixed_markup
            except ET.ParseError as parse_err:
                logger.debug("First sanitization attempt failed: %s", parse_err)

                fixed_markup2 = omml_markup
                max_iterations = 10
                for _ in range(max_iterations):
                    match = re.search(
                        r"<m:groupChr>\s*<m:groupChrPr[^>]*>(?:(?!</m:groupChrPr>).)*?</m:groupChr>",
                        fixed_markup2,
                        flags=re.DOTALL,
                    )
                    if not match:
                        break
                    block = match.group(0)
                    fixed_block = block.replace(
                        "</m:groupChr>", "</m:groupChrPr></m:groupChr>", 1
                    )
                    fixed_markup2 = (
                        fixed_markup2[: match.start()]
                        + fixed_block
                        + fixed_markup2[match.end() :]
                    )

                try:
                    ET.fromstring(fixed_markup2)
                    logger.info(
                        "Successfully sanitized OMML with iterative groupChr fix"
                    )
                    return fixed_markup2
                except ET.ParseError:
                    logger.debug("Iterative sanitization failed, trying fallback")

                    fixed_markup3 = re.sub(
                        r"<m:groupChr>\s*<m:groupChrPr[^>]*>.*?</m:groupChr>",
                        lambda m: (
                            m.group(0)
                            .replace("<m:groupChrPr", "<!--m:groupChrPr")
                            .replace("</m:groupChrPr>", "-->")
                        ),
                        omml_markup,
                        flags=re.DOTALL,
                    )

                    try:
                        ET.fromstring(fixed_markup3)
                        logger.info(
                            "Successfully sanitized OMML by commenting out groupChrPr"
                        )
                        return fixed_markup3
                    except ET.ParseError:
                        pass

        return omml_markup


def _ensure_omml_namespace(omml_markup: str) -> str:
    """Ensure the OMML markup carries the math namespace and an <m:oMath> root."""
    if "xmlns:m" not in omml_markup.split("\n", 1)[0]:
        omml_markup = omml_markup.replace(
            "<m:oMath",
            f'<m:oMath xmlns:m="{MATH_NAMESPACE}"',
            1,
        )

    if not omml_markup.strip().startswith("<m:oMath"):
        omml_markup = f'<m:oMath xmlns:m="{MATH_NAMESPACE}">{omml_markup}</m:oMath>'

    return omml_markup


def _wrap_omml(omml_markup: str, display: bool) -> str:
    """Wrap OMML in a block-level ``m:oMathPara`` only for display formulas.

    Per ECMA-376 (Part 1, 22.1.2.78) ``m:oMathPara`` is a *math paragraph*:
    display-mode math rendered on its own line with its own justification.
    Inline formulas must be bare ``m:oMath`` children of ``w:p`` interleaved
    with the surrounding runs, or Word breaks the sentence around a centered
    equation. ``_ensure_omml_namespace`` guarantees the bare markup already
    carries the required ``xmlns:m`` declaration.
    """
    if display:
        return f'<m:oMathPara xmlns:m="{MATH_NAMESPACE}">{omml_markup}</m:oMathPara>'
    return omml_markup


def add_math_to_paragraph(
    paragraph: Any, latex_code: str, display: bool = False
) -> None:
    """Append *latex_code* to *paragraph* as a native Word equation (OMML).

    A formula that fails to convert is retried in simplified form and, failing
    that, written as italic text.

    Args:
        display: True for ``$$...$$`` display math (block-level, own line);
            False for ``$...$`` inline math flowing with the text.
    """
    try:
        mathml = latex_to_mathml(latex_code)
        omml_markup = _ensure_omml_namespace(mathml2omml.convert(mathml))

        omml_para = _wrap_omml(omml_markup, display)

        try:
            omml_element = parse_xml(omml_para)
        # python-docx parses with lxml, whose XMLSyntaxError derives from
        # SyntaxError (not from xml.etree's ParseError); without SyntaxError in
        # the tuple the sanitize-and-retry path below never runs and every
        # malformed equation degrades to italic text.
        except (ET.ParseError, ValueError, SyntaxError) as parse_error:
            logger.warning(
                "XML parsing failed, attempting to sanitize OMML: %s", parse_error
            )
            sanitized_omml = sanitize_omml_xml(omml_markup)
            sanitized_para = _wrap_omml(sanitized_omml, display)
            omml_element = parse_xml(sanitized_para)

        paragraph._p.append(omml_element)

    except Exception as exc:  # noqa: BLE001 - falls back to simplified LaTeX
        try:
            simplified_latex, simplifications = simplify_problematic_latex(latex_code)

            if simplified_latex != latex_code:
                logger.info(
                    "Attempting conversion with simplified LaTeX: %s",
                    ", ".join(simplifications),
                )
                mathml = latex_to_mathml(simplified_latex)
                omml_markup = _ensure_omml_namespace(mathml2omml.convert(mathml))

                omml_para = _wrap_omml(omml_markup, display)
                omml_element = parse_xml(omml_para)
                paragraph._p.append(omml_element)
                logger.info("Successfully converted simplified LaTeX")
                return
        except Exception:  # noqa: BLE001 - a failed retry falls back to text
            pass

        logger.warning(
            "Failed to convert LaTeX to MathML: %s. Displaying as text.", exc
        )
        logger.warning("Problematic LaTeX code: %s", latex_code[:200])
        run = paragraph.add_run(f" {sanitize_for_xml(latex_code)} ")
        run.font.name = "Cambria Math"
        run.italic = True


__all__ = [
    "add_math_to_paragraph",
    "normalize_latex_whitespace",
    "parse_latex_in_text",
    "sanitize_omml_xml",
    "simplify_problematic_latex",
]
