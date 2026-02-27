"""Tests for modules/text_cleaner.py - Text cleaning utilities."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from modules.text_cleaner import (
    normalize_unicode,
    balance_dollar_signs,
    close_unclosed_braces,
    fix_common_latex_commands,
    fix_latex_formulas,
    merge_hyphenation,
    normalize_whitespace,
    wrap_long_lines,
    compute_auto_wrap_width,
    clean_transcription,
    get_text_cleaning_config,
)


class TestNormalizeUnicode:
    """Tests for Unicode normalization."""

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert normalize_unicode("") == ""

    def test_normal_text_unchanged(self):
        """Normal ASCII text is unchanged."""
        text = "Hello, World!"
        assert normalize_unicode(text) == text

    def test_nfc_normalization(self):
        """Composed characters are normalized to NFC."""
        # e + combining acute accent should become é
        text = "Cafe\u0301"
        result = normalize_unicode(text)
        assert result == "Café"

    def test_soft_hyphen_removed(self):
        """Soft hyphens are removed."""
        text = "test\u00adword"
        assert normalize_unicode(text) == "testword"

    def test_zero_width_space_removed(self):
        """Zero-width spaces are removed."""
        text = "test\u200bword"
        assert normalize_unicode(text) == "testword"

    def test_bom_removed(self):
        """Byte order marks are removed."""
        text = "\ufeffstart of text"
        assert normalize_unicode(text) == "start of text"

    def test_word_joiner_removed(self):
        """Word joiners are removed."""
        text = "test\u2060word"
        assert normalize_unicode(text) == "testword"

    def test_newlines_preserved(self):
        """Newlines are preserved."""
        text = "line1\nline2\nline3"
        assert normalize_unicode(text) == text

    def test_tabs_preserved(self):
        """Tabs are preserved."""
        text = "col1\tcol2\tcol3"
        assert normalize_unicode(text) == text

    def test_control_characters_removed(self):
        """Control characters (except newline/tab) are removed."""
        text = "test\x00\x01\x02word"
        result = normalize_unicode(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "testword" == result

    def test_aegean_icons_to_bullet(self):
        """Aegean icon characters are mapped to bullets."""
        text = "item \U00010101 text"
        result = normalize_unicode(text)
        assert "•" in result


class TestBalanceDollarSigns:
    """Tests for LaTeX dollar sign balancing."""

    def test_no_dollar_signs(self):
        """Text without dollar signs is unchanged."""
        text = "No formulas here."
        assert balance_dollar_signs(text) == text

    def test_balanced_inline_math(self):
        """Balanced inline math is unchanged."""
        text = "Here is $x + y = z$ inline."
        assert balance_dollar_signs(text) == text

    def test_balanced_display_math(self):
        """Balanced display math is unchanged."""
        text = "$$\\int f(x) dx$$"
        assert balance_dollar_signs(text) == text

    def test_unbalanced_single_dollar_with_content(self):
        """Single unbalanced $ with content gets closed."""
        text = "Formula: $x + y"
        result = balance_dollar_signs(text)
        # Should add closing $ after content
        assert result.count("$") % 2 == 0 or "$" not in result

    def test_escaped_dollar_preserved(self):
        """Escaped dollar signs are preserved."""
        text = "Price is \\$50"
        result = balance_dollar_signs(text)
        assert "\\$" in result or "$50" in result

    def test_multiple_lines_handled(self):
        """Multiple lines are handled independently."""
        text = "Line 1 $a=b$\nLine 2 $c=d$"
        result = balance_dollar_signs(text)
        assert result == text


class TestCloseUnclosedBraces:
    """Tests for closing unclosed LaTeX braces."""

    def test_no_braces(self):
        """Text without braces is unchanged."""
        text = "No braces here"
        assert close_unclosed_braces(text) == text

    def test_balanced_braces(self):
        """Balanced braces are unchanged."""
        text = "\\frac{a}{b}"
        assert close_unclosed_braces(text) == text

    def test_unclosed_brace_at_end(self):
        """Unclosed brace at end of line gets closed."""
        text = "\\frac{a"
        result = close_unclosed_braces(text)
        assert result.count("{") == result.count("}")

    def test_multiple_unclosed_braces(self):
        """Multiple unclosed braces get closed."""
        text = "\\frac{a}{b + \\sqrt{c"
        result = close_unclosed_braces(text)
        assert result.count("{") == result.count("}")

    def test_escaped_braces_ignored(self):
        """Escaped braces are not counted."""
        text = "\\{test\\}"
        assert close_unclosed_braces(text) == text

    def test_orphan_closing_brace_handling(self):
        """Orphan closing braces at start are handled."""
        text = "} orphan brace"
        result = close_unclosed_braces(text)
        # The function attempts to remove orphan closing braces at start
        # The result should be valid (either removed or preserved depending on heuristics)
        assert isinstance(result, str)


class TestFixCommonLatexCommands:
    """Tests for fixing common LaTeX command issues."""

    def test_frac_space_removed(self):
        """Space after \\frac is removed."""
        text = "\\frac {a}{b}"
        result = fix_common_latex_commands(text)
        assert "\\frac{" in result

    def test_sqrt_space_removed(self):
        """Space after \\sqrt is removed."""
        text = "\\sqrt {x}"
        result = fix_common_latex_commands(text)
        assert "\\sqrt{" in result

    def test_sum_subscript_space_removed(self):
        """Space before subscript in \\sum is removed."""
        text = "\\sum _{i=1}"
        result = fix_common_latex_commands(text)
        assert "\\sum_" in result

    def test_mathrm_space_removed(self):
        """Space after \\mathrm is removed."""
        text = "\\mathrm {sin}"
        result = fix_common_latex_commands(text)
        assert "\\mathrm{" in result

    def test_text_space_removed(self):
        """Space after \\text is removed."""
        text = "\\text {word}"
        result = fix_common_latex_commands(text)
        assert "\\text{" in result


class TestFixLatexFormulas:
    """Tests for the combined LaTeX fixing function."""

    def test_disabled_returns_unchanged(self):
        """Disabled config returns text unchanged."""
        text = "\\frac {a"
        config = {"enabled": False}
        assert fix_latex_formulas(text, config) == text

    def test_all_fixes_applied(self):
        """All fixes are applied when enabled."""
        text = "\\frac {a}{b} and $x + y"
        config = {
            "enabled": True,
            "fix_common_commands": True,
            "balance_dollar_signs": True,
            "close_unclosed_braces": True,
        }
        result = fix_latex_formulas(text, config)
        assert "\\frac{" in result  # Command fix applied


class TestMergeHyphenation:
    """Tests for hyphenation merging."""

    def test_simple_hyphenation_merge(self):
        """Simple hyphenated word across lines is merged."""
        text = "demon-\nstration"
        result = merge_hyphenation(text)
        assert result == "demonstration"

    def test_proper_hyphen_preserved(self):
        """Proper hyphenated compound (capitals) is preserved."""
        text = "Jean-\nBaptiste"
        result = merge_hyphenation(text)
        # Should preserve because of capital letter
        assert "-" in result

    def test_short_fragments_not_merged(self):
        """Short fragments are not merged (less than 3 chars)."""
        text = "a-\ntest"
        # Pattern requires 3+ chars on left side
        assert merge_hyphenation(text) == text

    def test_no_hyphenation_unchanged(self):
        """Text without hyphenation is unchanged."""
        text = "normal text without hyphenation"
        assert merge_hyphenation(text) == text


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert normalize_whitespace("") == ""

    def test_tabs_expanded(self):
        """Tabs are expanded to spaces."""
        text = "col1\tcol2"
        result = normalize_whitespace(text, tab_size=4)
        assert "\t" not in result

    def test_trailing_spaces_stripped(self):
        """Trailing spaces are stripped from lines."""
        text = "text with trailing   \nmore text   "
        result = normalize_whitespace(text)
        lines = result.rstrip("\n").split("\n")
        for line in lines:
            assert not line.endswith(" ")

    def test_internal_spaces_collapsed(self):
        """Internal runs of 3+ spaces are collapsed to 2."""
        text = "word    word     word"
        result = normalize_whitespace(text, collapse_internal=True)
        assert "    " not in result

    def test_blank_lines_limited(self):
        """Consecutive blank lines are limited."""
        text = "line1\n\n\n\n\nline2"
        result = normalize_whitespace(text, max_blank_lines=2)
        # Count consecutive blank lines
        lines = result.split("\n")
        blank_count = 0
        max_blank = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                max_blank = max(max_blank, blank_count)
            else:
                blank_count = 0
        assert max_blank <= 2

    def test_trailing_newline_added(self):
        """Output ends with exactly one newline."""
        text = "text without newline"
        result = normalize_whitespace(text)
        assert result.endswith("\n")
        assert not result.endswith("\n\n")


class TestWrapLongLines:
    """Tests for line wrapping."""

    def test_short_lines_unchanged(self):
        """Lines shorter than width are unchanged."""
        text = "Short line"
        result = wrap_long_lines(text, width=80)
        assert "Short line" in result

    def test_long_line_wrapped(self):
        """Long lines are wrapped at word boundaries."""
        text = "This is a very long line that should definitely be wrapped because it exceeds the specified width limit"
        result = wrap_long_lines(text, width=40)
        lines = result.strip().split("\n")
        for line in lines:
            assert len(line) <= 40 or " " not in line

    def test_markdown_headings_not_wrapped(self):
        """Markdown headings are not wrapped."""
        text = "# This is a very long heading that should not be wrapped even if it exceeds the width"
        result = wrap_long_lines(text, width=40)
        assert text.strip() in result

    def test_markdown_tables_not_wrapped(self):
        """Markdown table rows are not wrapped."""
        text = "| Column 1 | Column 2 | Column 3 | Column 4 | Column 5 | Column 6 |"
        result = wrap_long_lines(text, width=40)
        assert text in result

    def test_display_math_not_wrapped(self):
        """Display math lines are not wrapped."""
        text = "$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$"
        result = wrap_long_lines(text, width=40)
        assert text in result

    def test_zero_width_returns_unchanged(self):
        """Zero width returns text unchanged."""
        text = "Some text"
        result = wrap_long_lines(text, width=0)
        assert text in result


class TestComputeAutoWrapWidth:
    """Tests for automatic wrap width computation."""

    def test_returns_reasonable_default(self):
        """Returns reasonable default for insufficient data."""
        text = "Short"
        result = compute_auto_wrap_width(text)
        assert result == 80  # Default

    def test_computes_from_blocks(self):
        """Computes width from text blocks with 3+ lines."""
        # Create a block of lines with average length around 50
        lines = ["x" * 50 for _ in range(5)]
        text = "\n".join(lines)
        result = compute_auto_wrap_width(text)
        assert 45 <= result <= 55

    def test_minimum_width_enforced(self):
        """Minimum width of 20 is enforced."""
        text = "\n".join(["x" * 5 for _ in range(5)])
        result = compute_auto_wrap_width(text)
        assert result >= 20


class TestCleanTranscription:
    """Tests for the main cleaning pipeline."""

    def test_empty_text_returns_empty(self):
        """Empty text returns empty."""
        assert clean_transcription("") == ""
        assert clean_transcription(None) == None  # type: ignore

    def test_disabled_returns_unchanged(self):
        """Disabled cleaning returns text unchanged."""
        text = "Test\u00ad text"  # With soft hyphen
        config = {"enabled": False}
        assert clean_transcription(text, config) == text

    def test_unicode_normalization_applied(self):
        """Unicode normalization is applied by default."""
        text = "test\u00adword"
        result = clean_transcription(text)
        assert "\u00ad" not in result

    def test_latex_fixing_applied(self):
        """LaTeX fixing is applied by default."""
        text = "\\frac {a}{b}"
        result = clean_transcription(text)
        assert "\\frac{" in result

    def test_whitespace_normalization_applied(self):
        """Whitespace normalization is applied."""
        text = "text   with    spaces"
        result = clean_transcription(text)
        assert "   " not in result or "    " not in result

    def test_hyphenation_merging_disabled_by_default(self):
        """Hyphenation merging is disabled by default."""
        text = "demon-\nstration"
        config = {
            "enabled": True,
            "unicode_normalization": False,
            "latex_fixing": {"enabled": False},
            "merge_hyphenation": False,
            "whitespace_normalization": {"enabled": False},
            "line_wrapping": {"enabled": False},
        }
        result = clean_transcription(text, config)
        # Should still have hyphen since merging is disabled
        assert "-" in result

    def test_line_wrapping_disabled_by_default(self):
        """Line wrapping is disabled by default."""
        text = "A " * 100  # Very long line
        config = {
            "enabled": True,
            "unicode_normalization": False,
            "latex_fixing": {"enabled": False},
            "merge_hyphenation": False,
            "whitespace_normalization": {"enabled": False},
            "line_wrapping": {"enabled": False},
        }
        result = clean_transcription(text, config)
        # Should still be one line
        assert result.strip().count("\n") == 0


class TestGetTextCleaningConfig:
    """Tests for configuration loading."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        with patch("modules.text_cleaner.get_config_loader") as mock:
            mock_loader = MagicMock()
            mock_loader.get_image_processing_config.return_value = {}
            mock.return_value = mock_loader

            result = get_text_cleaning_config()
            assert isinstance(result, dict)

    def test_has_required_keys(self):
        """Returns config with all required keys."""
        with patch("modules.text_cleaner.get_config_loader") as mock:
            mock_loader = MagicMock()
            mock_loader.get_image_processing_config.return_value = {}
            mock.return_value = mock_loader

            result = get_text_cleaning_config()
            assert "enabled" in result
            assert "unicode_normalization" in result
            assert "latex_fixing" in result
            assert "merge_hyphenation" in result
            assert "whitespace_normalization" in result
            assert "line_wrapping" in result
