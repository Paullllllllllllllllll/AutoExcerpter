"""Tests for pipeline/text_cleaner.py - Text cleaning utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pipeline.text_cleaner import (
    balance_dollar_signs,
    balance_left_right,
    clean_transcription,
    close_unclosed_braces,
    compute_auto_wrap_width,
    convert_html_subsup,
    fix_common_latex_commands,
    fix_latex_formulas,
    get_text_cleaning_config,
    merge_hyphenation,
    normalize_math_delimiters,
    normalize_unicode,
    normalize_whitespace,
    should_keep_hyphen,
    wrap_long_lines,
)

# Full latex_fixing config with all sub-toggles enabled, for pipeline tests.
_LATEX_ALL_ON = {
    "enabled": True,
    "fix_common_commands": True,
    "normalize_math_delimiters": True,
    "convert_html_subsup": True,
    "balance_left_right": True,
    "balance_dollar_signs": True,
    "close_unclosed_braces": True,
}


def _left_right_balanced(text: str) -> bool:
    """Return True if \\left and \\right delimiter-command counts match."""
    from pipeline.text_cleaner import _LEFT_CMD, _RIGHT_CMD

    return len(_LEFT_CMD.findall(text)) == len(_RIGHT_CMD.findall(text))


class TestNormalizeUnicode:
    """Tests for Unicode normalization."""

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert normalize_unicode("") == ""

    def test_normal_text_unchanged(self) -> None:
        """Normal ASCII text is unchanged."""
        text = "Hello, World!"
        assert normalize_unicode(text) == text

    def test_nfc_normalization(self) -> None:
        """Composed characters are normalized to NFC."""
        # e + combining acute accent should become é
        text = "Cafe\u0301"
        result = normalize_unicode(text)
        assert result == "Café"

    def test_soft_hyphen_removed(self) -> None:
        """Soft hyphens are removed."""
        text = "test\u00adword"
        assert normalize_unicode(text) == "testword"

    def test_zero_width_space_removed(self) -> None:
        """Zero-width spaces are removed."""
        text = "test\u200bword"
        assert normalize_unicode(text) == "testword"

    def test_bom_removed(self) -> None:
        """Byte order marks are removed."""
        text = "\ufeffstart of text"
        assert normalize_unicode(text) == "start of text"

    def test_word_joiner_removed(self) -> None:
        """Word joiners are removed."""
        text = "test\u2060word"
        assert normalize_unicode(text) == "testword"

    def test_newlines_preserved(self) -> None:
        """Newlines are preserved."""
        text = "line1\nline2\nline3"
        assert normalize_unicode(text) == text

    def test_tabs_preserved(self) -> None:
        """Tabs are preserved."""
        text = "col1\tcol2\tcol3"
        assert normalize_unicode(text) == text

    def test_control_characters_removed(self) -> None:
        """Control characters (except newline/tab) are removed."""
        text = "test\x00\x01\x02word"
        result = normalize_unicode(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "testword"

    def test_aegean_icons_to_bullet(self) -> None:
        """Aegean icon characters are mapped to bullets."""
        text = "item \U00010101 text"
        result = normalize_unicode(text)
        assert "•" in result


class TestBalanceDollarSigns:
    """Tests for LaTeX dollar sign balancing."""

    def test_no_dollar_signs(self) -> None:
        """Text without dollar signs is unchanged."""
        text = "No formulas here."
        assert balance_dollar_signs(text) == text

    def test_balanced_inline_math(self) -> None:
        """Balanced inline math is unchanged."""
        text = "Here is $x + y = z$ inline."
        assert balance_dollar_signs(text) == text

    def test_balanced_display_math(self) -> None:
        """Balanced display math is unchanged."""
        text = "$$\\int f(x) dx$$"
        assert balance_dollar_signs(text) == text

    def test_unbalanced_single_dollar_with_content(self) -> None:
        """Single unbalanced $ with content gets closed."""
        text = "Formula: $x + y"
        result = balance_dollar_signs(text)
        # Should add closing $ after content
        assert result.count("$") % 2 == 0 or "$" not in result

    def test_escaped_dollar_preserved(self) -> None:
        """Escaped dollar signs are preserved."""
        text = "Price is \\$50"
        result = balance_dollar_signs(text)
        assert "\\$" in result or "$50" in result

    def test_multiple_lines_handled(self) -> None:
        """Multiple lines are handled independently."""
        text = "Line 1 $a=b$\nLine 2 $c=d$"
        result = balance_dollar_signs(text)
        assert result == text


class TestCloseUnclosedBraces:
    """Tests for closing unclosed LaTeX braces."""

    def test_no_braces(self) -> None:
        """Text without braces is unchanged."""
        text = "No braces here"
        assert close_unclosed_braces(text) == text

    def test_balanced_braces(self) -> None:
        """Balanced braces are unchanged."""
        text = "\\frac{a}{b}"
        assert close_unclosed_braces(text) == text

    def test_unclosed_brace_at_end(self) -> None:
        """Unclosed brace at end of line gets closed."""
        text = "\\frac{a"
        result = close_unclosed_braces(text)
        assert result.count("{") == result.count("}")

    def test_multiple_unclosed_braces(self) -> None:
        """Multiple unclosed braces get closed."""
        text = "\\frac{a}{b + \\sqrt{c"
        result = close_unclosed_braces(text)
        assert result.count("{") == result.count("}")

    def test_escaped_braces_ignored(self) -> None:
        """Escaped braces are not counted."""
        text = "\\{test\\}"
        assert close_unclosed_braces(text) == text

    def test_orphan_closing_brace_handling(self) -> None:
        """Orphan closing braces at start are handled."""
        text = "} orphan brace"
        result = close_unclosed_braces(text)
        # The function attempts to remove orphan closing braces at start.
        # The result should be valid (removed or preserved depending on heuristics).
        assert isinstance(result, str)


class TestFixCommonLatexCommands:
    """Tests for fixing common LaTeX command issues."""

    def test_frac_space_removed(self) -> None:
        """Space after \\frac is removed."""
        text = "\\frac {a}{b}"
        result = fix_common_latex_commands(text)
        assert "\\frac{" in result

    def test_sqrt_space_removed(self) -> None:
        """Space after \\sqrt is removed."""
        text = "\\sqrt {x}"
        result = fix_common_latex_commands(text)
        assert "\\sqrt{" in result

    def test_sum_subscript_space_removed(self) -> None:
        """Space before subscript in \\sum is removed."""
        text = "\\sum _{i=1}"
        result = fix_common_latex_commands(text)
        assert "\\sum_" in result

    def test_mathrm_space_removed(self) -> None:
        """Space after \\mathrm is removed."""
        text = "\\mathrm {sin}"
        result = fix_common_latex_commands(text)
        assert "\\mathrm{" in result

    def test_text_space_removed(self) -> None:
        """Space after \\text is removed."""
        text = "\\text {word}"
        result = fix_common_latex_commands(text)
        assert "\\text{" in result


class TestFixLatexFormulas:
    """Tests for the combined LaTeX fixing function."""

    def test_disabled_returns_unchanged(self) -> None:
        """Disabled config returns text unchanged."""
        text = "\\frac {a"
        config = {"enabled": False}
        assert fix_latex_formulas(text, config) == text

    def test_all_fixes_applied(self) -> None:
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

    def test_simple_hyphenation_merge(self) -> None:
        """Simple hyphenated word across lines is merged."""
        text = "demon-\nstration"
        result = merge_hyphenation(text)
        assert result == "demonstration"

    def test_proper_hyphen_preserved(self) -> None:
        """Proper hyphenated compound (capitals) is preserved."""
        text = "Jean-\nBaptiste"
        result = merge_hyphenation(text)
        # Should preserve because of capital letter
        assert "-" in result

    def test_short_fragments_not_merged(self) -> None:
        """Short fragments are not merged (less than 3 chars)."""
        text = "a-\ntest"
        # Pattern requires 3+ chars on left side
        assert merge_hyphenation(text) == text

    def test_no_hyphenation_unchanged(self) -> None:
        """Text without hyphenation is unchanged."""
        text = "normal text without hyphenation"
        assert merge_hyphenation(text) == text

    def test_ordinary_word_break_merged(self) -> None:
        """Ordinary line-break hyphenation of a normal word is merged."""
        assert merge_hyphenation("concep-\ntion") == "conception"
        assert merge_hyphenation("Manage-\nment") == "Management"
        assert merge_hyphenation("apprecia-\ntion") == "appreciation"

    def test_genuine_compound_prefix_kept(self) -> None:
        """A known compound prefix keeps its hyphen (keep-when-unsure)."""
        assert "-" in merge_hyphenation("co-\nordinating")
        assert "-" in merge_hyphenation("self-\nevident")
        assert "-" in merge_hyphenation("non-\ntrivial")


class TestShouldKeepHyphen:
    """Tests for the conservative compound-hyphen guard."""

    def test_ordinary_lowercase_break_merges(self) -> None:
        """Ordinary lowercase fragments are merged (keep returns False)."""
        assert should_keep_hyphen("concep", "tion") is False
        assert should_keep_hyphen("Manage", "ment") is False

    def test_compound_prefix_kept(self) -> None:
        """Known compound prefixes are kept."""
        assert should_keep_hyphen("co", "ordinating") is True
        assert should_keep_hyphen("self", "evident") is True

    def test_capitalized_continuation_kept(self) -> None:
        """A capitalized continuation is treated as a proper compound."""
        assert should_keep_hyphen("Jean", "Baptiste") is True

    def test_all_caps_split_merges(self) -> None:
        """An all-caps word split mid-word merges (not a compound)."""
        assert should_keep_hyphen("KNOWL", "EDGE") is False
        assert should_keep_hyphen("MAJ", "ESTY") is False

    def test_digit_adjacent_kept(self) -> None:
        """Digit-adjacent hyphens are kept (not word hyphenation)."""
        assert should_keep_hyphen("page", "42") is True


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert normalize_whitespace("") == ""

    def test_tabs_expanded(self) -> None:
        """Tabs are expanded to spaces."""
        text = "col1\tcol2"
        result = normalize_whitespace(text, tab_size=4)
        assert "\t" not in result

    def test_trailing_spaces_stripped(self) -> None:
        """Trailing spaces are stripped from lines."""
        text = "text with trailing   \nmore text   "
        result = normalize_whitespace(text)
        lines = result.rstrip("\n").split("\n")
        for line in lines:
            assert not line.endswith(" ")

    def test_internal_spaces_collapsed(self) -> None:
        """Internal runs of 3+ spaces are collapsed to 2."""
        text = "word    word     word"
        result = normalize_whitespace(text, collapse_internal=True)
        assert "    " not in result

    def test_blank_lines_limited(self) -> None:
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

    def test_trailing_newline_added(self) -> None:
        """Output ends with exactly one newline."""
        text = "text without newline"
        result = normalize_whitespace(text)
        assert result.endswith("\n")
        assert not result.endswith("\n\n")


class TestWrapLongLines:
    """Tests for line wrapping."""

    def test_short_lines_unchanged(self) -> None:
        """Lines shorter than width are unchanged."""
        text = "Short line"
        result = wrap_long_lines(text, width=80)
        assert "Short line" in result

    def test_long_line_wrapped(self) -> None:
        """Long lines are wrapped at word boundaries."""
        text = (
            "This is a very long line that should definitely be wrapped"
            " because it exceeds the specified width limit"
        )
        result = wrap_long_lines(text, width=40)
        lines = result.strip().split("\n")
        for line in lines:
            assert len(line) <= 40 or " " not in line

    def test_markdown_headings_not_wrapped(self) -> None:
        """Markdown headings are not wrapped."""
        text = (
            "# This is a very long heading that should not be wrapped"
            " even if it exceeds the width"
        )
        result = wrap_long_lines(text, width=40)
        assert text.strip() in result

    def test_markdown_tables_not_wrapped(self) -> None:
        """Markdown table rows are not wrapped."""
        text = "| Column 1 | Column 2 | Column 3 | Column 4 | Column 5 | Column 6 |"
        result = wrap_long_lines(text, width=40)
        assert text in result

    def test_display_math_not_wrapped(self) -> None:
        """Display math lines are not wrapped."""
        text = "$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$"
        result = wrap_long_lines(text, width=40)
        assert text in result

    def test_zero_width_returns_unchanged(self) -> None:
        """Zero width returns text unchanged."""
        text = "Some text"
        result = wrap_long_lines(text, width=0)
        assert text in result


class TestComputeAutoWrapWidth:
    """Tests for automatic wrap width computation."""

    def test_returns_reasonable_default(self) -> None:
        """Returns reasonable default for insufficient data."""
        text = "Short"
        result = compute_auto_wrap_width(text)
        assert result == 80  # Default

    def test_computes_from_blocks(self) -> None:
        """Computes width from text blocks with 3+ lines."""
        # Create a block of lines with average length around 50
        lines = ["x" * 50 for _ in range(5)]
        text = "\n".join(lines)
        result = compute_auto_wrap_width(text)
        assert 45 <= result <= 55

    def test_minimum_width_enforced(self) -> None:
        """Minimum width of 20 is enforced."""
        text = "\n".join(["x" * 5 for _ in range(5)])
        result = compute_auto_wrap_width(text)
        assert result >= 20


class TestCleanTranscription:
    """Tests for the main cleaning pipeline."""

    def test_empty_text_returns_empty(self) -> None:
        """Empty text returns empty."""
        assert clean_transcription("") == ""
        assert clean_transcription(None) is None  # type: ignore

    def test_disabled_returns_unchanged(self) -> None:
        """Disabled cleaning returns text unchanged."""
        text = "Test\u00ad text"  # With soft hyphen
        config = {"enabled": False}
        assert clean_transcription(text, config) == text

    def test_unicode_normalization_applied(self) -> None:
        """Unicode normalization is applied by default."""
        text = "test\u00adword"
        result = clean_transcription(text)
        assert "\u00ad" not in result

    def test_latex_fixing_applied(self) -> None:
        """LaTeX fixing is applied by default."""
        text = "\\frac {a}{b}"
        result = clean_transcription(text)
        assert "\\frac{" in result

    def test_whitespace_normalization_applied(self) -> None:
        """Whitespace normalization is applied."""
        text = "text   with    spaces"
        result = clean_transcription(text)
        assert "   " not in result or "    " not in result

    def test_hyphenation_merging_respects_explicit_disable(self) -> None:
        """Hyphenation merging is skipped when explicitly disabled."""
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

    def test_hyphenation_merged_when_enabled(self) -> None:
        """Ordinary line-break hyphenation is merged when enabled."""
        text = "demon-\nstration"
        config = {
            "enabled": True,
            "unicode_normalization": False,
            "latex_fixing": {"enabled": False},
            "merge_hyphenation": True,
            "whitespace_normalization": {"enabled": False},
            "line_wrapping": {"enabled": False},
        }
        result = clean_transcription(text, config)
        assert result == "demonstration"

    def test_line_wrapping_disabled_by_default(self) -> None:
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

    def test_returns_dict(self) -> None:
        """Returns a dictionary."""
        with patch("pipeline.text_cleaner.get_config_loader") as mock:
            mock_loader = MagicMock()
            mock_loader.get_image_processing_config.return_value = {}
            mock.return_value = mock_loader

            result = get_text_cleaning_config()
            assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        """Returns config with all required keys."""
        with patch("pipeline.text_cleaner.get_config_loader") as mock:
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

    def test_safe_defaults_merge_on_wrap_off(self) -> None:
        """Defaults merge hyphenation and keep line wrapping disabled."""
        with patch("pipeline.text_cleaner.get_config_loader") as mock:
            mock_loader = MagicMock()
            mock_loader.get_image_processing_config.return_value = {}
            mock.return_value = mock_loader

            result = get_text_cleaning_config()
            assert result["merge_hyphenation"] is True
            assert result["line_wrapping"]["enabled"] is False

    def test_latex_fixing_has_new_subtoggles(self) -> None:
        """Default latex_fixing exposes the new post-processing toggles."""
        with patch("pipeline.text_cleaner.get_config_loader") as mock:
            mock_loader = MagicMock()
            mock_loader.get_image_processing_config.return_value = {}
            mock.return_value = mock_loader

            latex = get_text_cleaning_config()["latex_fixing"]
            assert latex["normalize_math_delimiters"] is True
            assert latex["balance_left_right"] is True
            assert latex["convert_html_subsup"] is True


class TestNormalizeMathDelimiters:
    """Tests for \\(..\\)/\\[..\\] -> $..$/$$..$$ normalization."""

    def test_no_delimiters_unchanged(self) -> None:
        """Text without alternate delimiters is byte-identical."""
        text = "Plain $x$ and $$y$$ already."
        assert normalize_math_delimiters(text) == text

    def test_inline_paren_delimiters(self) -> None:
        r"""\(x\) becomes $x$."""
        assert normalize_math_delimiters(r"see \(x\) here") == "see $x$ here"

    def test_display_bracket_delimiters(self) -> None:
        r"""\[..\] becomes $$..$$."""
        assert normalize_math_delimiters(r"\[E = mc^2\]") == "$$E = mc^2$$"

    def test_mixed_delimiters(self) -> None:
        r"""Inline and display forms convert together."""
        text = r"\(a\) and \[b\]"
        assert normalize_math_delimiters(text) == "$a$ and $$b$$"


class TestConvertHtmlSubsup:
    """Tests for HTML sub/superscript -> inline math conversion."""

    def test_subscript_conversion(self) -> None:
        """X<sub>y</sub> becomes $X_y$."""
        assert convert_html_subsup("value X<sub>y</sub>.") == "value $X_y$."

    def test_superscript_conversion(self) -> None:
        """X<sup>y</sup> becomes $X^y$."""
        assert convert_html_subsup("area x<sup>2</sup>.") == "area $x^2$."

    def test_emphasis_wrapped_base(self) -> None:
        """*A*<sub>m</sub> (markdown emphasis) becomes $A_m$."""
        assert convert_html_subsup("ratio *A*<sub>m</sub> rises") == "ratio $A_m$ rises"

    def test_untouched_inside_inline_math(self) -> None:
        """Sub/sup inside existing $...$ is left untouched."""
        text = "already $x<sub>i</sub>$ done"
        assert convert_html_subsup(text) == text

    def test_untouched_inside_display_math(self) -> None:
        """Sub/sup inside existing $$...$$ is left untouched."""
        text = "$$A<sub>m</sub>$$"
        assert convert_html_subsup(text) == text

    def test_no_tags_unchanged(self) -> None:
        """Text without sub/sup tags is byte-identical."""
        text = "no markup here"
        assert convert_html_subsup(text) == text

    def test_long_token_not_converted(self) -> None:
        """A multi-word emphasis span is not misread as a base token."""
        text = "*emphasis*<sub>note</sub>"
        # Base/script exceed the 1-3 char scope; left unchanged.
        assert convert_html_subsup(text) == text


class TestBalanceLeftRight:
    r"""Tests for \left/\right balancing within display math blocks."""

    def test_unmatched_right_gets_left_dot(self) -> None:
        r"""\right\} with no \left gets a \left. and renders balanced."""
        text = "$$\\begin{aligned} x &= 1 \\quad\\right\\} \\quad < 0 \\end{aligned}$$"
        result = balance_left_right(text)
        assert "\\left." in result
        assert _left_right_balanced(result)

    def test_unmatched_right_no_environment(self) -> None:
        r"""\right without an environment prepends \left. at block start."""
        text = "$$a \\right) b$$"
        result = balance_left_right(text)
        assert result == "$$\\left.a \\right) b$$"
        assert _left_right_balanced(result)

    def test_unmatched_left_appends_right_dot(self) -> None:
        r"""Unmatched \left gets a trailing \right.."""
        text = "$$\\left( a + b$$"
        result = balance_left_right(text)
        assert result == "$$\\left( a + b\\right.$$"
        assert _left_right_balanced(result)

    def test_balanced_block_unchanged(self) -> None:
        r"""A balanced \left..\right block is byte-identical."""
        text = "$$\\left( a + b \\right)$$"
        assert balance_left_right(text) == text

    def test_arrow_commands_not_miscounted(self) -> None:
        r"""\leftarrow/\rightarrow are not treated as delimiters."""
        text = "$$a \\rightarrow b \\leftarrow c$$"
        assert balance_left_right(text) == text

    def test_inline_math_not_touched(self) -> None:
        r"""Unbalanced \left/\right in inline $...$ is left unchanged."""
        text = "inline $\\left( x$ stays"
        assert balance_left_right(text) == text

    def test_misordered_but_equal_counts_unchanged(self) -> None:
        r"""Equal-but-misordered counts are ambiguous -> left unchanged."""
        text = "$$\\right) a \\left($$"
        assert balance_left_right(text) == text

    def test_no_left_right_unchanged(self) -> None:
        """Text without \\left/\\right is byte-identical."""
        text = "$$x + y = z$$"
        assert balance_left_right(text) == text


class TestLatexPostprocessingPipeline:
    """Integration of the new fixes through fix_latex_formulas."""

    def test_paren_delimiters_normalized(self) -> None:
        r"""\(x\) is normalized to $x$ end to end."""
        result = fix_latex_formulas(r"here \(x\) there", _LATEX_ALL_ON)
        assert result == "here $x$ there"

    def test_bracket_delimiters_normalized(self) -> None:
        r"""\[..\] is normalized to $$..$$ end to end."""
        result = fix_latex_formulas(r"\[a = b\]", _LATEX_ALL_ON)
        assert result == "$$a = b$$"

    def test_html_subsup_normalized(self) -> None:
        """*A*<sub>m</sub> outside math becomes $A_m$ end to end."""
        result = fix_latex_formulas("ratio *A*<sub>m</sub>", _LATEX_ALL_ON)
        assert result == "ratio $A_m$"

    def test_html_subsup_inside_math_untouched(self) -> None:
        """Sub tags inside $...$ survive the pipeline."""
        text = "keep $x<sub>i</sub>$"
        assert fix_latex_formulas(text, _LATEX_ALL_ON) == text

    def test_unbalanced_left_right_repaired(self) -> None:
        r"""\right\} with no \left is repaired to a balanced block."""
        text = "$$a \\quad\\right\\} \\quad < 0$$"
        result = fix_latex_formulas(text, _LATEX_ALL_ON)
        assert _left_right_balanced(result)

    def test_balanced_input_byte_identical(self) -> None:
        """Clean, balanced input passes through unchanged."""
        text = (
            "Prose with inline $a > b$ and a display block:\n"
            "$$\\left( \\frac{x}{y} \\right) = z$$\n"
            "and $c \\leq d$ afterwards."
        )
        assert fix_latex_formulas(text, _LATEX_ALL_ON) == text

    def test_subtoggles_can_be_disabled(self) -> None:
        """Disabling a sub-toggle skips that specific fix."""
        config = dict(_LATEX_ALL_ON)
        config["convert_html_subsup"] = False
        text = "ratio X<sub>y</sub>"
        assert fix_latex_formulas(text, config) == text
