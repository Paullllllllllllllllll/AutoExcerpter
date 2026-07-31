"""Tests for modules/constants.py - Centralized constants."""

from __future__ import annotations

from config.constants import (
    ALL_COMMANDS,
    BACK_COMMANDS,
    BULLET_INDENT_PT,
    BULLET_SPACE_AFTER_PT,
    CONSECUTIVE_ERRORS_THRESHOLD,
    DEFAULT_CONCURRENT_REQUESTS,
    DEFAULT_HIGH_TARGET_HEIGHT,
    DEFAULT_HIGH_TARGET_WIDTH,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_LOW_MAX_SIDE_PX,
    # API Configuration
    DEFAULT_MODEL,
    DEFAULT_OPENAI_TIMEOUT,
    DEFAULT_RATE_LIMITS,
    # Image Processing
    DEFAULT_TARGET_DPI,
    DIVIDER_CHAR,
    DIVIDER_LENGTH,
    # Error Detection
    ERROR_MARKERS,
    ERROR_MULTIPLIER_DECREASE_RATE,
    ERROR_MULTIPLIER_INCREASE_OTHER,
    ERROR_MULTIPLIER_INCREASE_RATE_LIMIT,
    # CLI
    EXIT_COMMANDS,
    # Math Conversion
    MATH_NAMESPACE,
    MAX_SLEEP_TIME,
    # Rate Limiter
    MIN_SLEEP_TIME,
    PAGE_HEADING_LEVEL,
    PAGE_HEADING_SPACE_AFTER_PT,
    PAGE_HEADING_SPACE_BEFORE_PT,
    PDF_DPI_CONVERSION_FACTOR,
    REF_HEADING_SPACE_BEFORE_PT,
    REF_INDENT_PT,
    REFERENCES_HEADING_LEVEL,
    SUPPORTED_IMAGE_EXTENSIONS,
    # Document Formatting
    TITLE_HEADING_LEVEL,
    TITLE_SPACE_AFTER_PT,
    WHITE_BACKGROUND_COLOR,
)


class TestAPIConfigurationDefaults:
    """Tests for API configuration defaults.

    Drift-prone values are asserted EXACTLY, not merely "non-empty" or
    "positive": DEFAULT_MODEL sat at a stale "gpt-5-mini" for three minor
    releases after the shipped default moved on, and a shape-only assertion
    could never have caught it.
    """

    def test_default_model_exact(self) -> None:
        """DEFAULT_MODEL matches the shipped default model name."""
        assert DEFAULT_MODEL == "gpt-5.6-luna"

    def test_default_concurrent_requests_exact(self) -> None:
        """DEFAULT_CONCURRENT_REQUESTS is the expected conservative fallback."""
        assert isinstance(DEFAULT_CONCURRENT_REQUESTS, int)
        assert DEFAULT_CONCURRENT_REQUESTS == 4

    def test_default_openai_timeout_exact(self) -> None:
        """DEFAULT_OPENAI_TIMEOUT matches the flex-tier timeout in the template."""
        assert isinstance(DEFAULT_OPENAI_TIMEOUT, int)
        assert DEFAULT_OPENAI_TIMEOUT == 900

    def test_dead_api_timeout_constant_removed(self) -> None:
        """DEFAULT_API_TIMEOUT is gone: the live fallback is the OpenAI one.

        It was referenced only by this test file; config/accessors.py uses
        DEFAULT_OPENAI_TIMEOUT for the real api_timeout fallback.
        """
        import config.constants as constants

        assert not hasattr(constants, "DEFAULT_API_TIMEOUT")
        assert "DEFAULT_API_TIMEOUT" not in constants.__all__

    def test_default_rate_limits_exact(self) -> None:
        """DEFAULT_RATE_LIMITS has the expected values and structure."""
        assert DEFAULT_RATE_LIMITS == [(120, 1), (15000, 60), (15000, 3600)]

        for limit in DEFAULT_RATE_LIMITS:
            assert isinstance(limit, tuple)
            assert len(limit) == 2
            assert all(isinstance(x, int) for x in limit)
            assert all(x > 0 for x in limit)


class TestImageProcessingDefaults:
    """Tests for image processing defaults."""

    def test_default_target_dpi_exact(self) -> None:
        """DEFAULT_TARGET_DPI matches the shipped image_processing template."""
        assert DEFAULT_TARGET_DPI == 300
        assert 72 <= DEFAULT_TARGET_DPI <= 600

    def test_default_jpeg_quality_exact(self) -> None:
        """DEFAULT_JPEG_QUALITY matches the shipped image_processing template."""
        assert DEFAULT_JPEG_QUALITY == 95
        assert 1 <= DEFAULT_JPEG_QUALITY <= 100

    def test_default_low_max_side_exact(self) -> None:
        """DEFAULT_LOW_MAX_SIDE_PX is the low-detail cap the API expects."""
        assert DEFAULT_LOW_MAX_SIDE_PX == 512

    def test_default_high_dimensions_exact(self) -> None:
        """High-detail target dimensions match the API tile geometry."""
        assert DEFAULT_HIGH_TARGET_WIDTH == 768
        assert DEFAULT_HIGH_TARGET_HEIGHT == 1536

    def test_white_background_color_valid(self) -> None:
        """WHITE_BACKGROUND_COLOR is valid RGB tuple."""
        assert isinstance(WHITE_BACKGROUND_COLOR, tuple)
        assert len(WHITE_BACKGROUND_COLOR) == 3
        assert all(0 <= c <= 255 for c in WHITE_BACKGROUND_COLOR)

    def test_pdf_dpi_conversion_factor_positive(self) -> None:
        """PDF_DPI_CONVERSION_FACTOR is positive."""
        assert isinstance(PDF_DPI_CONVERSION_FACTOR, float)
        assert PDF_DPI_CONVERSION_FACTOR > 0

    def test_supported_image_extensions_non_empty(self) -> None:
        """SUPPORTED_IMAGE_EXTENSIONS is non-empty set."""
        assert isinstance(SUPPORTED_IMAGE_EXTENSIONS, frozenset)
        assert len(SUPPORTED_IMAGE_EXTENSIONS) > 0

    def test_supported_extensions_are_lowercase(self) -> None:
        """All supported extensions are lowercase with dot prefix."""
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            assert ext.startswith(".")
            assert ext == ext.lower()

    def test_common_formats_supported(self) -> None:
        """Common image formats are supported."""
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS


class TestRateLimiterConstants:
    """Tests for rate limiter constants."""

    def test_min_sleep_time_positive(self) -> None:
        """MIN_SLEEP_TIME is positive."""
        assert isinstance(MIN_SLEEP_TIME, float)
        assert MIN_SLEEP_TIME > 0

    def test_max_sleep_time_greater_than_min(self) -> None:
        """MAX_SLEEP_TIME is greater than MIN_SLEEP_TIME."""
        assert MAX_SLEEP_TIME > MIN_SLEEP_TIME

    def test_error_multiplier_decrease_rate(self) -> None:
        """ERROR_MULTIPLIER_DECREASE_RATE is less than 1."""
        assert isinstance(ERROR_MULTIPLIER_DECREASE_RATE, float)
        assert 0 < ERROR_MULTIPLIER_DECREASE_RATE < 1

    def test_error_multiplier_increase_rates(self) -> None:
        """Error multiplier increase rates are greater than 1."""
        assert ERROR_MULTIPLIER_INCREASE_RATE_LIMIT > 1
        assert ERROR_MULTIPLIER_INCREASE_OTHER > 1

    def test_consecutive_errors_threshold_positive(self) -> None:
        """CONSECUTIVE_ERRORS_THRESHOLD is positive."""
        assert isinstance(CONSECUTIVE_ERRORS_THRESHOLD, int)
        assert CONSECUTIVE_ERRORS_THRESHOLD > 0


class TestDocumentFormattingConstants:
    """Tests for document formatting constants."""

    def test_heading_levels_ordered(self) -> None:
        """Heading levels are in correct order."""
        assert TITLE_HEADING_LEVEL <= PAGE_HEADING_LEVEL
        assert PAGE_HEADING_LEVEL <= REFERENCES_HEADING_LEVEL

    def test_spacing_values_non_negative(self) -> None:
        """Spacing values are non-negative."""
        assert TITLE_SPACE_AFTER_PT >= 0
        assert PAGE_HEADING_SPACE_BEFORE_PT >= 0
        assert PAGE_HEADING_SPACE_AFTER_PT >= 0
        assert REF_HEADING_SPACE_BEFORE_PT >= 0
        assert BULLET_SPACE_AFTER_PT >= 0

    def test_indent_values_non_negative(self) -> None:
        """Indent values are non-negative."""
        assert REF_INDENT_PT >= 0
        assert BULLET_INDENT_PT >= 0


class TestErrorDetectionConstants:
    """Tests for error detection constants."""

    def test_error_markers_non_empty(self) -> None:
        """ERROR_MARKERS is non-empty list."""
        assert isinstance(ERROR_MARKERS, list)
        assert len(ERROR_MARKERS) > 0

    def test_error_markers_are_strings(self) -> None:
        """All error markers are strings."""
        assert all(isinstance(m, str) for m in ERROR_MARKERS)

    def test_common_error_markers_present(self) -> None:
        """Blank sentinels and bracketed placeholder prefixes are present."""
        markers_lower = [m.lower() for m in ERROR_MARKERS]
        assert any("empty" in m for m in markers_lower)
        assert any("error" in m for m in markers_lower)

    def test_no_bare_error_marker(self) -> None:
        """A bare ``error`` entry is gone: it matched legitimate prose.

        Error-placeholder markers are now the exact bracketed prefixes the
        summary layer emits, so every non-blank marker starts with ``[``.
        """
        assert "error" not in ERROR_MARKERS
        blank = {
            "no transcribable text",
            "transcription not possible",
            "no transcription possible",
            "empty page",
        }
        for marker in ERROR_MARKERS:
            assert marker in blank or marker.startswith("[")
        assert "[error generating summary" in ERROR_MARKERS
        assert "[summary generation failed" in ERROR_MARKERS


class TestMathConversionConstants:
    """Tests for math conversion constants."""

    def test_math_namespace_is_url(self) -> None:
        """MATH_NAMESPACE is a valid URL."""
        assert isinstance(MATH_NAMESPACE, str)
        assert MATH_NAMESPACE.startswith("http")


class TestCLIConstants:
    """Tests for CLI constants."""

    def test_exit_commands_non_empty(self) -> None:
        """EXIT_COMMANDS is non-empty set."""
        assert isinstance(EXIT_COMMANDS, frozenset)
        assert len(EXIT_COMMANDS) > 0

    def test_back_commands_non_empty(self) -> None:
        """BACK_COMMANDS is non-empty set."""
        assert isinstance(BACK_COMMANDS, frozenset)
        assert len(BACK_COMMANDS) > 0

    def test_all_commands_non_empty(self) -> None:
        """ALL_COMMANDS is non-empty set."""
        assert isinstance(ALL_COMMANDS, frozenset)
        assert len(ALL_COMMANDS) > 0

    def test_common_exit_commands(self) -> None:
        """Common exit commands are present."""
        assert "exit" in EXIT_COMMANDS or "quit" in EXIT_COMMANDS

    def test_divider_char_single(self) -> None:
        """DIVIDER_CHAR is a single character."""
        assert isinstance(DIVIDER_CHAR, str)
        assert len(DIVIDER_CHAR) == 1

    def test_divider_length_positive(self) -> None:
        """DIVIDER_LENGTH is positive."""
        assert isinstance(DIVIDER_LENGTH, int)
        assert DIVIDER_LENGTH > 0
