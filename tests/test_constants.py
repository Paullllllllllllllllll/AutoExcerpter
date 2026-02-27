"""Tests for modules/constants.py - Centralized constants."""

from __future__ import annotations

import pytest

from modules.constants import (
    # API Configuration
    DEFAULT_MODEL,
    DEFAULT_CONCURRENT_REQUESTS,
    DEFAULT_API_TIMEOUT,
    DEFAULT_OPENAI_TIMEOUT,
    DEFAULT_RATE_LIMITS,
    # Image Processing
    DEFAULT_TARGET_DPI,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_LOW_MAX_SIDE_PX,
    DEFAULT_HIGH_TARGET_WIDTH,
    DEFAULT_HIGH_TARGET_HEIGHT,
    WHITE_BACKGROUND_COLOR,
    MAX_EXTRACTION_WORKERS,
    PDF_DPI_CONVERSION_FACTOR,
    SUPPORTED_IMAGE_EXTENSIONS,
    # Rate Limiter
    MIN_SLEEP_TIME,
    MAX_SLEEP_TIME,
    ERROR_MULTIPLIER_DECREASE_RATE,
    ERROR_MULTIPLIER_INCREASE_RATE_LIMIT,
    ERROR_MULTIPLIER_INCREASE_OTHER,
    CONSECUTIVE_ERRORS_THRESHOLD,
    # Document Formatting
    TITLE_HEADING_LEVEL,
    PAGE_HEADING_LEVEL,
    REFERENCES_HEADING_LEVEL,
    TITLE_SPACE_AFTER_PT,
    PAGE_HEADING_SPACE_BEFORE_PT,
    PAGE_HEADING_SPACE_AFTER_PT,
    REF_HEADING_SPACE_BEFORE_PT,
    BULLET_SPACE_AFTER_PT,
    REF_INDENT_PT,
    BULLET_INDENT_PT,
    # Error Detection
    ERROR_MARKERS,
    # Math Conversion
    MATH_NAMESPACE,
    # CLI
    EXIT_COMMANDS,
    BACK_COMMANDS,
    ALL_COMMANDS,
    DIVIDER_CHAR,
    DIVIDER_LENGTH,
)


class TestAPIConfigurationDefaults:
    """Tests for API configuration defaults."""

    def test_default_model_is_string(self):
        """DEFAULT_MODEL is a non-empty string."""
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0

    def test_default_concurrent_requests_positive(self):
        """DEFAULT_CONCURRENT_REQUESTS is positive integer."""
        assert isinstance(DEFAULT_CONCURRENT_REQUESTS, int)
        assert DEFAULT_CONCURRENT_REQUESTS > 0

    def test_default_api_timeout_positive(self):
        """DEFAULT_API_TIMEOUT is positive integer."""
        assert isinstance(DEFAULT_API_TIMEOUT, int)
        assert DEFAULT_API_TIMEOUT > 0

    def test_default_openai_timeout_positive(self):
        """DEFAULT_OPENAI_TIMEOUT is positive integer."""
        assert isinstance(DEFAULT_OPENAI_TIMEOUT, int)
        assert DEFAULT_OPENAI_TIMEOUT > 0
        assert DEFAULT_OPENAI_TIMEOUT >= DEFAULT_API_TIMEOUT

    def test_default_rate_limits_structure(self):
        """DEFAULT_RATE_LIMITS has correct structure."""
        assert isinstance(DEFAULT_RATE_LIMITS, list)
        assert len(DEFAULT_RATE_LIMITS) > 0

        for limit in DEFAULT_RATE_LIMITS:
            assert isinstance(limit, tuple)
            assert len(limit) == 2
            assert all(isinstance(x, int) for x in limit)
            assert all(x > 0 for x in limit)


class TestImageProcessingDefaults:
    """Tests for image processing defaults."""

    def test_default_target_dpi_reasonable(self):
        """DEFAULT_TARGET_DPI is reasonable value."""
        assert isinstance(DEFAULT_TARGET_DPI, int)
        assert 72 <= DEFAULT_TARGET_DPI <= 600

    def test_default_jpeg_quality_in_range(self):
        """DEFAULT_JPEG_QUALITY is in valid range."""
        assert isinstance(DEFAULT_JPEG_QUALITY, int)
        assert 1 <= DEFAULT_JPEG_QUALITY <= 100

    def test_default_low_max_side_positive(self):
        """DEFAULT_LOW_MAX_SIDE_PX is positive."""
        assert isinstance(DEFAULT_LOW_MAX_SIDE_PX, int)
        assert DEFAULT_LOW_MAX_SIDE_PX > 0

    def test_default_high_dimensions_positive(self):
        """High detail target dimensions are positive."""
        assert isinstance(DEFAULT_HIGH_TARGET_WIDTH, int)
        assert isinstance(DEFAULT_HIGH_TARGET_HEIGHT, int)
        assert DEFAULT_HIGH_TARGET_WIDTH > 0
        assert DEFAULT_HIGH_TARGET_HEIGHT > 0

    def test_white_background_color_valid(self):
        """WHITE_BACKGROUND_COLOR is valid RGB tuple."""
        assert isinstance(WHITE_BACKGROUND_COLOR, tuple)
        assert len(WHITE_BACKGROUND_COLOR) == 3
        assert all(0 <= c <= 255 for c in WHITE_BACKGROUND_COLOR)

    def test_max_extraction_workers_positive(self):
        """MAX_EXTRACTION_WORKERS is positive."""
        assert isinstance(MAX_EXTRACTION_WORKERS, int)
        assert MAX_EXTRACTION_WORKERS > 0

    def test_pdf_dpi_conversion_factor_positive(self):
        """PDF_DPI_CONVERSION_FACTOR is positive."""
        assert isinstance(PDF_DPI_CONVERSION_FACTOR, float)
        assert PDF_DPI_CONVERSION_FACTOR > 0

    def test_supported_image_extensions_non_empty(self):
        """SUPPORTED_IMAGE_EXTENSIONS is non-empty set."""
        assert isinstance(SUPPORTED_IMAGE_EXTENSIONS, frozenset)
        assert len(SUPPORTED_IMAGE_EXTENSIONS) > 0

    def test_supported_extensions_are_lowercase(self):
        """All supported extensions are lowercase with dot prefix."""
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            assert ext.startswith(".")
            assert ext == ext.lower()

    def test_common_formats_supported(self):
        """Common image formats are supported."""
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS


class TestRateLimiterConstants:
    """Tests for rate limiter constants."""

    def test_min_sleep_time_positive(self):
        """MIN_SLEEP_TIME is positive."""
        assert isinstance(MIN_SLEEP_TIME, float)
        assert MIN_SLEEP_TIME > 0

    def test_max_sleep_time_greater_than_min(self):
        """MAX_SLEEP_TIME is greater than MIN_SLEEP_TIME."""
        assert MAX_SLEEP_TIME > MIN_SLEEP_TIME

    def test_error_multiplier_decrease_rate(self):
        """ERROR_MULTIPLIER_DECREASE_RATE is less than 1."""
        assert isinstance(ERROR_MULTIPLIER_DECREASE_RATE, float)
        assert 0 < ERROR_MULTIPLIER_DECREASE_RATE < 1

    def test_error_multiplier_increase_rates(self):
        """Error multiplier increase rates are greater than 1."""
        assert ERROR_MULTIPLIER_INCREASE_RATE_LIMIT > 1
        assert ERROR_MULTIPLIER_INCREASE_OTHER > 1

    def test_consecutive_errors_threshold_positive(self):
        """CONSECUTIVE_ERRORS_THRESHOLD is positive."""
        assert isinstance(CONSECUTIVE_ERRORS_THRESHOLD, int)
        assert CONSECUTIVE_ERRORS_THRESHOLD > 0


class TestDocumentFormattingConstants:
    """Tests for document formatting constants."""

    def test_heading_levels_ordered(self):
        """Heading levels are in correct order."""
        assert TITLE_HEADING_LEVEL <= PAGE_HEADING_LEVEL
        assert PAGE_HEADING_LEVEL <= REFERENCES_HEADING_LEVEL

    def test_spacing_values_non_negative(self):
        """Spacing values are non-negative."""
        assert TITLE_SPACE_AFTER_PT >= 0
        assert PAGE_HEADING_SPACE_BEFORE_PT >= 0
        assert PAGE_HEADING_SPACE_AFTER_PT >= 0
        assert REF_HEADING_SPACE_BEFORE_PT >= 0
        assert BULLET_SPACE_AFTER_PT >= 0

    def test_indent_values_non_negative(self):
        """Indent values are non-negative."""
        assert REF_INDENT_PT >= 0
        assert BULLET_INDENT_PT >= 0


class TestErrorDetectionConstants:
    """Tests for error detection constants."""

    def test_error_markers_non_empty(self):
        """ERROR_MARKERS is non-empty list."""
        assert isinstance(ERROR_MARKERS, list)
        assert len(ERROR_MARKERS) > 0

    def test_error_markers_are_strings(self):
        """All error markers are strings."""
        assert all(isinstance(m, str) for m in ERROR_MARKERS)

    def test_common_error_markers_present(self):
        """Common error markers are present."""
        markers_lower = [m.lower() for m in ERROR_MARKERS]
        assert any("empty" in m for m in markers_lower)
        assert any("error" in m for m in markers_lower)


class TestMathConversionConstants:
    """Tests for math conversion constants."""

    def test_math_namespace_is_url(self):
        """MATH_NAMESPACE is a valid URL."""
        assert isinstance(MATH_NAMESPACE, str)
        assert MATH_NAMESPACE.startswith("http")


class TestCLIConstants:
    """Tests for CLI constants."""

    def test_exit_commands_non_empty(self):
        """EXIT_COMMANDS is non-empty set."""
        assert isinstance(EXIT_COMMANDS, frozenset)
        assert len(EXIT_COMMANDS) > 0

    def test_back_commands_non_empty(self):
        """BACK_COMMANDS is non-empty set."""
        assert isinstance(BACK_COMMANDS, frozenset)
        assert len(BACK_COMMANDS) > 0

    def test_all_commands_non_empty(self):
        """ALL_COMMANDS is non-empty set."""
        assert isinstance(ALL_COMMANDS, frozenset)
        assert len(ALL_COMMANDS) > 0

    def test_common_exit_commands(self):
        """Common exit commands are present."""
        assert "exit" in EXIT_COMMANDS or "quit" in EXIT_COMMANDS

    def test_divider_char_single(self):
        """DIVIDER_CHAR is a single character."""
        assert isinstance(DIVIDER_CHAR, str)
        assert len(DIVIDER_CHAR) == 1

    def test_divider_length_positive(self):
        """DIVIDER_LENGTH is positive."""
        assert isinstance(DIVIDER_LENGTH, int)
        assert DIVIDER_LENGTH > 0
