"""Centralized constants for AutoExcerpter.

This module provides a single source of truth for all default values,
constants, and configuration defaults used throughout the application.
"""

from __future__ import annotations

# ============================================================================
# API Configuration Defaults
# ============================================================================
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_CONCURRENT_REQUESTS = 4
DEFAULT_API_TIMEOUT = 320
DEFAULT_OPENAI_TIMEOUT = 900
DEFAULT_RATE_LIMITS: list[tuple[int, int]] = [(120, 1), (15000, 60), (15000, 3600)]

# ============================================================================
# Image Processing Defaults
# ============================================================================
DEFAULT_TARGET_DPI = 300
DEFAULT_JPEG_QUALITY = 95
DEFAULT_LOW_MAX_SIDE_PX = 512
DEFAULT_HIGH_TARGET_WIDTH = 768
DEFAULT_HIGH_TARGET_HEIGHT = 1536
WHITE_BACKGROUND_COLOR = (255, 255, 255)
MAX_EXTRACTION_WORKERS = 8
PDF_DPI_CONVERSION_FACTOR = 72.0

SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'
})

# ============================================================================
# Rate Limiter Constants
# ============================================================================
MIN_SLEEP_TIME = 0.05
MAX_SLEEP_TIME = 0.50
ERROR_MULTIPLIER_DECREASE_RATE = 0.9
ERROR_MULTIPLIER_INCREASE_RATE_LIMIT = 1.5
ERROR_MULTIPLIER_INCREASE_OTHER = 1.2
CONSECUTIVE_ERRORS_THRESHOLD = 2

# ============================================================================
# Document Formatting Constants
# ============================================================================
TITLE_HEADING_LEVEL = 0
PAGE_HEADING_LEVEL = 1
REFERENCES_HEADING_LEVEL = 2
TITLE_SPACE_AFTER_PT = 6
PAGE_HEADING_SPACE_BEFORE_PT = 6
PAGE_HEADING_SPACE_AFTER_PT = 3
REF_HEADING_SPACE_BEFORE_PT = 4
BULLET_SPACE_AFTER_PT = 2
REF_INDENT_PT = 18
BULLET_INDENT_PT = 18

# ============================================================================
# Error Detection Constants
# ============================================================================
ERROR_MARKERS = ["[empty page", "no transcription possible", "empty page", "error"]

# ============================================================================
# Math Conversion Constants
# ============================================================================
MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# ============================================================================
# CLI Constants
# ============================================================================
EXIT_COMMANDS = frozenset({'exit', 'quit', 'q'})
BACK_COMMANDS = frozenset({'back', 'b'})
ALL_COMMANDS = frozenset({'all', 'a'})
DIVIDER_CHAR = '─'
DIVIDER_LENGTH = 70

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # API defaults
    "DEFAULT_MODEL",
    "DEFAULT_CONCURRENT_REQUESTS",
    "DEFAULT_API_TIMEOUT",
    "DEFAULT_OPENAI_TIMEOUT",
    "DEFAULT_RATE_LIMITS",
    # Image processing
    "DEFAULT_TARGET_DPI",
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_LOW_MAX_SIDE_PX",
    "DEFAULT_HIGH_TARGET_WIDTH",
    "DEFAULT_HIGH_TARGET_HEIGHT",
    "WHITE_BACKGROUND_COLOR",
    "MAX_EXTRACTION_WORKERS",
    "PDF_DPI_CONVERSION_FACTOR",
    "SUPPORTED_IMAGE_EXTENSIONS",
    # Rate limiter
    "MIN_SLEEP_TIME",
    "MAX_SLEEP_TIME",
    "ERROR_MULTIPLIER_DECREASE_RATE",
    "ERROR_MULTIPLIER_INCREASE_RATE_LIMIT",
    "ERROR_MULTIPLIER_INCREASE_OTHER",
    "CONSECUTIVE_ERRORS_THRESHOLD",
    # Document formatting
    "TITLE_HEADING_LEVEL",
    "PAGE_HEADING_LEVEL",
    "REFERENCES_HEADING_LEVEL",
    "TITLE_SPACE_AFTER_PT",
    "PAGE_HEADING_SPACE_BEFORE_PT",
    "PAGE_HEADING_SPACE_AFTER_PT",
    "REF_HEADING_SPACE_BEFORE_PT",
    "BULLET_SPACE_AFTER_PT",
    "REF_INDENT_PT",
    "BULLET_INDENT_PT",
    # Error detection
    "ERROR_MARKERS",
    # Math conversion
    "MATH_NAMESPACE",
    # CLI
    "EXIT_COMMANDS",
    "BACK_COMMANDS",
    "ALL_COMMANDS",
    "DIVIDER_CHAR",
    "DIVIDER_LENGTH",
]
