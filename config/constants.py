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
PDF_DPI_CONVERSION_FACTOR = 72.0

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
)

# ============================================================================
# Rate Limiter Constants
# ============================================================================
MIN_SLEEP_TIME = 0.05
MAX_SLEEP_TIME = 0.50
ERROR_MULTIPLIER_DECREASE_RATE = 0.9
ERROR_MULTIPLIER_INCREASE_RATE_LIMIT = 1.5
ERROR_MULTIPLIER_INCREASE_OTHER = 1.2
CONSECUTIVE_ERRORS_THRESHOLD = 2
MAX_ERROR_MULTIPLIER = 5.0

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
# Working-Log Format
# ============================================================================
# Bumped whenever the on-disk working-log layout changes. Resume refuses logs
# that lack this exact marker (no migration; re-run from scratch instead).
LOG_FORMAT_VERSION = 2

# ============================================================================
# Transcription Markers
# ============================================================================
EMPTY_PAGE_MARKER = "[empty page]"
NO_TRANSCRIPTION_MARKER = "[no transcription possible]"

# Substrings that identify a blank / untranscribable page regardless of the
# exact wrapper the transcription layer emits. The transcription layer produces
# forms such as "[<img>: no transcribable text — ...]" and
# "[<img>: transcription not possible — ...]" (llm/transcription.py), while the
# legacy markers above are still recognized. Match case-insensitively as
# substrings so both variants short-circuit the summary API call.
BLANK_PAGE_SENTINELS: tuple[str, ...] = (
    "no transcribable text",
    "transcription not possible",
    "empty page",
    "no transcription possible",
)


def is_blank_transcription(text: str | None) -> bool:
    """Return True if *text* is a blank/untranscribable-page sentinel."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in BLANK_PAGE_SENTINELS)


# ============================================================================
# Model Provider Prefixes
# ============================================================================
OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4")

# ============================================================================
# Error Detection Constants
# ============================================================================
# Markers indicating a page carries no usable summary content. Kept in sync with
# BLANK_PAGE_SENTINELS plus the generic "error" marker used by summary filtering.
ERROR_MARKERS = [
    "no transcribable text",
    "transcription not possible",
    "no transcription possible",
    "empty page",
    "error",
]

# ============================================================================
# Math Conversion Constants
# ============================================================================
MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# ============================================================================
# ETA Calculation Constants
# ============================================================================
MIN_SAMPLES_FOR_ETA = 5
RECENT_SAMPLES_FOR_ETA = 10
ETA_BLEND_WEIGHT_OVERALL = 0.7
ETA_BLEND_WEIGHT_RECENT = 0.3

# ============================================================================
# CLI Constants
# ============================================================================
EXIT_COMMANDS = frozenset({"exit", "quit", "q"})
BACK_COMMANDS = frozenset({"back", "b"})
ALL_COMMANDS = frozenset({"all", "a"})
DIVIDER_CHAR = "="
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
    "PDF_DPI_CONVERSION_FACTOR",
    "SUPPORTED_IMAGE_EXTENSIONS",
    # Rate limiter
    "MIN_SLEEP_TIME",
    "MAX_SLEEP_TIME",
    "ERROR_MULTIPLIER_DECREASE_RATE",
    "ERROR_MULTIPLIER_INCREASE_RATE_LIMIT",
    "ERROR_MULTIPLIER_INCREASE_OTHER",
    "CONSECUTIVE_ERRORS_THRESHOLD",
    "MAX_ERROR_MULTIPLIER",
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
    # Working-log format
    "LOG_FORMAT_VERSION",
    # Transcription markers
    "EMPTY_PAGE_MARKER",
    "NO_TRANSCRIPTION_MARKER",
    "BLANK_PAGE_SENTINELS",
    "is_blank_transcription",
    # Model provider prefixes
    "OPENAI_MODEL_PREFIXES",
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
    # ETA calculation
    "MIN_SAMPLES_FOR_ETA",
    "RECENT_SAMPLES_FOR_ETA",
    "ETA_BLEND_WEIGHT_OVERALL",
    "ETA_BLEND_WEIGHT_RECENT",
]
