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
# Base per-request penalty (seconds) imposed once the error multiplier is
# elevated but no window is saturated. Without this the multiplier would scale
# a zero wait and stay a no-op, admitting at full speed after repeated 429s.
ERROR_BASE_PENALTY_SECONDS = 0.5

# ============================================================================
# Document Formatting Constants
# ============================================================================
# Heading levels (python-docx built-in style indices, retained for structure).
TITLE_HEADING_LEVEL = 0
PAGE_HEADING_LEVEL = 1
REFERENCES_HEADING_LEVEL = 2
# Semantic heading levels used by the DOCX writer: section headings ("Document
# Structure", "Consolidated References") map to Word's Heading 1, per-page item
# headings ("Page 12") map to Heading 2.
SECTION_HEADING_LEVEL = 1
PAGE_ITEM_HEADING_LEVEL = 2

# --- Page setup (centimeters, A4) ---
PAGE_WIDTH_CM = 21.0
PAGE_HEIGHT_CM = 29.7
PAGE_MARGIN_CM = 2.0

# --- Typography: font family and sizes (points) ---
BODY_FONT_NAME = "Times New Roman"
BODY_FONT_PT = 11.0
TITLE_FONT_PT = 16.0
METADATA_FONT_PT = 9.0
SECTION_HEADING_FONT_PT = 13.0
PAGE_HEADING_FONT_PT = 11.5
BULLET_FONT_PT = 11.0
REF_FONT_PT = 10.0
REF_META_FONT_PT = 9.0
FOOTER_FONT_PT = 9.0

# --- Colors (0xRRGGBB, consumed by docx.shared.RGBColor) ---
COLOR_BLACK = 0x000000
COLOR_METADATA_GRAY = 0x595959
COLOR_SECTION_RULE = 0xAAAAAA
COLOR_PAGE_HEADING = 0x1F3864
COLOR_REF_META_GRAY = 0x595959

# --- Vertical spacing (points) ---
BODY_SPACE_AFTER_PT = 4
TITLE_SPACE_AFTER_PT = 6
SECTION_HEADING_SPACE_BEFORE_PT = 10
SECTION_HEADING_SPACE_AFTER_PT = 4
PAGE_HEADING_SPACE_BEFORE_PT = 8
PAGE_HEADING_SPACE_AFTER_PT = 2
REF_HEADING_SPACE_BEFORE_PT = 10
BULLET_SPACE_AFTER_PT = 2
REF_SPACE_AFTER_PT = 2

# --- Indentation (centimeters) ---
BULLET_LEFT_INDENT_CM = 0.5
BULLET_HANGING_INDENT_CM = 0.25
REF_HANGING_INDENT_CM = 0.5

# --- Legacy indent constants (points), retained for compatibility ---
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
    "ERROR_BASE_PENALTY_SECONDS",
    # Document formatting
    "TITLE_HEADING_LEVEL",
    "PAGE_HEADING_LEVEL",
    "REFERENCES_HEADING_LEVEL",
    "PAGE_WIDTH_CM",
    "PAGE_HEIGHT_CM",
    "PAGE_MARGIN_CM",
    "BODY_FONT_NAME",
    "BODY_FONT_PT",
    "TITLE_FONT_PT",
    "METADATA_FONT_PT",
    "SECTION_HEADING_FONT_PT",
    "PAGE_HEADING_FONT_PT",
    "BULLET_FONT_PT",
    "REF_FONT_PT",
    "REF_META_FONT_PT",
    "FOOTER_FONT_PT",
    "COLOR_BLACK",
    "COLOR_METADATA_GRAY",
    "COLOR_SECTION_RULE",
    "COLOR_PAGE_HEADING",
    "COLOR_REF_META_GRAY",
    "BODY_SPACE_AFTER_PT",
    "TITLE_SPACE_AFTER_PT",
    "SECTION_HEADING_SPACE_BEFORE_PT",
    "SECTION_HEADING_SPACE_AFTER_PT",
    "PAGE_HEADING_SPACE_BEFORE_PT",
    "PAGE_HEADING_SPACE_AFTER_PT",
    "REF_HEADING_SPACE_BEFORE_PT",
    "BULLET_SPACE_AFTER_PT",
    "REF_SPACE_AFTER_PT",
    "BULLET_LEFT_INDENT_CM",
    "BULLET_HANGING_INDENT_CM",
    "REF_HANGING_INDENT_CM",
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
