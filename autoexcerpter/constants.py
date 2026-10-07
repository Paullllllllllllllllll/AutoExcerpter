"""Named constants shared across AutoExcerpter."""

from __future__ import annotations

# Page setup (centimeters, A4).
PAGE_WIDTH_CM = 21.0
PAGE_HEIGHT_CM = 29.7
PAGE_MARGIN_CM = 2.0

# Typography: font family and sizes (points).
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

# Colors (0xRRGGBB, consumed by docx.shared.RGBColor).
COLOR_BLACK = 0x000000
COLOR_METADATA_GRAY = 0x595959
COLOR_SECTION_RULE = 0xAAAAAA
COLOR_PAGE_HEADING = 0x1F3864
COLOR_REF_META_GRAY = 0x595959

# Vertical spacing (points).
BODY_SPACE_AFTER_PT = 4
TITLE_SPACE_AFTER_PT = 6
SECTION_HEADING_SPACE_BEFORE_PT = 10
SECTION_HEADING_SPACE_AFTER_PT = 4
PAGE_HEADING_SPACE_BEFORE_PT = 8
PAGE_HEADING_SPACE_AFTER_PT = 2
BULLET_SPACE_AFTER_PT = 2
REF_SPACE_AFTER_PT = 2

# Indentation (centimeters).
BULLET_LEFT_INDENT_CM = 0.5
BULLET_HANGING_INDENT_CM = 0.25
REF_HANGING_INDENT_CM = 0.5

# Resume refuses working logs without this exact marker.
LOG_FORMAT_VERSION = 2

# Phrases of the blank and untranscribable-page markers that
# llm/transcription.py emits, matched case-insensitively as substrings so
# that every variant skips the summary call.
BLANK_PAGE_SENTINELS: tuple[str, ...] = (
    "no transcribable text",
    "transcription not possible",
    "empty page",
    "no transcription possible",
)

# A sentinel is a short bracketed marker; longer or unbracketed text that
# mentions a sentinel phrase is page content.
BLANK_TRANSCRIPTION_MAX_LEN = 200


def is_blank_transcription(text: str | None) -> bool:
    """Return True if *text* is a short bracketed blank or untranscribable marker."""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) >= BLANK_TRANSCRIPTION_MAX_LEN:
        return False
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    lowered = stripped.lower()
    return any(marker in lowered for marker in BLANK_PAGE_SENTINELS)


OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4")

# Markers of a single-bullet summary without usable content: the blank-page
# phrases, then the bracketed prefixes of llm/summary.py's failure
# placeholders. A bare "error" is left out because it matches prose such as
# "measurement error"; filtering applies only to a bullet that starts with "[".
ERROR_MARKERS = [
    "no transcribable text",
    "transcription not possible",
    "no transcription possible",
    "empty page",
    "[error generating summary",
    "[summary generation failed",
    "[transcription failed",
    "[transcription error",
]

MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"

MIN_SAMPLES_FOR_ETA = 5
RECENT_SAMPLES_FOR_ETA = 10
ETA_BLEND_WEIGHT_OVERALL = 0.7
ETA_BLEND_WEIGHT_RECENT = 0.3

__all__ = [
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
    "BULLET_SPACE_AFTER_PT",
    "REF_SPACE_AFTER_PT",
    "BULLET_LEFT_INDENT_CM",
    "BULLET_HANGING_INDENT_CM",
    "REF_HANGING_INDENT_CM",
    "LOG_FORMAT_VERSION",
    "is_blank_transcription",
    "OPENAI_MODEL_PREFIXES",
    "ERROR_MARKERS",
    "MATH_NAMESPACE",
    "MIN_SAMPLES_FOR_ETA",
    "RECENT_SAMPLES_FOR_ETA",
    "ETA_BLEND_WEIGHT_OVERALL",
    "ETA_BLEND_WEIGHT_RECENT",
]
