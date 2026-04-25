"""Type definitions for LLM API payloads and capabilities.

- ``TranscriptionResult``, ``PageInformation``, ``SummaryContent``,
  ``SummaryResult`` — structured payloads produced by
  :mod:`llm.transcription` and :mod:`llm.summary`.
- ``CustomEndpointCapabilities`` — capability flags declared in
  ``config/defaults/model.yaml`` for the ``custom`` provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class TranscriptionResult(TypedDict, total=False):
    """Type definition for transcription API results."""

    page: int
    image: str
    transcription: str
    processing_time: float
    retries: int
    api_retries: int
    schema_retries: dict[str, int]
    error: str | None
    original_input_order_index: int


class PageInformation(TypedDict, total=False):
    """Type definition for page information metadata."""

    page_number_integer: int | None
    page_number_type: str  # "roman", "arabic", "none"
    page_types: list[str]  # 1-3 classifications: content, bibliography, abstract, etc.


class SummaryContent(TypedDict, total=False):
    """Type definition for summary content structure."""

    page_information: PageInformation
    bullet_points: list[str] | None
    references: list[str] | None


class SummaryResult(TypedDict, total=False):
    """Type definition for summary API results.

    Uses flat structure with all fields at top level (no nested "summary" key).
    """

    # Metadata fields
    page: int
    original_input_order_index: int
    image_filename: str

    # Content fields (from LLM output, merged at top level)
    page_information: PageInformation
    bullet_points: list[str] | None
    references: list[str] | None

    # Processing metadata
    processing_time: float
    provider: str
    api_response: dict[str, Any]
    schema_retries: dict[str, int]
    error: str | None


@dataclass(frozen=True)
class CustomEndpointCapabilities:
    """Capabilities declared for a custom OpenAI-compatible endpoint.

    Three usage patterns:
      A) supports_structured_output=True, use_plain_text_prompt=False
         → Identical to commercial providers (response_format enforcement).
      B) supports_structured_output=False, use_plain_text_prompt=True
         → Simplified plain-text prompt, raw text response, no JSON.
      C) supports_structured_output=False, use_plain_text_prompt=False
         → Normal prompts with schema in prompt text; relies on
           validation_failure retry to catch malformed JSON responses.
    """

    supports_vision: bool = True
    supports_structured_output: bool = False
    use_plain_text_prompt: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CustomEndpointCapabilities:
        """Create from a capabilities configuration dictionary.

        Enforces: use_plain_text_prompt=True forces
        supports_structured_output=False.
        """
        supports_vision = bool(data.get("supports_vision", True))
        supports_structured_output = bool(
            data.get("supports_structured_output", False)
        )
        use_plain_text_prompt = bool(data.get("use_plain_text_prompt", False))

        if use_plain_text_prompt and supports_structured_output:
            logger.warning(
                "use_plain_text_prompt=true forces "
                "supports_structured_output=false"
            )
            supports_structured_output = False

        return cls(
            supports_vision=supports_vision,
            supports_structured_output=supports_structured_output,
            use_plain_text_prompt=use_plain_text_prompt,
        )


__all__ = [
    "TranscriptionResult",
    "PageInformation",
    "SummaryContent",
    "SummaryResult",
    "CustomEndpointCapabilities",
]
