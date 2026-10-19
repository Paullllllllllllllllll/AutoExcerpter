"""The OpenAI image detail a transcription request sends."""

from __future__ import annotations

import logging
from typing import Any

from autoexcerpter.llm.capabilities import detect_capabilities

DETAIL_VALUES = ("low", "high", "auto", "original")


def resolve_request_detail(
    model_config: dict[str, Any],
    provider: str | None,
    model_name: str,
    log: logging.Logger,
) -> str | None:
    """Return the ``detail`` of the image block, or None to send none.

    Reads the ``image_size`` option. Only OpenAI requests carry a detail;
    ``original`` falls back to ``high`` on models that do not accept it.
    """
    if provider != "openai":
        return None
    image_size = model_config.get("image_size")
    if not image_size:
        return None
    detail = str(image_size).strip().lower()
    if detail not in DETAIL_VALUES:
        log.warning(
            f"Ignoring unsupported image_size '{image_size}' for "
            f"{model_name}; expected low/high/auto/original."
        )
        return None
    capabilities = detect_capabilities(model_name)
    if not capabilities.supports_image_detail:
        log.warning(
            f"Model '{model_name}' does not support the image detail "
            "parameter; ignoring image_size."
        )
        return None
    if detail == "original" and not capabilities.supports_original_image_detail:
        log.warning(
            f"Model '{model_name}' does not support image_size "
            "'original'; falling back to 'high'."
        )
        return "high"
    return detail
