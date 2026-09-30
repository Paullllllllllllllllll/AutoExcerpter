"""Model detection utilities for provider-specific image processing.

Provides utilities for detecting the underlying model type from provider and model
names, particularly useful when using models through OpenRouter or other proxy services.

This enables correct image preprocessing even when using models via OpenRouter.
For example, 'google/gemini-2.5-flash' via OpenRouter should use Google config.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from config.constants import OPENAI_MODEL_PREFIXES

ModelType = Literal["openai", "google", "anthropic", "custom"]


def resolve_request_detail(
    model_config: dict[str, Any],
    provider: str | None,
    model_name: str,
    log: logging.Logger,
) -> str | None:
    """Resolve exactly the OpenAI detail sent by the transcription client."""
    from llm.capabilities import detect_capabilities

    if provider != "openai":
        return None
    image_size = model_config.get("image_size")
    if not image_size:
        return None
    detail = str(image_size).strip().lower()
    if detail not in ("low", "high", "auto", "original"):
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


def detect_model_type(provider: str, model_name: str | None = None) -> ModelType:
    """Detect the underlying model type from provider and model name.

    This allows correct preprocessing even when using models via OpenRouter.
    For example, 'google/gemini-2.5-flash' via OpenRouter should use Google config.

    Args:
        provider: The LLM provider name (e.g., 'openai', 'anthropic', 'google',
            'openrouter')
        model_name: The model name (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-2.5-flash')

    Returns:
        Model type: 'google', 'anthropic', 'openai', or 'custom'
    """
    provider = provider.lower()
    model_name = model_name.lower() if model_name else ""

    # Direct providers take precedence
    if provider == "custom":
        return "custom"
    if provider == "google":
        return "google"
    if provider == "anthropic":
        return "anthropic"
    if provider == "openai":
        return "openai"

    # For OpenRouter or unknown providers, detect from model name
    if model_name:
        # Google models
        if "gemini" in model_name or "google/" in model_name:
            return "google"
        # Anthropic models
        if "claude" in model_name or "anthropic/" in model_name:
            return "anthropic"
        # OpenAI models (shared prefix list plus the OpenRouter proxy prefix)
        if any(x in model_name for x in (*OPENAI_MODEL_PREFIXES, "openai/")):
            return "openai"

    # Default to OpenAI-style config
    return "openai"


def get_image_config_section_name(model_type: ModelType) -> str:
    """Get the image processing config section name for a model type.

    Args:
        model_type: The model type ('google', 'anthropic', or 'openai')

    Returns:
        Config section name (e.g., 'google_image_processing', 'api_image_processing')
    """
    section_map: dict[str, str] = {
        "custom": "custom_image_processing",
        "google": "google_image_processing",
        "anthropic": "anthropic_image_processing",
    }
    return section_map.get(model_type, "api_image_processing")


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "detect_model_type",
    "get_image_config_section_name",
    "ModelType",
]
