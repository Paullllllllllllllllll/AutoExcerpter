"""Model detection utilities for provider-specific image processing.

Provides utilities for detecting the underlying model type from provider and model names,
particularly useful when using models through OpenRouter or other proxy services.

This enables correct image preprocessing even when using models via OpenRouter.
For example, 'google/gemini-2.5-flash' via OpenRouter should use Google config.
"""

from __future__ import annotations

from typing import Literal

ModelType = Literal["openai", "google", "anthropic"]


def detect_model_type(provider: str, model_name: str | None = None) -> ModelType:
    """Detect the underlying model type from provider and model name.
    
    This allows correct preprocessing even when using models via OpenRouter.
    For example, 'google/gemini-2.5-flash' via OpenRouter should use Google config.
    
    Args:
        provider: The LLM provider name (e.g., 'openai', 'anthropic', 'google', 'openrouter')
        model_name: The model name (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-2.5-flash')
    
    Returns:
        Model type: 'google', 'anthropic', or 'openai'
    """
    provider = provider.lower()
    model_name = model_name.lower() if model_name else ""
    
    # Direct providers take precedence
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
        # OpenAI models
        if any(x in model_name for x in ["gpt-", "o1", "o3", "o4", "openai/"]):
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
    if model_type == "google":
        return "google_image_processing"
    elif model_type == "anthropic":
        return "anthropic_image_processing"
    else:
        return "api_image_processing"


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "detect_model_type",
    "get_image_config_section_name",
    "ModelType",
]
