"""Multi-provider LLM integration package.

Provides a unified interface for multiple LLM providers:
- OpenAI (GPT-5, GPT-4o, o-series)
- Anthropic (Claude 4.5, Claude 3.x)
- Google (Gemini 3, 2.5, 2.0, 1.5)
- OpenRouter (200+ models via unified API)

Usage:
    >>> from api.providers import get_provider, ProviderType
    >>> provider = get_provider(model="gpt-5", provider="openai")
    >>> result = await provider.transcribe_image(image_path, system_prompt="...")

Lazy imports are used to avoid circular dependencies with config loading.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    
    # Base classes and types
    if name in (
        "BaseProvider",
        "ProviderCapabilities",
        "TranscriptionResult",
    ):
        from api.providers.base import (
            BaseProvider,
            ProviderCapabilities,
            TranscriptionResult,
        )
        return {
            "BaseProvider": BaseProvider,
            "ProviderCapabilities": ProviderCapabilities,
            "TranscriptionResult": TranscriptionResult,
        }[name]
    
    # Factory functions and types
    if name in (
        "ProviderType",
        "get_provider",
        "get_available_providers",
        "detect_provider_from_model",
        "get_api_key_for_provider",
    ):
        from api.providers.factory import (
            ProviderType,
            get_provider,
            get_available_providers,
            detect_provider_from_model,
            get_api_key_for_provider,
        )
        return {
            "ProviderType": ProviderType,
            "get_provider": get_provider,
            "get_available_providers": get_available_providers,
            "detect_provider_from_model": detect_provider_from_model,
            "get_api_key_for_provider": get_api_key_for_provider,
        }[name]
    
    # Individual provider classes (lazy load)
    if name == "OpenAIProvider":
        from api.providers.openai_provider import OpenAIProvider
        return OpenAIProvider
    
    if name == "AnthropicProvider":
        from api.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider
    
    if name == "GoogleProvider":
        from api.providers.google_provider import GoogleProvider
        return GoogleProvider
    
    if name == "OpenRouterProvider":
        from api.providers.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider
    
    raise AttributeError(f"module 'api.providers' has no attribute '{name}'")


__all__ = [
    # Base classes
    "BaseProvider",
    "ProviderCapabilities",
    "TranscriptionResult",
    # Factory
    "ProviderType",
    "get_provider",
    "get_available_providers",
    "detect_provider_from_model",
    "get_api_key_for_provider",
    # Provider classes
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenRouterProvider",
]
