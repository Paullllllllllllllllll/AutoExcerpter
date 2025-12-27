"""Provider factory for dynamic LLM provider selection.

Creates provider instances based on configuration or explicit parameters.
Supports auto-detection of provider from model name.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional, Type

from api.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


# Lazy import mapping to avoid circular imports and unnecessary dependencies
_PROVIDER_CLASSES: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "api.providers.openai_provider.OpenAIProvider",
    ProviderType.ANTHROPIC: "api.providers.anthropic_provider.AnthropicProvider",
    ProviderType.GOOGLE: "api.providers.google_provider.GoogleProvider",
    ProviderType.OPENROUTER: "api.providers.openrouter_provider.OpenRouterProvider",
}

# Environment variable names for API keys
_API_KEY_ENV_VARS: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "OPENAI_API_KEY",
    ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderType.GOOGLE: "GOOGLE_API_KEY",
    ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
}


def _import_provider_class(provider_type: ProviderType) -> Type[BaseProvider]:
    """Dynamically import a provider class."""
    module_path = _PROVIDER_CLASSES[provider_type]
    module_name, class_name = module_path.rsplit(".", 1)
    
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_available_providers() -> list[ProviderType]:
    """Return list of provider types that have API keys configured."""
    available = []
    for provider_type, env_var in _API_KEY_ENV_VARS.items():
        if os.environ.get(env_var):
            available.append(provider_type)
    return available


def detect_provider_from_model(model_name: str) -> ProviderType:
    """Attempt to detect the provider from the model name.
    
    Args:
        model_name: The model name/identifier
    
    Returns:
        Best-guess ProviderType based on model name patterns
    """
    m = model_name.lower().strip()
    
    # OpenRouter format: provider/model (e.g., "anthropic/claude-sonnet-4-5")
    if "/" in m:
        prefix = m.split("/")[0]
        if prefix in ("openai", "anthropic", "google", "meta", "mistral", "deepseek"):
            return ProviderType.OPENROUTER
    
    # OpenAI models (GPT-5.1, GPT-5, GPT-4.1, GPT-4o, o-series)
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt", "text-")):
        return ProviderType.OPENAI
    
    # Anthropic models (Claude family)
    if "claude" in m:
        return ProviderType.ANTHROPIC
    
    # Google models (Gemini family)
    if "gemini" in m or m.startswith("models/"):
        return ProviderType.GOOGLE
    
    # Models commonly accessed via OpenRouter
    if any(x in m for x in ["llama", "mistral", "mixtral", "qwen", "deepseek"]):
        return ProviderType.OPENROUTER
    
    # Default to OpenAI
    logger.debug(f"Could not detect provider for model '{model_name}', defaulting to OpenAI")
    return ProviderType.OPENAI


def get_api_key_for_provider(
    provider_type: ProviderType,
    api_key: Optional[str] = None,
) -> str:
    """Get the API key for a provider.
    
    Args:
        provider_type: The provider type
        api_key: Optional explicit API key
    
    Returns:
        The API key to use
    
    Raises:
        ValueError: If no API key is available
    """
    if api_key:
        return api_key
    
    env_var = _API_KEY_ENV_VARS.get(provider_type)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key
    
    # Fallback: try OPENAI_API_KEY for OpenRouter if OPENROUTER_API_KEY not set
    if provider_type == ProviderType.OPENROUTER:
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            logger.warning(
                "OPENROUTER_API_KEY not set, falling back to OPENAI_API_KEY. "
                "This may not work with all OpenRouter features."
            )
            return key
    
    raise ValueError(
        f"No API key found for provider {provider_type.value}. "
        f"Set {env_var} environment variable or pass api_key parameter."
    )


def get_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: Optional[float] = None,
    **kwargs,
) -> BaseProvider:
    """Create a provider instance.
    
    This is the main entry point for creating LLM provider instances.
    It handles provider detection, API key resolution, and configuration.
    
    Args:
        provider: Provider name ("openai", "anthropic", "google", "openrouter")
                  If None, attempts to detect from model name
        model: Model name/identifier
        api_key: Optional explicit API key
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        timeout: Request timeout in seconds
        **kwargs: Provider-specific configuration (e.g., reasoning_config)
    
    Returns:
        Configured provider instance
    
    Raises:
        ValueError: If provider cannot be determined or API key is missing
    
    Example:
        >>> provider = get_provider(model="gpt-5", temperature=0.0)
        >>> result = await provider.transcribe_image(image_path, system_prompt="...")
    """
    # Load defaults from config if not provided
    if model is None or provider is None:
        try:
            from modules.config_loader import get_config_loader
            config = get_config_loader().get_model_config()
            tm = config.get("transcription_model", {})
            
            if model is None:
                model = tm.get("name", "gpt-4o")
            if provider is None:
                provider = tm.get("provider")  # May still be None
            
            # Load other defaults from config
            if "temperature" not in kwargs and tm.get("temperature") is not None:
                temperature = float(tm.get("temperature", 0.0))
            if tm.get("max_output_tokens") is not None:
                max_tokens = int(tm.get("max_output_tokens", max_tokens))
            elif tm.get("max_tokens") is not None:
                max_tokens = int(tm.get("max_tokens", max_tokens))
            
            # Load optional parameters
            for key in ["top_p", "frequency_penalty", "presence_penalty", "top_k"]:
                if key not in kwargs and tm.get(key) is not None:
                    kwargs[key] = tm.get(key)
            
            # Load reasoning config (cross-provider)
            if "reasoning_config" not in kwargs and tm.get("reasoning") is not None:
                kwargs["reasoning_config"] = tm.get("reasoning")
                    
        except Exception as e:
            logger.warning(f"Could not load config defaults: {e}")
            if model is None:
                model = "gpt-4o"
    
    # Determine provider type
    if provider is None:
        provider_type = detect_provider_from_model(model)
        logger.info(f"Auto-detected provider '{provider_type.value}' for model '{model}'")
    else:
        try:
            provider_type = ProviderType(provider.lower())
        except ValueError:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported: {', '.join(p.value for p in ProviderType)}"
            )
    
    # Get API key
    resolved_api_key = get_api_key_for_provider(provider_type, api_key)
    
    # Import and instantiate provider
    provider_class = _import_provider_class(provider_type)
    
    return provider_class(
        api_key=resolved_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )


def get_provider_for_transcription(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> BaseProvider:
    """Create a provider configured for transcription.
    
    Loads configuration from model.yaml and concurrency.yaml.
    This is a convenience function that handles all config loading.
    
    Args:
        api_key: Optional explicit API key
        model: Optional model override
        provider: Optional provider override
    
    Returns:
        Configured provider instance ready for transcription
    """
    from modules.config_loader import get_config_loader
    from modules.concurrency_helper import get_service_tier
    
    # Load model config
    config = get_config_loader().get_model_config()
    tm = config.get("transcription_model", {})
    
    # Load service tier from concurrency config
    service_tier = get_service_tier("transcription")
    
    # Build kwargs
    kwargs: Dict[str, Any] = {}
    
    if service_tier:
        kwargs["service_tier"] = service_tier
    
    # Load optional parameters from config
    for key in ["top_p", "frequency_penalty", "presence_penalty", "top_k"]:
        if tm.get(key) is not None:
            kwargs[key] = tm.get(key)
    
    # Load reasoning config
    if tm.get("reasoning") is not None:
        kwargs["reasoning_config"] = tm.get("reasoning")
    
    return get_provider(
        provider=provider or tm.get("provider"),
        model=model or tm.get("name"),
        api_key=api_key,
        temperature=float(tm.get("temperature", 0.0)),
        max_tokens=int(
            tm.get("max_output_tokens")
            or tm.get("max_tokens", 4096)
        ),
        **kwargs,
    )
