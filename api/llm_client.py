"""Multi-provider LLM client using LangChain's unified interface.

This module provides a provider-agnostic interface for interacting with various LLM providers
through LangChain. It supports:

- OpenAI (GPT-4, GPT-5, o-series models)
- Anthropic (Claude models)
- Google (Gemini models)
- OpenRouter (access to multiple providers through unified API)

The module uses LangChain's `init_chat_model` for automatic provider detection and
standardized message handling across all providers.

Usage:
    >>> from api.llm_client import get_chat_model, LLMConfig
    >>> config = LLMConfig(model="gpt-5-mini", provider="openai")
    >>> model = get_chat_model(config)
    >>> response = model.invoke([SystemMessage(content="..."), HumanMessage(content="...")])

Provider Detection:
- If model string contains ":" (e.g., "openai:gpt-5"), provider is auto-detected
- Otherwise, provider must be specified in config or will be inferred from model name
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from api.model_capabilities import (
    ProviderCapabilities,
    detect_capabilities,
    detect_provider as _detect_provider_from_caps,
)
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Type alias for supported providers
ProviderType = Literal["openai", "anthropic", "google", "openrouter"]

# Supported providers and their LangChain package requirements
# Updated November 2025 with latest model prefixes
SUPPORTED_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "package": "langchain-openai",
        "class": "ChatOpenAI",
        "env_key": "OPENAI_API_KEY",
        # GPT-5.1, GPT-5, GPT-4.1, GPT-4o, GPT-4, o-series (o1, o3, o4)
        "model_prefixes": ["gpt-5", "gpt-4", "o1", "o3", "o4", "text-"],
    },
    "anthropic": {
        "package": "langchain-anthropic",
        "class": "ChatAnthropic",
        "env_key": "ANTHROPIC_API_KEY",
        # Claude 4.5, Claude 4, Claude 3.x families
        "model_prefixes": ["claude-"],
    },
    "google": {
        "package": "langchain-google-genai",
        "class": "ChatGoogleGenerativeAI",
        "env_key": "GOOGLE_API_KEY",
        # Gemini 3, 2.5, 2.0, 1.5 families
        "model_prefixes": ["gemini-"],
    },
    "openrouter": {
        "package": "langchain-openai",  # Uses OpenAI-compatible API
        "class": "ChatOpenAI",
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_prefixes": [],  # OpenRouter supports many models from all providers
    },
}

# Backward-compatible re-exports from api.model_capabilities.
# New code should import directly from api.model_capabilities.
MODEL_CAPABILITIES = None  # Removed; use detect_capabilities() instead


def get_model_capabilities(model_name: str) -> dict[str, bool]:
    """Backward-compatible wrapper around detect_capabilities().
    
    Returns a dict[str, bool] that mirrors the old MODEL_CAPABILITIES format
    so existing callers (base_llm_client, transcribe_api, tests) continue to work.
    
    New code should use ``detect_capabilities()`` from ``api.model_capabilities``
    which returns a typed ``ProviderCapabilities`` dataclass.
    """
    caps = detect_capabilities(model_name)
    return {
        "reasoning": caps.is_reasoning_model,
        "text_verbosity": caps.supports_text_verbosity,
        "thinking": caps.is_reasoning_model and caps.provider_name == "google",
        "extended_thinking": caps.is_reasoning_model and caps.provider_name == "anthropic",
        "temperature": caps.supports_temperature,
        "max_tokens": True,  # Always allow setting max tokens
        "structured_output": caps.supports_structured_output,
        "multimodal": caps.supports_vision,
    }


@dataclass
class LLMConfig:
    """Configuration for LLM client initialization.
    
    Attributes:
        model: Model identifier (e.g., "gpt-5-mini", "claude-sonnet-4-5-20250929")
        provider: Provider name ("openai", "anthropic", "google", "openrouter")
        api_key: Optional API key (defaults to environment variable)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts. LangChain handles retry with exponential backoff.
        temperature: Model temperature (0.0 - 2.0)
        max_tokens: Maximum output tokens
        service_tier: OpenAI service tier ("flex", "default", "auto")
        extra_kwargs: Additional provider-specific parameters
    """
    model: str
    provider: ProviderType | None = None
    api_key: str | None = None
    timeout: int = 900
    max_retries: int = 5  # LangChain handles exponential backoff automatically
    temperature: float | None = None
    max_tokens: int | None = None
    service_tier: str | None = None  # OpenAI-specific: "flex", "default", "auto"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Infer provider from model name if not specified."""
        if self.provider is None:
            self.provider = _infer_provider(self.model)
            logger.debug(f"Inferred provider '{self.provider}' for model '{self.model}'")


def _infer_provider(model: str) -> ProviderType:
    """Infer the provider from the model name.
    
    Args:
        model: Model identifier string
        
    Returns:
        Inferred provider name
        
    Raises:
        ValueError: If provider cannot be inferred
    """
    # Check for explicit provider prefix (e.g., "openai:gpt-5")
    if ":" in model:
        provider = model.split(":")[0].lower()
        if provider in SUPPORTED_PROVIDERS:
            return provider  # type: ignore
    
    # Check model name patterns
    model_lower = model.lower()
    
    for provider_name, provider_info in SUPPORTED_PROVIDERS.items():
        prefixes = provider_info.get("model_prefixes", [])
        for prefix in prefixes:
            if model_lower.startswith(prefix):
                return provider_name  # type: ignore
    
    # Default to OpenAI for unknown models
    logger.warning(f"Could not infer provider for model '{model}', defaulting to 'openai'")
    return "openai"


def _get_api_key(provider: ProviderType, config_key: str | None = None) -> str:
    """Get API key for the specified provider.
    
    Args:
        provider: Provider name
        config_key: Optional API key from configuration
        
    Returns:
        API key string
        
    Raises:
        EnvironmentError: If API key is not found
    """
    if config_key:
        return config_key
    
    provider_info = SUPPORTED_PROVIDERS.get(provider, {})
    env_key = provider_info.get("env_key", f"{provider.upper()}_API_KEY")
    
    api_key = os.environ.get(env_key)
    if not api_key:
        raise EnvironmentError(
            f"API key not found for provider '{provider}'. "
            f"Please set the {env_key} environment variable."
        )
    
    return api_key


def get_chat_model(config: LLMConfig) -> BaseChatModel:
    """Create a LangChain chat model instance for the specified configuration.
    
    This function creates a provider-appropriate chat model using LangChain's
    unified interface. It handles:
    - Provider detection and initialization
    - API key resolution
    - Timeout and retry configuration
    - Provider-specific parameters
    
    Args:
        config: LLM configuration object
        
    Returns:
        Configured BaseChatModel instance
        
    Raises:
        ImportError: If required provider package is not installed
        EnvironmentError: If API key is not found
        ValueError: If provider is not supported
    """
    provider = config.provider or _infer_provider(config.model)
    
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Get API key
    api_key = _get_api_key(provider, config.api_key)
    
    # Parse model name (remove provider prefix if present)
    model_name = config.model
    if ":" in model_name:
        model_name = model_name.split(":", 1)[1]
    
    # Build common kwargs
    kwargs: dict[str, Any] = {
        "model": model_name,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }
    
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    
    # Add extra kwargs
    kwargs.update(config.extra_kwargs)
    
    # Provider-specific initialization
    if provider == "openai":
        return _create_openai_model(api_key, kwargs, config.service_tier)
    elif provider == "anthropic":
        return _create_anthropic_model(api_key, kwargs)
    elif provider == "google":
        return _create_google_model(api_key, kwargs)
    elif provider == "openrouter":
        return _create_openrouter_model(api_key, kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _create_openai_model(
    api_key: str, 
    kwargs: dict[str, Any], 
    service_tier: str | None = None,
) -> BaseChatModel:
    """Create an OpenAI chat model instance.
    
    Uses LangChain's built-in retry with exponential backoff via max_retries parameter.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for OpenAI models. "
            "Install with: pip install langchain-openai"
        )
    
    kwargs["api_key"] = api_key
    
    # Add service tier if specified (OpenAI-specific feature)
    if service_tier:
        kwargs["service_tier"] = service_tier
    
    logger.debug(f"Creating OpenAI model: {kwargs.get('model')} (max_retries={kwargs.get('max_retries', 2)})")
    return ChatOpenAI(**kwargs)


def _create_anthropic_model(api_key: str, kwargs: dict[str, Any]) -> BaseChatModel:
    """Create an Anthropic chat model instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic package is required for Anthropic models. "
            "Install with: pip install langchain-anthropic"
        )
    
    kwargs["api_key"] = api_key
    
    # Anthropic uses 'max_tokens' instead of 'max_output_tokens'
    if "max_output_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
    
    logger.debug(f"Creating Anthropic model: {kwargs.get('model')}")
    return ChatAnthropic(**kwargs)


def _create_google_model(api_key: str, kwargs: dict[str, Any]) -> BaseChatModel:
    """Create a Google Generative AI chat model instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai package is required for Google models. "
            "Install with: pip install langchain-google-genai"
        )
    
    kwargs["google_api_key"] = api_key
    
    # Google uses different parameter names
    if "max_tokens" in kwargs:
        kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
    
    logger.debug(f"Creating Google model: {kwargs.get('model')}")
    return ChatGoogleGenerativeAI(**kwargs)


def _create_openrouter_model(api_key: str, kwargs: dict[str, Any]) -> BaseChatModel:
    """Create an OpenRouter chat model instance (OpenAI-compatible API)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for OpenRouter models. "
            "Install with: pip install langchain-openai"
        )
    
    kwargs["api_key"] = api_key
    kwargs["base_url"] = SUPPORTED_PROVIDERS["openrouter"]["base_url"]
    
    # OpenRouter recommends including site info in headers
    default_headers = kwargs.get("default_headers", {})
    default_headers.setdefault("HTTP-Referer", "https://github.com/autoexcerpter")
    default_headers.setdefault("X-Title", "AutoExcerpter")
    kwargs["default_headers"] = default_headers
    
    logger.debug(f"Creating OpenRouter model: {kwargs.get('model')}")
    return ChatOpenAI(**kwargs)


def get_provider_for_model(model: str) -> ProviderType:
    """Get the provider name for a given model.
    
    Args:
        model: Model identifier string
        
    Returns:
        Provider name
    """
    return _infer_provider(model)


def is_provider_available(provider: ProviderType) -> bool:
    """Check if a provider's API key is available.
    
    Args:
        provider: Provider name
        
    Returns:
        True if API key is available
    """
    provider_info = SUPPORTED_PROVIDERS.get(provider, {})
    env_key = provider_info.get("env_key", f"{provider.upper()}_API_KEY")
    return bool(os.environ.get(env_key))


def get_available_providers() -> list[ProviderType]:
    """Get a list of providers with available API keys.
    
    Returns:
        List of available provider names
    """
    return [p for p in SUPPORTED_PROVIDERS.keys() if is_provider_available(p)]  # type: ignore


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "LLMConfig",
    "get_chat_model",
    "get_model_capabilities",
    "detect_capabilities",
    "SUPPORTED_PROVIDERS",
    "ProviderType",
]
