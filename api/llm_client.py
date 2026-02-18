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

# Model capability profiles for parameter guarding
# These define which parameters each model family supports to prevent API errors
# Updated November 2025 with latest models from all providers
MODEL_CAPABILITIES: dict[str, dict[str, bool]] = {
    # ============== OpenAI Models ==============
    # GPT-5.1 family (Nov 2025) - latest flagship with improved reasoning
    "gpt-5.1-thinking": {"reasoning": True, "text_verbosity": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-5.1-instant": {"reasoning": True, "text_verbosity": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-5.1": {"reasoning": True, "text_verbosity": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # GPT-5.2 family (Feb 2026) - flagship coding and agentic model, 400k context
    "gpt-5.2": {"reasoning": True, "text_verbosity": True, "temperature": False, "max_tokens": True, "structured_output": True, "multimodal": True},
    # GPT-5 family (Aug 2025) - supports reasoning and text verbosity
    "gpt-5-mini": {"reasoning": True, "text_verbosity": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-5-nano": {"reasoning": True, "text_verbosity": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-5": {"reasoning": True, "text_verbosity": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # O-series reasoning models (no temperature support)
    "o4-mini": {"reasoning": True, "temperature": False, "max_tokens": True, "structured_output": True, "multimodal": True},
    "o4": {"reasoning": True, "temperature": False, "max_tokens": True, "structured_output": True, "multimodal": True},
    "o3-mini": {"reasoning": True, "temperature": False, "max_tokens": True, "structured_output": True, "multimodal": False},
    "o3": {"reasoning": True, "temperature": False, "max_tokens": True, "structured_output": True, "multimodal": False},
    "o1-mini": {"reasoning": True, "temperature": False, "max_tokens": True, "structured_output": True, "multimodal": False},
    "o1": {"reasoning": True, "temperature": False, "max_tokens": True, "structured_output": True, "multimodal": False},
    # GPT-4.1 family (no reasoning support)
    "gpt-4.1-mini": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-4.1-nano": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-4.1": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # GPT-4o family (no reasoning support)
    "gpt-4o-mini": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-4o": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # GPT-4 legacy (no reasoning, no multimodal for base)
    "gpt-4-turbo": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gpt-4": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": False},
    
    # ============== Anthropic Claude Models ==============
    # Claude 4.6 family (Feb 2026) - native extended thinking (Opus 4.6), 200k context
    "claude-opus-4-6": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": True},
    "claude-sonnet-4-6": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": True},
    # Claude 4.5 family (Oct-Nov 2025) - latest generation
    "claude-opus-4-5": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": True},
    "claude-sonnet-4-5": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": True},
    "claude-haiku-4-5": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": False},
    # Claude 4 family
    "claude-opus-4": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": True},
    "claude-sonnet-4": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": True},
    # Claude 3.5/3.7 family
    "claude-3-7-sonnet": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": True},
    "claude-3-5-sonnet": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": False},
    "claude-3-5-haiku": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": False},
    # Claude 3 family (legacy)
    "claude-3-opus": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": False},
    "claude-3-sonnet": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": False},
    "claude-3-haiku": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True, "extended_thinking": False},
    # Fallback for any claude model
    "claude": {"reasoning": False, "text_verbosity": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    
    # ============== Google Gemini Models ==============
    # Gemini 3 Pro Preview (Feb 2026) - 1M token context, thinking
    "gemini-3-pro-preview": {"reasoning": False, "thinking": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # Gemini 3 Flash Preview (Feb 2026) - 1M token context, thinking
    "gemini-3-flash-preview": {"reasoning": False, "thinking": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # Gemini 3 Pro (Nov 2025) - latest flagship with thinking
    "gemini-3-pro": {"reasoning": False, "thinking": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # Gemini 3 Flash (catch-all for flash variants)
    "gemini-3-flash": {"reasoning": False, "thinking": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # Gemini 2.5 family (Mar 2025) - thinking models
    "gemini-2.5-pro": {"reasoning": False, "thinking": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gemini-2.5-flash": {"reasoning": False, "thinking": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gemini-2.5-flash-lite": {"reasoning": False, "thinking": True, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # Gemini 2.0 family
    "gemini-2.0-flash": {"reasoning": False, "thinking": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gemini-2.0-flash-lite": {"reasoning": False, "thinking": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # Gemini 1.5 family (legacy)
    "gemini-1.5-pro": {"reasoning": False, "thinking": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    "gemini-1.5-flash": {"reasoning": False, "thinking": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
    # Fallback for any gemini model
    "gemini": {"reasoning": False, "thinking": False, "temperature": True, "max_tokens": True, "structured_output": True, "multimodal": True},
}


def get_model_capabilities(model_name: str) -> dict[str, bool]:
    """
    Get capability profile for a model based on its name prefix.
    
    This enables parameter guarding - filtering out unsupported parameters
    before they're sent to the API, preventing errors like:
    "Unsupported parameter: 'reasoning_effort' is not supported with this model"
    
    Args:
        model_name: The model name (e.g., "gpt-5-mini", "o3-mini", "claude-sonnet-4-5-20250929")
        
    Returns:
        Dictionary of capabilities (reasoning, temperature, etc.)
    """
    model_lower = model_name.lower()
    
    # Check for exact prefix matches (order matters - more specific first)
    # OpenAI models (most specific to least specific)
    openai_prefixes = [
        "gpt-5.1-thinking", "gpt-5.1-instant", "gpt-5.1",
        "gpt-5.2",
        "gpt-5-mini", "gpt-5-nano", "gpt-5",
        "o4-mini", "o4", "o3-mini", "o3", "o1-mini", "o1",
        "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1",
        "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4",
    ]
    
    # Anthropic models (most specific to least specific)
    anthropic_prefixes = [
        "claude-opus-4-6", "claude-sonnet-4-6",
        "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
        "claude-opus-4", "claude-sonnet-4",
        "claude-3-7-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku",
        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
        "claude",
    ]
    
    # Google models (most specific to least specific)
    google_prefixes = [
        "gemini-3-pro-preview", "gemini-3-flash-preview",
        "gemini-3-pro", "gemini-3-flash",
        "gemini-2.5-pro", "gemini-2.5-flash-lite", "gemini-2.5-flash",
        "gemini-2.0-flash-lite", "gemini-2.0-flash",
        "gemini-1.5-pro", "gemini-1.5-flash",
        "gemini",
    ]
    
    # Check all prefixes in order
    for prefix in openai_prefixes + anthropic_prefixes + google_prefixes:
        if model_lower.startswith(prefix):
            caps = MODEL_CAPABILITIES.get(prefix, {})
            if caps:
                logger.debug(f"Model '{model_name}' matched profile '{prefix}'")
                return caps
    
    # Default capabilities (conservative - no special features)
    logger.debug(f"Model '{model_name}' using default capability profile")
    return {
        "reasoning": False,
        "text_verbosity": False,
        "thinking": False,
        "extended_thinking": False,
        "temperature": True,
        "max_tokens": True,
        "structured_output": False,
        "multimodal": False,
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
    "SUPPORTED_PROVIDERS",
    "MODEL_CAPABILITIES",
    "ProviderType",
]
