"""LLM client layer for AutoExcerpter.

Unified interface over multiple providers (OpenAI, Anthropic, Google,
OpenRouter, custom endpoints).

Public interface:

- ``TranscriptionManager`` — image-to-text via LLM vision
- ``SummaryManager`` — structured text summarization
- ``LLMConfig``, ``get_chat_model``, ``get_model_capabilities`` — factory
- ``ProviderType``, ``ProviderCapabilities``, ``CapabilityError``,
  ``ensure_image_support`` — capability model
- ``detect_provider``, ``detect_capabilities`` — provider detection
- ``CustomEndpointCapabilities`` — capability flags for custom endpoints
- ``DailyTokenTracker``, ``get_token_tracker`` — token usage tracking

Internal (not re-exported; import from the sub-module if needed for testing):

- ``llm.base.LLMClientBase`` — abstract base class
- ``llm.rate_limit.RateLimiter`` — adaptive rate limiter (managers own it)

Example:
    >>> from llm import get_chat_model, LLMConfig
    >>> config = LLMConfig(model="gpt-5-mini", provider="openai")
    >>> model = get_chat_model(config)
"""

from llm.capabilities import (
    CapabilityError,
    ProviderCapabilities,
    detect_capabilities,
    detect_provider,
    ensure_image_support,
)
from llm.client import (
    LLMConfig,
    ProviderType,
    get_chat_model,
    get_model_capabilities,
)
from llm.summary import SummaryManager
from llm.token_tracker import DailyTokenTracker, get_token_tracker
from llm.transcription import TranscriptionManager
from llm.types import CustomEndpointCapabilities

__all__ = [
    "TranscriptionManager",
    "SummaryManager",
    "LLMConfig",
    "get_chat_model",
    "get_model_capabilities",
    "ProviderType",
    "ProviderCapabilities",
    "CapabilityError",
    "ensure_image_support",
    "detect_provider",
    "detect_capabilities",
    "CustomEndpointCapabilities",
    "DailyTokenTracker",
    "get_token_tracker",
]
