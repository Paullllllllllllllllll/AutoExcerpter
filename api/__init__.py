"""API layer for AutoExcerpter.

This package provides the LLM API integration layer:

- **llm_client**: Multi-provider LLM client via LangChain
- **model_capabilities**: Provider/model capability detection
- **transcribe_api**: Image transcription via LLM vision APIs
- **summary_api**: Text summarization via LLM APIs
- **rate_limiter**: Adaptive rate limiting for API calls

Example:
    >>> from api.llm_client import get_chat_model, LLMConfig
    >>> config = LLMConfig(model="gpt-5-mini", provider="openai")
    >>> model = get_chat_model(config)
"""

from api.transcribe_api import TranscriptionManager
from api.summary_api import SummaryManager
from api.rate_limiter import RateLimiter
from api.llm_client import (
    LLMConfig,
    get_chat_model,
    get_model_capabilities,
    ProviderType,
)
from api.model_capabilities import detect_capabilities, ProviderCapabilities

__all__ = [
    # LLM Client
    "LLMConfig",
    "get_chat_model",
    "get_model_capabilities",
    "ProviderType",
    # Capabilities
    "detect_capabilities",
    "ProviderCapabilities",
    # API Managers
    "TranscriptionManager",
    "SummaryManager",
    # Rate Limiting
    "RateLimiter",
]
