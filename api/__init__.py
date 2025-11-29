"""API layer for AutoExcerpter.

This package provides the LLM API integration layer:

- **llm_client**: Multi-provider LLM client factory using LangChain
- **base_llm_client**: Base class for LLM clients with retry and rate limiting
- **transcribe_api**: Image transcription via LLM vision APIs
- **summary_api**: Text summarization via LLM APIs
- **rate_limiter**: Adaptive rate limiting for API calls

Example:
    >>> from api import TranscriptionManager, SummaryManager
    >>> transcriber = TranscriptionManager(model_name="gpt-5-mini")
    >>> result = transcriber.transcribe_image(image_path)
"""

from api.llm_client import (
    LLMConfig,
    get_chat_model,
    get_model_capabilities,
    get_provider_for_model,
    is_provider_available,
    get_available_providers,
    ProviderType,
    SUPPORTED_PROVIDERS,
    MODEL_CAPABILITIES,
)
from api.base_llm_client import LLMClientBase, DEFAULT_MAX_RETRIES
from api.transcribe_api import TranscriptionManager
from api.summary_api import SummaryManager
from api.rate_limiter import RateLimiter

__all__ = [
    # LLM Client Factory
    "LLMConfig",
    "get_chat_model",
    "get_model_capabilities",
    "get_provider_for_model",
    "is_provider_available",
    "get_available_providers",
    "ProviderType",
    "SUPPORTED_PROVIDERS",
    "MODEL_CAPABILITIES",
    # Base Client
    "LLMClientBase",
    "DEFAULT_MAX_RETRIES",
    # API Managers
    "TranscriptionManager",
    "SummaryManager",
    # Rate Limiting
    "RateLimiter",
]
