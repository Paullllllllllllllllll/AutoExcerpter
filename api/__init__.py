"""API layer for AutoExcerpter.

This package provides the LLM API integration layer:

- **providers**: Multi-provider abstraction with capability gating and cross-provider reasoning
- **transcribe_api**: Image transcription via LLM vision APIs
- **summary_api**: Text summarization via LLM APIs
- **rate_limiter**: Adaptive rate limiting for API calls

Example:
    >>> from api.providers import get_provider
    >>> provider = get_provider(model="gpt-5", provider="openai")
    >>> result = await provider.transcribe_image(image_path, system_prompt="...")
"""

from api.providers import (
    BaseProvider,
    ProviderCapabilities,
    TranscriptionResult,
    ProviderType,
    get_provider,
    get_available_providers,
    detect_provider_from_model,
)
from api.transcribe_api import TranscriptionManager
from api.summary_api import SummaryManager
from api.rate_limiter import RateLimiter

__all__ = [
    # Provider Abstraction
    "BaseProvider",
    "ProviderCapabilities",
    "TranscriptionResult",
    "ProviderType",
    "get_provider",
    "get_available_providers",
    "detect_provider_from_model",
    # API Managers
    "TranscriptionManager",
    "SummaryManager",
    # Rate Limiting
    "RateLimiter",
]
