"""Base OpenAI client - DEPRECATED.

This module is deprecated and maintained for backward compatibility.
Use `api.base_llm_client.LLMClientBase` instead, which supports
multiple LLM providers (OpenAI, Anthropic, Google, OpenRouter).

Example migration:
    # Old (deprecated):
    from api.base_openai_client import OpenAIClientBase
    
    # New (recommended):
    from api.base_llm_client import LLMClientBase
"""

from __future__ import annotations

import warnings

# Re-export from new module for backward compatibility
from api.base_llm_client import LLMClientBase, LLMClientBase as OpenAIClientBase, DEFAULT_MAX_RETRIES

# Issue deprecation warning when this module is imported
warnings.warn(
    "api.base_openai_client is deprecated. Use api.base_llm_client instead. "
    "OpenAIClientBase is now LLMClientBase with multi-provider support.",
    DeprecationWarning,
    stacklevel=2
)

# Public API
__all__ = [
    "OpenAIClientBase",
    "LLMClientBase",
    "DEFAULT_MAX_RETRIES",
]
