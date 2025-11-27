"""OpenAI API client for structured summarization - DEPRECATED.

This module is deprecated and maintained for backward compatibility.
Use `api.summary_api.SummaryManager` instead, which supports
multiple LLM providers (OpenAI, Anthropic, Google, OpenRouter).

Example migration:
    # Old (deprecated):
    from api.openai_api import OpenAISummaryManager
    manager = OpenAISummaryManager(api_key, model_name)
    
    # New (recommended):
    from api.summary_api import SummaryManager
    manager = SummaryManager(model_name=model_name, provider="openai")
"""

from __future__ import annotations

import warnings

# Re-export from new module for backward compatibility
from api.summary_api import SummaryManager, SummaryManager as OpenAISummaryManager

# Issue deprecation warning when this module is imported
warnings.warn(
    "api.openai_api is deprecated. Use api.summary_api instead. "
    "OpenAISummaryManager is now SummaryManager with provider='openai'.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["OpenAISummaryManager", "SummaryManager"]

# Legacy imports for any code that might use them directly
from api.base_llm_client import DEFAULT_MAX_RETRIES
