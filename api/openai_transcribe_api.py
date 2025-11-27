"""OpenAI transcription API client - DEPRECATED.

This module is deprecated and maintained for backward compatibility.
Use `api.transcribe_api.TranscriptionManager` instead, which supports
multiple LLM providers (OpenAI, Anthropic, Google, OpenRouter).

Example migration:
    # Old (deprecated):
    from api.openai_transcribe_api import OpenAITranscriptionManager
    manager = OpenAITranscriptionManager(api_key, model_name)
    
    # New (recommended):
    from api.transcribe_api import TranscriptionManager
    manager = TranscriptionManager(model_name=model_name, provider="openai")
"""

from __future__ import annotations

import warnings

# Re-export from new module for backward compatibility
from api.transcribe_api import TranscriptionManager, TranscriptionManager as OpenAITranscriptionManager

# Issue deprecation warning when this module is imported
warnings.warn(
    "api.openai_transcribe_api is deprecated. Use api.transcribe_api instead. "
    "OpenAITranscriptionManager is now TranscriptionManager with provider='openai'.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["OpenAITranscriptionManager", "TranscriptionManager"]

# Legacy imports for any code that might use them directly
from api.base_llm_client import DEFAULT_MAX_RETRIES
