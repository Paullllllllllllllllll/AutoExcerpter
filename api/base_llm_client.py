"""Base LLM client leveraging LangChain's built-in capabilities.

This module provides the foundational LLM client implementation with:

1. **Multi-Provider Support**: Works with OpenAI, Anthropic, Google, and OpenRouter
   through LangChain's unified interface.

2. **LangChain Built-in Retry**: API error retries (rate limits, timeouts, server errors)
   are handled by LangChain's built-in exponential backoff via `max_retries` parameter.
   
3. **Schema-Specific Retries**: Optional retries based on model-returned boolean flags
   in responses (e.g., no_transcribable_text, contains_no_semantic_content).

4. **Rate Limiting Integration**: Works with RateLimiter to throttle requests and prevent
   API quota exhaustion (complementary to LangChain's retry).

5. **Configuration Loading**: Dynamically loads model parameters from YAML configuration files.

6. **Statistics Tracking**: Monitors request success rates, processing times, and error patterns.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from api.llm_client import LLMConfig, get_chat_model, get_model_capabilities, ProviderType
from modules import app_config as config
from modules.concurrency_helper import get_service_tier
from modules.config_loader import get_config_loader
from modules.logger import setup_logger

logger = setup_logger(__name__)


def _load_retry_config() -> dict[str, Any]:
    """Load retry and backoff configuration from concurrency.yaml."""
    try:
        config_loader = get_config_loader()
        retry_cfg = config_loader.get_concurrency_config().get("retry", {})
        
        if not retry_cfg:
            logger.debug("No retry config found in concurrency.yaml, using defaults")
            
        return retry_cfg
    except Exception as e:
        logger.warning(f"Error loading retry config: {e}. Using defaults.")
        return {}


# Load retry configuration from concurrency.yaml
_RETRY_CONFIG = _load_retry_config()

# Constants for retry logic (loaded from config with fallback defaults)
DEFAULT_MAX_RETRIES = _RETRY_CONFIG.get("max_attempts", 5)

_JITTER = _RETRY_CONFIG.get("jitter", {})
JITTER_MIN = _JITTER.get("min", 0.5)
JITTER_MAX = _JITTER.get("max", 1.0)


class LLMClientBase:
    """
    Base class for LLM clients leveraging LangChain's built-in capabilities.
    
    This class provides:
    - Multi-provider support through LangChain
    - API error retries handled by LangChain's built-in exponential backoff
    - Optional schema-specific retry logic for content validation flags
    - Rate limiting integration (complementary to LangChain's retry)
    - Model configuration loading from YAML
    - Request statistics tracking
    
    Subclasses should implement specific API endpoint logic (transcription, summarization, etc.)
    """

    def __init__(
        self,
        model_name: str,
        provider: ProviderType | None = None,
        api_key: str | None = None,
        timeout: int = config.OPENAI_API_TIMEOUT,
        rate_limiter: Any | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        service_tier: str | None = None,
    ) -> None:
        """
        Initialize the base LLM client.

        Args:
            model_name: Model to use (e.g., 'gpt-5-mini', 'claude-sonnet-4-5-20250929').
            provider: Provider name (openai, anthropic, google, openrouter).
                     If None, inferred from model name.
            api_key: Optional API key. If None, uses environment variable.
            timeout: Request timeout in seconds.
            rate_limiter: Optional RateLimiter instance for request throttling.
            max_retries: Max retry attempts. LangChain handles exponential backoff automatically.
            service_tier: OpenAI service tier ("flex", "default", "auto"). OpenAI-only.
        """
        self.model_name = model_name
        self.provider = provider
        self.timeout = timeout
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries

        # Create LLM configuration - LangChain handles retry with exponential backoff
        llm_config = LLMConfig(
            model=model_name,
            provider=provider,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,  # LangChain handles exponential backoff
            service_tier=service_tier,
        )
        
        # Store the resolved provider
        self.provider = llm_config.provider

        # Instantiate LangChain chat model (with built-in retry)
        self.chat_model: BaseChatModel = get_chat_model(llm_config)

        # Statistics tracking
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times: deque = deque(maxlen=50)
        
        # Model configuration and service tier (loaded by subclasses)
        self.model_config: dict[str, Any] = {}
        self.service_tier: str = service_tier or "auto"
        
        # Schema retry configuration (loaded by subclasses)
        self.schema_retry_config: dict[str, Any] = {}
        
        logger.info(
            f"Initialized LLM client: provider={self.provider}, model={model_name}, "
            f"max_retries={max_retries} (handled by LangChain)"
        )

    def _load_model_config(self, config_key: str) -> dict[str, Any]:
        """Load model configuration from model.yaml."""
        try:
            config_loader = get_config_loader()
            model_cfg = config_loader.get_model_config()
            config_dict = model_cfg.get(config_key, {})
            
            if config_dict:
                logger.debug(
                    f"Loaded {config_key} config: {config_dict.get('name', 'unknown')}"
                )
            else:
                logger.debug(f"No {config_key} config found, using defaults")
                
            return config_dict
        except Exception as e:
            logger.warning(f"Error loading model config for {config_key}: {e}")
            return {}

    def _determine_service_tier(self, api_type: str = "transcription") -> str:
        """
        Determine service tier from configuration.
        
        Service tier controls request prioritization (OpenAI-specific):
        - 'flex': Lower cost, may queue during high demand
        - 'default': Standard processing speed
        - 'priority': Fastest processing, higher cost
        - 'auto': Let provider decide
        
        For non-OpenAI providers, this returns 'auto' as it's not applicable.
        
        Args:
            api_type: Type of API request ('transcription' or 'summary').
            
        Returns:
            Service tier string.
        """
        # Service tier is primarily an OpenAI concept
        if self.provider != "openai":
            return "auto"
        
        # Use centralized service tier configuration
        return get_service_tier(api_type)

    def _wait_for_rate_limit(self) -> None:
        """Wait for rate limiter capacity if rate limiter is configured."""
        if self.rate_limiter is not None:
            self.rate_limiter.wait_for_capacity()

    def _report_success(self) -> None:
        """Report successful request to rate limiter and update stats."""
        self.successful_requests += 1
        if self.rate_limiter is not None:
            self.rate_limiter.report_success()

    def _report_error(self, is_rate_limit_or_server: bool) -> None:
        """
        Report error to rate limiter.
        
        Args:
            is_rate_limit_or_server: True if error is rate limit or server error.
        """
        if self.rate_limiter is not None:
            self.rate_limiter.report_error(is_rate_limit_or_server)

    @staticmethod
    def _extract_output_text(data: Any) -> str:
        """
        Normalize LLM response output into a single text string.

        Args:
            data: Response data from LLM API.

        Returns:
            Extracted text content.
        """
        # Handle LangChain AIMessage responses directly
        if isinstance(data, AIMessage):
            parts: list[str] = []
            content = data.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        text = block.get("text") or block.get("output_text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
            elif isinstance(content, str) and content.strip():
                parts.append(content)

            result = "".join(parts).strip()
            if not result:
                logger.warning(
                    "Empty content extracted from LangChain AIMessage response."
                )
            return result

        # Try SDK convenience attribute
        try:
            text_attr = getattr(data, "output_text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                return text_attr.strip()
        except Exception:
            pass

        # Try dict-style access
        if isinstance(data, dict) and isinstance(data.get("output_text"), str):
            output_text = data["output_text"].strip()
            if output_text:
                return output_text

        # Fallback: reconstruct from output list
        try:
            obj = data
            if not isinstance(obj, dict):
                conv = getattr(data, "to_dict", None) or getattr(
                    data, "model_dump", None
                )
                if callable(conv):
                    obj = conv()

            output = obj.get("output") if isinstance(obj, dict) else None
            parts = []
            if isinstance(output, list):
                for item in output:
                    if not isinstance(item, dict):
                        continue

                    item_content = item.get("content", [])
                    if not item_content:
                        continue

                    for content_item in item_content:
                        if not isinstance(content_item, dict):
                            continue

                        content_type = content_item.get("type")
                        if content_type in ["output_text", "text"]:
                            text = content_item.get("text")
                            if isinstance(text, str) and text.strip():
                                parts.append(text)

            result = "".join(parts).strip()

            if not result:
                logger.warning(
                    "Empty content extracted from API response. "
                    f"Response structure: output={'list' if isinstance(output, list) else type(output).__name__}, "
                    f"items={len(output) if isinstance(output, list) else 0}"
                )

            return result
        except Exception as e:
            logger.warning(f"Error extracting output text: {e}")
            return ""

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about API usage."""
        import statistics

        avg_time = (
            statistics.mean(self.processing_times) if self.processing_times else 0
        )
        total_requests = self.successful_requests + self.failed_requests
        success_rate = (self.successful_requests / max(1, total_requests)) * 100

        return {
            "provider": self.provider,
            "model": self.model_name,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_processing_time": round(avg_time, 2),
            "recent_success_rate": round(success_rate, 1),
        }

    def _load_schema_retry_config(self, api_type: str) -> dict[str, Any]:
        """Load schema-specific retry configuration from concurrency.yaml."""
        try:
            config_loader = get_config_loader()
            concurrency_cfg = config_loader.get_concurrency_config()
            
            schema_retries = (
                concurrency_cfg
                .get("retry", {})
                .get("schema_retries", {})
                .get(api_type, {})
            )
            
            if schema_retries:
                logger.debug(f"Loaded schema retry config for {api_type}: {list(schema_retries.keys())}")
            else:
                logger.debug(f"No schema retry config found for {api_type}")
                
            return schema_retries
        except Exception as e:
            logger.warning(f"Error loading schema retry config for {api_type}: {e}")
            return {}

    def _should_retry_for_schema_flag(
        self, 
        flag_name: str, 
        flag_value: Any,
        current_attempt: int
    ) -> tuple[bool, float, int]:
        """Check if we should retry based on a schema flag value."""
        # Get configuration for this specific flag
        flag_config = self.schema_retry_config.get(flag_name, {})
        
        # Check if retries are enabled for this flag
        if not flag_config.get("enabled", False):
            return False, 0.0, 0
        
        # Check if flag is set to a truthy value
        if not flag_value:
            return False, 0.0, 0
        
        # Get max attempts for this flag
        max_attempts = flag_config.get("max_attempts", 0)
        
        # Check if we've exceeded max attempts
        if current_attempt >= max_attempts:
            return False, 0.0, max_attempts
        
        # Calculate backoff time
        backoff_base = flag_config.get("backoff_base", 2.0)
        backoff_multiplier = flag_config.get("backoff_multiplier", 1.5)
        jitter = random.uniform(JITTER_MIN, JITTER_MAX)
        
        backoff_time = backoff_base * (backoff_multiplier ** current_attempt) * jitter
        
        return True, backoff_time, max_attempts

    def _build_invoke_kwargs(self) -> dict[str, Any]:
        """Build provider-appropriate invocation kwargs with capability guarding."""
        invoke_kwargs: dict[str, Any] = {}
        
        # Get model capabilities for parameter guarding
        capabilities = get_model_capabilities(self.model_name)
        
        # Add max_output_tokens from config (most models support this)
        if capabilities.get("max_tokens", True):
            max_tokens = self.model_config.get("max_output_tokens")
            if max_tokens:
                if self.provider == "openai":
                    invoke_kwargs["max_output_tokens"] = max_tokens
                elif self.provider == "anthropic":
                    invoke_kwargs["max_tokens"] = max_tokens
                elif self.provider == "google":
                    invoke_kwargs["max_output_tokens"] = max_tokens
                else:
                    invoke_kwargs["max_tokens"] = max_tokens
        
        # OpenAI-specific: service tier (supported by all OpenAI models)
        if self.provider == "openai" and self.service_tier:
            invoke_kwargs["service_tier"] = self.service_tier
        
        # OpenAI-specific: reasoning parameters (only GPT-5 and o-series)
        # GUARDED: Only add if model supports reasoning
        if self.provider == "openai" and capabilities.get("reasoning", False):
            if "reasoning" in self.model_config:
                reasoning_cfg = self.model_config["reasoning"]
                if isinstance(reasoning_cfg, dict) and "effort" in reasoning_cfg:
                    invoke_kwargs["reasoning"] = {"effort": reasoning_cfg["effort"]}
                    logger.debug(f"Added reasoning params for {self.model_name}")
        elif "reasoning" in self.model_config:
            logger.debug(
                f"Skipping reasoning params for {self.model_name} (not supported)"
            )
        
        # OpenAI-specific: text verbosity parameters (only GPT-5 family)
        # GUARDED: Only add if model supports text_verbosity
        if self.provider == "openai" and capabilities.get("text_verbosity", False):
            if "text" in self.model_config:
                text_cfg = self.model_config["text"]
                if isinstance(text_cfg, dict):
                    text_params = {}
                    if "verbosity" in text_cfg:
                        text_params["verbosity"] = text_cfg["verbosity"]
                    if text_params:
                        invoke_kwargs["text"] = text_params
                        logger.debug(f"Added text verbosity params for {self.model_name}")
        elif "text" in self.model_config:
            logger.debug(
                f"Skipping text verbosity params for {self.model_name} (not supported)"
            )
        
        return invoke_kwargs


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "LLMClientBase",
    "DEFAULT_MAX_RETRIES",
]
