"""Base OpenAI client with shared retry logic and error handling."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from modules import app_config as config
from modules.config_loader import ConfigLoader
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Public API
__all__ = [
    "OpenAIClientBase",
    "DEFAULT_MAX_RETRIES",
]


def _load_retry_config() -> Dict[str, Any]:
    """
    Load retry and backoff configuration from concurrency.yaml.
    
    Returns:
        Dictionary with retry configuration, or defaults if loading fails.
    """
    try:
        config_loader = ConfigLoader()
        config_loader.load_configs()
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
BACKOFF_BASE = _RETRY_CONFIG.get("backoff_base", 1.0)

_BACKOFF_MULTIPLIERS = _RETRY_CONFIG.get("backoff_multipliers", {})
BACKOFF_MULTIPLIER_RATE_LIMIT = _BACKOFF_MULTIPLIERS.get("rate_limit", 2.0)
BACKOFF_MULTIPLIER_TIMEOUT = _BACKOFF_MULTIPLIERS.get("timeout", 1.5)
BACKOFF_MULTIPLIER_SERVER = _BACKOFF_MULTIPLIERS.get("server_error", 2.0)
BACKOFF_MULTIPLIER_OTHER = _BACKOFF_MULTIPLIERS.get("other", 2.0)

_JITTER = _RETRY_CONFIG.get("jitter", {})
JITTER_MIN = _JITTER.get("min", 0.5)
JITTER_MAX = _JITTER.get("max", 1.0)


class OpenAIClientBase:
    """
    Base class for OpenAI API clients with common retry and error handling logic.
    
    This class provides:
    - Retry logic with exponential backoff for API errors
    - Schema-specific retry logic for content validation flags
    - Error classification and handling
    - Rate limiting integration
    - Model configuration loading
    - Service tier determination
    - Request statistics tracking
    
    Subclasses should implement specific API endpoint logic (transcription, summarization, etc.)
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout: int = config.OPENAI_API_TIMEOUT,
        rate_limiter: Optional[Any] = None,
    ) -> None:
        """
        Initialize the base OpenAI client.

        Args:
            api_key: OpenAI API key.
            model_name: Model to use (e.g., 'gpt-5-mini').
            timeout: Request timeout in seconds.
            rate_limiter: Optional RateLimiter instance for request throttling.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.rate_limiter = rate_limiter

        # Statistics tracking
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times = deque(maxlen=50)
        
        # Model configuration and service tier (loaded by subclasses)
        self.model_config: Dict[str, Any] = {}
        self.service_tier: str = "auto"
        
        # Schema retry configuration (loaded by subclasses)
        self.schema_retry_config: Dict[str, Any] = {}

    def _load_model_config(self, config_key: str) -> Dict[str, Any]:
        """
        Load model configuration from model.yaml.
        
        Args:
            config_key: Key for model config (e.g., 'transcription_model', 'summary_model').
            
        Returns:
            Model configuration dictionary.
        """
        try:
            config_loader = ConfigLoader()
            config_loader.load_configs()
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
        Determine OpenAI service tier from configuration.
        
        Service tier controls request prioritization:
        - 'flex': Lower cost, may queue during high demand
        - 'default': Standard processing speed
        - 'priority': Fastest processing, higher cost
        - 'auto': Let OpenAI decide
        
        Args:
            api_type: Type of API request ('transcription' or 'summary').
            
        Returns:
            Service tier string.
        """
        try:
            config_loader = ConfigLoader()
            config_loader.load_configs()
            concurrency_cfg = config_loader.get_concurrency_config()
            
            # Get service tier from new config structure
            service_tier = (
                concurrency_cfg
                .get("api_requests", {})
                .get(api_type, {})
                .get("service_tier")
            )
            
            if service_tier:
                return service_tier
                
            # Fallback to legacy OPENAI_USE_FLEX setting
            return "flex" if config.OPENAI_USE_FLEX else "auto"
            
        except Exception as e:
            logger.debug(f"Error determining service tier: {e}")
            return "flex" if config.OPENAI_USE_FLEX else "auto"

    def _classify_error(self, error_message: str) -> tuple[bool, str]:
        """
        Classify an error and determine if it's retryable.

        Args:
            error_message: The error message to classify.

        Returns:
            Tuple of (is_retryable, error_type).
        """
        error_lower = error_message.lower()

        # Check for rate limit errors
        if any(
            err in error_lower
            for err in ["rate limit", "too many", "429", "retry-after"]
        ):
            return True, "rate_limit"

        # Check for server errors
        if any(
            err in error_lower
            for err in [
                "server error",
                "500",
                "502",
                "503",
                "504",
                "service unavailable",
            ]
        ):
            return True, "server"

        # Check for timeout errors
        if any(err in error_lower for err in ["timeout", "timed out", "deadline"]):
            return True, "timeout"

        # Check for network errors
        if any(
            err in error_lower
            for err in ["connection", "network", "temporarily unavailable"]
        ):
            return True, "network"

        # Check for resource unavailable
        if "resource unavailable" in error_lower:
            return True, "resource_unavailable"

        return False, "other"

    def _calculate_backoff_time(
        self, retry_count: int, error_type: str, base: float = BACKOFF_BASE
    ) -> float:
        """
        Calculate backoff time with jitter based on error type.

        Args:
            retry_count: Current retry attempt number.
            error_type: Type of error encountered.
            base: Base time for backoff calculation.

        Returns:
            Wait time in seconds.
        """
        jitter = random.uniform(JITTER_MIN, JITTER_MAX)

        if error_type == "rate_limit":
            multiplier = BACKOFF_MULTIPLIER_RATE_LIMIT
        elif error_type == "timeout":
            multiplier = BACKOFF_MULTIPLIER_TIMEOUT
        elif error_type == "server":
            multiplier = BACKOFF_MULTIPLIER_SERVER
        else:
            multiplier = BACKOFF_MULTIPLIER_OTHER

        return base * (multiplier ** retry_count) * jitter

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
        Normalize Responses API output into a single text string.

        Args:
            data: Response data from OpenAI API.

        Returns:
            Extracted text content.
        """
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
                    # Skip items with null role or empty content
                    if not isinstance(item, dict):
                        continue

                    item_content = item.get("content", [])
                    if not item_content:
                        continue

                    # Extract text from content items
                    for content_item in item_content:
                        if not isinstance(content_item, dict):
                            continue

                        # Handle both "output_text" and "text" types
                        content_type = content_item.get("type")
                        if content_type in ["output_text", "text"]:
                            text = content_item.get("text")
                            if isinstance(text, str) and text.strip():
                                parts.append(text)

            result = "".join(parts).strip()

            # Log warning if empty content detected
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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API usage.

        Returns:
            Dictionary containing request statistics.
        """
        import statistics

        avg_time = (
            statistics.mean(self.processing_times) if self.processing_times else 0
        )
        total_requests = self.successful_requests + self.failed_requests
        success_rate = (self.successful_requests / max(1, total_requests)) * 100

        return {
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_processing_time": round(avg_time, 2),
            "recent_success_rate": round(success_rate, 1),
        }

    def _load_schema_retry_config(self, api_type: str) -> Dict[str, Any]:
        """
        Load schema-specific retry configuration from concurrency.yaml.
        
        Args:
            api_type: Type of API request ('transcription' or 'summary').
            
        Returns:
            Schema retry configuration dictionary.
        """
        try:
            config_loader = ConfigLoader()
            config_loader.load_configs()
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
    ) -> Tuple[bool, float, int]:
        """
        Check if we should retry based on a schema flag value.
        
        Args:
            flag_name: Name of the schema flag (e.g., 'no_transcribable_text').
            flag_value: Value of the flag from the API response.
            current_attempt: Current retry attempt number for this flag.
            
        Returns:
            Tuple of (should_retry, backoff_time, max_attempts).
        """
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
