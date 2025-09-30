"""Base OpenAI client with shared retry logic and error handling."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from typing import Any, Dict, Optional

from openai import OpenAI

from modules import app_config as config
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Public API
__all__ = [
    "OpenAIClientBase",
    "DEFAULT_MAX_RETRIES",
    "BACKOFF_BASE",
    "BACKOFF_MULTIPLIER_RATE_LIMIT",
    "BACKOFF_MULTIPLIER_TIMEOUT",
    "BACKOFF_MULTIPLIER_OTHER",
    "JITTER_MIN",
    "JITTER_MAX",
]

# Constants for retry logic
DEFAULT_MAX_RETRIES = 5
BACKOFF_BASE = 1.0
BACKOFF_MULTIPLIER_RATE_LIMIT = 2.0
BACKOFF_MULTIPLIER_TIMEOUT = 1.5
BACKOFF_MULTIPLIER_OTHER = 2.0
JITTER_MIN = 0.5
JITTER_MAX = 1.0


class OpenAIClientBase:
    """Base class for OpenAI API clients with common retry and error handling logic."""

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
            api_key: OpenAI API key
            model_name: Model to use (e.g., 'gpt-5-mini')
            timeout: Request timeout in seconds
            rate_limiter: Optional RateLimiter instance
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

    def _classify_error(self, error_message: str) -> tuple[bool, str]:
        """
        Classify an error and determine if it's retryable.

        Args:
            error_message: The error message to classify (lowercased)

        Returns:
            Tuple of (is_retryable, error_type)
        """
        error_lower = error_message.lower()

        # Check for rate limit errors
        if any(err in error_lower for err in ["rate limit", "too many", "429", "retry-after"]):
            return True, "rate_limit"

        # Check for server errors
        if any(err in error_lower for err in ["server error", "500", "502", "503", "504", "service unavailable"]):
            return True, "server"

        # Check for timeout errors
        if any(err in error_lower for err in ["timeout", "timed out", "deadline"]):
            return True, "timeout"

        # Check for network errors
        if any(err in error_lower for err in ["connection", "network", "temporarily unavailable"]):
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
            retry_count: Current retry attempt number
            error_type: Type of error encountered
            base: Base time for backoff calculation

        Returns:
            Wait time in seconds
        """
        jitter = random.uniform(JITTER_MIN, JITTER_MAX)

        if error_type == "rate_limit":
            return base * (BACKOFF_MULTIPLIER_RATE_LIMIT ** retry_count) * jitter
        elif error_type == "timeout":
            return base * (BACKOFF_MULTIPLIER_TIMEOUT ** retry_count) * jitter
        else:
            return base * (BACKOFF_MULTIPLIER_OTHER ** (retry_count - 1)) * jitter

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
        """Report error to rate limiter."""
        if self.rate_limiter is not None:
            self.rate_limiter.report_error(is_rate_limit_or_server)

    @staticmethod
    def _extract_output_text(data: Any) -> str:
        """
        Normalize Responses API output into a single text string.

        Args:
            data: Response data from OpenAI API

        Returns:
            Extracted text content
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
                conv = getattr(data, "to_dict", None) or getattr(data, "model_dump", None)
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
            Dictionary containing request statistics
        """
        import statistics

        avg_time = statistics.mean(self.processing_times) if self.processing_times else 0
        total_requests = self.successful_requests + self.failed_requests
        success_rate = (self.successful_requests / max(1, total_requests)) * 100

        return {
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_processing_time": round(avg_time, 2),
            "recent_success_rate": round(success_rate, 1),
        }
