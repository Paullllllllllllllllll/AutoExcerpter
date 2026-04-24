"""Base LLM client leveraging LangChain's built-in capabilities.

This module provides the foundational LLM client implementation with:

1. **Multi-Provider Support**: Works with OpenAI, Anthropic, Google, and OpenRouter
   through LangChain's unified interface.

2. **Application-Level Retry**: API error retries (rate limits, timeouts, server errors)
   are handled by ``_invoke_with_retry`` with exponential backoff and per-attempt token
   tracking. SDK-level retries are disabled (``max_retries=0``) so that every attempt
   is visible to the token tracker.

3. **Schema-Specific Retries**: Optional retries based on model-returned flags
   in responses (e.g., no_transcribable_text, page_type_null_bullets).

4. **Rate Limiting Integration**: Works with RateLimiter to throttle requests and prevent
   API quota exhaustion (complementary to application-level retry).

5. **Configuration Loading**: Dynamically loads model parameters from YAML configuration files.

6. **Statistics Tracking**: Monitors request success rates, processing times, and error patterns.
"""

from __future__ import annotations

import json as _json
import random
import statistics
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from api.llm_client import (
    LLMConfig,
    get_chat_model,
    get_model_capabilities,
    ProviderType,
)
from modules.concurrency_helper import get_api_timeout, get_service_tier
from modules.config_loader import get_config_loader
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker

logger = setup_logger(__name__)


def _load_retry_config() -> dict[str, Any]:
    """Load retry and backoff configuration from concurrency.yaml."""
    try:
        config_loader = get_config_loader()
        retry_cfg: dict[str, Any] = config_loader.get_concurrency_config().get("retry", {})

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

_ANTHROPIC_EFFORT_TO_BUDGET: dict[str, int] = {
    "none": 0,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}

_GOOGLE_EFFORT_TO_BUDGET: dict[str, int] = {
    "none": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}


# ============================================================================
# Output Text Extraction Chain
# ============================================================================
def _extract_from_aimessage(data: Any) -> str | None:
    """Extract text from a LangChain AIMessage response."""
    if not isinstance(data, AIMessage):
        return None
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


def _extract_from_output_attribute(data: Any) -> str | None:
    """Try SDK convenience ``output_text`` attribute."""
    try:
        text_attr = getattr(data, "output_text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            return text_attr.strip()
    except Exception:
        pass
    return None


def _extract_from_dict(data: Any) -> str | None:
    """Try dict-style ``output_text`` key access."""
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        output_text: str = data["output_text"].strip()
        if output_text:
            return output_text
    return None


def _extract_from_nested_output(data: Any) -> str | None:
    """Reconstruct text from nested ``output`` list structure."""
    try:
        obj = data
        if not isinstance(obj, dict):
            conv = getattr(data, "to_dict", None) or getattr(
                data, "model_dump", None
            )
            if callable(conv):
                obj = conv()

        output = obj.get("output") if isinstance(obj, dict) else None
        parts: list[str] = []
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
        return None


def _extract_from_structured_output_wrapper(data: Any) -> str | None:
    """Extract text from ``with_structured_output(include_raw=True)`` response.

    LangChain's ``with_structured_output(include_raw=True)`` returns a dict
    with keys ``raw`` (AIMessage), ``parsed`` (dict or pydantic), and
    ``parsing_error``.  This extractor handles that shape.
    """
    if not isinstance(data, dict) or "raw" not in data:
        return None

    # Prefer the parsed dict (already schema-validated by LangChain)
    parsed = data.get("parsed")
    if isinstance(parsed, dict):
        try:
            return _json.dumps(parsed, ensure_ascii=False)
        except (TypeError, ValueError):
            pass

    # Fallback: extract text from the raw AIMessage
    raw = data.get("raw")
    if raw is not None:
        return _extract_from_aimessage(raw)

    return None


# Ordered extraction chain: first non-None result wins
_EXTRACTORS: list[Callable[[Any], str | None]] = [
    _extract_from_aimessage,
    _extract_from_output_attribute,
    _extract_from_structured_output_wrapper,
    _extract_from_dict,
    _extract_from_nested_output,
]


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
        timeout: int | None = None,
        rate_limiter: Any | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        service_tier: str | None = None,
        section_hint: str | None = None,
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
            section_hint: YAML config section for custom endpoint disambiguation.
        """
        self.model_name = model_name
        self.provider = provider
        self.timeout = timeout if timeout is not None else get_api_timeout()
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries

        # Create LLM configuration — SDK retries disabled; handled by _invoke_with_retry
        llm_config = LLMConfig(
            model=model_name,
            provider=provider,
            api_key=api_key,
            timeout=self.timeout,  # Use resolved timeout (never None)
            max_retries=0,  # Disable SDK retries; handled by _invoke_with_retry
            service_tier=service_tier,
            section_hint=section_hint,
        )

        # Store the resolved provider
        self.provider = llm_config.provider

        # Instantiate LangChain chat model (no SDK-level retry)
        self.chat_model: BaseChatModel = get_chat_model(llm_config)

        # Statistics tracking
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times: deque[float] = deque(maxlen=50)

        # Model configuration and service tier (loaded by subclasses)
        self.model_config: dict[str, Any] = {}
        self.service_tier: str = service_tier or "auto"

        # Schema retry configuration (loaded by subclasses)
        self.schema_retry_config: dict[str, Any] = {}

        # Output schema for structured output (set by subclasses)
        self._output_schema: dict[str, Any] | None = None

        logger.info(
            f"Initialized LLM client: provider={self.provider}, model={model_name}, "
            f"max_retries={max_retries} (application-level via _invoke_with_retry)"
        )

    def _load_model_config(self, config_key: str) -> dict[str, Any]:
        """Load model configuration from model.yaml."""
        try:
            config_loader = get_config_loader()
            model_cfg = config_loader.get_model_config()
            config_dict: dict[str, Any] = model_cfg.get(config_key, {})

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
        """Normalize LLM response output into a single text string.

        Tries a chain of extraction strategies in order:
        1. LangChain AIMessage
        2. SDK ``output_text`` attribute
        3. Dict-style ``output_text`` key
        4. Nested ``output`` list reconstruction

        Args:
            data: Response data from LLM API.

        Returns:
            Extracted text content.
        """
        for extractor in _EXTRACTORS:
            result = extractor(data)
            if result is not None:
                return result
        logger.warning("Could not extract output text from response")
        return ""

    def _build_text_format(
        self, default_name: str = "json_schema"
    ) -> dict[str, Any] | None:
        """Build the structured output format specification from the output schema.

        Args:
            default_name: Default schema name if not specified in the schema dict.

        Returns:
            JSON schema format dict, or None if no valid schema is available.
        """
        schema = self._output_schema
        if not isinstance(schema, dict):
            return None

        name = schema.get("name", default_name)
        strict = bool(schema.get("strict", True))
        schema_obj = schema.get("schema", schema)

        if not isinstance(schema_obj, dict) or not schema_obj:
            return None

        return {
            "type": "json_schema",
            "name": name,
            "schema": schema_obj,
            "strict": strict,
        }

    def _get_structured_chat_model(self) -> Any:
        """Get chat model with structured output for each provider.

        Provider-specific approaches:
        - OpenAI: Native response_format parameter (guaranteed JSON) - handled separately
        - Custom: Same as OpenAI when supports_structured_output=True (Mode A),
          otherwise bare model with no structured output (Modes B/C)
        - Anthropic/Google: Prompt-based JSON with markdown stripping fallback
          (LangChain's with_structured_output has tool naming compatibility issues)
        - OpenRouter: Default tool-based structured output (OpenAI-compatible)

        Returns:
            Chat model with structured output, or base chat model.
        """
        if self.provider == "openai":
            return self.chat_model

        # Custom: structured output via response_format kwargs (Mode A) or
        # none at all (Modes B/C).  Never use with_structured_output since
        # most custom endpoints do not support tool-based function calling.
        if self.provider == "custom":
            return self.chat_model

        if self.provider in ("anthropic", "google"):
            return self.chat_model

        # OpenRouter: Use tool-based structured output (OpenAI-compatible)
        if self._output_schema:
            schema = self._output_schema.get("schema", self._output_schema)
            if isinstance(schema, dict) and schema:
                return self.chat_model.with_structured_output(
                    schema,
                    include_raw=True,
                )

        return self.chat_model

    def _apply_structured_output_kwargs(self, invoke_kwargs: dict[str, Any]) -> None:
        """Apply provider-specific structured output parameters to invoke kwargs.

        Modifies invoke_kwargs in-place to add structured output format for
        OpenAI (response_format), Custom Mode A (response_format), and
        Google (response_mime_type + response_schema).

        Args:
            invoke_kwargs: The invocation kwargs dict to modify.
        """
        if self.provider == "openai":
            text_format = self._build_text_format()
            if text_format:
                if "text" in invoke_kwargs:
                    invoke_kwargs["text"]["format"] = text_format
                else:
                    invoke_kwargs["response_format"] = text_format

        if self.provider == "custom":
            custom_caps = getattr(self, "custom_capabilities", None)
            if custom_caps and custom_caps.supports_structured_output:
                # Mode A: apply response_format for Chat Completions API.
                # _build_text_format() returns the Responses API shape
                # (top-level name/schema/strict).  Custom endpoints use
                # the Chat Completions API which nests those fields under
                # a "json_schema" key.
                text_format = self._build_text_format()
                if text_format:
                    chat_completions_format = {
                        "type": "json_schema",
                        "json_schema": {
                            k: v
                            for k, v in text_format.items()
                            if k != "type"
                        },
                    }
                    invoke_kwargs["response_format"] = chat_completions_format

        if self.provider == "google":
            schema_obj = (
                self._output_schema.get("schema")
                if isinstance(self._output_schema, dict)
                and "schema" in self._output_schema
                else self._output_schema
            )
            if isinstance(schema_obj, dict) and schema_obj:
                invoke_kwargs.setdefault("response_mime_type", "application/json")
                invoke_kwargs.setdefault("response_schema", schema_obj)

    def _report_token_usage(self, response: Any, context_label: str) -> None:
        """Report token usage from a LangChain response to the token tracker.

        Args:
            response: LangChain model response with usage_metadata.
            context_label: Description for the log message (e.g. "Transcription for page_001.png").
        """
        try:
            usage_meta = getattr(response, "usage_metadata", None)
            if not usage_meta or not isinstance(usage_meta, dict):
                logger.warning(
                    f"[TOKEN] {context_label}: usage_metadata missing or not a dict"
                )
                return

            total_tokens = usage_meta.get("total_tokens")

            # Fallback: compute from input_tokens + output_tokens (Anthropic-style)
            if not total_tokens or not isinstance(total_tokens, int):
                input_tokens = usage_meta.get("input_tokens", 0)
                output_tokens = usage_meta.get("output_tokens", 0)
                if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                    total_tokens = input_tokens + output_tokens

            if total_tokens and isinstance(total_tokens, int):
                token_tracker = get_token_tracker()
                token_tracker.add_tokens(total_tokens)
                logger.debug(
                    f"[TOKEN] {context_label}: "
                    f"added {total_tokens} tokens (total now: {token_tracker.get_tokens_used_today():,})"
                )
            else:
                logger.warning(
                    f"[TOKEN] {context_label}: no usable token counts in usage_metadata"
                )
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(
                f"Error reporting token usage for {context_label}: {e}"
            )

    def _extract_tokens_from_exception(self, exc: Exception, context_label: str) -> None:
        """Extract token usage from a failed API call's exception.

        Provider SDK exceptions often carry usage data in ``body.usage`` or
        ``response.json()["usage"]``.  This method attempts both strategies and
        reports recovered tokens to the tracker.  It never raises.

        Args:
            exc: The exception from a failed ``invoke()`` call.
            context_label: Description for the log message.
        """
        try:
            usage: dict[str, Any] | None = None

            # Strategy 1: exc.body["usage"] (OpenAI / Anthropic SDK exceptions)
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                usage = body.get("usage")

            # Strategy 2: exc.response.json()["usage"] (raw httpx response)
            if usage is None:
                resp = getattr(exc, "response", None)
                if resp is not None:
                    try:
                        resp_json = resp.json()
                        if isinstance(resp_json, dict):
                            usage = resp_json.get("usage")
                    except Exception:
                        pass

            if not isinstance(usage, dict):
                logger.debug(
                    f"[TOKEN] {context_label}: no usage data in exception"
                )
                return

            # Try total_tokens, then prompt_tokens+completion_tokens (OpenAI),
            # then input_tokens+output_tokens (Anthropic)
            total = usage.get("total_tokens")
            if not isinstance(total, int) or total <= 0:
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                if isinstance(prompt, int) and isinstance(completion, int) and (prompt + completion) > 0:
                    total = prompt + completion
                else:
                    inp = usage.get("input_tokens", 0)
                    out = usage.get("output_tokens", 0)
                    if isinstance(inp, int) and isinstance(out, int) and (inp + out) > 0:
                        total = inp + out

            if isinstance(total, int) and total > 0:
                token_tracker = get_token_tracker()
                token_tracker.add_tokens(total)
                logger.info(
                    f"[TOKEN] {context_label}: recovered {total} tokens from failed request "
                    f"(total now: {token_tracker.get_tokens_used_today():,})"
                )
            else:
                logger.debug(
                    f"[TOKEN] {context_label}: usage dict present but no usable counts"
                )
        except Exception:
            # Never propagate — token tracking is best-effort
            logger.debug(
                f"[TOKEN] {context_label}: exception while extracting tokens from error"
            )

    @staticmethod
    def _classify_error(exc: Exception) -> tuple[bool, str]:
        """Classify an exception as retryable or terminal.

        Returns:
            ``(is_retryable, error_type)`` where *error_type* is one of
            ``"rate_limit"``, ``"server_error"``, ``"timeout"``, or ``"other"``.
        """
        # Check for HTTP status code on the exception
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(status, int):
            if status == 429:
                return True, "rate_limit"
            if 500 <= status < 600:
                return True, "server_error"

        # Check exception class name and message for common transient patterns
        exc_str = f"{type(exc).__name__}: {exc}".lower()
        if "timeout" in exc_str or "timed out" in exc_str:
            return True, "timeout"
        if "connection" in exc_str or "connect" in exc_str:
            return True, "timeout"
        if "rate" in exc_str and "limit" in exc_str:
            return True, "rate_limit"
        if "server" in exc_str and "error" in exc_str:
            return True, "server_error"
        if "overloaded" in exc_str or "capacity" in exc_str:
            return True, "server_error"
        if "502" in exc_str or "503" in exc_str or "504" in exc_str:
            return True, "server_error"

        return False, "other"

    def _calculate_backoff(self, attempt: int, error_type: str) -> float:
        """Calculate backoff delay for the given attempt and error type.

        Uses retry config from ``concurrency.yaml`` (``_RETRY_CONFIG``).

        Args:
            attempt: Zero-based attempt number.
            error_type: One of ``"rate_limit"``, ``"server_error"``, ``"timeout"``, ``"other"``.

        Returns:
            Backoff delay in seconds.
        """
        backoff_base = _RETRY_CONFIG.get("backoff_base", 0.5)
        multipliers = _RETRY_CONFIG.get("backoff_multipliers", {})
        multiplier = multipliers.get(error_type, 2.0)

        jitter = random.uniform(JITTER_MIN, JITTER_MAX)
        return float(backoff_base * (multiplier ** attempt) + jitter)

    def _invoke_with_retry(
        self,
        structured_model: Any,
        messages: list[Any],
        invoke_kwargs: dict[str, Any],
        context_label: str,
    ) -> Any:
        """Invoke model with application-level retry and per-attempt token tracking.

        SDK-level retries are disabled (``max_retries=0``), so every HTTP attempt
        passes through this method, enabling token extraction from both successful
        and failed attempts.

        Args:
            structured_model: Chat model (possibly with structured output).
            messages: List of LangChain messages.
            invoke_kwargs: Additional kwargs for ``invoke()``.
            context_label: Description for log messages (e.g. "Transcription for page_001.png").

        Returns:
            The model response on success.

        Raises:
            Exception: Re-raises the last exception after exhausting retries or on
                non-retryable errors.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_rate_limit()
                response = structured_model.invoke(messages, **invoke_kwargs)
                return response
            except Exception as e:
                last_exception = e
                retryable, error_type = self._classify_error(e)

                # Track tokens from this failed attempt (when available)
                self._extract_tokens_from_exception(
                    e, f"{context_label} (attempt {attempt + 1}/{self.max_retries + 1})"
                )

                # Report to rate limiter
                self._report_error(error_type in ("rate_limit", "server_error"))

                if not retryable or attempt >= self.max_retries:
                    raise

                backoff = self._calculate_backoff(attempt, error_type)
                logger.warning(
                    f"Retryable {error_type} on attempt {attempt + 1}/{self.max_retries + 1} "
                    f"for {context_label}: {type(e).__name__}. "
                    f"Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)

        raise last_exception  # type: ignore[misc]  # unreachable; satisfies mypy

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about API usage."""
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

            schema_retries: dict[str, Any] = (
                concurrency_cfg.get("retry", {})
                .get("schema_retries", {})
                .get(api_type, {})
            )

            if schema_retries:
                logger.debug(
                    f"Loaded schema retry config for {api_type}: {list(schema_retries.keys())}"
                )
            else:
                logger.debug(f"No schema retry config found for {api_type}")

            return schema_retries
        except Exception as e:
            logger.warning(f"Error loading schema retry config for {api_type}: {e}")
            return {}

    def _should_retry_for_schema_flag(
        self, flag_name: str, flag_value: Any, current_attempt: int
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

        backoff_time = backoff_base * (backoff_multiplier**current_attempt) * jitter

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
                elif self.provider == "custom":
                    # Custom endpoints use Chat Completions API (max_tokens),
                    # not the Responses API (max_output_tokens).
                    invoke_kwargs["max_tokens"] = max_tokens
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

        # Anthropic-specific: extended thinking (Claude 4.5+, Opus models)
        elif (
            self.provider == "anthropic"
            and capabilities.get("extended_thinking", False)
        ):
            if "reasoning" in self.model_config:
                reasoning_cfg = self.model_config["reasoning"]
                if isinstance(reasoning_cfg, dict) and "effort" in reasoning_cfg:
                    effort = reasoning_cfg["effort"]
                    budget = _ANTHROPIC_EFFORT_TO_BUDGET.get(effort)
                    if budget is not None and budget > 0:
                        invoke_kwargs["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": budget,
                        }
                        logger.debug(
                            f"Added Anthropic extended thinking for "
                            f"{self.model_name}: budget_tokens={budget}"
                        )

        # Google-specific: thinking mode (Gemini 2.5+, 3.x)
        elif self.provider == "google" and capabilities.get("thinking", False):
            if "reasoning" in self.model_config:
                reasoning_cfg = self.model_config["reasoning"]
                if isinstance(reasoning_cfg, dict) and "effort" in reasoning_cfg:
                    effort = reasoning_cfg["effort"]
                    budget = _GOOGLE_EFFORT_TO_BUDGET.get(effort)
                    if budget is not None and budget > 0:
                        invoke_kwargs["thinking_config"] = {
                            "thinking_budget": budget,
                        }
                        logger.debug(
                            f"Added Google thinking for "
                            f"{self.model_name}: thinking_budget={budget}"
                        )

        elif "reasoning" in self.model_config:
            logger.debug(
                f"Skipping reasoning params for {self.model_name} "
                f"(not supported)"
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
                        logger.debug(
                            f"Added text verbosity params for {self.model_name}"
                        )
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
