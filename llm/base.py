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

4. **Rate Limiting Integration**: Works with RateLimiter to throttle requests and
   prevent API quota exhaustion (complementary to application-level retry).

5. **Configuration Loading**: Dynamically loads model parameters from YAML configuration
   files.

6. **Statistics Tracking**: Monitors request success rates, processing times, and error
   patterns.
"""

from __future__ import annotations

import json as _json
import random
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from config.accessors import get_api_timeout, get_service_tier
from config.loader import get_config_loader
from config.logger import setup_logger
from llm.client import (
    LLMConfig,
    ProviderType,
    get_chat_model,
    get_model_capabilities,
)
from llm.token_tracker import get_token_tracker

if TYPE_CHECKING:
    from llm.rate_limit import RateLimiter

logger = setup_logger(__name__)


def _load_retry_config() -> dict[str, Any]:
    """Load retry and backoff configuration from concurrency.yaml."""
    try:
        config_loader = get_config_loader()
        retry_cfg: dict[str, Any] = config_loader.get_concurrency_config().get(
            "retry", {}
        )

        if not retry_cfg:
            logger.debug("No retry config found in concurrency.yaml, using defaults")

        return retry_cfg
    except Exception as e:
        logger.warning(f"Error loading retry config: {e}. Using defaults.")
        return {}


# Load retry configuration from concurrency.yaml
_RETRY_CONFIG = _load_retry_config()


def _cfg_int(value: Any, default: int) -> int:
    """Coerce a config value to ``int``, returning *default* on malformed input.

    A hand-edited concurrency.yaml can carry a non-numeric value (e.g.
    ``max_attempts: fast``); guarding the conversion keeps such a value from
    crashing at import time.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _cfg_float(value: Any, default: float) -> float:
    """Coerce a config value to ``float``, returning *default* on malformed
    input."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# Constants for retry logic (loaded from config with fallback defaults).
# retry.max_attempts is the TOTAL number of attempts (initial call included),
# so the retry count is one less. Default reduced from 15 to 8: with honored
# Retry-After and a 120 s backoff cap, 8 transient attempts already spans
# several minutes of retrying.
DEFAULT_MAX_RETRIES = max(0, _cfg_int(_RETRY_CONFIG.get("max_attempts", 8), 8) - 1)

# Hard ceiling on any single backoff wait (seconds), including a server-provided
# Retry-After. Keeps a hostile Retry-After header from stalling a worker for an
# unbounded time.
BACKOFF_CAP_S = _cfg_float(_RETRY_CONFIG.get("backoff_cap", 120), 120.0)

# Time-based retry horizon (seconds) bounding the TOTAL time spent retrying one
# request from its first attempt. Precedence: a retryable error is retried while
# EITHER attempts remain (attempt < max_retries) OR the window is still open
# (elapsed < MAX_ELAPSED_S); it stops only once BOTH are exhausted. A value of 0
# disables the window, restoring legacy attempts-only behavior (governed solely
# by DEFAULT_MAX_RETRIES). Recommended ~900 for OpenAI flex-tier queuing.
MAX_ELAPSED_S = _cfg_float(_RETRY_CONFIG.get("max_elapsed", 0), 0.0)

# A ``jitter:`` key present but null yields None here; fall back to an empty
# dict so the ``.get`` calls below cannot raise AttributeError at import time.
_JITTER = _RETRY_CONFIG.get("jitter", {})
if not isinstance(_JITTER, dict):
    _JITTER = {}
JITTER_MIN = _cfg_float(_JITTER.get("min", 0.5), 0.5)
JITTER_MAX = _cfg_float(_JITTER.get("max", 1.0), 1.0)

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


def _sanitize_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip union types (e.g. ["string", "null"]) to the primary non-null
    type.  Anthropic's ``with_structured_output(method="json_schema")``
    does not support JSON Schema union types and raises
    ``AssertionError: Expected code to be unreachable``.
    """
    import copy

    schema = copy.deepcopy(schema)

    def _walk(node: dict[str, Any]) -> dict[str, Any]:
        if "type" in node and isinstance(node["type"], list):
            non_null = [t for t in node["type"] if t != "null"]
            node["type"] = non_null[0] if non_null else "string"
        for _key, value in node.items():
            if isinstance(value, dict):
                _walk(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _walk(item)
        return node

    return _walk(schema)


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
        logger.warning("Empty content extracted from LangChain AIMessage response.")
    return result


def _coerce_token_count(value: Any) -> int | None:
    """Coerce a provider-reported token count to an ``int``.

    Accepts ``int`` and ``float`` (some providers report counts as floats such
    as ``123.0``); rejects ``bool`` and non-numeric values. Floats are rounded
    to the nearest integer so a float count is committed to the budget rather
    than silently dropped.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    return None


def _extract_from_output_attribute(data: Any) -> str | None:
    """Try SDK convenience ``output_text`` attribute."""
    try:
        text_attr = getattr(data, "output_text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            return text_attr.strip()
    except Exception as e:
        # Defensive: attribute access on exotic SDK objects may raise; fall
        # through to the next extractor but keep the cause observable.
        logger.debug(f"output_text attribute extraction failed: {e}")
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
            conv = getattr(data, "to_dict", None) or getattr(data, "model_dump", None)
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
            output_type = "list" if isinstance(output, list) else type(output).__name__
            output_items = len(output) if isinstance(output, list) else 0
            logger.warning(
                "Empty content extracted from API response. "
                f"Response structure: output={output_type}, "
                f"items={output_items}"
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

    # Prefer the parsed value (already schema-validated by LangChain)
    parsed = data.get("parsed")
    if isinstance(parsed, dict):
        try:
            return _json.dumps(parsed, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    elif parsed is not None:
        # A pydantic model instance rather than a dict: serialize via
        # model_dump() so a structured answer is not dropped.
        model_dump = getattr(parsed, "model_dump", None)
        if callable(model_dump):
            try:
                return _json.dumps(model_dump(), ensure_ascii=False)
            except (TypeError, ValueError):
                pass

    # Fallback: extract text from the raw AIMessage
    raw = data.get("raw")
    if raw is not None:
        text = _extract_from_aimessage(raw)
        if text:
            return text
        # OpenRouter tool-mode and Anthropic json_schema can return an
        # AIMessage with EMPTY content while the model's JSON lives in the
        # first tool call's args (parsed is None with a parsing_error).
        # Serialize those args so a complete answer is not reported as empty.
        tool_calls = getattr(raw, "tool_calls", None)
        if tool_calls:
            first = tool_calls[0]
            args = first.get("args") if isinstance(first, dict) else None
            if args is not None:
                try:
                    return _json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    pass
        return text

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

    Subclasses should implement specific API endpoint logic
    (transcription, summarization, etc.)
    """

    model_name: str
    provider: ProviderType | None
    timeout: int
    rate_limiter: RateLimiter | None
    max_retries: int
    # Time-based retry horizon (seconds); see MAX_ELAPSED_S. Class-level default
    # of 0.0 (disabled) so instances built via ``__new__`` (e.g. in unit tests,
    # bypassing ``__init__``) fall back to legacy attempts-only behavior; the
    # real horizon is assigned from MAX_ELAPSED_S in ``__init__``.
    max_elapsed: float = 0.0
    chat_model: BaseChatModel
    successful_requests: int
    failed_requests: int
    processing_times: deque[float]
    model_config: dict[str, Any]
    service_tier: str
    schema_retry_config: dict[str, Any]
    _output_schema: dict[str, Any] | None
    custom_capabilities: Any
    _stats_lock: threading.Lock

    def __init__(
        self,
        model_name: str,
        provider: ProviderType | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        rate_limiter: RateLimiter | None = None,
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
            max_retries: Max retry attempts. LangChain handles exponential backoff
                automatically.
            service_tier: OpenAI service tier ("flex", "default", "auto"). OpenAI-only.
            section_hint: YAML config section for custom endpoint disambiguation.
        """
        self.model_name = model_name
        self.provider = provider
        self.timeout = timeout if timeout is not None else get_api_timeout()
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        # Time-based retry horizon from concurrency.yaml (0 = disabled/legacy).
        self.max_elapsed = MAX_ELAPSED_S

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

        # Resolve and stash the NAME of the env var that serves this client's
        # key (never the key value) so token usage can be stamped per key-pool
        # bucket. Best-effort: a client built with a literal api_key or a custom
        # endpoint lacking an api_key_env_var simply stamps as unattributed.
        self.key_env: str | None = None
        if self.provider is not None:
            try:
                from llm.client import resolve_key_env

                self.key_env = resolve_key_env(self.provider, section_hint)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Could not resolve key env var for token stamping: %s", exc
                )

        # Instantiate LangChain chat model (no SDK-level retry)
        self.chat_model: BaseChatModel = get_chat_model(llm_config)

        # Statistics tracking. Managers are shared across ThreadPoolExecutor
        # workers, so the request counters are guarded by a lock to keep their
        # read-modify-writes atomic.
        self._stats_lock = threading.Lock()
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
        with self._stats_lock:
            self.successful_requests += 1
        if self.rate_limiter is not None:
            self.rate_limiter.report_success()

    def _report_failure(self) -> None:
        """Increment the failed-request counter (thread-safe)."""
        with self._stats_lock:
            self.failed_requests += 1

    def _record_processing_time(self, elapsed: float) -> None:
        """Append a per-attempt duration to the stats deque (thread-safe).

        Guarded by the same lock get_stats() snapshots under, so concurrent
        worker appends never race a reader iterating the deque.
        """
        with self._stats_lock:
            self.processing_times.append(elapsed)

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

    def _get_structured_chat_model(
        self, invoke_kwargs: dict[str, Any] | None = None
    ) -> Any:
        """Get chat model with structured output for each provider.

        Provider-specific approaches:
        - OpenAI: Native text.format parameter (Responses API) - handled
          in _apply_structured_output_kwargs, guarded by capability flag
        - Custom: Same as OpenAI when supports_structured_output=True (Mode A),
          otherwise bare model with no structured output (Modes B/C)
        - Anthropic: Native json_schema constrained decoding via
          with_structured_output(method="json_schema"), guarded by capability flag
        - Google: response_mime_type + response_schema in invoke kwargs
        - OpenRouter: Tool-based structured output (OpenAI-compatible)

        ``with_structured_output(include_raw=True)`` returns a
        ``RunnableMap(raw=llm) | parser`` that never forwards call-time kwargs
        to the inner model, so resolved model parameters (max_tokens,
        temperature, extended-thinking config) are bound onto the model here
        before wrapping. Bare-model providers (OpenAI/Custom/Google) still
        receive those kwargs at invoke time, so they are not bound here.

        Args:
            invoke_kwargs: Resolved invocation kwargs to bind onto the model on
                the structured-output paths (Anthropic/OpenRouter). ``None`` or
                empty leaves the model unbound.

        Returns:
            Chat model with structured output, or base chat model.
        """

        def _bind(model: Any) -> Any:
            return model.bind(**invoke_kwargs) if invoke_kwargs else model

        if self.provider == "openai":
            return self.chat_model

        # Custom: structured output via response_format kwargs (Mode A) or
        # none at all (Modes B/C).  Never use with_structured_output since
        # most custom endpoints do not support tool-based function calling.
        if self.provider == "custom":
            return self.chat_model

        if self.provider == "anthropic":
            capabilities = get_model_capabilities(self.model_name)
            if capabilities.get("structured_output", False) and self._output_schema:
                schema = self._output_schema.get("schema", self._output_schema)
                if isinstance(schema, dict) and schema:
                    schema = _sanitize_schema_for_anthropic(schema)
                    if "title" not in schema:
                        schema["title"] = self._output_schema.get(
                            "name", "structured_output"
                        )
                    return _bind(self.chat_model).with_structured_output(
                        schema,
                        method="json_schema",
                        include_raw=True,
                    )
            return self.chat_model

        if self.provider == "google":
            return self.chat_model

        # OpenRouter: Use tool-based structured output (OpenAI-compatible).
        # LangChain's function-calling path requires a top-level "title" on
        # the JSON schema to use as the function name; without it every call
        # fails with "Unsupported function". Inject the schema's declared
        # name (never mutating the shared schema dict) when it is missing.
        if self._output_schema:
            schema = self._output_schema.get("schema", self._output_schema)
            if isinstance(schema, dict) and schema:
                if "title" not in schema:
                    schema = {
                        **schema,
                        "title": self._output_schema.get("name", "structured_output"),
                    }
                return _bind(self.chat_model).with_structured_output(
                    schema,
                    include_raw=True,
                )

        return self.chat_model

    def _apply_structured_output_kwargs(self, invoke_kwargs: dict[str, Any]) -> None:
        """Apply provider-specific structured output parameters to invoke kwargs.

        Modifies invoke_kwargs in-place to add structured output format for
        OpenAI (text.format via Responses API), Custom Mode A (response_format
        via Chat Completions API), and Google (response_mime_type +
        response_schema). Guarded by supports_structured_output capability.

        Args:
            invoke_kwargs: The invocation kwargs dict to modify.
        """
        if self.provider == "openai":
            capabilities = get_model_capabilities(self.model_name)
            if capabilities.get("structured_output", False):
                text_format = self._build_text_format()
                if text_format:
                    if "text" not in invoke_kwargs:
                        invoke_kwargs["text"] = {}
                    invoke_kwargs["text"]["format"] = text_format

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
                            k: v for k, v in text_format.items() if k != "type"
                        },
                    }
                    invoke_kwargs["response_format"] = chat_completions_format

        if self.provider == "google":
            if "thinking_config" in invoke_kwargs:
                logger.debug(
                    f"Google thinking active for {self.model_name}: "
                    "suppressing response_schema to avoid API conflict"
                )
            else:
                schema_obj = (
                    self._output_schema.get("schema")
                    if isinstance(self._output_schema, dict)
                    and "schema" in self._output_schema
                    else self._output_schema
                )
                if isinstance(schema_obj, dict) and schema_obj:
                    invoke_kwargs.setdefault("response_mime_type", "application/json")
                    invoke_kwargs.setdefault("response_schema", schema_obj)

    @staticmethod
    def _usage_metadata_folds_cache(usage_meta: dict[str, Any]) -> bool:
        """Whether a LangChain usage_metadata already folds cache into the total.

        LangChain normalizes prompt-cache tokens into ``input_tokens`` and
        exposes the breakdown under ``input_token_details`` (keys ``cache_read``
        / ``cache_creation``). OpenAI's cached-prompt tokens surface the same
        way (a subset of the prompt count). When that nested detail dict is
        present the cache is already inside ``total_tokens``, so it must not be
        added again.
        """
        return isinstance(usage_meta.get("input_token_details"), dict)

    @staticmethod
    def _additive_cache_tokens(usage_meta: dict[str, Any], target: Any) -> int:
        """Cache tokens reported SEPARATELY from a cache-exclusive input.

        Raw-Anthropic usage reports ``cache_read_input_tokens`` /
        ``cache_creation_input_tokens`` alongside an ``input_tokens`` that
        EXCLUDES them, so ``input + output`` is short by the cache total. This
        reads those fields from the usage_metadata dict and, defensively, from
        the raw provider usage on the unwrapped response
        (``response_metadata['usage']`` or a ``usage`` attribute). Only invoked
        when the usage_metadata does not already fold cache in, so no double
        counting occurs. Returns the full-weight cache total to add.
        """
        candidates: list[dict[str, Any]] = [usage_meta]
        response_metadata = getattr(target, "response_metadata", None)
        if isinstance(response_metadata, dict):
            raw_usage = response_metadata.get("usage")
            if isinstance(raw_usage, dict):
                candidates.append(raw_usage)
        raw_usage_attr = getattr(target, "usage", None)
        if isinstance(raw_usage_attr, dict):
            candidates.append(raw_usage_attr)

        cache_read = 0
        cache_creation = 0
        for usage in candidates:
            if not isinstance(usage, dict):
                continue
            cache_read = cache_read or int(usage.get("cache_read_input_tokens", 0) or 0)
            cache_creation = cache_creation or int(
                usage.get("cache_creation_input_tokens", 0) or 0
            )
        return cache_read + cache_creation

    def _report_token_usage(self, response: Any, context_label: str) -> None:
        """Report token usage from a LangChain response to the token tracker.

        Cache tokens count at FULL weight in the daily budget. LangChain-
        normalized and OpenAI cached shapes fold cache into ``total_tokens``
        already (committed as-is); raw-Anthropic shapes report cache separately
        from a cache-exclusive input, so those are added on top. Never double
        counts.

        Args:
            response: LangChain model response with usage_metadata.
            context_label: Description for the log message
                (e.g. "Transcription for page_001.png").
        """
        try:
            target = response
            if isinstance(response, dict) and "raw" in response:
                target = response["raw"]
            usage_meta = getattr(target, "usage_metadata", None)
            if not usage_meta or not isinstance(usage_meta, dict):
                logger.warning(
                    f"[TOKEN] {context_label}: usage_metadata missing or not a dict"
                )
                return

            # Accept int and float counts (some providers report e.g. 123.0);
            # bool is excluded and floats are rounded to int.
            total_tokens = _coerce_token_count(usage_meta.get("total_tokens"))

            # Fallback: compute from input_tokens + output_tokens (Anthropic-style)
            if not total_tokens:
                input_tokens = _coerce_token_count(usage_meta.get("input_tokens", 0))
                output_tokens = _coerce_token_count(usage_meta.get("output_tokens", 0))
                if input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens

            # Add raw-Anthropic-shape cache tokens at full weight ONLY when the
            # usage_metadata does not already fold them into the total.
            cache_add = 0
            if not self._usage_metadata_folds_cache(usage_meta):
                cache_add = self._additive_cache_tokens(usage_meta, target)

            if total_tokens or cache_add:
                committed = (total_tokens or 0) + cache_add
                token_tracker = get_token_tracker()
                token_tracker.add_tokens(
                    committed,
                    provider=getattr(self, "provider", None),
                    key_env=getattr(self, "key_env", None),
                    model=getattr(self, "model_name", None),
                )
                logger.debug(
                    f"[TOKEN] {context_label}: added {committed} tokens "
                    f"(cache +{cache_add}; "
                    f"total now: {token_tracker.get_tokens_used_today():,})"
                )
            else:
                logger.warning(
                    f"[TOKEN] {context_label}: no usable token counts in usage_metadata"
                )
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Error reporting token usage for {context_label}: {e}")

    def _extract_tokens_from_exception(
        self, exc: Exception, context_label: str
    ) -> None:
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
                logger.debug(f"[TOKEN] {context_label}: no usage data in exception")
                return

            # Try total_tokens, then prompt_tokens+completion_tokens (OpenAI),
            # then input_tokens+output_tokens (Anthropic). Route each field
            # through _coerce_token_count so float counts (e.g. 123.0) are
            # accepted the same way the success path accepts them, rather than
            # dropped by an int-only isinstance check.
            total = _coerce_token_count(usage.get("total_tokens"))
            if not total or total <= 0:
                prompt = _coerce_token_count(usage.get("prompt_tokens"))
                completion = _coerce_token_count(usage.get("completion_tokens"))
                if (
                    prompt is not None
                    and completion is not None
                    and (prompt + completion) > 0
                ):
                    total = prompt + completion
                else:
                    inp = _coerce_token_count(usage.get("input_tokens"))
                    out = _coerce_token_count(usage.get("output_tokens"))
                    if inp is not None and out is not None and (inp + out) > 0:
                        total = inp + out

            if total is not None and total > 0:
                token_tracker = get_token_tracker()
                token_tracker.add_tokens(
                    total,
                    provider=getattr(self, "provider", None),
                    key_env=getattr(self, "key_env", None),
                    model=getattr(self, "model_name", None),
                )
                logger.info(
                    f"[TOKEN] {context_label}: recovered {total} tokens"
                    " from failed request "
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
        # Check for an HTTP status code on the exception. Providers expose it
        # under different attribute names: OpenAI/Anthropic use ``status_code``;
        # google.genai ``APIError`` carries the numeric HTTP status on ``code``
        # (an int) while its ``status`` holds a non-numeric label such as
        # "RESOURCE_EXHAUSTED". Only ints count as an HTTP status, so a string
        # ``status`` never surfaces where an int is expected (and a 429/500 from
        # Google is no longer misclassified as terminal).
        for status in (
            getattr(exc, "status_code", None),
            getattr(exc, "code", None),
            getattr(exc, "status", None),
        ):
            if isinstance(status, bool) or not isinstance(status, int):
                continue
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
        # Google quota exhaustion surfaces as a "RESOURCE_EXHAUSTED" status or a
        # "quota"/"resource has been exhausted" message; treat as retryable.
        if (
            "quota" in exc_str
            or "resource_exhausted" in exc_str
            or "resource has been exhausted" in exc_str
        ):
            return True, "rate_limit"
        if "server" in exc_str and "error" in exc_str:
            return True, "server_error"
        if "internal error" in exc_str:
            return True, "server_error"
        if "overloaded" in exc_str or "capacity" in exc_str:
            return True, "server_error"
        if "502" in exc_str or "503" in exc_str or "504" in exc_str:
            return True, "server_error"

        return False, "other"

    def _calculate_backoff(self, attempt: int, error_type: str) -> float:
        """Calculate backoff delay for the given attempt and error type.

        Uses retry config from ``concurrency.yaml`` (``_RETRY_CONFIG``). The
        result is capped at ``BACKOFF_CAP_S`` so a high attempt count cannot
        produce an unbounded wait. The exponent is additionally clamped so a
        very large ``attempt`` (possible under time-based retry windows) cannot
        raise ``OverflowError`` before the cap is applied.

        Args:
            attempt: Zero-based attempt number.
            error_type: One of ``"rate_limit"``, ``"server_error"``, ``"timeout"``,
                ``"other"``.

        Returns:
            Backoff delay in seconds (<= ``BACKOFF_CAP_S``).
        """
        backoff_base = _cfg_float(_RETRY_CONFIG.get("backoff_base", 0.5), 0.5)
        multipliers = _RETRY_CONFIG.get("backoff_multipliers", {})
        if not isinstance(multipliers, dict):
            multipliers = {}
        multiplier = _cfg_float(multipliers.get(error_type, 2.0), 2.0)

        jitter = random.uniform(JITTER_MIN, JITTER_MAX)
        return min(
            BACKOFF_CAP_S,
            float(backoff_base * (multiplier ** min(attempt, 64)) + jitter),
        )

    @staticmethod
    def _parse_retry_after(exc: BaseException | None) -> float | None:
        """Extract a Retry-After delay (seconds) from an exception's headers.

        Reads the ``Retry-After`` header off the exception's response (or a
        top-level ``headers`` attribute) across the OpenAI/Anthropic SDK
        exception shapes, tolerating both the integer/float seconds form and the
        HTTP-date form. Returns ``None`` when no usable value is present.
        Defensive: any parsing problem yields ``None`` rather than raising.
        """
        if exc is None:
            return None
        try:
            headers = None
            resp = getattr(exc, "response", None)
            if resp is not None:
                headers = getattr(resp, "headers", None)
            if headers is None:
                headers = getattr(exc, "headers", None)
            if headers is None:
                return None

            getter = getattr(headers, "get", None)
            if not callable(getter):
                return None
            raw = getter("retry-after")
            if raw is None:
                raw = getter("Retry-After")
            if raw is None:
                return None

            value = str(raw).strip()
            if not value:
                return None

            # Seconds form (integer or float).
            try:
                return max(0.0, float(value))
            except ValueError:
                pass

            # HTTP-date form (e.g. "Wed, 21 Oct 2026 07:28:00 GMT").
            from email.utils import parsedate_to_datetime

            try:
                target = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if target is None:
                return None
            import datetime as _dt

            now = (
                _dt.datetime.now(target.tzinfo) if target.tzinfo else _dt.datetime.now()
            )
            return max(0.0, (target - now).total_seconds())
        except Exception:
            return None

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
            context_label: Description for log messages
                (e.g. "Transcription for page_001.png").

        Returns:
            The model response on success.

        Raises:
            Exception: Re-raises the last exception after exhausting retries or on
                non-retryable errors.

        Retry horizon (precedence): a retryable error is retried while EITHER
        attempts remain (``attempt < self.max_retries``) OR the time window is
        still open (``elapsed < self.max_elapsed``); it stops only once BOTH are
        exhausted. ``max_attempts`` is thus a legacy floor, not a ceiling, when a
        window is configured. ``self.max_elapsed == 0`` disables the window and
        restores attempts-only behavior. Each sleep is capped at ``BACKOFF_CAP_S``
        and additionally clamped so it never overshoots the remaining window by
        more than one ``BACKOFF_CAP_S``.
        """
        first_attempt_start = time.monotonic()
        max_elapsed = self.max_elapsed
        attempt = 0

        # Loop exits only via ``return`` (success) or ``raise`` (terminal), so
        # there is no fall-through path after the ``while``.
        while True:
            try:
                self._wait_for_rate_limit()
                response = structured_model.invoke(messages, **invoke_kwargs)
                return response
            except Exception as e:
                retryable, error_type = self._classify_error(e)

                # Track tokens from this failed attempt (when available)
                self._extract_tokens_from_exception(
                    e, f"{context_label} (attempt {attempt + 1})"
                )

                # A server-provided Retry-After is itself a rate-limit signal.
                # Feed it into the shared limiter as a rate-limit error even
                # when the exception was not otherwise classified as one (a
                # single report per failure — never double-counted).
                retry_after = self._parse_retry_after(e)
                is_rate_signal = (
                    error_type in ("rate_limit", "server_error")
                    or retry_after is not None
                )
                self._report_error(is_rate_signal)

                # Continue while attempts remain OR the time window is still open.
                elapsed = time.monotonic() - first_attempt_start
                attempts_left = attempt < self.max_retries
                window_open = max_elapsed > 0 and elapsed < max_elapsed
                if not retryable or not (attempts_left or window_open):
                    raise

                # Honor Retry-After: wait at least as long as the server asks,
                # never below the computed backoff and never above the cap.
                backoff = self._calculate_backoff(attempt, error_type)
                if retry_after is not None:
                    backoff = min(BACKOFF_CAP_S, max(backoff, retry_after))
                # Never overshoot the remaining window by more than one
                # backoff_cap (no-op once attempts-only or window disabled).
                if max_elapsed > 0:
                    remaining = max(0.0, max_elapsed - elapsed)
                    backoff = min(backoff, remaining + BACKOFF_CAP_S)
                logger.warning(
                    f"Retryable {error_type} on attempt {attempt + 1} "
                    f"for {context_label}: {type(e).__name__}. "
                    f"Retrying in {backoff:.1f}s (elapsed {elapsed:.1f}s"
                    f"{f'/{max_elapsed:.0f}s window' if max_elapsed > 0 else ''})"
                    f"{' (Retry-After honored)' if retry_after is not None else ''}..."
                )
                time.sleep(backoff)
                attempt += 1

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about API usage."""
        # Snapshot the deque under the lock: worker threads append to (and evict
        # from) processing_times concurrently, so iterating it lazily via
        # statistics.mean() outside the lock can raise "deque mutated during
        # iteration".
        with self._stats_lock:
            times = list(self.processing_times)
            successful = self.successful_requests
            failed = self.failed_requests
        avg_time = statistics.mean(times) if times else 0
        total_requests = successful + failed
        success_rate = (successful / max(1, total_requests)) * 100

        return {
            "provider": self.provider,
            "model": self.model_name,
            "successful_requests": successful,
            "failed_requests": failed,
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
                    f"Loaded schema retry config for {api_type}: "
                    f"{list(schema_retries.keys())}"
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

        # Get max attempts for this flag. Route through _cfg_int so a
        # hand-edited concurrency.yaml carrying a string (e.g. max_attempts:
        # "three") falls back to the default rather than raising a TypeError
        # in the comparison below.
        max_attempts = _cfg_int(flag_config.get("max_attempts", 0), 0)

        # Check if we've exceeded max attempts
        if current_attempt >= max_attempts:
            return False, 0.0, max_attempts

        # Calculate backoff time. Additive jitter (base * multiplier^attempt +
        # jitter), matching the documented formula in concurrency.example.yaml
        # and the sibling _calculate_backoff; a multiplicative jitter in
        # [0.5, 1.0] would instead halve the intended wait. The base and
        # multiplier are coerced through _cfg_float so a malformed config value
        # cannot raise inside the arithmetic below.
        backoff_base = _cfg_float(flag_config.get("backoff_base", 2.0), 2.0)
        backoff_multiplier = _cfg_float(flag_config.get("backoff_multiplier", 1.5), 1.5)
        jitter = random.uniform(JITTER_MIN, JITTER_MAX)

        backoff_time = backoff_base * (backoff_multiplier**current_attempt) + jitter

        return True, backoff_time, max_attempts

    def _build_invoke_kwargs(self) -> dict[str, Any]:
        """Build provider-appropriate invocation kwargs with capability guarding."""
        invoke_kwargs: dict[str, Any] = {}

        # Get model capabilities for parameter guarding
        capabilities = get_model_capabilities(self.model_name)

        # Add max_output_tokens from config (most models support this)
        if capabilities.get("max_tokens", True):
            max_tokens = self.model_config.get("max_output_tokens")
            if not max_tokens and self.provider == "anthropic":
                # langchain-anthropic silently defaults to 1024 output tokens
                # when max_tokens is unset, truncating long transcriptions and
                # summaries. Fall back to the capability registry's default.
                from llm.capabilities import detect_capabilities

                max_tokens = detect_capabilities(self.model_name).max_output_tokens
                logger.debug(
                    "Anthropic max_tokens omitted; defaulting to registry value "
                    f"{max_tokens} for {self.model_name}"
                )
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
        elif self.provider == "anthropic" and capabilities.get(
            "extended_thinking", False
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
                        logger.debug(
                            f"Added text verbosity params for {self.model_name}"
                        )
        elif "text" in self.model_config:
            logger.debug(
                f"Skipping text verbosity params for {self.model_name} (not supported)"
            )

        # Temperature (capability-guarded). Skipped for reasoning models
        # (whose capability flag is False) and whenever reasoning/extended
        # thinking is active, since those paths reject an explicit temperature.
        temperature = self.model_config.get("temperature")
        reasoning_active = any(
            key in invoke_kwargs for key in ("reasoning", "thinking", "thinking_config")
        )
        if (
            temperature is not None
            and capabilities.get("temperature", False)
            and not reasoning_active
        ):
            invoke_kwargs["temperature"] = float(temperature)
            logger.debug(f"Added temperature={temperature} for {self.model_name}")
        elif temperature is not None:
            logger.debug(
                f"Skipping temperature for {self.model_name} "
                "(unsupported or reasoning active)"
            )

        return invoke_kwargs

    def close(self) -> None:
        """No-op teardown, retained for API compatibility.

        Managers are constructed per item, so an earlier version closed the
        chat model's SDK/httpx clients here to avoid leaking connection pools.
        That was actively harmful: every langchain provider we use hands its
        ChatModel an httpx client drawn from a process-wide ``@lru_cache``
        keyed on (base_url, timeout, socket_options) -- see langchain_openai
        and langchain_anthropic ``_client_utils`` (``_cached_sync_httpx_client``
        / ``_get_default_httpx_client``). Every manager built for the same
        provider therefore SHARES one httpx client. Closing it at the end of
        item 1 poisoned that shared pool, so item 2+ failed instantly with
        ``APIConnectionError`` ("Connection error."). langchain_google_genai
        holds a per-instance ``google.genai`` client with no such cache, but
        it too needs no per-item teardown: the pool is bounded and dies with
        the process. Hence closing is both unnecessary and dangerous, and this
        method now only logs at debug level.
        """
        logger.debug("close() is a no-op (provider httpx clients are shared)")


# ============================================================================
# Module exports
# ============================================================================
# ``LLMClientBase`` is a package-private base class; prefer the concrete
# ``TranscriptionManager`` / ``SummaryManager`` subclasses exposed via
# ``llm/__init__.py``. It remains importable via ``llm.base.LLMClientBase``
# for testing.
__all__ = [
    "DEFAULT_MAX_RETRIES",
]
