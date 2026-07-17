"""Multi-provider transcription API client using LangChain with structured outputs.

This module provides image transcription using LangChain's unified interface,
supporting multiple LLM providers:
- OpenAI (GPT-5, GPT-4o, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- OpenRouter (access to multiple providers)

The client consumes preprocessed in-memory page payloads
(``imaging.payload.PagePayload``) and handles:
- Structured JSON output parsing
- Dual-level retry logic (API errors + schema validation)
- Rate limiting integration
- Token usage tracking
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.loader import PROMPTS_DIR, SCHEMAS_DIR
from config.logger import setup_logger
from imaging.payload import PagePayload
from llm.base import LLMClientBase
from llm.capabilities import detect_capabilities
from llm.client import ProviderType, get_model_capabilities
from llm.prompts import render_prompt_with_schema, strip_markdown_code_block
from llm.rate_limit import RateLimiter, get_shared_rate_limiter
from llm.types import CustomEndpointCapabilities

logger = setup_logger(__name__)

# Constants
TRANSCRIPTION_SCHEMA_FILE = "transcription_schema.json"
SYSTEM_PROMPT_FILE = "transcription_system_prompt.txt"
PLAIN_TEXT_PROMPT_FILE = "transcription_plain_text_prompt.txt"

# Required top-level keys in the transcription JSON schema response
TRANSCRIPTION_REQUIRED_KEYS = frozenset(
    {
        "image_analysis",
        "transcription",
        "no_transcribable_text",
        "transcription_not_possible",
    }
)


class TranscriptionManager(LLMClientBase):
    """
    Transcribes images using LangChain with structured outputs.

    This class handles image transcription using various LLM providers through
    LangChain's unified interface. It consumes ``PagePayload`` objects that
    were preprocessed and base64-encoded upstream by ``imaging.payload``.

    Supports:
    - OpenAI with Responses API structured outputs
    - Anthropic Claude with tool-based structured outputs
    - Google Gemini with JSON mode
    - OpenRouter with provider-appropriate handling
    """

    def __init__(
        self,
        model_name: str,
        provider: ProviderType | None = None,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: int | None = None,
        custom_capabilities: CustomEndpointCapabilities | None = None,
    ) -> None:
        """
        Initialize the transcription manager.

        Args:
            model_name: Model to use (e.g., 'gpt-5-mini', 'claude-sonnet-4-5-20250929').
            provider: Provider name (openai, anthropic, google, openrouter, custom).
            api_key: Optional API key. If None, uses environment variable.
            rate_limiter: Optional injected RateLimiter. When omitted, one is
                constructed from ``config.accessors.get_rate_limits()``.
            timeout: Request timeout in seconds.
            custom_capabilities: Declared capabilities for custom endpoints.

        Raises:
            ValueError: If the selected model doesn't support multimodal (image) input.
        """
        if rate_limiter is None:
            rate_limiter = get_shared_rate_limiter(provider)
        super().__init__(
            model_name,
            provider,
            api_key,
            timeout,
            rate_limiter,
            section_hint="transcription_model",
        )

        # Store custom endpoint capabilities (used for routing decisions)
        self.custom_capabilities = custom_capabilities

        # Check multimodal capability (required for image transcription)
        capabilities = get_model_capabilities(model_name)
        has_multimodal = capabilities.get("multimodal", False)
        # Override with user-declared capability for custom endpoints
        if custom_capabilities is not None:
            has_multimodal = custom_capabilities.supports_vision
        if not has_multimodal:
            logger.warning(
                f"Model '{model_name}' may not support multimodal (image) input. "
                "Image transcription may fail. "
                "Consider using gpt-5, gpt-4o, claude, or gemini."
            )

        # Load schema and system prompt
        self.transcription_schema: dict[str, Any] | None = None
        self.system_prompt: str = ""

        self._load_schema_and_prompt()
        self._output_schema = self.transcription_schema

        # Load model configuration and determine service tier
        self.model_config = self._load_model_config("transcription_model")
        self.service_tier = self._determine_service_tier("transcription")

        # Load schema-specific retry configuration
        self.schema_retry_config = self._load_schema_retry_config("transcription")

    @property
    def is_plain_text_mode(self) -> bool:
        """Whether this manager operates in plain-text (non-JSON) mode."""
        return (
            self.custom_capabilities is not None
            and self.custom_capabilities.use_plain_text_prompt
        )

    def _load_schema_and_prompt(self) -> None:
        """Load transcription schema and system prompt from configuration files.

        In plain-text mode (Mode B): loads a simplified prompt with no JSON
        schema injection.  In standard mode (Modes A/C): loads the full schema
        and renders the prompt with schema injection.
        """
        try:
            if self.is_plain_text_mode:
                # Mode B: no schema, plain-text prompt
                self.transcription_schema = None
                prompt_file = PLAIN_TEXT_PROMPT_FILE
                logger.info(
                    "Plain-text mode: skipping transcription schema, "
                    f"using {PLAIN_TEXT_PROMPT_FILE}"
                )
            else:
                # Modes A/C: load schema as usual
                prompt_file = SYSTEM_PROMPT_FILE
                schema_path = (SCHEMAS_DIR / TRANSCRIPTION_SCHEMA_FILE).resolve()
                if schema_path.exists():
                    with open(schema_path, encoding="utf-8") as f:
                        self.transcription_schema = json.load(f)
                    logger.info(
                        f"Loaded transcription schema from "
                        f"{TRANSCRIPTION_SCHEMA_FILE} "
                        f"({len(json.dumps(self.transcription_schema))} bytes)"
                    )
                else:
                    logger.error(f"Transcription schema not found at {schema_path}")
                    raise FileNotFoundError(
                        f"Required schema file missing: {schema_path}"
                    )

            # Load prompt
            prompt_path = (PROMPTS_DIR / prompt_file).resolve()
            if prompt_path.exists():
                with open(prompt_path, encoding="utf-8") as f:
                    raw_prompt = f.read()

                if self.is_plain_text_mode:
                    # No schema injection for plain-text prompt
                    self.system_prompt = raw_prompt
                    logger.info(
                        f"Loaded plain-text transcription prompt "
                        f"({len(self.system_prompt)} chars)"
                    )
                elif self.transcription_schema is not None:
                    # Render prompt with schema injection
                    bare_schema = self.transcription_schema.get(
                        "schema", self.transcription_schema
                    )
                    self.system_prompt = render_prompt_with_schema(
                        raw_prompt, bare_schema
                    )
                    logger.info(
                        f"Loaded and rendered transcription system prompt "
                        f"({len(self.system_prompt)} chars)"
                    )
                else:
                    self.system_prompt = raw_prompt
                    logger.info(
                        "Loaded transcription system prompt without schema injection"
                    )
            else:
                logger.error(f"Transcription prompt not found at {prompt_path}")
                raise FileNotFoundError(f"Required prompt file missing: {prompt_path}")

        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading schema/prompt: {e}")
            raise

    @staticmethod
    def _format_image_name(image_name: str | None) -> str:
        """Format image name for display in failure messages.

        Args:
            image_name: Image filename (e.g., 'page_001.png', 'image_42.jpg').

        Returns:
            Formatted image name string (keeps original filename).
        """
        if not image_name:
            return "unknown_image"
        return image_name

    @staticmethod
    def _truncate_analysis(text: str | None, max_chars: int = 100) -> str:
        """Truncate image analysis text to a reasonable length.

        Args:
            text: Full image analysis text.
            max_chars: Maximum characters to include.

        Returns:
            Truncated text with ellipsis if needed.
        """
        if not text:
            return "no details available"
        text = text.strip()
        if len(text) <= max_chars:
            return text
        # Truncate at word boundary if possible
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.7:  # Only use word boundary if not too short
            truncated = truncated[:last_space]
        return truncated.rstrip(".,;:") + "..."

    def _validate_transcription_schema(self, raw_text: str) -> tuple[bool, str]:
        """Validate that *raw_text* is valid JSON with required schema keys.

        Args:
            raw_text: Raw text response from the API.

        Returns:
            ``(is_valid, reason)`` -- *is_valid* is True when JSON is valid
            and contains all required top-level keys.
        """
        stripped = strip_markdown_code_block(raw_text)
        try:
            obj = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return False, "invalid JSON"
        if not isinstance(obj, dict):
            return False, f"expected JSON object, got {type(obj).__name__}"
        missing = TRANSCRIPTION_REQUIRED_KEYS - obj.keys()
        if missing:
            return False, f"missing keys: {', '.join(sorted(missing))}"
        return True, ""

    def _parse_transcription_from_text(self, text: str, image_name: str = "") -> str:
        """
        Parse transcription from API response, handling special flags.

        In plain-text mode the response is treated as raw text with sentinel
        string checks.  In standard mode the response is parsed as JSON.

        Args:
            text: Raw text response from API.
            image_name: Name of the image being processed.

        Returns:
            Parsed transcription text or error/status message.
        """
        if not text:
            return f"[transcription error: {image_name or '[unknown image]'}]"

        # Plain-text mode: response IS the transcription (no JSON)
        if self.is_plain_text_mode:
            stripped = text.strip()
            img_name = self._format_image_name(image_name)
            if stripped == "[no transcribable text]":
                return f"[{img_name}: no transcribable text]"
            if stripped == "[transcription not possible]":
                return f"[{img_name}: transcription not possible]"
            return stripped

        # Standard JSON mode
        stripped = strip_markdown_code_block(text)

        if not stripped.startswith("{"):
            return text

        # Try to parse JSON response
        try:
            obj = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            # Attempt to salvage JSON from the string
            last_close = stripped.rfind("}")
            obj = None
            if last_close != -1:
                for i in range(last_close, -1, -1):
                    if stripped[i] == "{":
                        candidate = stripped[i : last_close + 1]
                        try:
                            obj = json.loads(candidate)
                            break
                        except (json.JSONDecodeError, ValueError):
                            continue

            if obj is None:
                return text

        # Handle special flags in parsed JSON
        if isinstance(obj, dict):
            img_name = self._format_image_name(image_name)

            # Check for "no_transcribable_text" flag (current schema)
            if obj.get("no_transcribable_text") is True:
                image_analysis = obj.get("image_analysis", "")
                brief_reason = self._truncate_analysis(image_analysis)
                return f"[{img_name}: no transcribable text — {brief_reason}]"

            # Check for "transcription_not_possible" flag (current schema)
            if obj.get("transcription_not_possible") is True:
                image_analysis = obj.get("image_analysis", "")
                brief_reason = self._truncate_analysis(image_analysis)
                return f"[{img_name}: transcription not possible — {brief_reason}]"

            # Extract transcription text
            transcription = obj.get("transcription")
            if isinstance(transcription, str):
                return transcription

        # Fallback: return original text
        return text

    def _resolve_image_detail(self) -> str | None:
        """Resolve the OpenAI per-image ``detail`` value from model config.

        Reads the ``image_size`` knob (low/high/auto/original) from the
        transcription model config. Returns the value to place inside the
        ``image_url`` dict, or ``None`` when nothing should be sent (knob
        unset, non-OpenAI provider, or model lacks detail support). When
        ``original`` is configured for a model that does not accept it, logs a
        warning and falls back to ``high`` rather than sending an invalid value.
        """
        # OpenAI-only: leave anthropic/google/openrouter/custom shapes untouched.
        if self.provider != "openai":
            return None
        image_size = self.model_config.get("image_size")
        if not image_size:
            return None
        detail = str(image_size).strip().lower()
        if detail not in ("low", "high", "auto", "original"):
            logger.warning(
                f"Ignoring unsupported image_size '{image_size}' for "
                f"{self.model_name}; expected low/high/auto/original."
            )
            return None
        capabilities = detect_capabilities(self.model_name)
        if not capabilities.supports_image_detail:
            logger.warning(
                f"Model '{self.model_name}' does not support the image detail "
                "parameter; ignoring image_size."
            )
            return None
        if detail == "original" and not capabilities.supports_original_image_detail:
            logger.warning(
                f"Model '{self.model_name}' does not support image_size "
                "'original'; falling back to 'high'."
            )
            return "high"
        return detail

    def _build_model_inputs(
        self, base64_image: str
    ) -> tuple[list[Any], dict[str, Any]]:
        """Build messages and invocation kwargs for the chat model."""
        system_msg = SystemMessage(content=self.system_prompt)

        # Build image content based on provider
        user_content: list[dict[str, Any]]
        if self.provider == "anthropic":
            # Anthropic format
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                }
            ]
        elif self.provider == "google":
            # Google Gemini format
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
            ]
        else:
            # OpenAI / OpenRouter format. The optional "detail" key is added
            # only for OpenAI (via _resolve_image_detail); langchain-openai
            # forwards image_url.detail into the Responses API "input_image"
            # block. Absent detail keeps the request byte-identical to before.
            image_url: dict[str, Any] = {
                "url": f"data:image/jpeg;base64,{base64_image}",
            }
            detail = self._resolve_image_detail()
            if detail is not None:
                image_url["detail"] = detail
            user_content = [
                {
                    "type": "image_url",
                    "image_url": image_url,
                }
            ]

        user_msg = HumanMessage(content=user_content)  # type: ignore[arg-type]

        # Build invoke kwargs using base class method
        invoke_kwargs = self._build_invoke_kwargs()
        self._apply_structured_output_kwargs(invoke_kwargs)

        return [system_msg, user_msg], invoke_kwargs

    def transcribe_payload(
        self,
        payload: PagePayload,
        max_schema_retries: int = 3,
    ) -> dict[str, Any]:
        """Transcribe a preprocessed in-memory page payload."""
        start_time = time.time()
        image_name = payload.image_name
        sequence_number = payload.sequence_number
        base64_image = payload.base64

        schema_retry_attempts: dict[str, int] = {
            "validation_failure": 0,
            "no_transcribable_text": 0,
            "transcription_not_possible": 0,
        }
        raw_text = ""

        # Schema retry loop (API retries handled by _invoke_with_retry)
        for _ in range(max_schema_retries + 1):
            try:
                attempt_start = time.time()

                # Build messages and invocation kwargs
                messages, invoke_kwargs = self._build_model_inputs(base64_image)

                # Get structured chat model (binds invoke kwargs on the
                # structured-output paths where invoke-time kwargs are dropped)
                structured_model = self._get_structured_chat_model(invoke_kwargs)

                # Application-level retry with per-attempt token tracking
                response = self._invoke_with_retry(
                    structured_model,
                    messages,
                    invoke_kwargs,
                    f"Transcription for {image_name}",
                )

                # Success - extract and parse response. The result carries the
                # cumulative page time (processing_time, what the ETA consumes),
                # while the per-request stats deque records only THIS attempt:
                # appending the cumulative value re-counted every earlier schema
                # attempt and its backoff sleep on each retry, inflating the
                # "average API processing time" statistic.
                processing_time = time.time() - start_time
                self._record_processing_time(time.time() - attempt_start)
                self._report_success()

                # Report token usage (built into LangChain's response metadata)
                self._report_token_usage(response, f"Transcription for {image_name}")

                raw_text = self._extract_output_text(response)

                # --- Validation retry (fires BEFORE schema-flag retries) ---
                # Active in standard mode (Modes A/C) when response must be
                # valid JSON conforming to the transcription schema.
                if not self.is_plain_text_mode:
                    is_valid, reason = self._validate_transcription_schema(raw_text)
                    if not is_valid:
                        should_retry, backoff_time, max_attempts = (
                            self._should_retry_for_schema_flag(
                                "validation_failure",
                                True,
                                schema_retry_attempts["validation_failure"],
                            )
                        )
                        if should_retry:
                            schema_retry_attempts["validation_failure"] += 1
                            logger.warning(
                                f"Validation failure for {image_name}: "
                                f"{reason}. Retrying "
                                f"({schema_retry_attempts['validation_failure']}"
                                f"/{max_attempts}) in {backoff_time:.2f}s..."
                            )
                            time.sleep(backoff_time)
                            continue

                transcription = self._parse_transcription_from_text(
                    raw_text, image_name
                )

                # --- Plain-text mode: sentinel-based retries ---
                if self.is_plain_text_mode:
                    stripped_text = raw_text.strip()
                    if stripped_text == "[no transcribable text]":
                        should_retry, backoff_time, max_attempts = (
                            self._should_retry_for_schema_flag(
                                "no_transcribable_text",
                                True,
                                schema_retry_attempts["no_transcribable_text"],
                            )
                        )
                        if should_retry:
                            schema_retry_attempts["no_transcribable_text"] += 1
                            logger.warning(
                                f"Sentinel '[no transcribable text]' for "
                                f"{image_name}. Retrying "
                                f"({schema_retry_attempts['no_transcribable_text']}"
                                f"/{max_attempts}) in {backoff_time:.2f}s..."
                            )
                            time.sleep(backoff_time)
                            continue

                    elif stripped_text == "[transcription not possible]":
                        should_retry, backoff_time, max_attempts = (
                            self._should_retry_for_schema_flag(
                                "transcription_not_possible",
                                True,
                                schema_retry_attempts["transcription_not_possible"],
                            )
                        )
                        if should_retry:
                            schema_retry_attempts["transcription_not_possible"] += 1
                            logger.warning(
                                f"Sentinel '[transcription not possible]' for "
                                f"{image_name}. Retrying "
                                f"({schema_retry_attempts['transcription_not_possible']}"
                                f"/{max_attempts}) in {backoff_time:.2f}s..."
                            )
                            time.sleep(backoff_time)
                            continue

                # --- Standard JSON mode: schema-flag retries ---
                if not self.is_plain_text_mode:
                    # Parse the raw text as JSON to check for schema flags.
                    # Strip any markdown code fence first (as validation and
                    # parsing do) so a fenced response does not bypass the
                    # configured schema-flag retries.
                    parsed_response = None
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        parsed_response = json.loads(
                            strip_markdown_code_block(raw_text)
                        )

                    # Check for schema-specific retry conditions
                    if isinstance(parsed_response, dict):
                        # Check no_transcribable_text flag
                        if parsed_response.get("no_transcribable_text") is True:
                            should_retry, backoff_time, max_attempts = (
                                self._should_retry_for_schema_flag(
                                    "no_transcribable_text",
                                    True,
                                    schema_retry_attempts["no_transcribable_text"],
                                )
                            )
                            if should_retry:
                                schema_retry_attempts["no_transcribable_text"] += 1
                                logger.warning(
                                    "Schema flag 'no_transcribable_text' detected "
                                    f"for {image_name}. Retrying "
                                    f"({schema_retry_attempts['no_transcribable_text']}"
                                    f"/{max_attempts}) in {backoff_time:.2f}s..."
                                )
                                time.sleep(backoff_time)
                                continue

                        # Check transcription_not_possible flag
                        if parsed_response.get("transcription_not_possible") is True:
                            should_retry, backoff_time, max_attempts = (
                                self._should_retry_for_schema_flag(
                                    "transcription_not_possible",
                                    True,
                                    schema_retry_attempts["transcription_not_possible"],
                                )
                            )
                            if should_retry:
                                schema_retry_attempts["transcription_not_possible"] += 1
                                logger.warning(
                                    "Schema flag 'transcription_not_possible' detected "
                                    f"for {image_name}. Retrying "
                                    f"({schema_retry_attempts['transcription_not_possible']}"
                                    f"/{max_attempts}) in {backoff_time:.2f}s..."
                                )
                                time.sleep(backoff_time)
                                continue

                result = {
                    "image": image_name,
                    "sequence_number": sequence_number,
                    "transcription": transcription,
                    "processing_time": round(processing_time, 2),
                    "provider": self.provider,
                }

                # Include schema retry statistics if any occurred
                total_schema_retries = sum(schema_retry_attempts.values())
                if total_schema_retries > 0:
                    result["schema_retries"] = schema_retry_attempts.copy()

                return result

            except Exception as e:
                # _invoke_with_retry has exhausted all retries if we get here
                self._report_failure()
                logger.error(
                    f"Transcription API error for {image_name} after retries: "
                    f"{type(e).__name__} - {e}"
                )
                return {
                    "image": image_name,
                    "sequence_number": sequence_number,
                    "transcription": f"[transcription error: {e}]",
                    "processing_time": round(time.time() - start_time, 2),
                    "error": str(e),
                    "error_type": "api_failure",
                    "provider": self.provider,
                }

        # Fallback if schema retry loop exits (all schema retries exhausted)
        return {
            "image": image_name,
            "sequence_number": sequence_number,
            "transcription": self._parse_transcription_from_text(raw_text, image_name),
            "processing_time": round(time.time() - start_time, 2),
            "schema_retries": schema_retry_attempts.copy(),
            "provider": self.provider,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about API usage."""
        stats = super().get_stats()
        stats["service_tier"] = self.service_tier
        return stats


# ============================================================================
# Public API
# ============================================================================
__all__ = ["TranscriptionManager"]
