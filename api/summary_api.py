"""Multi-provider summary API client using LangChain with structured outputs.

This module provides text summarization using LangChain's unified interface,
supporting multiple LLM providers:
- OpenAI (GPT-5, GPT-4o, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- OpenRouter (access to multiple providers)

The client handles:
- Structured JSON output parsing
- Dual-level retry logic (API errors + schema validation)
- Rate limiting integration
- Token usage tracking
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from api.base_llm_client import LLMClientBase, DEFAULT_MAX_RETRIES
from api.llm_client import ProviderType
from api.rate_limiter import RateLimiter
from modules.concurrency_helper import get_api_timeout, get_rate_limits
from modules.prompt_utils import render_prompt_with_schema
from modules.config_loader import PROMPTS_DIR, SCHEMAS_DIR
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker

logger = setup_logger(__name__)

# Constants
SUMMARY_SCHEMA_FILE = "summary_schema.json"
SUMMARY_PROMPT_FILE = "summary_system_prompt.txt"


def _transform_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform JSON schema to be Anthropic-compatible.
    
    Anthropic's SDK doesn't support union types like ["string", "null"].
    This function converts them to simple types.
    Also adds required 'title' and 'description' keys for LangChain compatibility.
    """
    import copy
    result = copy.deepcopy(schema)
    
    def transform_type(obj: dict[str, Any]) -> None:
        if not isinstance(obj, dict):
            return
            
        # Handle union types like ["string", "null"]
        if "type" in obj and isinstance(obj["type"], list):
            # Filter out "null" and keep the first non-null type
            non_null_types = [t for t in obj["type"] if t != "null"]
            if non_null_types:
                obj["type"] = non_null_types[0]
            else:
                obj["type"] = "string"  # fallback
        
        # Recursively handle properties
        if "properties" in obj and isinstance(obj["properties"], dict):
            for prop in obj["properties"].values():
                transform_type(prop)
        
        # Handle items in arrays
        if "items" in obj and isinstance(obj["items"], dict):
            transform_type(obj["items"])
        
        # Handle anyOf/oneOf/allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    transform_type(item)
    
    transform_type(result)
    
    # Add required top-level keys for LangChain/Anthropic compatibility
    if "title" not in result:
        result["title"] = "SummarySchema"
    if "description" not in result:
        result["description"] = "Schema for document summary output"
    
    return result


class SummaryManager(LLMClientBase):
    """
    Manages LLM API requests for generating structured summaries.
    
    This class handles summarization of transcribed text using various LLM providers
    through LangChain's unified interface. It loads schemas and prompts from 
    configuration files and applies model-specific parameters.
    
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
        summary_context: str | None = None,
    ) -> None:
        """
        Initialize the summary manager.

        Args:
            model_name: Model to use (e.g., 'gpt-5-mini', 'claude-sonnet-4-5-20250929').
            provider: Provider name (openai, anthropic, google, openrouter).
            api_key: Optional API key. If None, uses environment variable.
            summary_context: Optional context string for guiding summarization focus.
        """
        # Initialize rate limiter with provider-agnostic configuration
        rate_limiter = RateLimiter(get_rate_limits())
        super().__init__(model_name, provider, api_key, get_api_timeout(), rate_limiter)

        # Load schema and system prompt
        self.summary_schema: dict[str, Any] | None = None
        self.summary_system_prompt_text: str = ""

        self._load_schema_and_prompt()
        
        # Store summary context for prompt injection
        self.summary_context = summary_context
        
        # Load model configuration and determine service tier
        self.model_config = self._load_model_config("summary_model")
        self.service_tier = self._determine_service_tier("summary")
        
        # Load schema-specific retry configuration
        self.schema_retry_config = self._load_schema_retry_config("summary")

    def _load_schema_and_prompt(self) -> None:
        """Load summary schema and system prompt from configuration files."""
        try:
            # Load schema
            schema_path = (SCHEMAS_DIR / SUMMARY_SCHEMA_FILE).resolve()
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.summary_schema = json.load(f)
                logger.info(f"Loaded summary schema from {SUMMARY_SCHEMA_FILE}")
            else:
                logger.error(f"Summary schema not found at {schema_path}")
                raise FileNotFoundError(f"Required schema file missing: {schema_path}")

            # Load prompt
            prompt_path = (PROMPTS_DIR / SUMMARY_PROMPT_FILE).resolve()
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self.summary_system_prompt_text = f.read()
                logger.info(
                    f"Loaded summary system prompt "
                    f"({len(self.summary_system_prompt_text)} chars)"
                )
            else:
                logger.error(f"Summary prompt not found at {prompt_path}")
                raise FileNotFoundError(f"Required prompt file missing: {prompt_path}")
                
        except Exception as e:
            logger.error(f"Error loading schema/prompt: {e}")
            raise

    def _build_text_format(self) -> dict[str, Any]:
        """Build the structured output format specification."""
        schema_obj = self.summary_schema or {}
        name = (
            schema_obj.get("name", "article_page_summary")
            if isinstance(schema_obj, dict)
            else "article_page_summary"
        )
        strict = (
            bool(schema_obj.get("strict", True))
            if isinstance(schema_obj, dict)
            else True
        )
        schema = (
            schema_obj.get("schema", schema_obj)
            if isinstance(schema_obj, dict)
            else {}
        )

        return {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": strict,
        }

    def _create_placeholder_summary(
        self, page_num: int, error_message: str = "", page_type: str = "other"
    ) -> dict[str, Any]:
        """Create a placeholder summary for pages that couldn't be processed."""
        bullet_text = (
            f"[Error generating summary: {error_message}]"
            if error_message
            else "[Summary generation failed]"
        )

        result = {
            "page": page_num,
            "summary": {
                "page_information": {
                    "page_number_integer": page_num,
                    "page_number_type": "arabic",
                    "page_type": page_type,
                },
                "bullet_points": [bullet_text],
                "references": None,
            },
        }

        if error_message:
            result["error"] = error_message

        return result

    def _build_model_inputs(self, transcription: str) -> tuple[list, dict[str, Any]]:
        """Build messages and invocation kwargs for summary generation."""
        # Prepare system prompt with schema
        schema_obj = (
            self.summary_schema.get("schema")
            if isinstance(self.summary_schema, dict) and "schema" in self.summary_schema
            else self.summary_schema
        ) or {}
        system_text = render_prompt_with_schema(
            self.summary_system_prompt_text, schema_obj, context=self.summary_context
        )

        system_msg = SystemMessage(content=system_text)
        
        # Build user message based on provider
        if self.provider in ("openai", "openrouter"):
            user_msg = HumanMessage(
                content=[{"type": "text", "text": transcription}]
            )
        else:
            # Anthropic, Google use simpler format
            user_msg = HumanMessage(content=transcription)

        # Build invoke kwargs using base class method
        invoke_kwargs = self._build_invoke_kwargs()

        # Add structured output format for OpenAI (native response_format)
        if self.provider == "openai":
            text_format = self._build_text_format()
            if text_format:
                if "text" in invoke_kwargs:
                    invoke_kwargs["text"]["format"] = text_format
                else:
                    invoke_kwargs["response_format"] = text_format

        return [system_msg, user_msg], invoke_kwargs
    
    def _get_structured_chat_model(self):
        """Get chat model with structured output for each provider.
        
        Provider-specific approaches:
        - OpenAI: Native response_format parameter (guaranteed JSON) - handled separately
        - Anthropic/Google: Prompt-based JSON with markdown stripping fallback
          (LangChain's with_structured_output has tool naming compatibility issues)
        - OpenRouter: Default tool-based structured output (OpenAI-compatible)
        
        Returns:
            Chat model with structured output, or base chat model.
        """
        # OpenAI uses native response_format - no need for with_structured_output
        if self.provider == "openai":
            return self.chat_model
        
        # Anthropic/Google: Don't use with_structured_output due to compatibility issues
        # Anthropic: Tool names must match ^[a-zA-Z0-9_-]{1,128}$ pattern
        # Google: Function names have strict requirements
        # Both rely on schema in system prompt + markdown stripping fallback
        if self.provider in ("anthropic", "google"):
            return self.chat_model
        
        # OpenRouter: Use tool-based structured output (OpenAI-compatible)
        if self.summary_schema:
            schema = self.summary_schema.get("schema", self.summary_schema)
            if isinstance(schema, dict) and schema:
                return self.chat_model.with_structured_output(
                    schema,
                    include_raw=True,
                )
        
        return self.chat_model

    def _ensure_page_information_structure(
        self, summary_json: dict[str, Any], page_num: int
    ) -> None:
        """Ensure page_information structure is correct in the summary JSON."""
        if "page_information" not in summary_json:
            summary_json["page_information"] = {
                "page_number_integer": page_num,
                "page_number_type": "arabic",
                "page_types": ["content"],
            }
        elif not isinstance(summary_json["page_information"], dict):
            summary_json["page_information"] = {
                "page_number_integer": page_num,
                "page_number_type": "arabic",
                "page_types": ["content"],
            }
        else:
            # Ensure all required fields exist
            page_info = summary_json["page_information"]
            if "page_number_integer" not in page_info:
                page_info["page_number_integer"] = page_num
            if "page_number_type" not in page_info:
                page_info["page_number_type"] = "arabic" if page_info.get("page_number_integer") else "none"
            # Handle both page_types (array) and legacy page_type (string)
            if "page_types" not in page_info:
                if "page_type" in page_info:
                    # Convert legacy page_type to page_types array
                    legacy_type = page_info.pop("page_type")
                    page_info["page_types"] = [legacy_type] if legacy_type else ["content"]
                else:
                    page_info["page_types"] = ["content"]

    def generate_summary(
        self, transcription: str, page_num: int, max_schema_retries: int = 3
    ) -> dict[str, Any]:
        """Generate a structured summary from transcription text."""
        start_time = time.time()

        schema_retry_attempts = {
            "page_type_null_bullets": 0,
        }
        
        summary_json = None

        # Schema retry loop (API retries handled by LangChain)
        for _ in range(max_schema_retries + 1):
            try:
                self._wait_for_rate_limit()

                # Build messages and invocation kwargs
                messages, invoke_kwargs = self._build_model_inputs(transcription)

                # Get structured chat model (uses with_structured_output for non-OpenAI providers)
                structured_model = self._get_structured_chat_model()
                
                # LangChain handles API retries with exponential backoff internally
                response = structured_model.invoke(messages, **invoke_kwargs)

                # Success - extract and parse response
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self._report_success()
                
                # Report token usage (built into LangChain's response metadata)
                try:
                    usage_meta = getattr(response, "usage_metadata", None)
                    if usage_meta and isinstance(usage_meta, dict):
                        total_tokens = usage_meta.get("total_tokens")
                        if total_tokens and isinstance(total_tokens, int):
                            token_tracker = get_token_tracker()
                            token_tracker.add_tokens(total_tokens)
                            logger.debug(
                                f"[TOKEN] Summary for page {page_num}: "
                                f"added {total_tokens} tokens (total now: {token_tracker.get_tokens_used_today():,})"
                            )
                except Exception as e:
                    logger.warning(f"Error reporting token usage for page {page_num}: {e}")

                summary_json_str = self._extract_output_text(response)
                if not summary_json_str:
                    raise ValueError("LLM API returned empty content for summary.")

                # Strip markdown code blocks if present
                summary_json_str = summary_json_str.strip()
                if summary_json_str.startswith("```json"):
                    summary_json_str = summary_json_str[7:]  # Remove ```json
                elif summary_json_str.startswith("```"):
                    summary_json_str = summary_json_str[3:]  # Remove ```
                if summary_json_str.endswith("```"):
                    summary_json_str = summary_json_str[:-3]  # Remove ```
                summary_json_str = summary_json_str.strip()

                # Parse JSON with better error handling
                try:
                    summary_json = json.loads(summary_json_str)
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"JSON decode error for page {page_num}: {json_err}. "
                        f"Raw content (first 500 chars): {summary_json_str[:500]}"
                    )
                    raise ValueError(f"Invalid JSON in API response: {json_err}")

                # Ensure page_information structure is correct
                self._ensure_page_information_structure(summary_json, page_num)

                result = {
                    "page": page_num,
                    "summary": summary_json,
                    "processing_time": round(processing_time, 2),
                    "provider": self.provider,
                }
                
                # Include schema retry statistics if any occurred
                total_schema_retries = sum(schema_retry_attempts.values())
                if total_schema_retries > 0:
                    result["schema_retries"] = schema_retry_attempts.copy()

                # Add response metadata for logging
                response_meta = getattr(response, "response_metadata", {})
                if isinstance(response_meta, dict):
                    result["api_response"] = response_meta

                return result

            except Exception as e:
                # LangChain has already exhausted its retries if we get here
                self.failed_requests += 1
                self._report_error(True)
                logger.error(
                    f"Summary API error for page {page_num} after LangChain retries: "
                    f"{type(e).__name__} - {e}"
                )
                placeholder = self._create_placeholder_summary(page_num, str(e))
                placeholder["error_type"] = "api_failure"
                placeholder["provider"] = self.provider
                return placeholder

        # Fallback if schema retry loop exits (all schema retries exhausted)
        result = {
            "page": page_num,
            "summary": summary_json,
            "processing_time": round(time.time() - start_time, 2),
            "schema_retries": schema_retry_attempts.copy(),
            "provider": self.provider,
        }
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about API usage."""
        stats = super().get_stats()
        stats["service_tier"] = self.service_tier
        return stats


# ============================================================================
# Public API
# ============================================================================
__all__ = ["SummaryManager"]
