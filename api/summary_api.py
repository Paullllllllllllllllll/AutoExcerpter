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
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from modules import app_config as config
from api.base_llm_client import LLMClientBase, DEFAULT_MAX_RETRIES
from api.llm_client import ProviderType
from api.rate_limiter import RateLimiter
from modules.prompt_utils import render_prompt_with_schema
from modules.config_loader import PROMPTS_DIR, SCHEMAS_DIR
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker

logger = setup_logger(__name__)

# Constants
SUMMARY_SCHEMA_FILE = "summary_schema.json"
SUMMARY_PROMPT_FILE = "summary_system_prompt.txt"


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
        provider: Optional[ProviderType] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the summary manager.

        Args:
            model_name: Model to use (e.g., 'gpt-5-mini', 'claude-sonnet-4-5-20250929').
            provider: Provider name (openai, anthropic, google, openrouter).
            api_key: Optional API key. If None, uses environment variable.
        """
        # Initialize rate limiter
        rate_limiter = RateLimiter(config.OPENAI_RATE_LIMITS)
        super().__init__(model_name, provider, api_key, config.OPENAI_API_TIMEOUT, rate_limiter)

        # Load schema and system prompt
        self.summary_schema: Dict[str, Any] | None = None
        self.summary_system_prompt_text: str = ""

        self._load_schema_and_prompt()
        
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

    def _build_text_format(self) -> Dict[str, Any]:
        """
        Build the structured output format specification.
        
        For OpenAI: Returns json_schema format for Responses API
        For other providers: Returns simplified JSON format instruction

        Returns:
            Format specification dictionary.
        """
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
        self, page_num: int, error_message: str = ""
    ) -> Dict[str, Any]:
        """
        Create a placeholder summary for pages that couldn't be processed.

        Args:
            page_num: Page number.
            error_message: Error message to include.

        Returns:
            Placeholder summary dictionary.
        """
        bullet_text = (
            f"[Error generating summary: {error_message}]"
            if error_message
            else "[Summary generation failed]"
        )

        result = {
            "page": page_num,
            "summary": {
                "page_number": {
                    "page_number_integer": page_num,
                    "contains_no_page_number": False,
                },
                "bullet_points": [bullet_text],
                "references": [],
                "contains_no_semantic_content": True,
            },
        }

        if error_message:
            result["error"] = error_message

        return result

    def _build_model_inputs(self, transcription: str) -> Tuple[list, Dict[str, Any]]:
        """
        Build messages and invocation kwargs for summary generation.
        
        Args:
            transcription: The transcribed text to summarize.
            
        Returns:
            Tuple of (messages, invoke_kwargs).
        """
        # Prepare system prompt with schema
        schema_obj = (
            self.summary_schema.get("schema")
            if isinstance(self.summary_schema, dict) and "schema" in self.summary_schema
            else self.summary_schema
        ) or {}
        system_text = render_prompt_with_schema(
            self.summary_system_prompt_text, schema_obj
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

        # Add structured output format for OpenAI
        if self.provider == "openai":
            text_format = self._build_text_format()
            if text_format:
                if "text" in invoke_kwargs:
                    invoke_kwargs["text"]["format"] = text_format
                else:
                    invoke_kwargs["response_format"] = text_format

        return [system_msg, user_msg], invoke_kwargs

    def _ensure_page_number_structure(
        self, summary_json: Dict[str, Any], page_num: int
    ) -> None:
        """
        Ensure page_number structure is correct in the summary JSON.
        
        Args:
            summary_json: The summary JSON to validate/fix.
            page_num: The page number to use as fallback.
        """
        if "page_number" not in summary_json:
            summary_json["page_number"] = {
                "page_number_integer": page_num,
                "contains_no_page_number": False,
            }
        elif not isinstance(summary_json["page_number"], dict):
            contains_no_page_number = summary_json.get(
                "contains_no_page_number", False
            )
            summary_json["page_number"] = {
                "page_number_integer": page_num,
                "contains_no_page_number": contains_no_page_number,
            }
            if "contains_no_page_number" in summary_json:
                del summary_json["contains_no_page_number"]

    def generate_summary(
        self, transcription: str, page_num: int, max_schema_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a structured summary from transcription text.
        
        API error retries (rate limits, timeouts, server errors) are handled automatically
        by LangChain via the max_retries parameter set during client initialization.
        
        This method only handles schema-specific retries for content validation flags
        like contains_no_semantic_content and contains_no_page_number.

        Args:
            transcription: The transcribed text to summarize.
            page_num: Page number for context.
            max_schema_retries: Maximum schema-specific retry attempts.

        Returns:
            Dictionary containing summary result and metadata.
        """
        start_time = time.time()

        schema_retry_attempts = {
            "contains_no_semantic_content": 0,
            "contains_no_page_number": 0,
        }
        
        summary_json = None

        # Schema retry loop (API retries handled by LangChain)
        for _ in range(max_schema_retries + 1):
            try:
                self._wait_for_rate_limit()

                # Build messages and invocation kwargs
                messages, invoke_kwargs = self._build_model_inputs(transcription)

                # LangChain handles API retries with exponential backoff internally
                response = self.chat_model.invoke(messages, **invoke_kwargs)

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

                # Parse JSON with better error handling
                try:
                    summary_json = json.loads(summary_json_str)
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"JSON decode error for page {page_num}: {json_err}. "
                        f"Raw content (first 500 chars): {summary_json_str[:500]}"
                    )
                    raise ValueError(f"Invalid JSON in API response: {json_err}")

                # Ensure page_number structure is correct
                self._ensure_page_number_structure(summary_json, page_num)

                # Check for schema-specific retry conditions
                if summary_json.get("contains_no_semantic_content") is True:
                    should_retry, backoff_time, max_attempts = self._should_retry_for_schema_flag(
                        "contains_no_semantic_content",
                        True,
                        schema_retry_attempts["contains_no_semantic_content"]
                    )
                    
                    if should_retry:
                        schema_retry_attempts["contains_no_semantic_content"] += 1
                        logger.warning(
                            f"Schema flag 'contains_no_semantic_content' detected for page {page_num}. "
                            f"Retrying ({schema_retry_attempts['contains_no_semantic_content']}/{max_attempts}) "
                            f"in {backoff_time:.2f}s..."
                        )
                        time.sleep(backoff_time)
                        continue
                
                # Check contains_no_page_number flag
                page_number_obj = summary_json.get("page_number", {})
                if isinstance(page_number_obj, dict):
                    if page_number_obj.get("contains_no_page_number") is True:
                        should_retry, backoff_time, max_attempts = self._should_retry_for_schema_flag(
                            "contains_no_page_number",
                            True,
                            schema_retry_attempts["contains_no_page_number"]
                        )
                        
                        if should_retry:
                            schema_retry_attempts["contains_no_page_number"] += 1
                            logger.warning(
                                f"Schema flag 'contains_no_page_number' detected for page {page_num}. "
                                f"Retrying ({schema_retry_attempts['contains_no_page_number']}/{max_attempts}) "
                                f"in {backoff_time:.2f}s..."
                            )
                            time.sleep(backoff_time)
                            continue

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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API usage.

        Returns:
            Dictionary containing request statistics.
        """
        stats = super().get_stats()
        stats["service_tier"] = self.service_tier
        return stats


# Backward compatibility alias
OpenAISummaryManager = SummaryManager
