"""Multi-provider transcription API client using LangChain with structured outputs.

This module provides image transcription using LangChain's unified interface,
supporting multiple LLM providers:
- OpenAI (GPT-5, GPT-4o, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- OpenRouter (access to multiple providers)

The client handles:
- Image preprocessing and base64 encoding
- Structured JSON output parsing
- Dual-level retry logic (API errors + schema validation)
- Rate limiting integration
- Token usage tracking
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from modules import app_config as config
from api.base_llm_client import LLMClientBase, DEFAULT_MAX_RETRIES
from api.llm_client import ProviderType, get_model_capabilities
from api.rate_limiter import RateLimiter
from modules.prompt_utils import render_prompt_with_schema
from modules.image_utils import ImageProcessor
from modules.config_loader import PROMPTS_DIR, SCHEMAS_DIR
from modules.logger import setup_logger
from modules.token_tracker import get_token_tracker

logger = setup_logger(__name__)

# Constants
TRANSCRIPTION_SCHEMA_FILE = "transcription_schema.json"
SYSTEM_PROMPT_FILE = "transcription_system_prompt.txt"


class TranscriptionManager(LLMClientBase):
    """
    Transcribes images using LangChain with structured outputs.
    
    This class handles image transcription using various LLM providers through
    LangChain's unified interface. Images are preprocessed and encoded in-memory
    before being sent to the API.
    
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
        rate_limiter: Optional[RateLimiter] = None,
        timeout: int = config.OPENAI_API_TIMEOUT,
    ) -> None:
        """
        Initialize the transcription manager.

        Args:
            model_name: Model to use (e.g., 'gpt-5-mini', 'claude-sonnet-4-5-20250929').
            provider: Provider name (openai, anthropic, google, openrouter).
            api_key: Optional API key. If None, uses environment variable.
            rate_limiter: Optional RateLimiter instance.
            timeout: Request timeout in seconds.
            
        Raises:
            ValueError: If the selected model doesn't support multimodal (image) input.
        """
        super().__init__(model_name, provider, api_key, timeout, rate_limiter)
        
        # Check multimodal capability (required for image transcription)
        capabilities = get_model_capabilities(model_name)
        if not capabilities.get("multimodal", False):
            logger.warning(
                f"Model '{model_name}' may not support multimodal (image) input. "
                "Image transcription may fail. Consider using gpt-5, gpt-4o, claude, or gemini."
            )

        # Load schema and system prompt
        self.transcription_schema: Optional[Dict[str, Any]] = None
        self.system_prompt: str = ""

        self._load_schema_and_prompt()
        
        # Load model configuration and determine service tier
        self.model_config = self._load_model_config("transcription_model")
        self.service_tier = self._determine_service_tier("transcription")
        
        # Load schema-specific retry configuration
        self.schema_retry_config = self._load_schema_retry_config("transcription")

    def _load_schema_and_prompt(self) -> None:
        """Load transcription schema and system prompt from configuration files."""
        try:
            # Load schema
            schema_path = (SCHEMAS_DIR / TRANSCRIPTION_SCHEMA_FILE).resolve()
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.transcription_schema = json.load(f)
                logger.info(
                    f"Loaded transcription schema from {TRANSCRIPTION_SCHEMA_FILE} "
                    f"({len(json.dumps(self.transcription_schema))} bytes)"
                )
            else:
                logger.error(f"Transcription schema not found at {schema_path}")
                raise FileNotFoundError(f"Required schema file missing: {schema_path}")

            # Load and render prompt
            prompt_path = (PROMPTS_DIR / SYSTEM_PROMPT_FILE).resolve()
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    raw_prompt = f.read()

                # Render prompt with schema injection
                if self.transcription_schema is not None:
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
                    logger.info("Loaded transcription system prompt without schema injection")
            else:
                logger.error(f"Transcription prompt not found at {prompt_path}")
                raise FileNotFoundError(f"Required prompt file missing: {prompt_path}")
                
        except Exception as e:
            logger.error(f"Error loading schema/prompt: {e}")
            raise

    def _extract_sequence_number(self, image_path: Path) -> int:
        """
        Extract page/sequence number from image filename.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted sequence number, or 0 if not found.
        """
        try:
            # Pattern: page_0001 -> 1
            stem = image_path.stem
            parts = stem.split("_")
            last = parts[-1]
            if last.isdigit():
                return int(last)
        except Exception:
            pass

        # Fallback: extract last number from filename
        try:
            nums = [int(s) for s in re.findall(r"\d+", image_path.stem)]
            return nums[-1] if nums else 0
        except Exception:
            return 0

    def _build_text_format(self) -> Optional[Dict[str, Any]]:
        """
        Build the structured output format specification.
        
        For OpenAI: Returns json_schema format for Responses API
        For other providers: Returns simplified JSON format instruction

        Returns:
            Format specification dictionary, or None if schema not available.
        """
        if not isinstance(self.transcription_schema, dict):
            return None

        name = self.transcription_schema.get("name", "markdown_transcription_schema")
        strict = bool(self.transcription_schema.get("strict", True))
        schema_obj = self.transcription_schema.get(
            "schema", self.transcription_schema
        )

        if not isinstance(schema_obj, dict) or not schema_obj:
            return None

        return {
            "type": "json_schema",
            "name": name,
            "schema": schema_obj,
            "strict": strict,
        }

    def _parse_transcription_from_text(self, text: str, image_name: str = "") -> str:
        """
        Parse transcription from JSON response, handling special flags.

        Args:
            text: Raw text response from API.
            image_name: Name of the image being processed.

        Returns:
            Parsed transcription text or error/status message.
        """
        if not text:
            return f"[transcription error: {image_name or '[unknown image]'}]"

        stripped = text.lstrip()
        if not stripped.startswith("{"):
            return text

        # Try to parse JSON response
        try:
            obj = json.loads(stripped)
        except Exception:
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
                        except Exception:
                            continue

            if obj is None:
                return text

        # Handle special flags in parsed JSON
        if isinstance(obj, dict):
            # Check for "contains_no_text" flag
            if obj.get("contains_no_text") is True:
                return "[NO TEXT ON PAGE]"

            # Check for "cannot_transcribe" flag
            if obj.get("cannot_transcribe") is True:
                reason = obj.get("reason", "unknown reason")
                return f"[CANNOT TRANSCRIBE: {reason}]"

            # Extract transcription text
            transcription = obj.get("transcription")
            if isinstance(transcription, str):
                return transcription

        # Fallback: return original text
        return text

    def _build_model_inputs(self, base64_image: str) -> Tuple[list, Dict[str, Any]]:
        """
        Build messages and invocation kwargs for the chat model.
        
        Adapts message format based on provider capabilities:
        - OpenAI: Uses input_image type with data URL
        - Anthropic: Uses image type with base64 and media_type
        - Google: Uses inline_data format
        - OpenRouter: Uses OpenAI-compatible format
        
        Args:
            base64_image: Base64-encoded image string.
            
        Returns:
            Tuple of (messages, invoke_kwargs).
        """
        system_msg = SystemMessage(content=self.system_prompt)
        
        # Build image content based on provider
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
            # OpenAI / OpenRouter format
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
            ]
        
        user_msg = HumanMessage(content=user_content)

        # Build invoke kwargs using base class method
        invoke_kwargs = self._build_invoke_kwargs()

        # Add structured output format for OpenAI
        if self.provider == "openai":
            text_format = self._build_text_format()
            if text_format:
                # Check if we already have text params
                if "text" in invoke_kwargs:
                    invoke_kwargs["text"]["format"] = text_format
                else:
                    invoke_kwargs["response_format"] = text_format

        return [system_msg, user_msg], invoke_kwargs

    def transcribe_image(
        self,
        image_path: Path,
        max_schema_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Transcribe a single image using the configured LLM provider.
        
        API error retries (rate limits, timeouts, server errors) are handled automatically
        by LangChain via the max_retries parameter set during client initialization.
        
        This method only handles schema-specific retries for content validation flags
        like no_transcribable_text and transcription_not_possible.

        Args:
            image_path: Path to the image file.
            max_schema_retries: Maximum schema-specific retry attempts.

        Returns:
            Dictionary containing transcription result and metadata.
        """
        start_time = time.time()
        sequence_number = self._extract_sequence_number(image_path)

        # Preprocess and encode image in-memory
        try:
            image_processor = ImageProcessor(image_path)
            pil_image = image_processor.process_image_to_memory()
            jpeg_quality = image_processor.img_cfg.get('jpeg_quality', 95)
            base64_image = ImageProcessor.pil_image_to_base64(pil_image, jpeg_quality)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path.name}: {e}")
            return {
                "image": image_path.name,
                "sequence_number": sequence_number,
                "transcription": f"[preprocessing error: {e}]",
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e),
                "error_type": "preprocessing_failure",
            }

        schema_retry_attempts = {
            "no_transcribable_text": 0,
            "transcription_not_possible": 0,
        }

        # Schema retry loop (API retries handled by LangChain)
        for _ in range(max_schema_retries + 1):
            try:
                self._wait_for_rate_limit()

                # Build messages and invocation kwargs
                messages, invoke_kwargs = self._build_model_inputs(base64_image)

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
                                f"[TOKEN] Transcription for {image_path.name}: "
                                f"added {total_tokens} tokens (total now: {token_tracker.get_tokens_used_today():,})"
                            )
                except Exception as e:
                    logger.warning(f"Error reporting token usage for {image_path.name}: {e}")

                raw_text = self._extract_output_text(response)
                transcription = self._parse_transcription_from_text(
                    raw_text, image_path.name
                )
                
                # Parse the raw text as JSON to check for schema flags
                parsed_response = None
                try:
                    parsed_response = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Check for schema-specific retry conditions
                if isinstance(parsed_response, dict):
                    # Check no_transcribable_text flag
                    if parsed_response.get("no_transcribable_text") is True:
                        should_retry, backoff_time, max_attempts = self._should_retry_for_schema_flag(
                            "no_transcribable_text",
                            True,
                            schema_retry_attempts["no_transcribable_text"]
                        )
                        
                        if should_retry:
                            schema_retry_attempts["no_transcribable_text"] += 1
                            logger.warning(
                                f"Schema flag 'no_transcribable_text' detected for {image_path.name}. "
                                f"Retrying ({schema_retry_attempts['no_transcribable_text']}/{max_attempts}) "
                                f"in {backoff_time:.2f}s..."
                            )
                            time.sleep(backoff_time)
                            continue
                    
                    # Check transcription_not_possible flag
                    if parsed_response.get("transcription_not_possible") is True:
                        should_retry, backoff_time, max_attempts = self._should_retry_for_schema_flag(
                            "transcription_not_possible",
                            True,
                            schema_retry_attempts["transcription_not_possible"]
                        )
                        
                        if should_retry:
                            schema_retry_attempts["transcription_not_possible"] += 1
                            logger.warning(
                                f"Schema flag 'transcription_not_possible' detected for {image_path.name}. "
                                f"Retrying ({schema_retry_attempts['transcription_not_possible']}/{max_attempts}) "
                                f"in {backoff_time:.2f}s..."
                            )
                            time.sleep(backoff_time)
                            continue

                result = {
                    "image": image_path.name,
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
                # LangChain has already exhausted its retries if we get here
                self.failed_requests += 1
                self._report_error(True)
                logger.error(
                    f"Transcription API error for {image_path.name} after LangChain retries: "
                    f"{type(e).__name__} - {e}"
                )
                return {
                    "image": image_path.name,
                    "sequence_number": sequence_number,
                    "transcription": f"[transcription error: {e}]",
                    "processing_time": round(time.time() - start_time, 2),
                    "error": str(e),
                    "error_type": "api_failure",
                    "provider": self.provider,
                }

        # Fallback if schema retry loop exits (all schema retries exhausted)
        return {
            "image": image_path.name,
            "sequence_number": sequence_number,
            "transcription": self._parse_transcription_from_text(raw_text, image_path.name),
            "processing_time": round(time.time() - start_time, 2),
            "schema_retries": schema_retry_attempts.copy(),
            "provider": self.provider,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API usage.

        Returns:
            Dictionary containing request statistics.
        """
        stats = super().get_stats()
        stats["service_tier"] = self.service_tier
        return stats
