"""OpenAI transcription API client using Responses API with structured outputs."""

from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from modules import app_config as config
from api.base_openai_client import OpenAIClientBase, DEFAULT_MAX_RETRIES
from api.rate_limiter import RateLimiter
from modules.prompt_utils import render_prompt_with_schema
from modules.image_utils import ImageProcessor
from modules.config_loader import ConfigLoader, PROMPTS_DIR, SCHEMAS_DIR
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Constants
TRANSCRIPTION_SCHEMA_FILE = "transcription_schema.json"
SYSTEM_PROMPT_FILE = "transcription_system_prompt.txt"


class OpenAITranscriptionManager(OpenAIClientBase):
    """
    Transcribes images using OpenAI Responses API with structured outputs.
    
    This class handles image transcription using the OpenAI Responses API with
    structured JSON outputs. Images are preprocessed and encoded in-memory before
    being sent to the API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: int = config.OPENAI_API_TIMEOUT,
    ) -> None:
        """
        Initialize the transcription manager.

        Args:
            api_key: OpenAI API key.
            model_name: Model to use (e.g., 'gpt-5-mini').
            rate_limiter: Optional RateLimiter instance.
            timeout: Request timeout in seconds.
        """
        super().__init__(api_key, model_name, timeout, rate_limiter)

        # Load schema and system prompt
        self.transcription_schema: Optional[Dict[str, Any]] = None
        self.system_prompt: str = ""

        self._load_schema_and_prompt()
        
        # Load model configuration and determine service tier using base class methods
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
        Build the Responses API text.format object for Structured Outputs.

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

    def transcribe_image(
        self,
        image_path: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Dict[str, Any]:
        """
        Transcribe a single image using OpenAI Responses API.
        
        This method implements two levels of retries:
        1. API error retries: For rate limits, timeouts, server errors, etc.
        2. Schema-specific retries: For content validation flags like no_transcribable_text.

        Args:
            image_path: Path to the image file.
            max_retries: Maximum number of retry attempts for API errors.

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

        # Prepare API request
        input_messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ],
            },
        ]

        text_format = self._build_text_format()
        api_retries = 0
        schema_retry_attempts = {
            "no_transcribable_text": 0,
            "transcription_not_possible": 0,
        }

        # Main retry loop
        while api_retries <= max_retries:
            try:
                self._wait_for_rate_limit()

                payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": input_messages,
                    "service_tier": self.service_tier,
                }

                # Add max_output_tokens from config
                max_tokens = self.model_config.get("max_output_tokens", 16384)
                payload["max_output_tokens"] = max_tokens

                # Add reasoning parameters if specified
                if "reasoning" in self.model_config:
                    reasoning_cfg = self.model_config["reasoning"]
                    if isinstance(reasoning_cfg, dict) and "effort" in reasoning_cfg:
                        payload["reasoning"] = {"effort": reasoning_cfg["effort"]}

                # Add text parameters if specified
                if "text" in self.model_config:
                    text_cfg = self.model_config["text"]
                    if isinstance(text_cfg, dict):
                        text_params = {}
                        if "verbosity" in text_cfg:
                            text_params["verbosity"] = text_cfg["verbosity"]

                        if text_format:
                            text_params["format"] = text_format

                        if text_params:
                            payload["text"] = text_params
                elif text_format:
                    payload["text"] = {"format": text_format}

                response = self.client.responses.create(**payload)

                # Success - extract and parse response
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self._report_success()

                raw_text = self._extract_output_text(response)
                transcription = self._parse_transcription_from_text(
                    raw_text, image_path.name
                )
                
                # Parse the raw text as JSON to check for schema flags
                parsed_response = None
                try:
                    parsed_response = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON, continue with transcription as-is
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
                            continue  # Retry the request
                    
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
                            continue  # Retry the request

                result = {
                    "image": image_path.name,
                    "sequence_number": sequence_number,
                    "transcription": transcription,
                    "processing_time": round(processing_time, 2),
                }
                
                # Include schema retry statistics if any occurred
                total_schema_retries = sum(schema_retry_attempts.values())
                if total_schema_retries > 0:
                    result["schema_retries"] = schema_retry_attempts.copy()

                return result

            except Exception as e:
                api_retries += 1
                is_retryable, error_type = self._classify_error(str(e))
                self._report_error(
                    error_type in ["rate_limit", "server", "resource_unavailable"]
                )

                if not is_retryable or api_retries > max_retries:
                    self.failed_requests += 1
                    logger.error(
                        f"Transcription API error for {image_path.name} (final attempt): "
                        f"{type(e).__name__} - {e}"
                    )
                    return {
                        "image": image_path.name,
                        "sequence_number": sequence_number,
                        "transcription": f"[transcription error: {e}]",
                        "processing_time": round(time.time() - start_time, 2),
                        "error": str(e),
                        "error_type": "api_failure",
                        "api_retries": api_retries - 1,
                    }

                # Calculate backoff and retry
                wait_time = self._calculate_backoff_time(api_retries, error_type)
                logger.warning(
                    f"Transcription API error for {image_path.name} "
                    f"(attempt {api_retries}/{max_retries + 1}). "
                    f"Retrying in {wait_time:.2f}s... "
                    f"Error: {type(e).__name__} - {e}"
                )
                time.sleep(wait_time)

        # Fallback if loop exits unexpectedly
        self.failed_requests += 1
        return {
            "image": image_path.name,
            "sequence_number": sequence_number,
            "transcription": "[transcription error: max retries exceeded]",
            "processing_time": round(time.time() - start_time, 2),
            "error": "Max retries exceeded",
            "error_type": "max_retries_exceeded",
            "api_retries": max_retries,
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
