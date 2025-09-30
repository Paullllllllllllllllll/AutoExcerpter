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
TRANSCRIPTION_SCHEMA_FILE = "markdown_transcription_schema.json"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
DEFAULT_SYSTEM_PROMPT = "You are an expert OCR system. Return only the transcription."
MAX_OUTPUT_TOKENS = 8192
REASONING_EFFORT = "medium"


class OpenAITranscriptionManager(OpenAIClientBase):
    """Transcribes images using OpenAI Responses API with structured outputs."""

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
            api_key: OpenAI API key
            model_name: Model to use (e.g., 'gpt-5-mini')
            rate_limiter: Optional RateLimiter instance
            timeout: Request timeout in seconds
        """
        super().__init__(api_key, model_name, timeout, rate_limiter)

        # Load schema and system prompt
        self.transcription_schema: Optional[Dict[str, Any]] = None
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT

        self._load_schema_and_prompt()
        self._determine_service_tier()

    def _load_schema_and_prompt(self) -> None:
        """Load transcription schema and system prompt from configuration files."""
        try:
            schema_path = (SCHEMAS_DIR / TRANSCRIPTION_SCHEMA_FILE).resolve()
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.transcription_schema = json.load(f)

            prompt_path = (PROMPTS_DIR / SYSTEM_PROMPT_FILE).resolve()
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    raw_prompt = f.read()

                # Render prompt with schema injection
                if self.transcription_schema is not None:
                    bare_schema = self.transcription_schema.get("schema", self.transcription_schema)
                    self.system_prompt = render_prompt_with_schema(raw_prompt, bare_schema)
                else:
                    self.system_prompt = raw_prompt
        except Exception as e:
            logger.warning(f"Error loading schema/prompt: {e}. Using defaults.")

    def _determine_service_tier(self) -> None:
        """Determine service tier from configuration."""
        try:
            config_loader = ConfigLoader()
            config_loader.load_configs()
            service_tier = (
                config_loader.get_concurrency_config()
                .get("concurrency", {})
                .get("transcription", {})
                .get("service_tier")
            )
            self.service_tier = service_tier if service_tier else (
                "flex" if config.OPENAI_USE_FLEX else "auto"
            )
        except Exception:
            self.service_tier = "flex" if config.OPENAI_USE_FLEX else "auto"

    def _extract_sequence_number(self, image_path: Path) -> int:
        """
        Extract page/sequence number from image filename.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted sequence number, or 0 if not found
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
            Format specification dictionary, or None if schema not available
        """
        if not isinstance(self.transcription_schema, dict):
            return None

        name = self.transcription_schema.get("name", "markdown_transcription_schema")
        strict = bool(self.transcription_schema.get("strict", True))
        schema_obj = self.transcription_schema.get("schema", self.transcription_schema)

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
            text: Raw text response from API
            image_name: Name of the image being processed

        Returns:
            Parsed transcription text or error/status message
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
                        candidate = stripped[i:last_close + 1]
                        try:
                            obj = json.loads(candidate)
                            break
                        except Exception:
                            pass

        if isinstance(obj, dict):
            # Check for special flags
            if obj.get("no_transcribable_text", False):
                return "[empty page]"
            if obj.get("transcription_not_possible", False):
                return "[no transcription possible]"
            if "transcription" in obj:
                val = obj.get("transcription")
                return (val or "").strip() if isinstance(val, str) else ""

        return text

    def _preprocess_image(self, image_path: Path) -> Optional[str]:
        """
        Preprocess image and encode to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image string, or None on error
        """
        try:
            pre_dir = image_path.parent / "preprocessed_images"
            pre_dir.mkdir(exist_ok=True)
            out_base = pre_dir / image_path.stem

            processor = ImageProcessor(image_path)
            processor.process_image(out_base)

            processed_file_path = out_base.with_suffix('.jpg')
            if not processed_file_path.exists():
                raise FileNotFoundError(f"Processed image not found: {processed_file_path}")

            with open(processed_file_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path.name}: {e}")
            return None

    def transcribe_image(
        self,
        image_path: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> Dict[str, Any]:
        """
        Transcribe an image using OpenAI Responses API.

        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing transcription result and metadata
        """
        sequence_num = self._extract_sequence_number(image_path)
        start_time = time.time()

        # Preprocess and encode image
        base64_image = self._preprocess_image(image_path)
        if base64_image is None:
            return self._create_error_result(
                sequence_num, image_path.name, "Image preprocessing failed", "processing", 0
            )

        # Prepare API request
        input_messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Please transcribe the text from this image following the schema.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            },
        ]

        text_format = self._build_text_format()
        retries = 0

        # Retry loop
        while retries <= max_retries:
            try:
                self._wait_for_rate_limit()

                payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": input_messages,
                    "service_tier": self.service_tier,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "reasoning": {"effort": REASONING_EFFORT},
                }
                if text_format is not None:
                    payload["text"] = {"format": text_format}

                response = self.client.responses.create(**payload)

                # Success - extract and parse response
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self._report_success()

                out_text = self._extract_output_text(response)
                transcription_text = self._parse_transcription_from_text(out_text, image_path.name)

                return {
                    "page": sequence_num,
                    "image": image_path.name,
                    "transcription": transcription_text,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": round(processing_time, 2),
                }

            except Exception as e:
                retries += 1
                is_retryable, error_type = self._classify_error(str(e))
                self._report_error(error_type in ["rate_limit", "server", "resource_unavailable"])

                if not is_retryable or retries > max_retries:
                    self.failed_requests += 1
                    return self._create_error_result(
                        sequence_num, image_path.name, str(e), error_type, retries - 1
                    )

                # Calculate backoff and retry
                wait_time = self._calculate_backoff_time(retries, error_type)
                logger.warning(
                    f"{'Rate limit' if error_type == 'rate_limit' else 'Error'} for {image_path.name} "
                    f"(attempt {retries}/{max_retries + 1}). Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)

        # Should not reach here, but provide fallback
        return self._create_error_result(
            sequence_num, image_path.name, "Max retries exceeded", "max_retries", max_retries
        )

    def _create_error_result(
        self, page: int, image_name: str, error: str, error_type: str, retries: int
    ) -> Dict[str, Any]:
        """
        Create a standardized error result dictionary.

        Args:
            page: Page/sequence number
            image_name: Name of the image file
            error: Error message
            error_type: Type of error
            retries: Number of retries attempted

        Returns:
            Error result dictionary
        """
        return {
            "page": page,
            "image": image_name,
            "transcription": f"[ERROR] {error}",
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "error_type": error_type,
            "retries": retries,
        }
