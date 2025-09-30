import json
import time
from typing import Any, Dict

from modules import app_config as config
from api.base_openai_client import OpenAIClientBase, DEFAULT_MAX_RETRIES
from api.rate_limiter import RateLimiter
from modules.prompt_utils import render_prompt_with_schema
from modules.config_loader import ConfigLoader, PROMPTS_DIR, SCHEMAS_DIR
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Constants
SUMMARY_SCHEMA_FILE = "summary_schema.json"
SUMMARY_PROMPT_FILE = "summary_system_prompt.txt"
DEFAULT_SUMMARY_PROMPT = "Summarize the provided text according to the JSON schema below."
MAX_OUTPUT_TOKENS = 8192
REASONING_EFFORT = "high"


class OpenAISummaryManager(OpenAIClientBase):
    """Manages OpenAI API requests for generating structured summaries."""

    def __init__(self, api_key: str, model_name: str) -> None:
        """
        Initialize the summary manager.

        Args:
            api_key: OpenAI API key
            model_name: Model to use (e.g., 'gpt-5-mini')
        """
        # Initialize rate limiter with OpenAI-specific limits
        rate_limiter = RateLimiter(config.OPENAI_RATE_LIMITS)
        super().__init__(api_key, model_name, config.OPENAI_API_TIMEOUT, rate_limiter)

        # Load schema and system prompt
        self.summary_schema: Dict[str, Any] | None = None
        self.summary_system_prompt_text: str = DEFAULT_SUMMARY_PROMPT
        
        # Load model configuration parameters
        self.model_config: Dict[str, Any] = {}

        self._load_schema_and_prompt()
        self._load_model_config()
        self._determine_service_tier()

    def _load_schema_and_prompt(self) -> None:
        """Load summary schema and system prompt from configuration files."""
        try:
            schema_path = (SCHEMAS_DIR / SUMMARY_SCHEMA_FILE).resolve()
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.summary_schema = json.load(f)
                logger.info(f"Loaded summary schema from {SUMMARY_SCHEMA_FILE}")
            else:
                logger.warning(f"Summary schema not found at {schema_path}")

            prompt_path = (PROMPTS_DIR / SUMMARY_PROMPT_FILE).resolve()
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    self.summary_system_prompt_text = f.read()
                logger.info(f"Loaded summary system prompt ({len(self.summary_system_prompt_text)} chars)")
            else:
                logger.warning(f"Summary prompt not found at {prompt_path}")
        except Exception as e:
            logger.warning(f"Error loading schema/prompt: {e}. Using defaults.")

    def _load_model_config(self) -> None:
        """Load model configuration from model.yaml."""
        try:
            config_loader = ConfigLoader()
            config_loader.load_configs()
            model_cfg = config_loader.get_model_config()
            self.model_config = model_cfg.get("summary_model", {})
            
            if self.model_config:
                logger.debug(f"Loaded model config: {self.model_config.get('name', 'unknown')}")
        except Exception as e:
            logger.warning(f"Error loading model config: {e}. Using defaults.")
            self.model_config = {}

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

    def _build_text_format(self) -> Dict[str, Any]:
        """
        Build the Responses API text.format object for Structured Outputs.

        Returns:
            Format specification dictionary
        """
        schema_obj = self.summary_schema or {}
        name = schema_obj.get("name", "article_page_summary") if isinstance(schema_obj, dict) else "article_page_summary"
        strict = bool(schema_obj.get("strict", True)) if isinstance(schema_obj, dict) else True
        schema = schema_obj.get("schema", schema_obj) if isinstance(schema_obj, dict) else {}

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
            page_num: Page number
            error_message: Error message to include

        Returns:
            Placeholder summary dictionary
        """
        bullet_text = f"[Error generating summary: {error_message}]" if error_message else "[Summary generation failed]"

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

    def generate_summary(
        self, transcription: str, page_num: int, max_retries: int = DEFAULT_MAX_RETRIES
    ) -> Dict[str, Any]:
        """
        Generate a structured summary from transcription text.

        Args:
            transcription: The transcribed text to summarize
            page_num: Page number for context
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing summary result and metadata
        """
        start_time = time.time()

        # Prepare request payload
        schema_obj = (
            self.summary_schema.get("schema")
            if isinstance(self.summary_schema, dict) and "schema" in self.summary_schema
            else self.summary_schema
        ) or {}
        system_text = render_prompt_with_schema(self.summary_system_prompt_text, schema_obj)

        input_messages = [
            {
                "role": "system",
                "content": system_text,
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": transcription}],
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
                }
                
                # Add max_output_tokens from config (default to 8192 if not specified)
                max_tokens = self.model_config.get("max_output_tokens", MAX_OUTPUT_TOKENS)
                payload["max_output_tokens"] = max_tokens
                
                # Add reasoning parameters if specified (for GPT-5 and o-series models)
                if "reasoning" in self.model_config:
                    reasoning_cfg = self.model_config["reasoning"]
                    if isinstance(reasoning_cfg, dict) and "effort" in reasoning_cfg:
                        payload["reasoning"] = {"effort": reasoning_cfg["effort"]}
                else:
                    # Fallback to default
                    payload["reasoning"] = {"effort": REASONING_EFFORT}
                
                # Add text parameters if specified (for GPT-5 family)
                if "text" in self.model_config:
                    text_cfg = self.model_config["text"]
                    if isinstance(text_cfg, dict):
                        text_params = {}
                        if "verbosity" in text_cfg:
                            text_params["verbosity"] = text_cfg["verbosity"]
                        
                        # Add format if we have a schema
                        if text_format:
                            text_params["format"] = text_format
                        
                        if text_params:
                            payload["text"] = text_params
                elif text_format:
                    # No text config, but we have a format
                    payload["text"] = {"format": text_format}
                
                response = self.client.responses.create(**payload)

                # Success - extract and parse response
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self._report_success()

                summary_json_str = self._extract_output_text(response)
                if not summary_json_str:
                    raise ValueError("OpenAI API returned empty content for summary.")

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
                if "page_number" not in summary_json:
                    summary_json["page_number"] = {
                        "page_number_integer": page_num,
                        "contains_no_page_number": False,
                    }
                elif not isinstance(summary_json["page_number"], dict):
                    # Handle malformed responses
                    contains_no_page_number = summary_json.get("contains_no_page_number", False)
                    summary_json["page_number"] = {
                        "page_number_integer": page_num,
                        "contains_no_page_number": contains_no_page_number,
                    }
                    if "contains_no_page_number" in summary_json:
                        del summary_json["contains_no_page_number"]

                result = {
                    "page": page_num,
                    "summary": summary_json,
                    "processing_time": round(processing_time, 2),
                }
                
                # Add full API response object for logging
                try:
                    result["api_response"] = {
                        "id": response.id if hasattr(response, 'id') else None,
                        "model": response.model if hasattr(response, 'model') else None,
                        "created_at": response.created_at if hasattr(response, 'created_at') else None,
                        "output": [
                            {
                                "role": item.role if hasattr(item, 'role') else None,
                                "content": [
                                    {
                                        "type": c.type if hasattr(c, 'type') else None,
                                        "text": c.text if hasattr(c, 'text') else None,
                                    } for c in (item.content if hasattr(item, 'content') and item.content else [])
                                ] if hasattr(item, 'content') else []
                            } for item in (response.output if hasattr(response, 'output') and response.output else [])
                        ] if hasattr(response, 'output') else [],
                        "usage": {
                            "input_tokens": response.usage.input_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens') else None,
                            "output_tokens": response.usage.output_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens') else None,
                            "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else None,
                        } if hasattr(response, 'usage') and response.usage else None,
                    }
                except Exception as e:
                    logger.debug(f"Could not serialize full API response: {e}")
                    result["api_response"] = {"serialization_error": str(e)}
                
                return result

            except Exception as e:
                retries += 1
                is_retryable, error_type = self._classify_error(str(e))
                self._report_error(error_type in ["rate_limit", "server", "resource_unavailable"])

                if not is_retryable or retries > max_retries:
                    self.failed_requests += 1
                    logger.error(
                        f"Summary API error for page {page_num} (final attempt): {type(e).__name__} - {e}"
                    )
                    placeholder = self._create_placeholder_summary(page_num, str(e))
                    placeholder["error_type"] = "api_failure"
                    return placeholder

                # Calculate backoff and retry
                wait_time = self._calculate_backoff_time(retries, error_type)
                logger.warning(
                    f"Summary API error for page {page_num} (attempt {retries}/{max_retries + 1}). "
                    f"Retrying in {wait_time:.2f}s... Error: {type(e).__name__} - {e}"
                )
                time.sleep(wait_time)

        # Fallback if loop exits unexpectedly
        self.failed_requests += 1
        placeholder = self._create_placeholder_summary(page_num, "Max retries exceeded")
        placeholder["error_type"] = "max_retries_exceeded"
        return placeholder

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API usage including flex processing status.

        Returns:
            Dictionary containing request statistics
        """
        stats = super().get_stats()
        stats["flex_processing"] = "Enabled" if config.OPENAI_USE_FLEX else "Disabled"
        return stats
