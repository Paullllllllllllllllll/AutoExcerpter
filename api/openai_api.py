import json
import random
import time
from collections import deque
from typing import Dict, Any

from openai import OpenAI
 
from modules import app_config as config
from modules.logger import setup_logger
from api.rate_limiter import RateLimiter
from modules.prompt_utils import render_prompt_with_schema
from modules.config_loader import ConfigLoader, PROMPTS_DIR, SCHEMAS_DIR


logger = setup_logger(__name__)


class OpenAISummaryManager:
	"""Manages OpenAI API requests for generating structured summaries."""

	def __init__(self, api_key: str, model_name: str):
		self.api_key = api_key
		self.model_name = model_name  # e.g., "gpt-5-mini"
		self.client = OpenAI(api_key=api_key, timeout=config.OPENAI_API_TIMEOUT)
		self.successful_requests = 0
		self.failed_requests = 0
		self.processing_times = deque(maxlen=50)
		self.rate_limiter = RateLimiter(
			config.OPENAI_RATE_LIMITS)  # Use configured OpenAI rate limits
		# Determine service tier preference from concurrency config, fallback to config flag
		try:
			cc = ConfigLoader()
			cc.load_configs()
			st = (
				(cc.get_concurrency_config().get("concurrency", {}) or {})
				.get("transcription", {})
				.get("service_tier")
			)
		except Exception:
			st = None
		self.service_tier = st if st else ("flex" if config.OPENAI_USE_FLEX else "auto")

		# Load summary schema and system prompt from modules
		self.summary_schema: Dict[str, Any] | None = None
		self.summary_system_prompt_text: str = "Summarize the provided text according to the JSON schema below."
		try:
			schema_path = (SCHEMAS_DIR / "summary_schema.json").resolve()
			if schema_path.exists():
				with open(schema_path, "r", encoding="utf-8") as f:
					self.summary_schema = json.load(f)
			prompt_path = (PROMPTS_DIR / "summary_system_prompt.txt").resolve()
			if prompt_path.exists():
				with open(prompt_path, "r", encoding="utf-8") as f:
					self.summary_system_prompt_text = f.read()
		except Exception:
			# leave defaults
			pass

	def _build_text_format(self) -> Dict[str, Any]:
		"""Build the Responses API text.format object for Structured Outputs."""
		schema_obj = self.summary_schema or {}
		# Accept wrapper form { name, strict, schema } or bare schema
		name = schema_obj.get("name", "article_page_summary") if isinstance(schema_obj, dict) else "article_page_summary"
		strict = bool(schema_obj.get("strict", True)) if isinstance(schema_obj, dict) else True
		schema = schema_obj.get("schema", schema_obj) if isinstance(schema_obj, dict) else {}
		return {
			"type": "json_schema",
			"name": name,
			"schema": schema,
			"strict": strict,
		}

	@staticmethod
	def _extract_output_text(data: Any) -> str:
		"""Normalize Responses output into a single text string."""
		# Try SDK convenience attr
		try:
			text_attr = getattr(data, "output_text", None)
			if isinstance(text_attr, str):
				return text_attr.strip()
		except Exception:
			pass
		# Try dict-style
		if isinstance(data, dict) and isinstance(data.get("output_text"), str):
			return data["output_text"].strip()
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
					if isinstance(item, dict) and item.get("type") == "message":
						for c in item.get("content", []):
							t = c.get("text")
							if isinstance(t, str):
								parts.append(t)
			return "".join(parts).strip()
		except Exception:
			return ""

	def generate_summary(self, transcription: str, page_num: int,
	                     max_retries: int = 3) -> Dict[str, Any]:
		start_time = time.time()
		retries = 0

		# Responses API payload components
		# Render system prompt with embedded schema
		schema_obj = (self.summary_schema.get("schema") if isinstance(self.summary_schema, dict) and "schema" in self.summary_schema else self.summary_schema) or {}
		system_text = render_prompt_with_schema(self.summary_system_prompt_text, schema_obj)
		input_messages = [
			{
				"role": "system",
				"content": [
					{"type": "input_text", "text": system_text}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "input_text", "text": transcription}
				]
			}
		]
		service_tier = self.service_tier
		text_format = self._build_text_format()

		try:
			while retries <= max_retries:
				try:
					_ = self.rate_limiter.wait_for_capacity()

					payload: Dict[str, Any] = {
						"model": self.model_name,
						"input": input_messages,
						"service_tier": service_tier,
						"max_output_tokens": 8192,
						"reasoning": {"effort": "high"},
					}
					if text_format:
						payload["text"] = {"format": text_format}

					response = self.client.responses.create(**payload)

					processing_time = time.time() - start_time
					self.processing_times.append(processing_time)
					self.successful_requests += 1
					self.rate_limiter.report_success()

					summary_json_str = self._extract_output_text(response)
					if not summary_json_str:
						raise ValueError("OpenAI API returned an empty content string for summary.")

					# The model should return JSON adhering to the schema
					summary_json = json.loads(summary_json_str)

					# Create page number object if it doesn't exist in the response
					if "page_number" not in summary_json:
						summary_json["page_number"] = {
							"page_number_integer": page_num,
							"contains_no_page_number": False
						}
					# Ensure page_number is correctly structured as an object
					elif not isinstance(summary_json["page_number"], dict):
						# Handle malformed responses with old schema format
						contains_no_page_number = summary_json.get(
							"contains_no_page_number", False)
						summary_json["page_number"] = {
							"page_number_integer": page_num,
							"contains_no_page_number": contains_no_page_number
						}
						# Remove old format field if present
						if "contains_no_page_number" in summary_json:
							del summary_json["contains_no_page_number"]

					return {
						"page": page_num,
						"summary": summary_json,
						"processing_time": round(processing_time, 2)
					}

				except Exception as e:
					retries += 1
					error_message = str(e).lower()
					is_rate_limit = any(err in error_message for err in
					                    ["rate limit", "too many", "429"])
					is_server_error = any(err in error_message for err in
					                      ["server error", "500", "502", "503",
					                       "504", "service unavailable"])
					is_timeout = any(err in error_message for err in
					                 ["timeout", "timed out"])
					is_network_error = any(err in error_message for err in
					                       ["connection", "network"])
					is_resource_unavailable = "resource unavailable" in error_message
					is_retryable = is_rate_limit or is_server_error or is_timeout or is_network_error or is_resource_unavailable

					self.rate_limiter.report_error(
						is_rate_limit or is_server_error or is_resource_unavailable)  # Backoff for rate limits and server errors

					if not is_retryable or retries > max_retries:
						self.failed_requests += 1
						logger.error(
                            f"Summary API error for page {page_num} (final attempt): {type(e).__name__} - {e}")
						return {
							"page": page_num,
							"summary": {
								"page_number": {
									"page_number_integer": page_num,
									"contains_no_page_number": False
								},
								"bullet_points": [
									f"[Error generating summary: {e}]"],
								"references": [],
								"contains_no_semantic_content": True
							},
							"error": str(e),
							"error_type": "api_failure"
						}

					wait_time = (2 ** retries) * (
								0.5 + 0.5 * random.random())  # Exponential backoff with jitter
					logger.warning(
                        f"Summary API error for page {page_num} (attempt {retries}/{max_retries + 1}). Retrying in {wait_time:.2f}s... Error: {type(e).__name__} - {e}")
					time.sleep(wait_time)

		except Exception as outer_e:  # Catch unexpected errors in the retry loop itself
			logger.exception(
                f"Critical error in generate_summary for page {page_num}: {type(outer_e).__name__} - {outer_e}")
			self.failed_requests += 1
			return {
				"page": page_num,
				"summary": {
					"page_number": {
						"page_number_integer": page_num,
						"contains_no_page_number": False
					},
					"bullet_points": [
						f"[Critical error generating summary: {outer_e}]"],
					"references": [],
					"contains_no_semantic_content": True
				},
				"error": str(outer_e),
				"error_type": "task_unexpected"
			}

		# Fallback if loop finishes unexpectedly (should be caught by max_retries logic)
		self.failed_requests += 1
		return {
			"page": page_num,
			"summary": {
				"page_number": {
					"page_number_integer": page_num,
					"contains_no_page_number": False
				},
				"bullet_points": [
					"[Summary generation failed after maximum retries]"],
				"references": [],
				"contains_no_semantic_content": True
			},
			"error": "Max retries exceeded",
			"error_type": "max_retries_exceeded"
		}

	def get_stats(self) -> Dict[str, Any]:
		import statistics
		avg_time = statistics.mean(
			self.processing_times) if self.processing_times else 0
		success_rate = (
				self.successful_requests / max(1,
				                               self.successful_requests + self.failed_requests) * 100
		)
		return {
			"successful_requests": self.successful_requests,
			"failed_requests": self.failed_requests,
			"average_processing_time": round(avg_time, 2),
			"recent_success_rate": round(success_rate, 1),
			"flex_processing": "Enabled" if config.OPENAI_USE_FLEX else "Disabled"
		}
