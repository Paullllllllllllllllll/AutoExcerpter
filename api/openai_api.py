import json
import random
import time
from collections import deque
from typing import Dict, Any

from openai import OpenAI

import config
from utils.rate_limiter import RateLimiter


class OpenAISummaryManager:
	"""Manages OpenAI API requests for generating structured summaries."""

	def __init__(self, api_key: str, model_name: str):
		self.api_key = api_key
		self.model_name = model_name  # e.g., "o4-mini"
		self.client = OpenAI(api_key=api_key, timeout=config.OPENAI_API_TIMEOUT)
		self.successful_requests = 0
		self.failed_requests = 0
		self.processing_times = deque(maxlen=50)
		self.rate_limiter = RateLimiter(
			config.OPENAI_RATE_LIMITS)  # Use configured OpenAI rate limits

	def generate_summary(self, transcription: str, page_num: int,
	                     max_retries: int = 3) -> Dict[str, Any]:
		start_time = time.time()
		retries = 0

		messages_payload = [
			{
				"role": "system",
				"content": [
					{"type": "text", "text": config.SUMMARY_SYSTEM_PROMPT}]
			},
			{
				"role": "user",
				"content": [{"type": "text", "text": transcription}]
			}
		]

		# Defines the expected JSON structure for the API response.
		# 'name', 'strict', and 'schema' are part of the ChatCompletionResponseFormatJSONSchema.
		response_format_payload = {
			"type": "json_schema",
			"json_schema": {
				"name": config.SUMMARY_SCHEMA["name"],
				"strict": config.SUMMARY_SCHEMA["strict"],
				"schema": config.SUMMARY_SCHEMA["schema"]
			}
		}

		try:
			while retries <= max_retries:
				try:
					_ = self.rate_limiter.wait_for_capacity()

					# Add flex processing configuration
					service_tier = "flex" if config.OPENAI_USE_FLEX else "auto"

					response = self.client.chat.completions.create(
						model=self.model_name,
						messages=messages_payload,
						reasoning_effort="high",
						response_format=response_format_payload,
						service_tier=service_tier,
					)

					processing_time = time.time() - start_time
					self.processing_times.append(processing_time)
					self.successful_requests += 1
					self.rate_limiter.report_success()

					if not response.choices or not response.choices[
						0].message or not response.choices[0].message.content:
						raise ValueError(
							"OpenAI API returned an invalid or empty response structure.")

					summary_json_str = response.choices[0].message.content
					if not summary_json_str:
						raise ValueError(
							"OpenAI API returned an empty content string for summary.")

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
						print(
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
					print(
						f"Summary API error for page {page_num} (attempt {retries}/{max_retries + 1}). Retrying in {wait_time:.2f}s... Error: {type(e).__name__} - {e}")
					time.sleep(wait_time)

		except Exception as outer_e:  # Catch unexpected errors in the retry loop itself
			print(
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
