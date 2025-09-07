import random
import re
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional

from openai import Client as OpenAIClient
from PIL import Image

import config
from utils.rate_limiter import RateLimiter
from utils.image_processor import ImageProcessor

class APIRequestManager:
    """Manages API requests with concurrency and rate limiting."""

    def __init__(
            self, api_key: str, base_url: str, model_name: str,
            rate_limiter: RateLimiter, timeout: int = config.SAIA_API_TIMEOUT,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.rate_limiter = rate_limiter
        self.timeout = timeout
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times = deque(maxlen=50)  # For stats
        self.client = OpenAIClient(api_key=api_key, base_url=base_url, timeout=timeout)

    def _get_sequence_number(self, image_path: Path) -> int:
        try:
            return int(image_path.stem.split("_")[-1])
        except (IndexError, ValueError):
            try:
                # Fallback: extract last number from filename
                nums = [int(s) for s in re.findall(r'\d+', image_path.stem)]
                return nums[-1] if nums else 0
            except Exception:  # If no numbers or other parsing error
                return 0

    def transcribe_image(
            self, image_path: Path, image_processor: ImageProcessor,
            max_retries: int = 5
    ) -> Dict[str, Any]:
        sequence_num = self._get_sequence_number(image_path)
        retries = 0
        start_time = time.time()
        backoff_base = 1.0

        pil_img: Optional[Image.Image] = None
        processed_pil_img: Optional[Image.Image] = None

        try:
            try:
                pil_img = Image.open(image_path)
                processed_pil_img = image_processor.process_pil_image(pil_img)  # Process original
                base64_image = image_processor.encode_pil_to_base64(processed_pil_img)
            except Exception as proc_err:
                print(f"Error processing image {image_path.name}: {proc_err}")
                return {
                    "page": sequence_num, "image": image_path.name,
                    "transcription": f"[ERROR] Image loading/processing: {proc_err}",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(proc_err),
                    "error_type": "processing", "retries": 0,
                }

            if not base64_image:
                return {
                    "page": sequence_num, "image": image_path.name,
                    "transcription": "[ERROR] Failed to encode image.",
                    "timestamp": datetime.now().isoformat(),
                    "error": "Image encoding failed",
                    "error_type": "encoding", "retries": 0,
                }

            while retries <= max_retries:
                try:
                    _ = self.rate_limiter.wait_for_capacity()
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": config.SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Transcribe this image."},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            },
                        ],
                        temperature=0.05, max_tokens=4096,
                        frequency_penalty=0.1, presence_penalty=0.1,
                        timeout=self.timeout,
                    )
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    self.successful_requests += 1
                    self.rate_limiter.report_success()
                    return {
                        "page": sequence_num, "image": image_path.name,
                        "transcription": response.choices[0].message.content,
                        "timestamp": datetime.now().isoformat(),
                        "processing_time": round(processing_time, 2),
                    }
                except Exception as e:
                    retries += 1
                    error_message = str(e).lower()
                    is_rate_limit = any(err in error_message for err in ["rate limit", "too many", "429"])
                    is_server_error = any(err in error_message for err in ["server error", "500", "502", "503", "504"])
                    is_timeout = any(err in error_message for err in ["timeout", "timed out"])
                    is_network_error = any(err in error_message for err in ["connection", "network"])
                    is_retryable = is_rate_limit or is_server_error or is_timeout or is_network_error

                    self.rate_limiter.report_error(is_rate_limit)

                    if not is_retryable or retries > max_retries:
                        error_type = "other"
                        if is_rate_limit:
                            error_type = "rate_limit"
                        elif is_server_error:
                            error_type = "server"
                        elif is_timeout:
                            error_type = "timeout"
                        elif is_network_error:
                            error_type = "network"
                        self.failed_requests += 1
                        return {
                            "page": sequence_num, "image": image_path.name,
                            "transcription": f"[ERROR] API failure after {retries - 1} retries: {e}",
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                            "error_type": error_type, "retries": retries - 1,
                        }

                    # Calculate wait time with jitter
                    if is_rate_limit:
                        wait_time = backoff_base * (2 ** retries) * (0.8 + 0.4 * random.random())
                        msg_prefix = "Rate limit"
                    elif is_timeout:
                        wait_time = backoff_base * (1.5 ** retries) * (0.5 + 0.5 * random.random())
                        msg_prefix = "API timeout"
                    else:  # Other retryable errors
                        wait_time = backoff_base * (2 ** (retries - 1)) * (0.5 + random.random())
                        msg_prefix = "Error"
                    print(
                        f"{msg_prefix} for {image_path.name} (attempt {retries}/{max_retries + 1}). "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
        except Exception as outer_e:
            print(f"Critical error in transcribe_image for {image_path.name}: {outer_e}")
            return {
                "page": sequence_num, "image": image_path.name,
                "transcription": f"[CRITICAL ERROR] Task-level: {outer_e}",
                "timestamp": datetime.now().isoformat(), "error": str(outer_e),
                "error_type": "task_unexpected", "retries": retries,
            }
        finally:
            if processed_pil_img and hasattr(processed_pil_img, 'close'):
                processed_pil_img.close()
            if pil_img and hasattr(pil_img, 'close'):
                pil_img.close()

        # Fallback if loop finishes unexpectedly (should be caught by max_retries)
        return {
            "page": sequence_num, "image": image_path.name,
            "transcription": f"[ERROR] Unknown failure after {max_retries} retries.",
            "timestamp": datetime.now().isoformat(),
            "error": "Max retries loop completed",
            "error_type": "unknown_max_retries", "retries": max_retries,
        }

    def get_stats(self) -> Dict[str, Any]:
        import statistics
        avg_time = statistics.mean(self.processing_times) if self.processing_times else 0
        success_rate = (
                self.successful_requests / max(1, self.successful_requests + self.failed_requests) * 100
        )
        return {
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_processing_time": round(avg_time, 2),
            "recent_success_rate": round(success_rate, 1),
        }
