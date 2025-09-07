import json
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
import base64

from modules import app_config as config
from api.rate_limiter import RateLimiter
from modules.prompt_utils import render_prompt_with_schema
from modules.image_utils import ImageProcessor as ModImageProcessor
from modules.config_loader import ConfigLoader, PROMPTS_DIR, SCHEMAS_DIR


class OpenAITranscriptionManager:
    """Transcribes images using OpenAI Responses API with gpt-5-mini.

    Uses a JSON schema (from modules/markdown_transcription_schema.json) and a
    system prompt (from modules/system_prompt.txt) to enforce structured output.
    Returns a dict compatible with the previous workflow: includes keys
    'page', 'image', 'transcription', optional 'processing_time', 'error', etc.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: int = config.OPENAI_API_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.client = OpenAI(api_key=api_key, timeout=timeout)

        self.rate_limiter = rate_limiter or RateLimiter(config.OPENAI_RATE_LIMITS)
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times = deque(maxlen=50)

        # Load schema and system prompt from modules/
        self.transcription_schema: Optional[Dict[str, Any]] = None
        self.system_prompt: str = "You are an expert OCR system. Return only the transcription."
        try:
            schema_path = (SCHEMAS_DIR / "markdown_transcription_schema.json").resolve()
            if schema_path.exists():
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.transcription_schema = json.load(f)
            prompt_path = (PROMPTS_DIR / "system_prompt.txt").resolve()
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    raw_prompt = f.read()
                # Render prompt with schema injection using prompt_utils
                if self.transcription_schema is not None:
                    bare_schema = self.transcription_schema.get("schema", self.transcription_schema)
                    self.system_prompt = render_prompt_with_schema(raw_prompt, bare_schema)
                else:
                    self.system_prompt = raw_prompt
        except Exception:
            # Fallback silently; defaults are already set
            pass

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

    def _get_sequence_number(self, image_path: Path) -> int:
        try:
            # typical pattern: page_0001 -> 1
            stem = image_path.stem
            parts = stem.split("_")
            last = parts[-1]
            if last.isdigit():
                return int(last)
        except Exception:
            pass
        # Fallback: try to extract last number anywhere in the stem
        try:
            import re
            nums = [int(s) for s in re.findall(r"\d+", image_path.stem)]
            return nums[-1] if nums else 0
        except Exception:
            return 0

    def _build_text_format(self) -> Optional[Dict[str, Any]]:
        if not isinstance(self.transcription_schema, dict):
            return None
        # Accept wrapper form { name, strict, schema } or bare schema
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

    @staticmethod
    def _extract_output_text(data: Any) -> str:
        # Attempt to normalize Responses API result into a plain string
        try:
            # SDK object may expose output_text directly
            text_attr = getattr(data, "output_text", None)
            if isinstance(text_attr, str):
                return text_attr.strip()
        except Exception:
            pass
        # Try dict-style
        try:
            if isinstance(data, dict) and isinstance(data.get("output_text"), str):
                return data["output_text"].strip()
        except Exception:
            pass
        # Fallback: attempt to collect from output list
        try:
            obj = data
            if not isinstance(obj, dict):
                # try to_dict
                to_dict = getattr(data, "to_dict", None) or getattr(data, "model_dump", None)
                if callable(to_dict):
                    obj = to_dict()
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

    @staticmethod
    def _parse_transcription_from_text(text: str, image_name: str = "") -> str:
        # If JSON is returned, prefer the 'transcription' field with flags
        if not text:
            return f"[transcription error: {image_name or '[unknown image]'}]"
        stripped = text.lstrip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
            except Exception:
                # try to salvage the last JSON object in the string
                last_close = stripped.rfind("}")
                obj = None
                if last_close != -1:
                    i = last_close
                    while i >= 0:
                        if stripped[i] == "{":
                            candidate = stripped[i:last_close + 1]
                            try:
                                obj = json.loads(candidate)
                                break
                            except Exception:
                                pass
                        i -= 1
            if isinstance(obj, dict):
                if obj.get("no_transcribable_text", False):
                    return "[empty page]"
                if obj.get("transcription_not_possible", False):
                    return "[no transcription possible]"
                if "transcription" in obj:
                    val = obj.get("transcription")
                    return (val or "").strip() if isinstance(val, str) else ""
        return text

    def transcribe_image(
        self,
        image_path: Path,
        image_processor: Any,
        max_retries: int = 5,
    ) -> Dict[str, Any]:
        sequence_num = self._get_sequence_number(image_path)
        retries = 0
        start_time = time.time()
        backoff_base = 1.0

        processed_file_path: Optional[Path] = None

        try:
            try:
                # Preprocess to a dedicated folder using modules.image_utils settings
                pre_dir = image_path.parent / "preprocessed_images"
                pre_dir.mkdir(exist_ok=True)
                out_base = pre_dir / image_path.stem
                proc = ModImageProcessor(image_path)
                _msg = proc.process_image(out_base)
                processed_file_path = out_base.with_suffix('.jpg')
                if not processed_file_path.exists():
                    raise FileNotFoundError(f"Processed image not found: {processed_file_path}")
                # Read and encode processed file
                with open(processed_file_path, 'rb') as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
            except Exception as proc_err:
                print(f"Error processing image {image_path.name}: {proc_err}")
                return {
                    "page": sequence_num,
                    "image": image_path.name,
                    "transcription": f"[ERROR] Image loading/processing: {proc_err}",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(proc_err),
                    "error_type": "processing",
                    "retries": 0,
                }

            if not base64_image:
                return {
                    "page": sequence_num,
                    "image": image_path.name,
                    "transcription": "[ERROR] Failed to encode image.",
                    "timestamp": datetime.now().isoformat(),
                    "error": "Image encoding failed",
                    "error_type": "encoding",
                    "retries": 0,
                }

            service_tier = self.service_tier
            text_format = self._build_text_format()

            while retries <= max_retries:
                try:
                    _ = self.rate_limiter.wait_for_capacity()

                    input_messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": self.system_prompt}
                            ],
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

                    payload: Dict[str, Any] = {
                        "model": self.model_name,
                        "input": input_messages,
                        "service_tier": service_tier,
                        "max_output_tokens": 8192,
                        "reasoning": {"effort": "medium"},
                    }
                    if text_format is not None:
                        payload["text"] = {"format": text_format}

                    response = self.client.responses.create(**payload)

                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    self.successful_requests += 1
                    self.rate_limiter.report_success()

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
                    error_message = str(e).lower()
                    is_rate_limit = any(err in error_message for err in ["rate limit", "too many", "429"]) or "retry-after" in error_message
                    is_server_error = any(err in error_message for err in ["server error", "500", "502", "503", "504", "service unavailable"])
                    is_timeout = any(err in error_message for err in ["timeout", "timed out"]) or "deadline" in error_message
                    is_network_error = any(err in error_message for err in ["connection", "network", "temporarily unavailable"])
                    is_retryable = is_rate_limit or is_server_error or is_timeout or is_network_error

                    self.rate_limiter.report_error(is_rate_limit or is_server_error)

                    if not is_retryable or retries > max_retries:
                        error_type = (
                            "rate_limit"
                            if is_rate_limit
                            else "server"
                            if is_server_error
                            else "timeout"
                            if is_timeout
                            else "network"
                            if is_network_error
                            else "other"
                        )
                        self.failed_requests += 1
                        return {
                            "page": sequence_num,
                            "image": image_path.name,
                            "transcription": f"[ERROR] API failure after {retries - 1} retries: {e}",
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                            "error_type": error_type,
                            "retries": retries - 1,
                        }

                    # backoff with jitter
                    if is_rate_limit:
                        wait_time = backoff_base * (2 ** retries) * (0.8 + 0.4 * random.random())
                        msg_prefix = "Rate limit"
                    elif is_timeout:
                        wait_time = backoff_base * (1.5 ** retries) * (0.5 + 0.5 * random.random())
                        msg_prefix = "API timeout"
                    else:
                        wait_time = backoff_base * (2 ** (retries - 1)) * (0.5 + random.random())
                        msg_prefix = "Error"
                    print(
                        f"{msg_prefix} for {image_path.name} (attempt {retries}/{max_retries + 1}). Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
        except Exception as outer_e:
            print(f"Critical error in transcribe_image for {image_path.name}: {outer_e}")
            return {
                "page": sequence_num,
                "image": image_path.name,
                "transcription": f"[CRITICAL ERROR] Task-level: {outer_e}",
                "timestamp": datetime.now().isoformat(),
                "error": str(outer_e),
                "error_type": "task_unexpected",
                "retries": retries,
            }
        finally:
            # Keep processed files in working dir for potential inspection; no cleanup here
            pass

        # Fallback (should not be reached under normal conditions)
        return {
            "page": sequence_num,
            "image": image_path.name,
            "transcription": f"[ERROR] Unknown failure after {max_retries} retries.",
            "timestamp": datetime.now().isoformat(),
            "error": "Max retries loop completed",
            "error_type": "unknown_max_retries",
            "retries": max_retries,
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
