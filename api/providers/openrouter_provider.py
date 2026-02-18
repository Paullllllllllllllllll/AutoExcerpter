"""OpenRouter provider implementation using LangChain.

OpenRouter provides access to 200+ models from multiple providers
through a unified OpenAI-compatible API.

Supported model families include:
- OpenAI models (GPT-5, GPT-4o, o1/o3, GPT-OSS-120B/20B)
- Anthropic models (Claude 4.5, Claude 3.5)
- Google models (Gemini 3, Gemini 2.5)
- DeepSeek models (R1, V3.2, V3.1)
- Meta models (Llama 3.2/3.3)
- Mistral models (Mistral Large, Pixtral)
- And many more

LangChain handles:
- Retry logic with exponential backoff (max_retries parameter)
- Token usage tracking (response_metadata)
- Structured output parsing (with_structured_output)
- Parameter filtering via disabled_params for unsupported models
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from api.model_capabilities import detect_capabilities
from api.providers.base import (
    BaseProvider,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _effort_to_ratio(effort: str) -> float:
    """Convert reasoning effort string to ratio for budget calculation."""
    e = (effort or "").strip().lower()
    return {
        "xhigh": 0.95,
        "high": 0.80,
        "medium": 0.50,
        "low": 0.20,
        "minimal": 0.10,
        "none": 0.0,
    }.get(e, 0.50)


def _compute_openrouter_reasoning_max_tokens(*, max_tokens: int, effort: str) -> int:
    """Compute reasoning max_tokens budget based on effort level.
    
    Used for Anthropic and Gemini models via OpenRouter, where reasoning
    is controlled via max_tokens rather than effort string.
    """
    ratio = _effort_to_ratio(effort)
    if ratio <= 0:
        return 0
    # Keep budget within OpenRouter docs guidance for Anthropic reasoning.
    # Ensure some tokens remain for the final response.
    reserve_for_answer = 256
    upper = max(0, int(max_tokens) - reserve_for_answer)
    budget = int(int(max_tokens) * ratio)
    budget = min(budget, 32000, upper)
    budget = max(budget, 1024)
    return budget


def _load_max_retries() -> int:
    """Load max retries from concurrency.yaml."""
    try:
        from modules.config_loader import get_config_loader
        conc_cfg = get_config_loader().get_concurrency_config() or {}
        retry_cfg = conc_cfg.get("retry", {}) or {}
        attempts = int(retry_cfg.get("max_attempts", 5))
        return max(1, attempts)
    except Exception:
        return 5


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider using LangChain.
    
    Uses the OpenAI-compatible API endpoint provided by OpenRouter
    to access 200+ models from various providers.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )
        
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.site_url = site_url
        self.app_name = app_name or "AutoExcerpter"
        self.reasoning_config = reasoning_config
        
        self._capabilities = detect_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build disabled_params for models that don't support certain features
        disabled_params = self._build_disabled_params()
        
        # Build model kwargs - include all params, LangChain will filter via disabled_params
        model_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        # Apply OpenRouter unified reasoning controls.
        # OpenRouter accepts a top-level `reasoning` object and will route/translate
        # it when supported by the selected model/provider.
        if self._capabilities.supports_reasoning_effort and reasoning_config:
            reasoning_payload: Dict[str, Any] = {}

            effort = reasoning_config.get("effort")
            if effort:
                reasoning_payload["effort"] = str(effort)

            max_reasoning_tokens = reasoning_config.get("max_tokens")
            if max_reasoning_tokens is not None:
                try:
                    reasoning_payload["max_tokens"] = int(max_reasoning_tokens)
                except Exception:
                    pass

            exclude = reasoning_config.get("exclude")
            if exclude is not None:
                reasoning_payload["exclude"] = bool(exclude)

            enabled = reasoning_config.get("enabled")
            if enabled is not None:
                reasoning_payload["enabled"] = bool(enabled)

            if reasoning_payload:
                # Avoid sending both effort and max_tokens for models where OpenRouter expects
                # one or the other.
                m = (model or "").lower().strip()

                # For Anthropic and Gemini thinking models, OpenRouter supports reasoning.max_tokens.
                # Map effort -> max_tokens budget when max_tokens isn't explicitly provided.
                if ("anthropic/" in m or "claude" in m or "gemini" in m) and "max_tokens" not in reasoning_payload:
                    eff = (reasoning_payload.get("effort") or "medium")
                    budget = _compute_openrouter_reasoning_max_tokens(max_tokens=max_tokens, effort=str(eff))
                    if budget > 0:
                        reasoning_payload.pop("effort", None)
                        reasoning_payload["max_tokens"] = budget

                # For DeepSeek models, OpenRouter docs emphasize enabling reasoning, rather than effort.
                if "deepseek/" in m or "deepseek" in m:
                    eff = str(reasoning_payload.get("effort") or "medium").lower().strip()
                    reasoning_payload.pop("effort", None)
                    if "enabled" not in reasoning_payload:
                        reasoning_payload["enabled"] = eff != "none"

                # OpenRouter's OpenAI-compatible endpoint expects this under extra_body.
                extra_body = model_kwargs.get("extra_body")
                if not isinstance(extra_body, dict):
                    extra_body = {}
                extra_body["reasoning"] = reasoning_payload
                model_kwargs["extra_body"] = extra_body

                logger.info(f"Using OpenRouter reasoning={reasoning_payload} for model {model}")
        
        # OpenRouter-specific headers
        default_headers = {
            "HTTP-Referer": site_url or "https://github.com/autoexcerpter",
            "X-Title": self.app_name,
        }
        
        # Initialize LangChain ChatOpenAI with OpenRouter endpoint
        # LangChain handles:
        # - Retry logic with exponential backoff (max_retries)
        # - Parameter filtering for unsupported models (disabled_params)
        self._llm = ChatOpenAI(  # type: ignore[call-arg]
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            base_url=OPENROUTER_BASE_URL,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            disabled_params=disabled_params,
            default_headers=default_headers,
            **model_kwargs,
        )
    
    def _build_disabled_params(self) -> Optional[Dict[str, Any]]:
        """Build disabled_params dict based on model capabilities."""
        caps = self._capabilities
        disabled: Dict[str, Any] = {}
        
        if not caps.supports_temperature:
            disabled["temperature"] = None
        if not caps.supports_top_p:
            disabled["top_p"] = None
        if not caps.supports_frequency_penalty:
            disabled["frequency_penalty"] = None
        if not caps.supports_presence_penalty:
            disabled["presence_penalty"] = None
        
        return disabled if disabled else None
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
    
    def get_capabilities(self) -> ProviderCapabilities:
        return self._capabilities
    
    async def transcribe_image(
        self,
        image_path: Path,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: Optional[str] = None,
        media_resolution: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe text from an image file."""
        base64_data, mime_type = self.encode_image_to_base64(image_path)
        return await self.transcribe_image_from_base64(
            image_base64=base64_data,
            mime_type=mime_type,
            system_prompt=system_prompt,
            user_instruction=user_instruction,
            json_schema=json_schema,
            image_detail=image_detail,
            media_resolution=media_resolution,
        )
    
    async def transcribe_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
        *,
        system_prompt: str,
        user_instruction: str = "Please transcribe the text from this image.",
        json_schema: Optional[Dict[str, Any]] = None,
        image_detail: Optional[str] = None,
        media_resolution: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe text from a base64-encoded image using LangChain."""
        caps = self._capabilities
        
        if not caps.supports_vision:
            return TranscriptionResult(
                content="",
                error=f"Model {self.model} does not support vision/image inputs.",
                transcription_not_possible=True,
            )
        
        # Normalize image detail
        detail = image_detail
        if detail:
            detail = detail.lower().strip()
            if detail not in ("low", "high"):
                detail = None
        if detail is None:
            detail = caps.default_image_detail if caps.supports_image_detail else None
        
        # Build data URL
        data_url = self.create_data_url(image_base64, mime_type)
        
        # Build message content with image (OpenAI format)
        image_content: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        if detail and caps.supports_image_detail:
            image_content["image_url"]["detail"] = detail
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]
        
        # Use structured output if schema provided and supported
        # Use include_raw=True to get token usage from the underlying AIMessage
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_output:
            # Unwrap schema if wrapped
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            
            # Note: Not all OpenRouter models support structured output
            # For those that do, use json_mode method as it's more widely supported
            try:
                llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                    actual_schema,
                    method="json_mode",
                    include_raw=True,
                )
            except Exception as e:
                logger.warning(
                    f"Structured output not available for {self.model}, "
                    f"falling back to standard output: {e}"
                )
        
        return await self._invoke_llm(llm_to_use, messages)
    
    async def _invoke_llm(
        self,
        llm,
        messages: List,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response.
        
        LangChain handles retry logic internally.
        
        When using with_structured_output(include_raw=True), the response is a dict:
        - "raw": The underlying AIMessage with response_metadata containing token usage
        - "parsed": The parsed dict
        - "parsing_error": Any parsing error that occurred
        """
        try:
            response = await llm.ainvoke(messages)
            
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            raw_response = {}
            raw_message = None
            parsed_output = None
            
            if isinstance(response, dict) and "raw" in response and "parsed" in response:
                raw_message = response.get("raw")
                parsed_data = response.get("parsed")
                
                if parsed_data is not None:
                    if hasattr(parsed_data, 'model_dump'):
                        content = parsed_data.model_dump_json()
                        parsed_output = parsed_data.model_dump()
                    elif isinstance(parsed_data, dict):
                        content = json.dumps(parsed_data)
                        parsed_output = parsed_data
                    else:
                        content = str(parsed_data)
                else:
                    content = raw_message.content if raw_message and hasattr(raw_message, 'content') else ""
                    if isinstance(content, dict):
                        parsed_output = content
                        content = json.dumps(content)
            elif hasattr(response, 'content'):
                raw_message = response
                content = response.content
                if isinstance(content, dict):
                    parsed_output = content
                    content = json.dumps(content)
                elif not isinstance(content, str):
                    content = str(content)
            elif isinstance(response, dict):
                content = json.dumps(response)
                parsed_output = response
            else:
                content = str(response)
            
            # Extract token usage from response_metadata
            if raw_message and hasattr(raw_message, 'response_metadata'):
                metadata = raw_message.response_metadata
                if isinstance(metadata, dict):
                    raw_response = metadata
                    usage = metadata.get('token_usage', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('prompt_tokens', 0)
                        output_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)
            
            # Track tokens
            if total_tokens > 0:
                try:
                    from modules.token_tracker import get_token_tracker
                    token_tracker = get_token_tracker()
                    token_tracker.add_tokens(total_tokens)
                    logger.debug(f"[TOKEN] API call consumed {total_tokens:,} tokens")
                except Exception as e:
                    logger.warning(f"Error tracking tokens: {e}")
            
            result = TranscriptionResult(
                content=content,
                raw_response=raw_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            
            if parsed_output and isinstance(parsed_output, dict):
                result.parsed_output = parsed_output
                result.no_transcribable_text = parsed_output.get('no_transcribable_text', False)
                result.transcription_not_possible = parsed_output.get('transcription_not_possible', False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error invoking OpenRouter: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
