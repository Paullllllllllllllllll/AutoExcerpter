"""OpenRouter provider implementation using LangChain.

OpenRouter provides access to 200+ models from various providers through a unified API:
- OpenAI models (GPT-5, GPT-4o, o-series)
- Anthropic models (Claude family)
- Google models (Gemini family)
- Meta models (Llama)
- Mistral models
- DeepSeek models
- And many more

Uses OpenAI-compatible API with custom base URL and headers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from api.providers.base import (
    BaseProvider,
    ProviderCapabilities,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on OpenRouter model name.
    
    OpenRouter model format: provider/model (e.g., "anthropic/claude-sonnet-4-5")
    
    Capabilities are inferred from the model name prefix/pattern.
    """
    m = model_name.lower().strip()
    
    # Extract the actual model name if in provider/model format
    if "/" in m:
        provider_prefix, model_part = m.split("/", 1)
    else:
        provider_prefix = ""
        model_part = m
    
    # OpenAI models via OpenRouter
    if provider_prefix == "openai" or model_part.startswith(("gpt-", "o1", "o3", "o4")):
        # GPT-5.1/5 family
        if "gpt-5" in model_part:
            return ProviderCapabilities(
                provider_name="openrouter",
                model_name=model_name,
                supports_vision=True,
                supports_image_detail=True,
                default_image_detail="high",
                supports_structured_output=True,
                supports_json_mode=True,
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                supports_temperature=False,
                supports_top_p=False,
                supports_frequency_penalty=False,
                supports_presence_penalty=False,
                max_context_tokens=256000,
                max_output_tokens=128000,
            )
        # o-series
        if model_part.startswith(("o1", "o3", "o4")):
            supports_vision = not ("mini" in model_part and ("o1" in model_part or "o3" in model_part))
            return ProviderCapabilities(
                provider_name="openrouter",
                model_name=model_name,
                supports_vision=supports_vision,
                supports_image_detail=supports_vision,
                default_image_detail="high",
                supports_structured_output=True,
                supports_json_mode=True,
                is_reasoning_model=True,
                supports_reasoning_effort=True,
                supports_temperature=False,
                supports_top_p=False,
                supports_frequency_penalty=False,
                supports_presence_penalty=False,
                max_context_tokens=200000,
                max_output_tokens=100000,
            )
        # GPT-4o/4.1 family
        if "gpt-4" in model_part:
            return ProviderCapabilities(
                provider_name="openrouter",
                model_name=model_name,
                supports_vision=True,
                supports_image_detail=True,
                default_image_detail="high",
                supports_structured_output=True,
                supports_json_mode=True,
                is_reasoning_model=False,
                supports_reasoning_effort=False,
                supports_temperature=True,
                supports_top_p=True,
                supports_frequency_penalty=True,
                supports_presence_penalty=True,
                max_context_tokens=128000,
                max_output_tokens=16384,
            )
    
    # Anthropic models via OpenRouter
    if provider_prefix == "anthropic" or "claude" in model_part:
        is_45 = "4-5" in model_part or "4.5" in model_part
        is_haiku = "haiku" in model_part
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=not (is_45 and is_haiku),
            supports_json_mode=not (is_45 and is_haiku),
            is_reasoning_model=is_45,
            supports_reasoning_effort=is_45,
            supports_temperature=True,
            supports_top_p=not is_45,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384 if not is_haiku else 8192,
        )
    
    # Google models via OpenRouter
    if provider_prefix == "google" or "gemini" in model_part:
        is_thinking = "2.5" in model_part or "3" in model_part
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_media_resolution=True,
            default_media_resolution="high",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=is_thinking,
            supports_reasoning_effort=is_thinking,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=1000000,
            max_output_tokens=32768,
        )
    
    # DeepSeek models
    if provider_prefix == "deepseek" or "deepseek" in model_part:
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision="v3" not in model_part,  # V3 is text-only
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model="r1" in model_part,
            supports_reasoning_effort="r1" in model_part,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=128000,
            max_output_tokens=8192,
        )
    
    # Meta Llama models
    if provider_prefix == "meta" or "llama" in model_part:
        has_vision = "vision" in model_part or "90b" in model_part
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision=has_vision,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=128000,
            max_output_tokens=8192,
        )
    
    # Mistral models
    if provider_prefix == "mistral" or "mistral" in model_part or "mixtral" in model_part:
        has_vision = "pixtral" in model_part
        return ProviderCapabilities(
            provider_name="openrouter",
            model_name=model_name,
            supports_vision=has_vision,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=False,
            supports_reasoning_effort=False,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=128000,
            max_output_tokens=8192,
        )
    
    # Default/fallback (conservative, assume basic capabilities)
    return ProviderCapabilities(
        provider_name="openrouter",
        model_name=model_name,
        supports_vision=True,
        supports_image_detail=True,
        default_image_detail="auto",
        supports_structured_output=True,
        supports_json_mode=True,
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_temperature=True,
        supports_top_p=True,
        supports_frequency_penalty=True,
        supports_presence_penalty=True,
        max_context_tokens=128000,
        max_output_tokens=4096,
    )


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
    """OpenRouter LLM provider using LangChain (OpenAI-compatible API).
    
    Features:
    - Access to 200+ models from various providers
    - OpenAI-compatible API interface
    - Automatic capability detection based on model name
    - LangChain handles retry logic with exponential backoff
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
        self.reasoning_config = reasoning_config
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build LangChain kwargs
        llm_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "base_url": OPENROUTER_BASE_URL,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        
        # OpenRouter recommends including site info in headers
        default_headers = {
            "HTTP-Referer": "https://github.com/autoexcerpter",
            "X-Title": "AutoExcerpter",
        }
        llm_kwargs["default_headers"] = default_headers
        
        # Handle reasoning vs non-reasoning models
        caps = self._capabilities
        if caps.is_reasoning_model:
            llm_kwargs["max_completion_tokens"] = max_tokens
            
            # Pass reasoning effort if supported
            if caps.supports_reasoning_effort and reasoning_config:
                effort = reasoning_config.get("effort")
                if effort:
                    llm_kwargs["reasoning_effort"] = effort
                    logger.info(f"Using reasoning_effort={effort} for model {model}")
        else:
            llm_kwargs["max_tokens"] = max_tokens
            if caps.supports_temperature:
                llm_kwargs["temperature"] = temperature
            if caps.supports_top_p:
                llm_kwargs["top_p"] = top_p
            if caps.supports_frequency_penalty:
                llm_kwargs["frequency_penalty"] = frequency_penalty
            if caps.supports_presence_penalty:
                llm_kwargs["presence_penalty"] = presence_penalty
        
        self._llm = ChatOpenAI(**llm_kwargs)
    
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
        # OpenRouter routes to many providers - use default tool-based method for reliability
        # json_schema with strict=True only works for OpenAI models
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_output:
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            llm_to_use = self._llm.with_structured_output(
                actual_schema,
                include_raw=True,
            )
        
        return await self._invoke_llm(llm_to_use, messages)
    
    async def _invoke_llm(
        self,
        llm,
        messages: List,
    ) -> TranscriptionResult:
        """Invoke the LLM and process the response."""
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
