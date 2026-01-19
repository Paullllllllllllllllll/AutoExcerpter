"""OpenAI provider implementation using LangChain.

Supports all OpenAI models including:
- GPT-5.1, GPT-5 family (with reasoning controls)
- GPT-4.1, GPT-4o family
- o1, o3, o4 reasoning models

LangChain handles:
- Retry logic with exponential backoff (max_retries parameter)
- Token usage tracking (response_metadata)
- Structured output parsing (with_structured_output)
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


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on OpenAI model name.
    
    Supports (as of late 2025):
    - GPT-5.1: gpt-5.1, gpt-5.1-mini, gpt-5.1-nano, gpt-5.1-instant, gpt-5.1-thinking
    - GPT-5: gpt-5, gpt-5-mini, gpt-5-nano
    - o4: o4, o4-mini
    - o3: o3, o3-pro, o3-mini
    - o1: o1, o1-pro, o1-mini
    - GPT-4.1: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
    - GPT-4o: gpt-4o, gpt-4o-mini
    """
    m = model_name.lower().strip()
    
    # GPT-5.1 family (newest, with adaptive thinking)
    if m.startswith("gpt-5.1"):
        return ProviderCapabilities(
            provider_name="openai",
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
    
    # GPT-5 family
    if m.startswith("gpt-5"):
        return ProviderCapabilities(
            provider_name="openai",
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
    
    # o4 family
    if m.startswith("o4"):
        return ProviderCapabilities(
            provider_name="openai",
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
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o3-pro
    if m.startswith("o3-pro"):
        return ProviderCapabilities(
            provider_name="openai",
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
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o3-mini (NO vision)
    if m.startswith("o3-mini"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=False,
            supports_image_detail=False,
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
    
    # o3 (not o3-mini, not o3-pro)
    if m == "o3" or m.startswith("o3-20"):
        return ProviderCapabilities(
            provider_name="openai",
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
            max_context_tokens=200000,
            max_output_tokens=100000,
        )
    
    # o1-pro
    if m.startswith("o1-pro"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=False,
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
    
    # o1-mini (NO vision)
    if m.startswith("o1-mini"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=False,
            supports_image_detail=False,
            default_image_detail="high",
            supports_structured_output=False,
            supports_json_mode=False,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=128000,
            max_output_tokens=65536,
        )
    
    # o1 (not o1-mini, not o1-pro)
    if m == "o1" or m.startswith("o1-20"):
        return ProviderCapabilities(
            provider_name="openai",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=True,
            default_image_detail="high",
            supports_structured_output=False,
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
    
    # GPT-4o family (multimodal workhorse)
    if m.startswith("gpt-4o"):
        return ProviderCapabilities(
            provider_name="openai",
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
    
    # GPT-4.1 family (million token context)
    if m.startswith("gpt-4.1"):
        return ProviderCapabilities(
            provider_name="openai",
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
            max_context_tokens=1000000,
            max_output_tokens=32768,
        )
    
    # Default/fallback (conservative)
    return ProviderCapabilities(
        provider_name="openai",
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


class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider using LangChain.
    
    LangChain handles:
    - Automatic retry with exponential backoff (via max_retries)
    - Token usage tracking (via response_metadata)
    - Structured output parsing (via with_structured_output)
    - Parameter filtering for unsupported models (via disabled_params)
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
        service_tier: Optional[str] = None,
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
        self.service_tier = service_tier
        self.reasoning_config = reasoning_config
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build disabled_params for models that don't support certain features
        disabled_params = self._build_disabled_params()
        
        # Initialize LangChain ChatOpenAI
        llm_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        
        if disabled_params:
            llm_kwargs["disabled_params"] = disabled_params
        
        # For reasoning models, use max_completion_tokens and skip sampler params
        caps = self._capabilities
        if caps.is_reasoning_model:
            llm_kwargs["max_completion_tokens"] = max_tokens
            logger.info(f"Using max_completion_tokens={max_tokens} for reasoning model {model}")
            
            # Apply reasoning controls for models that support them
            if caps.supports_reasoning_effort and reasoning_config:
                effort = reasoning_config.get("effort")
                if effort:
                    llm_kwargs["reasoning_effort"] = effort
                    logger.info(f"Using reasoning_effort={effort} for model {model}")
        else:
            llm_kwargs["max_tokens"] = max_tokens
            llm_kwargs["temperature"] = temperature
            llm_kwargs["top_p"] = top_p
            llm_kwargs["frequency_penalty"] = frequency_penalty
            llm_kwargs["presence_penalty"] = presence_penalty
        
        # Add service tier if specified
        if service_tier:
            llm_kwargs["service_tier"] = service_tier
        
        self._llm = ChatOpenAI(**llm_kwargs)
    
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
        return "openai"
    
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
        
        # Build message content with image
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
        
        # Use LangChain's structured output if schema provided and supported
        llm_to_use = self._llm
        
        if json_schema and caps.supports_structured_output:
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
                actual_schema,
                method="json_schema",
                strict=True,
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
            
            # Extract token usage and content
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
            logger.error(f"Error invoking OpenAI: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
