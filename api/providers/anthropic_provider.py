"""Anthropic (Claude) provider implementation using LangChain.

Supports Claude model families:
- Claude 4.5: Opus, Sonnet, Haiku (with extended thinking on Opus/Sonnet)
- Claude 4.1: Opus
- Claude 4: Opus, Sonnet
- Claude 3.5/3.7: Sonnet, Haiku
- Claude 3: Opus, Sonnet, Haiku

LangChain handles:
- Retry logic with exponential backoff (max_retries parameter)
- Token usage tracking (response_metadata)
- Structured output parsing (with_structured_output)
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from api.providers.base import (
    BaseProvider,
    ProviderCapabilities,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


def _transform_schema_for_anthropic(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Transform JSON schema to be Anthropic-compatible.
    
    Anthropic's SDK doesn't support union types like ["string", "null"].
    This function converts them to simple types.
    """
    result = copy.deepcopy(schema)
    
    def transform_type(obj: Dict[str, Any]) -> None:
        if not isinstance(obj, dict):
            return
            
        # Handle union types like ["string", "null"]
        if "type" in obj and isinstance(obj["type"], list):
            non_null_types = [t for t in obj["type"] if t != "null"]
            if non_null_types:
                obj["type"] = non_null_types[0]
            else:
                obj["type"] = "string"
        
        # Recursively handle properties
        if "properties" in obj and isinstance(obj["properties"], dict):
            for prop in obj["properties"].values():
                transform_type(prop)
        
        # Handle items in arrays
        if "items" in obj and isinstance(obj["items"], dict):
            transform_type(obj["items"])
        
        # Handle anyOf/oneOf/allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    transform_type(item)
    
    transform_type(result)
    
    # Add required top-level keys for LangChain/Anthropic compatibility
    if "title" not in result:
        result["title"] = "TranscriptionSchema"
    if "description" not in result:
        result["description"] = "Schema for document transcription output"
    
    return result


def _get_model_capabilities(model_name: str) -> ProviderCapabilities:
    """Determine capabilities based on Anthropic model name.
    
    Supports (as of late 2025):
    - Claude 4.5: claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5
    - Claude 4.1: claude-opus-4-1
    - Claude 4: claude-sonnet-4, claude-opus-4
    - Claude 3.5/3.7: claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku
    - Claude 3: claude-3-opus, claude-3-sonnet, claude-3-haiku
    """
    m = model_name.lower().strip()
    
    # Claude 4.5 Opus (most capable, extended thinking)
    if "claude-opus-4-5" in m or "claude-opus-4.5" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,  # Claude 4.5 doesn't allow both temp AND top_p
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=32768,
        )
    
    # Claude 4.5 Sonnet (balanced, extended thinking)
    if "claude-sonnet-4-5" in m or "claude-sonnet-4.5" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 4.5 Haiku (fastest, NO structured output)
    if "claude-haiku-4-5" in m or "claude-haiku-4.5" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=False,
            supports_json_mode=False,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 4.6 Opus (native extended thinking, 200k context)
    if "claude-opus-4-6" in m or "claude-opus-4.6" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=32768,
        )
    
    # Claude 4.6 Sonnet (extended thinking via beta header, 200k context)
    if "claude-sonnet-4-6" in m or "claude-sonnet-4.6" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=False,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 4.1 Opus
    if "claude-opus-4-1" in m or "claude-opus-4.1" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 4 Sonnet
    if "claude-sonnet-4" in m and "4-5" not in m and "4.5" not in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
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
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 4 Opus
    if "claude-opus-4" in m and "4-1" not in m and "4.1" not in m and "4-5" not in m and "4.5" not in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
            supports_image_detail=False,
            default_image_detail="auto",
            supports_structured_output=True,
            supports_json_mode=True,
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_frequency_penalty=False,
            supports_presence_penalty=False,
            max_context_tokens=200000,
            max_output_tokens=16384,
        )
    
    # Claude 3.7/3.5 Sonnet
    if "claude-3-7-sonnet" in m or "claude-3-5-sonnet" in m or "claude-3.5-sonnet" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
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
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 3.5 Haiku
    if "claude-3-5-haiku" in m or "claude-3.5-haiku" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
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
            max_context_tokens=200000,
            max_output_tokens=8192,
        )
    
    # Claude 3 Opus
    if "claude-3-opus" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
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
            max_context_tokens=200000,
            max_output_tokens=4096,
        )
    
    # Claude 3 Sonnet/Haiku
    if "claude-3-sonnet" in m or "claude-3-haiku" in m:
        return ProviderCapabilities(
            provider_name="anthropic",
            model_name=model_name,
            supports_vision=True,
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
            max_context_tokens=200000,
            max_output_tokens=4096,
        )
    
    # Default/fallback for Claude models
    return ProviderCapabilities(
        provider_name="anthropic",
        model_name=model_name,
        supports_vision=True,
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
        max_context_tokens=200000,
        max_output_tokens=8192,
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


class AnthropicProvider(BaseProvider):
    """Anthropic (Claude) LLM provider using LangChain.
    
    Features:
    - Extended thinking support for Claude 4.5/4.1 (maps reasoning.effort to budget_tokens)
    - Automatic schema transformation for Anthropic compatibility
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
        top_k: Optional[int] = None,
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
        self.top_k = top_k
        self.reasoning_config = reasoning_config
        
        self._capabilities = _get_model_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build LangChain model kwargs
        model_kwargs: Dict[str, Any] = {}
        if self._capabilities.supports_temperature:
            model_kwargs["temperature"] = temperature
        if self._capabilities.supports_top_p:
            model_kwargs["top_p"] = top_p
        if top_k is not None:
            model_kwargs["top_k"] = top_k
        
        # Apply extended thinking for Claude 4.5+ models
        # Maps reasoning_config.effort to Anthropic's thinking parameter
        if self._capabilities.supports_reasoning_effort and reasoning_config:
            effort = reasoning_config.get("effort", "medium")
            # Map effort levels to thinking budget tokens
            effort_to_budget = {
                "low": 1024,
                "medium": 4096,
                "high": 16384,
            }
            budget = effort_to_budget.get(effort, 4096)
            model_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            logger.info(f"Using extended thinking (budget={budget}) for model {model}")
        
        # Initialize LangChain ChatAnthropic
        self._llm = ChatAnthropic(  # type: ignore[call-arg]
            api_key=api_key,  # type: ignore[arg-type]
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **model_kwargs,
        )
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
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
        
        # Anthropic uses a different image format
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_base64,
            },
        }
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]
        
        # Use structured output if schema provided
        # For Anthropic, use default method (tool-based) for reliable structured output
        # method="json_mode" just prompts for JSON and can return markdown-wrapped responses
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_output:
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            
            # Transform schema for Anthropic compatibility
            actual_schema = _transform_schema_for_anthropic(actual_schema)
            
            # Use default method (tool-based) for guaranteed structured output
            llm_to_use = self._llm.with_structured_output(  # type: ignore[assignment]
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
                    if isinstance(parsed_data, dict):
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
                    usage = metadata.get('usage', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        total_tokens = input_tokens + output_tokens
            
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
            logger.error(f"Error invoking Anthropic: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
