"""Google Gemini provider implementation using LangChain.

Supports Gemini models:
- Gemini 3: Pro, Flash (state-of-the-art reasoning)
- Gemini 2.5: Pro, Flash (adaptive thinking)
- Gemini 2.0: Flash
- Gemini 1.5: Pro, Flash

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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from api.model_capabilities import detect_capabilities
from api.providers.base import (
    BaseProvider,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


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


class GoogleProvider(BaseProvider):
    """Google Gemini LLM provider using LangChain.
    
    Features:
    - Thinking mode support for Gemini 2.5+/3 (maps reasoning.effort to thinking_level)
    - Media resolution control for image inputs
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
        
        self._capabilities = detect_capabilities(model)
        max_retries = _load_max_retries()
        
        # Build LLM kwargs
        llm_kwargs: Dict[str, Any] = {
            "google_api_key": api_key,
            "model": model,
            "max_output_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        
        if self._capabilities.supports_temperature:
            llm_kwargs["temperature"] = temperature
        if self._capabilities.supports_top_p:
            llm_kwargs["top_p"] = top_p
        if top_k is not None:
            llm_kwargs["top_k"] = top_k
        
        # Apply thinking mode for Gemini 2.5+/3 models
        # Maps reasoning_config.effort to Google's thinking_level parameter
        if self._capabilities.supports_reasoning_effort and reasoning_config:
            effort = reasoning_config.get("effort", "medium")
            # Map effort levels to Gemini thinking_level
            if effort == "low":
                llm_kwargs["thinking_level"] = "low"
            else:
                # medium and high both map to "high" thinking
                llm_kwargs["thinking_level"] = "high"
            logger.info(f"Using thinking_level={llm_kwargs['thinking_level']} for model {model}")
        
        self._llm = ChatGoogleGenerativeAI(**llm_kwargs)
    
    @property
    def provider_name(self) -> str:
        return "google"
    
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
        
        # Build data URL for Gemini
        data_url = self.create_data_url(image_base64, mime_type)
        
        # Gemini uses standard image_url format
        image_content = {
            "type": "image_url",
            "image_url": data_url,
        }
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                image_content,
            ]),
        ]
        
        # Use structured output if schema provided
        # For Google Gemini, use default method (tool-based) for reliable structured output
        # method="json_schema" may not be properly supported and can return markdown-wrapped responses
        llm_to_use = self._llm
        if json_schema and caps.supports_structured_output:
            if isinstance(json_schema, dict) and "schema" in json_schema:
                actual_schema = json_schema["schema"]
            else:
                actual_schema = json_schema
            
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
                    elif isinstance(content, list):
                        # Gemini can return content as list of parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = "".join(text_parts)
            elif hasattr(response, 'content'):
                raw_message = response
                content = response.content
                if isinstance(content, dict):
                    parsed_output = content
                    content = json.dumps(content)
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "".join(text_parts)
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
                    usage = metadata.get('usage_metadata', {})
                    if isinstance(usage, dict):
                        input_tokens = usage.get('prompt_token_count', 0)
                        output_tokens = usage.get('candidates_token_count', 0)
                        total_tokens = usage.get('total_token_count', 0)
                        if total_tokens == 0:
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
            logger.error(f"Error invoking Google Gemini: {e}")
            return TranscriptionResult(
                content="",
                error=str(e),
                transcription_not_possible=True,
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        pass
