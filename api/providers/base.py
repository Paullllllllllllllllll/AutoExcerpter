"""Base provider abstraction for LLM integrations.

Defines the common interface that all LLM providers must implement.
Adapted from ChronoTranscriber's multi-provider architecture.
"""

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)

# Supported image formats and their MIME types
SUPPORTED_IMAGE_FORMATS: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

ImageDetail = Literal["auto", "high", "low"]
MediaResolution = Literal["low", "medium", "high", "ultra_high", "auto"]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Describes capabilities of an LLM provider/model combination.
    
    This enables parameter guarding - filtering out unsupported parameters
    before they're sent to the API, preventing errors like:
    "Unsupported parameter: 'reasoning_effort' is not supported with this model"
    
    Attributes:
        provider_name: Name of the provider (openai, anthropic, google, openrouter)
        model_name: Full model identifier
        supports_vision: Whether the model can process image inputs
        supports_image_detail: Whether OpenAI-style "detail" parameter is supported
        default_image_detail: Default detail level for images
        supports_media_resolution: Whether Google-style media_resolution is supported
        default_media_resolution: Default resolution for Google
        supports_structured_output: Whether native structured output is supported
        supports_json_mode: Whether JSON mode is available
        is_reasoning_model: Whether this is a reasoning-capable model family
        supports_reasoning_effort: Whether reasoning_effort/thinking parameters work
        supports_temperature: Whether temperature sampling is supported
        supports_top_p: Whether top_p sampling is supported
        supports_frequency_penalty: Whether frequency_penalty is supported
        supports_presence_penalty: Whether presence_penalty is supported
        supports_streaming: Whether streaming responses are supported
        max_context_tokens: Maximum input context tokens
        max_output_tokens: Maximum output tokens
    """
    
    provider_name: str
    model_name: str
    
    # Vision/multimodal
    supports_vision: bool = False
    supports_image_detail: bool = True  # OpenAI-style "detail" parameter
    default_image_detail: ImageDetail = "high"
    supports_media_resolution: bool = False  # Google-style media_resolution
    default_media_resolution: MediaResolution = "high"
    
    # Structured outputs
    supports_structured_output: bool = False
    supports_json_mode: bool = False
    
    # Reasoning models
    is_reasoning_model: bool = False
    supports_reasoning_effort: bool = False
    
    # Sampler controls
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_frequency_penalty: bool = True
    supports_presence_penalty: bool = True
    
    # Streaming
    supports_streaming: bool = True
    
    # Context window
    max_context_tokens: int = 128000
    max_output_tokens: int = 4096


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.
    
    Provides a standardized result format across all providers.
    
    Attributes:
        content: The transcription text content
        raw_response: Raw metadata from the API response
        parsed_output: Parsed structured output (if schema was provided)
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens used
        no_transcribable_text: Schema flag indicating no text found
        transcription_not_possible: Schema flag indicating transcription failed
        error: Error message if transcription failed
    """
    
    # Core result
    content: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    # Parsed structured output (if schema was provided)
    parsed_output: Optional[Dict[str, Any]] = None
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Transcription status flags (from schema response)
    no_transcribable_text: bool = False
    transcription_not_possible: bool = False
    
    # Error information
    error: Optional[str] = None
    
    def __post_init__(self):
        """Parse transcription status flags from content if available."""
        if self.content and not self.parsed_output:
            try:
                stripped = self.content.strip()
                if stripped.startswith("{"):
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        self.parsed_output = parsed
                        self.no_transcribable_text = parsed.get("no_transcribable_text", False)
                        self.transcription_not_possible = parsed.get("transcription_not_possible", False)
            except json.JSONDecodeError:
                pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            "output_text": self.content,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
        }
        
        if self.raw_response:
            result["metadata"] = self.raw_response
        
        if self.parsed_output:
            result["parsed"] = self.parsed_output
        
        if self.error:
            result["error"] = self.error
        
        return result


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.
    
    All providers must implement these methods to work with the transcription pipeline.
    This provides a unified interface regardless of the underlying LLM provider.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the provider.
        
        Args:
            api_key: API key for the provider
            model: Model name/identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            **kwargs: Provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_config = kwargs
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider/model combination."""
        pass
    
    @abstractmethod
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
        """Transcribe text from an image.
        
        Args:
            image_path: Path to the image file
            system_prompt: System prompt for the model
            user_instruction: User instruction text
            json_schema: Optional JSON schema for structured output
            image_detail: Image detail level for OpenAI ("low", "high", "auto")
            media_resolution: Media resolution for Google
        
        Returns:
            TranscriptionResult with the transcription and metadata
        """
        pass
    
    @abstractmethod
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
        """Transcribe text from a base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image data
            mime_type: MIME type of the image (e.g., "image/jpeg")
            system_prompt: System prompt for the model
            user_instruction: User instruction text
            json_schema: Optional JSON schema for structured output
            image_detail: Image detail level for OpenAI
            media_resolution: Media resolution for Google
        
        Returns:
            TranscriptionResult with the transcription and metadata
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g., HTTP sessions)."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False
    
    @staticmethod
    def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
        """Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Tuple of (base64_data, mime_type)
        
        Raises:
            ValueError: If the image format is not supported
        """
        ext = image_path.suffix.lower()
        mime_type = SUPPORTED_IMAGE_FORMATS.get(ext)
        if not mime_type:
            raise ValueError(f"Unsupported image format: {ext}")
        
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return data, mime_type
    
    @staticmethod
    def create_data_url(base64_data: str, mime_type: str) -> str:
        """Create a data URL from base64 data.
        
        Args:
            base64_data: Base64-encoded image data
            mime_type: MIME type of the image
        
        Returns:
            Data URL string
        """
        return f"data:{mime_type};base64,{base64_data}"


class CapabilityError(ValueError):
    """Raised when a selected model is incompatible with the configured pipeline."""
    pass


def ensure_image_support(model_name: str, capabilities: ProviderCapabilities) -> None:
    """Fail fast if the model doesn't support image inputs.
    
    Args:
        model_name: Selected model id/alias
        capabilities: Model capabilities
        
    Raises:
        CapabilityError: If model doesn't support images
    """
    if not capabilities.supports_vision:
        raise CapabilityError(
            f"Model '{model_name}' does not support image inputs. "
            "Choose an image-capable model (e.g., gpt-5, gpt-4o, claude, gemini) "
            "or use a text-only flow."
        )
