"""Unified model capabilities detection and gating.

This module is the single source of truth for model capability detection across
all providers (OpenAI, Anthropic, Google, OpenRouter). It uses a registry pattern
with provider-level base defaults and per-model overrides for concise, maintainable
capability definitions.

Architecture:
=============
1. Provider base dicts define the typical capability profile for each provider class.
2. _MODEL_REGISTRY entries specify only the delta from the provider base.
3. detect_capabilities() walks the registry and returns a typed ProviderCapabilities.
4. Adding a new model requires a single line in _MODEL_REGISTRY.

This replaces the previous scattered per-provider _get_model_capabilities() functions
and the MODEL_CAPABILITIES dict in llm_client.py.

LangChain Integration:
======================
LangChain does NOT automatically handle capability guarding (e.g., it will pass
temperature to reasoning models that don't support it, causing API errors).

The capabilities detected here are used by:
1. Provider classes — to set ``disabled_params`` for unsupported parameters
2. base_llm_client.py — to guard invocation kwargs
3. transcribe_api.py — to verify multimodal support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Type aliases
ImageDetail = Literal["auto", "high", "low"]
MediaResolution = Literal["low", "medium", "high", "ultra_high", "auto"]
ProviderType = Literal["openai", "anthropic", "google", "openrouter", "unknown"]


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Describes capabilities of an LLM provider/model combination.

    This enables parameter guarding - filtering out unsupported parameters
    before they're sent to the API, preventing errors like:
    "Unsupported parameter: 'reasoning_effort' is not supported with this model"

    Attributes:
        provider_name: Name of the provider (openai, anthropic, google, openrouter)
        model_name: Full model identifier
        family: Model family identifier (e.g., "gpt-5", "claude-opus-4.5")
        supports_vision: Whether the model can process image inputs
        supports_image_detail: Whether OpenAI-style "detail" parameter is supported
        default_image_detail: Default detail level for images
        supports_media_resolution: Whether Google-style media_resolution is supported
        default_media_resolution: Default resolution for Google
        supports_structured_output: Whether native structured output is supported
        supports_json_mode: Whether JSON mode is available
        is_reasoning_model: Whether this is a reasoning-capable model family
        supports_reasoning_effort: Whether reasoning_effort/thinking parameters work
        supports_text_verbosity: Whether text verbosity parameters work (GPT-5 family)
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
    family: str = "unknown"

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

    # Text verbosity (GPT-5 family only)
    supports_text_verbosity: bool = False

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


def _norm(name: str) -> str:
    """Normalize model name for matching."""
    return name.strip().lower()


def detect_provider(model_name: str) -> ProviderType:
    """Detect LLM provider from model name.

    This is the canonical provider detection function.  All other modules
    should use this function or delegate to it.

    Args:
        model_name: The model identifier string.

    Returns:
        ProviderType: "openai", "anthropic", "google", "openrouter", or "unknown"
    """
    m = _norm(model_name)

    # OpenRouter models (contain / separator)
    if "/" in m:
        return "openrouter"

    # Anthropic models
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"

    # Google models
    if m.startswith("gemini") or "google" in m:
        return "google"

    # OpenAI models
    if any(m.startswith(p) for p in ("gpt", "o1", "o3", "o4", "text-", "chatgpt")):
        return "openai"

    return "unknown"


# ---------------------------------------------------------------------------
# Provider-level capability defaults.  Each model entry in the registry below
# only needs to declare the fields that *differ* from its provider default.
# ---------------------------------------------------------------------------

_OPENAI_REASONING_BASE: dict = dict(
    provider_name="openai",
    supports_vision=True,
    supports_image_detail=True,
    default_image_detail="high",
    supports_structured_output=True,
    supports_json_mode=True,
    is_reasoning_model=True,
    supports_reasoning_effort=True,
    supports_text_verbosity=False,
    supports_temperature=False,
    supports_top_p=False,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=200000,
    max_output_tokens=100000,
)

_OPENAI_STANDARD_BASE: dict = dict(
    provider_name="openai",
    supports_vision=True,
    supports_image_detail=True,
    default_image_detail="high",
    supports_structured_output=True,
    supports_json_mode=True,
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_text_verbosity=False,
    supports_temperature=True,
    supports_top_p=True,
    supports_frequency_penalty=True,
    supports_presence_penalty=True,
    max_context_tokens=128000,
    max_output_tokens=16384,
)

_ANTHROPIC_BASE: dict = dict(
    provider_name="anthropic",
    supports_vision=True,
    supports_image_detail=False,
    default_image_detail="auto",
    supports_structured_output=True,
    supports_json_mode=True,
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_text_verbosity=False,
    supports_temperature=True,
    supports_top_p=True,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=200000,
    max_output_tokens=8192,
)

_GOOGLE_BASE: dict = dict(
    provider_name="google",
    supports_vision=True,
    supports_image_detail=False,
    default_image_detail="auto",
    supports_media_resolution=True,
    default_media_resolution="high",
    supports_structured_output=True,
    supports_json_mode=True,
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_text_verbosity=False,
    supports_temperature=True,
    supports_top_p=True,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=1000000,
    max_output_tokens=8192,
)

_OPENROUTER_BASE: dict = dict(
    provider_name="openrouter",
    supports_vision=True,
    supports_image_detail=False,
    default_image_detail="auto",
    supports_structured_output=True,
    supports_json_mode=True,
    is_reasoning_model=False,
    supports_reasoning_effort=False,
    supports_text_verbosity=False,
    supports_temperature=True,
    supports_top_p=True,
    supports_frequency_penalty=False,
    supports_presence_penalty=False,
    max_context_tokens=128000,
    max_output_tokens=4096,
)

# ---------------------------------------------------------------------------
# Static model registry.  Each entry is (prefixes, family, base, overrides).
# Order matters: more specific prefixes MUST come before less specific ones.
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: list[tuple[tuple[str, ...], str, dict, dict]] = [
    # ===== OpenAI GPT-5.x family (reasoning + text verbosity) =====
    (("gpt-5.2",), "gpt-5.2", _OPENAI_REASONING_BASE, dict(
        supports_text_verbosity=True, max_context_tokens=400000, max_output_tokens=128000,
    )),
    (("gpt-5.1",), "gpt-5.1", _OPENAI_REASONING_BASE, dict(
        supports_text_verbosity=True, max_context_tokens=256000, max_output_tokens=128000,
    )),
    (("gpt-5",), "gpt-5", _OPENAI_REASONING_BASE, dict(
        supports_text_verbosity=True, max_context_tokens=256000, max_output_tokens=128000,
    )),
    # ===== OpenAI o-series reasoning models =====
    (("o4",), "o4", _OPENAI_REASONING_BASE, {}),
    (("o3-pro",), "o3-pro", _OPENAI_REASONING_BASE, {}),
    (("o3-mini",), "o3-mini", _OPENAI_REASONING_BASE, dict(
        supports_vision=False, supports_image_detail=False,
    )),
    # o3 (not o3-mini, not o3-pro) — handled below in detect_capabilities()
    (("o1-mini",), "o1-mini", _OPENAI_REASONING_BASE, dict(
        supports_vision=False, supports_image_detail=False,
        supports_structured_output=False, supports_json_mode=False,
        max_context_tokens=128000, max_output_tokens=65536,
    )),
    # o1-pro and o1 — handled below in detect_capabilities()
    # ===== OpenAI GPT-4.x standard models =====
    (("gpt-4o",), "gpt-4o", _OPENAI_STANDARD_BASE, {}),
    (("gpt-4.1",), "gpt-4.1", _OPENAI_STANDARD_BASE, dict(
        max_context_tokens=1000000, max_output_tokens=32768,
    )),
    (("gpt-4-turbo",), "gpt-4-turbo", _OPENAI_STANDARD_BASE, dict(
        max_output_tokens=4096,
    )),
    (("gpt-4",), "gpt-4", _OPENAI_STANDARD_BASE, dict(
        supports_vision=False, max_output_tokens=4096,
    )),
    # ===== Anthropic Claude models (most-specific first) =====
    (("claude-opus-4-6", "claude-opus-4.6"), "claude-opus-4.6", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=32768,
    )),
    (("claude-sonnet-4-6", "claude-sonnet-4.6"), "claude-sonnet-4.6", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=16384,
    )),
    (("claude-opus-4-5", "claude-opus-4.5"), "claude-opus-4.5", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=32768,
    )),
    (("claude-sonnet-4-5", "claude-sonnet-4.5"), "claude-sonnet-4.5", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, max_output_tokens=16384,
    )),
    (("claude-haiku-4-5", "claude-haiku-4.5"), "claude-haiku-4.5", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        supports_top_p=False, supports_structured_output=False, supports_json_mode=False,
    )),
    (("claude-opus-4-1", "claude-opus-4.1"), "claude-opus-4.1", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=16384,
    )),
    (("claude-opus-4",), "claude-opus-4", _ANTHROPIC_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=16384,
    )),
    (("claude-sonnet-4",), "claude-sonnet-4", _ANTHROPIC_BASE, {}),
    (("claude-3-7-sonnet", "claude-3.7-sonnet"), "claude-3.7-sonnet", _ANTHROPIC_BASE, {}),
    (("claude-3-5-sonnet", "claude-3.5-sonnet"), "claude-3.5-sonnet", _ANTHROPIC_BASE, {}),
    (("claude-3-5-haiku", "claude-3.5-haiku"), "claude-3.5-haiku", _ANTHROPIC_BASE, {}),
    (("claude-3-opus",), "claude-3-opus", _ANTHROPIC_BASE, dict(max_output_tokens=4096)),
    (("claude-3-sonnet",), "claude-3-sonnet", _ANTHROPIC_BASE, dict(max_output_tokens=4096)),
    (("claude-3-haiku",), "claude-3-haiku", _ANTHROPIC_BASE, dict(max_output_tokens=4096)),
    (("claude",), "claude", _ANTHROPIC_BASE, {}),
    # ===== Google Gemini models (most-specific first) =====
    (("gemini-3-flash", "gemini-3.0-flash"), "gemini-3-flash", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=1048576, max_output_tokens=65536,
    )),
    (("gemini-3-pro", "gemini-3.0-pro"), "gemini-3-pro", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=2000000, max_output_tokens=65536,
    )),
    (("gemini-3",), "gemini-3", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=65536,
    )),
    (("gemini-2.5-pro", "gemini-2-5-pro"), "gemini-2.5-pro", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_context_tokens=2000000, max_output_tokens=65536,
    )),
    (("gemini-2.5-flash-lite", "gemini-2-5-flash-lite"), "gemini-2.5-flash-lite", _GOOGLE_BASE, dict(
        max_context_tokens=1048576,
    )),
    (("gemini-2.5-flash", "gemini-2-5-flash"), "gemini-2.5-flash", _GOOGLE_BASE, dict(
        is_reasoning_model=True, supports_reasoning_effort=True,
        max_output_tokens=32768,
    )),
    (("gemini-2.0", "gemini-2-0"), "gemini-2.0", _GOOGLE_BASE, {}),
    (("gemini-1.5-pro", "gemini-1-5-pro"), "gemini-1.5-pro", _GOOGLE_BASE, dict(
        max_context_tokens=2000000,
    )),
    (("gemini-1.5-flash", "gemini-1-5-flash"), "gemini-1.5-flash", _GOOGLE_BASE, {}),
    (("gemini",), "gemini", _GOOGLE_BASE, {}),
]


def _build_caps(model_name: str, family: str, base: dict, overrides: dict) -> ProviderCapabilities:
    """Merge *base* defaults with *overrides* and return a ProviderCapabilities instance."""
    merged = {**base, **overrides}
    merged["model_name"] = model_name
    merged["family"] = family
    return ProviderCapabilities(**merged)


def detect_capabilities(model_name: str) -> ProviderCapabilities:
    """Detect model capabilities from model name.

    This is the single entry point for capability detection.  It checks the
    static registry first, then handles special-case models that require
    negative-prefix logic, then OpenRouter dynamic matching.

    Args:
        model_name: Model identifier (e.g., "gpt-5-mini", "claude-sonnet-4-5-20250929")

    Returns:
        ProviderCapabilities with all capability flags set appropriately.
    """
    m = _norm(model_name)

    # --- Static registry lookup -------------------------------------------
    for prefixes, family, base, overrides in _MODEL_REGISTRY:
        if any(m.startswith(p) for p in prefixes):
            logger.debug(f"Model '{model_name}' matched registry family '{family}'")
            return _build_caps(model_name, family, base, overrides)

    # --- o3 (not o3-mini, not o3-pro) — requires negative-prefix logic ----
    if m == "o3" or (m.startswith("o3-") and not m.startswith("o3-mini") and not m.startswith("o3-pro")):
        return _build_caps(model_name, "o3", _OPENAI_REASONING_BASE, {})

    # --- o1-pro -----------------------------------------------------------
    if m.startswith("o1-pro"):
        return _build_caps(model_name, "o1-pro", _OPENAI_REASONING_BASE, dict(
            supports_structured_output=False,
        ))

    # --- o1 (not o1-mini, not o1-pro) -------------------------------------
    if m == "o1" or m.startswith("o1-20") or (m.startswith("o1") and not m.startswith("o1-mini") and not m.startswith("o1-pro")):
        return _build_caps(model_name, "o1", _OPENAI_REASONING_BASE, dict(
            supports_structured_output=False,
        ))

    # --- OpenRouter models (dynamic matching on underlying model) ---------
    if "/" in m:
        return _detect_openrouter_capabilities(model_name, m)

    # --- Fallback: conservative -------------------------------------------
    provider = detect_provider(model_name)
    logger.debug(f"Model '{model_name}' using fallback capability profile (provider={provider})")
    return ProviderCapabilities(
        provider_name=provider,
        model_name=model_name,
        family="unknown",
        supports_vision=False,
        supports_image_detail=False,
        default_image_detail="auto",
        supports_structured_output=False,
        supports_json_mode=False,
        is_reasoning_model=False,
        supports_reasoning_effort=False,
        supports_temperature=True,
        supports_top_p=True,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
    )


def _detect_openrouter_capabilities(model_name: str, m: str) -> ProviderCapabilities:
    """Detect capabilities for OpenRouter models based on underlying model name."""
    underlying = m.split("/")[-1] if "/" in m else m

    # DeepSeek via OpenRouter
    if "deepseek" in m:
        is_r1 = "deepseek-r1" in m or "r1" in m
        is_terminus = "terminus" in m
        return _build_caps(model_name, "openrouter-deepseek", _OPENROUTER_BASE, dict(
            is_reasoning_model=is_r1 or is_terminus,
            supports_reasoning_effort=is_r1 or is_terminus,
        ))

    # GPT-OSS via OpenRouter
    if "gpt-oss" in m:
        return _build_caps(model_name, "openrouter-gpt-oss", _OPENROUTER_BASE, dict(
            supports_image_detail=True,
            default_image_detail="high",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
        ))

    # GPT-5 via OpenRouter
    if "gpt-5" in m:
        return _build_caps(model_name, "openrouter-gpt5", _OPENROUTER_BASE, dict(
            supports_image_detail=True,
            default_image_detail="high",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
        ))

    # o-series via OpenRouter
    if any(x in m for x in ("/o1", "/o3", "/o4")):
        return _build_caps(model_name, "openrouter-o-series", _OPENROUTER_BASE, dict(
            supports_image_detail=True,
            default_image_detail="high",
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            supports_temperature=False,
            supports_top_p=False,
        ))

    # Other OpenAI models via OpenRouter
    if "openai/" in m or "gpt-4" in m:
        return _build_caps(model_name, "openrouter-openai", _OPENROUTER_BASE, dict(
            supports_image_detail=True,
            default_image_detail="high",
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
        ))

    # Claude via OpenRouter
    if "claude" in underlying or "anthropic/" in m:
        return _build_caps(model_name, "openrouter-claude", _OPENROUTER_BASE, dict(
            is_reasoning_model=True,
            supports_reasoning_effort=True,
            max_context_tokens=200000,
        ))

    # Gemini via OpenRouter
    if "gemini" in underlying or "google/" in m:
        is_thinking = any(x in m for x in ("gemini-2.5", "gemini-3", "gemini-2-5", "gemini-3-"))
        return _build_caps(model_name, "openrouter-gemini", _OPENROUTER_BASE, dict(
            supports_media_resolution=True,
            default_media_resolution="high",
            is_reasoning_model=is_thinking,
            supports_reasoning_effort=True,
            max_context_tokens=1000000,
            max_output_tokens=8192,
        ))

    # Llama via OpenRouter
    if "llama" in underlying or "meta/" in m:
        return _build_caps(model_name, "openrouter-llama", _OPENROUTER_BASE, dict(
            supports_vision="vision" in m or "llama-3.2" in m,
            supports_frequency_penalty=True,
            supports_presence_penalty=True,
        ))

    # Mistral via OpenRouter
    if "mistral" in m or "mixtral" in m:
        return _build_caps(model_name, "openrouter-mistral", _OPENROUTER_BASE, dict(
            supports_vision="pixtral" in m,
        ))

    # Generic OpenRouter fallback
    return _build_caps(model_name, "openrouter", _OPENROUTER_BASE, {})


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "ProviderCapabilities",
    "ProviderType",
    "ImageDetail",
    "MediaResolution",
    "CapabilityError",
    "ensure_image_support",
    "detect_provider",
    "detect_capabilities",
]
