"""Resolve the page-image settings of one transcription model.

The per-provider sections below are the recommended values; the run's
``ImageSpec`` overrides the fields it sets. The resolved dict is recorded in
the working log, and its fingerprint guards resume.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any, Literal

from autoexcerpter.common.images import DEFAULT_JPEG_QUALITY
from autoexcerpter.common.native import model_image_cap, validate_image_settings
from autoexcerpter.llm.capabilities import detect_capabilities, detect_provider
from autoexcerpter.llm.image_detail import resolve_request_detail
from autoexcerpter.spec import ImageSpec, recommended_images

logger = logging.getLogger(__name__)

ModelType = Literal["openai", "google", "anthropic", "custom"]

# Recommended image settings per model family. Numeric runs render directly to
# the profile unless a section sets the supersample strategy; native runs
# render to the resolved model cap or profile. The optional OpenAI keys for
# the original side and pixel limits and the Anthropic key for the high side
# limit only tighten the model caps.
IMAGE_PROCESSING: Mapping[str, Any] = {
    "render_strategy": "direct",
    # Numeric rendering only; 0 disables the memory guard.
    "max_pixels_per_page": 24_000_000,
    # Downscaling filter: bilinear (fast) or lanczos (quality).
    "resampling_algorithm": "bilinear",
    "api_image_processing": {
        "target_dpi": "native",
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 50_000_000,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "jpeg_quality": 95,
        "llm_detail": "original",
        "resize_profile": "high",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    },
    "anthropic_image_processing": {
        "target_dpi": "native",
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 10_000_000,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "jpeg_quality": 95,
        "resize_profile": "auto",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    },
    "google_image_processing": {
        "target_dpi": 300,
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 20_000_000,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "jpeg_quality": 95,
        "media_resolution": "high",
        "resize_profile": "high",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    },
    "custom_image_processing": {
        "target_dpi": 150,
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 0,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "jpeg_quality": 85,
        "llm_detail": "high",
        "resize_profile": "low",
        "low_max_side_px": 768,
        "high_target_box": [512, 1024],
    },
}

# The section key that carries the detail of each model family.
_DETAIL_KEY: Mapping[str, str] = {
    "openai": "llm_detail",
    "anthropic": "resize_profile",
    "google": "media_resolution",
    "custom": "llm_detail",
}

_DIRECT_PROVIDERS: Mapping[str, ModelType] = {
    "custom": "custom",
    "google": "google",
    "anthropic": "anthropic",
    "openai": "openai",
}

_SECTION_NAMES: Mapping[str, str] = {
    "custom": "custom_image_processing",
    "google": "google_image_processing",
    "anthropic": "anthropic_image_processing",
}


def detect_model_type(provider: str, model_name: str | None = None) -> ModelType:
    """Return the model family whose image settings apply.

    Direct providers decide; OpenRouter and unknown providers are detected
    from the model name, defaulting to OpenAI.
    """
    provider = provider.lower()
    model_name = model_name.lower() if model_name else ""
    if provider in _DIRECT_PROVIDERS:
        return _DIRECT_PROVIDERS[provider]
    if "gemini" in model_name or "google/" in model_name:
        return "google"
    if "claude" in model_name or "anthropic/" in model_name:
        return "anthropic"
    return "openai"


def get_image_config_section_name(model_type: ModelType) -> str:
    """Return the settings section of *model_type*."""
    return _SECTION_NAMES.get(model_type, "api_image_processing")


def request_detail_option(
    provider: str | None, images: ImageSpec | None = None
) -> str | None:
    """Return the ``image_size`` option of an OpenAI transcription model.

    The run's detail wins; otherwise the recommended OpenAI detail. Other
    providers send no detail parameter.
    """
    if provider != "openai":
        return None
    if images is not None and images.detail is not None:
        return images.detail
    return recommended_images("openai").detail


def _apply_overrides(
    cfg: dict[str, Any], images: ImageSpec, model_type: ModelType
) -> None:
    """Write the fields *images* sets into one provider section."""
    if images.dpi is not None:
        cfg["target_dpi"] = images.dpi
    if images.format is not None:
        cfg["payload_format"] = images.format
    if images.jpeg_quality is not None:
        cfg["jpeg_quality"] = images.jpeg_quality
    if images.grayscale is not None:
        cfg["grayscale_conversion"] = images.grayscale
    if images.detail is not None:
        cfg[_DETAIL_KEY[model_type]] = images.detail


def resolve_image_settings(
    provider: str | None,
    model_name: str | None,
    images: ImageSpec | None = None,
) -> tuple[dict[str, Any], ModelType, str]:
    """Resolve section settings, wire detail, caps and rendering options."""
    model = model_name or ""
    provider = (provider or detect_provider(model)).lower()
    model_type = detect_model_type(provider, model)
    section = get_image_config_section_name(model_type)
    full_cfg = IMAGE_PROCESSING
    cfg = copy.deepcopy(dict(full_cfg.get(section, {})))
    if images is not None:
        _apply_overrides(cfg, images, model_type)
    validate_image_settings(cfg, section)
    result = {
        "target_dpi": 300,
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 0,
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "grayscale_conversion": True,
        "handle_transparency": True,
        "resize_profile": "auto",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
        **cfg,
    }
    model_options = {"image_size": request_detail_option(provider, images)}
    request_detail = resolve_request_detail(model_options, provider, model, logger)
    if model_type == "google":
        detail = str(result.get("media_resolution", "high") or "high")
    elif model_type == "anthropic":
        detail = str(result.get("resize_profile", "auto") or "auto")
    elif provider == "openai":
        detail = request_detail or "auto"
    else:
        # OpenRouter and custom requests carry no OpenAI detail parameter, so the
        # local profile decides. On OpenRouter, original keeps the conservative
        # 10,000-patch cap (the target model's default detail decides remotely);
        # custom endpoints fall back to the high profile.
        local = str(result.get("llm_detail", "high") or "high").strip().lower()
        allowed: tuple[str, ...] = ("low", "high", "auto", "original")
        if provider != "openrouter":
            allowed = ("low", "high", "auto")
        detail = local if local in allowed else "high"
    strategy = (
        str(cfg.get("render_strategy") or full_cfg.get("render_strategy") or "direct")
        .strip()
        .lower()
    )
    if strategy not in ("direct", "supersample"):
        strategy = "direct"
    caps = detect_capabilities(model.split("/")[-1])
    result.update(
        model_name=model,
        provider=provider,
        model_type=model_type,
        request_detail=request_detail,
        resolved_detail=detail.strip().lower(),
        image_original_patch_cap_30k=(
            caps.image_original_patch_cap_30k and provider == "openai"
        ),
        image_high_res_tier=caps.image_high_res_tier,
        max_pixels_per_page=int(full_cfg.get("max_pixels_per_page", 0)),
        resampling_algorithm=full_cfg.get("resampling_algorithm", "bilinear"),
        render_strategy=strategy,
    )
    # The OpenAI section's llm_detail reports the effective local detail.
    if model_type == "openai":
        result["llm_detail"] = result["resolved_detail"]
    cap = model_image_cap(model_type, model, result["resolved_detail"], result)
    result["cap_policy"] = cap.policy if cap else "profile-v1"
    if (
        result["target_dpi"] == "native"
        and not cap
        and result["resize_profile"] == "none"
    ):
        raise ValueError(f"Invalid resize_profile in {section}: native needs a bound")
    return result, model_type, section


__all__ = [
    "IMAGE_PROCESSING",
    "ModelType",
    "detect_model_type",
    "get_image_config_section_name",
    "request_detail_option",
    "resolve_image_settings",
]
