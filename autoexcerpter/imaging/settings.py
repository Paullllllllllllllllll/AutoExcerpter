"""Resolve the page-image settings of one transcription model.

The per-provider sections of :data:`IMAGE_PROCESSING` are the recommended
values of ``spec.IMAGE_PRESETS``; the run's ``ImageSpec`` overrides the
fields it sets, and the detail follows the presets' detail rules. The
resolved dict is recorded with the document in the store, and resume
compares it key by key.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from autoexcerpter.common import image_presets
from autoexcerpter.common.image_presets import ModelType
from autoexcerpter.common.images import DEFAULT_JPEG_QUALITY
from autoexcerpter.common.native import model_image_cap, validate_image_settings
from autoexcerpter.llm.capabilities import (
    ORIGINAL_MAX_SIDE,
    detect_capabilities,
    detect_provider,
)
from autoexcerpter.llm.image_detail import resolve_request_detail
from autoexcerpter.spec import IMAGE_PRESETS, ImageSpec, recommended_images

logger = logging.getLogger(__name__)

# Recommended image settings per model family. Numeric runs render directly to
# the profile unless a section sets the supersample strategy; native runs
# render to the resolved model cap or profile. ``max_pixels_per_page`` guards
# numeric rendering only (0 disables it); ``resampling_algorithm`` is the
# downscaling filter, bilinear (fast) or lanczos (quality).
IMAGE_PROCESSING: Mapping[str, Any] = {
    "render_strategy": IMAGE_PRESETS.render_strategy,
    "max_pixels_per_page": IMAGE_PRESETS.max_pixels_per_page,
    "resampling_algorithm": IMAGE_PRESETS.resampling,
    **IMAGE_PRESETS.sections(),
}


def detect_model_type(provider: str, model_name: str | None = None) -> ModelType:
    """Return the model family whose image settings apply.

    Direct providers decide; OpenRouter and unknown providers are detected
    from the model name, defaulting to OpenAI.
    """
    return image_presets.detect_model_type(provider, model_name)


def get_image_config_section_name(model_type: ModelType) -> str:
    """Return the settings section of *model_type*."""
    return image_presets.section_name(model_type)


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


def resolve_image_settings(
    provider: str | None,
    model_name: str | None,
    images: ImageSpec | None = None,
    original_max_side_px: int | None = None,
) -> tuple[dict[str, Any], ModelType, str]:
    """Resolve section settings, wire detail, caps and rendering options.

    OpenAI pages at ``original`` detail are limited on the long side to
    *original_max_side_px*, else to the model's default limit, if any.
    """
    model = model_name or ""
    provider = (provider or detect_provider(model)).lower()
    model_type = detect_model_type(provider, model)
    section = get_image_config_section_name(model_type)
    full_cfg = IMAGE_PROCESSING
    rules = IMAGE_PRESETS.detail
    cfg = image_presets.section_settings(full_cfg, section, images, rules, provider)
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
    image_size = request_detail_option(provider, images)
    model_options = {"image_size": image_size}
    request_detail = resolve_request_detail(model_options, provider, model, logger)
    detail = image_presets.page_detail(
        provider,
        model_type,
        result,
        detect_capabilities(model),
        rules,
        request=image_size,
    )
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
        resolved_detail=detail,
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
    if provider == "openai" and result["resolved_detail"] == "original":
        side = original_max_side_px
        if side is None:
            side = caps.extra(ORIGINAL_MAX_SIDE)
        if side is not None:
            result["original_max_side_px"] = int(side)
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
