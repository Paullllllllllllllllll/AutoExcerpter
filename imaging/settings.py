"""Connect portable image policy to AE settings and its capability registry."""

from typing import Any

from config.constants import DEFAULT_JPEG_QUALITY
from config.loader import ConfigLoader
from config.logger import setup_logger
from imaging._provider import (
    ModelType,
    detect_model_type,
    get_image_config_section_name,
    resolve_request_detail,
)
from imaging.native import model_image_cap, validate_image_settings

logger = setup_logger(__name__)


def resolve_image_settings(
    loader: ConfigLoader, provider: str | None, model_name: str | None
) -> tuple[dict[str, Any], ModelType, str]:
    """Resolve section-only settings, wire detail, caps and rendering options."""
    from llm.capabilities import detect_capabilities, detect_provider

    model = model_name or ""
    provider = (provider or detect_provider(model)).lower()
    model_type = detect_model_type(provider, model)
    section = get_image_config_section_name(model_type)
    full_cfg = loader.get_image_processing_config()
    cfg = full_cfg.get(section, {})
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
    model_cfg = loader.get_model_config().get("transcription_model", {})
    request_detail = resolve_request_detail(model_cfg, provider, model, logger)
    if model_type == "google":
        detail = str(result.get("media_resolution", "high") or "high")
    elif model_type == "anthropic":
        detail = str(result.get("resize_profile", "auto") or "auto")
    elif provider == "openai":
        detail = request_detail or "auto"
    else:
        # OpenRouter and custom requests carry no OpenAI detail parameter.
        detail = "high"
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
    # Keep the effective local detail in the section for existing consumers.
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
