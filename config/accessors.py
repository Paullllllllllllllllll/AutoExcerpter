"""Concurrency configuration helper utilities.

This module provides convenient access to concurrency settings from configuration
files, with sensible defaults and validation.
"""

from __future__ import annotations

from typing import Any

from config.loader import get_config_loader
from config.constants import DEFAULT_CONCURRENT_REQUESTS as DEFAULT_CONCURRENT_REQUESTS
from config.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Private Helper
# ============================================================================
def _get_config_value(
    config_method: str,
    path: list[str],
    default: Any,
    log_context: str = "",
) -> Any:
    """Navigate nested config dicts along *path*, returning *default* on any failure."""
    try:
        cfg_loader = get_config_loader()
        getter = getattr(cfg_loader, config_method)
        result: Any = getter()
        for key in path:
            result = result.get(key, {}) if isinstance(result, dict) else {}
        return result if result != {} else default
    except Exception as e:
        if log_context:
            logger.debug(f"Error loading {log_context}: {e}")
        return default


# ============================================================================
# Concurrency Configuration Access
# ============================================================================
def get_api_concurrency(api_type: str = "transcription") -> tuple[int, float]:
    """Get concurrency settings for API requests."""
    try:
        cfg_loader = get_config_loader()
        concurrency_cfg = cfg_loader.get_concurrency_config()
        api_cfg = concurrency_cfg.get("api_requests", {}).get(api_type, {})

        max_workers = api_cfg.get("concurrency_limit", DEFAULT_CONCURRENT_REQUESTS)
        delay = api_cfg.get("delay_between_tasks", 0.05)

        return max_workers, delay
    except Exception as e:
        logger.warning(f"Error loading {api_type} concurrency config: {e}")
        return DEFAULT_CONCURRENT_REQUESTS, 0.05


def get_transcription_concurrency() -> tuple[int, float]:
    """Get concurrency settings for transcription API requests."""
    return get_api_concurrency("transcription")


def get_summary_concurrency() -> tuple[int, float]:
    """Get concurrency settings for summary API requests."""
    return get_api_concurrency("summary")


def get_image_processing_concurrency() -> tuple[int, float]:
    """Get concurrency settings for local image processing."""
    try:
        cfg_loader = get_config_loader()
        concurrency_cfg = cfg_loader.get_concurrency_config()
        img_cfg = concurrency_cfg.get("image_processing", {})

        max_workers = img_cfg.get("concurrency_limit", 24)
        delay = img_cfg.get("delay_between_tasks", 0)

        return max_workers, delay
    except Exception as e:
        logger.warning(f"Error loading image processing concurrency config: {e}")
        return 24, 0


def get_service_tier(api_type: str = "transcription") -> str:
    """Get OpenAI service tier for the specified API type."""
    tier: Any = _get_config_value(
        "get_concurrency_config",
        ["api_requests", api_type, "service_tier"],
        default=None,
        log_context="service tier",
    )
    if isinstance(tier, str) and tier:
        return tier
    return "flex"


def get_api_timeout() -> int:
    """Get API request timeout in seconds from concurrency.yaml."""
    timeout: Any = _get_config_value(
        "get_concurrency_config",
        ["api_requests", "api_timeout"],
        default=900,
        log_context="API timeout",
    )
    return int(timeout)


def get_rate_limits() -> list[tuple[int, int]]:
    """Get rate limiting configuration from concurrency.yaml.

    Returns:
        List of (max_requests, time_window_seconds) tuples.
    """
    default_limits = [(120, 1), (15000, 60), (15000, 3600)]
    try:
        cfg_loader = get_config_loader()
        concurrency_cfg = cfg_loader.get_concurrency_config()
        raw_limits = concurrency_cfg.get("api_requests", {}).get("rate_limits")

        if not isinstance(raw_limits, list):
            return default_limits

        limits: list[tuple[int, int]] = []
        for item in raw_limits:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    limits.append((int(item[0]), int(item[1])))
                except (ValueError, TypeError):
                    continue

        return limits if limits else default_limits
    except Exception as e:
        logger.debug(f"Error loading rate limits: {e}")
        return default_limits


def get_target_dpi() -> int:
    """Get target DPI for PDF page extraction."""
    dpi: Any = _get_config_value(
        "get_image_processing_config",
        ["api_image_processing", "target_dpi"],
        default=300,
        log_context="target DPI",
    )
    return int(dpi)


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "get_api_concurrency",
    "get_transcription_concurrency",
    "get_summary_concurrency",
    "get_image_processing_concurrency",
    "get_service_tier",
    "get_api_timeout",
    "get_rate_limits",
    "get_target_dpi",
]
