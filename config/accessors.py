"""Concurrency configuration helper utilities.

This module provides convenient access to concurrency settings from configuration
files, with sensible defaults and validation.
"""

from __future__ import annotations

from typing import Any

from config.constants import DEFAULT_CONCURRENT_REQUESTS as DEFAULT_CONCURRENT_REQUESTS
from config.constants import (
    DEFAULT_OPENAI_TIMEOUT,
    DEFAULT_RATE_LIMITS,
    DEFAULT_TARGET_DPI,
)
from config.loader import get_config_loader
from config.logger import setup_logger

logger = setup_logger(__name__)

_warned: set[str] = set()


def _warn_once(message: str) -> None:
    """Log *message* at WARNING the first time it is seen this process.

    The accessors below are called several times per run (pre-run overview,
    log header, each client), so an unconditional warning would repeat the
    same notice a handful of times; the console handler shows WARNING and
    above, which is the only channel the user actually reads.
    """
    if message in _warned:
        return
    _warned.add(message)
    logger.warning(message)


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


def _coerce_int(raw: Any, default: int, minimum: int, label: str) -> int:
    """Coerce *raw* to an int of at least *minimum*, warning once on repair.

    ``None`` means the key is absent, which is the normal case and stays
    silent; anything else that cannot be honored is a hand-edit the user
    should hear about rather than have quietly overruled.
    """
    if raw is None:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        _warn_once(f"Config {label}={raw!r} is not a number; using {default}.")
        return default
    if value < minimum:
        _warn_once(f"Config {label}={value} is invalid; using the minimum {minimum}.")
        return minimum
    return value


# ============================================================================
# Concurrency Configuration Access
# ============================================================================
def get_api_concurrency(api_type: str = "transcription") -> int:
    """Get the parallel-worker count for API requests of *api_type*."""
    try:
        cfg_loader = get_config_loader()
        concurrency_cfg = cfg_loader.get_concurrency_config()
        api_cfg = concurrency_cfg.get("api_requests", {}).get(api_type, {})

        # A quoted YAML value (e.g. "80") would survive to min(max_workers, ...)
        # in the pipeline and raise TypeError mid-run; a zero or negative one
        # would be silently clamped much further downstream, after the run log
        # header had already recorded the unclamped value.
        raw = api_cfg.get("concurrency_limit")
        return _coerce_int(
            raw,
            DEFAULT_CONCURRENT_REQUESTS,
            1,
            f"api_requests.{api_type}.concurrency_limit",
        )
    except Exception as e:
        logger.warning(f"Error loading {api_type} concurrency config: {e}")
        return DEFAULT_CONCURRENT_REQUESTS


def get_transcription_concurrency() -> int:
    """Get the parallel-worker count for transcription API requests."""
    return get_api_concurrency("transcription")


def get_service_tier(api_type: str = "transcription") -> str:
    """Get OpenAI service tier for the specified API type.

    A phase without its own ``service_tier`` falls back to the transcription
    tier (the behavior documented in ``concurrency.example.yaml`` and shown
    by the pre-run overview), then to ``"flex"``.
    """
    tier: Any = _get_config_value(
        "get_concurrency_config",
        ["api_requests", api_type, "service_tier"],
        default=None,
        log_context="service tier",
    )
    if isinstance(tier, str) and tier:
        return tier
    if api_type != "transcription":
        return get_service_tier("transcription")
    return "flex"


def get_api_timeout() -> int:
    """Get API request timeout in seconds from concurrency.yaml."""
    timeout: Any = _get_config_value(
        "get_concurrency_config",
        ["api_requests", "api_timeout"],
        default=None,
        log_context="API timeout",
    )
    # llm/base.py hands this straight to the SDK, so a zero or negative
    # timeout would arrive there unclamped.
    return _coerce_int(timeout, DEFAULT_OPENAI_TIMEOUT, 1, "api_requests.api_timeout")


def get_rate_limits() -> list[tuple[int, int]]:
    """Get rate limiting configuration from concurrency.yaml.

    Returns:
        List of (max_requests, time_window_seconds) tuples.
    """
    default_limits = list(DEFAULT_RATE_LIMITS)
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
                    max_requests = int(item[0])
                    time_window = int(item[1])
                except (ValueError, TypeError):
                    continue
                if max_requests <= 0 or time_window <= 0:
                    _warn_once(f"Ignoring non-positive rate_limits entry {item!r}.")
                    continue
                limits.append((max_requests, time_window))

        # An all-invalid throttle list falls back to the built-in limits, which
        # are the MOST permissive setting the tool has; say so rather than let
        # a typo quietly uncap the request rate.
        if not limits and raw_limits:
            _warn_once(
                "No usable api_requests.rate_limits entries; using built-in defaults."
            )

        return limits if limits else default_limits
    except Exception as e:
        logger.debug(f"Error loading rate limits: {e}")
        return default_limits


def get_target_dpi() -> int:
    """Get target DPI for PDF page extraction."""
    dpi: Any = _get_config_value(
        "get_image_processing_config",
        ["api_image_processing", "target_dpi"],
        default=None,
        log_context="target DPI",
    )
    return _coerce_int(dpi, DEFAULT_TARGET_DPI, 1, "api_image_processing.target_dpi")


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "get_api_concurrency",
    "get_transcription_concurrency",
    "get_service_tier",
    "get_api_timeout",
    "get_rate_limits",
    "get_target_dpi",
]
