"""Concurrency configuration helper utilities.

This module provides convenient access to concurrency settings from configuration
files, with sensible defaults and validation.
"""

from __future__ import annotations

from typing import Tuple

from modules import app_config as config
from modules.config_loader import get_config_loader
from modules.logger import setup_logger

logger = setup_logger(__name__)

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "get_transcription_concurrency",
    "get_summary_concurrency",
    "get_image_processing_concurrency",
    "get_service_tier",
]


# ============================================================================
# Concurrency Configuration Access
# ============================================================================
def get_transcription_concurrency() -> Tuple[int, float]:
    """
    Get concurrency settings for transcription API requests.
    
    Returns:
        Tuple of (max_workers, delay_between_tasks).
    """
    try:
        cfg_loader = get_config_loader()
        concurrency_cfg = cfg_loader.get_concurrency_config()
        trans_cfg = concurrency_cfg.get("api_requests", {}).get("transcription", {})
        
        max_workers = trans_cfg.get("concurrency_limit", config.CONCURRENT_REQUESTS)
        delay = trans_cfg.get("delay_between_tasks", 0.05)
        
        return max_workers, delay
    except Exception as e:
        logger.warning(f"Error loading transcription concurrency config: {e}")
        return config.CONCURRENT_REQUESTS, 0.05


def get_summary_concurrency() -> Tuple[int, float]:
    """
    Get concurrency settings for summary API requests.
    
    Returns:
        Tuple of (max_workers, delay_between_tasks).
    """
    try:
        cfg_loader = get_config_loader()
        concurrency_cfg = cfg_loader.get_concurrency_config()
        summ_cfg = concurrency_cfg.get("api_requests", {}).get("summary", {})
        
        max_workers = summ_cfg.get("concurrency_limit", config.CONCURRENT_REQUESTS)
        delay = summ_cfg.get("delay_between_tasks", 0.05)
        
        return max_workers, delay
    except Exception as e:
        logger.warning(f"Error loading summary concurrency config: {e}")
        return config.CONCURRENT_REQUESTS, 0.05


def get_image_processing_concurrency() -> Tuple[int, float]:
    """
    Get concurrency settings for local image processing.
    
    Returns:
        Tuple of (max_workers, delay_between_tasks).
    """
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
    """
    Get OpenAI service tier for the specified API type.
    
    Args:
        api_type: Type of API request ('transcription' or 'summary').
    
    Returns:
        Service tier string ('auto', 'default', 'flex', or 'priority').
    """
    try:
        cfg_loader = get_config_loader()
        concurrency_cfg = cfg_loader.get_concurrency_config()
        
        service_tier = (
            concurrency_cfg
            .get("api_requests", {})
            .get(api_type, {})
            .get("service_tier")
        )
        
        if service_tier:
            return service_tier
        
        # Fallback to legacy OPENAI_USE_FLEX setting
        return "flex" if config.OPENAI_USE_FLEX else "auto"
    except Exception as e:
        logger.debug(f"Error determining service tier: {e}")
        return "flex" if config.OPENAI_USE_FLEX else "auto"


def get_target_dpi() -> int:
    """
    Get target DPI for PDF page extraction.
    
    Returns:
        Target DPI value.
    """
    try:
        cfg_loader = get_config_loader()
        img_cfg = cfg_loader.get_image_processing_config()
        dpi = img_cfg.get('api_image_processing', {}).get('target_dpi', 300)
        return int(dpi)
    except Exception as e:
        logger.debug(f"Error loading target DPI: {e}")
        return 300
