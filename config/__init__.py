"""Unified configuration package for AutoExcerpter.

Public interface:

- ``config.app`` — application-level settings as module constants
  (``CLI_MODE``, ``SUMMARIZE``, ``INPUT_FOLDER_PATH``, API keys,
  ``DAILY_TOKEN_LIMIT``, citation settings) plus ``get_available_providers``
  and ``require_api_key``.
- ``config.constants`` — hardcoded default values and named constants.
- ``config.get_config_loader()`` — singleton YAML loader for
  ``image_processing.yaml``, ``concurrency.yaml``, ``model.yaml``.
- Typed accessor functions for concurrency/rate-limit/DPI settings.
- Path constants (``PROJECT_ROOT``, ``CONFIG_DIR``, ``PROMPTS_DIR``,
  ``SCHEMAS_DIR``).
"""

from config import app, constants
from config.accessors import (
    get_api_concurrency,
    get_api_timeout,
    get_image_processing_concurrency,
    get_rate_limits,
    get_service_tier,
    get_summary_concurrency,
    get_target_dpi,
    get_transcription_concurrency,
)
from config.loader import (
    CONFIG_DIR,
    ConfigLoader,
    PROJECT_ROOT,
    PROMPTS_DIR,
    SCHEMAS_DIR,
    get_config_loader,
)

__all__ = [
    "app",
    "constants",
    "ConfigLoader",
    "get_config_loader",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "PROMPTS_DIR",
    "SCHEMAS_DIR",
    "get_api_concurrency",
    "get_api_timeout",
    "get_image_processing_concurrency",
    "get_rate_limits",
    "get_service_tier",
    "get_summary_concurrency",
    "get_target_dpi",
    "get_transcription_concurrency",
]
