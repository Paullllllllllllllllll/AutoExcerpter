"""Application configuration loader from YAML files.

This module loads configuration from modules/config/app.yaml and exposes
settings as module-level constants. Import as:
    from modules import app_config as config
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Constants
DEFAULT_CONCURRENT_REQUESTS = 4
DEFAULT_API_TIMEOUT = 320
DEFAULT_OPENAI_TIMEOUT = 900
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_RATE_LIMITS: List[Tuple[int, int]] = [(120, 1), (15000, 60), (15000, 3600)]

# Path resolution
_MODULES_DIR = Path(__file__).resolve().parent
_APP_CONFIG_PATH = _MODULES_DIR / "config" / "app.yaml"


def _load_yaml_app_config() -> Dict[str, Any]:
    """
    Load the application config YAML.

    Returns:
        Configuration dictionary, or empty dict on error
    """
    if not _APP_CONFIG_PATH.exists():
        logger.warning(f"App config file not found: {_APP_CONFIG_PATH}. Using defaults.")
        return {}

    try:
        with _APP_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        if not isinstance(data, dict):
            logger.warning("App config is not a dictionary. Using defaults.")
            return {}
            
        return data
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in app.yaml: {e}. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Error loading app config: {e}. Using defaults.")
        return {}


def _as_rate_limits(val: Any) -> List[Tuple[int, int]]:
    """
    Normalize YAML rate_limits into a list of (int, int) tuples.

    Args:
        val: Value from YAML configuration

    Returns:
        List of (max_requests, time_window_seconds) tuples
    """
    if not isinstance(val, list):
        return DEFAULT_RATE_LIMITS

    out: List[Tuple[int, int]] = []
    for item in val:
        try:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((int(item[0]), int(item[1])))
        except (ValueError, TypeError):
            # Skip malformed entries
            continue

    return out if out else DEFAULT_RATE_LIMITS


def _get_str(data: Dict[str, Any], key: str, default: str) -> str:
    """Safely get a string value from config dictionary."""
    value = data.get(key, default)
    return str(value) if value is not None else default


def _get_int(data: Dict[str, Any], key: str, default: int) -> int:
    """Safely get an integer value from config dictionary."""
    try:
        return int(data.get(key, default))
    except (ValueError, TypeError):
        logger.warning(f"Invalid integer for '{key}', using default: {default}")
        return default


def _get_bool(data: Dict[str, Any], key: str, default: bool) -> bool:
    """Safely get a boolean value from config dictionary."""
    value = data.get(key, default)
    return bool(value)


# Load configuration
_APP_CFG: Dict[str, Any] = _load_yaml_app_config()
_OA: Dict[str, Any] = _APP_CFG.get("openai", {}) if isinstance(_APP_CFG.get("openai"), dict) else {}

# Feature toggle
SUMMARIZE = _get_bool(_APP_CFG, "summarize", True)

# Folder Paths
INPUT_FOLDER_PATH = _get_str(_APP_CFG, "input_folder_path", r"C:\Users\paulg\OneDrive\Desktop\New Literature")
OUTPUT_FOLDER_PATH = _get_str(_APP_CFG, "output_folder_path", r"C:\Users\paulg\OneDrive\Desktop\New Literature")

# Cleanup Settings
DELETE_TEMP_WORKING_DIR = _get_bool(_APP_CFG, "delete_temp_working_dir", True)

# Performance Settings
CONCURRENT_REQUESTS = _get_int(_APP_CFG, "concurrent_requests", DEFAULT_CONCURRENT_REQUESTS)
API_TIMEOUT = _get_int(_APP_CFG, "api_timeout", DEFAULT_API_TIMEOUT)

# Rate Limiting Configuration (OpenAI)
OPENAI_RATE_LIMITS = _as_rate_limits(_OA.get("rate_limits"))

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

OPENAI_MODEL = _get_str(_OA, "model", DEFAULT_MODEL)
OPENAI_TRANSCRIPTION_MODEL = _get_str(_OA, "transcription_model", DEFAULT_MODEL)
OPENAI_API_TIMEOUT = _get_int(_OA, "api_timeout", DEFAULT_OPENAI_TIMEOUT)
OPENAI_USE_FLEX = _get_bool(_OA, "use_flex", True)

# Log loaded configuration
logger.debug(f"Configuration loaded: SUMMARIZE={SUMMARIZE}, CONCURRENT_REQUESTS={CONCURRENT_REQUESTS}")
logger.debug(f"Models: transcription={OPENAI_TRANSCRIPTION_MODEL}, summary={OPENAI_MODEL}")
