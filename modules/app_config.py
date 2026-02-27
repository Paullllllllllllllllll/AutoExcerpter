"""Application configuration loader from YAML files.

This module loads configuration from modules/config/app.yaml and exposes
settings as module-level constants. Configuration includes:
- Execution mode (CLI vs interactive)
- Feature toggles (summarization, cleanup)
- File paths (input/output directories)
- Citation management settings (OpenAlex integration)
- Daily token limits

Import as:
    from modules import app_config as config

Configuration is loaded at module import time and exposed as constants:
    - CLI_MODE: bool
    - SUMMARIZE: bool
    - INPUT_FOLDER_PATH: str
    - OUTPUT_FOLDER_PATH: str
    - etc.

Note: Model configuration is centralized in model.yaml
Note: API concurrency settings (rate_limits, api_timeout, service_tier) are in concurrency.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Configuration File Paths
# ============================================================================
_MODULES_DIR = Path(__file__).resolve().parent
_APP_CONFIG_PATH = _MODULES_DIR / "config" / "app.yaml"


# ============================================================================
# Configuration Loading Functions
# ============================================================================
def _load_yaml_app_config() -> dict[str, Any]:
    """
    Load the application config YAML.

    Returns:
        Configuration dictionary, or empty dict on error
    """
    if not _APP_CONFIG_PATH.exists():
        logger.warning(
            f"App config file not found: {_APP_CONFIG_PATH}. Using defaults."
        )
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


def _get_str(data: dict[str, Any], key: str, default: str) -> str:
    """Safely get a string value from config dictionary."""
    value = data.get(key, default)
    return str(value) if value is not None else default


def _get_int(data: dict[str, Any], key: str, default: int) -> int:
    """Safely get an integer value from config dictionary."""
    try:
        return int(data.get(key, default))
    except (ValueError, TypeError):
        logger.warning(f"Invalid integer for '{key}', using default: {default}")
        return default


def _get_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    """Safely get a boolean value from config dictionary."""
    value = data.get(key, default)
    return bool(value)


# ============================================================================
# Configuration Values (loaded at module import)
# ============================================================================
# Load configuration
_APP_CFG: dict[str, Any] = _load_yaml_app_config()
_CITATION: dict[str, Any] = (
    _APP_CFG.get("citation", {}) if isinstance(_APP_CFG.get("citation"), dict) else {}
)
_TOKEN_LIMIT: dict[str, Any] = (
    _APP_CFG.get("daily_token_limit", {})
    if isinstance(_APP_CFG.get("daily_token_limit"), dict)
    else {}
)
_SUMMARY_OUTPUT: dict[str, Any] = (
    _APP_CFG.get("summary_output", {})
    if isinstance(_APP_CFG.get("summary_output"), dict)
    else {}
)

# --- Execution Mode ---
CLI_MODE = _get_bool(_APP_CFG, "cli_mode", False)

# --- Feature Toggles ---
SUMMARIZE = _get_bool(_APP_CFG, "summarize", True)

# --- Summary Output Formats ---
OUTPUT_DOCX = _get_bool(_SUMMARY_OUTPUT, "docx", True)
OUTPUT_MARKDOWN = _get_bool(_SUMMARY_OUTPUT, "markdown", True)

# --- File Paths ---
INPUT_FOLDER_PATH = _get_str(
    _APP_CFG, "input_folder_path", r"C:\Users\paulg\OneDrive\Desktop\New Literature"
)
OUTPUT_FOLDER_PATH = _get_str(
    _APP_CFG, "output_folder_path", r"C:\Users\paulg\OneDrive\Desktop\New Literature"
)
INPUT_PATHS_IS_OUTPUT_PATH = _get_bool(_APP_CFG, "input_paths_is_output_path", False)

# --- Cleanup Settings ---
DELETE_TEMP_WORKING_DIR = _get_bool(_APP_CFG, "delete_temp_working_dir", True)

# --- Citation Management Settings ---
CITATION_OPENALEX_EMAIL = _get_str(
    _CITATION, "openalex_email", "your-email@example.com"
)
CITATION_MAX_API_REQUESTS = _get_int(_CITATION, "max_api_requests", 50)
CITATION_ENABLE_OPENALEX = _get_bool(_CITATION, "enable_openalex_enrichment", True)

# --- LLM Provider API Keys ---
# API keys are loaded from environment variables
# At least one provider's API key must be set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Check that at least one API key is configured
_available_providers = []
if OPENAI_API_KEY:
    _available_providers.append("openai")
if ANTHROPIC_API_KEY:
    _available_providers.append("anthropic")
if GOOGLE_API_KEY:
    _available_providers.append("google")
if OPENROUTER_API_KEY:
    _available_providers.append("openrouter")

if not _available_providers:
    error_msg = (
        "No LLM provider API key found. Please set at least one of: "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY"
    )
    logger.error(error_msg)
    raise EnvironmentError(error_msg)

logger.info(f"Available LLM providers: {', '.join(_available_providers)}")

# --- Daily Token Limit ---
DAILY_TOKEN_LIMIT_ENABLED = _get_bool(_TOKEN_LIMIT, "enabled", False)
DAILY_TOKEN_LIMIT = _get_int(_TOKEN_LIMIT, "daily_tokens", 10000000)

# ============================================================================
# Logging
# ============================================================================
logger.debug(f"Configuration loaded: CLI_MODE={CLI_MODE}, SUMMARIZE={SUMMARIZE}")
logger.debug("Model config in model.yaml, API concurrency in concurrency.yaml")
logger.debug(
    f"Daily token limit: enabled={DAILY_TOKEN_LIMIT_ENABLED}, limit={DAILY_TOKEN_LIMIT}"
)
