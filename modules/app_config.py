from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

# YAML-backed app configuration. Mirrors the old config.py interface so callers
# can continue using `config.VAR` after importing as:
#   from modules import app_config as config

_MODULES_DIR = Path(__file__).resolve().parent
_APP_CONFIG_PATH = _MODULES_DIR / "config" / "app.yaml"


def _load_yaml_app_config() -> Dict[str, Any]:
    """Load the application config YAML. Returns {} on any error."""
    try:
        if _APP_CONFIG_PATH.exists():
            with _APP_CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _as_rate_limits(val: Any) -> List[Tuple[int, int]]:
    """Normalize YAML rate_limits into a list of (int, int) tuples."""
    default_limits: List[Tuple[int, int]] = [(120, 1), (15000, 60), (15000, 3600)]
    if not isinstance(val, list):
        return default_limits
    out: List[Tuple[int, int]] = []
    for item in val:
        try:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((int(item[0]), int(item[1])))
        except Exception:
            # Skip malformed entries
            continue
    return out if out else default_limits


_APP_CFG: Dict[str, Any] = _load_yaml_app_config()
_OA: Dict[str, Any] = (_APP_CFG.get("openai") if isinstance(_APP_CFG.get("openai"), dict) else {}) or {}

# Feature toggle
SUMMARIZE = bool(_APP_CFG.get("summarize", True))  # When False, only transcription will be performed

# Folder Paths
INPUT_FOLDER_PATH = str(_APP_CFG.get("input_folder_path", r"C:\\Users\\paulg\\OneDrive\\Desktop\\New Literature"))
OUTPUT_FOLDER_PATH = str(_APP_CFG.get("output_folder_path", r"C:\\Users\\paulg\\OneDrive\\Desktop\\New Literature"))

# Cleanup Settings:
DELETE_TEMP_WORKING_DIR = bool(_APP_CFG.get("delete_temp_working_dir", True))  # Delete the item's temporary working directory (including all temp files/images)

# Performance Settings
CONCURRENT_REQUESTS = int(_APP_CFG.get("concurrent_requests", 4))
API_TIMEOUT = int(_APP_CFG.get("api_timeout", 320))  # Timeout for each API call in seconds

# Rate Limiting Configuration (OpenAI)
# Format: List of (max_requests, time_period_seconds) tuples
OPENAI_RATE_LIMITS = _as_rate_limits(_OA.get("rate_limits"))

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
OPENAI_MODEL = str(_OA.get("model", "gpt-5-mini"))  # Summary/excerpting and default OpenAI model
OPENAI_TRANSCRIPTION_MODEL = str(_OA.get("transcription_model", "gpt-5-mini"))  # Transcription model
OPENAI_API_TIMEOUT = int(_OA.get("api_timeout", 900))  # 15 minutes for flex processing
OPENAI_USE_FLEX = bool(_OA.get("use_flex", True))  # Enable flex processing for cost savings
