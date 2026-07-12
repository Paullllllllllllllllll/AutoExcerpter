"""Application configuration loader from YAML files.

This module loads configuration from config/defaults/app.yaml and exposes
settings as module-level constants. Configuration includes:
- Execution mode (CLI vs interactive)
- Feature toggles (summarization, cleanup)
- File paths (input/output directories)
- Citation management settings (OpenAlex integration)
- Daily token limits

Import as:
    from config import app as config

Configuration is loaded at module import time and exposed as constants:
    - CLI_MODE: bool
    - SUMMARIZE: bool
    - INPUT_FOLDER_PATH: str
    - OUTPUT_FOLDER_PATH: str
    - etc.

API keys are read from environment variables but validated lazily via
``require_api_key(provider)`` / ``get_available_providers()``. Import of
this module does NOT raise if no API keys are set; the check happens at
first actual use.

Note: Model configuration is centralized in model.yaml
Note: API concurrency settings (rate_limits, api_timeout, service_tier) are in
concurrency.yaml
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import yaml

from config.logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Configuration File Paths
# ============================================================================
_CONFIG_DIR = Path(__file__).resolve().parent
_APP_CONFIG_PATH = _CONFIG_DIR / "defaults" / "app.yaml"


# ============================================================================
# Configuration Loading Functions
# ============================================================================
def _load_yaml_app_config() -> dict[str, Any]:
    """Load the application config YAML.

    Resolution order:
    1. Real file (``config/defaults/app.yaml``).
    2. Bundled example (``config/defaults/app.example.yaml``) with an
       informational message prompting the user to copy and customize it.
    3. Neither present: return {} with a WARNING.

    Returns:
        Configuration dictionary, or empty dict on error
    """
    app_config_path = _APP_CONFIG_PATH

    if not app_config_path.exists():
        # Look for the bundled example next to the expected real file.
        stem = app_config_path.stem
        example_path = app_config_path.parent / f"{stem}.example.yaml"
        if example_path.exists():
            logger.info(
                f"Config '{app_config_path.name}' not found; using bundled "
                f"defaults from '{example_path.name}'. Copy it to "
                f"'{app_config_path.name}' and edit it to set your own values."
            )
            app_config_path = example_path
        else:
            logger.warning(
                f"App config file not found: {_APP_CONFIG_PATH}. Using defaults."
            )
            return {}

    try:
        with app_config_path.open("r", encoding="utf-8") as f:
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
_SHARED_TOKEN_BUDGET: dict[str, Any] = (
    _APP_CFG.get("shared_token_budget", {})
    if isinstance(_APP_CFG.get("shared_token_budget"), dict)
    else {}
)

# --- Execution Mode ---
CLI_MODE: bool = _get_bool(_APP_CFG, "cli_mode", False)

# --- Feature Toggles ---
SUMMARIZE: bool = _get_bool(_APP_CFG, "summarize", True)

# --- Summary Output Formats ---
OUTPUT_DOCX = _get_bool(_SUMMARY_OUTPUT, "docx", True)
OUTPUT_MARKDOWN = _get_bool(_SUMMARY_OUTPUT, "markdown", True)

# --- File Paths ---
INPUT_FOLDER_PATH = _get_str(_APP_CFG, "input_folder_path", "")
OUTPUT_FOLDER_PATH = _get_str(_APP_CFG, "output_folder_path", "")
INPUT_PATHS_IS_OUTPUT_PATH = _get_bool(_APP_CFG, "input_paths_is_output_path", False)

# --- Cleanup Settings ---
DELETE_TEMP_WORKING_DIR = _get_bool(_APP_CFG, "delete_temp_working_dir", True)

# --- Citation Management Settings ---
CITATION_OPENALEX_EMAIL = _get_str(
    _CITATION, "openalex_email", "your-email@example.com"
)
CITATION_MAX_API_REQUESTS = _get_int(_CITATION, "max_api_requests", 50)
CITATION_ENABLE_OPENALEX = _get_bool(_CITATION, "enable_openalex_enrichment", True)


def _get_float(data: dict[str, Any], key: str, default: float) -> float:
    """Safely get a float value from config dictionary."""
    try:
        return float(data.get(key, default))
    except (ValueError, TypeError):
        logger.warning(f"Invalid float for '{key}', using default: {default}")
        return default


# Conservative fuzzy-merge thresholds (decision 9: prefer under-merging).
# Variants merge only within a (first-author surname, year) block when EITHER
# similarity gate passes; volume differences never merge.
CITATION_MERGE_RATIO = _get_float(_CITATION, "merge_ratio", 0.90)
CITATION_MERGE_JACCARD = _get_float(_CITATION, "merge_jaccard", 0.85)
# Strict OpenAlex linking: title overlap must clear this AND a year/author check.
CITATION_MATCH_TITLE_OVERLAP = _get_float(_CITATION, "match_title_overlap", 0.5)

# --- State Directory (token-budget + OpenAlex cache) ---
# User-level by default (~/.autoexcerpter); override with paths.state_dir. A
# legacy CWD state file is adopted once when the user-level file is absent.
_PATHS: dict[str, Any] = (
    _APP_CFG.get("paths", {}) if isinstance(_APP_CFG.get("paths"), dict) else {}
)
STATE_DIR = _get_str(_PATHS, "state_dir", "")

# --- LLM Provider API Keys ---
# Read from environment. Validation is lazy: import of this module no longer
# raises when no API keys are set. Callers that need a key must use
# require_api_key(provider) or get_available_providers() at use time.
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY: str | None = os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY")

_PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def get_available_providers() -> list[str]:
    """Return the list of providers whose API key is currently set.

    Honors the optional ``api_keys.yaml`` remapping so that availability
    agrees with the key actually resolved at call time.
    """
    from config.loader import resolve_env_var

    return [
        name
        for name, env in _PROVIDER_ENV_VARS.items()
        if os.environ.get(resolve_env_var(name, env))
    ]


def require_api_key(provider: str) -> str:
    """Return the API key for *provider* or raise ``EnvironmentError`` if unset.

    Preferred call site: just before the first LLM invocation, so that
    configuration errors surface with a clear, actionable message rather than
    crashing at module import. The optional ``api_keys.yaml`` mapping may
    remap the provider to a different env-var name (defaults apply when
    omitted).
    """
    from config.loader import resolve_env_var

    default_env_var = _PROVIDER_ENV_VARS.get(provider.lower())
    if default_env_var is None:
        raise OSError(
            f"Unknown provider: {provider!r}. "
            f"Known providers: {sorted(_PROVIDER_ENV_VARS)}"
        )
    env_var = resolve_env_var(provider.lower(), default_env_var)
    key = os.environ.get(env_var)
    if not key:
        raise OSError(
            f"No API key for provider {provider!r}. "
            f"Please set the {env_var} environment variable."
        )
    return key


# --- Daily Token Limit ---
DAILY_TOKEN_LIMIT_ENABLED = _get_bool(_TOKEN_LIMIT, "enabled", False)
DAILY_TOKEN_LIMIT = _get_int(_TOKEN_LIMIT, "daily_tokens", 10000000)

# Combined-guard scope: "pooled" (default) enforces the combined daily_tokens
# cap only against pooled buckets and unstamped legacy usage; "all" enforces it
# against every stamped bucket too.
DAILY_TOKEN_LIMIT_SCOPE = _get_str(_TOKEN_LIMIT, "scope", "pooled")


def _parse_pool_caps(
    raw: Any,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, list[str]]]]:
    """Parse a ``per_key_pool_caps`` block into ``(caps, pool_models)``.

    Each pool entry under a provider is EITHER a bare integer (cap only; the
    pool's model list comes from the ledger built-ins) OR a mapping with the
    optional keys ``cap`` (daily token cap) and ``models`` (model-name prefixes
    defining the pool; configured lists replace the built-ins for that
    provider). A ``models`` list without a ``cap`` yields a tracked but
    uncapped pool. The ``enabled`` toggle is skipped; malformed entries are
    dropped silently so a hand-edited config never crashes the run.

    Returns:
        ``caps``: ``{provider: {pool_label: cap}}`` for entries carrying a cap.
        ``pool_models``: ``{provider: {pool_label: [prefix, ...]}}`` for
        entries carrying a ``models`` list (input to ``compile_pools``).
    """
    caps: dict[str, dict[str, int]] = {}
    pool_models: dict[str, dict[str, list[str]]] = {}
    if not isinstance(raw, dict):
        return caps, pool_models
    for provider, pools in raw.items():
        if provider == "enabled" or not isinstance(pools, dict):
            continue
        provider_caps: dict[str, int] = {}
        provider_models: dict[str, list[str]] = {}
        for pool, value in pools.items():
            label = str(pool)
            if isinstance(value, dict):
                cap_raw = value.get("cap")
                if cap_raw is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        provider_caps[label] = int(str(cap_raw).replace("_", ""))
                models_raw = value.get("models")
                if isinstance(models_raw, (list, tuple)):
                    prefixes = [
                        str(m).strip()
                        for m in models_raw
                        if isinstance(m, str) and m.strip()
                    ]
                    if prefixes:
                        provider_models[label] = prefixes
            else:
                try:
                    provider_caps[label] = int(str(value).replace("_", ""))
                except (ValueError, TypeError):
                    continue
        if provider_caps:
            caps[str(provider)] = provider_caps
        if provider_models:
            pool_models[str(provider)] = provider_models
    return caps, pool_models


# Per-key-pool caps and definitions (primary enforcement). Absent block ->
# defaults apply (enabled, built-in pools and DEFAULT_POOL_CAPS from the shared
# ledger). ``enabled: false`` turns the pool gate off entirely, leaving only
# the combined guard.
_PER_KEY_POOL_CAPS: dict[str, Any] = (
    _TOKEN_LIMIT.get("per_key_pool_caps", {})
    if isinstance(_TOKEN_LIMIT.get("per_key_pool_caps"), dict)
    else {}
)
PER_KEY_POOL_CAPS_ENABLED = _get_bool(_PER_KEY_POOL_CAPS, "enabled", True)
PER_KEY_POOL_CAPS, PER_KEY_POOL_MODELS = _parse_pool_caps(_PER_KEY_POOL_CAPS)


def reload_pool_settings() -> dict[str, Any] | None:
    """Re-read ``scope`` and ``per_key_pool_caps`` fresh from ``app.yaml``.

    Bypasses the import-time constants so a user editing the config while a
    token-limit wait loop is polling changes the combined scope, toggles the
    pool gate, or remaps caps and pool definitions without a restart. Returns
    ``None`` when the file cannot be read (callers keep current values).
    Degrades gracefully: never raises.
    """
    try:
        data = _load_yaml_app_config()
        token_limit = data.get("daily_token_limit", {})
        if not isinstance(token_limit, dict):
            return None
        raw_caps = token_limit.get("per_key_pool_caps", {})
        enabled = (
            _get_bool(raw_caps, "enabled", True) if isinstance(raw_caps, dict) else True
        )
        caps, pool_models = _parse_pool_caps(raw_caps)
        return {
            "scope": str(token_limit.get("scope", "pooled")),
            "pool_caps_enabled": enabled,
            "pool_caps": caps,
            "pool_models": pool_models,
        }
    except Exception:
        return None


def reload_daily_token_limit() -> int | None:
    """Re-read ``daily_token_limit.daily_tokens`` fresh from ``app.yaml``.

    Bypasses the import-time module constant so a user editing the config while
    a token-limit wait loop is polling lifts (or lowers) the cap without a
    restart. Returns ``None`` when the value is absent or the file cannot be
    read, so callers keep the current limit on failure. Degrades gracefully:
    never raises.
    """
    try:
        data = _load_yaml_app_config()
        token_limit = data.get("daily_token_limit", {})
        if not isinstance(token_limit, dict):
            return None
        raw = token_limit.get("daily_tokens")
        if raw is None:
            return None
        return int(str(raw).replace("_", ""))
    except Exception:
        return None


# Chunk/page-level enforcement tuning (see llm/token_tracker.DailyTokenTracker).
DAILY_TOKEN_CHUNK_ESTIMATE_SEED = _get_int(_TOKEN_LIMIT, "chunk_estimate_seed", 25000)
DAILY_TOKEN_ESTIMATE_SMOOTHING = float(_TOKEN_LIMIT.get("estimate_smoothing", 0.3))

# --- Shared Cross-Tool Token Budget (opt-in) ---
# When enabled, the daily token limit is enforced against the COMBINED usage of
# every participating ChronoPipeline tool (ChronoMiner, ChronoTranscriber,
# AutoExcerpter) via a shared on-disk ledger, and that ledger replaces the
# private token-state file as persistence. An empty ledger_dir means the ledger
# default (~/.chronopipeline). Disabled by default: bit-for-bit the private
# per-tool tracker with zero ledger I/O.
SHARED_TOKEN_BUDGET_ENABLED = _get_bool(_SHARED_TOKEN_BUDGET, "enabled", False)
SHARED_TOKEN_BUDGET_LEDGER_DIR = _get_str(_SHARED_TOKEN_BUDGET, "ledger_dir", "")

# ============================================================================
# Logging
# ============================================================================
logger.debug(f"Configuration loaded: CLI_MODE={CLI_MODE}, SUMMARIZE={SUMMARIZE}")
logger.debug("Model config in model.yaml, API concurrency in concurrency.yaml")
logger.debug(
    f"Daily token limit: enabled={DAILY_TOKEN_LIMIT_ENABLED}, limit={DAILY_TOKEN_LIMIT}"
)
