"""Configuration loader for YAML-based settings.

This module provides centralized configuration loading for the AutoExcerpter application.
It loads YAML configuration files from the modules/config/ directory and provides
type-safe access to configuration values.

Supported Configuration Files:
1. **image_processing.yaml**: Image preprocessing settings (DPI, quality, resize, etc.)
2. **concurrency.yaml**: API concurrency, service tiers, and retry configuration
3. **model.yaml**: Model-specific parameters (GPT-5 settings, reasoning effort, etc.)

Usage Pattern:
    >>> from modules.config_loader import ConfigLoader
    >>> loader = ConfigLoader()
    >>> loader.load_configs()
    >>> img_config = loader.get_image_processing_config()
    >>> concurrency_config = loader.get_concurrency_config()
    >>> model_config = loader.get_model_config()

Path Constants:
- PROJECT_ROOT: Root directory of the project
- MODULES_DIR: modules/ directory
- CONFIG_DIR: modules/config/ directory
- PROMPTS_DIR: modules/prompts/ directory
- SCHEMAS_DIR: modules/schemas/ directory

The loader handles missing files gracefully, returning empty dictionaries and logging
warnings when configuration files are not found or contain invalid YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from modules.logger import setup_logger

logger = setup_logger(__name__)

# ============================================================================
# Path Resolution
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = Path(__file__).resolve().parent
CONFIG_DIR = MODULES_DIR / "config"
PROMPTS_DIR = MODULES_DIR / "prompts"
SCHEMAS_DIR = MODULES_DIR / "schemas"


# ============================================================================
# Configuration Loader Class
# ============================================================================
class ConfigLoader:
    """
    Lightweight loader for YAML/JSON configs residing under modules/.

    This class loads configuration files for image processing, concurrency,
    and model settings from the modules/config/ directory.
    
    Example:
        >>> loader = ConfigLoader()
        >>> loader.load_configs()
        >>> img_config = loader.get_image_processing_config()
    """

    def __init__(self) -> None:
        """Initialize the configuration loader with empty config dictionaries."""
        self._image_processing: dict[str, Any] = {}
        self._concurrency: dict[str, Any] = {}
        self._model: dict[str, Any] = {}

    def load_configs(self) -> None:
        """
        Load all configuration files from the config directory.

        This method loads:
        - image_processing.yaml: Image preprocessing settings
        - concurrency.yaml: Concurrency and service tier settings
        - model.yaml: Model-specific settings (optional)

        Errors during loading are logged but do not raise exceptions.
        """
        self._image_processing = self._load_yaml_config("image_processing.yaml")
        self._concurrency = self._load_yaml_config("concurrency.yaml")
        self._model = self._load_yaml_config("model.yaml")

    def _load_yaml_config(self, filename: str) -> dict[str, Any]:
        """Load a single YAML configuration file."""
        config_path = CONFIG_DIR / filename
        
        if not config_path.exists():
            logger.debug(f"Config file not found: {config_path}")
            return {}

        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                
            if not isinstance(data, dict):
                logger.warning(f"Config file {filename} did not contain a dictionary. Using empty config.")
                return {}
                
            return data
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {filename}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config {filename}: {e}")
            return {}

    def get_image_processing_config(self) -> dict[str, Any]:
        """Get the image processing configuration."""
        return dict(self._image_processing)

    def get_concurrency_config(self) -> dict[str, Any]:
        """Get the concurrency configuration."""
        return dict(self._concurrency)

    def get_model_config(self) -> dict[str, Any]:
        """Get the model configuration."""
        return dict(self._model)

    def is_loaded(self) -> bool:
        """Check if configurations have been loaded."""
        return bool(self._image_processing or self._concurrency or self._model)


# ============================================================================
# Singleton Pattern for Config Loader
# ============================================================================
_config_loader_instance: ConfigLoader | None = None


def get_config_loader() -> ConfigLoader:
    """Get or create a singleton ConfigLoader instance."""
    global _config_loader_instance
    
    if _config_loader_instance is None:
        _config_loader_instance = ConfigLoader()
        _config_loader_instance.load_configs()
        logger.debug("Initialized singleton ConfigLoader")
    
    return _config_loader_instance


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "ConfigLoader",
    "get_config_loader",
    "PROJECT_ROOT",
    "MODULES_DIR",
    "CONFIG_DIR",
    "PROMPTS_DIR",
    "SCHEMAS_DIR",
]
