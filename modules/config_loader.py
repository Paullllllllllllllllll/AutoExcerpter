"""Configuration loader for YAML-based settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Public API
__all__ = [
    "ConfigLoader",
    "PROJECT_ROOT",
    "MODULES_DIR",
    "CONFIG_DIR",
    "PROMPTS_DIR",
    "SCHEMAS_DIR",
]

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = Path(__file__).resolve().parent
CONFIG_DIR = MODULES_DIR / "config"
PROMPTS_DIR = MODULES_DIR / "prompts"
SCHEMAS_DIR = MODULES_DIR / "schemas"


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
        self._image_processing: Dict[str, Any] = {}
        self._concurrency: Dict[str, Any] = {}
        self._model: Dict[str, Any] = {}

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

    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """
        Load a single YAML configuration file.

        Args:
            filename: Name of the YAML file to load from CONFIG_DIR

        Returns:
            Configuration dictionary, or empty dict if loading fails
        """
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

    def get_image_processing_config(self) -> Dict[str, Any]:
        """
        Get the image processing configuration.

        Returns:
            Dictionary containing image processing settings
        """
        return dict(self._image_processing)

    def get_concurrency_config(self) -> Dict[str, Any]:
        """
        Get the concurrency configuration.

        Returns:
            Dictionary containing concurrency and service tier settings
        """
        return dict(self._concurrency)

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.

        Returns:
            Dictionary containing model-specific settings
        """
        return dict(self._model)

    def is_loaded(self) -> bool:
        """
        Check if configurations have been loaded.

        Returns:
            True if at least one configuration has been loaded
        """
        return bool(self._image_processing or self._concurrency or self._model)
