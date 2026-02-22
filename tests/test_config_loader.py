"""Tests for modules/config_loader.py - Configuration loading utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from modules.config_loader import (
    ConfigLoader,
    get_config_loader,
    PROJECT_ROOT,
    MODULES_DIR,
    CONFIG_DIR,
    PROMPTS_DIR,
    SCHEMAS_DIR,
)


class TestPathConstants:
    """Tests for path constants."""

    def test_project_root_exists(self):
        """PROJECT_ROOT path exists."""
        assert PROJECT_ROOT.exists()

    def test_modules_dir_exists(self):
        """MODULES_DIR path exists."""
        assert MODULES_DIR.exists()

    def test_config_dir_exists(self):
        """CONFIG_DIR path exists."""
        assert CONFIG_DIR.exists()

    def test_prompts_dir_exists(self):
        """PROMPTS_DIR path exists."""
        assert PROMPTS_DIR.exists()

    def test_schemas_dir_exists(self):
        """SCHEMAS_DIR path exists."""
        assert SCHEMAS_DIR.exists()

    def test_config_dir_is_under_modules(self):
        """CONFIG_DIR is under MODULES_DIR."""
        assert CONFIG_DIR.parent == MODULES_DIR


class TestConfigLoaderInit:
    """Tests for ConfigLoader initialization."""

    def test_init_creates_empty_configs(self):
        """Initialization creates empty config dictionaries."""
        loader = ConfigLoader()
        
        assert loader._image_processing == {}
        assert loader._concurrency == {}
        assert loader._model == {}

    def test_is_loaded_false_initially(self):
        """is_loaded returns False before loading."""
        loader = ConfigLoader()
        
        assert loader.is_loaded() is False


class TestConfigLoaderLoadConfigs:
    """Tests for load_configs method."""

    def test_load_configs_populates_dictionaries(self):
        """load_configs populates configuration dictionaries."""
        loader = ConfigLoader()
        loader.load_configs()
        
        # At least one config should be loaded
        assert loader.is_loaded() is True

    def test_load_configs_handles_missing_file(self, temp_dir: Path):
        """load_configs handles missing config files gracefully."""
        loader = ConfigLoader()
        
        with patch.object(loader, "_load_yaml_config", return_value={}):
            loader.load_configs()
        
        # Should not raise, should have empty configs
        assert loader._image_processing == {}

    def test_load_yaml_config_returns_dict(self, temp_dir: Path):
        """_load_yaml_config returns dictionary from valid YAML."""
        # Create a temporary config file
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text("key: value\nnested:\n  subkey: subvalue")
        
        loader = ConfigLoader()
        
        with patch("modules.config_loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("test_config.yaml")
        
        assert result == {"key": "value", "nested": {"subkey": "subvalue"}}

    def test_load_yaml_config_handles_invalid_yaml(self, temp_dir: Path):
        """_load_yaml_config handles invalid YAML gracefully."""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text("key: [unclosed bracket")
        
        loader = ConfigLoader()
        
        with patch("modules.config_loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("invalid.yaml")
        
        assert result == {}

    def test_load_yaml_config_handles_non_dict(self, temp_dir: Path):
        """_load_yaml_config handles non-dict YAML content."""
        config_path = temp_dir / "list.yaml"
        config_path.write_text("- item1\n- item2")
        
        loader = ConfigLoader()
        
        with patch("modules.config_loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("list.yaml")
        
        assert result == {}


class TestConfigLoaderGetters:
    """Tests for configuration getter methods."""

    def test_get_image_processing_config(self):
        """get_image_processing_config returns copy of config."""
        loader = ConfigLoader()
        loader._image_processing = {"key": "value"}
        
        result = loader.get_image_processing_config()
        
        assert result == {"key": "value"}
        # Should be a copy
        result["new_key"] = "new_value"
        assert "new_key" not in loader._image_processing

    def test_get_concurrency_config(self):
        """get_concurrency_config returns copy of config."""
        loader = ConfigLoader()
        loader._concurrency = {"limit": 100}
        
        result = loader.get_concurrency_config()
        
        assert result == {"limit": 100}

    def test_get_model_config(self):
        """get_model_config returns copy of config."""
        loader = ConfigLoader()
        loader._model = {"name": "gpt-5"}
        
        result = loader.get_model_config()
        
        assert result == {"name": "gpt-5"}


class TestModelOverrides:
    """Tests for runtime model override application."""

    def test_apply_model_overrides_deep_merge(self):
        """apply_model_overrides deep-merges nested dictionaries."""
        loader = ConfigLoader()
        loader._model = {
            "transcription_model": {
                "name": "gpt-5-mini",
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "medium"},
            },
            "summary_model": {
                "name": "gpt-5-mini",
                "max_output_tokens": 4096,
            },
        }

        loader.apply_model_overrides(
            {
                "transcription_model": {"reasoning": {"effort": "high"}},
                "summary_model": {"max_output_tokens": 8192},
            }
        )

        merged = loader.get_model_config()
        assert merged["transcription_model"]["name"] == "gpt-5-mini"
        assert merged["transcription_model"]["reasoning"]["effort"] == "high"
        assert merged["transcription_model"]["text"]["verbosity"] == "medium"
        assert merged["summary_model"]["max_output_tokens"] == 8192

    def test_apply_model_overrides_ignores_empty(self):
        """apply_model_overrides ignores empty override payloads."""
        loader = ConfigLoader()
        loader._model = {"transcription_model": {"name": "gpt-5-mini"}}

        loader.apply_model_overrides({})

        result = loader.get_model_config()
        assert result["transcription_model"]["name"] == "gpt-5-mini"


class TestGetConfigLoader:
    """Tests for get_config_loader singleton function."""

    def test_returns_config_loader(self):
        """Returns a ConfigLoader instance."""
        # Reset singleton for test
        import modules.config_loader as config_module
        config_module._config_loader_instance = None
        
        loader = get_config_loader()
        
        assert isinstance(loader, ConfigLoader)

    def test_returns_same_instance(self):
        """Returns same instance on repeated calls."""
        # Reset singleton for test
        import modules.config_loader as config_module
        config_module._config_loader_instance = None
        
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        
        assert loader1 is loader2

    def test_instance_is_loaded(self):
        """Singleton instance has configs loaded."""
        # Reset singleton for test
        import modules.config_loader as config_module
        config_module._config_loader_instance = None
        
        loader = get_config_loader()
        
        assert loader.is_loaded() is True


class TestConfigLoaderIntegration:
    """Integration tests using actual config files."""

    def test_loads_image_processing_yaml(self):
        """Loads actual image_processing.yaml file."""
        loader = ConfigLoader()
        loader.load_configs()
        
        config = loader.get_image_processing_config()
        
        # Should have some expected keys (if file exists)
        if config:
            assert isinstance(config, dict)

    def test_loads_concurrency_yaml(self):
        """Loads actual concurrency.yaml file."""
        loader = ConfigLoader()
        loader.load_configs()
        
        config = loader.get_concurrency_config()
        
        if config:
            assert isinstance(config, dict)

    def test_loads_model_yaml(self):
        """Loads actual model.yaml file."""
        loader = ConfigLoader()
        loader.load_configs()
        
        config = loader.get_model_config()
        
        if config:
            assert isinstance(config, dict)
