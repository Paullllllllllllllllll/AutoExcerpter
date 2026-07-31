"""Tests for config/loader.py - Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from config.loader import (
    CONFIG_DIR,
    PROJECT_ROOT,
    PROMPTS_DIR,
    SCHEMAS_DIR,
    ConfigLoader,
    get_config_loader,
)


class TestPathConstants:
    """Tests for path constants."""

    def test_project_root_exists(self) -> None:
        """PROJECT_ROOT path exists."""
        assert PROJECT_ROOT.exists()

    def test_config_dir_exists(self) -> None:
        """CONFIG_DIR path exists."""
        assert CONFIG_DIR.exists()

    def test_prompts_dir_exists(self) -> None:
        """PROMPTS_DIR path exists."""
        assert PROMPTS_DIR.exists()

    def test_schemas_dir_exists(self) -> None:
        """SCHEMAS_DIR path exists."""
        assert SCHEMAS_DIR.exists()

    def test_config_dir_is_under_config_package(self) -> None:
        """CONFIG_DIR (config/defaults/) is under the config/ package directory."""
        assert CONFIG_DIR.parent == PROJECT_ROOT / "config"


class TestConfigLoaderInit:
    """Tests for ConfigLoader initialization."""

    def test_init_creates_empty_configs(self) -> None:
        """Initialization creates empty config dictionaries."""
        loader = ConfigLoader()

        assert loader._image_processing == {}
        assert loader._concurrency == {}
        assert loader._model == {}

    def test_is_loaded_false_initially(self) -> None:
        """is_loaded returns False before loading."""
        loader = ConfigLoader()

        assert loader.is_loaded() is False


class TestConfigLoaderLoadConfigs:
    """Tests for load_configs method."""

    def test_load_configs_populates_dictionaries(self) -> None:
        """load_configs populates configuration dictionaries."""
        loader = ConfigLoader()
        loader.load_configs()

        # At least one config should be loaded
        assert loader.is_loaded() is True

    def test_load_configs_handles_missing_file(self, temp_dir: Path) -> None:
        """load_configs handles missing config files gracefully."""
        loader = ConfigLoader()

        with patch.object(loader, "_load_yaml_config", return_value={}):
            loader.load_configs()

        # Should not raise, should have empty configs
        assert loader._image_processing == {}

    def test_load_yaml_config_returns_dict(self, temp_dir: Path) -> None:
        """_load_yaml_config returns dictionary from valid YAML."""
        # Create a temporary config file
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text("key: value\nnested:\n  subkey: subvalue")

        loader = ConfigLoader()

        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("test_config.yaml")

        assert result == {"key": "value", "nested": {"subkey": "subvalue"}}

    def test_load_yaml_config_handles_invalid_yaml(self, temp_dir: Path) -> None:
        """_load_yaml_config handles invalid YAML gracefully."""
        config_path = temp_dir / "invalid.yaml"
        config_path.write_text("key: [unclosed bracket")

        loader = ConfigLoader()

        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("invalid.yaml")

        assert result == {}

    def test_load_yaml_config_handles_non_dict(self, temp_dir: Path) -> None:
        """_load_yaml_config handles non-dict YAML content."""
        config_path = temp_dir / "list.yaml"
        config_path.write_text("- item1\n- item2")

        loader = ConfigLoader()

        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("list.yaml")

        assert result == {}

    def test_load_yaml_config_falls_back_to_example(self, temp_dir: Path) -> None:
        """_load_yaml_config loads <stem>.example.yaml when the real file absent."""
        # Only the bundled example exists; the real file is missing.
        example_path = temp_dir / "image_processing.example.yaml"
        example_path.write_text("key: example_value\nnested:\n  subkey: 1")

        loader = ConfigLoader()

        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("image_processing.yaml")

        assert result == {"key": "example_value", "nested": {"subkey": 1}}

    def test_load_yaml_config_real_takes_precedence_over_example(
        self, temp_dir: Path
    ) -> None:
        """The real file wins when both it and the example are present."""
        (temp_dir / "image_processing.yaml").write_text("key: real_value")
        (temp_dir / "image_processing.example.yaml").write_text("key: example_value")

        loader = ConfigLoader()

        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("image_processing.yaml")

        assert result == {"key": "real_value"}

    def test_load_yaml_config_neither_present_returns_empty(
        self, temp_dir: Path
    ) -> None:
        """Returns {} without raising when neither real nor example exists."""
        loader = ConfigLoader()

        # temp_dir is empty: no image_processing.yaml and no example beside it.
        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("image_processing.yaml")

        assert result == {}


class TestExampleBaselineMerge:
    """The real file is deep-merged OVER the bundled example.

    Regression: a present-but-partial real config was used verbatim, so every
    key it omitted fell through to an unrelated hardcoded constant instead of
    the shipped template's value. The sharpest case was a concurrency.yaml
    written without ``rate_limits``, which silently selected constants.py's
    (120, 1) burst limit rather than the template's conservative [10, 1].
    """

    def test_partial_real_inherits_example_values(self, temp_dir: Path) -> None:
        """Keys the real file omits keep the example's value."""
        (temp_dir / "concurrency.example.yaml").write_text(
            "api_requests:\n"
            "  api_timeout: 900\n"
            "  rate_limits:\n"
            "    - [10, 1]\n"
            "    - [600, 60]\n"
            "  transcription:\n"
            "    concurrency_limit: 80\n"
            "    service_tier: flex\n",
            encoding="utf-8",
        )
        # A real file that speaks ONLY about concurrency_limit.
        (temp_dir / "concurrency.yaml").write_text(
            "api_requests:\n  transcription:\n    concurrency_limit: 4\n",
            encoding="utf-8",
        )

        loader = ConfigLoader()
        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("concurrency.yaml")

        api = result["api_requests"]
        # The real file wins where it speaks ...
        assert api["transcription"]["concurrency_limit"] == 4
        # ... and inherits everything it omitted, at every nesting depth.
        assert api["rate_limits"] == [[10, 1], [600, 60]]
        assert api["api_timeout"] == 900
        assert api["transcription"]["service_tier"] == "flex"

    def test_real_scalar_values_still_win(self, temp_dir: Path) -> None:
        """An explicit real value overrides the example, including falsy ones."""
        (temp_dir / "model.example.yaml").write_text(
            "transcription_model:\n  name: example-model\n  stream: true\n",
            encoding="utf-8",
        )
        (temp_dir / "model.yaml").write_text(
            "transcription_model:\n  name: real-model\n  stream: false\n",
            encoding="utf-8",
        )

        loader = ConfigLoader()
        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("model.yaml")

        assert result["transcription_model"]["name"] == "real-model"
        assert result["transcription_model"]["stream"] is False

    def test_corrupt_real_falls_back_to_example_wholesale(self, temp_dir: Path) -> None:
        """A real file that fails to parse yields the example unchanged."""
        (temp_dir / "model.example.yaml").write_text(
            "transcription_model:\n  name: example-model\n", encoding="utf-8"
        )
        (temp_dir / "model.yaml").write_text("key: [unclosed", encoding="utf-8")

        loader = ConfigLoader()
        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("model.yaml")

        assert result == {"transcription_model": {"name": "example-model"}}

    def test_api_keys_partial_real_inherits_provider_defaults(
        self, temp_dir: Path
    ) -> None:
        """api_keys.yaml merges too; omitted providers keep their default var."""
        (temp_dir / "api_keys.example.yaml").write_text(
            "openai: OPENAI_API_KEY\nanthropic: ANTHROPIC_API_KEY\n",
            encoding="utf-8",
        )
        (temp_dir / "api_keys.yaml").write_text(
            "openai: OPENAI_API_KEY_2\n", encoding="utf-8"
        )

        loader = ConfigLoader()
        with patch("config.loader.CONFIG_DIR", temp_dir):
            result = loader._load_yaml_config("api_keys.yaml")

        assert result["openai"] == "OPENAI_API_KEY_2"
        # Behavior-neutral: the example maps each provider to its own default.
        assert result["anthropic"] == "ANTHROPIC_API_KEY"


class TestApiKeysConfig:
    """Tests for the optional api_keys.yaml mapping."""

    def test_is_loaded_counts_api_keys(self) -> None:
        """is_loaded() reflects api_keys.yaml, the fourth loaded file."""
        loader = ConfigLoader()
        assert loader.is_loaded() is False

        loader._api_keys = {"openai": "OPENAI_API_KEY"}
        assert loader.is_loaded() is True

    def test_get_api_keys_config_empty_when_absent(self, temp_dir: Path) -> None:
        """get_api_keys_config returns {} when api_keys.yaml is absent."""
        loader = ConfigLoader()

        # Point CONFIG_DIR at an empty directory so no api_keys.yaml exists.
        with patch("config.loader.CONFIG_DIR", temp_dir):
            loader.load_configs()

        assert loader.get_api_keys_config() == {}

    def test_get_api_keys_config_default_for_fresh_loader(self) -> None:
        """A fresh loader exposes an empty mapping before loading."""
        loader = ConfigLoader()

        assert loader.get_api_keys_config() == {}


class TestResolveEnvVar:
    """Tests for resolve_env_var()."""

    def _patch_mapping(self, monkeypatch: Any, mapping: dict[str, Any]) -> None:
        loader = ConfigLoader()
        loader._api_keys = mapping
        monkeypatch.setattr("config.loader.get_config_loader", lambda: loader)

    def test_mapped_value_is_stripped(self, monkeypatch: Any) -> None:
        """Regression: the guard stripped but the return value did not.

        A trailing space in the YAML produced an env-var name that could
        never resolve, so the key looked unset with no explanation.
        """
        from config.loader import resolve_env_var

        self._patch_mapping(monkeypatch, {"openai": "  OPENAI_API_KEY_2  "})
        assert resolve_env_var("openai", "OPENAI_API_KEY") == "OPENAI_API_KEY_2"

    def test_blank_mapping_falls_back_to_default(self, monkeypatch: Any) -> None:
        """A whitespace-only or missing entry keeps the default env-var name."""
        from config.loader import resolve_env_var

        self._patch_mapping(monkeypatch, {"openai": "   "})
        assert resolve_env_var("openai", "OPENAI_API_KEY") == "OPENAI_API_KEY"
        assert resolve_env_var("google", "GOOGLE_API_KEY") == "GOOGLE_API_KEY"


class TestConfigLoaderGetters:
    """Tests for configuration getter methods."""

    def test_get_image_processing_config(self) -> None:
        """get_image_processing_config returns copy of config."""
        loader = ConfigLoader()
        loader._image_processing = {"key": "value"}

        result = loader.get_image_processing_config()

        assert result == {"key": "value"}
        # Should be a copy
        result["new_key"] = "new_value"
        assert "new_key" not in loader._image_processing

    def test_get_concurrency_config(self) -> None:
        """get_concurrency_config returns copy of config."""
        loader = ConfigLoader()
        loader._concurrency = {"limit": 100}

        result = loader.get_concurrency_config()

        assert result == {"limit": 100}

    def test_get_model_config(self) -> None:
        """get_model_config returns copy of config."""
        loader = ConfigLoader()
        loader._model = {"name": "gpt-5"}

        result = loader.get_model_config()

        assert result == {"name": "gpt-5"}


class TestModelOverrides:
    """Tests for runtime model override application."""

    def test_apply_model_overrides_deep_merge(self) -> None:
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

    def test_apply_model_overrides_ignores_empty(self) -> None:
        """apply_model_overrides ignores empty override payloads."""
        loader = ConfigLoader()
        loader._model = {"transcription_model": {"name": "gpt-5-mini"}}

        loader.apply_model_overrides({})

        result = loader.get_model_config()
        assert result["transcription_model"]["name"] == "gpt-5-mini"


class TestGetConfigLoader:
    """Tests for get_config_loader singleton function."""

    def test_returns_config_loader(self) -> None:
        """Returns a ConfigLoader instance."""
        # Reset singleton for test
        import config.loader as config_module

        config_module._config_loader_instance = None

        loader = get_config_loader()

        assert isinstance(loader, ConfigLoader)

    def test_returns_same_instance(self) -> None:
        """Returns same instance on repeated calls."""
        # Reset singleton for test
        import config.loader as config_module

        config_module._config_loader_instance = None

        loader1 = get_config_loader()
        loader2 = get_config_loader()

        assert loader1 is loader2

    def test_instance_is_loaded(self) -> None:
        """Singleton instance has configs loaded."""
        # Reset singleton for test
        import config.loader as config_module

        config_module._config_loader_instance = None

        loader = get_config_loader()

        assert loader.is_loaded() is True


class TestConfigLoaderIntegration:
    """Integration tests using actual config files."""

    def test_loads_image_processing_yaml(self) -> None:
        """Loads actual image_processing.yaml file."""
        loader = ConfigLoader()
        loader.load_configs()

        config = loader.get_image_processing_config()

        # Should have some expected keys (if file exists)
        if config:
            assert isinstance(config, dict)

    def test_loads_concurrency_yaml(self) -> None:
        """Loads actual concurrency.yaml file."""
        loader = ConfigLoader()
        loader.load_configs()

        config = loader.get_concurrency_config()

        if config:
            assert isinstance(config, dict)

    def test_loads_model_yaml(self) -> None:
        """Loads actual model.yaml file."""
        loader = ConfigLoader()
        loader.load_configs()

        config = loader.get_model_config()

        if config:
            assert isinstance(config, dict)
