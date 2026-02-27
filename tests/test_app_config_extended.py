"""Extended tests for modules/app_config.py - coverage gap filling.

Covers the private helper functions that are loaded at import time:
- _load_yaml_app_config: file exists, file missing, invalid YAML, non-dict YAML
- _get_str: normal value, None value, default
- _get_int: normal int, invalid value falls back to default
- _get_bool: truthy/falsy values

Because app_config runs module-level code that checks API keys, these tests
import the helper functions only after the module has already been loaded
(the test environment must have at least one API key set, which conftest.py
handles via mock_api_keys or the real environment).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml


# ---------------------------------------------------------------------------
# Direct helper function tests
# ---------------------------------------------------------------------------
# We import the module (which triggers the top-level code) and then exercise
# the private helpers.  The import succeeds because the CI/test environment
# has at least one API key in the environment (or the conftest sets one).
from modules.app_config import (
    _load_yaml_app_config,
    _get_str,
    _get_int,
    _get_bool,
)


# ============================================================================
# _load_yaml_app_config
# ============================================================================
class TestLoadYamlAppConfig:
    """Tests for _load_yaml_app_config()."""

    def test_file_exists_valid_yaml(self, tmp_path: Path, monkeypatch):
        """Returns parsed dict when a valid YAML file exists."""
        yaml_content = {"cli_mode": True, "summarize": False}
        config_file = tmp_path / "app.yaml"
        config_file.write_text(yaml.dump(yaml_content), encoding="utf-8")

        import modules.app_config as ac
        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", config_file)

        result = _load_yaml_app_config()
        assert isinstance(result, dict)
        assert result["cli_mode"] is True
        assert result["summarize"] is False

    def test_file_missing_returns_empty_dict(self, tmp_path: Path, monkeypatch):
        """Returns empty dict when the config file does not exist."""
        nonexistent = tmp_path / "does_not_exist.yaml"

        import modules.app_config as ac
        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", nonexistent)

        result = _load_yaml_app_config()
        assert result == {}

    def test_invalid_yaml_returns_empty_dict(self, tmp_path: Path, monkeypatch):
        """Returns empty dict when the YAML is malformed."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(":::not: valid: yaml: [[[", encoding="utf-8")

        import modules.app_config as ac
        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", bad_yaml)

        result = _load_yaml_app_config()
        assert result == {}

    def test_non_dict_yaml_returns_empty_dict(self, tmp_path: Path, monkeypatch):
        """Returns empty dict when the YAML root is a list instead of a dict."""
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

        import modules.app_config as ac
        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", list_yaml)

        result = _load_yaml_app_config()
        assert result == {}


# ============================================================================
# _get_str
# ============================================================================
class TestGetStr:
    """Tests for _get_str()."""

    def test_normal_value(self):
        """Returns the value as a string when key is present."""
        data = {"key": "hello"}
        assert _get_str(data, "key", "default") == "hello"

    def test_none_value_returns_default(self):
        """Returns default when value is None."""
        data = {"key": None}
        assert _get_str(data, "key", "fallback") == "fallback"

    def test_missing_key_returns_default(self):
        """Returns default when key is absent."""
        data = {}
        assert _get_str(data, "missing", "fallback") == "fallback"

    def test_integer_value_converted_to_string(self):
        """Non-string values are converted via str()."""
        data = {"key": 42}
        assert _get_str(data, "key", "default") == "42"

    def test_empty_string_value(self):
        """Empty string is returned as is (not replaced by default)."""
        data = {"key": ""}
        assert _get_str(data, "key", "default") == ""


# ============================================================================
# _get_int
# ============================================================================
class TestGetInt:
    """Tests for _get_int()."""

    def test_normal_int(self):
        """Returns the integer when key holds a valid int."""
        data = {"key": 10}
        assert _get_int(data, "key", 99) == 10

    def test_string_int(self):
        """Returns parsed int when key holds a numeric string."""
        data = {"key": "42"}
        assert _get_int(data, "key", 0) == 42

    def test_invalid_value_returns_default(self):
        """Returns default when value cannot be converted to int."""
        data = {"key": "not_a_number"}
        assert _get_int(data, "key", 77) == 77

    def test_none_value_returns_default(self):
        """Returns default when value is None (TypeError)."""
        data = {"key": None}
        assert _get_int(data, "key", 5) == 5

    def test_missing_key_returns_default(self):
        """Returns default when key is absent."""
        data = {}
        assert _get_int(data, "missing", 123) == 123

    def test_float_truncated_to_int(self):
        """Float values are truncated to int."""
        data = {"key": 3.9}
        assert _get_int(data, "key", 0) == 3


# ============================================================================
# _get_bool
# ============================================================================
class TestGetBool:
    """Tests for _get_bool()."""

    def test_true_value(self):
        """Returns True for True value."""
        data = {"key": True}
        assert _get_bool(data, "key", False) is True

    def test_false_value(self):
        """Returns False for False value."""
        data = {"key": False}
        assert _get_bool(data, "key", True) is False

    def test_truthy_string(self):
        """Non-empty strings are truthy."""
        data = {"key": "yes"}
        assert _get_bool(data, "key", False) is True

    def test_falsy_empty_string(self):
        """Empty string is falsy."""
        data = {"key": ""}
        assert _get_bool(data, "key", True) is False

    def test_truthy_int(self):
        """Non-zero int is truthy."""
        data = {"key": 1}
        assert _get_bool(data, "key", False) is True

    def test_falsy_zero(self):
        """Zero is falsy."""
        data = {"key": 0}
        assert _get_bool(data, "key", True) is False

    def test_none_value_is_falsy(self):
        """None is falsy."""
        data = {"key": None}
        assert _get_bool(data, "key", True) is False

    def test_missing_key_returns_default_true(self):
        """Missing key returns the default value (True)."""
        data = {}
        assert _get_bool(data, "missing", True) is True

    def test_missing_key_returns_default_false(self):
        """Missing key returns the default value (False)."""
        data = {}
        assert _get_bool(data, "missing", False) is False
