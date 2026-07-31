"""Extended tests for config/app.py - coverage gap filling.

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
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Direct helper function tests
# ---------------------------------------------------------------------------
# We import the module (which triggers the top-level code) and then exercise
# the private helpers.  The import succeeds because the CI/test environment
# has at least one API key in the environment (or the conftest sets one).
from config.app import (
    _get_bool,
    _get_int,
    _get_str,
    _load_yaml_app_config,
)


# ============================================================================
# _load_yaml_app_config
# ============================================================================
class TestLoadYamlAppConfig:
    """Tests for _load_yaml_app_config()."""

    def test_file_exists_valid_yaml(self, tmp_path: Path, monkeypatch) -> None:
        """Returns parsed dict when a valid YAML file exists."""
        yaml_content = {"cli_mode": True, "summarize": False}
        config_file = tmp_path / "app.yaml"
        config_file.write_text(yaml.dump(yaml_content), encoding="utf-8")

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", config_file)

        result = _load_yaml_app_config()
        assert isinstance(result, dict)
        assert result["cli_mode"] is True
        assert result["summarize"] is False

    def test_file_missing_returns_empty_dict(self, tmp_path: Path, monkeypatch) -> None:
        """Returns empty dict when the config file does not exist."""
        nonexistent = tmp_path / "does_not_exist.yaml"

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", nonexistent)

        result = _load_yaml_app_config()
        assert result == {}

    def test_falls_back_to_example(self, tmp_path: Path, monkeypatch) -> None:
        """Loads app.example.yaml when the real app.yaml is absent."""
        yaml_content = {"cli_mode": False, "summarize": True}
        # Only the bundled example exists next to the expected real file.
        (tmp_path / "app.example.yaml").write_text(
            yaml.dump(yaml_content), encoding="utf-8"
        )
        real_path = tmp_path / "app.yaml"

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", real_path)

        result = _load_yaml_app_config()
        assert isinstance(result, dict)
        assert result["cli_mode"] is False
        assert result["summarize"] is True

    def test_real_takes_precedence_over_example(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The real app.yaml wins when both it and the example are present."""
        real_path = tmp_path / "app.yaml"
        real_path.write_text(yaml.dump({"cli_mode": True}), encoding="utf-8")
        (tmp_path / "app.example.yaml").write_text(
            yaml.dump({"cli_mode": False}), encoding="utf-8"
        )

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", real_path)

        result = _load_yaml_app_config()
        assert result["cli_mode"] is True

    def test_invalid_yaml_returns_empty_dict(self, tmp_path: Path, monkeypatch) -> None:
        """Returns empty dict when the YAML is malformed."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(":::not: valid: yaml: [[[", encoding="utf-8")

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", bad_yaml)

        result = _load_yaml_app_config()
        assert result == {}

    def test_non_dict_yaml_returns_empty_dict(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Returns empty dict when the YAML root is a list instead of a dict."""
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", list_yaml)

        result = _load_yaml_app_config()
        assert result == {}


# ============================================================================
# _load_yaml_app_config: example-baseline deep merge
# ============================================================================
class TestAppConfigExampleMerge:
    """The real app.yaml is deep-merged OVER app.example.yaml.

    Regression: a present-but-partial real app.yaml was used verbatim, so
    every key it omitted fell through to a hardcoded default that could
    contradict the shipped template.
    """

    def _write_example(self, tmp_path: Path) -> Path:
        (tmp_path / "app.example.yaml").write_text(
            yaml.dump(
                {
                    "cli_mode": False,
                    "summarize": True,
                    "citation": {
                        "openalex_email": "",
                        "max_api_requests": 300,
                        "match_title_overlap": 0.5,
                    },
                    "paths": {"state_dir": ""},
                }
            ),
            encoding="utf-8",
        )
        return tmp_path / "app.yaml"

    def test_partial_real_inherits_example_values(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Omitted keys and omitted sub-keys keep the template's value."""
        real_path = self._write_example(tmp_path)
        real_path.write_text(
            yaml.dump({"cli_mode": True, "citation": {"max_api_requests": 50}}),
            encoding="utf-8",
        )

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", real_path)

        result = _load_yaml_app_config()
        # The real file wins where it speaks ...
        assert result["cli_mode"] is True
        assert result["citation"]["max_api_requests"] == 50
        # ... and inherits the rest, including whole omitted sections.
        assert result["summarize"] is True
        assert result["citation"]["match_title_overlap"] == 0.5
        assert result["paths"] == {"state_dir": ""}

    def test_corrupt_real_falls_back_to_example_wholesale(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """An unparseable real app.yaml yields the example unchanged."""
        real_path = self._write_example(tmp_path)
        real_path.write_text(":::not: valid: yaml: [[[", encoding="utf-8")

        import config.app as ac

        monkeypatch.setattr(ac, "_APP_CONFIG_PATH", real_path)

        result = _load_yaml_app_config()
        assert result["summarize"] is True
        assert result["citation"]["max_api_requests"] == 300


# ============================================================================
# Citation defaults
# ============================================================================
class TestOpenAlexEmailDefault:
    """A missing openalex_email must not become a fake mailto address."""

    def test_default_is_blank(self) -> None:
        """The template documents "leave blank to skip"; the default matches."""
        from config.app import _get_str

        assert _get_str({}, "openalex_email", "") == ""

    def test_no_placeholder_default_in_module(self) -> None:
        """The old ``your-email@example.com`` placeholder default is gone."""
        import config.app as ac

        assert ac.CITATION_OPENALEX_EMAIL != "your-email@example.com"


# ============================================================================
# _get_str
# ============================================================================
class TestGetStr:
    """Tests for _get_str()."""

    def test_normal_value(self) -> None:
        """Returns the value as a string when key is present."""
        data = {"key": "hello"}
        assert _get_str(data, "key", "default") == "hello"

    def test_none_value_returns_default(self) -> None:
        """Returns default when value is None."""
        data = {"key": None}
        assert _get_str(data, "key", "fallback") == "fallback"

    def test_missing_key_returns_default(self) -> None:
        """Returns default when key is absent."""
        data: dict[str, Any] = {}
        assert _get_str(data, "missing", "fallback") == "fallback"

    def test_integer_value_converted_to_string(self) -> None:
        """Non-string values are converted via str()."""
        data = {"key": 42}
        assert _get_str(data, "key", "default") == "42"

    def test_empty_string_value(self) -> None:
        """Empty string is returned as is (not replaced by default)."""
        data = {"key": ""}
        assert _get_str(data, "key", "default") == ""


# ============================================================================
# _get_int
# ============================================================================
class TestGetInt:
    """Tests for _get_int()."""

    def test_normal_int(self) -> None:
        """Returns the integer when key holds a valid int."""
        data = {"key": 10}
        assert _get_int(data, "key", 99) == 10

    def test_string_int(self) -> None:
        """Returns parsed int when key holds a numeric string."""
        data = {"key": "42"}
        assert _get_int(data, "key", 0) == 42

    def test_invalid_value_returns_default(self) -> None:
        """Returns default when value cannot be converted to int."""
        data = {"key": "not_a_number"}
        assert _get_int(data, "key", 77) == 77

    def test_none_value_returns_default(self) -> None:
        """Returns default when value is None (TypeError)."""
        data = {"key": None}
        assert _get_int(data, "key", 5) == 5

    def test_missing_key_returns_default(self) -> None:
        """Returns default when key is absent."""
        data: dict[str, Any] = {}
        assert _get_int(data, "missing", 123) == 123

    def test_float_truncated_to_int(self) -> None:
        """Float values are truncated to int."""
        data = {"key": 3.9}
        assert _get_int(data, "key", 0) == 3


# ============================================================================
# _get_bool
# ============================================================================
class TestGetBool:
    """Tests for _get_bool()."""

    def test_true_value(self) -> None:
        """Returns True for True value."""
        data = {"key": True}
        assert _get_bool(data, "key", False) is True

    def test_false_value(self) -> None:
        """Returns False for False value."""
        data = {"key": False}
        assert _get_bool(data, "key", True) is False

    def test_recognized_true_literals(self) -> None:
        """Quoted true-ish literals map to True, case- and space-insensitively."""
        for raw in ("true", "TRUE", " True ", "yes", "on", "1"):
            assert _get_bool({"key": raw}, "key", False) is True

    def test_recognized_false_literals(self) -> None:
        """Regression: a quoted "false"/"no"/"off"/"0" must NOT read as True.

        ``bool("false")`` is True, so the previous implementation silently
        inverted every quoted negative the user wrote in YAML.
        """
        for raw in ("false", "FALSE", " False ", "no", "off", "0"):
            assert _get_bool({"key": raw}, "key", True) is False

    def test_unrecognized_string_returns_default(self) -> None:
        """An unparseable string falls back to the default with a warning.

        Mirrors _get_int / _get_float, which also reject garbage rather than
        coercing it. The empty string is unrecognized too.
        """
        assert _get_bool({"key": "maybe"}, "key", True) is True
        assert _get_bool({"key": "maybe"}, "key", False) is False
        assert _get_bool({"key": ""}, "key", True) is True
        assert _get_bool({"key": ""}, "key", False) is False

    def test_truthy_int(self) -> None:
        """Non-zero int is truthy."""
        data = {"key": 1}
        assert _get_bool(data, "key", False) is True

    def test_falsy_zero(self) -> None:
        """Zero is falsy."""
        data = {"key": 0}
        assert _get_bool(data, "key", True) is False

    def test_none_value_returns_default(self) -> None:
        """An explicit null (e.g. a blank YAML key) falls back to the default.

        A key written with no value parses to None; treating it as False would
        silently disable features whose default is True. Mirrors _get_str /
        _get_int, which also fall back to the default on None.
        """
        data = {"key": None}
        assert _get_bool(data, "key", True) is True
        assert _get_bool(data, "key", False) is False

    def test_missing_key_returns_default_true(self) -> None:
        """Missing key returns the default value (True)."""
        data: dict[str, Any] = {}
        assert _get_bool(data, "missing", True) is True

    def test_missing_key_returns_default_false(self) -> None:
        """Missing key returns the default value (False)."""
        data: dict[str, Any] = {}
        assert _get_bool(data, "missing", False) is False
