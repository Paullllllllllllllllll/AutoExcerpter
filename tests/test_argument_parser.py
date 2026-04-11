"""Comprehensive tests for CLI argument parsing and override logic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cli.argument_parser import (
    PROVIDER_CHOICES,
    REASONING_EFFORT_CHOICES,
    VERBOSITY_CHOICES,
    _apply_app_config_overrides,
    _build_cli_model_overrides,
    _parse_cli_selection,
    _parse_execution_mode,
    _positive_int,
    _temperature_float,
)
from modules import app_config as config
from modules.types import ItemSpec


# ============================================================================
# Helpers
# ============================================================================


def _make_cli_args(**kwargs: Any) -> argparse.Namespace:
    """Build a Namespace mimicking parsed CLI-mode arguments.

    Defaults are set to ``None`` for every optional flag so that override
    functions see the same shape they get from ``argparse``.
    """
    defaults: dict[str, Any] = {
        # Positional / path args
        "input": "/tmp/in",
        "output": "/tmp/out",
        "input_path": None,
        "output_path": None,
        # Selection
        "all": False,
        "select": None,
        "context": None,
        # Resume
        "resume": None,
        "force": None,
        # App config overrides
        "summarize": None,
        "cleanup": None,
        # Model overrides
        "model": None,
        "transcription_model": None,
        "summary_model": None,
        "reasoning_effort": None,
        "transcription_reasoning_effort": None,
        "summary_reasoning_effort": None,
        "verbosity": None,
        "transcription_verbosity": None,
        "summary_verbosity": None,
        "max_output_tokens": None,
        "transcription_max_output_tokens": None,
        "summary_max_output_tokens": None,
        "temperature": None,
        "transcription_temperature": None,
        "summary_temperature": None,
        "provider": None,
        "transcription_provider": None,
        "summary_provider": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ============================================================================
# Validators
# ============================================================================


class TestPositiveInt:
    """Tests for _positive_int() argparse validator."""

    def test_valid_positive(self) -> None:
        assert _positive_int("1") == 1
        assert _positive_int("42") == 42
        assert _positive_int("100000") == 100_000

    def test_zero_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
            _positive_int("0")

    def test_negative_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be > 0"):
            _positive_int("-5")

    def test_non_integer_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be an integer"):
            _positive_int("abc")

    def test_float_string_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be an integer"):
            _positive_int("1.5")


class TestTemperatureFloat:
    """Tests for _temperature_float() argparse validator."""

    def test_valid_values(self) -> None:
        assert _temperature_float("0.0") == 0.0
        assert _temperature_float("0.5") == 0.5
        assert _temperature_float("1.0") == 1.0
        assert _temperature_float("1.5") == 1.5
        assert _temperature_float("2.0") == 2.0

    def test_zero_is_valid(self) -> None:
        result = _temperature_float("0")
        assert result == 0.0

    def test_boundary_lower(self) -> None:
        assert _temperature_float("0.0") == 0.0

    def test_boundary_upper(self) -> None:
        assert _temperature_float("2.0") == 2.0

    def test_below_range_rejected(self) -> None:
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be between 0.0 and 2.0"
        ):
            _temperature_float("-0.1")

    def test_above_range_rejected(self) -> None:
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be between 0.0 and 2.0"
        ):
            _temperature_float("2.1")

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="must be a number"):
            _temperature_float("abc")


# ============================================================================
# _build_cli_model_overrides
# ============================================================================


class TestBuildCliModelOverrides:
    """Tests for _build_cli_model_overrides() override dict construction."""

    @patch.object(config, "CLI_MODE", True)
    def test_no_flags_returns_empty(self) -> None:
        args = _make_cli_args()
        result = _build_cli_model_overrides(args)
        assert result == {}

    @patch.object(config, "CLI_MODE", False)
    def test_non_cli_mode_returns_empty(self) -> None:
        args = _make_cli_args(model="gpt-5")
        result = _build_cli_model_overrides(args)
        assert result == {}

    # --- Model name overrides ---

    @patch.object(config, "CLI_MODE", True)
    def test_shared_model_sets_both(self) -> None:
        args = _make_cli_args(model="gpt-5-mini")
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["name"] == "gpt-5-mini"
        assert result["summary_model"]["name"] == "gpt-5-mini"

    @patch.object(config, "CLI_MODE", True)
    def test_specific_model_overrides_shared(self) -> None:
        args = _make_cli_args(
            model="gpt-5-mini",
            transcription_model="gpt-5",
            summary_model="claude-4",
        )
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["name"] == "gpt-5"
        assert result["summary_model"]["name"] == "claude-4"

    @patch.object(config, "CLI_MODE", True)
    def test_only_transcription_model(self) -> None:
        args = _make_cli_args(transcription_model="gpt-5")
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["name"] == "gpt-5"
        assert "summary_model" not in result

    # --- Reasoning effort overrides ---

    @patch.object(config, "CLI_MODE", True)
    def test_shared_reasoning_sets_both(self) -> None:
        args = _make_cli_args(reasoning_effort="high")
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["reasoning"]["effort"] == "high"
        assert result["summary_model"]["reasoning"]["effort"] == "high"

    @patch.object(config, "CLI_MODE", True)
    def test_specific_reasoning_overrides_shared(self) -> None:
        args = _make_cli_args(
            reasoning_effort="low",
            transcription_reasoning_effort="high",
        )
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["reasoning"]["effort"] == "high"
        assert result["summary_model"]["reasoning"]["effort"] == "low"

    # --- Verbosity overrides ---

    @patch.object(config, "CLI_MODE", True)
    def test_shared_verbosity_sets_both(self) -> None:
        args = _make_cli_args(verbosity="high")
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["text"]["verbosity"] == "high"
        assert result["summary_model"]["text"]["verbosity"] == "high"

    @patch.object(config, "CLI_MODE", True)
    def test_specific_verbosity_overrides_shared(self) -> None:
        args = _make_cli_args(
            verbosity="medium",
            summary_verbosity="low",
        )
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["text"]["verbosity"] == "medium"
        assert result["summary_model"]["text"]["verbosity"] == "low"

    # --- Max output tokens overrides ---

    @patch.object(config, "CLI_MODE", True)
    def test_shared_max_tokens_sets_both(self) -> None:
        args = _make_cli_args(max_output_tokens=8000)
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["max_output_tokens"] == 8000
        assert result["summary_model"]["max_output_tokens"] == 8000

    @patch.object(config, "CLI_MODE", True)
    def test_specific_max_tokens_overrides_shared(self) -> None:
        args = _make_cli_args(
            max_output_tokens=8000,
            transcription_max_output_tokens=12000,
        )
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["max_output_tokens"] == 12000
        assert result["summary_model"]["max_output_tokens"] == 8000

    # --- Temperature overrides ---

    @patch.object(config, "CLI_MODE", True)
    def test_shared_temperature_sets_both(self) -> None:
        args = _make_cli_args(temperature=0.7)
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["temperature"] == 0.7
        assert result["summary_model"]["temperature"] == 0.7

    @patch.object(config, "CLI_MODE", True)
    def test_specific_temperature_overrides_shared(self) -> None:
        args = _make_cli_args(
            temperature=1.0,
            transcription_temperature=0.5,
            summary_temperature=1.5,
        )
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["temperature"] == 0.5
        assert result["summary_model"]["temperature"] == 1.5

    @patch.object(config, "CLI_MODE", True)
    def test_temperature_zero_does_not_fall_through(self) -> None:
        """Temperature 0.0 is a valid value and must not fall through to shared."""
        args = _make_cli_args(
            temperature=1.0,
            transcription_temperature=0.0,
        )
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["temperature"] == 0.0
        assert result["summary_model"]["temperature"] == 1.0

    @patch.object(config, "CLI_MODE", True)
    def test_only_transcription_temperature(self) -> None:
        args = _make_cli_args(transcription_temperature=0.3)
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["temperature"] == 0.3
        assert "summary_model" not in result or "temperature" not in result.get(
            "summary_model", {}
        )

    # --- Provider overrides ---

    @patch.object(config, "CLI_MODE", True)
    def test_shared_provider_sets_both(self) -> None:
        args = _make_cli_args(provider="anthropic")
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["provider"] == "anthropic"
        assert result["summary_model"]["provider"] == "anthropic"

    @patch.object(config, "CLI_MODE", True)
    def test_specific_provider_overrides_shared(self) -> None:
        args = _make_cli_args(
            provider="openai",
            transcription_provider="google",
            summary_provider="anthropic",
        )
        result = _build_cli_model_overrides(args)
        assert result["transcription_model"]["provider"] == "google"
        assert result["summary_model"]["provider"] == "anthropic"

    @patch.object(config, "CLI_MODE", True)
    def test_only_summary_provider(self) -> None:
        args = _make_cli_args(summary_provider="openrouter")
        result = _build_cli_model_overrides(args)
        assert "transcription_model" not in result or "provider" not in result.get(
            "transcription_model", {}
        )
        assert result["summary_model"]["provider"] == "openrouter"

    # --- Combined overrides ---

    @patch.object(config, "CLI_MODE", True)
    def test_combined_overrides(self) -> None:
        args = _make_cli_args(
            model="gpt-5",
            reasoning_effort="high",
            verbosity="low",
            max_output_tokens=16000,
            temperature=0.5,
            provider="openai",
        )
        result = _build_cli_model_overrides(args)
        for model_key in ("transcription_model", "summary_model"):
            assert result[model_key]["name"] == "gpt-5"
            assert result[model_key]["reasoning"]["effort"] == "high"
            assert result[model_key]["text"]["verbosity"] == "low"
            assert result[model_key]["max_output_tokens"] == 16000
            assert result[model_key]["temperature"] == 0.5
            assert result[model_key]["provider"] == "openai"


# ============================================================================
# _apply_app_config_overrides
# ============================================================================


class TestApplyAppConfigOverrides:
    """Tests for _apply_app_config_overrides() config mutation."""

    @pytest.fixture(autouse=True)
    def _save_config(self) -> Any:
        """Save and restore original config values around each test."""
        orig_summarize = config.SUMMARIZE
        orig_cleanup = config.DELETE_TEMP_WORKING_DIR
        yield
        config.SUMMARIZE = orig_summarize
        config.DELETE_TEMP_WORKING_DIR = orig_cleanup

    @patch.object(config, "CLI_MODE", True)
    def test_no_summarize_disables_summarization(self) -> None:
        config.SUMMARIZE = True
        args = _make_cli_args(summarize=False)
        _apply_app_config_overrides(args)
        assert config.SUMMARIZE is False

    @patch.object(config, "CLI_MODE", True)
    def test_summarize_enables_summarization(self) -> None:
        config.SUMMARIZE = False
        args = _make_cli_args(summarize=True)
        _apply_app_config_overrides(args)
        assert config.SUMMARIZE is True

    @patch.object(config, "CLI_MODE", True)
    def test_no_cleanup_keeps_working_dir(self) -> None:
        config.DELETE_TEMP_WORKING_DIR = True
        args = _make_cli_args(cleanup=False)
        _apply_app_config_overrides(args)
        assert config.DELETE_TEMP_WORKING_DIR is False

    @patch.object(config, "CLI_MODE", True)
    def test_cleanup_enables_cleanup(self) -> None:
        config.DELETE_TEMP_WORKING_DIR = False
        args = _make_cli_args(cleanup=True)
        _apply_app_config_overrides(args)
        assert config.DELETE_TEMP_WORKING_DIR is True

    @patch.object(config, "CLI_MODE", True)
    def test_no_flags_leaves_config_unchanged(self) -> None:
        config.SUMMARIZE = True
        config.DELETE_TEMP_WORKING_DIR = True
        args = _make_cli_args()
        _apply_app_config_overrides(args)
        assert config.SUMMARIZE is True
        assert config.DELETE_TEMP_WORKING_DIR is True

    @patch.object(config, "CLI_MODE", False)
    def test_non_cli_mode_no_ops(self) -> None:
        config.SUMMARIZE = True
        args = _make_cli_args(summarize=False)
        _apply_app_config_overrides(args)
        assert config.SUMMARIZE is True


# ============================================================================
# _parse_execution_mode
# ============================================================================


class TestParseExecutionMode:
    """Tests for _parse_execution_mode() resume and path parsing."""

    @patch.object(config, "CLI_MODE", True)
    def test_force_sets_overwrite(self) -> None:
        args = _make_cli_args(force=True)
        _, _, _, _, _, resume_mode = _parse_execution_mode(args)
        assert resume_mode == "overwrite"

    @patch.object(config, "CLI_MODE", True)
    def test_default_resume_mode_is_skip(self) -> None:
        args = _make_cli_args()
        _, _, _, _, _, resume_mode = _parse_execution_mode(args)
        assert resume_mode == "skip"

    @patch.object(config, "CLI_MODE", True)
    def test_resume_flag_keeps_skip(self) -> None:
        args = _make_cli_args(resume=True)
        _, _, _, _, _, resume_mode = _parse_execution_mode(args)
        assert resume_mode == "skip"

    @patch.object(config, "CLI_MODE", True)
    def test_cli_paths_resolved(self) -> None:
        args = _make_cli_args(input="C:/abs/path/in", output="C:/abs/path/out")
        input_path, output_path, _, _, _, _ = _parse_execution_mode(args)
        assert input_path == Path("C:/abs/path/in")
        assert output_path == Path("C:/abs/path/out")

    @patch.object(config, "CLI_MODE", True)
    def test_named_paths_override_positional(self) -> None:
        args = _make_cli_args(
            input="C:/pos/in",
            output="C:/pos/out",
            input_path="C:/named/in",
            output_path="C:/named/out",
        )
        input_path, output_path, _, _, _, _ = _parse_execution_mode(args)
        assert input_path == Path("C:/named/in")
        assert output_path == Path("C:/named/out")

    @patch.object(config, "CLI_MODE", True)
    def test_process_all_flag(self) -> None:
        args = _make_cli_args(all=True)
        _, _, process_all, _, _, _ = _parse_execution_mode(args)
        assert process_all is True

    @patch.object(config, "CLI_MODE", True)
    def test_select_pattern_passed_through(self) -> None:
        args = _make_cli_args(select="1-5")
        _, _, _, select_pattern, _, _ = _parse_execution_mode(args)
        assert select_pattern == "1-5"

    @patch.object(config, "CLI_MODE", True)
    def test_context_passed_through(self) -> None:
        args = _make_cli_args(context="Food History")
        _, _, _, _, context, _ = _parse_execution_mode(args)
        assert context == "Food History"


# ============================================================================
# _parse_cli_selection
# ============================================================================


def _make_item(name: str) -> ItemSpec:
    """Create a minimal ItemSpec for selection testing."""
    return ItemSpec(kind="pdf", path=Path(f"/test/{name}"))


class TestParseCliSelection:
    """Tests for _parse_cli_selection() item filtering."""

    def setup_method(self) -> None:
        self.items = [
            _make_item("Mennell_Food_History.pdf"),
            _make_item("Laudan_Cuisine.pdf"),
            _make_item("Mintz_Sweetness.pdf"),
            _make_item("Goody_Cooking.pdf"),
            _make_item("Pilcher_Food_History.pdf"),
        ]

    def test_single_number(self) -> None:
        result = _parse_cli_selection(self.items, "2")
        assert len(result) == 1
        assert result[0].output_stem == "Laudan_Cuisine"

    def test_comma_list(self) -> None:
        result = _parse_cli_selection(self.items, "1,3,5")
        assert len(result) == 3
        stems = [r.output_stem for r in result]
        assert stems == [
            "Mennell_Food_History",
            "Mintz_Sweetness",
            "Pilcher_Food_History",
        ]

    def test_range(self) -> None:
        result = _parse_cli_selection(self.items, "2-4")
        assert len(result) == 3
        stems = [r.output_stem for r in result]
        assert stems == [
            "Laudan_Cuisine",
            "Mintz_Sweetness",
            "Goody_Cooking",
        ]

    def test_filename_pattern(self) -> None:
        result = _parse_cli_selection(self.items, "Mennell")
        assert len(result) == 1
        assert result[0].output_stem == "Mennell_Food_History"

    def test_filename_pattern_case_insensitive(self) -> None:
        result = _parse_cli_selection(self.items, "mennell")
        assert len(result) == 1

    def test_filename_pattern_multiple_matches(self) -> None:
        result = _parse_cli_selection(self.items, "Food_History")
        assert len(result) == 2
        stems = [r.output_stem for r in result]
        assert "Mennell_Food_History" in stems
        assert "Pilcher_Food_History" in stems

    def test_out_of_range_number_ignored(self) -> None:
        result = _parse_cli_selection(self.items, "99")
        assert len(result) == 0

    def test_mixed_valid_invalid_numbers(self) -> None:
        result = _parse_cli_selection(self.items, "1,99,3")
        assert len(result) == 2

    def test_no_match_returns_empty(self) -> None:
        result = _parse_cli_selection(self.items, "Nonexistent")
        assert len(result) == 0

    def test_semicolons_treated_as_commas(self) -> None:
        result = _parse_cli_selection(self.items, "1;3")
        assert len(result) == 2


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    """Verify that choice constants are consistent and non-empty."""

    def test_reasoning_effort_choices(self) -> None:
        assert len(REASONING_EFFORT_CHOICES) >= 3
        assert "medium" in REASONING_EFFORT_CHOICES

    def test_verbosity_choices(self) -> None:
        assert len(VERBOSITY_CHOICES) >= 3
        assert "medium" in VERBOSITY_CHOICES

    def test_provider_choices(self) -> None:
        assert "openai" in PROVIDER_CHOICES
        assert "anthropic" in PROVIDER_CHOICES
        assert "google" in PROVIDER_CHOICES
        assert "openrouter" in PROVIDER_CHOICES
        assert "custom" in PROVIDER_CHOICES
