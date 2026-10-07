"""Tests for settings.py: the example file, typed values and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoexcerpter import settings as settings_module
from autoexcerpter.common.options import Source
from autoexcerpter.common.retry import RetryPolicy
from autoexcerpter.settings import (
    EXAMPLE_SETTINGS_PATH,
    Endpoint,
    RetryRule,
    Settings,
    SettingsError,
    SpecError,
    Timeouts,
    load,
    resolve,
)
from autoexcerpter.spec import OPTIONS


@pytest.fixture
def settings_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the default settings file and the example into ``tmp_path``."""
    monkeypatch.setattr(
        settings_module, "DEFAULT_SETTINGS_PATH", tmp_path / "settings.yaml"
    )
    monkeypatch.setattr(
        settings_module, "EXAMPLE_SETTINGS_PATH", tmp_path / "settings.example.yaml"
    )
    return tmp_path


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_tracked_example_loads_and_matches_code_values() -> None:
    loaded = load(EXAMPLE_SETTINGS_PATH)
    assert loaded.source == EXAMPLE_SETTINGS_PATH
    assert loaded == Settings(source=EXAMPLE_SETTINGS_PATH)


def test_example_is_used_when_the_real_file_is_missing(settings_paths: Path) -> None:
    _write(settings_paths / "settings.example.yaml", "openalex:\n  max_requests: 5\n")
    assert load().openalex.max_requests == 5


def test_real_file_wins_over_the_example(settings_paths: Path) -> None:
    _write(settings_paths / "settings.example.yaml", "openalex:\n  max_requests: 5\n")
    _write(settings_paths / "settings.yaml", "openalex:\n  max_requests: 7\n")
    assert load().openalex.max_requests == 7


def test_missing_file_gives_a_clear_error(settings_paths: Path) -> None:
    with pytest.raises(SettingsError, match="settings.yaml"):
        load()


def test_explicit_settings_path_wins(settings_paths: Path) -> None:
    _write(settings_paths / "settings.yaml", "openalex:\n  max_requests: 7\n")
    other = _write(settings_paths / "other.yaml", "openalex:\n  max_requests: 9\n")
    assert load(other).openalex.max_requests == 9


def test_partial_file_keeps_code_values(settings_paths: Path) -> None:
    _write(
        settings_paths / "settings.yaml",
        "timeouts:\n  request: 300\nretry:\n  schema_retries:\n"
        "    summary:\n      validation_failure:\n        max_attempts: 5\n",
    )
    value = load()
    assert value.timeouts == Timeouts(300.0, 10.0, 30.0, 30.0)
    assert value.retry.max_attempts == 8
    summary_rule = value.retry.schema_retries["summary"]["validation_failure"]
    assert summary_rule == RetryRule(True, 5, 0.5, 1.5)
    assert value.retry.schema_retries["transcription"]["validation_failure"] == (
        RetryRule(True, 3, 0.5, 1.5)
    )


def test_endpoints_and_key_variables(settings_paths: Path) -> None:
    _write(
        settings_paths / "settings.yaml",
        "api_keys:\n  openai: MY_OPENAI_KEY\n"
        "endpoints:\n  local:\n    base_url: http://127.0.0.1:8000/v1/\n"
        "    api_key_env: LOCAL_KEY\n    supports_schema: true\n",
    )
    value = load()
    assert value.endpoint("local") == Endpoint(
        "local", "http://127.0.0.1:8000/v1/", "LOCAL_KEY", True, True
    )
    assert value.api_key_env("openai") == "MY_OPENAI_KEY"
    assert value.api_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert value.api_key_env("custom", "local") == "LOCAL_KEY"
    with pytest.raises(SettingsError, match="unknown endpoint"):
        value.endpoint("remote")
    with pytest.raises(SettingsError):
        value.api_key_env("custom")


@pytest.mark.parametrize(
    "text",
    [
        "unknown: 1\n",
        "retry:\n  call_timeout: auto\n",
        "timeouts:\n  request: fast\n",
        "rate_limits:\n  - [10, 0]\n",
        "endpoints:\n  local:\n    base_url: http://x/\n",
        "openalex:\n  max_requests: -1\n",
        "defaults:\n  modle: gpt-5.6-terra\n",
        "defaults:\n  dry_run: true\n",
        "defaults:\n  force: true\n",
        "defaults:\n  retranscribe: true\n",
        "defaults:\n  beside_input: true\n",
        "defaults:\n  concurrency: 0\n",
    ],
)
def test_invalid_settings_are_errors(settings_paths: Path, text: str) -> None:
    _write(settings_paths / "settings.yaml", text)
    with pytest.raises(SettingsError, match="settings.yaml"):
        load()


def test_precedence_flag_then_settings_default_then_code_default(
    settings_paths: Path,
) -> None:
    _write(
        settings_paths / "settings.yaml",
        "defaults:\n  model: gpt-5.6-terra\n  concurrency: 40\n"
        "  summary_verbosity: medium\n  formats: [summary-md]\n",
    )
    value = load()
    flags = OPTIONS.parse(["--input", "x.pdf", "--concurrency", "8"])
    resolution = resolve(value, flags)
    spec, sources = resolution.spec, resolution.sources
    assert spec.run.concurrency == 8
    assert sources["run.concurrency"] is Source.FLAG
    assert spec.transcription_model.model == "gpt-5.6-terra"
    assert sources["transcription_model.model"] is Source.SETTINGS
    assert spec.summary_model.verbosity == "medium"
    assert sources["summary_model.verbosity"] is Source.SETTINGS
    assert spec.output.formats == ("summary-md",)
    assert spec.transcription_model.verbosity == "medium"
    assert sources["transcription_model.verbosity"] is Source.DEFAULT
    assert spec.transcription_model.reasoning_effort == "high"
    assert sources["transcription_model.reasoning_effort"] is Source.DEFAULT


def test_prefixed_flag_beats_unprefixed_settings_default(settings_paths: Path) -> None:
    _write(settings_paths / "settings.yaml", "defaults:\n  model: gpt-5.6-terra\n")
    spec = resolve(load(), {"input": Path("x"), "summary_model": "m2"}).spec
    assert spec.transcription_model.model == "gpt-5.6-terra"
    assert spec.summary_model.model == "m2"


def test_resolve_checks_endpoint_names() -> None:
    value = Settings(
        endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
    )
    spec = resolve(value, {"input": Path("x"), "endpoint": "local"}).spec
    assert spec.transcription_model.provider == "custom"
    with pytest.raises(SettingsError, match="remote"):
        resolve(value, {"input": Path("x"), "endpoint": "remote"})
    with pytest.raises(SpecError):
        resolve(value, {"input": Path("x"), "provider": "custom"})


def test_retry_policy_and_state_directory(tmp_path: Path) -> None:
    policy = Settings().retry.policy()
    assert isinstance(policy, RetryPolicy)
    assert (policy.max_attempts, policy.backoff_base, policy.backoff_cap) == (
        8,
        0.5,
        120.0,
    )
    assert Settings(state_dir=tmp_path).state_directory() == tmp_path
    assert Settings().state_directory().name == ".autoexcerpter"
