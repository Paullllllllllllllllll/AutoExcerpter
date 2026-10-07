"""Provider inference from the model name, through spec.parse and settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoexcerpter import __main__ as entry
from autoexcerpter.common.options import Source
from autoexcerpter.pipeline.job import job_from_spec
from autoexcerpter.providers import implied_provider, infer_provider
from autoexcerpter.settings import Endpoint, Settings, SpecError, resolve
from autoexcerpter.spec import OPTIONS, Resolution, parse, resolve_spec, to_argv


def _resolve(argv: list[str], settings: Settings | None = None) -> Resolution:
    return resolve(settings or Settings(), OPTIONS.parse(["--input", "x", *argv]))


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gpt-5.6-luna", "openai"),
        ("o3-mini", "openai"),
        ("claude-sonnet-4-5", "anthropic"),
        ("gemini-2.5-flash", "google"),
        ("anthropic/claude-sonnet-4.5", "openrouter"),
        ("anthropic:claude-sonnet-4-5", "anthropic"),
        ("ft:gpt-4o-mini:org:id", "openai"),
        ("my-local-model", "openai"),
    ],
)
def test_infer_provider(model: str, provider: str) -> None:
    assert infer_provider(model) == provider


def test_only_known_names_imply_a_provider() -> None:
    assert implied_provider("my-local-model") is None
    assert implied_provider("Claude-3-Opus") == "anthropic"


def test_default_arguments_resolve_to_openai_from_the_code_default() -> None:
    resolution = _resolve([])
    for phase in ("transcription", "summary"):
        assert resolution.spec.model(phase).provider == "openai"
        assert resolution.sources[f"{phase}_model.provider"] is Source.DEFAULT


def test_claude_model_selects_anthropic_and_its_key_variable() -> None:
    resolution = _resolve(["--model", "claude-sonnet-4-5"])
    model = resolution.spec.transcription_model
    assert model.provider == "anthropic"
    assert resolution.sources["transcription_model.provider"] is Source.FLAG
    job = job_from_spec(resolution.spec, Settings())
    assert job.transcription.provider == "anthropic"
    assert job.transcription.api_key_env == "ANTHROPIC_API_KEY"


@pytest.mark.parametrize(
    ("model", "provider", "name"),
    [
        ("gemini-2.5-flash", "google", "gemini-2.5-flash"),
        ("anthropic/claude-sonnet-4.5", "openrouter", "anthropic/claude-sonnet-4.5"),
        ("anthropic:claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
        ("google:gemini-2.5-flash", "google", "gemini-2.5-flash"),
    ],
)
def test_model_flag_selects_the_provider(model: str, provider: str, name: str) -> None:
    spec = parse(["--input", "x", "--model", model])
    for phase in ("transcription", "summary"):
        assert spec.model(phase).provider == provider
        assert spec.model(phase).model == name


def test_settings_provider_yields_to_a_model_flag() -> None:
    settings = Settings(defaults={"provider": "openai"})
    resolution = _resolve(["--model", "claude-sonnet-4-5"], settings)
    assert resolution.spec.summary_model.provider == "anthropic"
    assert resolution.sources["summary_model.provider"] is Source.FLAG


def test_settings_provider_stays_for_a_name_that_implies_none() -> None:
    settings = Settings(defaults={"provider": "anthropic"})
    spec = _resolve(["--model", "my-local-model"], settings).spec
    assert spec.summary_model.provider == "anthropic"


def test_provider_flag_wins_over_a_settings_model() -> None:
    settings = Settings(defaults={"model": "claude-sonnet-4-5"})
    spec = _resolve(["--provider", "openrouter"], settings).spec
    assert spec.transcription_model.provider == "openrouter"


def test_provider_flag_wins_over_an_implying_model_flag() -> None:
    spec = parse(["--input", "x", "--provider", "openai", "--model", "claude-x"])
    assert spec.transcription_model.provider == "openai"


def test_settings_model_prefix_selects_the_provider() -> None:
    settings = Settings(defaults={"model": "anthropic:claude-sonnet-4-5"})
    resolution = _resolve([], settings)
    assert resolution.spec.transcription_model.provider == "anthropic"
    assert resolution.spec.transcription_model.model == "claude-sonnet-4-5"
    assert resolution.sources["transcription_model.provider"] is Source.SETTINGS


def test_named_endpoint_keeps_provider_custom() -> None:
    settings = Settings(
        endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
    )
    spec = _resolve(["--endpoint", "local", "--model", "org/model"], settings).spec
    for phase in ("transcription", "summary"):
        assert spec.model(phase).provider == "custom"
        assert spec.model(phase).model == "org/model"


def test_prefix_contradicting_a_named_endpoint_is_an_error() -> None:
    with pytest.raises(SpecError, match="names provider anthropic"):
        parse(["--input", "x", "--endpoint", "local", "--model", "anthropic:c"])


def test_prefix_contradicting_the_provider_flag_is_an_error() -> None:
    with pytest.raises(SpecError, match="names provider anthropic"):
        parse(
            [
                "--input",
                "x",
                "--provider",
                "openai",
                "--model",
                "anthropic:claude-sonnet-4-5",
            ]
        )


def test_prefix_contradiction_exits_2_through_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("openalex:\n  max_requests: 5\n", encoding="utf-8")
    code = entry.main(
        [
            "run",
            "--input",
            str(tmp_path),
            "--settings",
            str(settings),
            "--provider",
            "openai",
            "--model",
            "anthropic:claude-sonnet-4-5",
        ]
    )
    assert code == 2
    assert "names provider anthropic" in capsys.readouterr().err


def test_code_defaults_round_trip_to_the_input_alone() -> None:
    path = Path("in.pdf")
    assert to_argv(resolve_spec({"input": path}).spec) == ["--input", str(path)]


def test_to_argv_spells_out_a_provider_the_name_does_not_imply() -> None:
    spec = parse(["--input", "x", "--provider", "openai", "--model", "claude-x"])
    argv = to_argv(spec)
    assert "--provider" in argv
    assert parse(argv) == spec
    implied = parse(["--input", "x", "--model", "anthropic:claude-sonnet-4-5"])
    assert to_argv(implied) == ["--input", "x", "--model", "claude-sonnet-4-5"]


@pytest.mark.parametrize(
    ("defaults", "extra"),
    [
        ({"provider": "openai"}, []),
        ({"provider": "anthropic"}, ["--provider", "openai"]),
        ({"model": "anthropic:claude-sonnet-4-5"}, ["--model", "gpt-5.6-luna"]),
        ({"summary_endpoint": "local"}, ["--provider", "openai"]),
    ],
)
def test_to_argv_spells_out_a_provider_only_when_a_default_changes_it(
    defaults: dict[str, str], extra: list[str]
) -> None:
    checked = OPTIONS.check_defaults(defaults)
    spec = parse(["--input", "x"])
    argv = to_argv(spec, checked)
    assert argv == ["--input", "x", *extra]
    assert resolve(Settings(defaults=checked), OPTIONS.parse(argv)).spec == spec
