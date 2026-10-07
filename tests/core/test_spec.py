"""Tests for spec.py: option table, code defaults, round trip, phase flags."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.llm.capabilities import detect_capabilities
from autoexcerpter.providers import infer_provider
from autoexcerpter.spec import (
    CONTEXT_NONE,
    DEFAULT_MODEL,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    MODEL_SUGGESTIONS,
    OPTIONS,
    OUTPUT_FORMATS,
    PHASES,
    PROVIDERS,
    REASONING_EFFORTS,
    RESPONSE_FORMATS,
    SERVICE_TIERS,
    TRANSCRIPTION_FORMATS,
    VERBOSITIES,
    CitationSpec,
    Dpi,
    ImageSpec,
    InputSpec,
    ModelSpec,
    OutputSpec,
    RunControl,
    RunSpec,
    SpecError,
    SummarySpec,
    effective_spec,
    model_suggestions,
    parse,
    recommended_images,
    resolve_spec,
    to_argv,
)

_DPIS: list[Dpi | None] = [None, "native", 150, 300]


def _code_defaults(path: str) -> RunSpec:
    return resolve_spec({"input": Path(path)}).spec


def _model(rng: random.Random) -> ModelSpec:
    provider = rng.choice(PROVIDERS)
    return ModelSpec(
        provider=provider,
        model=rng.choice(
            ["gpt-5.6-luna", "gpt-5.6-terra", "claude-x", "gemini-y", "v/m", "m y"]
        ),
        endpoint=rng.choice(["local", "remote"]) if provider == "custom" else None,
        reasoning_effort=rng.choice(REASONING_EFFORTS),
        verbosity=rng.choice(VERBOSITIES),
        max_output_tokens=rng.choice([None, 1, 4096, 128_000]),
        temperature=rng.choice([0.0, 0.7, 1.0, 2.0]),
        top_p=rng.choice([None, 0.0, 0.5, 1.0]),
        service_tier=rng.choice(SERVICE_TIERS),
        response_format=rng.choice([None, *RESPONSE_FORMATS]),
    )


def _spec(rng: random.Random) -> RunSpec:
    select = rng.choice([None, "1,3", "2-4", "Mennell"])
    transcription = _model(rng)
    summary = transcription if rng.random() < 0.3 else _model(rng)
    return RunSpec(
        input=InputSpec(
            path=Path(rng.choice(["in.pdf", "scans", "a b/c.pdf"])),
            select=select,
            all=select is None and rng.random() < 0.5,
        ),
        output=OutputSpec(
            directory=rng.choice([None, Path("out"), Path("x y")]),
            transcription_format=rng.choice(["md", "txt"]),
            formats=tuple(f for f in OUTPUT_FORMATS if rng.random() < 0.6),
            keep_working_files=rng.random() < 0.5,
        ),
        summary=SummarySpec(
            enabled=rng.random() < 0.5,
            context=rng.choice([None, CONTEXT_NONE, "Food History, Wages", "ctx.txt"]),
            fallback_context=rng.choice([None, "Prices, Wages", "fallback.txt"]),
        ),
        citations=CitationSpec(openalex=rng.random() < 0.5),
        transcription_model=transcription,
        summary_model=summary,
        images=ImageSpec(
            dpi=rng.choice(_DPIS),
            detail=rng.choice([None, *IMAGE_DETAILS]),
            format=rng.choice([None, *IMAGE_FORMATS]),
            jpeg_quality=rng.choice([None, 1, 85, 100]),
            grayscale=rng.choice([None, True, False]),
        ),
        run=RunControl(
            force=rng.random() < 0.5,
            retranscribe=rng.random() < 0.5,
            concurrency=rng.choice([1, 16, 80]),
            dry_run=rng.random() < 0.5,
            json=rng.random() < 0.5,
        ),
    )


_MODEL_DEFAULT_VALUES: dict[str, list[Any]] = {
    "provider": list(PROVIDERS),
    "model": [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "claude-x",
        "anthropic:claude-x",
        "v/m",
        "my-model",
    ],
    "endpoint": ["local", "remote"],
    "reasoning_effort": list(REASONING_EFFORTS),
    "verbosity": list(VERBOSITIES),
    "max_output_tokens": [1000, 4096],
    "temperature": [0.3, 1.0],
    "top_p": [0.5, 1.0],
    "service_tier": list(SERVICE_TIERS),
    "response_format": list(RESPONSE_FORMATS),
}

_DEFAULT_VALUES: dict[str, list[Any]] = {
    "select": ["1", "2-3", "Laudan"],
    "all": [True, False],
    "output": ["out", "x y"],
    "transcription_format": list(TRANSCRIPTION_FORMATS),
    "formats": [["summary-md"], ["sqlite", "summary-docx"], list(OUTPUT_FORMATS)],
    "keep_working_files": [True, False],
    "summarize": [True, False],
    "context": ["auto", "none", "Default topics", "ctx.txt"],
    "fallback_context": ["none", "Fallback topics", "fallback.txt"],
    "openalex": [True, False],
    "dpi": ["native", 200, 300],
    "image_detail": list(IMAGE_DETAILS),
    "image_format": list(IMAGE_FORMATS),
    "jpeg_quality": [80, 95],
    "grayscale": [True, False],
    "concurrency": [4, 16],
    **_MODEL_DEFAULT_VALUES,
    **{
        f"{phase}_{name}": values
        for phase in PHASES
        for name, values in _MODEL_DEFAULT_VALUES.items()
    },
}

_IMAGE_FIELDS = {
    "dpi": "dpi",
    "image_detail": "detail",
    "image_format": "format",
    "jpeg_quality": "jpeg_quality",
    "grayscale": "grayscale",
}
_MODEL_FIELDS_WITHOUT_UNSET = {"max_output_tokens", "top_p", "response_format"}


def _expressible(spec: RunSpec, key: str, value: Any) -> bool:
    """Whether a settings default *key* leaves *spec* expressible as flags.

    Conservative: a default that touches an unset value without a spelling
    is dropped even where a more specific default would shadow it.
    """
    if key in _IMAGE_FIELDS:
        return getattr(spec.images, _IMAGE_FIELDS[key]) is not None
    if key == "select":
        return spec.input.select is not None or spec.input.all
    if key == "all":
        return not value or spec.input.all or spec.input.select is not None
    phase, _, name = key.partition("_")
    phases = [phase] if phase in PHASES else list(PHASES)
    name = name if phase in PHASES else key
    if name in _MODEL_FIELDS_WITHOUT_UNSET:
        return all(getattr(spec.model(p), name) is not None for p in phases)
    return True


def _defaults(rng: random.Random, spec: RunSpec) -> dict[str, Any]:
    """Return a random checked settings defaults block that *spec* can undo."""
    keys = rng.sample(sorted(_DEFAULT_VALUES), rng.randint(0, 10))
    if "select" in keys and "all" in keys:
        keys.remove("all")
    raw = {key: rng.choice(_DEFAULT_VALUES[key]) for key in keys}
    kept = {key: value for key, value in raw.items() if _expressible(spec, key, value)}
    return OPTIONS.check_defaults(kept)


def _resolved(argv: list[str], defaults: dict[str, Any]) -> RunSpec:
    return resolve_spec(OPTIONS.parse(argv), defaults).spec


@pytest.mark.parametrize("seed", range(300))
def test_parse_of_to_argv_returns_the_spec(seed: int) -> None:
    spec = _spec(random.Random(seed))
    assert parse(to_argv(spec)) == spec


@pytest.mark.parametrize("seed", range(400))
def test_to_argv_reproduces_the_spec_with_and_without_settings_defaults(
    seed: int,
) -> None:
    rng = random.Random(seed)
    spec = _spec(rng)
    defaults = _defaults(rng, spec)

    argv = to_argv(spec, defaults)

    assert _resolved(argv, defaults) == spec
    assert _resolved(argv, {}) == spec


def test_to_argv_spells_out_values_the_settings_defaults_change() -> None:
    spec = _code_defaults("in.pdf")

    def argv(**defaults: Any) -> list[str]:
        return to_argv(spec, OPTIONS.check_defaults(defaults))[2:]

    assert argv(output="out") == ["--beside-input"]
    assert argv(context="Topics") == ["--context", "auto"]
    assert argv(fallback_context="Topics") == ["--fallback-context", "none"]
    assert argv(model="gpt-5.6-terra") == ["--model", DEFAULT_MODEL]
    assert argv(endpoint="local") == ["--provider", "openai"]
    assert argv(provider="openai", concurrency=16, context="auto") == []
    assert argv(summarize=False) == ["--summarize"]


@pytest.mark.parametrize(
    ("defaults", "option"),
    [
        ({"top_p": 0.5}, "top_p"),
        ({"summary_max_output_tokens": 100}, "summary_max_output_tokens"),
        ({"response_format": "json"}, "response_format"),
        ({"dpi": 300}, "dpi"),
        ({"grayscale": False}, "grayscale"),
        ({"select": "1"}, "select"),
        ({"all": True}, "all"),
    ],
)
def test_to_argv_rejects_a_default_without_an_unset_spelling(
    defaults: dict[str, Any], option: str
) -> None:
    with pytest.raises(SpecError, match=option):
        to_argv(_code_defaults("in.pdf"), OPTIONS.check_defaults(defaults))


@pytest.mark.parametrize(
    ("provider", "name"),
    [
        (provider, name)
        for provider, names in MODEL_SUGGESTIONS.items()
        for name in names
    ],
)
def test_every_suggestion_reads_images_and_implies_its_provider(
    provider: str, name: str
) -> None:
    assert detect_capabilities(name).supports_vision
    assert infer_provider(name) == provider


def test_model_suggestions_per_provider() -> None:
    assert set(MODEL_SUGGESTIONS) == {"openai", "anthropic", "google", "openrouter"}
    assert model_suggestions("openai")[0] == DEFAULT_MODEL
    assert model_suggestions("custom") == ()
    assert model_suggestions()[0] == DEFAULT_MODEL
    assert set(model_suggestions()) == {
        name for names in MODEL_SUGGESTIONS.values() for name in names
    }


def test_model_options_suggest_names_without_enforcing_them() -> None:
    for name in ("model", "transcription_model", "summary_model"):
        option = OPTIONS[name]
        assert option.choice_provider is not None
        assert tuple(option.choice_provider()) == model_suggestions()
        assert option.choices is None
    assert parse(["--input", "x", "--model", "my-model"]).summary_model.model == (
        "my-model"
    )


def test_code_defaults_need_no_arguments_beyond_the_input() -> None:
    assert to_argv(_code_defaults("in.pdf")) == ["--input", "in.pdf"]


def test_to_argv_collapses_equal_phase_values() -> None:
    spec = _code_defaults("in.pdf")
    both_low = replace(
        spec, transcription_model=replace(spec.transcription_model, verbosity="low")
    )
    assert to_argv(both_low) == ["--input", "in.pdf", "--verbosity", "low"]
    split = replace(spec, summary_model=replace(spec.summary_model, model="m2"))
    assert to_argv(split) == ["--input", "in.pdf", "--summary-model", "m2"]


def test_custom_provider_without_endpoint_is_an_error() -> None:
    with pytest.raises(SpecError, match="endpoint"):
        resolve_spec({"input": Path("x"), "summary_provider": "custom"})


def test_missing_input_is_an_error() -> None:
    with pytest.raises(SpecError, match="input"):
        resolve_spec({})


def test_sources_cover_every_field() -> None:
    resolution = resolve_spec(
        {"input": Path("x"), "concurrency": 4}, {"summary_verbosity": "high"}
    )
    payload = effective_spec(resolution.spec, resolution.sources)
    assert payload["run.concurrency"] == {"value": 4, "source": "flag"}
    assert payload["summary_model.verbosity"] == {"value": "high", "source": "settings"}
    assert payload["output.formats"]["value"] == list(OUTPUT_FORMATS)
    assert payload["input.path"] == {"value": "x", "source": "flag"}
    assert len(payload) == len([o for o in OPTIONS if o.target is not None])


@pytest.mark.parametrize(
    "argv",
    [
        ["--input", "x", "--formats", "summary-txt"],
        ["--input", "x", "--temperature", "2.5"],
        ["--input", "x", "--top-p", "nan"],
        ["--input", "x", "--jpeg-quality", "0"],
        ["--input", "x", "--dpi", "-3"],
        ["--input", "x", "--all", "--select", "1"],
        ["--input", "x", "--output", "o", "--beside-input"],
        ["--input", "x", "--response-format", "xml"],
    ],
)
def test_invalid_arguments_exit_with_status_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        parse(argv)
    assert info.value.code == 2


def test_service_tier_applies_only_to_openai() -> None:
    values = {"transcription_provider": "anthropic", "transcription_service_tier": "x"}
    assert OPTIONS.inapplicable(values) == ["transcription_service_tier"]


def test_recommended_images_fill_unset_fields() -> None:
    spec = ImageSpec(jpeg_quality=70)
    assert spec.over(recommended_images("openai")) == ImageSpec(
        "native", "original", "jpeg", 70, True
    )
    assert recommended_images("google").dpi == 300
    assert recommended_images("unknown") == recommended_images("custom")
