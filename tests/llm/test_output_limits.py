"""The output limit each provider receives, from the spec to the invoke kwargs."""

from __future__ import annotations

from typing import Any

import pytest

from autoexcerpter.llm.options import build_invoke_kwargs
from autoexcerpter.pipeline.job import job_from_spec
from autoexcerpter.settings import Endpoint, Settings, resolve
from autoexcerpter.spec import OPTIONS, RunSpec, parse

SETTINGS = Settings(
    endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
)
LIMIT_KEYS = ("max_tokens", "max_output_tokens")


def _spec_kwargs(spec: RunSpec) -> dict[str, Any]:
    phase = job_from_spec(spec, SETTINGS).summary
    assert phase.provider is not None
    return build_invoke_kwargs(
        phase.provider, phase.name, phase.options, phase.service_tier
    )


def _kwargs(*argv: str) -> dict[str, Any]:
    return _spec_kwargs(parse(["--input", "x", *argv]))


def _limit(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: kwargs[key] for key in LIMIT_KEYS if key in kwargs}


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--model", "claude-sonnet-4-5"], {"max_tokens": 64000}),
        (["--model", "gemini-2.5-flash"], {"max_output_tokens": 65536}),
        (["--model", "gemini-2.0-flash"], {"max_output_tokens": 8192}),
        (["--model", "gpt-4o"], {"max_output_tokens": 16384}),
        ([], {"max_output_tokens": 128000}),
        (["--provider", "anthropic", "--model", "my-model"], {"max_tokens": 4096}),
        (["--model", "anthropic/claude-sonnet-4.5"], {}),
        (["--endpoint", "local", "--model", "gpt-4o"], {}),
        (["--model", "my-finetune"], {}),
        (["--provider", "google", "--model", "my-model"], {}),
    ],
)
def test_an_unset_limit_takes_the_model_limit(
    argv: list[str], expected: dict[str, Any]
) -> None:
    assert _limit(_kwargs(*argv)) == expected


def test_an_unset_limit_is_left_out_of_the_options() -> None:
    job = job_from_spec(parse(["--input", "x"]), SETTINGS)
    assert "max_output_tokens" not in job.transcription.options
    assert "max_output_tokens" not in job.summary.options


@pytest.mark.parametrize(
    ("argv", "key"),
    [
        (["--model", "gpt-5.6-luna"], "max_output_tokens"),
        (["--model", "claude-sonnet-4-5"], "max_tokens"),
        (["--model", "gemini-2.5-flash"], "max_output_tokens"),
        (["--model", "anthropic/claude-sonnet-4.5"], "max_tokens"),
        (["--endpoint", "local", "--model", "org/model"], "max_tokens"),
        (["--model", "my-finetune"], "max_output_tokens"),
    ],
)
def test_an_explicit_limit_is_sent_as_given(argv: list[str], key: str) -> None:
    kwargs = _kwargs(*argv, "--max-output-tokens", "5000")
    assert _limit(kwargs) == {key: 5000}


def test_a_settings_default_limit_is_sent_as_given() -> None:
    settings = Settings(defaults={"max_output_tokens": 7000})
    spec = resolve(settings, OPTIONS.parse(["--input", "x"])).spec
    assert _limit(_spec_kwargs(spec)) == {"max_output_tokens": 7000}
