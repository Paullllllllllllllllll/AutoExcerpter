"""Characterization of the request options per provider, model and option.

Every case records what one model sends in each phase: the invoke kwargs of
the synchronous call, the OpenAI image detail of a transcription request and,
for the providers with a batch API, the batch request body of a page
transcription and of a summary. The cases cover the providers,
representative models and the options that shape a request (reasoning effort,
verbosity, temperature, top_p, service tier, output limit and image detail).
``golden/request_matrix.json`` holds the requests of v2.4.0 before the
request options moved to ``common/request_options.py``; the move must leave
every case identical.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from autoexcerpter.batch.bodies import ROLE_SUMMARY, ROLE_TRANSCRIPTION, build_plan
from autoexcerpter.common.batch import BatchRequest
from autoexcerpter.common.testing import WriteGuard
from autoexcerpter.llm.image_detail import resolve_request_detail
from autoexcerpter.llm.options import build_invoke_kwargs
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.pipeline.job import phase_model
from autoexcerpter.settings import Endpoint, Settings
from autoexcerpter.spec import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DEFAULT_TEMPERATURE,
    DEFAULT_VERBOSITY,
    PHASES,
    ImageSpec,
    ModelSpec,
)
from tests.matrix_golden import check_matrix

SETTINGS = Settings(
    endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
)
MODELS: tuple[tuple[str, str], ...] = (
    ("openai", "gpt-6-luna"),
    ("openai", "gpt-5.6-luna"),
    ("openai", "gpt-5.3"),
    ("anthropic", "claude-haiku-5-5"),
    ("anthropic", "claude-sonnet-4-0"),
    ("google", "gemini-3.1-pro"),
    ("google", "gemini-2.5-flash"),
    ("openrouter", "anthropic/claude-sonnet-4.5"),
    ("custom", "org/fake-model"),
)
BATCH_PROVIDERS = frozenset({"openai", "anthropic", "google"})

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "defaults": {},
    "effort-none": {"reasoning_effort": "none"},
    "effort-high": {"reasoning_effort": "high"},
    "effort-max": {"reasoning_effort": "max"},
    "effort-none-limit-2000": {"reasoning_effort": "none", "max_output_tokens": 2000},
    "effort-max-limit-2000": {"reasoning_effort": "max", "max_output_tokens": 2000},
    "verbosity-low": {"verbosity": "low"},
    "verbosity-high": {"verbosity": "high"},
    "temperature-0.3": {"temperature": 0.3},
    "top-p-0.5": {"top_p": 0.5},
    "samplers-effort-none": {
        "reasoning_effort": "none",
        "temperature": 0.3,
        "top_p": 0.5,
    },
    "tier-flex": {"service_tier": "flex"},
    "tier-priority": {"service_tier": "priority"},
    "tier-default": {"service_tier": "default"},
    "tier-auto": {"service_tier": "auto"},
    "limit-unset": {"max_output_tokens": None},
    "limit-2000": {"max_output_tokens": 2000},
    "limit-500000": {"max_output_tokens": 500000},
}
DETAILS: tuple[str | None, ...] = (None, "low", "high", "auto", "original")
IMAGE = BatchRequest(
    custom_id="page-0", image_base64="AAAA", mime_type="image/jpeg", order_index=0
)
TEXT = BatchRequest(custom_id="summary-0", text="## Page text", order_index=0)
LOGGER = logging.getLogger(__name__)


def _model(provider: str, model: str, phase: str, **changes: Any) -> ModelSpec:
    values: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "endpoint": "local" if provider == "custom" else None,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "verbosity": DEFAULT_VERBOSITY[phase],
        "max_output_tokens": None,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": None,
        "service_tier": DEFAULT_SERVICE_TIER,
        "response_format": None,
    }
    values.update(changes)
    return ModelSpec(**values)


def _outcome(build: Callable[..., Any], *args: Any) -> Any:
    """Return what *build* returns for *args*, or the error it raises."""
    try:
        return build(*args)
    except Exception as exc:  # noqa: BLE001 - the error is the recorded outcome
        return {"<error>": f"{type(exc).__name__}: {exc}"}


def _invoke_kwargs(phase: PhaseModel) -> dict[str, Any]:
    return build_invoke_kwargs(
        phase.provider, phase.name, phase.options, phase.service_tier
    )


def _sync_detail(phase: PhaseModel) -> str | None:
    return resolve_request_detail(
        dict(phase.options), phase.provider, phase.name, LOGGER
    )


def _body(phase: PhaseModel, role: str) -> dict[str, Any]:
    plan = build_plan(phase, role)
    return plan.body(IMAGE if role == ROLE_TRANSCRIPTION else TEXT)


def _phase_case(phase: PhaseModel, role: str) -> dict[str, Any]:
    case: dict[str, Any] = {
        "options": phase.options,
        "sync": _outcome(_invoke_kwargs, phase),
    }
    if role == ROLE_TRANSCRIPTION:
        case["sync_detail"] = _outcome(_sync_detail, phase)
    if phase.provider in BATCH_PROVIDERS:
        case["batch"] = _outcome(_body, phase, role)
    return case


def _case(
    provider: str,
    model: str,
    changes: Mapping[str, Any],
    images: ImageSpec | None,
    phases: tuple[str, ...] = PHASES,
) -> dict[str, Any]:
    case: dict[str, Any] = {}
    for name in phases:
        spec = _model(provider, model, name, **changes)
        phase = phase_model(spec, SETTINGS, name, images)
        role = ROLE_TRANSCRIPTION if name == "transcription" else ROLE_SUMMARY
        case[name] = _phase_case(phase, role)
    return case


def _cases() -> Iterator[tuple[str, dict[str, Any]]]:
    for provider, model in MODELS:
        for name, changes in VARIANTS.items():
            yield f"{provider}:{model}:{name}", _case(provider, model, changes, None)
        for detail in DETAILS[1:]:
            images = ImageSpec(detail=detail)
            key = f"{provider}:{model}:detail-{detail}"
            yield key, _case(provider, model, {}, images, ("transcription",))


def test_requests_match_the_golden(write_guard: WriteGuard) -> None:
    check_matrix(__file__, "request_matrix", dict(_cases()), write_guard.consume)
