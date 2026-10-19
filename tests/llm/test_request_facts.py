"""Requests follow the settled capability facts in both paths.

Synchronous calls send :func:`build_invoke_kwargs`; batch bodies are built
from the same kwargs, so each fact is checked on the kwargs and, where the
batch body could differ (the service tier), on the body too. The vendors'
sources are cited in ``common/capability_table.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoexcerpter.batch.bodies import ROLE_TRANSCRIPTION, build_plan
from autoexcerpter.common.batch import BatchRequest
from autoexcerpter.llm.capabilities import detect_capabilities, detect_provider
from autoexcerpter.llm.options import build_invoke_kwargs
from autoexcerpter.pipeline.job import phase_model
from autoexcerpter.settings import Settings
from autoexcerpter.spec import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
    DEFAULT_VERBOSITY,
    ModelSpec,
)

IMAGE = BatchRequest(
    custom_id="page-0", image_base64="AAAA", mime_type="image/jpeg", order_index=0
)


def _kwargs(provider: str, model: str, **options: Any) -> dict[str, Any]:
    tier = options.pop("service_tier", None)
    return build_invoke_kwargs(provider, model, options, tier)


def _effort(effort: str) -> dict[str, Any]:
    return {"reasoning": {"effort": effort}}


def _batch_body(model: str, tier: str) -> dict[str, Any]:
    spec = ModelSpec(
        provider="openai",
        model=model,
        endpoint=None,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        verbosity=DEFAULT_VERBOSITY["transcription"],
        max_output_tokens=None,
        temperature=DEFAULT_TEMPERATURE,
        top_p=None,
        service_tier=tier,
        response_format=None,
    )
    phase = phase_model(spec, Settings(), "transcription", None)
    return build_plan(phase, ROLE_TRANSCRIPTION).body(IMAGE)


@pytest.mark.parametrize(
    ("model", "sync_tier", "batch_tier"),
    [
        ("gpt-6-luna", "flex", "auto"),
        ("gpt-5.4-mini", "flex", "auto"),
        ("o3", "flex", "auto"),
        ("gpt-5.2-pro", None, None),
        ("gpt-5.3-codex", None, None),
        ("gpt-5.3-chat-latest", None, None),
        ("gpt-4o", None, None),
    ],
)
def test_flex_goes_only_to_the_listed_models(
    model: str, sync_tier: str | None, batch_tier: str | None
) -> None:
    assert _kwargs("openai", model, service_tier="flex").get("service_tier") == (
        sync_tier
    )
    assert _batch_body(model, "flex").get("service_tier") == batch_tier


def test_other_tiers_still_reach_every_openai_model() -> None:
    assert _kwargs("openai", "gpt-4o", service_tier="priority")["service_tier"] == (
        "priority"
    )


@pytest.mark.parametrize(
    ("model", "effort", "sent"),
    [
        ("gpt-6-astra", "none", "low"),
        ("gpt-6-astra", "minimal", "low"),
        ("gpt-6.1-sol", "none", "low"),
        ("gpt-6-luna", "minimal", "low"),
        ("gpt-6-luna", "none", "none"),
        ("gpt-6-luna", "max", "max"),
        ("gpt-5.6-terra", "minimal", "low"),
        ("gpt-5.5", "max", "xhigh"),
        ("gpt-5.1", "xhigh", "high"),
        ("gpt-5", "none", "minimal"),
        ("gpt-5.5-pro", "low", "medium"),
        ("gpt-5.4-pro", "max", "xhigh"),
        ("gpt-5.3-codex", "none", "low"),
    ],
)
def test_openai_effort_is_one_the_model_documents(
    model: str, effort: str, sent: str
) -> None:
    assert _kwargs("openai", model, **_effort(effort))["reasoning"] == {"effort": sent}


def test_an_effort_clamp_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    model = "gpt-6.1-sol-2026-09-01"
    with caplog.at_level("WARNING"):
        _kwargs("openai", model, **_effort("none"))
        _kwargs("openai", model, **_effort("none"))

    hits = [r for r in caplog.records if model in r.getMessage()]
    assert len(hits) == 1


@pytest.mark.parametrize("model", ["o1", "o1-mini"])
def test_o1_models_get_no_reasoning_effort(model: str) -> None:
    assert "reasoning" not in _kwargs("openai", model, **_effort("high"))


@pytest.mark.parametrize(
    ("model", "effort", "level"),
    [
        ("gemini-3.1-pro", "minimal", "low"),
        ("gemini-3.8-flash", "minimal", "low"),
        ("gemini-3-flash", "minimal", "minimal"),
        ("gemma-4-31b-it", "low", "high"),
    ],
)
def test_gemini_level_is_one_the_model_documents(
    model: str, effort: str, level: str
) -> None:
    kwargs = _kwargs("google", model, **_effort(effort))

    assert kwargs["thinking_config"] == {"thinking_level": level}


@pytest.mark.parametrize("model", ["gemini-3-pro-image", "gemini-3.1-flash-image"])
def test_gemini_image_models_get_no_thinking_level(model: str) -> None:
    assert "thinking_config" not in _kwargs("google", model, **_effort("high"))


@pytest.mark.parametrize(
    "model", ["gpt-5-chat-latest", "gpt-5.2-chat-latest", "gpt-5.3-chat-latest"]
)
def test_chat_latest_models_are_not_reasoning_models(model: str) -> None:
    kwargs = _kwargs(
        "openai",
        model,
        **_effort("high"),
        text={"verbosity": "low"},
        temperature=0.3,
        top_p=0.5,
    )

    assert kwargs == {"max_output_tokens": 16384, "service_tier": "default"}


@pytest.mark.parametrize("model", ["gpt-5.2-codex", "gpt-5.3-codex"])
def test_codex_models_take_images_and_their_effort_range(model: str) -> None:
    kwargs = _kwargs("openai", model, **_effort("max"), text={"verbosity": "low"})

    assert detect_capabilities(model).supports_vision
    assert kwargs["reasoning"] == {"effort": "xhigh"}
    assert "text" not in kwargs


@pytest.mark.parametrize(
    ("model", "sent"),
    [("gpt-6-luna", True), ("gpt-5.1", True), ("gpt-5.5-pro", False), ("o3", False)],
)
def test_verbosity_goes_to_the_documented_models(model: str, sent: bool) -> None:
    kwargs = _kwargs("openai", model, text={"verbosity": "low"})

    assert ("text" in kwargs) is sent


def test_openrouter_samplers_follow_the_model_list() -> None:
    r1 = _kwargs("openrouter", "deepseek/deepseek-r1", temperature=0.3)
    opus = _kwargs("openrouter", "anthropic/claude-opus-5.5", temperature=0.3)

    assert r1["temperature"] == 0.3
    assert "temperature" not in opus


def test_gemini_3_5_flash_takes_no_samplers() -> None:
    kwargs = _kwargs("google", "gemini-3.5-flash", temperature=0.3)

    assert "temperature" not in kwargs


@pytest.mark.parametrize(
    ("model", "provider", "family"),
    [
        ("models/gemini-2.5-flash", "google", "gemini-2.5-flash"),
        ("openai:gpt-5.6-luna", "openai", "gpt-5.6-luna"),
    ],
)
def test_prefixed_ids_resolve_like_the_other_tools(
    model: str, provider: str, family: str
) -> None:
    assert detect_provider(model) == provider
    assert detect_capabilities(model).family == family
