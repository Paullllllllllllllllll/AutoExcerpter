"""Request options per provider and the response format of a phase.

:func:`build_invoke_kwargs` turns a phase's options (output limit, reasoning
effort, verbosity, temperature, top_p) into the invoke kwargs its provider
accepts, guarded by the model's capabilities. An unset output limit takes the
model's limit from the capability table on OpenAI, Anthropic and Google
(:data:`LIMITS`). The capability table also decides the OpenAI service tier
(``flex`` only for the models listed for it), the reasoning effort or
thinking level a model documents and whether it takes a verbosity. The
shared tables and mechanics come from ``common.request_options``; the tables
below are this tool's variants. :func:`response_format_for` decides between
schema, json and text.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from autoexcerpter.common.capabilities import UNKNOWN, Capabilities
from autoexcerpter.common.request_options import (
    ANTHROPIC_EFFORT_LEVELS,
    ANTHROPIC_EFFORTS,
    GOOGLE_LEVELS,
    OutputLimits,
    anthropic_effort,
    clamp_effort,
    google_thinking_level,
    openai_service_tier,
)
from autoexcerpter.common.structured import RESPONSE_FORMATS, ResponseFormat
from autoexcerpter.llm.capabilities import (
    TABLE,
    ProviderCapabilities,
    detect_capabilities,
)

__all__ = [
    "DEFAULT_SERVICE_TIER",
    "LIMITS",
    "build_invoke_kwargs",
    "request_service_tier",
    "response_format_for",
]

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_TIER = "flex"

LIMITS = OutputLimits(
    conservative=int(
        TABLE.unknown.get(
            "max_output_tokens",
            Capabilities(provider_name=UNKNOWN, model_name="").max_output_tokens,
        )
    )
)
"""A set output limit is sent as given; an unset one takes a known family's
limit on OpenAI, Anthropic and Google (the conservative limit for an
unmatched Anthropic model, since langchain-anthropic would otherwise send
1024) and none elsewhere."""

_ANTHROPIC_EFFORT_TO_BUDGET: Mapping[str, int] = {
    "none": 0,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}

# Adaptive-thinking Claude models reject thinking.budget_tokens and take only
# the effort. "none" sends no effort here.
_ANTHROPIC_EFFORTS: Mapping[str, str] = {
    effort: level for effort, level in ANTHROPIC_EFFORTS.items() if effort != "none"
}

_GOOGLE_EFFORT_TO_BUDGET: Mapping[str, int] = {
    "none": 0,
    "minimal": 512,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}

# Gemini 3 and later take discrete thinking levels instead of a budget; xhigh
# takes high here.
_GOOGLE_EFFORT_TO_LEVEL: Mapping[str, str] = {**GOOGLE_LEVELS, "xhigh": "high"}

_REASONING_KEYS = ("reasoning", "thinking", "thinking_config", "output_config")


def response_format_for(
    requested: str | None, provider: str, model_name: str
) -> ResponseFormat:
    """Return the response format a phase uses.

    Unset takes schema when the model supports structured output, else json.
    A custom endpoint's format is decided by its settings, so unset means
    json there. Schema on a model without structured output falls back to
    json with a warning.

    Raises:
        ValueError: *requested* is not a response format.
    """
    if requested is not None and requested not in RESPONSE_FORMATS:
        raise ValueError(f"Unknown response format: {requested!r}")
    if provider == "custom":
        return requested or "json"
    supported = detect_capabilities(model_name).supports_structured_output
    if requested is None:
        return "schema" if supported else "json"
    if requested == "schema" and not supported:
        logger.warning(
            "Model %s has no structured output; using prompted JSON", model_name
        )
        return "json"
    return requested


def request_service_tier(provider: str, service_tier: str | None) -> str:
    """Return the service tier a call asks for ("auto" outside OpenAI).

    :func:`build_invoke_kwargs` sends it where the model takes it.
    """
    if provider != "openai":
        return "auto"
    return service_tier or DEFAULT_SERVICE_TIER


def _effort(options: Mapping[str, Any]) -> str | None:
    reasoning = options.get("reasoning")
    if isinstance(reasoning, Mapping) and "effort" in reasoning:
        return str(reasoning["effort"])
    return None


def _max_tokens(
    provider: str, caps: ProviderCapabilities, options: Mapping[str, Any]
) -> dict[str, Any]:
    max_tokens = LIMITS.resolve(provider, options.get("max_output_tokens"), caps)
    if max_tokens is None:
        return {}
    key = "max_output_tokens" if provider in ("openai", "google") else "max_tokens"
    return {key: max_tokens}


def _anthropic_effort(family: str, effort: str) -> str | None:
    """Return the effort level *family* takes for *effort*, or None for none."""
    return anthropic_effort(
        family, effort, efforts=_ANTHROPIC_EFFORTS, levels=ANTHROPIC_EFFORT_LEVELS
    )


def _reasoning(
    provider: str, caps: ProviderCapabilities, effort: str | None
) -> dict[str, Any]:
    """Return the reasoning kwargs the provider and model accept.

    OpenAI and Gemini models get an effort or thinking level they document
    (``clamp_effort``); models without an effort control get none.
    """
    reasoning = caps.is_reasoning_model
    if provider == "openai" and reasoning:
        if effort is None or not caps.supports_reasoning_effort:
            return {}
        return {"reasoning": {"effort": clamp_effort(effort, caps, log=logger)}}
    family = caps.provider_name
    adaptive = caps.uses_adaptive_thinking and family == "anthropic"
    extended = reasoning and family == "anthropic" and not caps.uses_adaptive_thinking
    if provider == "anthropic" and (adaptive or extended):
        if effort is None:
            return {}
        level = _anthropic_effort(caps.family, effort)
        output_config = {"output_config": {"effort": level}} if level else {}
        if adaptive:
            return output_config
        budget = _ANTHROPIC_EFFORT_TO_BUDGET.get(effort)
        if budget:
            thinking = {"type": "enabled", "budget_tokens": budget}
            return {"thinking": thinking, **output_config}
        return {}
    if provider == "google" and caps.supports_reasoning_effort and family == "google":
        if effort is None:
            return {}
        if not caps.family.startswith("gemini-2"):
            level = google_thinking_level(effort, caps, _GOOGLE_EFFORT_TO_LEVEL)
            return {"thinking_config": {"thinking_level": level}} if level else {}
        budget = _GOOGLE_EFFORT_TO_BUDGET.get(effort)
        return {"thinking_config": {"thinking_budget": budget}} if budget else {}
    return {}


def build_invoke_kwargs(
    provider: str,
    model_name: str,
    options: Mapping[str, Any],
    service_tier: str | None = None,
) -> dict[str, Any]:
    """Return the invoke kwargs of one request, guarded by capabilities.

    Temperature and top_p are left out while reasoning or thinking is
    active, and for models that reject them.
    """
    caps = detect_capabilities(model_name)
    kwargs: dict[str, Any] = _max_tokens(provider, caps, options)
    if provider == "openai":
        tier = openai_service_tier(request_service_tier(provider, service_tier), caps)
        if tier is not None:
            kwargs["service_tier"] = tier
    kwargs.update(_reasoning(provider, caps, _effort(options)))

    text = options.get("text")
    if (
        provider == "openai"
        and caps.supports_text_verbosity
        and isinstance(text, Mapping)
        and "verbosity" in text
    ):
        kwargs["text"] = {"verbosity": text["verbosity"]}

    reasoning_active = any(key in kwargs for key in _REASONING_KEYS)
    temperature = options.get("temperature")
    if temperature is not None and caps.supports_temperature and not reasoning_active:
        kwargs["temperature"] = float(temperature)
    top_p = options.get("top_p")
    if top_p is not None and caps.supports_top_p and not reasoning_active:
        kwargs["top_p"] = float(top_p)
    return kwargs
