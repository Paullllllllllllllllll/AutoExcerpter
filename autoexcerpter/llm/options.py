"""Request options per provider and the response format of a phase.

:func:`build_invoke_kwargs` turns a phase's options (output limit, reasoning
effort, verbosity, temperature, top_p) into the invoke kwargs its provider
accepts, guarded by the model's capabilities. An unset output limit takes the
model's limit from the capability table on OpenAI, Anthropic and Google.
:func:`response_format_for` decides between schema, json and text.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from autoexcerpter.common.capabilities import UNKNOWN, Capabilities
from autoexcerpter.common.structured import RESPONSE_FORMATS, ResponseFormat
from autoexcerpter.llm.capabilities import (
    TABLE,
    ProviderCapabilities,
    detect_capabilities,
)

__all__ = [
    "DEFAULT_SERVICE_TIER",
    "build_invoke_kwargs",
    "request_service_tier",
    "response_format_for",
]

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_TIER = "flex"

_LIMITED_PROVIDERS = frozenset({"openai", "anthropic", "google"})

_ANTHROPIC_EFFORT_TO_BUDGET: Mapping[str, int] = {
    "none": 0,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}

# Anthropic's output_config.effort level for each reasoning effort.
# Adaptive-thinking Claude models reject thinking.budget_tokens and take only
# the effort. "none" is absent: no parameter is sent.
_ANTHROPIC_EFFORTS: Mapping[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

_ANTHROPIC_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")

# The effort levels of each Claude family that takes output_config.effort. A
# level a family lacks steps down to the next lower level it has; families
# not listed send no effort.
_ANTHROPIC_EFFORT_LEVELS: Mapping[str, tuple[str, ...]] = {
    "claude-opus-4.5": ("low", "medium", "high"),
    "claude-opus-4.6": ("low", "medium", "high", "max"),
    "claude-sonnet-4.6": ("low", "medium", "high", "max"),
    **dict.fromkeys(
        (
            "claude-opus-4.7",
            "claude-opus-4.8",
            "claude-opus-5",
            "claude-fable-5",
            "claude-sonnet-5",
            "claude-opus-5.5",
            "claude-haiku-5.5",
        ),
        _ANTHROPIC_EFFORT_ORDER,
    ),
}

_GOOGLE_EFFORT_TO_BUDGET: Mapping[str, int] = {
    "none": 0,
    "minimal": 512,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}

# Gemini 3 and later take discrete thinking levels instead of a budget.
_GOOGLE_EFFORT_TO_LEVEL: Mapping[str, str] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

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
    """Return the service tier sent with each call ("auto" outside OpenAI)."""
    if provider != "openai":
        return "auto"
    return service_tier or DEFAULT_SERVICE_TIER


def _effort(options: Mapping[str, Any]) -> str | None:
    reasoning = options.get("reasoning")
    if isinstance(reasoning, Mapping) and "effort" in reasoning:
        return str(reasoning["effort"])
    return None


def _default_limit(provider: str, caps: ProviderCapabilities) -> int | None:
    """Return the output limit sent when a phase sets none (None sends none).

    OpenAI, Anthropic and Google models matched by a capability family take
    the family's limit. An unmatched Anthropic model takes the conservative
    profile's limit, since langchain-anthropic would otherwise send 1024.
    """
    if provider not in _LIMITED_PROVIDERS:
        return None
    if caps.family != UNKNOWN and caps.provider_name == provider:
        return caps.max_output_tokens
    if provider == "anthropic":
        conservative = Capabilities(provider_name=provider, model_name="")
        limit = TABLE.unknown.get("max_output_tokens", conservative.max_output_tokens)
        return int(limit)
    return None


def _max_tokens(
    provider: str, caps: ProviderCapabilities, options: Mapping[str, Any]
) -> dict[str, Any]:
    max_tokens = options.get("max_output_tokens")
    if max_tokens is None:
        max_tokens = _default_limit(provider, caps)
    if max_tokens is None:
        return {}
    key = "max_output_tokens" if provider in ("openai", "google") else "max_tokens"
    return {key: max_tokens}


def _anthropic_effort(family: str, effort: str) -> str | None:
    """Return the effort level *family* takes for *effort*, or None for none.

    The level maps through :data:`_ANTHROPIC_EFFORTS` and steps down to the
    next lower level the family has.
    """
    levels = _ANTHROPIC_EFFORT_LEVELS.get(family)
    wanted = _ANTHROPIC_EFFORTS.get(effort)
    if levels is None or wanted is None:
        return None
    rank = _ANTHROPIC_EFFORT_ORDER.index(wanted)
    lower = reversed(_ANTHROPIC_EFFORT_ORDER[: rank + 1])
    return next((level for level in lower if level in levels), None)


def _reasoning(
    provider: str, model_name: str, caps: ProviderCapabilities, effort: str | None
) -> dict[str, Any]:
    """Return the reasoning kwargs the provider and model accept."""
    reasoning = caps.is_reasoning_model
    if provider == "openai" and reasoning:
        return {"reasoning": {"effort": effort}} if effort is not None else {}
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
    if provider == "google" and reasoning and family == "google":
        if effort is None:
            return {}
        if "gemini-3" in model_name.lower():
            level = _GOOGLE_EFFORT_TO_LEVEL.get(effort)
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
        kwargs["service_tier"] = request_service_tier(provider, service_tier)
    kwargs.update(_reasoning(provider, model_name, caps, _effort(options)))

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
