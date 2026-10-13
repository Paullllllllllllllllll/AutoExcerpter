"""Provider request options: output limits, service tiers, samplers and thinking.

The tools share the mechanics and tables here and inject their own policy.
:class:`OutputLimits` decides the output limit sent: a known family's limit,
the conservative limit of an unmatched Anthropic model, a routed default for
providers outside OpenAI, Anthropic and Google, and a clamp for the providers
a tool names. :class:`RequestRules` holds a tool's choices for the
synchronous options that :func:`openai_options`, :func:`anthropic_options`,
:func:`google_options` and :func:`openrouter_options` build. The other
functions build what synchronous calls and batch bodies share: the OpenAI
and batch service tiers, the reasoning effort a model documents, the OpenAI
prompt-cache parameter, sampler values, Anthropic's effort level and
thinking block, Gemini's thinking level (Gemini 3) or budget (Gemini 2) and
OpenRouter's reasoning object. Each table is a default that a tool may
replace with its own.

The capability table decides what an OpenAI request carries, in synchronous
calls and batch bodies alike: ``flex`` only for the models OpenAI lists for
Flex processing (:func:`openai_service_tier`), a reasoning effort the model
documents (:func:`clamp_effort`), the verbosity only where
``supports_text_verbosity`` is set, and the prompt-cache parameter the model
takes (:func:`openai_prompt_cache`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from .capabilities import UNKNOWN, Capabilities

__all__ = [
    "ANSWER_RESERVE",
    "ANTHROPIC_BUDGETS",
    "ANTHROPIC_DEFAULT_BUDGET",
    "ANTHROPIC_EFFORTS",
    "ANTHROPIC_EFFORT_LEVELS",
    "ANTHROPIC_EFFORT_ORDER",
    "BATCH_SERVICE_TIERS",
    "EFFORT_ORDER",
    "GOOGLE_BUDGET_CAP",
    "GOOGLE_BUDGET_CAPS",
    "GOOGLE_BUDGET_RATIOS",
    "GOOGLE_LEVELS",
    "LIMITED_PROVIDERS",
    "OPENROUTER_BUDGET_CAP",
    "OPENROUTER_DEFAULT_RATIO",
    "OPENROUTER_RATIOS",
    "PROMPT_CACHE_TTL",
    "ROUTED_LIMIT",
    "SAMPLERS",
    "THINKING_FLOOR",
    "ModelOptions",
    "OutputLimits",
    "RequestRules",
    "anthropic_effort",
    "anthropic_options",
    "anthropic_thinking",
    "batch_reasoning",
    "batch_service_tier",
    "clamp_effort",
    "disabled_params",
    "google_options",
    "google_thinking_budget",
    "google_thinking_level",
    "openai_options",
    "openai_prompt_cache",
    "openai_service_tier",
    "openrouter_options",
    "openrouter_reasoning",
    "sampler_values",
]

logger = logging.getLogger(__name__)

ROUTED_LIMIT = 128000
"""Output limit sent to OpenRouter and custom endpoints when none is set."""
LIMITED_PROVIDERS = frozenset({"openai", "anthropic", "google"})
"""Providers whose unset output limit takes the model family's limit."""
BATCH_SERVICE_TIERS = frozenset({"auto", "default", "priority"})
"""The service tiers a batch body carries; ``flex`` becomes ``auto``."""
SAMPLERS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
THINKING_FLOOR = 1024
"""The smallest thinking budget sent."""
ANSWER_RESERVE = 256
"""Tokens of the output limit a thinking budget leaves for the answer."""
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
"""Every reasoning effort and thinking level, lowest first."""
PROMPT_CACHE_TTL = "30m"
"""The ``prompt_cache_options.ttl`` of GPT-5.6 and later, its only value."""

_logged: set[tuple[str, str, str]] = set()

ANTHROPIC_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_EFFORTS: Mapping[str, str] = MappingProxyType(
    {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
)
"""Anthropic's ``output_config.effort`` level for each reasoning effort.

The Messages API takes low, medium, high, xhigh and max
(``anthropic.types.OutputConfigParam``); none and minimal take low, the
lowest level.
"""
ANTHROPIC_EFFORT_LEVELS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
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
                "claude-sonnet-5.5",
                "claude-haiku-5.5",
            ),
            ANTHROPIC_EFFORT_ORDER,
        ),
    }
)
"""The effort levels of each Claude family that takes ``output_config.effort``.

A level a family lacks steps down to the next lower level it has; families
not listed send no effort.
"""
ANTHROPIC_BUDGETS: Mapping[str, int] = MappingProxyType(
    {"low": 1024, "medium": 4096, "high": 16384}
)
"""The thinking budget of each effort on Claude models without adaptive thinking."""
ANTHROPIC_DEFAULT_BUDGET = 4096

GOOGLE_LEVELS: Mapping[str, str] = MappingProxyType(
    {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": "high",
    }
)
"""Gemini's ``thinking_level`` for each reasoning effort (Gemini 3 and later)."""
GOOGLE_BUDGET_RATIOS: Mapping[str, float] = MappingProxyType(
    {
        "minimal": 0.10,
        "low": 0.10,
        "medium": 0.25,
        "high": 0.50,
        "xhigh": 0.50,
        "max": 0.50,
    }
)
"""Share of the output limit a Gemini 2 thinking budget takes, per effort."""
GOOGLE_BUDGET_CAPS: Mapping[str, int] = MappingProxyType(
    {"gemini-2.5-flash": 24576, "gemini-2.5-pro": 32768}
)
"""The largest thinking budget each Gemini 2 family accepts."""
GOOGLE_BUDGET_CAP = 24576

OPENROUTER_RATIOS: Mapping[str, float] = MappingProxyType(
    {
        "max": 1.0,
        "xhigh": 0.95,
        "high": 0.80,
        "medium": 0.50,
        "low": 0.20,
        "minimal": 0.10,
        "none": 0.0,
    }
)
"""Share of the output limit an OpenRouter reasoning budget takes, per effort."""
OPENROUTER_DEFAULT_RATIO = 0.50
OPENROUTER_BUDGET_CAP = 32000


class ModelOptions(Protocol):
    """The options of one model that shape its requests."""

    @property
    def model(self) -> str: ...

    @property
    def reasoning_effort(self) -> str: ...

    @property
    def verbosity(self) -> str: ...

    @property
    def temperature(self) -> float: ...

    @property
    def top_p(self) -> float | None: ...

    @property
    def service_tier(self) -> str: ...


@dataclass(frozen=True)
class OutputLimits:
    """How a tool decides the output limit of a request.

    *conservative* is the limit of an unmatched Anthropic model, which
    langchain-anthropic would otherwise send as 1024. A set limit is sent
    as given, clamped to a known family's limit on the providers in
    *clamped*. An unset limit takes a known family's limit on OpenAI,
    Anthropic and Google and *routed* (None: no limit) elsewhere.
    """

    conservative: int
    clamped: frozenset[str] = frozenset()
    routed: int | None = None

    def resolve(
        self, provider: str, requested: int | None, caps: Capabilities
    ) -> int | None:
        """Return the output limit sent to *provider*, or None to send none."""
        known = caps.family != UNKNOWN and caps.provider_name == provider
        if requested is not None:
            if provider in self.clamped and known:
                return min(requested, caps.max_output_tokens)
            return requested
        if provider not in LIMITED_PROVIDERS:
            return self.routed
        if known:
            return caps.max_output_tokens
        if provider == "anthropic":
            return self.conservative
        return None


@dataclass(frozen=True)
class RequestRules:
    """A tool's choices for the synchronous options of each provider.

    With *reasoning_only_service_tier* the OpenAI service tier is sent to
    reasoning models only; with *gemini2_budget* a Gemini 2 model takes a
    thinking budget (:func:`google_thinking_budget`).
    """

    limits: OutputLimits
    reasoning_only_service_tier: bool = False
    gemini2_budget: bool = False


def _log_once(log: logging.Logger, key: tuple[str, str, str], message: str) -> None:
    """Log *message* with the parts of *key*, once per *key* and process."""
    if key in _logged:
        return
    _logged.add(key)
    log.warning(message, *key)


def clamp_effort(
    effort: str, caps: Capabilities, *, log: logging.Logger = logger
) -> str:
    """Return the reasoning effort or thinking level sent for *effort*.

    A level in ``caps.reasoning_efforts`` stays; another steps up to the next
    level the model documents, or down to its highest level when none lies
    above. A model without documented levels, or an effort outside
    :data:`EFFORT_ORDER`, keeps *effort*. Each substitution is logged once per
    model and effort.
    """
    levels = caps.reasoning_efforts
    if not levels or effort in levels or effort not in EFFORT_ORDER:
        return effort
    rank = EFFORT_ORDER.index(effort)
    ranked = sorted(levels, key=EFFORT_ORDER.index)
    sent = next(
        (level for level in ranked if EFFORT_ORDER.index(level) > rank), ranked[-1]
    )
    _log_once(
        log,
        (caps.model_name, effort, sent),
        "Model %s does not take reasoning effort %r; sending %r",
    )
    return sent


def openai_service_tier(
    tier: str | None, caps: Capabilities, *, log: logging.Logger = logger
) -> str | None:
    """Return the service tier an OpenAI request carries, or None to send none.

    ``flex`` goes only to the models OpenAI lists for Flex processing
    (``caps.supports_flex_tier``); elsewhere no tier is sent, which is
    logged once per model.
    """
    if not tier:
        return None
    if tier == "flex" and not caps.supports_flex_tier:
        _log_once(
            log,
            (caps.model_name, tier, "no service tier"),
            "Model %s does not take service tier %r; sending %s",
        )
        return None
    return tier


def batch_service_tier(tier: str | None) -> str | None:
    """Return the service tier a batch body carries, or None to send none.

    The Batch API takes no ``flex``; it becomes ``auto``. Pass the tier of
    :func:`openai_service_tier`, so a model without Flex sends none.
    """
    if tier == "flex":
        return "auto"
    return tier if tier in BATCH_SERVICE_TIERS else None


def openai_prompt_cache(caps: Capabilities, retention: str | None) -> dict[str, Any]:
    """Return the prompt-cache parameter of an OpenAI request.

    With a configured *retention*, GPT-5.6 and later take
    ``prompt_cache_options.ttl`` (:data:`PROMPT_CACHE_TTL`), the models
    documented for extended retention ``prompt_cache_retention``, and every
    other model nothing.
    """
    if not retention:
        return {}
    if caps.prompt_cache_parameter == "prompt_cache_options":
        return {"prompt_cache_options": {"ttl": PROMPT_CACHE_TTL}}
    if caps.prompt_cache_parameter == "prompt_cache_retention":
        return {"prompt_cache_retention": retention}
    return {}


def disabled_params(caps: Capabilities) -> dict[str, None] | None:
    """Return the sampler parameters LangChain must drop for *caps*."""
    if not caps.supports_temperature:
        return dict.fromkeys(SAMPLERS)
    flags = {
        "top_p": caps.supports_top_p,
        "frequency_penalty": caps.supports_frequency_penalty,
        "presence_penalty": caps.supports_presence_penalty,
    }
    disabled = {name: None for name, supported in flags.items() if not supported}
    return disabled or None


def sampler_values(
    caps: Capabilities, temperature: Any, top_p: Any, penalty: float
) -> dict[str, Any]:
    """Return the sampler values a model with *caps* accepts.

    The penalties take *penalty*.
    """
    values: dict[str, Any] = {}
    if caps.supports_temperature:
        values["temperature"] = temperature
    if caps.supports_top_p:
        values["top_p"] = top_p
    if caps.supports_frequency_penalty:
        values["frequency_penalty"] = penalty
    if caps.supports_presence_penalty:
        values["presence_penalty"] = penalty
    return values


def openai_options(
    model: ModelOptions, caps: Capabilities, limit: int | None, rules: RequestRules
) -> dict[str, Any]:
    """Return the service tier, limit, samplers, reasoning and verbosity.

    A reasoning model takes ``max_completion_tokens``, no sampler values and,
    where the model takes one, a reasoning effort it documents; another model
    takes ``max_tokens`` and the samplers it accepts. The verbosity goes to
    models with ``supports_text_verbosity``, ``flex`` to the models listed
    for it (:func:`openai_service_tier`).
    """
    options: dict[str, Any] = {}
    tier = openai_service_tier(model.service_tier, caps)
    if tier and (caps.is_reasoning_model or not rules.reasoning_only_service_tier):
        options["service_tier"] = tier
    if caps.is_reasoning_model:
        if limit is not None:
            options["max_completion_tokens"] = limit
        options.update(dict.fromkeys(SAMPLERS))
        if caps.supports_reasoning_effort:
            options["reasoning"] = {
                "effort": clamp_effort(model.reasoning_effort, caps)
            }
    else:
        if limit is not None:
            options["max_tokens"] = limit
        options.update(sampler_values(caps, model.temperature, model.top_p, 0.0))
    if caps.supports_text_verbosity and model.verbosity:
        options["verbosity"] = model.verbosity
    return options


def batch_reasoning(model: ModelOptions, caps: Capabilities) -> dict[str, Any]:
    """Return the verbosity and reasoning of an OpenAI batch body.

    As in :func:`openai_options`: the verbosity where the model has
    ``supports_text_verbosity``, ``reasoning`` with a documented effort where
    it takes one.
    """
    options: dict[str, Any] = {}
    if caps.supports_text_verbosity and model.verbosity:
        options["verbosity"] = model.verbosity
    if caps.supports_reasoning_effort:
        options["reasoning"] = {"effort": clamp_effort(model.reasoning_effort, caps)}
    return options


def anthropic_effort(
    family: str,
    effort: str | None,
    efforts: Mapping[str, str] = ANTHROPIC_EFFORTS,
    levels: Mapping[str, tuple[str, ...]] = ANTHROPIC_EFFORT_LEVELS,
) -> str | None:
    """Return Anthropic's ``output_config.effort`` level, or None to send none.

    *effort* maps through *efforts* and steps down to the next lower level
    that *family* has in *levels*.
    """
    wanted = efforts.get(effort) if effort is not None else None
    allowed = levels.get(family)
    if allowed is None or wanted is None:
        return None
    rank = ANTHROPIC_EFFORT_ORDER.index(wanted)
    lower = reversed(ANTHROPIC_EFFORT_ORDER[: rank + 1])
    return next((level for level in lower if level in allowed), None)


def anthropic_thinking(
    model: ModelOptions,
    caps: Capabilities,
    limit: int | None,
    conservative: int,
    *,
    budgets: Mapping[str, int] = ANTHROPIC_BUDGETS,
    default_budget: int = ANTHROPIC_DEFAULT_BUDGET,
    log: logging.Logger = logger,
) -> dict[str, Any] | None:
    """Return Anthropic's ``thinking`` block, or None when thinking stays off.

    Adaptive-thinking models take ``{"type": "adaptive"}``; the others the
    effort's budget below the output limit *limit* (None: *conservative*),
    leaving room for the answer. Effort none sends no thinking.
    """
    if not caps.supports_reasoning_effort:
        return None
    if caps.uses_adaptive_thinking:
        return {"type": "adaptive"}
    effort = model.reasoning_effort
    if effort == "none":
        return None
    if limit is None:
        limit = conservative
    upper = limit - ANSWER_RESERVE
    if upper < THINKING_FLOOR:
        log.info(
            "No thinking for model %s: max_tokens=%d leaves no room for a "
            "thinking budget and an answer",
            model.model,
            limit,
        )
        return None
    budget = budgets.get(effort, default_budget)
    return {"type": "enabled", "budget_tokens": min(budget, upper)}


def anthropic_options(
    model: ModelOptions,
    caps: Capabilities,
    limit: int,
    conservative: int,
    *,
    log: logging.Logger = logger,
) -> dict[str, Any]:
    """Return the effort, the thinking block or the samplers of a Claude model.

    With thinking on, no sampler values are sent.
    """
    options: dict[str, Any] = {}
    effort = anthropic_effort(caps.family, model.reasoning_effort)
    if effort is not None:
        options["effort"] = effort
    thinking = anthropic_thinking(model, caps, limit, conservative, log=log)
    if thinking is not None:
        options["thinking"] = thinking
        return options
    if caps.supports_temperature:
        options["temperature"] = model.temperature
    if caps.supports_top_p:
        options["top_p"] = model.top_p
    return options


def google_thinking_level(
    effort: str, caps: Capabilities, levels: Mapping[str, str] = GOOGLE_LEVELS
) -> str | None:
    """Return Gemini's ``thinking_level`` for *effort*, or None to send none.

    Gemini 2 models, models without a thinking-level control and efforts
    without a level send none; a level the model does not document becomes
    one it does (:func:`clamp_effort`).
    """
    level = levels.get(effort)
    if (
        caps.supports_reasoning_effort
        and level is not None
        and not caps.family.startswith("gemini-2")
    ):
        return clamp_effort(level, caps)
    return None


def google_thinking_budget(
    effort: str,
    caps: Capabilities,
    limit: int | None,
    *,
    ratios: Mapping[str, float] = GOOGLE_BUDGET_RATIOS,
    family_caps: Mapping[str, int] = GOOGLE_BUDGET_CAPS,
    default_cap: int = GOOGLE_BUDGET_CAP,
) -> int | None:
    """Return the ``thinking_budget`` of a Gemini 2 model, or None to send none.

    The budget is the share of the output limit *limit* that *ratios* give
    the effort, at least :data:`THINKING_FLOOR`, at most the family's largest
    budget and below the limit. Effort none, an unknown limit or a limit
    without room for the floor send none.
    """
    ratio = ratios.get(effort)
    if (
        not caps.supports_reasoning_effort
        or not caps.family.startswith("gemini-2")
        or ratio is None
        or limit is None
    ):
        return None
    cap = family_caps.get(caps.family, default_cap)
    budget = max(THINKING_FLOOR, min(int(limit * ratio), cap))
    budget = min(budget, limit - 1)
    return budget if budget >= THINKING_FLOOR else None


def google_options(
    model: ModelOptions, caps: Capabilities, limit: int | None, rules: RequestRules
) -> dict[str, Any]:
    """Return the output limit and the thinking level or budget of a Gemini model."""
    options: dict[str, Any] = {}
    if limit is not None:
        options["max_tokens"] = limit
    level = google_thinking_level(model.reasoning_effort, caps)
    if level is not None:
        options["thinking_level"] = level
    if rules.gemini2_budget:
        budget = google_thinking_budget(model.reasoning_effort, caps, limit)
        if budget is not None:
            options["thinking_budget"] = budget
    return options


def _openrouter_budget(limit: int, effort: str) -> int:
    """Return the reasoning token budget, or 0 to keep effort-based reasoning."""
    ratio = OPENROUTER_RATIOS.get(effort.strip().lower(), OPENROUTER_DEFAULT_RATIO)
    upper = max(0, limit - ANSWER_RESERVE)
    if ratio <= 0 or upper < THINKING_FLOOR:
        return 0
    budget = min(int(limit * ratio), OPENROUTER_BUDGET_CAP, upper)
    return min(max(budget, THINKING_FLOOR), upper)


def openrouter_reasoning(
    model_name: str, effort: str, caps: Capabilities, limit: int | None
) -> dict[str, Any] | None:
    """Return OpenRouter's unified ``reasoning`` object, or None.

    Anthropic and Gemini models get a token budget when the output limit is
    known; DeepSeek models are switched on or off instead of an effort.
    """
    if not caps.supports_reasoning_effort:
        return None
    payload: dict[str, Any] = {"effort": effort} if effort else {}
    name = model_name.lower().strip()
    budgeted = any(part in name for part in ("anthropic/", "claude", "gemini"))
    if payload and budgeted and limit is not None:
        budget = _openrouter_budget(limit, effort)
        if budget > 0:
            payload = {"max_tokens": budget}
    if payload and "deepseek" in name:
        level = str(payload.pop("effort", None) or "medium").lower().strip()
        payload["enabled"] = level != "none"
    return payload or None


def openrouter_options(
    model: ModelOptions, caps: Capabilities, limit: int | None
) -> dict[str, Any]:
    """Return the output limit, samplers and reasoning of an OpenRouter model."""
    options: dict[str, Any] = {}
    if limit is not None:
        options["max_tokens"] = limit
    options.update(sampler_values(caps, model.temperature, model.top_p, 0.0))
    reasoning = openrouter_reasoning(model.model, model.reasoning_effort, caps, limit)
    if reasoning is not None:
        options["extra_body"] = {"reasoning": reasoning}
    return options
