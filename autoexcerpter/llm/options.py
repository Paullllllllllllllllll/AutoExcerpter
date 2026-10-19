"""Request plan of a phase and its response format.

:func:`request_plan` turns a phase's options (output limit, reasoning
effort, verbosity, temperature, top_p) and service tier into the shared
:class:`~autoexcerpter.common.provider_requests.RequestPlan`, whose
parameters (:func:`build_invoke_kwargs`) every synchronous call sends as
invoke kwargs and every batch body carries. The shared builder decides,
from the capability table, the OpenAI service tier (``flex`` only for the
models listed for it), the reasoning effort or thinking level a model
documents, Anthropic's adaptive thinking and effort, Gemini's thinking level
or budget, OpenRouter's reasoning object and whether a model takes a
verbosity. An unset output limit takes the model's limit on OpenAI,
Anthropic and Google and the documented limit of an OpenRouter model the
capability facts list (:data:`LIMITS`). :func:`response_format_for` decides
between schema, json and text.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from autoexcerpter.common.capabilities import UNKNOWN, Capabilities
from autoexcerpter.common.provider_requests import RequestPlan, request_params
from autoexcerpter.common.request_options import OutputLimits, RequestRules
from autoexcerpter.common.settings_schema import PromptCaching
from autoexcerpter.common.structured import RESPONSE_FORMATS, ResponseFormat
from autoexcerpter.llm.capabilities import TABLE, detect_capabilities

__all__ = [
    "DEFAULT_SERVICE_TIER",
    "LIMITS",
    "RULES",
    "build_invoke_kwargs",
    "request_plan",
    "request_service_tier",
    "response_format_for",
]

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_TIER = "default"

LIMITS = OutputLimits(
    conservative=int(
        TABLE.unknown.get(
            "max_output_tokens",
            Capabilities(provider_name=UNKNOWN, model_name="").max_output_tokens,
        )
    )
)
"""A set output limit is sent as given, clamped only to the documented limit
of an OpenRouter model the capability facts list; an unset one takes a known
family's limit on OpenAI, Anthropic and Google (the conservative limit for
an unmatched Anthropic model, since langchain-anthropic would otherwise send
1024), such a documented limit on OpenRouter and none elsewhere."""

RULES = RequestRules(limits=LIMITS)
"""This tool's request rules: :data:`LIMITS`, and every OpenAI model takes
the service tier."""


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

    The request builder sends it where the model takes it.
    """
    if provider != "openai":
        return "auto"
    return service_tier or DEFAULT_SERVICE_TIER


def _effort(options: Mapping[str, Any]) -> str | None:
    reasoning = options.get("reasoning")
    if isinstance(reasoning, Mapping) and reasoning.get("effort"):
        return str(reasoning["effort"])
    return None


def _verbosity(options: Mapping[str, Any]) -> str | None:
    text = options.get("text")
    if isinstance(text, Mapping) and text.get("verbosity"):
        return str(text["verbosity"])
    return None


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def request_plan(
    provider: str,
    model_name: str,
    options: Mapping[str, Any],
    service_tier: str | None = None,
    caching: PromptCaching | None = None,
) -> RequestPlan:
    """Return the request plan of one phase; *caching* None sends no
    prompt-cache parameter or breakpoint."""
    return RequestPlan.resolve(
        provider,
        model_name,
        detect_capabilities(model_name),
        RULES,
        requested_limit=options.get("max_output_tokens"),
        effort=_effort(options),
        temperature=_number(options.get("temperature")),
        top_p=_number(options.get("top_p")),
        verbosity=_verbosity(options),
        service_tier=request_service_tier(provider, service_tier),
        caching=caching,
    )


def build_invoke_kwargs(
    provider: str,
    model_name: str,
    options: Mapping[str, Any],
    service_tier: str | None = None,
    caching: PromptCaching | None = None,
) -> dict[str, Any]:
    """Return the invoke kwargs of one request: the parameters of the shared
    request builder (:func:`~autoexcerpter.common.provider_requests.
    request_params`) for the phase's plan."""
    return request_params(
        request_plan(provider, model_name, options, service_tier, caching)
    )
