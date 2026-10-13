"""Request bodies of batch transcriptions and summaries, one builder per provider.

:func:`build_plan` reads the parts of a phase that every request shares: the
system prompt with the schema and, for a summary, the item's context; the
schema the provider enforces, if any; the request kwargs of
:func:`~autoexcerpter.llm.options.build_invoke_kwargs`; and the OpenAI image
detail. :meth:`BodyPlan.body` turns each
:class:`~autoexcerpter.common.batch.BatchRequest` into the provider-native
body the shared backends wrap: Responses API bodies for OpenAI, Messages API
parameters for Anthropic and ``InlinedRequest`` bodies for Google. A request
with image data is a page transcription; any other request sends its text,
a page's transcription to summarize.

The bodies carry what the synchronous calls send: the same system prompt
(with the Anthropic cache breakpoint), the page image alone as the user
message of a transcription and the page text alone as that of a summary, the
same output limit, reasoning, verbosity and sampler values, and the schema
enforcement of :func:`~autoexcerpter.common.structured.prepare_request`
(OpenAI ``text.format``; Anthropic ``output_config.format`` on the schema the
synchronous request sends, unless the schema has more union parameters than
Anthropic compiles; Google ``response_json_schema``, left out while thinking
is configured). A summary always asks for JSON; the ``text`` format only
changes how a failed answer is kept. The OpenAI ``flex`` tier, which batches
do not take, becomes ``auto``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from autoexcerpter.common.batch import BatchRequest, supports_batch
from autoexcerpter.common.request_options import batch_service_tier
from autoexcerpter.common.settings_schema import PromptCaching
from autoexcerpter.common.structured import (
    ResponseFormat,
    anthropic_output_schema,
    build_text_format,
    fit_response_format,
    schema_body,
)
from autoexcerpter.llm.image_detail import resolve_request_detail
from autoexcerpter.llm.options import build_invoke_kwargs, response_format_for
from autoexcerpter.llm.prompts import cache_breakpoint, openai_cache_options
from autoexcerpter.llm.summary import summary_prompt, summary_request_format
from autoexcerpter.llm.transcription import transcription_prompt
from autoexcerpter.llm.types import PhaseModel

__all__ = [
    "ROLE_SUMMARY",
    "ROLE_TRANSCRIPTION",
    "BodyPlan",
    "build_plan",
]

logger = logging.getLogger(__name__)

ROLE_TRANSCRIPTION: Final = "transcription"
ROLE_SUMMARY: Final = "summary"

_ANTHROPIC_FALLBACK_LIMIT = 4096
_SAMPLERS = ("temperature", "top_p")


@dataclass(frozen=True)
class BodyPlan:
    """What every request body of one phase's submission shares.

    *response_format* is the phase's format; *schema* is the schema spec
    the provider enforces, None under prompted JSON and plain text. *kwargs*
    are the request kwargs of the synchronous call; *detail* the OpenAI
    image detail of a transcription. *cache_control* is the Anthropic
    system prompt's cache breakpoint (None: none); *prompt_cache* the
    OpenAI prompt-cache parameter, as the synchronous call sends them.
    """

    role: str
    provider: str
    model: str
    system_prompt: str
    response_format: ResponseFormat
    schema: Mapping[str, Any] | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    detail: str | None = None
    cache_control: Mapping[str, Any] | None = field(
        default_factory=lambda: {"type": "ephemeral"}
    )
    prompt_cache: Mapping[str, Any] = field(default_factory=dict)

    def body(self, request: BatchRequest) -> dict[str, Any]:
        """Return the provider-native body of *request*.

        Raises:
            ValueError: The request carries neither image data nor text.
        """
        if not request.is_visual and not request.text:
            raise ValueError(f"Request {request.custom_id} has no text or image")
        if self.provider == "openai":
            return self._openai(request)
        if self.provider == "anthropic":
            return self._anthropic(request)
        return self._google(request)

    def _samplers(self) -> dict[str, Any]:
        return {name: self.kwargs[name] for name in _SAMPLERS if name in self.kwargs}

    def _openai(self, request: BatchRequest) -> dict[str, Any]:
        content: list[dict[str, Any]]
        if request.is_visual:
            image: dict[str, Any] = {
                "type": "input_image",
                "image_url": f"data:{request.mime_type};base64,{request.image_base64}",
            }
            if self.detail is not None:
                image["detail"] = self.detail
            content = [image]
        else:
            content = [{"type": "input_text", "text": request.text}]
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {"role": "user", "content": content},
            ],
        }
        kwargs = self.kwargs
        if "max_output_tokens" in kwargs:
            body["max_output_tokens"] = kwargs["max_output_tokens"]
        tier = batch_service_tier(kwargs.get("service_tier"))
        if tier is not None:
            body["service_tier"] = tier
        if "reasoning" in kwargs:
            body["reasoning"] = dict(kwargs["reasoning"])
        text = dict(kwargs.get("text") or {})
        if self.schema is not None:
            text["format"] = build_text_format(self.schema)
        if text:
            body["text"] = text
        body.update(self._samplers())
        body.update(self.prompt_cache)
        return body

    def _anthropic(self, request: BatchRequest) -> dict[str, Any]:
        content: Any
        if request.is_visual:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": request.mime_type,
                        "data": request.image_base64,
                    },
                }
            ]
        else:
            content = request.text
        kwargs = self.kwargs
        system: dict[str, Any] = {"type": "text", "text": self.system_prompt}
        if self.cache_control is not None:
            system["cache_control"] = dict(self.cache_control)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", _ANTHROPIC_FALLBACK_LIMIT),
            "system": [system],
            "messages": [{"role": "user", "content": content}],
        }
        if "thinking" in kwargs:
            params["thinking"] = dict(kwargs["thinking"])
        params.update(self._samplers())
        output_config = dict(kwargs.get("output_config") or {})
        if self.schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic_output_schema(self.schema),
            }
        if output_config:
            params["output_config"] = output_config
        return params

    def _google(self, request: BatchRequest) -> dict[str, Any]:
        parts: list[dict[str, Any]]
        if request.is_visual:
            parts = [
                {
                    "inline_data": {
                        "mime_type": request.mime_type,
                        "data": request.image_base64,
                    }
                }
            ]
        else:
            parts = [{"text": request.text}]
        kwargs = self.kwargs
        config: dict[str, Any] = {}
        if "max_output_tokens" in kwargs:
            config["max_output_tokens"] = kwargs["max_output_tokens"]
        config.update(self._samplers())
        if "thinking_config" in kwargs:
            config["thinking_config"] = dict(kwargs["thinking_config"])
        config["system_instruction"] = self.system_prompt
        body = schema_body(self.schema)
        if body is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = body
        return {
            "model": self.model,
            "contents": [{"role": "user", "parts": parts}],
            "config": config,
        }


def _enforced(
    provider: str,
    request_format: ResponseFormat,
    schema: Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the schema the provider enforces, as the synchronous call does."""
    if schema is None or request_format != "schema":
        return None
    if fit_response_format(provider, request_format, schema) != "schema":
        return None
    if provider == "google" and "thinking_config" in kwargs:
        return None
    return schema


def build_plan(
    phase: PhaseModel,
    role: str,
    *,
    context: str | None = None,
    transcription_format: str = "schema",
    caching: PromptCaching | None = None,
) -> BodyPlan:
    """Return the body plan of *phase*'s requests in *role*.

    *role* is ``transcription`` or ``summary``. A summary's prompt carries
    *context*, and *transcription_format* (the transcription's response
    format) selects its plain-text variant. *caching* is the prompt-caching
    setting (by default the code values).

    Raises:
        ValueError: The provider has no batch API, or *role* is unknown.
        OSError: A prompt or schema file cannot be read.
    """
    provider = phase.provider
    if phase.endpoint is not None or not supports_batch(provider):
        raise ValueError(f"Provider '{provider}' has no batch API")
    response_format = response_format_for(phase.response_format, provider, phase.name)
    kwargs = build_invoke_kwargs(
        provider, phase.name, phase.options, phase.service_tier
    )
    detail: str | None = None
    if role == ROLE_TRANSCRIPTION:
        schema, prompt = transcription_prompt(response_format == "text")
        request_format = response_format
        detail = resolve_request_detail(
            dict(phase.options), provider, phase.name, logger
        )
    elif role == ROLE_SUMMARY:
        schema, prompt = summary_prompt(context, transcription_format)
        request_format = summary_request_format(response_format)
    else:
        raise ValueError(f"Unknown batch role {role!r}")
    return BodyPlan(
        role=role,
        provider=provider,
        model=phase.name,
        system_prompt=prompt,
        response_format=response_format,
        schema=_enforced(provider, request_format, schema, kwargs),
        kwargs=kwargs,
        detail=detail,
        cache_control=cache_breakpoint(provider, phase.name, caching),
        prompt_cache=openai_cache_options(provider, phase.name, caching),
    )
