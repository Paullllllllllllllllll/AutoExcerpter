"""Build the request bodies of batch transcriptions and summaries.

The shared request builder does the work. :func:`build_plan` reads the
parts of a phase that every request shares: the system prompt with the
schema and, for a summary, the item's context; the schema the provider
enforces, if any; the request plan of
:func:`~autoexcerpter.llm.options.request_plan`; and the page detail.
:meth:`BodyPlan.body` turns each
:class:`~autoexcerpter.common.batch.BatchRequest` into the provider-native
body the shared backends wrap, through
:func:`~autoexcerpter.common.provider_requests.batch_body`: Responses API
bodies for OpenAI, Messages API parameters for Anthropic and
``InlinedRequest`` bodies for Google. A request with image data is a page
transcription; any other request sends its text, a page's transcription to
summarize.

The bodies carry what the synchronous calls send, from the same
:class:`~autoexcerpter.common.provider_requests.RequestPlan`: the system
prompt (with the prompt-cache breakpoint or parameter), the page image alone
as the user message of a transcription (with its OpenAI detail or Gemini
media resolution) and the page text alone as that of a summary, the output
limit, reasoning effort or thinking, verbosity and sampler values, and the
schema enforcement of :func:`~autoexcerpter.common.structured.prepare_request`
(OpenAI ``text.format``; Anthropic ``output_config.format`` on the schema the
synchronous request sends, unless the schema has more union parameters than
Anthropic compiles; Google ``response_json_schema``). A summary always asks
for JSON; the ``text`` format only changes how a failed answer is kept. The
OpenAI ``flex`` tier, which batches do not take, becomes ``auto``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from autoexcerpter.common.batch import BatchRequest, supports_batch
from autoexcerpter.common.provider_requests import Part, RequestPlan, batch_body
from autoexcerpter.common.settings_schema import PromptCaching
from autoexcerpter.common.structured import ResponseFormat, fit_response_format
from autoexcerpter.llm.options import request_plan, response_format_for
from autoexcerpter.llm.summary import summary_prompt, summary_request_format
from autoexcerpter.llm.transcription import page_detail, transcription_prompt
from autoexcerpter.llm.types import PhaseModel

__all__ = [
    "ROLE_SUMMARY",
    "ROLE_TRANSCRIPTION",
    "BodyPlan",
    "build_plan",
]

ROLE_TRANSCRIPTION: Final = "transcription"
ROLE_SUMMARY: Final = "summary"


@dataclass(frozen=True)
class BodyPlan:
    """What every request body of one phase's submission shares.

    *response_format* is the phase's format; *schema* is the schema spec
    the provider enforces, None under prompted JSON and plain text. *detail*
    is the page detail of a transcription's images, None for a summary.
    """

    role: str
    plan: RequestPlan
    system_prompt: str
    response_format: ResponseFormat
    schema: Mapping[str, Any] | None = None
    detail: str | None = None

    @property
    def provider(self) -> str:
        """The provider of the submission."""
        return self.plan.provider

    @property
    def model(self) -> str:
        """The model every body names."""
        return self.plan.model

    def body(self, request: BatchRequest) -> dict[str, Any]:
        """Return the provider-native body of *request*.

        Raises:
            ValueError: The request carries neither image data nor text.
        """
        if request.is_visual:
            part = Part(
                data=request.image_base64, mime=request.mime_type, detail=self.detail
            )
        elif request.text:
            part = Part(text=request.text)
        else:
            raise ValueError(f"Request {request.custom_id} has no text or image")
        return batch_body(self.plan, self.system_prompt, [part], self.schema)


def _enforced(
    provider: str, request_format: ResponseFormat, schema: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """Return the schema the provider enforces, as the synchronous call does."""
    if schema is None or request_format != "schema":
        return None
    if fit_response_format(provider, request_format, schema) != "schema":
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
    detail: str | None = None
    if role == ROLE_TRANSCRIPTION:
        schema, prompt = transcription_prompt(response_format == "text")
        request_format = response_format
        detail = page_detail(provider, phase.name, phase.options)
    elif role == ROLE_SUMMARY:
        schema, prompt = summary_prompt(context, transcription_format)
        request_format = summary_request_format(response_format)
    else:
        raise ValueError(f"Unknown batch role {role!r}")
    return BodyPlan(
        role=role,
        plan=request_plan(
            provider,
            phase.name,
            phase.options,
            phase.service_tier,
            caching if caching is not None else PromptCaching(),
        ),
        system_prompt=prompt,
        response_format=response_format,
        schema=_enforced(provider, request_format, schema),
        detail=detail,
    )
