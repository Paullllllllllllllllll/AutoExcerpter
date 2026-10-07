"""The explicit settings and resume state handed to one item's processor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from autoexcerpter.imaging.settings import request_detail_option
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.rendering.citations import OpenAlexClient
from autoexcerpter.settings import Settings
from autoexcerpter.spec import ImageSpec, ModelSpec, RunSpec


@dataclass(frozen=True)
class ItemJob:
    """Run settings shared by every item of one run.

    *context* is the run's summary context: None uses the item's sidecars,
    ``"none"`` disables context, and text or a path to a text file overrides
    the sidecars. *fallback_context* (text or a path) applies to items without
    a sidecar. The plan resolves each item's prompt context from both and the
    item's sidecars. *cleanup* deletes the working
    directory of an item that finished complete outside resume mode.
    *openalex* is the run's OpenAlex client, shared by every item so that its
    cache and budget state carry across items; None when lookups are off.
    Citations are looked up only when *openalex_enabled* is set and
    *openalex* holds a client, at most *openalex_max_requests* per item.
    *transcription_format* is the transcription file's extension;
    *output_sqlite* writes the output folder's database.
    """

    transcription: PhaseModel
    summary: PhaseModel
    summarize: bool = True
    output_docx: bool = True
    output_markdown: bool = True
    output_sqlite: bool = True
    transcription_format: str = "md"
    concurrency: int = 1
    timeout: float | None = None
    context: str | None = None
    fallback_context: str | None = None
    cleanup: bool = False
    openalex_enabled: bool = True
    openalex_max_requests: int = 300
    images: ImageSpec | None = None
    openalex: OpenAlexClient | None = None


@dataclass(frozen=True)
class ItemResume:
    """Working-log state of an item found by the resume check.

    The parsed lists and header are reused instead of re-reading the logs;
    a field left None falls back to a disk read.
    """

    completed_page_indices: frozenset[int] = field(default_factory=frozenset)
    transcription_results: list[dict[str, Any]] | None = None
    summary_results: list[dict[str, Any]] | None = None
    log_header: dict[str, Any] | None = None


def _seconds(value: float) -> float:
    """Return *value* as an int when it is whole (``900`` rather than 900.0)."""
    return int(value) if float(value).is_integer() else value


def _model_options(
    model: ModelSpec, phase: str, images: ImageSpec | None
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if model.max_output_tokens is not None:
        options["max_output_tokens"] = model.max_output_tokens
    options["reasoning"] = {"effort": model.reasoning_effort}
    options["text"] = {"verbosity": model.verbosity}
    options["temperature"] = model.temperature
    if model.top_p is not None:
        options["top_p"] = model.top_p
    if phase == "transcription":
        detail = request_detail_option(model.provider, images)
        if detail is not None:
            options["image_size"] = detail
    return options


def phase_model(
    model: ModelSpec, settings: Settings, phase: str, images: ImageSpec | None = None
) -> PhaseModel:
    """Build the model of one phase from its spec and the machine settings.

    A named endpoint selects the base URL, the key variable and the declared
    image support; an unset response format takes schema when the endpoint
    enforces schemas, else json. For other providers an unset format is
    decided by the model's capabilities when the client is built.

    Raises:
        SettingsError: The named endpoint is not defined.
    """
    base_url: str | None = None
    supports_vision: bool | None = None
    response_format = model.response_format
    if model.endpoint is not None:
        endpoint = settings.endpoint(model.endpoint)
        base_url = endpoint.base_url
        supports_vision = endpoint.supports_vision
        response_format = response_format or (
            "schema" if endpoint.supports_schema else "json"
        )
    return PhaseModel(
        name=model.model,
        provider=model.provider,
        response_format=response_format,
        options=_model_options(model, phase, images),
        service_tier=model.service_tier,
        api_key_env=settings.api_key_env(model.provider, model.endpoint),
        base_url=base_url,
        endpoint=model.endpoint,
        supports_vision=supports_vision,
    )


def job_from_spec(spec: RunSpec, settings: Settings) -> ItemJob:
    """Build the item job of a run from its spec and the machine settings.

    The OpenAlex key is read from the environment variable the settings
    name. With OpenAlex lookups on, the job holds the run's one client;
    building it reads nothing, since the cache loads on first use.

    Raises:
        SettingsError: A named endpoint is not defined.
    """
    key_env = settings.openalex.api_key_env
    openalex_key = os.environ.get(key_env, "").strip() if key_env else ""
    formats = spec.output.formats
    openalex = (
        OpenAlexClient(
            settings.openalex.email or None,
            openalex_key or None,
            state_dir=settings.state_directory(),
        )
        if spec.citations.openalex
        else None
    )
    return ItemJob(
        transcription=phase_model(
            spec.transcription_model, settings, "transcription", spec.images
        ),
        summary=phase_model(spec.summary_model, settings, "summary"),
        summarize=spec.summary.enabled,
        output_docx="summary-docx" in formats,
        output_markdown="summary-md" in formats,
        output_sqlite="sqlite" in formats,
        transcription_format=spec.output.transcription_format,
        concurrency=spec.run.concurrency,
        timeout=_seconds(settings.timeouts.request),
        context=spec.summary.context,
        fallback_context=spec.summary.fallback_context,
        cleanup=not spec.output.keep_working_files,
        openalex_enabled=spec.citations.openalex,
        openalex_max_requests=settings.openalex.max_requests,
        images=spec.images,
        openalex=openalex,
    )


__all__ = [
    "ItemJob",
    "ItemResume",
    "PhaseModel",
    "job_from_spec",
    "phase_model",
]
