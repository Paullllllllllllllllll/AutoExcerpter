"""The explicit settings handed to one item's processor."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from autoexcerpter.common.selection import PageRange, parse_page_range
from autoexcerpter.imaging.settings import (
    request_detail_option,
    resolve_image_settings,
)
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
    item's sidecars. *openalex* is the run's OpenAlex client, shared by every
    item so that its cache and budget state carry across items; None when
    lookups are off. Citations are looked up only when *openalex_enabled* is
    set and *openalex* holds a client, at most *openalex_max_requests* per
    item. *transcription_format* is the transcription file's extension.
    *original_max_side_px* is the ``images`` setting of that name.
    *start_delay* is the pause in seconds between starting two page
    requests (``concurrency.delay_between_tasks``). *pages* is the run's
    ``--pages`` range; None works every page (see :meth:`selected_pages`).
    """

    transcription: PhaseModel
    summary: PhaseModel
    summarize: bool = True
    output_docx: bool = True
    output_markdown: bool = True
    transcription_format: str = "md"
    concurrency: int = 1
    start_delay: float = 0.0
    timeout: float | None = None
    context: str | None = None
    fallback_context: str | None = None
    openalex_enabled: bool = True
    openalex_max_requests: int = 300
    images: ImageSpec | None = None
    openalex: OpenAlexClient | None = None
    original_max_side_px: int | None = None
    pages: PageRange | None = None

    def selected_pages(self, count: int) -> list[int]:
        """Return the 0-based pages the run works in an item of *count* pages.

        These are every page, or those *pages* selects, in source order.
        """
        if self.pages is None:
            return list(range(count))
        return self.pages.resolve(count)


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
        if model.provider in ("google", "openrouter"):
            options["page_detail"] = _page_detail(model, images)
    return options


def _page_detail(model: ModelSpec, images: ImageSpec | None) -> str | None:
    """Return the detail a page's image settings resolve, or None if they do not.

    A Gemini page's media resolution follows it; an OpenRouter page sends it
    where the model takes a detail.
    """
    try:
        cfg, _model_type, _section = resolve_image_settings(
            model.provider, model.model, images
        )
    except ValueError:
        return None
    value = cfg.get("resolved_detail")
    return str(value) if value else None


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
    formats = spec.output.summary_formats
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
        transcription_format=spec.output.transcription_format,
        concurrency=spec.run.concurrency,
        start_delay=settings.concurrency.delay_between_tasks,
        timeout=_seconds(settings.timeouts.request),
        context=spec.summary.context,
        fallback_context=spec.summary.fallback_context,
        openalex_enabled=spec.citations.openalex,
        openalex_max_requests=settings.openalex.max_requests,
        images=spec.images,
        openalex=openalex,
        original_max_side_px=settings.images.original_max_side_px,
        pages=parse_page_range(spec.input.pages) if spec.input.pages else None,
    )


__all__ = [
    "ItemJob",
    "PhaseModel",
    "job_from_spec",
    "phase_model",
]
