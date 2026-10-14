"""The run specification and the option table that builds it.

``OPTIONS`` is the single source of run options: the CLI parser, the settings
``defaults`` block and the equivalent command line all derive from it. Model
options exist three times: the unprefixed flag sets both phases, and the
``--transcription-*`` and ``--summary-*`` flags override one phase. Image
options left unset take the provider's recommended value
(:func:`recommended_images`). :func:`model_suggestions` lists model names
per provider for the front ends; any other name is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from autoexcerpter.common.image_presets import (
    DetailCheck,
    DetailRules,
    ImagePresets,
    anthropic_preset,
    custom_preset,
    google_preset,
    openai_preset,
)
from autoexcerpter.common.options import (
    Kind,
    Option,
    OptionTable,
    Resolved,
    Source,
)
from autoexcerpter.common.polling import DEFAULT_POLL_MINUTES, wait_options
from autoexcerpter.common.selection import parse_page_range
from autoexcerpter.common.standard_options import (
    DRY_RUN_HELP,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    JSON_HELP,
    MODEL_FIELDS,
    REASONING_EFFORTS,
    RESPONSE_FORMATS,
    SERVICE_TIERS,
    VERBOSITIES,
    Dpi,
    abandon_option,
    adopt_option,
    batch_cancel_options,
    batch_ids_option,
    batch_option,
    check_input_option,
    concurrency_option,
    dry_run_option,
    force_option,
    image_options,
    json_option,
    model_options,
    nonempty_text,
    output_options,
    render_options,
    settings_option,
)
from autoexcerpter.providers import (
    PROVIDERS,
    implied_provider,
    infer_provider,
    split_provider_prefix,
)

__all__ = [
    "BATCH_CANCEL_OPTIONS",
    "BATCH_CHECK_OPTIONS",
    "CONTEXT_AUTO",
    "CONTEXT_NONE",
    "DEFAULT_FORMATS",
    "DEFAULT_MODEL",
    "EXTENSION_COMMANDS",
    "IMAGE_DETAILS",
    "IMAGE_FORMATS",
    "IMAGE_PRESETS",
    "MIGRATE_OPTIONS",
    "MODEL_SUGGESTIONS",
    "OPTIONS",
    "OUTPUT_FORMATS",
    "PHASES",
    "RENDER_FORMATS",
    "RENDER_OPTIONS",
    "PROVIDERS",
    "REASONING_EFFORTS",
    "RESPONSE_FORMATS",
    "SERVICE_TIERS",
    "SUMMARY_FORMATS",
    "TRANSCRIPTION_FORMATS",
    "TRANSCRIPTION_FORMAT_ALIAS",
    "VERBOSITIES",
    "BatchCancelSpec",
    "BatchCheckSpec",
    "CitationSpec",
    "CommandSpec",
    "Dpi",
    "ImageSpec",
    "InputSpec",
    "MigrateSpec",
    "ModelSpec",
    "OutputSpec",
    "RenderSpec",
    "Resolution",
    "RunControl",
    "RunSpec",
    "SpecError",
    "SummarySpec",
    "check_formats",
    "effective_spec",
    "fold_transcription_format",
    "model_suggestions",
    "parse",
    "recommended_images",
    "resolve_batch_cancel",
    "resolve_batch_check",
    "resolve_migrate",
    "resolve_render",
    "resolve_spec",
    "spec_from_effective",
    "to_argv",
    "transcription_format_of",
    "with_transcription_format",
]

TRANSCRIPTION_FORMATS = ("md", "txt")
RENDER_FORMATS = ("transcription-md", "transcription-txt", "summary-md", "summary-docx")
OUTPUT_FORMATS = RENDER_FORMATS
SUMMARY_FORMATS = ("summary-md", "summary-docx")
DEFAULT_FORMATS = ("transcription-md", *SUMMARY_FORMATS)
TRANSCRIPTION_PREFIX = "transcription-"
PHASES = ("transcription", "summary")

EXTENSION_COMMANDS: frozenset[tuple[str, ...]] = frozenset(
    {("run",), ("batch", "check"), ("batch", "cancel"), ("ui",)}
)
"""The commands that call providers and so take the extension's arguments
(``ext.register_args``); ``render`` and ``migrate`` do not."""

TRANSCRIPTION_FORMAT_ALIAS = "transcription_format"
"""The deprecated ``--transcription-format`` (and settings key of that name):
``md`` or ``txt``, folded into ``--formats`` during v3.x."""

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_VERBOSITY = {"transcription": "high", "summary": "low"}
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SERVICE_TIER = "flex"
DEFAULT_CONCURRENCY = 16

CONTEXT_AUTO = "auto"
CONTEXT_NONE = "none"

MODEL_SUGGESTIONS: Mapping[str, tuple[str, ...]] = {
    "openai": (
        DEFAULT_MODEL,
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.5",
    ),
    "anthropic": ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
    "google": ("gemini-3.7-flash", "gemini-3.1-pro", "gemini-3.5-flash"),
    "openrouter": (
        "openai/gpt-5.6-luna",
        "anthropic/claude-sonnet-5-5",
        "google/gemini-3.1-pro",
    ),
}


class SpecError(ValueError):
    """The options do not form a valid run specification."""


@dataclass(frozen=True)
class InputSpec:
    """The input path, the selection of discovered items and of their pages.

    ``select`` is a selection expression, ``files`` names items exactly and
    ``all`` takes every item; at most one of them is set. ``pages`` is a page
    range (:func:`~autoexcerpter.common.selection.parse_page_range`) applied
    to every selected item; None takes every page.
    """

    path: Path
    select: str | None
    all: bool
    files: tuple[str, ...] = ()
    pages: str | None = None


@dataclass(frozen=True)
class OutputSpec:
    """Where and what to write; ``directory`` None means beside the input.

    ``formats`` names the rendered files (:data:`OUTPUT_FORMATS`): exactly
    one transcription format and the summary formats. The transcription
    file and the database are always written.
    """

    directory: Path | None
    formats: tuple[str, ...]

    @property
    def transcription_format(self) -> str:
        """The transcription file's extension, ``md`` or ``txt``."""
        return transcription_format_of(self.formats) or "md"

    @property
    def summary_formats(self) -> tuple[str, ...]:
        """The summary formats among ``formats``."""
        return tuple(name for name in self.formats if name in SUMMARY_FORMATS)


def transcription_format_of(formats: Sequence[str]) -> str | None:
    """Return ``md`` or ``txt`` from the transcription format in *formats*."""
    for name in formats:
        if name.startswith(TRANSCRIPTION_PREFIX):
            return name.removeprefix(TRANSCRIPTION_PREFIX)
    return None


def with_transcription_format(
    formats: Sequence[str], extension: str | None
) -> tuple[str, ...]:
    """Return *formats* with exactly one transcription format.

    *extension* (``md`` or ``txt``) replaces the transcription format of
    *formats*; None keeps it, and a list without one takes ``md``.

    Raises:
        SpecError: *formats* names both transcription formats.
    """
    given = [name for name in formats if name.startswith(TRANSCRIPTION_PREFIX)]
    if len(given) > 1:
        raise SpecError(
            "--formats takes one transcription format: transcription-md or "
            "transcription-txt"
        )
    if extension is None:
        extension = transcription_format_of(given) or "md"
    if extension not in TRANSCRIPTION_FORMATS:
        raise SpecError(
            f"{TRANSCRIPTION_FORMAT_ALIAS}: {extension!r} is not one of md, txt"
        )
    wanted = {TRANSCRIPTION_PREFIX + extension} | {
        name for name in formats if name in SUMMARY_FORMATS
    }
    return tuple(name for name in OUTPUT_FORMATS if name in wanted)


@dataclass(frozen=True)
class SummarySpec:
    """Whether to summarize, and the summary context.

    ``context`` None uses the item's file or folder sidecar, :data:`CONTEXT_NONE`
    disables context, and text or a path to a text file overrides the
    sidecars. ``fallback_context`` (text or a path) applies to items without
    a sidecar.
    """

    enabled: bool
    context: str | None
    fallback_context: str | None


@dataclass(frozen=True)
class CitationSpec:
    """Citation enrichment."""

    openalex: bool


@dataclass(frozen=True)
class ModelSpec:
    """The model of one phase.

    ``provider`` is a provider name once resolved, and ``model`` carries no
    ``provider:`` prefix. ``endpoint`` names a custom endpoint from the
    settings. ``max_output_tokens`` None takes the model's output limit;
    ``top_p`` None is not sent. ``response_format`` None means schema when the
    model supports it, else json.
    """

    provider: str
    model: str
    endpoint: str | None
    reasoning_effort: str
    verbosity: str
    max_output_tokens: int | None
    temperature: float
    top_p: float | None
    service_tier: str
    response_format: str | None


@dataclass(frozen=True)
class ImageSpec:
    """Page image settings; None takes the provider's recommended value."""

    dpi: Dpi | None = None
    detail: str | None = None
    format: str | None = None
    jpeg_quality: int | None = None
    grayscale: bool | None = None

    def over(self, base: ImageSpec) -> ImageSpec:
        """Return these values with unset fields taken from ``base``."""
        return ImageSpec(
            *(
                getattr(self, f.name)
                if getattr(self, f.name) is not None
                else getattr(base, f.name)
                for f in fields(self)
            )
        )


@dataclass(frozen=True)
class RunControl:
    """Resume, concurrency, batch and agent-contract switches.

    ``batch`` submits the model calls to the providers' batch APIs instead
    of calling the models directly.
    """

    force: bool
    retranscribe: bool
    concurrency: int
    dry_run: bool
    json: bool
    batch: bool = False


@dataclass(frozen=True)
class RunSpec:
    """Everything one run needs besides the machine settings."""

    input: InputSpec
    output: OutputSpec
    summary: SummarySpec
    citations: CitationSpec
    transcription_model: ModelSpec
    summary_model: ModelSpec
    images: ImageSpec
    run: RunControl

    def model(self, phase: str) -> ModelSpec:
        """Return the model of ``phase`` (transcription or summary)."""
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        spec: ModelSpec = getattr(self, f"{phase}_model")
        return spec


IMAGE_PRESETS = ImagePresets(
    families={
        "openai": openai_preset(grayscale=False),
        "anthropic": anthropic_preset(grayscale=False),
        "google": google_preset(grayscale=False),
        "custom": custom_preset(jpeg_quality=95, grayscale=False),
    },
    detail=DetailRules(
        filter_overrides=False,
        openrouter_openai_detail=False,
        checks={
            "openai": DetailCheck(
                kept=frozenset({"low", "high", "auto"}),
                original=True,
                unsupported_original="high",
                fallback="auto",
                without_detail="auto",
                source="request",
            ),
            "openrouter": DetailCheck(
                kept=frozenset({"low", "high", "auto", "original"}),
                model_types=frozenset({"openai"}),
            ),
        },
        other=DetailCheck(
            kept=frozenset({"low", "high", "auto"}),
            model_types=frozenset({"openai", "custom"}),
        ),
    ),
    resampling="bilinear",
    version=2,
)
"""The image presets of each model family: the values the wizard recommends
and the request sections of ``imaging.settings``. Version 2 sends every
family's pages in colour at JPEG quality 95 (version 1: grayscale, and
quality 85 for custom endpoints); an item stored under version 1 resumes
with the values of its stored record (``run.plan``). A run's detail is written
into the section as given. OpenAI pages follow the request's ``image_size``
option: low, high and auto stay, original stays where the model takes it and
becomes high elsewhere, and any other detail, or a model without image
detail, gives auto. OpenRouter models of the OpenAI family keep low, high,
auto and original, other providers' OpenAI and custom sections low, high and
auto; anything else becomes high."""

_RECOMMENDED_IMAGES: Mapping[str, ImageSpec] = {
    family: ImageSpec(**asdict(preset.recommended))
    for family, preset in IMAGE_PRESETS.families.items()
}


def recommended_images(family: str) -> ImageSpec:
    """Return the recommended image settings of a model family.

    ``family`` is openai, anthropic, google or custom; OpenRouter models use
    the family of the underlying model, unknown families the custom values.
    """
    return _RECOMMENDED_IMAGES.get(family, _RECOMMENDED_IMAGES["custom"])


def model_suggestions(provider: str | None = None) -> tuple[str, ...]:
    """Return suggested model names of *provider*, or of every provider.

    Suggestions are not enforced; a custom endpoint has none.
    """
    if provider is None:
        names = (name for names in MODEL_SUGGESTIONS.values() for name in names)
        return tuple(dict.fromkeys(names))
    return MODEL_SUGGESTIONS.get(provider, ())


def _context(value: str) -> str | None:
    keyword = value.strip().lower()
    if keyword == CONTEXT_AUTO:
        return None
    if keyword == CONTEXT_NONE:
        return CONTEXT_NONE
    return nonempty_text(value)


def _pages(value: str) -> str:
    parse_page_range(value)
    return value.strip()


def _fallback_context(value: str) -> str | None:
    if value.strip().lower() == CONTEXT_NONE:
        return None
    return nonempty_text(value)


def _model_options() -> list[Option]:
    """Return the unprefixed model options and the per-phase copies."""
    return model_options(
        providers=PROVIDERS,
        suggestions=model_suggestions,
        defaults={
            "model": DEFAULT_MODEL,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "temperature": DEFAULT_TEMPERATURE,
            "service_tier": DEFAULT_SERVICE_TIER,
        },
        response_formats=RESPONSE_FORMATS,
        phases=PHASES,
        phase_defaults={
            phase: {"verbosity": DEFAULT_VERBOSITY[phase]} for phase in PHASES
        },
    )


def _build_table() -> OptionTable:
    options = [
        Option(
            "input",
            "--input",
            "PDF file, image folder, or folder of PDFs and image folders",
            group="input",
            type=Path,
            metavar="PATH",
            target="input.path",
        ),
        Option(
            "select",
            "--select",
            "select items by number (1,3), range (1-5) or name search",
            group="input",
            type=nonempty_text,
            metavar="EXPR",
            exclusive="selection",
            target="input.select",
        ),
        Option(
            "files",
            "--files",
            "comma-separated names of items inside the input folder",
            group="input",
            kind=Kind.LIST,
            type=nonempty_text,
            metavar="LIST",
            default=(),
            exclusive="selection",
            target="input.files",
            settable=False,
        ),
        Option(
            "all",
            "--all",
            "process every discovered item",
            group="input",
            kind=Kind.FLAG,
            default=False,
            exclusive="selection",
            target="input.all",
        ),
        Option(
            "pages",
            "--pages",
            "pages or images: 5, first:5, last:5, 3-7, 3-, -7 or 1,3,5-8",
            group="input",
            type=_pages,
            metavar="RANGE",
            default_help="all",
            target="input.pages",
            settable=False,
        ),
        *output_options(),
        Option(
            "formats",
            "--formats",
            (
                "comma-separated output formats, at most one transcription "
                "format (transcription-md when none is given)"
            ),
            group="output",
            kind=Kind.LIST,
            choices=OUTPUT_FORMATS,
            default=DEFAULT_FORMATS,
            metavar="LIST",
            target="output.formats",
        ),
        Option(
            "summarize",
            "--summarize",
            "summarize after transcribing",
            group="summary",
            kind=Kind.SWITCH,
            default=True,
            target="summary.enabled",
        ),
        Option(
            "context",
            "--context",
            (
                "summary context: topics as text or a path to a text file "
                "(overrides sidecars), auto (the item's file or folder sidecar) "
                "or none (no context)"
            ),
            group="summary",
            type=_context,
            metavar="TEXT|PATH|auto|none",
            default_help=CONTEXT_AUTO,
            target="summary.context",
        ),
        Option(
            "fallback_context",
            "--fallback-context",
            (
                "summary context for items without a sidecar under --context "
                "auto: topics as text, a path to a text file, or none"
            ),
            group="summary",
            type=_fallback_context,
            metavar="TEXT|PATH|none",
            default_help=CONTEXT_NONE,
            target="summary.fallback_context",
        ),
        Option(
            "openalex",
            "--openalex",
            "enrich citations with OpenAlex metadata",
            group="citations",
            kind=Kind.SWITCH,
            default=True,
            target="citations.openalex",
        ),
        *_model_options(),
        *image_options(),
        force_option(
            "reprocess items whose outputs exist (resume is the default): every "
            "stored page of the item is reset, also outside --pages"
        ),
        Option(
            "retranscribe",
            "--retranscribe",
            (
                "transcribe again instead of reusing stored transcriptions; "
                "with --pages only those of the selected pages, under the "
                "stored image settings"
            ),
            group="run",
            kind=Kind.FLAG,
            default=False,
            target="run.retranscribe",
            settable=False,
        ),
        concurrency_option(DEFAULT_CONCURRENCY),
        batch_option("submit the pages and summaries to the providers' batch APIs"),
        dry_run_option("run.dry_run"),
        json_option("run.json"),
        settings_option(),
    ]
    titles = {
        "input": "input",
        "output": "output",
        "summary": "summary",
        "citations": "citations",
        "model": "model (both phases)",
        "transcription model": "transcription model",
        "summary model": "summary model",
        "images": "images",
        "run": "run control",
    }
    return OptionTable(options, group_titles=titles)


OPTIONS = _build_table()


RENDER_OPTIONS = OptionTable(
    [
        *render_options(
            input_help="output folder or autoexcerpter.sqlite to render",
            formats=RENDER_FORMATS,
            dry_run_help=DRY_RUN_HELP,
            json_help=JSON_HELP,
        ),
        settings_option(target="settings"),
    ],
    group_titles={"input": "input", "output": "output", "run": "run control"},
)


@dataclass(frozen=True)
class RenderSpec:
    """The options of ``render``; ``formats`` None renders the registered ones.

    ``settings`` names the settings file (its ``logs_dir``); render makes no
    model call.
    """

    input: Path
    formats: tuple[str, ...] | None = None
    force: bool = False
    dry_run: bool = False
    json: bool = False
    settings: Path | None = None


def resolve_render(flags: Mapping[str, Any]) -> RenderSpec:
    """Resolve render options; no option takes a settings default.

    Raises:
        SpecError: An unknown option or an empty ``--formats``.
    """
    unknown = set(flags) - set(RENDER_OPTIONS.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    values, _ = RENDER_OPTIONS.by_targets(RENDER_OPTIONS.resolve(flags, None))
    if values["formats"] == ():
        raise SpecError("--formats needs at least one format")
    return RenderSpec(**values)


MIGRATE_OPTIONS = OptionTable(
    [
        Option(
            "input",
            "--input",
            "output folder, working folder or autoexcerpter.sqlite to import",
            group="input",
            type=Path,
            metavar="PATH",
            target="input",
            settable=False,
        ),
        settings_option(target="settings"),
        dry_run_option("dry_run"),
        json_option("json"),
    ],
    group_titles={"input": "input", "run": "run control"},
)


@dataclass(frozen=True)
class MigrateSpec:
    """The options of ``migrate``."""

    input: Path
    settings: Path | None = None
    dry_run: bool = False
    json: bool = False


def resolve_migrate(flags: Mapping[str, Any]) -> MigrateSpec:
    """Resolve migrate options; no option takes a settings default.

    Raises:
        SpecError: An unknown option or no input path.
    """
    unknown = set(flags) - set(MIGRATE_OPTIONS.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    values, _ = MIGRATE_OPTIONS.by_targets(MIGRATE_OPTIONS.resolve(flags, None))
    if values["input"] is None:
        raise SpecError("an input path is required (--input PATH)")
    return MigrateSpec(**values)


_BATCH_TITLES = {"batch": "batches", "run": "run control"}

WAIT_HOURS = 98.0
"""Default ``--wait-hours``: a check waits for two batch rounds in sequence,
the transcriptions and then the summaries, each up to the longest provider
batch window (Gemini, 48 hours)."""

BATCH_CHECK_OPTIONS = OptionTable(
    [
        check_input_option(),
        batch_ids_option("comma-separated batch IDs to check; default every batch"),
        adopt_option(),
        abandon_option(
            "give up an unresolved submission (its upload is deleted) or a "
            "submitted one whose results are lost: its pages return to pending "
            "and its documents wait for a run"
        ),
        *wait_options(group="batch", default_hours=WAIT_HOURS),
        Option(
            "formats",
            "--formats",
            (
                "comma-separated formats of the finished documents, replacing "
                "those of the submitting run (whose transcription format stays "
                "when none is given)"
            ),
            group="output",
            kind=Kind.LIST,
            choices=OUTPUT_FORMATS,
            metavar="LIST",
            default_help="the formats of the submitting run",
            target="formats",
            settable=False,
        ),
        json_option("json"),
        settings_option(target="settings"),
    ],
    group_titles={**_BATCH_TITLES, "output": "output"},
)

BATCH_CANCEL_OPTIONS = OptionTable(
    batch_cancel_options(settings_target="settings"),
    group_titles=_BATCH_TITLES,
)


@dataclass(frozen=True)
class BatchCheckSpec:
    """Check and finalize the batch submissions found below ``directory``.

    ``adopt`` names a provider batch to record for an unresolved submission;
    ``abandon`` names an unresolved submission whose pages return to pending.
    With ``wait`` the check repeats every ``poll_minutes`` until no batch
    work is pending; no check starts after ``wait_hours``. ``formats``, when
    given, replaces the formats of the submitting run (see
    :func:`check_formats`).
    """

    directory: Path
    batch_ids: tuple[str, ...] = ()
    adopt: str | None = None
    abandon: str | None = None
    wait: bool = False
    poll_minutes: float = DEFAULT_POLL_MINUTES
    wait_hours: float = WAIT_HOURS
    formats: tuple[str, ...] | None = None
    json: bool = False
    settings: Path | None = None


def fold_transcription_format(defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Return *defaults* with the deprecated transcription format in ``formats``.

    The settings key :data:`TRANSCRIPTION_FORMAT_ALIAS` becomes the
    transcription format of the ``formats`` default (the code default
    formats when the block sets none), so the block holds option names only.
    """
    folded = dict(defaults)
    alias = folded.pop(TRANSCRIPTION_FORMAT_ALIAS, None)
    if alias is not None:
        formats = folded.get("formats", DEFAULT_FORMATS)
        folded["formats"] = with_transcription_format(formats, str(alias))
    return folded


def check_formats(
    override: Sequence[str] | None, stored: Sequence[str]
) -> tuple[str, ...]:
    """Return the formats ``batch check`` renders a document in.

    *override* is ``batch check --formats`` (None: *stored*, the formats of
    the submitting run); an override without a transcription format keeps
    the stored one.
    """
    if override is None:
        return tuple(stored)
    keep = None
    if transcription_format_of(override) is None:
        keep = transcription_format_of(stored)
    return with_transcription_format(override, keep)


@dataclass(frozen=True)
class BatchCancelSpec:
    """Cancel the named batches, every unfinished batch with ``all``, or the
    open submissions of the databases below ``directory``."""

    batch_ids: tuple[str, ...] = ()
    all: bool = False
    directory: Path | None = None
    json: bool = False
    settings: Path | None = None


type CommandSpec = RunSpec | BatchCheckSpec | BatchCancelSpec
"""The specs the extension's startup receives."""


def resolve_batch_check(flags: Mapping[str, Any]) -> BatchCheckSpec:
    """Resolve batch check options; no option takes a settings default.

    Raises:
        SpecError: An unknown option, ``--adopt`` with ``--abandon``, or
            ``--poll-minutes`` or ``--wait-hours`` without ``--wait``.
    """
    unknown = set(flags) - set(BATCH_CHECK_OPTIONS.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    if flags.get("adopt") and flags.get("abandon"):
        raise SpecError("--adopt and --abandon exclude each other")
    if not flags.get("wait"):
        for name in ("poll_minutes", "wait_hours"):
            if name in flags:
                flag = BATCH_CHECK_OPTIONS[name].flag
                raise SpecError(f"{flag} needs --wait")
    resolved = BATCH_CHECK_OPTIONS.resolve(flags, None)
    values, _ = BATCH_CHECK_OPTIONS.by_targets(resolved)
    formats = values["formats"]
    if formats is not None:
        if not formats:
            raise SpecError("--formats needs at least one format")
        with_transcription_format(formats, None)
    return BatchCheckSpec(**values)


def resolve_batch_cancel(flags: Mapping[str, Any]) -> BatchCancelSpec:
    """Resolve batch cancel options; exactly one target is required.

    Raises:
        SpecError: An unknown option, or not exactly one of ``--batch-ids``,
            ``--all`` and ``--input``.
    """
    unknown = set(flags) - set(BATCH_CANCEL_OPTIONS.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    values, _ = BATCH_CANCEL_OPTIONS.by_targets(
        BATCH_CANCEL_OPTIONS.resolve(flags, None)
    )
    spec = BatchCancelSpec(**values)
    targets = sum((bool(spec.batch_ids), spec.all, spec.directory is not None))
    if targets != 1:
        raise SpecError("give exactly one of --batch-ids, --all and --input")
    return spec


@dataclass(frozen=True)
class Resolution:
    """The effective spec and the source of each field (dotted path)."""

    spec: RunSpec
    sources: Mapping[str, Source]


def effective_spec(
    spec: RunSpec, sources: Mapping[str, Source] | None = None
) -> dict[str, dict[str, Any]]:
    """Return every field of *spec* (dotted path) with its value and source.

    Values are JSON values; fields missing from *sources* were given by the
    caller and count as flags.
    """
    known = sources or {}
    return {
        path: {
            "value": _jsonable(value),
            "source": known.get(path, Source.FLAG).value,
        }
        for path, value in _flatten(spec).items()
    }


_PATH_FIELDS = frozenset({"input.path", "output.directory"})
_LEGACY_FORMAT_FIELD = "output.transcription_format"


def spec_from_effective(stored: Mapping[str, Any]) -> RunSpec:
    """Return the :class:`RunSpec` that :func:`effective_spec` recorded.

    *stored* maps dotted field paths to ``{"value": ..., "source": ...}``,
    as the store keeps a run's spec; lists become tuples and the path fields
    paths. A field the stored spec lacks, from an earlier version, takes its
    dataclass default. A spec of version 2 kept the transcription format in
    ``output.transcription_format`` and the summary formats in
    ``output.formats``; both join ``output.formats``.

    Raises:
        SpecError: A field without a default is missing or a value is invalid.
    """
    values: dict[str, Any] = {}
    for path, entry in stored.items():
        value = entry.get("value") if isinstance(entry, Mapping) else entry
        if path in _PATH_FIELDS and isinstance(value, str):
            value = Path(value)
        elif isinstance(value, list):
            value = tuple(value)
        values[path] = value
    legacy = values.pop(_LEGACY_FORMAT_FIELD, None)
    if legacy is not None and isinstance(values.get("output.formats"), tuple):
        values["output.formats"] = with_transcription_format(
            values["output.formats"], str(legacy)
        )
    try:
        return RunSpec(
            input=InputSpec(**_part(values, "input")),
            output=OutputSpec(**_part(values, "output")),
            summary=SummarySpec(**_part(values, "summary")),
            citations=CitationSpec(**_part(values, "citations")),
            transcription_model=ModelSpec(**_part(values, "transcription_model")),
            summary_model=ModelSpec(**_part(values, "summary_model")),
            images=ImageSpec(**_part(values, "images")),
            run=RunControl(**_part(values, "run")),
        )
    except TypeError as exc:
        raise SpecError(f"the stored run spec is incomplete: {exc}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        path = f"{prefix}{f.name}"
        if is_dataclass(value):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def _resolve_endpoint(resolved: dict[str, Resolved], phase: str) -> None:
    """Make a named endpoint imply provider custom, by precedence."""
    provider = resolved[f"{phase}_provider"]
    endpoint = resolved[f"{phase}_endpoint"]
    if endpoint.value is not None and provider.value != "custom":
        if endpoint.source.rank >= provider.source.rank:
            resolved[f"{phase}_provider"] = Resolved("custom", endpoint.source)
        else:
            resolved[f"{phase}_endpoint"] = Resolved(None, provider.source)


def _resolve_provider(resolved: dict[str, Resolved], phase: str) -> None:
    """Settle the provider from the model name and strip a provider prefix.

    An unset provider is inferred and takes the model's source. A provider
    set below the model's level yields to the provider the name implies; a
    prefix contradicting a provider set at the model's level or above, or a
    custom provider, is an error.
    """
    provider = resolved[f"{phase}_provider"]
    model = resolved[f"{phase}_model"]
    prefixed, name = split_provider_prefix(model.value)
    if prefixed is not None:
        resolved[f"{phase}_model"] = Resolved(name, model.source)
    if provider.value is None:
        inferred = prefixed or infer_provider(name)
        resolved[f"{phase}_provider"] = Resolved(inferred, model.source)
    elif prefixed is not None and prefixed != provider.value:
        if provider.value == "custom" or provider.source.rank >= model.source.rank:
            raise SpecError(
                f"{phase} model {model.value!r} names provider {prefixed}, "
                f"but the provider is {provider.value}"
            )
        resolved[f"{phase}_provider"] = Resolved(prefixed, model.source)
    elif provider.value != "custom" and provider.source.rank < model.source.rank:
        implied = implied_provider(name)
        if implied is not None and implied != provider.value:
            resolved[f"{phase}_provider"] = Resolved(implied, model.source)
    if (
        resolved[f"{phase}_provider"].value == "custom"
        and resolved[f"{phase}_endpoint"].value is None
    ):
        raise SpecError(
            f"{phase} provider custom needs --endpoint NAME (or "
            f"--{phase}-endpoint) naming an endpoint from the settings"
        )


def resolve_spec(
    flags: Mapping[str, Any], defaults: Mapping[str, Any] | None = None
) -> Resolution:
    """Resolve option values into a spec: flag, settings default, code default.

    ``flags`` are converted values keyed by option name, as returned by
    ``OPTIONS.parse``; ``defaults`` is a checked settings defaults block.
    Both may hold the deprecated :data:`TRANSCRIPTION_FORMAT_ALIAS` (``md``
    or ``txt``); see :func:`_resolve_formats`.

    Raises:
        SpecError: No input path, a custom provider without an endpoint, a
            model prefix that contradicts the provider, or two transcription
            formats.
    """
    flags = dict(flags)
    settings_defaults = dict(defaults or {})
    aliases = (
        (Source.FLAG, flags.pop(TRANSCRIPTION_FORMAT_ALIAS, None)),
        (Source.SETTINGS, settings_defaults.pop(TRANSCRIPTION_FORMAT_ALIAS, None)),
    )
    unknown = set(flags) - set(OPTIONS.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    resolved = OPTIONS.resolve(flags, settings_defaults)
    for phase in PHASES:
        _resolve_endpoint(resolved, phase)
        _resolve_provider(resolved, phase)
    resolved["formats"] = _resolve_formats(resolved["formats"], aliases)
    values, sources = OPTIONS.by_targets(resolved)
    if values["input.path"] is None:
        raise SpecError("an input path is required (--input PATH)")
    spec = RunSpec(
        input=InputSpec(**_part(values, "input")),
        output=OutputSpec(**_part(values, "output")),
        summary=SummarySpec(**_part(values, "summary")),
        citations=CitationSpec(**_part(values, "citations")),
        transcription_model=ModelSpec(**_part(values, "transcription_model")),
        summary_model=ModelSpec(**_part(values, "summary_model")),
        images=ImageSpec(**_part(values, "images")),
        run=RunControl(**_part(values, "run")),
    )
    return Resolution(spec, sources)


def _resolve_formats(
    formats: Resolved, aliases: Sequence[tuple[Source, Any]]
) -> Resolved:
    """Give the resolved formats exactly one transcription format.

    The first given alias of *aliases* (the deprecated transcription format
    as a flag, then as a settings default) replaces the transcription format
    of *formats* when it is set at the level of *formats* or above, and
    fills it in when *formats* names none; otherwise ``md`` does.
    """
    level, alias = next(
        ((source, value) for source, value in aliases if value is not None),
        (Source.DEFAULT, None),
    )
    value = tuple(formats.value)
    source = formats.source
    replaces = alias is not None and level.rank >= source.rank
    if alias is not None and (replaces or transcription_format_of(value) is None):
        value = with_transcription_format(
            [name for name in value if not name.startswith(TRANSCRIPTION_PREFIX)],
            str(alias),
        )
        if level.rank > source.rank:
            source = level
    return Resolved(with_transcription_format(value, None), source)


def _part(values: Mapping[str, Any], name: str) -> dict[str, Any]:
    prefix = name + "."
    return {
        key.removeprefix(prefix): value
        for key, value in values.items()
        if key.startswith(prefix)
    }


def parse(argv: Sequence[str], defaults: Mapping[str, Any] | None = None) -> RunSpec:
    """Parse run arguments into a spec; argparse exits with status 2 on errors."""
    return resolve_spec(OPTIONS.parse(argv), defaults).spec


_UNSET = object()

# Unset values with a spelling; other options cannot be reset to None.
_UNSET_SPELLINGS = {"context": CONTEXT_AUTO, "fallback_context": CONTEXT_NONE}


def _settings_value(option: Option, defaults: Mapping[str, Any]) -> Any:
    """Return what the settings defaults give *option*, or ``_UNSET``."""
    name: str | None = option.name
    while name is not None:
        if name in defaults:
            return defaults[name]
        name = OPTIONS[name].fallback
    return _UNSET


def _spell(given: dict[str, Any], option: Option, value: Any) -> None:
    """Add *option* with *value* to *given*, or raise when it has no spelling."""
    if option.name == "output" and value is None:
        given["beside_input"] = True
    elif option.name in _UNSET_SPELLINGS or (
        value is not None and (option.kind is not Kind.FLAG or value)
    ):
        given[option.name] = value
    else:
        raise SpecError(
            f"the settings default of {option.name} cannot be undone on the "
            f"command line: {option.flag} has no spelling for {value!r}"
        )


def _alternative_given(option: Option, given: Mapping[str, Any]) -> bool:
    """Whether an alternative of *option* is given, which resets *option*."""
    return option.exclusive is not None and any(
        other.exclusive == option.exclusive and other.name in given for other in OPTIONS
    )


def _provider_holds(
    spec: RunSpec, given: Mapping[str, Any], defaults: Mapping[str, Any], phase: str
) -> bool:
    """Whether *phase* keeps its provider and endpoint without a provider flag."""
    trial = dict(given)
    for other in PHASES:
        if other != phase:
            trial[f"{other}_provider"] = spec.model(other).provider
    try:
        resolved = resolve_spec(trial, defaults).spec.model(phase)
    except SpecError:
        return False
    model = spec.model(phase)
    return (resolved.provider, resolved.endpoint) == (model.provider, model.endpoint)


def to_argv(spec: RunSpec, defaults: Mapping[str, Any] | None = None) -> list[str]:
    """Return arguments that reproduce ``spec`` with and without ``defaults``.

    ``defaults`` is a checked settings defaults block. Every value that
    differs from its code default is spelled out, and so is every value the
    defaults would change; a provider the model name implies is omitted
    unless the defaults would change it. A model value equal in both phases
    uses the unprefixed flag.

    Raises:
        SpecError: The defaults change a value that has no spelling, such
            as an unset ``--top-p``.
    """
    settings_defaults = fold_transcription_format(defaults or {})
    flat = _flatten(spec)
    targeted = [
        option
        for option in OPTIONS
        if option.target is not None and not option.target.endswith(".provider")
    ]
    given: dict[str, Any] = {}
    _spell_code_changes(given, targeted, flat)
    _spell_settings_changes(given, targeted, flat, settings_defaults)
    _spell_providers(given, spec, settings_defaults)
    _merge_phases(given, flat)
    for name, keyword in _UNSET_SPELLINGS.items():
        if name in given and given[name] is None:
            given[name] = keyword
    return OPTIONS.to_argv(given)


def _spell_code_changes(
    given: dict[str, Any], targeted: Sequence[Option], flat: Mapping[str, Any]
) -> None:
    """Spell the input and every value that differs from its code default."""
    for option in targeted:
        value = flat[option.target or ""]
        if option.target == "input.path" or value != option.default:
            _spell(given, option, value)


def _spell_settings_changes(
    given: dict[str, Any],
    targeted: Sequence[Option],
    flat: Mapping[str, Any],
    settings_defaults: Mapping[str, Any],
) -> None:
    """Spell every value not yet given that the settings defaults would change."""
    for option in targeted:
        value = flat[option.target or ""]
        if (
            option.name in given
            or _alternative_given(option, given)
            or (option.target or "").endswith(".endpoint")
        ):
            continue
        setting = _settings_value(option, settings_defaults)
        if setting is not _UNSET and setting != value:
            _spell(given, option, value)


def _spell_providers(
    given: dict[str, Any], spec: RunSpec, settings_defaults: Mapping[str, Any]
) -> None:
    """Spell each provider the model name or the settings would not yield."""
    for phase in PHASES:
        model = spec.model(phase)
        if model.provider != infer_provider(model.model) or (
            settings_defaults
            and not _provider_holds(spec, given, settings_defaults, phase)
        ):
            given[f"{phase}_provider"] = model.provider


def _merge_phases(given: dict[str, Any], flat: Mapping[str, Any]) -> None:
    """Replace per-phase model values that agree with the unprefixed option."""
    for name in MODEL_FIELDS:
        per_phase = [f"{phase}_{name}" for phase in PHASES]
        if not any(key in given for key in per_phase):
            continue
        phase_values = [flat[OPTIONS[key].target or ""] for key in per_phase]
        if phase_values[0] == phase_values[1]:
            for key in per_phase:
                given.pop(key, None)
            given[name] = phase_values[0]
