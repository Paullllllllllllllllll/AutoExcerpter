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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

from autoexcerpter.common.options import (
    Condition,
    Kind,
    Option,
    OptionTable,
    Resolved,
    Source,
)
from autoexcerpter.common.polling import DEFAULT_POLL_MINUTES, wait_options
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
    "DEFAULT_MODEL",
    "IMAGE_DETAILS",
    "IMAGE_FORMATS",
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
    "TRANSCRIPTION_FORMATS",
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
    "effective_spec",
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
]

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
VERBOSITIES = ("low", "medium", "high")
SERVICE_TIERS = ("auto", "default", "flex", "priority")
RESPONSE_FORMATS = ("schema", "json", "text")
IMAGE_DETAILS = ("low", "high", "auto", "original")
IMAGE_FORMATS = ("jpeg", "png")
TRANSCRIPTION_FORMATS = ("md", "txt")
OUTPUT_FORMATS = ("summary-md", "summary-docx")
RENDER_FORMATS = ("transcription-md", "transcription-txt", "summary-md", "summary-docx")
PHASES = ("transcription", "summary")

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_VERBOSITY = {"transcription": "high", "summary": "low"}
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SERVICE_TIER = "flex"
DEFAULT_CONCURRENCY = 16

CONTEXT_AUTO = "auto"
CONTEXT_NONE = "none"

MODEL_SUGGESTIONS: Mapping[str, tuple[str, ...]] = {
    "openai": (DEFAULT_MODEL, "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.5"),
    "anthropic": ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
    "google": ("gemini-3.7-flash", "gemini-3.1-pro", "gemini-3.5-flash"),
    "openrouter": (
        "openai/gpt-5.6-luna",
        "anthropic/claude-sonnet-5-5",
        "google/gemini-3.1-pro",
    ),
}

Dpi = int | Literal["native"]


class SpecError(ValueError):
    """The options do not form a valid run specification."""


@dataclass(frozen=True)
class InputSpec:
    """The input path and the selection of discovered items."""

    path: Path
    select: str | None
    all: bool


@dataclass(frozen=True)
class OutputSpec:
    """Where and what to write; ``directory`` None means beside the input.

    ``formats`` are the summary files; the transcription file and the
    database are always written.
    """

    directory: Path | None
    transcription_format: str
    formats: tuple[str, ...]


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


_RECOMMENDED_IMAGES: Mapping[str, ImageSpec] = {
    "openai": ImageSpec("native", "original", "jpeg", 95, True),
    "anthropic": ImageSpec("native", "auto", "jpeg", 95, True),
    "google": ImageSpec(300, "high", "jpeg", 95, True),
    "custom": ImageSpec(150, "high", "jpeg", 85, True),
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("must be > 0")
    return parsed


def _bounded_float(low: float, high: float) -> Any:
    def convert(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not low <= parsed <= high:
            raise ValueError(f"must be between {low} and {high}")
        return parsed

    return convert


def _dpi(value: str) -> Dpi:
    if value.strip().lower() == "native":
        return "native"
    return _positive_int(value)


def _jpeg_quality(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise ValueError("must be between 1 and 100")
    return parsed


def _text(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty")
    return value


def _context(value: str) -> str | None:
    keyword = value.strip().lower()
    if keyword == CONTEXT_AUTO:
        return None
    if keyword == CONTEXT_NONE:
        return CONTEXT_NONE
    return _text(value)


def _fallback_context(value: str) -> str | None:
    if value.strip().lower() == CONTEXT_NONE:
        return None
    return _text(value)


_MODEL_FIELDS: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    ("provider", {"choices": PROVIDERS}, dict.fromkeys(PHASES)),
    (
        "model",
        {"type": _text, "metavar": "NAME", "choice_provider": model_suggestions},
        dict.fromkeys(PHASES, DEFAULT_MODEL),
    ),
    ("endpoint", {"type": _text, "metavar": "NAME"}, dict.fromkeys(PHASES)),
    (
        "reasoning_effort",
        {"choices": REASONING_EFFORTS},
        dict.fromkeys(PHASES, DEFAULT_REASONING_EFFORT),
    ),
    ("verbosity", {"choices": VERBOSITIES}, DEFAULT_VERBOSITY),
    (
        "max_output_tokens",
        {"type": _positive_int, "metavar": "N"},
        dict.fromkeys(PHASES),
    ),
    (
        "temperature",
        {"type": _bounded_float(0.0, 2.0), "metavar": "T"},
        dict.fromkeys(PHASES, DEFAULT_TEMPERATURE),
    ),
    (
        "top_p",
        {"type": _bounded_float(0.0, 1.0), "metavar": "P"},
        dict.fromkeys(PHASES),
    ),
    (
        "service_tier",
        {"choices": SERVICE_TIERS},
        dict.fromkeys(PHASES, DEFAULT_SERVICE_TIER),
    ),
    ("response_format", {"choices": RESPONSE_FORMATS}, dict.fromkeys(PHASES)),
)

_MODEL_HELP = {
    "provider": "model provider",
    "model": "model name; a provider: prefix (anthropic:NAME) selects the provider",
    "endpoint": "named custom endpoint from the settings (implies provider custom)",
    "reasoning_effort": "reasoning effort",
    "verbosity": "output verbosity",
    "max_output_tokens": "maximum output tokens",
    "temperature": "sampling temperature, 0.0 to 2.0",
    "top_p": "nucleus sampling, 0.0 to 1.0; sent only when given and supported",
    "service_tier": "OpenAI service tier",
    "response_format": (
        "schema (API-enforced), json (prompted, validated) or text (plain)"
    ),
}

_MODEL_DEFAULT_HELP = {
    "provider": "inferred from the model name",
    "max_output_tokens": "the model's output limit",
    "endpoint": "none",
    "top_p": "not sent",
    "response_format": "schema when the model supports it, else json",
}


def _model_options() -> list[Option]:
    options: list[Option] = []
    for scope in (None, *PHASES):
        for name, extra, defaults in _MODEL_FIELDS:
            condition = None
            if name == "service_tier":
                provider = "provider" if scope is None else f"{scope}_provider"
                condition = Condition(provider, frozenset({"openai"}))
            if scope is None:
                options.append(
                    Option(
                        name=name,
                        flag="--" + name.replace("_", "-"),
                        help=_MODEL_HELP[name],
                        group="model",
                        default_help=_MODEL_DEFAULT_HELP.get(name, "per phase"),
                        condition=condition,
                        **extra,
                    )
                )
                continue
            default = defaults[scope]
            options.append(
                Option(
                    name=f"{scope}_{name}",
                    flag=f"--{scope}-{name.replace('_', '-')}",
                    help=_MODEL_HELP[name],
                    group=f"{scope} model",
                    default=default,
                    default_help=_MODEL_DEFAULT_HELP.get(name),
                    condition=condition,
                    fallback=name,
                    target=f"{scope}_model.{name}",
                    **extra,
                )
            )
    return options


def _build_table() -> OptionTable:
    recommended = "the provider's recommended value"
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
            type=_text,
            metavar="EXPR",
            exclusive="selection",
            target="input.select",
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
            "output",
            "--output",
            "output folder",
            group="output",
            type=Path,
            metavar="DIR",
            default_help="beside the input",
            exclusive="output_location",
            target="output.directory",
        ),
        Option(
            "beside_input",
            "--beside-input",
            "write outputs beside each input",
            group="output",
            kind=Kind.FLAG,
            default=False,
            exclusive="output_location",
            settable=False,
        ),
        Option(
            "transcription_format",
            "--transcription-format",
            "transcription file format",
            group="output",
            choices=TRANSCRIPTION_FORMATS,
            default="md",
            target="output.transcription_format",
        ),
        Option(
            "formats",
            "--formats",
            "comma-separated output formats",
            group="output",
            kind=Kind.LIST,
            choices=OUTPUT_FORMATS,
            default=OUTPUT_FORMATS,
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
        Option(
            "dpi",
            "--dpi",
            "render resolution: native or a DPI value",
            group="images",
            type=_dpi,
            metavar="native|N",
            default_help=recommended,
            target="images.dpi",
        ),
        Option(
            "image_detail",
            "--image-detail",
            "image detail sent to the model",
            group="images",
            choices=IMAGE_DETAILS,
            default_help=recommended,
            target="images.detail",
        ),
        Option(
            "image_format",
            "--image-format",
            "page image format",
            group="images",
            choices=IMAGE_FORMATS,
            default_help=recommended,
            target="images.format",
        ),
        Option(
            "jpeg_quality",
            "--jpeg-quality",
            "JPEG quality, 1 to 100",
            group="images",
            type=_jpeg_quality,
            metavar="N",
            default_help=recommended,
            target="images.jpeg_quality",
        ),
        Option(
            "grayscale",
            "--grayscale",
            "convert page images to grayscale",
            group="images",
            kind=Kind.SWITCH,
            default_help=recommended,
            target="images.grayscale",
        ),
        Option(
            "force",
            "--force",
            "reprocess items whose outputs exist (resume is the default)",
            group="run",
            kind=Kind.FLAG,
            default=False,
            target="run.force",
            settable=False,
        ),
        Option(
            "retranscribe",
            "--retranscribe",
            "transcribe again instead of reusing stored transcriptions",
            group="run",
            kind=Kind.FLAG,
            default=False,
            target="run.retranscribe",
            settable=False,
        ),
        Option(
            "concurrency",
            "--concurrency",
            "parallel page requests",
            group="run",
            type=_positive_int,
            metavar="N",
            default=DEFAULT_CONCURRENCY,
            target="run.concurrency",
        ),
        Option(
            "batch",
            "--batch",
            "submit the pages and summaries to the providers' batch APIs",
            group="run",
            kind=Kind.SWITCH,
            default=False,
            target="run.batch",
        ),
        Option(
            "dry_run",
            "--dry-run",
            "plan without API calls or writes",
            group="run",
            kind=Kind.FLAG,
            default=False,
            target="run.dry_run",
            settable=False,
        ),
        Option(
            "json",
            "--json",
            "print one JSON summary line on stdout",
            group="run",
            kind=Kind.FLAG,
            default=False,
            target="run.json",
            settable=False,
        ),
        Option(
            "settings",
            "--settings",
            "alternate settings file",
            group="run",
            type=Path,
            metavar="FILE",
            default_help="config/settings.yaml",
            settable=False,
        ),
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


def _render_flag(name: str, help: str) -> Option:
    return Option(
        name,
        "--" + name.replace("_", "-"),
        help,
        group="run",
        kind=Kind.FLAG,
        default=False,
        target=name,
        settable=False,
    )


RENDER_OPTIONS = OptionTable(
    [
        Option(
            "input",
            "--input",
            "output folder or autoexcerpter.sqlite to render",
            group="input",
            type=Path,
            metavar="PATH",
            default=Path("."),
            target="input",
            settable=False,
        ),
        Option(
            "formats",
            "--formats",
            "comma-separated formats to render",
            group="output",
            kind=Kind.LIST,
            choices=RENDER_FORMATS,
            metavar="LIST",
            default_help="the formats each document was rendered in",
            target="formats",
            settable=False,
        ),
        _render_flag("force", "replace files changed outside the tool"),
        _render_flag("dry_run", "print what would be written without writing"),
        _render_flag("json", "print one JSON summary line on stdout"),
    ],
    group_titles={"input": "input", "output": "output", "run": "run control"},
)


@dataclass(frozen=True)
class RenderSpec:
    """The options of ``render``; ``formats`` None renders the registered ones."""

    input: Path
    formats: tuple[str, ...] | None = None
    force: bool = False
    dry_run: bool = False
    json: bool = False


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
        Option(
            "settings",
            "--settings",
            "settings file whose state_dir holds the OpenAlex cache",
            group="run",
            type=Path,
            metavar="FILE",
            default_help="config/settings.yaml",
            target="settings",
            settable=False,
        ),
        _render_flag("dry_run", "print what would be imported without writing"),
        _render_flag("json", "print one JSON summary line on stdout"),
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


def _batch_ids_option(help: str, exclusive: str | None = None) -> Option:
    return Option(
        "batch_ids",
        "--batch-ids",
        help,
        group="batch",
        kind=Kind.LIST,
        type=_text,
        metavar="IDS",
        default=(),
        exclusive=exclusive,
        target="batch_ids",
        settable=False,
    )


def _settings_option() -> Option:
    return Option(
        "settings",
        "--settings",
        "alternate settings file",
        group="run",
        type=Path,
        metavar="FILE",
        default_help="config/settings.yaml",
        target="settings",
        settable=False,
    )


_BATCH_TITLES = {"batch": "batches", "run": "run control"}

WAIT_HOURS = 50.0
"""Default ``--wait-hours``: a check waits for two batch rounds in sequence,
the transcriptions and then the summaries, each of up to a day."""

BATCH_CHECK_OPTIONS = OptionTable(
    [
        Option(
            "input",
            "--input",
            "folder searched for batch submissions, subfolders included",
            group="batch",
            type=Path,
            metavar="DIR",
            default=Path("."),
            default_help="the current directory",
            target="directory",
            settable=False,
        ),
        _batch_ids_option("comma-separated batch IDs to check; default every batch"),
        Option(
            "adopt",
            "--adopt",
            "record this provider batch for the unresolved submission it belongs "
            "to, then finalize it",
            group="batch",
            metavar="BATCH_ID",
            exclusive="unresolved_action",
            target="adopt",
            settable=False,
        ),
        Option(
            "abandon",
            "--abandon",
            "give up an unresolved submission (its upload is deleted) or a "
            "submitted one whose results are lost: its pages return to pending "
            "and its documents wait for a run",
            group="batch",
            metavar="SUBMISSION_ID",
            exclusive="unresolved_action",
            target="abandon",
            settable=False,
        ),
        *wait_options(group="batch", default_hours=WAIT_HOURS),
        _render_flag("json", "print one JSON summary line on stdout"),
        _settings_option(),
    ],
    group_titles=_BATCH_TITLES,
)

BATCH_CANCEL_OPTIONS = OptionTable(
    [
        _batch_ids_option("comma-separated batch IDs to cancel", "cancel_target"),
        Option(
            "all",
            "--all",
            "cancel every unfinished batch of every provider",
            group="batch",
            kind=Kind.FLAG,
            default=False,
            exclusive="cancel_target",
            target="all",
            settable=False,
        ),
        Option(
            "input",
            "--input",
            "cancel the open batch submissions recorded in the databases below "
            "this folder",
            group="batch",
            type=Path,
            metavar="DIR",
            exclusive="cancel_target",
            target="directory",
            settable=False,
        ),
        _render_flag("json", "print one JSON summary line on stdout"),
        _settings_option(),
    ],
    group_titles=_BATCH_TITLES,
)


@dataclass(frozen=True)
class BatchCheckSpec:
    """Check and finalize the batch submissions found below ``directory``.

    ``adopt`` names a provider batch to record for an unresolved submission;
    ``abandon`` names an unresolved submission whose pages return to pending.
    With ``wait`` the check repeats every ``poll_minutes`` until no batch
    work is pending; no check starts after ``wait_hours``.
    """

    directory: Path
    batch_ids: tuple[str, ...] = ()
    adopt: str | None = None
    abandon: str | None = None
    wait: bool = False
    poll_minutes: float = DEFAULT_POLL_MINUTES
    wait_hours: float = WAIT_HOURS
    json: bool = False
    settings: Path | None = None


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


def spec_from_effective(stored: Mapping[str, Any]) -> RunSpec:
    """Return the :class:`RunSpec` that :func:`effective_spec` recorded.

    *stored* maps dotted field paths to ``{"value": ..., "source": ...}``,
    as the store keeps a run's spec; lists become tuples and the path fields
    paths. A field the stored spec lacks, from an earlier version, takes its
    dataclass default.

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

    Raises:
        SpecError: No input path, a custom provider without an endpoint, or
            a model prefix that contradicts the provider.
    """
    unknown = set(flags) - set(OPTIONS.names)
    if unknown:
        raise SpecError(f"unknown options: {', '.join(sorted(unknown))}")
    resolved = OPTIONS.resolve(flags, defaults)
    for phase in PHASES:
        _resolve_endpoint(resolved, phase)
        _resolve_provider(resolved, phase)
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
    settings_defaults = defaults or {}
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
    for name, _extra, _defaults in _MODEL_FIELDS:
        per_phase = [f"{phase}_{name}" for phase in PHASES]
        if not any(key in given for key in per_phase):
            continue
        phase_values = [flat[OPTIONS[key].target or ""] for key in per_phase]
        if phase_values[0] == phase_values[1]:
            for key in per_phase:
                given.pop(key, None)
            given[name] = phase_values[0]
