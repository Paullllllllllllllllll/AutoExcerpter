"""Review screen of the guided run: summary lines, plan preview, command.

The review resolves the answers like ``autoexcerpter run`` would, marks
values taken from the settings defaults, previews the plan with
:func:`autoexcerpter.run.plan` (no API calls, no writes) and withholds Run
while the plan fails or a needed API key variable is unset. The equivalent
command spells out every value that differs from the code default.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Final

from autoexcerpter import run as core
from autoexcerpter.common.options import Source
from autoexcerpter.common.wizard import Line, Part, Review, ReviewAction, State
from autoexcerpter.imaging.settings import detect_model_type
from autoexcerpter.pipeline.resume import ProcessingState
from autoexcerpter.settings import SettingsError
from autoexcerpter.spec import (
    CONTEXT_NONE,
    OPTIONS,
    ImageSpec,
    ModelSpec,
    RunSpec,
    recommended_images,
    to_argv,
)
from autoexcerpter.ui.session import (
    Session,
    absolute,
    inventory_of,
    plural,
    resolution,
    selection_of,
    support_of,
)

__all__ = [
    "FORCE_ACTION",
    "IMAGE_OPTIONS",
    "IMAGE_TEXT_INDEX",
    "PROGRAM",
    "build_review",
    "describe_images",
    "equivalent_command",
    "image_family",
    "plan_notes",
]

PROGRAM: Final = ("autoexcerpter", "run")
FORCE_ACTION: Final = "force"
IMAGE_OPTIONS: Final = (
    "dpi",
    "image_detail",
    "image_format",
    "jpeg_quality",
    "grayscale",
)
IMAGE_TEXT_INDEX: Final = {
    "dpi": 0,
    "image_detail": 1,
    "image_format": 2,
    "jpeg_quality": 2,
    "grayscale": 3,
}
PLAN_LIMIT: Final = 20

_Mark = Callable[[str, str], Part]


def equivalent_command(state: State, session: Session) -> list[str]:
    """Return the words of the ``autoexcerpter run`` command for the answers.

    Raises:
        ValueError: The answers do not resolve, or a settings default cannot
            be undone on the command line.
    """
    spec = resolution(state, session).spec
    words = [*PROGRAM, *to_argv(spec, session.settings.defaults)]
    if session.settings_path is not None:
        words += ["--settings", str(absolute(session.settings_path))]
    return words


def image_family(model: ModelSpec | None) -> str:
    """Return the image-settings family of a transcription model."""
    if model is None:
        return "openai"
    return detect_model_type(model.provider, model.model)


def describe_images(images: ImageSpec) -> list[str]:
    """Return the texts of resolved image settings: DPI, detail, format, color."""
    dpi = "native DPI" if images.dpi == "native" else f"{images.dpi} DPI"
    encoding = "PNG" if images.format == "png" else f"JPEG {images.jpeg_quality}"
    return [
        dpi,
        f"detail {images.detail}",
        encoding,
        "grayscale" if images.grayscale else "color",
    ]


def _short(text: str, width: int = 40) -> str:
    return textwrap.shorten(" ".join(text.split()), width, placeholder="...")


def _input_line(state: State, spec: RunSpec, mark: _Mark) -> Line:
    path = absolute(spec.input.path)
    inventory = inventory_of(state, path)
    count = len(inventory.items)
    parts = [mark("input.path", f"{path}  ({inventory.describe()})")]
    if count > 1:
        if spec.input.all:
            parts.append(mark("input.all", f"all {count} items"))
        elif spec.input.select:
            picked = len(selection_of(inventory, spec.input.select).indices)
            parts.append(
                mark(
                    "input.select",
                    f"select {spec.input.select}: {picked} of {count} items",
                )
            )
        else:
            parts.append(Part("no selection"))
    return Line("input", parts)


def _output_line(spec: RunSpec, mark: _Mark) -> Line:
    output = spec.output
    where = "beside input" if output.directory is None else str(output.directory)
    formats = ", ".join(output.formats) or "none"
    return Line(
        "output",
        [
            mark("output.directory", where),
            mark(
                "output.transcription_format",
                f"transcription {output.transcription_format}",
            ),
            mark("output.formats", f"formats {formats}"),
        ],
    )


def _context_text(value: str | None, label: str) -> str:
    if value is None:
        return "context from sidecars" if label == "context" else "no fallback"
    if value == CONTEXT_NONE:
        return "no context"
    path = absolute(value)
    if path.is_file():
        return f"{label} file {path}"
    return f"{label}: {_short(value)}"


def _summary_line(spec: RunSpec, mark: _Mark) -> Line:
    summary = spec.summary
    if not summary.enabled:
        return Line("summary", [mark("summary.enabled", "off")])
    parts = [
        mark("summary.enabled", "on"),
        mark("summary.context", _context_text(summary.context, "context")),
    ]
    if summary.context is None:
        parts.append(
            mark(
                "summary.fallback_context",
                _context_text(summary.fallback_context, "fallback"),
            )
        )
    return Line("summary", parts)


def _model_line(
    phase: str, spec: RunSpec, session: Session, sources: Mapping[str, Source]
) -> Line:
    model = spec.model(phase)
    support = support_of(model, session.settings)
    prefix = f"{phase}_model."

    def mark(field: str, text: str) -> Part:
        return Part(text, sources.get(prefix + field) is Source.SETTINGS)

    def given(field: str) -> bool:
        return sources.get(prefix + field, Source.FLAG) is not Source.DEFAULT

    if model.provider == "custom":
        name = f"endpoint {model.endpoint} / {model.model}"
    else:
        name = f"{model.provider} / {model.model}"
    parts = [mark("model", name)]
    if support.reasoning:
        parts.append(mark("reasoning_effort", f"reasoning {model.reasoning_effort}"))
    if support.service_tier:
        parts.append(mark("service_tier", f"tier {model.service_tier}"))
    if model.response_format is None:
        response = f"response automatic ({support.automatic_format})"
    else:
        response = f"response {model.response_format}"
    parts.append(mark("response_format", response))
    if support.verbosity and given("verbosity"):
        parts.append(mark("verbosity", f"verbosity {model.verbosity}"))
    if model.max_output_tokens is not None:
        parts.append(mark("max_output_tokens", f"max {model.max_output_tokens} tokens"))
    if support.temperature and given("temperature"):
        parts.append(mark("temperature", f"temperature {model.temperature}"))
    if support.top_p and model.top_p is not None:
        parts.append(mark("top_p", f"top p {model.top_p}"))
    return Line(f"{phase} model", parts)


def _images_line(spec: RunSpec, sources: Mapping[str, Source]) -> Line:
    family = image_family(spec.transcription_model)
    texts = describe_images(spec.images.over(recommended_images(family)))
    flagged = {
        IMAGE_TEXT_INDEX[name]
        for name in IMAGE_OPTIONS
        if sources.get(OPTIONS[name].target or "") is Source.SETTINGS
    }
    parts = [Part(text, index in flagged) for index, text in enumerate(texts)]
    if all(getattr(spec.images, field.name) is None for field in fields(ImageSpec)):
        parts.append(Part(f"recommended for {family}"))
    return Line("images", parts)


def _run_line(spec: RunSpec, mark: _Mark) -> Line:
    run = spec.run
    parts = [mark("run.force", "force" if run.force else "resume")]
    if run.retranscribe:
        parts.append(mark("run.retranscribe", "retranscribe"))
    if spec.output.keep_working_files:
        parts.append(mark("output.keep_working_files", "keep working files"))
    if spec.summary.enabled:
        openalex = "on" if spec.citations.openalex else "off"
        parts.append(mark("citations.openalex", f"OpenAlex {openalex}"))
    parts.append(mark("run.concurrency", f"concurrency {run.concurrency}"))
    if run.dry_run:
        parts.append(mark("run.dry_run", "dry run"))
    return Line("run", parts)


def _item_state(planned: core.PlannedItem) -> str:
    state = planned.resume.state
    if state is ProcessingState.NONE:
        return "new"
    reused = plural(planned.completed_pages, "page")
    if state is ProcessingState.TRANSCRIPTION_ONLY:
        return f"partial, transcription done ({reused} reused)"
    return f"partial ({reused} reused)"


def plan_notes(plan: core.RunPlan) -> list[str]:
    """Return the plan preview: counts, one line per item, warnings."""
    states = [_item_state(planned) for planned in plan.to_process]
    new = sum(1 for state in states if state == "new")
    partial = len(states) - new
    lines = [f"Plan: {new} new, {partial} partial, {len(plan.skipped)} complete"]
    entries = [
        f"  {planned.name}: {state}"
        for planned, state in zip(plan.to_process, states, strict=True)
    ]
    entries += [f"  {planned.name}: complete (skipped)" for planned in plan.skipped]
    lines += entries[:PLAN_LIMIT]
    if len(entries) > PLAN_LIMIT:
        lines.append(f"  ... and {len(entries) - PLAN_LIMIT} more")
    lines += [f"Warning: {warning}" for warning in plan.warnings]
    return lines


def _force(state: State) -> None:
    state.set("force", True)


def _missing_keys(spec: RunSpec, session: Session) -> tuple[list[str], list[str]]:
    """Return the notes and the withheld reasons for unset key variables."""
    phases = ["transcription", *(["summary"] if spec.summary.enabled else [])]
    missing: dict[str, list[str]] = {}
    errors: list[str] = []
    for phase in phases:
        model = spec.model(phase)
        try:
            variable = session.settings.api_key_env(model.provider, model.endpoint)
        except SettingsError as exc:
            errors.append(str(exc))
            continue
        if not session.environ.get(variable):
            missing.setdefault(variable, []).append(phase)
    notes = [
        f"Missing API key: {variable} is not set ({', '.join(used)})"
        for variable, used in missing.items()
    ]
    if missing:
        errors.append(f"set {', '.join(missing)} in the environment")
    return notes, errors


def build_review(state: State, session: Session) -> Review:
    """Return the review screen of the answers in *state*."""
    try:
        resolved = resolution(state, session)
    except ValueError as exc:
        return Review(lines=(), withheld=str(exc))
    spec, sources = resolved.spec, resolved.sources

    def mark(target: str, text: str) -> Part:
        return Part(text, sources.get(target) is Source.SETTINGS)

    lines = [
        _input_line(state, spec, mark),
        _output_line(spec, mark),
        _summary_line(spec, mark),
        _model_line("transcription", spec, session, sources),
    ]
    if spec.summary.enabled:
        lines.append(_model_line("summary", spec, session, sources))
    lines += [_images_line(spec, sources), _run_line(spec, mark)]

    notes: list[str] = []
    withheld: list[str] = []
    actions: list[ReviewAction] = []
    try:
        plan = core.plan(spec, session.settings, sources=sources)
    except core.PlanError as exc:
        notes.append(f"Plan error: {exc}")
        withheld.append("the plan failed; edit a step")
    else:
        notes += plan_notes(plan)
        if plan.skipped and not spec.run.force:
            count = len(plan.skipped)
            actions.append(
                ReviewAction(
                    FORCE_ACTION,
                    f"Reprocess {count} complete item(s) (--force)",
                    _force,
                )
            )
    if spec.run.dry_run:
        notes.append("Dry run: Run shows the plan only, with no API calls or writes.")
    else:
        key_notes, key_errors = _missing_keys(spec, session)
        notes += key_notes
        withheld += key_errors
    return Review(
        lines=lines,
        notes=notes,
        actions=actions,
        withheld="; ".join(withheld) if withheld else None,
    )
