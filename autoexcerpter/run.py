"""Core entry points: discover items, plan a run, run it.

:func:`plan` turns a resolved :class:`~autoexcerpter.spec.RunSpec` and the
machine settings into a :class:`RunPlan` without API calls or writes;
:func:`run` processes the plan's items on one event loop and reports
progress through :class:`~autoexcerpter.events.RunEvents`. Neither prints,
prompts or exits: a plan that cannot run raises a :class:`PlanError` that
carries its exit code. Cancelling the task that awaits :func:`run` stops the
run, retry waits included. Under ``--batch`` each item's missing units are
submitted to the providers' batch APIs instead of called
(:mod:`autoexcerpter.batch`), and ``batch check`` continues the work.

Every output folder holds one ``autoexcerpter.sqlite``, the store of its
documents (see :mod:`autoexcerpter.output.store`). Resume reads only the
store: an item whose rendered output exists without units in the store is
``unstored`` and skipped, an item leased by
another live run is ``blocked`` and skipped, and an item whose stored pages
used other image settings is ``refused`` and counts failed.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import functools
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from autoexcerpter import ext
from autoexcerpter.batch.clients import BatchClients
from autoexcerpter.batch.item import BatchItem, BatchItemOutcome
from autoexcerpter.common.batch import supports_batch
from autoexcerpter.common.contract import ExitCode, exit_code_for
from autoexcerpter.common.options import Source
from autoexcerpter.common.selection import PageRange, select_items
from autoexcerpter.common.spec_view import resolved_view
from autoexcerpter.common.sqlite import DEFAULT_BUSY_TIMEOUT, DEFAULT_WARN_INTERVAL
from autoexcerpter.common.store import (
    Lease,
    LeaseKeeper,
    LeaseLostError,
    Store,
    StoreError,
    UnitRow,
    WriterOptions,
)
from autoexcerpter.common.usage import Usage
from autoexcerpter.events import (
    ItemFinished,
    ItemStarted,
    NullEvents,
    RunEvents,
)
from autoexcerpter.imaging.settings import resolve_image_settings
from autoexcerpter.llm import CallEnv
from autoexcerpter.llm.options import request_plan, response_format_for
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.output.store import (
    KIND_TRANSCRIPTION,
    Classification,
    InputMismatchError,
    ItemState,
    ResumeRequest,
    begin_run,
    classify,
    document_kind,
    drop_batch_job,
    image_changes,
    image_settings_record,
    lease_reason,
    open_store,
    read_store,
)
from autoexcerpter.pipeline.context import ContextError, summary_context
from autoexcerpter.pipeline.item import (
    ItemFiles,
    ItemProcessor,
    input_sha256,
    open_source,
)
from autoexcerpter.pipeline.job import ItemJob, job_from_spec
from autoexcerpter.pipeline.scanner import scan_input_path
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.pipeline.units import DocumentHandle, Stamp
from autoexcerpter.settings import Settings, SettingsError
from autoexcerpter.spec import CONTEXT_NONE, ImageSpec, RunSpec, effective_spec

__all__ = [
    "BUSY_TIMEOUT",
    "BUSY_WARN_INTERVAL",
    "AmbiguousSelectionError",
    "ConfigurationError",
    "DuplicateOutputError",
    "ItemResult",
    "MissingKey",
    "NoItemsError",
    "PlanError",
    "PlannedItem",
    "RunPlan",
    "RunResult",
    "SelectionError",
    "StoreRefusedError",
    "discover",
    "idle_result",
    "missing_keys",
    "plan",
    "run",
]

logger = logging.getLogger(__name__)

BUSY_TIMEOUT: float = DEFAULT_BUSY_TIMEOUT
"""Seconds a store write or read waits on a busy database before it backs off."""

BUSY_WARN_INTERVAL: float = DEFAULT_WARN_INTERVAL
"""Seconds between two warnings about the same busy database."""

_RESUMABLE = (ItemState.PARTIAL, ItemState.TRANSCRIPTION_ONLY)
# Inputs hashed at once while planning; the threads overlap the file reads.
_HASH_WORKERS = 8


class PlanError(Exception):
    """The run cannot be planned.

    *items_total* is the number of items the run would have processed, for
    the run summary; ``exit_code`` is the process exit code of the error.
    """

    exit_code: ExitCode = ExitCode.USAGE

    def __init__(self, message: str, *, items_total: int = 0) -> None:
        super().__init__(message)
        self.items_total = items_total


class ConfigurationError(PlanError):
    """The spec and settings cannot serve the run (exit 2)."""


class StoreRefusedError(ConfigurationError):
    """An output folder's database is not a current store (exit 2).

    The database is newer or unreadable; it is left as it is.
    """


class NoItemsError(PlanError):
    """The input holds no PDF or image folder (exit 2)."""


class SelectionError(PlanError):
    """``--select`` matched no item, or ``--files`` named one not found (exit 2)."""


class AmbiguousSelectionError(PlanError):
    """Several items were found without a selection option (exit 2).

    The selection options are ``--all``, ``--select`` and ``--files``.
    """


class DuplicateOutputError(PlanError):
    """Two items, selected now or stored earlier, write the same outputs (exit 2)."""


@dataclass(frozen=True)
class PlannedItem:
    """One selected item: where it writes and what the store holds for it.

    *resume* is the item's classification from the store of *output_dir*;
    *completed_pages* counts the pages an earlier run transcribed that this
    run keeps. *context* is the item's summary context as it enters the
    prompt; *transcription_format* the transcription file's extension;
    *input_sha256* the hash of the input at planning. *images*, when set,
    replaces the run's image options for this item: the values stored with
    its pages (see :meth:`_Classifier._stored_images`).
    """

    item: ItemSpec
    output_dir: Path
    resume: Classification
    context: str | None = None
    transcription_format: str = "md"
    input_sha256: str | None = None
    completed_pages: int = 0
    images: ImageSpec | None = None

    @property
    def name(self) -> str:
        """The item's output stem, its document name in the store."""
        return self.item.output_stem

    @property
    def paths(self) -> ItemPaths:
        """The item's output paths."""
        return ItemPaths.for_item(self.name, self.output_dir, self.transcription_format)

    @property
    def state(self) -> str:
        """The resume state.

        One of none, partial, transcription_only, complete, unstored, blocked
        or refused.
        """
        return self.resume.state.value

    @property
    def reason(self) -> str:
        """Why the item has its resume state."""
        return self.resume.reason


@dataclass(frozen=True)
class RunPlan:
    """What a run will do.

    *output_dir* None writes beside each input. *to_process* and *skipped*
    keep the discovery order; *skipped* items are already complete, and one
    whose files are missing or stale is rendered again without model calls.
    *unstored* items have outputs without store rows, *blocked* items are
    leased by another run, *refused* items were stored with other image
    settings; none of them is processed. *warnings* are notes for the user,
    such as ignored ``--select`` parts and, in a dry run, unset key
    variables. *settings* are the machine settings the run uses. *sources*
    maps spec fields to where their values came from; fields it omits were
    given by the caller and count as flags. *missing_keys* are the key
    variables the run needs that are unset; a real run stops on them.
    """

    spec: RunSpec
    job: ItemJob
    input_path: Path
    output_dir: Path | None
    discovered: tuple[ItemSpec, ...]
    to_process: tuple[PlannedItem, ...]
    skipped: tuple[PlannedItem, ...]
    warnings: tuple[str, ...] = ()
    settings: Settings = field(default_factory=Settings)
    sources: Mapping[str, Source] = field(default_factory=dict)
    unstored: tuple[PlannedItem, ...] = ()
    blocked: tuple[PlannedItem, ...] = ()
    refused: tuple[PlannedItem, ...] = ()
    missing_keys: tuple[MissingKey, ...] = ()

    @property
    def force(self) -> bool:
        """Whether existing outputs are reprocessed instead of resumed."""
        return self.spec.run.force

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        return self.spec.run.dry_run

    @property
    def batch(self) -> bool:
        """Whether the units are submitted to the providers' batch APIs."""
        return self.spec.run.batch

    @property
    def render_only(self) -> tuple[PlannedItem, ...]:
        """The complete items whose files are written again without model calls."""
        return tuple(item for item in self.skipped if item.resume.render)

    @property
    def not_processed(self) -> tuple[PlannedItem, ...]:
        """The items the run leaves alone: complete, unstored and blocked ones."""
        return (*self.skipped, *self.unstored, *self.blocked)

    @property
    def leased(self) -> tuple[PlannedItem, ...]:
        """The blocked items another run's lease holds, not a batch."""
        return tuple(item for item in self.blocked if item.resume.leased_by)

    @property
    def has_work(self) -> bool:
        """Whether a run of the plan processes, renders or refuses an item."""
        return bool(self.to_process or self.refused or self.render_only)

    def effective_spec(self) -> dict[str, dict[str, Any]]:
        """Return ``{field: {"value": ..., "source": ...}}`` for every field.

        The store records it with the run, and a batch check rebuilds the
        run's spec from it.
        """
        return effective_spec(self.spec, self.sources)

    def resolved_spec(self) -> dict[str, dict[str, Any]]:
        """Return :meth:`effective_spec` as the dry run shows it.

        The unset output limit and response format of both models and the
        unset image options carry the values the requests use; a provider
        other than OpenAI shows no service tier
        (:func:`~autoexcerpter.common.spec_view.resolved_view`).
        """
        view = self.effective_spec()
        phases = (
            ("transcription_model", self.job.transcription),
            ("summary_model", self.job.summary),
        )
        for prefix, phase in phases:
            images: dict[str, Any] | None = None
            if prefix == "transcription_model":
                try:
                    images, _family, _section = resolve_image_settings(
                        phase.provider,
                        phase.name,
                        self.spec.images,
                        self.settings.images.original_max_side_px,
                    )
                except ValueError:
                    images = None
            plan = request_plan(phase.provider, phase.name, phase.options)
            view = resolved_view(
                view,
                model=prefix,
                provider=phase.provider,
                limit=plan.limit,
                response_format=response_format_for(
                    phase.response_format, phase.provider, phase.name
                ),
                images=images,
            )
        return view


def _absolute(path: Path) -> Path:
    """Expand ``~`` and make *path* absolute against the current directory."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def discover(input_path: Path) -> list[ItemSpec]:
    """Return the PDFs and image folders found at *input_path*."""
    return scan_input_path(_absolute(input_path))


def _item_output_dir(item: ItemSpec, base: Path | None) -> Path:
    return item.path.parent if base is None else base


def _select_files(items: list[ItemSpec], names: tuple[str, ...]) -> list[ItemSpec]:
    """Return the items named exactly by *names*, in discovery order.

    Raises:
        SelectionError: A name matches no discovered item.
    """
    found = {item.path.name for item in items}
    missing = [name for name in names if name not in found]
    if missing:
        raise SelectionError(
            f"--files names no item found in the input: {', '.join(missing)}"
        )
    wanted = set(names)
    return [item for item in items if item.path.name in wanted]


def _select(
    items: list[ItemSpec], spec: RunSpec
) -> tuple[list[ItemSpec], tuple[str, ...]]:
    """Apply ``--all``, ``--select`` or ``--files``; a single item needs none."""
    selection = spec.input
    if selection.all:
        return list(items), ()
    if selection.files:
        return _select_files(items, selection.files), ()
    if selection.select:
        result = select_items([item.path.name for item in items], selection.select)
        warnings: tuple[str, ...] = ()
        if result.unmatched:
            warnings = (
                "Ignored out-of-range/invalid --select part(s): "
                f"{', '.join(result.unmatched)} (valid: 1-{len(items)})",
            )
        selected = result.pick(items)
        if not selected:
            raise SelectionError(f"No items found matching '{selection.select}'")
        return selected, warnings
    if len(items) == 1:
        return list(items), ()
    raise AmbiguousSelectionError(
        f"{len(items)} items found but none of --all, --select and --files was "
        "given. Pass --all to process every item, --select to choose items by "
        "number, range or name search, or --files to name them exactly."
    )


def _guard_duplicate_outputs(items: list[ItemSpec], base: Path | None) -> None:
    """Refuse two items that resolve to the same output stem and folder.

    Runs before the resume check: otherwise one colliding item's outputs
    could classify the other as complete and hide the collision.
    """
    seen: dict[tuple[Path, str], ItemSpec] = {}
    for item in items:
        key = (_item_output_dir(item, base).resolve(), item.output_stem)
        prior = seen.get(key)
        if prior is not None:
            raise DuplicateOutputError(
                f"Duplicate output target: '{prior.path}' and '{item.path}' both "
                f"resolve to output stem '{item.output_stem}' in the same "
                "directory; their outputs would overwrite each other. Rename one "
                "input or send them to separate output directories.",
                items_total=len(items),
            )
        seen[key] = item


def _job(spec: RunSpec, settings: Settings) -> ItemJob:
    try:
        job = job_from_spec(spec, settings)
    except SettingsError as exc:
        raise ConfigurationError(str(exc)) from exc
    if job.summarize and not (job.output_markdown or job.output_docx):
        raise ConfigurationError(
            "Summaries are on but no output format is selected, so no summary "
            "would be written. Pass --no-summarize, or select summary-md or "
            "summary-docx with --formats."
        )
    return job


def _batch_phases(job: ItemJob) -> list[tuple[str, PhaseModel]]:
    """Return the phases a batch run submits: (role, phase model)."""
    phases: list[tuple[str, PhaseModel]] = [("transcription", job.transcription)]
    if job.summarize:
        phases.append(("summary", job.summary))
    return phases


@dataclass(frozen=True)
class MissingKey:
    """An unset API key variable and the phases of the run that need it."""

    variable: str
    provider: str
    roles: tuple[str, ...]

    def describe(self) -> str:
        """Return ``OPENAI_API_KEY (openai; transcription, summary)``."""
        return f"{self.variable} ({self.provider}; {', '.join(self.roles)})"


def missing_keys(
    job: ItemJob,
    settings: Settings,
    *,
    batch: bool = False,
    args: argparse.Namespace | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[MissingKey, ...]:
    """Return the key variables the run's phases need that are unset.

    A phase needs one of the variables the extension names for its provider
    (:func:`autoexcerpter.ext.key_variables`, batch clients under *batch*);
    the first of them is reported. *environ* defaults to ``os.environ``.

    Raises:
        ConfigurationError: The settings name no key variable for a phase.
    """
    env = os.environ if environ is None else environ
    phases = [("transcription", job.transcription)]
    if job.summarize:
        phases.append(("summary", job.summary))
    found: dict[str, MissingKey] = {}
    for role, phase in phases:
        try:
            variables = ext.key_variables(
                phase.provider,
                phase.endpoint,
                kind="batch" if batch else "chat",
                settings=settings,
                args=args,
            )
        except SettingsError as exc:
            raise ConfigurationError(str(exc)) from exc
        if not variables or any(env.get(name) for name in variables):
            continue
        known = found.get(variables[0])
        roles = (*known.roles, role) if known else (role,)
        found[variables[0]] = MissingKey(variables[0], phase.provider, roles)
    return tuple(found.values())


def _key_warning(missing: MissingKey, *, batch: bool = False) -> str:
    roles = " and ".join(missing.roles)
    if batch:
        needs = f"--batch needs it for the {roles} batches"
    else:
        plural = "s" if len(missing.roles) > 1 else ""
        needs = f"a real run needs it for the {roles} phase{plural}"
    return (
        f"The API key variable {missing.variable} of provider {missing.provider} "
        f"is not set; {needs}."
    )


def _check_batch(
    spec: RunSpec, job: ItemJob, settings: Settings, args: argparse.Namespace | None
) -> None:
    """Require a batch API for every phase and the extension's consent.

    Raises:
        ConfigurationError: A phase's provider has no batch API or uses a
            custom endpoint, or the extension refuses batch submission.
    """
    for role, phase in _batch_phases(job):
        if phase.endpoint is not None or not supports_batch(phase.provider):
            where = (
                f"the custom endpoint {phase.endpoint!r}"
                if phase.endpoint is not None
                else f"provider {phase.provider}"
            )
            raise ConfigurationError(
                "--batch needs a provider with a batch API (openai, anthropic, "
                f"google) for every phase; the {role} model uses {where}."
            )
    try:
        allowed = ext.batch_allowed(spec, settings=settings, args=args)
    except SettingsError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not allowed:
        raise ConfigurationError("Batch submission is not available for this run.")


def _context(item: ItemSpec, job: ItemJob) -> str | None:
    if not job.summarize or job.context == CONTEXT_NONE:
        return None
    try:
        return summary_context(item.path, job.context, default=job.fallback_context)
    except ContextError as exc:
        raise ConfigurationError(str(exc)) from exc


def _image_record(
    job: ItemJob, provider: str | None = None, name: str | None = None
) -> dict[str, Any] | None:
    """Return the run's image-settings record; None when it cannot be resolved.

    *provider* and *name* resolve the run's image options for another model
    than the transcription model. An image setting that cannot be resolved
    fails the item when its source opens.
    """
    model = job.transcription
    if name is None:
        provider, name = model.provider, model.name
    try:
        settings, _, _ = resolve_image_settings(
            provider, name, job.images, job.original_max_side_px
        )
    except ValueError:
        return None
    return image_settings_record(settings)


def _expected_outputs(paths: ItemPaths, job: ItemJob) -> list[Path]:
    expected = [paths.transcription]
    if job.summarize:
        if job.output_docx:
            expected.append(paths.summary_docx)
        if job.output_markdown:
            expected.append(paths.summary_md)
    return expected


def _completed_pages(root: Path, resume: Classification) -> int:
    """Return the number of transcribed pages a resumable item keeps."""
    if resume.state not in _RESUMABLE or resume.doc_id is None:
        return 0
    reader = read_store(root)
    if reader is None:
        return 0
    units = reader.units(resume.doc_id, kind=KIND_TRANSCRIPTION)
    redo = set(resume.indices(KIND_TRANSCRIPTION))
    return sum(1 for unit in units if unit.status == "done" and unit.idx not in redo)


@dataclass(frozen=True)
class _Classifier:
    """Classifies the items of one plan from the stores of their folders."""

    spec: RunSpec
    job: ItemJob
    image: Mapping[str, Any] | None

    def planned(
        self, item: ItemSpec, output_dir: Path, digest: str | None
    ) -> PlannedItem:
        """Return *item* planned with its resume state; write nothing.

        *digest* is the input's hash (see
        :func:`~autoexcerpter.pipeline.item.input_sha256`).

        Raises:
            StoreRefusedError: The folder's database is refused.
            DuplicateOutputError: The stored document belongs to another input.
        """
        job = self.job
        paths = ItemPaths.for_item(
            item.output_stem, output_dir, job.transcription_format
        )
        request = ResumeRequest(
            outputs=_expected_outputs(paths, job),
            input_path=item.path,
            input_sha256=digest,
            image_settings=self.image,
            model=job.transcription.name,
            summary_model=job.summary.name if job.summarize else None,
            summarize=job.summarize,
            settings_for=functools.partial(_image_record, job),
        )
        state = _classify(output_dir, item.output_stem, request)
        run = self.spec.run
        # --retranscribe --pages works some pages only, so it keeps the
        # stored image values in every state it resets pages from: a
        # document never mixes two image settings.
        keep_images = (
            run.retranscribe and self.spec.input.pages is not None and not run.force
        )
        images: ImageSpec | None = None
        if keep_images and state.state in (ItemState.REFUSED, *_RESUMABLE):
            state, images = self._stored_images(output_dir, item, request, state)
            state = self._kept_images(output_dir, item, request, state, images)
        if run.force and state.state is not ItemState.BLOCKED:
            state = Classification(
                ItemState.NONE, item.output_stem, state.doc_id, reason="overwrite mode"
            )
        elif run.retranscribe and not keep_images and state.state is ItemState.REFUSED:
            state = Classification(
                ItemState.PARTIAL,
                item.output_stem,
                state.doc_id,
                reason="every page is transcribed again",
            )
        return PlannedItem(
            item,
            output_dir,
            state,
            transcription_format=job.transcription_format,
            input_sha256=digest,
            completed_pages=_completed_pages(output_dir, state),
            images=images,
        )

    def _kept_images(
        self,
        output_dir: Path,
        item: ItemSpec,
        request: ResumeRequest,
        state: Classification,
        images: ImageSpec | None,
    ) -> Classification:
        """Refuse a ``--retranscribe --pages`` item whose image settings conflict.

        The item is refused, in any state it resumes from, when the run's
        pinned *images* cannot keep its stored image settings. The comparison
        is the one that refuses an item with transcription work, applied here
        to items without it as well. An item that keeps
        stored transcriptions without an image-settings record is refused
        too: its settings are unknown.
        """
        stored = _stored_record(output_dir, item.output_stem)
        if stored is None:
            if not _stores_transcriptions(output_dir, state):
                return state
            return Classification(
                ItemState.REFUSED,
                item.output_stem,
                state.doc_id,
                reason=(
                    f"The image settings of '{item.output_stem}' are unknown: its "
                    "stored pages have no image-settings record. Skipping it; use "
                    "--force, or --retranscribe without --pages, to replace the "
                    "existing run."
                ),
            )
        job = replace(
            self.job, images=images if images is not None else self.job.images
        )
        check = replace(
            request,
            image_settings=_image_record(job),
            settings_for=functools.partial(_image_record, job),
        )
        changed = image_changes(stored, check)
        if not changed:
            return state
        return Classification(
            ItemState.REFUSED,
            item.output_stem,
            state.doc_id,
            reason=(
                f"Image settings changed: {', '.join(changed)}. Skipping "
                f"'{item.output_stem}'; use --force, or --retranscribe without "
                "--pages, to replace the existing run."
            ),
        )

    def _stored_images(
        self,
        output_dir: Path,
        item: ItemSpec,
        request: ResumeRequest,
        refused: Classification,
    ) -> tuple[Classification, ImageSpec | None]:
        """Resume a ``--retranscribe --pages`` item with its stored image values.

        Each pinnable image option takes the value of the document's stored
        record, with a note naming the options the run set otherwise. The
        item keeps its state *refused* (refused, or the resumable state it
        was classified in) when a value the record does not hold still
        differs.
        """
        stored = _stored_record(output_dir, item.output_stem)
        if stored is None:
            return refused, None
        images = _pinned_images(self.spec.images, stored)
        if images == self.spec.images:
            return refused, None
        job = replace(self.job, images=images)
        retry = replace(
            request,
            image_settings=_image_record(job),
            settings_for=functools.partial(_image_record, job),
        )
        state = _classify(output_dir, item.output_stem, retry)
        if state.state is ItemState.REFUSED:
            return refused, None
        given = [
            name
            for name in _PINNED_KEYS
            if getattr(self.spec.images, name) not in (None, getattr(images, name))
        ]
        if given:
            kept_note = (
                f"'{item.output_stem}': --retranscribe --pages keeps the stored "
                f"image settings ({', '.join(given)}); use --force, or "
                "--retranscribe without --pages, to change them."
            )
            note = f"{state.note} {kept_note}" if state.note else kept_note
            state = replace(state, note=note)
        return state, images


def _classify(root: Path, name: str, request: ResumeRequest) -> Classification:
    """Classify one item, mapping store refusals to plan errors."""
    try:
        return classify(root, name, request)
    except StoreError as exc:
        raise StoreRefusedError(str(exc)) from exc
    except InputMismatchError as exc:
        raise DuplicateOutputError(str(exc)) from exc


def _stores_transcriptions(root: Path, state: Classification) -> bool:
    """Whether the classified document holds a transcribed page."""
    reader = read_store(root) if state.doc_id is not None else None
    if reader is None or state.doc_id is None:
        return False
    units = reader.units(state.doc_id, kind=KIND_TRANSCRIPTION)
    return any(unit.status == "done" for unit in units)


def _stored_record(root: Path, name: str) -> Mapping[str, Any] | None:
    """Return the image-settings record stored with document *name*."""
    reader = read_store(root)
    resume = reader.resume_state(name) if reader is not None else None
    record = resume.image_settings if resume is not None else None
    return record if isinstance(record, Mapping) else None


_PINNED_KEYS = {
    "dpi": "target_dpi",
    "format": "payload_format",
    "jpeg_quality": "jpeg_quality",
    "grayscale": "grayscale_conversion",
}
"""The image options a ``--retranscribe --pages`` item takes from its stored
record, with the record key of each; the detail is resolved per model and
stays the run's."""


def _pinned_images(images: ImageSpec, record: Mapping[str, Any]) -> ImageSpec:
    """Return *images* with each pinnable option that *record* holds from it."""
    values = {
        name: record[key]
        for name, key in _PINNED_KEYS.items()
        if record.get(key) is not None
    }
    return replace(images, **values)


def _plan_warnings(groups: Mapping[ItemState, list[PlannedItem]]) -> tuple[str, ...]:
    return tuple(
        f"Skipping '{planned.name}': {planned.reason}."
        for state in (ItemState.UNSTORED, ItemState.BLOCKED)
        for planned in groups[state]
    )


def plan(
    spec: RunSpec,
    settings: Settings,
    *,
    sources: Mapping[str, Source] | None = None,
    args: argparse.Namespace | None = None,
) -> RunPlan:
    """Discover, select and classify the items of a run; write nothing.

    *sources* maps spec fields to where their values came from; the plan
    reports them with the effective spec. *args* holds the parsed command
    line, if any, for the extension's batch consent and key variables.

    Raises:
        PlanError: The run cannot proceed; the subclass names the reason and
            ``exit_code`` the exit code. A database that is not a current
            store, newer or unreadable raises :class:`StoreRefusedError`; a
            ``--batch`` run
            that batch cannot serve a :class:`ConfigurationError`.
    """
    job = _job(spec, settings)
    if spec.run.batch:
        _check_batch(spec, job, settings, args)
    unset = missing_keys(job, settings, batch=spec.run.batch, args=args)
    key_warnings: tuple[str, ...] = ()
    if unset and spec.run.dry_run:
        key_warnings = tuple(_key_warning(missing) for missing in unset)
    elif unset and spec.run.batch:
        raise ConfigurationError(_key_warning(unset[0], batch=True))
    input_path = _absolute(spec.input.path)
    base = _absolute(spec.output.directory) if spec.output.directory else None
    logger.info("Input=%s, Output=%s", input_path, base or "beside input")

    discovered = discover(input_path)
    if not discovered:
        raise NoItemsError(
            f"No items found to process in: {input_path}. Pass --input with a "
            "PDF, an image folder, or a folder holding them."
        )
    selected, warnings = _select(discovered, spec)
    _guard_duplicate_outputs(selected, base)

    classifier = _Classifier(spec, job, _image_record(job))
    with concurrent.futures.ThreadPoolExecutor(_HASH_WORKERS) as pool:
        digests = list(pool.map(input_sha256, selected))
    groups: dict[ItemState, list[PlannedItem]] = {state: [] for state in ItemState}
    for item, digest in zip(selected, digests, strict=True):
        try:
            planned = classifier.planned(item, _item_output_dir(item, base), digest)
        except PlanError as exc:
            exc.items_total = len(selected)
            raise
        groups[planned.resume.state].append(planned)
    order = {id(item): position for position, item in enumerate(selected)}
    chosen = sorted(
        (
            planned
            for state in (ItemState.NONE, *_RESUMABLE)
            for planned in groups[state]
        ),
        key=lambda planned: order[id(planned.item)],
    )
    to_process = [
        replace(planned, context=_context(planned.item, job)) for planned in chosen
    ]
    skipped = groups[ItemState.COMPLETE]
    logger.info(
        "%d item(s) to process, %d already complete", len(to_process), len(skipped)
    )
    return RunPlan(
        spec=spec,
        job=job,
        input_path=input_path,
        output_dir=base,
        discovered=tuple(discovered),
        to_process=tuple(to_process),
        skipped=tuple(skipped),
        warnings=warnings
        + _plan_warnings(groups)
        + _unselected_warnings(job.pages, skipped)
        + key_warnings,
        settings=settings,
        sources=dict(sources or {}),
        unstored=tuple(groups[ItemState.UNSTORED]),
        blocked=tuple(groups[ItemState.BLOCKED]),
        refused=tuple(groups[ItemState.REFUSED]),
        missing_keys=unset,
    )


@dataclass(frozen=True)
class ItemResult:
    """The outcome of one item.

    *success* is True only when every page the run worked succeeded and
    every configured output was written, or, under ``--batch``, when the item's
    units were submitted (*submitted*). *partial* marks a successful
    ``--pages`` item whose document is still short of complete; it counts
    neither complete nor failed. *outputs* are the absolute paths written;
    *report* holds the page statistics (None when the item stopped before
    them or was batched). *failures* are the failed units of a failed item's
    document, transcriptions and summaries, with their error types and
    errors.
    """

    name: str
    success: bool
    paths: ItemPaths
    outputs: tuple[str, ...] = ()
    report: dict[str, Any] | None = None
    submitted: bool = False
    partial: bool = False
    failures: tuple[UnitRow, ...] = ()


@dataclass(frozen=True)
class RunResult:
    """The outcome of a run.

    *usage* holds the tokens of the run's model calls per role
    (transcription, summary), failed attempts included where the provider
    reported them. *skipped* names the items not processed: complete,
    unstored and blocked ones, and those of which ``--pages`` selects no
    page. *rendered* are the absolute paths of the files of skipped complete
    items that the run rendered again from the store. *databases* are the
    absolute paths of the database files the run opened. *blocked* names the
    skipped items another run's lease held or whose pages wait in a batch,
    and *blocked_reasons* gives the reason of each, in the same order, a
    lease with its expiry; two items of the same name in different folders
    keep their own reasons. *leased* names those of them that another run's
    lease held; they make the run exit 1, while items waiting for a batch
    are submitted work.
    """

    items: tuple[ItemResult, ...]
    skipped: tuple[str, ...] = ()
    seconds: float = 0.0
    usage: Mapping[str, Usage] = field(default_factory=dict)
    databases: tuple[str, ...] = ()
    rendered: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    leased: tuple[str, ...] = ()

    @property
    def total_usage(self) -> Usage:
        """The tokens of all roles together."""
        total = Usage()
        for usage in self.usage.values():
            total = total + usage
        return total

    @property
    def complete(self) -> int:
        """The number of items that finished with every page and output."""
        return sum(
            1
            for item in self.items
            if item.success and not item.submitted and not item.partial
        )

    @property
    def partial(self) -> int:
        """The number of items whose documents ``--pages`` left partial."""
        return sum(1 for item in self.items if item.success and item.partial)

    @property
    def submitted(self) -> int:
        """The number of items whose units wait in batches for ``batch check``."""
        return sum(1 for item in self.items if item.submitted)

    @property
    def failed(self) -> int:
        """The number of items that finished incomplete."""
        return sum(1 for item in self.items if not item.success)

    @property
    def total(self) -> int:
        """The number of items the run processed or skipped."""
        return len(self.items) + len(self.skipped)

    @property
    def exit_code(self) -> ExitCode:
        """1 when an item failed or another run's lease held one, else 0."""
        return exit_code_for(failed=self.failed + len(self.leased))

    @property
    def outputs(self) -> list[str]:
        """The absolute paths written by the run.

        The files rendered again come first, then those of the processed
        items, item by item, then the databases.
        """
        processed = [path for item in self.items for path in item.outputs]
        return [*self.rendered, *processed, *self.databases]


def _writer_options(events: RunEvents) -> WriterOptions:
    """Return the writer options of a run: busy waits warn through *events*."""

    def on_busy(path: Path, waited: float) -> None:
        events.warning(
            f"The database {path} is busy (another program holds it open); "
            f"waited {waited:.0f} s. Results are kept in memory and written "
            "once it is free."
        )

    return WriterOptions(
        busy_timeout=BUSY_TIMEOUT, warn_interval=BUSY_WARN_INTERVAL, on_busy=on_busy
    )


class _Stores:
    """The stores of a run: one per output folder, opened on first use.

    Opening a store stamps the run with its effective spec and settings
    fingerprint and starts the heartbeat that renews the run's leases.
    """

    def __init__(self, run_plan: RunPlan, events: RunEvents) -> None:
        self._plan = run_plan
        self._events = events
        self._stores: dict[Path, Store] = {}
        self._keepers: list[LeaseKeeper] = []

    @property
    def databases(self) -> tuple[str, ...]:
        """The database files of the opened stores, in opening order."""
        return tuple(str(store.path.resolve()) for store in self._stores.values())

    async def get(self, root: Path) -> Store:
        """Return the open store of output folder *root*.

        Raises:
            StoreRefusedError: The folder's database is refused.
        """
        key = root.resolve()
        store = self._stores.get(key)
        if store is not None:
            return store
        key.mkdir(parents=True, exist_ok=True)
        try:
            store = open_store(key, writer=_writer_options(self._events))
            await store.open()
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        self._stores[key] = store
        await begin_run(store, "run", self._plan.effective_spec())
        marked = await store.mark_stale_submissions()
        if marked:
            self._events.warning(
                f"{len(marked)} batch submission(s) in {store.path} were left "
                "without a provider batch id and are now unresolved; their pages "
                "stay blocked. Run 'autoexcerpter batch check' to reconcile them."
            )
        keeper = store.heartbeat(on_lost=self._lost)
        keeper.start()
        self._keepers.append(keeper)
        return store

    def _lost(self, lease: Lease) -> None:
        self._events.warning(
            f"The lease on document {lease.doc_id} expired; its remaining units "
            "are not stored by this run."
        )

    async def close(self, exit_code: int) -> None:
        """Stop the heartbeats, record the end of the run and close the stores."""
        for keeper in self._keepers:
            await keeper.stop()
        for store in self._stores.values():
            try:
                if store.run_id is not None:
                    await store.end_run(exit_code)
            except sqlite3.Error as exc:
                logger.warning("Could not record the run end in %s: %s", store, exc)
            try:
                await store.close()
            except sqlite3.Error as exc:
                logger.warning("Could not close %s: %s", store.path, exc)


async def _note(
    store: Store, doc_id: int | None, message: str, events: RunEvents
) -> None:
    events.warning(message)
    try:
        await store.add_note(message, doc_id=doc_id)
    except sqlite3.Error as exc:
        logger.warning("Could not record a note in %s: %s", store.path, exc)


@dataclass(frozen=True)
class _Blocked:
    """An item not processed.

    Another run leased its document after planning, or (*leased* False) a
    batch submitted since the plan holds its pages.
    """

    reason: str
    leased: bool = True


@dataclass
class _ItemRun:
    """One item of a run: its store document and lease.

    *held* is the reason the item could not be leased, once :meth:`lease`
    found another run holding it.
    """

    run_plan: RunPlan
    planned: PlannedItem
    stores: _Stores
    events: RunEvents
    held: str | None = None

    async def lease(self) -> tuple[Store, int, Lease] | None:
        """Ensure the item's document and lease it; None when another holds it.

        The input hash is recorded later, under the lease (:meth:`reset`).
        """
        planned = self.planned
        store = await self.stores.get(planned.output_dir)
        doc_id = await store.ensure_document(
            planned.name,
            input_path=planned.item.path,
            kind=document_kind(planned.item.kind),
        )
        lease = await store.acquire(doc_id)
        if lease is None:
            state = None
            with contextlib.suppress(StoreError, sqlite3.Error):
                state = store.resume_state(doc_id)
            self.held = lease_reason(
                state.lease_run_id if state is not None else None,
                state.lease_until if state is not None else None,
            )
            await _note(
                store,
                doc_id,
                f"Skipping '{planned.name}': another run holds its document in "
                f"{store.path} ({self.held}).",
                self.events,
            )
            return None
        return store, doc_id, lease

    def blocked(self) -> _Blocked:
        """Return the item as blocked by the lease :meth:`lease` found."""
        return _Blocked(self.held or "another run holds its document")

    def handle(self, store: Store, doc_id: int, lease: Lease) -> DocumentHandle:
        """Return the item's document handle with the run's model stamps."""
        job = self.run_plan.job
        summary = (
            Stamp(job.summary.name, job.summary.provider) if job.summarize else None
        )
        return DocumentHandle(
            store,
            doc_id,
            self.planned.name,
            lease,
            transcription=Stamp(job.transcription.name, job.transcription.provider),
            summary=summary,
        )

    async def reset(self, handle: DocumentHandle) -> None:
        """Reset the units this run works again.

        These are all units under ``--force`` or after an input change, and
        the transcriptions under ``--retranscribe`` (those of the pages
        ``--pages`` selects, when given). A reset transcription marks its
        page's summary stale. The input hash is recorded after the reset, so
        that an interruption between the two leaves the change detected, and
        the document's files are marked stale: they are rendered again once
        its units are stored. The document's batch job of an earlier ``run
        --batch`` ends, so ``batch check`` leaves the document to this run.
        """
        planned, spec = self.planned, self.run_plan.spec
        if spec.run.force or planned.resume.reset:
            await handle.reset()
        elif spec.run.retranscribe and planned.resume.state in _RESUMABLE:
            pages = self.run_plan.job.pages
            idxs = None
            if pages is not None:
                idxs = pages.resolve(len(handle.units(KIND_TRANSCRIPTION)))
            await handle.reset(KIND_TRANSCRIPTION, idxs)
        digest, doc_id = planned.input_sha256, handle.doc_id

        def prepare(connection: sqlite3.Connection) -> None:
            if digest is not None:
                connection.execute(
                    "UPDATE documents SET input_sha256 = ? WHERE doc_id = ?",
                    (digest, doc_id),
                )
            connection.execute(
                "UPDATE outputs SET stale = 1 WHERE doc_id = ?", (doc_id,)
            )
            drop_batch_job(connection, doc_id)

        await handle.store.write(handle.lease, prepare)


def _item_job(run_plan: RunPlan, planned: PlannedItem) -> ItemJob:
    """Return the run's item job with the item's own image options, if any."""
    if planned.images is None:
        return run_plan.job
    return replace(run_plan.job, images=planned.images)


def _page_count(planned: PlannedItem, job: ItemJob) -> int | None:
    """Return the item's page count; None when its input cannot be opened."""
    try:
        source = open_source(planned.item, job)
    except Exception:  # noqa: BLE001 - the item fails when it is processed
        return None
    try:
        return len(source)
    finally:
        source.close()


async def _beyond_pages(run_plan: RunPlan, planned: PlannedItem) -> str | None:
    """Return why ``--pages`` selects none of the item's pages, else None.

    An input that cannot be opened or holds no page is left to the item,
    which fails it.
    """
    pages = run_plan.job.pages
    if pages is None:
        return None
    count = await asyncio.to_thread(_page_count, planned, _item_job(run_plan, planned))
    if not count or pages.resolve(count):
        return None
    noun = "pages" if planned.item.kind == "pdf" else "images"
    return (
        f"Page range '{pages.describe()}' yielded no {noun} for '{planned.name}' "
        f"({count} {noun}). Skipping."
    )


def _left_partial(run_plan: RunPlan, store: Store, doc_id: int) -> bool:
    """Whether a ``--pages`` run leaves the document short of complete."""
    if run_plan.job.pages is None:
        return False
    document = store.document(doc_id)
    return document is not None and document.status != "complete"


def _unselected_warnings(
    pages: PageRange | None, skipped: Sequence[PlannedItem]
) -> tuple[str, ...]:
    """Return a warning for each complete item of which *pages* selects no page."""
    if pages is None:
        return ()
    warnings = []
    for planned in skipped:
        reader = read_store(planned.output_dir)
        doc_id = planned.resume.doc_id
        if reader is None or doc_id is None:
            continue
        count = len(reader.units(doc_id, kind=KIND_TRANSCRIPTION))
        if count and not pages.resolve(count):
            noun = "pages" if planned.item.kind == "pdf" else "images"
            warnings.append(
                f"Page range '{pages.describe()}' yielded no {noun} for "
                f"'{planned.name}' ({count} {noun}), which is complete. Skipping."
            )
    return tuple(warnings)


async def _submit_item(
    run_plan: RunPlan,
    planned: PlannedItem,
    handle: DocumentHandle,
    events: RunEvents,
    clients: BatchClients,
) -> BatchItemOutcome:
    """Submit one leased item's missing units to the batch APIs."""
    outcome = await BatchItem(
        planned.item,
        planned.output_dir,
        _item_job(run_plan, planned),
        handle,
        context=planned.context,
        clients=clients,
        events=events,
        force=run_plan.force,
    ).process()
    if outcome.detail:
        await _note(
            handle.store, handle.doc_id, f"'{planned.name}': {outcome.detail}", events
        )
    if outcome.status == "submitted":
        logger.info(
            "Submitted '%s' in %d batch(es); run 'autoexcerpter batch check' to "
            "collect them",
            planned.name,
            len(outcome.submission_ids),
        )
    return outcome


def _batch_result(  # noqa: PLR0913 - the item, its place, its outcome
    planned: PlannedItem,
    index: int,
    total: int,
    events: RunEvents,
    outcome: BatchItemOutcome,
    left_partial: Callable[[], bool],
    failed_units: Callable[[], tuple[UnitRow, ...]],
) -> ItemResult:
    """Report a batch item's end and return its result.

    *left_partial* says whether a successful item that submitted nothing is
    still short of complete; *failed_units* returns the failed units of a
    failed item's document.
    """
    outputs = tuple(str(path) for path in outcome.outputs)
    success = outcome.success
    submitted = outcome.status == "submitted"
    partial = success and not submitted and left_partial()
    events.item_finished(
        ItemFinished(
            planned.name,
            index,
            total,
            success,
            outputs,
            None,
            submitted=submitted,
            partial=partial,
        )
    )
    return ItemResult(
        planned.name,
        success,
        planned.paths,
        outputs,
        None,
        submitted=submitted,
        partial=partial,
        failures=() if success else failed_units(),
    )


async def _run_item(  # noqa: PLR0913 - the item, its place and the run's objects
    run_plan: RunPlan,
    planned: PlannedItem,
    index: int,
    env: CallEnv,
    stores: _Stores,
    clients: BatchClients | None = None,
) -> ItemResult | _Blocked | None:
    """Process one item; :class:`_Blocked` when another run holds its document.

    An item whose pages a batch submission took since the plan is blocked
    as well, though not by a lease; an item of which ``--pages`` selects no
    page returns None, before ``--force`` resets it. With *clients*, the
    item's missing units are submitted to the batch APIs instead of called.
    """
    events = env.events
    total = len(run_plan.to_process)
    name = planned.name
    item_run = _ItemRun(run_plan, planned, stores, events)
    leased = await item_run.lease()
    if leased is None:
        return item_run.blocked()
    store, doc_id, lease = leased
    held: _Blocked | None = None
    if store.blocked_units(doc_id):
        held = _Blocked("its pages wait in a batch submitted since the plan", False)
        skip: str | None = (
            f"Skipping '{name}': its pages wait in a batch submitted since the "
            "plan; run 'autoexcerpter batch check'."
        )
    else:
        skip = await _beyond_pages(run_plan, planned)
    if skip is not None:
        await _note(store, doc_id, skip, events)
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
        return held
    events.item_started(ItemStarted(name, planned.item.kind, index, total))
    processor: ItemProcessor | None = None
    success = False
    try:
        if planned.resume.note:
            await _note(store, doc_id, planned.resume.note, events)
        handle = item_run.handle(store, doc_id, lease)
        await item_run.reset(handle)
        if clients is not None:
            batched = await _submit_item(run_plan, planned, handle, events, clients)
            return _batch_result(
                planned,
                index,
                total,
                events,
                batched,
                lambda: _left_partial(run_plan, store, doc_id),
                lambda: _failed_units(store, doc_id),
            )
        processor = ItemProcessor(
            planned.item,
            planned.output_dir,
            _item_job(run_plan, planned),
            handle,
            context=planned.context,
            events=events,
            env=env,
            force=run_plan.force,
        )
        success = await processor.process()
        if not success:
            logger.error(
                "Item finished with page-level failures or missing outputs: %s",
                name,
            )
    except LeaseLostError as exc:
        await _note(store, doc_id, f"'{name}': its lease was lost: {exc}", events)
    except Exception as exc:
        logger.exception("Critical error in processing item '%s': %s", name, exc)
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)

    outputs = (
        tuple(str(path) for path in processor.written_outputs) if processor else ()
    )
    report = processor.last_run_report if processor else None
    partial = success and _left_partial(run_plan, store, doc_id)
    events.item_finished(
        ItemFinished(name, index, total, success, outputs, report, partial=partial)
    )
    return ItemResult(
        name,
        success,
        planned.paths,
        outputs,
        report,
        partial=partial,
        failures=() if success else _failed_units(store, doc_id),
    )


def _failed_units(store: Store, doc_id: int) -> tuple[UnitRow, ...]:
    """Return the failed units of the document *doc_id*; none when unreadable."""
    with contextlib.suppress(StoreError, sqlite3.Error):
        return tuple(store.failed_units(doc_id=doc_id))
    return ()


async def _render_only(
    run_plan: RunPlan, planned: PlannedItem, env: CallEnv, stores: _Stores
) -> ItemResult | _Blocked:
    """Write a complete item's missing or stale files from the store.

    No model and no OpenAlex request is made. A document without stored
    citations, as a run interrupted during their enrichment leaves it, gets
    them consolidated first, with metadata from the OpenAlex cache alone.
    Returns the item's result with the files that hold its rendering, failed
    when a file could not be written; :class:`_Blocked` when another run
    holds its document.
    """
    item_run = _ItemRun(run_plan, planned, stores, env.events)
    leased = await item_run.lease()
    if leased is None:
        return item_run.blocked()
    store, doc_id, lease = leased
    ok = False
    files: ItemFiles | None = None
    try:
        files = ItemFiles(
            planned.item,
            planned.output_dir,
            run_plan.job,
            item_run.handle(store, doc_id, lease),
            events=env.events,
        )
        if files.has_citations() or await files.store_citations(offline=True):
            status = await files.write()
            ok = status.transcription_ok and status.summaries_ok
    except (LeaseLostError, OSError) as exc:
        await _note(
            store, doc_id, f"'{planned.name}' was not rendered: {exc}", env.events
        )
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)
    outputs = tuple(str(path) for path in files.written) if files else ()
    return ItemResult(planned.name, ok, planned.paths, outputs)


@dataclass
class _Held:
    """The blocked items of a run, their reasons and those leased elsewhere.

    It starts from the plan's blocked items; items held at run time are
    added in the order the run meets them. *reasons* holds the reason of
    each blocked item, in the order of *blocked*.
    """

    blocked: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    leased: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, run_plan: RunPlan) -> _Held:
        held = cls(leased=[item.name for item in run_plan.leased])
        for item in run_plan.blocked:
            held.blocked.append(item.name)
            held.reasons.append(item.reason)
        return held

    def add(self, planned: PlannedItem, outcome: _Blocked) -> None:
        """Record *planned* as blocked for *outcome*'s reason."""
        self.blocked.append(planned.name)
        self.reasons.append(outcome.reason)
        if outcome.leased:
            self.leased.append(planned.name)


def idle_result(run_plan: RunPlan) -> RunResult:
    """Return the result of a plan with nothing to process, render or refuse.

    Every item is skipped; the plan's blocked items keep their reasons, and
    one another run leases makes the result exit 1.
    """
    held = _Held.of(run_plan)
    return RunResult(
        items=(),
        skipped=tuple(item.name for item in run_plan.not_processed),
        blocked=tuple(held.blocked),
        blocked_reasons=tuple(held.reasons),
        leased=tuple(held.leased),
    )


async def _render_complete(
    run_plan: RunPlan, env: CallEnv, stores: _Stores, held: _Held
) -> tuple[list[str], list[ItemResult]]:
    """Render the plan's complete items with missing or stale files.

    Returns the files of the items rendered and the results of those whose
    files could not all be written; an item another run holds goes to
    *held* and stays skipped.
    """
    rendered: list[str] = []
    failures: list[ItemResult] = []
    for planned in run_plan.render_only:
        result = await _render_only(run_plan, planned, env, stores)
        if isinstance(result, _Blocked):
            held.add(planned, result)
            continue
        if result.success:
            rendered.extend(result.outputs)
        else:
            failures.append(result)
    return rendered, failures


def _check_clients(run_plan: RunPlan, env: CallEnv) -> None:
    """Build the caller of every phase the run calls, before any item starts.

    The callers build their clients as the items do, so the attempt identity
    decides which key variable applies; the first item uses them.

    Raises:
        ConfigurationError: A key variable is unset or the settings cannot
            serve a phase.
    """
    job = run_plan.job
    phases = [("transcription", job.transcription)]
    if job.summarize:
        phases.append(("summary", job.summary))
    for role, phase in phases:
        try:
            env.prepare(role, phase)
        except (OSError, SettingsError) as exc:
            raise ConfigurationError(
                f"Cannot build the {role} model client: {exc}",
                items_total=len(run_plan.to_process),
            ) from exc


def _clients(
    run_plan: RunPlan, env: CallEnv, args: argparse.Namespace | None
) -> BatchClients | None:
    """Return the run's batch clients under ``--batch``; else check the callers.

    Raises:
        ConfigurationError: A phase's caller cannot be built.
    """
    if run_plan.batch:
        return BatchClients(run_plan.settings, "batch", args=args)
    if run_plan.to_process:
        _check_clients(run_plan, env)
    return None


async def run(
    run_plan: RunPlan,
    events: RunEvents | None = None,
    *,
    args: argparse.Namespace | None = None,
) -> RunResult:
    """Process the plan's items one after another on the running event loop.

    The run owns one rate-limiter registry and one usage total, shared by
    every item, and one store per output folder, stamped with the run.
    Before any item starts, the client of every phase the run calls is
    built once; the clients are closed when the run ends. Each item's
    document is leased while its units are written and its files written
    from the store; complete items with missing files are written without
    model calls. An item whose document another run holds, also a complete
    one to render again, is skipped with a note and counts as blocked; the
    run records exit code 1 for it like a failure. An item whose pages wait
    in a batch is blocked too, without changing the exit code. An item that
    crashes is logged and counted failed; the run
    continues with the next item. Refused items count as failed.
    Cancellation propagates. Under ``--batch`` no model client is built:
    each item's missing units are submitted through one set of batch
    clients, and *args*, the parsed command line if any, serves the
    extension's key variables.

    Raises:
        ValueError: The plan is a dry run.
        ConfigurationError: A phase's client cannot be built because its key
            variable is unset or the settings cannot serve it; nothing has
            been written. :class:`StoreRefusedError` when a database became
            unusable after planning.
    """
    if run_plan.dry_run:
        raise ValueError("a dry-run plan is reported, not run")
    sink: RunEvents = events if events is not None else NullEvents()
    env = CallEnv.create(run_plan.settings, sink)
    clients = _clients(run_plan, env, args)
    if run_plan.output_dir is not None and (
        run_plan.to_process or run_plan.render_only
    ):
        run_plan.output_dir.mkdir(parents=True, exist_ok=True)
    stores = _Stores(run_plan, sink)
    started = time.monotonic()
    skipped = [item.name for item in run_plan.not_processed]
    held = _Held.of(run_plan)
    exit_code: int = ExitCode.FAILURE
    try:
        rendered, results = await _render_complete(run_plan, env, stores, held)
        for failure in results:
            skipped.remove(failure.name)
        for index, planned in enumerate(run_plan.to_process, start=1):
            result = await _run_item(run_plan, planned, index, env, stores, clients)
            if isinstance(result, ItemResult):
                results.append(result)
                continue
            skipped.append(planned.name)
            if result is not None:
                held.add(planned, result)
        for planned in run_plan.refused:
            sink.warning(planned.reason)
            results.append(ItemResult(planned.name, False, planned.paths))
        failed = any(not result.success for result in results)
        exit_code = ExitCode.FAILURE if failed or held.leased else ExitCode.OK
    except (asyncio.CancelledError, KeyboardInterrupt):
        exit_code = ExitCode.INTERRUPTED
        raise
    finally:
        try:
            await stores.close(exit_code)
        finally:
            await env.aclose()
    return RunResult(
        items=tuple(results),
        skipped=tuple(skipped),
        seconds=time.monotonic() - started,
        usage={role: env.usage.get(role) for role in env.usage.roles},
        databases=stores.databases,
        rendered=tuple(rendered),
        blocked=tuple(held.blocked),
        blocked_reasons=tuple(held.reasons),
        leased=tuple(held.leased),
    )
