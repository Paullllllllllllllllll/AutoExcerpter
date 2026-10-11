"""Core entry points: discover items, plan a run, run it.

:func:`plan` turns a resolved :class:`~autoexcerpter.spec.RunSpec` and the
machine settings into a :class:`RunPlan` without API calls or writes;
:func:`run` processes the plan's items on one event loop and reports
progress through :class:`~autoexcerpter.events.RunEvents`. Neither prints,
prompts or exits: a plan that cannot run raises a :class:`PlanError` that
carries its exit code. Cancelling the task that awaits :func:`run` stops the
run, retry waits included.

Every output folder holds one ``autoexcerpter.sqlite``, the store of its
documents (see :mod:`autoexcerpter.output.store`). Resume reads only the
store: an item whose rendered output or version 1.0 working logs exist
without units in the store is ``legacy`` and skipped, an item leased by
another live run is ``blocked`` and skipped, and an item whose stored pages
used other image settings is ``refused`` and counts failed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from autoexcerpter.common.contract import ExitCode
from autoexcerpter.common.options import Source
from autoexcerpter.common.selection import select_items
from autoexcerpter.common.sqlite import DEFAULT_BUSY_TIMEOUT, DEFAULT_WARN_INTERVAL
from autoexcerpter.common.store import (
    Lease,
    LeaseKeeper,
    LeaseLostError,
    Store,
    StoreError,
    VersionlessDatabaseError,
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
    image_settings_record,
    open_store,
    read_store,
)
from autoexcerpter.pipeline.context import ContextError, summary_context
from autoexcerpter.pipeline.item import ItemFiles, ItemProcessor, input_sha256
from autoexcerpter.pipeline.job import ItemJob, job_from_spec
from autoexcerpter.pipeline.scanner import scan_input_path
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.pipeline.units import DocumentHandle, Stamp
from autoexcerpter.settings import Settings, SettingsError
from autoexcerpter.spec import CONTEXT_NONE, RunSpec, effective_spec

__all__ = [
    "BUSY_TIMEOUT",
    "BUSY_WARN_INTERVAL",
    "AmbiguousSelectionError",
    "ConfigurationError",
    "DuplicateOutputError",
    "ItemResult",
    "NoItemsError",
    "PlanError",
    "PlannedItem",
    "RunPlan",
    "RunResult",
    "SelectionError",
    "StoreRefusedError",
    "discover",
    "plan",
    "run",
]

logger = logging.getLogger(__name__)

BUSY_TIMEOUT: float = DEFAULT_BUSY_TIMEOUT
"""Seconds a store write or read waits on a busy database before it backs off."""

BUSY_WARN_INTERVAL: float = DEFAULT_WARN_INTERVAL
"""Seconds between two warnings about the same busy database."""

_RESUMABLE = (ItemState.PARTIAL, ItemState.TRANSCRIPTION_ONLY)


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
    """An output folder's database is versionless, newer or unreadable (exit 2).

    The database is left as it is; ``migrate`` converts a versionless one.
    """


class NoItemsError(PlanError):
    """The input holds no PDF or image folder (exit 1)."""

    exit_code = ExitCode.FAILURE


class SelectionError(PlanError):
    """``--select`` matched no item (exit 1)."""

    exit_code = ExitCode.FAILURE


class AmbiguousSelectionError(PlanError):
    """Several items were found without ``--all`` or ``--select`` (exit 2)."""


class DuplicateOutputError(PlanError):
    """Two items, selected now or stored earlier, write the same outputs (exit 2)."""


@dataclass(frozen=True)
class PlannedItem:
    """One selected item: where it writes and what the store holds for it.

    *resume* is the item's classification from the store of *output_dir*;
    *completed_pages* counts the pages an earlier run transcribed that this
    run keeps. *context* is the item's summary context as it enters the
    prompt; *transcription_format* the transcription file's extension;
    *input_sha256* the hash of the input at planning.
    """

    item: ItemSpec
    output_dir: Path
    resume: Classification
    context: str | None = None
    transcription_format: str = "md"
    input_sha256: str | None = None
    completed_pages: int = 0

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
        """The resume state: none, partial, transcription_only, complete,
        legacy, blocked or refused."""
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
    *legacy* items have outputs without store rows, *blocked* items are
    leased by another run, *refused* items were stored with other image
    settings; none of them is processed. *warnings* are notes for the user,
    such as ignored ``--select`` parts. *settings* are the machine settings
    the run uses. *sources* maps spec fields to where their values came
    from; fields it omits were given by the caller and count as flags.
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
    legacy: tuple[PlannedItem, ...] = ()
    blocked: tuple[PlannedItem, ...] = ()
    refused: tuple[PlannedItem, ...] = ()

    @property
    def force(self) -> bool:
        """Whether existing outputs are reprocessed instead of resumed."""
        return self.spec.run.force

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        return self.spec.run.dry_run

    @property
    def render_only(self) -> tuple[PlannedItem, ...]:
        """The complete items whose files are written again without model calls."""
        return tuple(item for item in self.skipped if item.resume.render)

    @property
    def not_processed(self) -> tuple[PlannedItem, ...]:
        """The items the run leaves alone: complete, legacy and blocked ones."""
        return (*self.skipped, *self.legacy, *self.blocked)

    @property
    def has_work(self) -> bool:
        """Whether a run of the plan processes, renders or refuses an item."""
        return bool(self.to_process or self.refused or self.render_only)

    def effective_spec(self) -> dict[str, dict[str, Any]]:
        """Return ``{field: {"value": ..., "source": ...}}`` for every field."""
        return effective_spec(self.spec, self.sources)


def _absolute(path: Path) -> Path:
    """Expand ``~`` and make *path* absolute against the current directory."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def discover(input_path: Path) -> list[ItemSpec]:
    """Return the PDFs and image folders found at *input_path*."""
    return scan_input_path(_absolute(input_path))


def _item_output_dir(item: ItemSpec, base: Path | None) -> Path:
    return item.path.parent if base is None else base


def _select(
    items: list[ItemSpec], spec: RunSpec
) -> tuple[list[ItemSpec], tuple[str, ...]]:
    """Apply ``--all`` or ``--select``; a single item needs neither."""
    selection = spec.input
    if selection.all:
        return list(items), ()
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
        f"{len(items)} items found but neither --all nor --select was given. "
        "Refusing to silently process only the first item; pass --all to "
        "process every item or --select to choose specific ones."
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


def _context(item: ItemSpec, job: ItemJob) -> str | None:
    if not job.summarize or job.context == CONTEXT_NONE:
        return None
    try:
        return summary_context(item.path, job.context, default=job.fallback_context)
    except ContextError as exc:
        raise ConfigurationError(str(exc)) from exc


def _image_record(job: ItemJob) -> dict[str, Any] | None:
    """Return the run's image-settings record; None when it cannot be resolved.

    An image setting that cannot be resolved fails the item when its source
    opens.
    """
    model = job.transcription
    try:
        settings, _, _ = resolve_image_settings(model.provider, model.name, job.images)
    except ValueError:
        return None
    return image_settings_record(settings)


def _refused(root: Path, exc: StoreError) -> StoreRefusedError:
    """Return the plan error of a database the store refuses."""
    message = str(exc)
    if isinstance(exc, VersionlessDatabaseError):
        message += (
            f' Run: autoexcerpter migrate --input "{root}" to convert it; '
            "the database is left as it is until then."
        )
    return StoreRefusedError(message)


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

    def planned(self, item: ItemSpec, output_dir: Path) -> PlannedItem:
        """Return *item* planned with its resume state; write nothing.

        Raises:
            StoreRefusedError: The folder's database is refused.
            DuplicateOutputError: The stored document belongs to another input.
        """
        job = self.job
        paths = ItemPaths.for_item(
            item.output_stem, output_dir, job.transcription_format
        )
        digest = input_sha256(item)
        request = ResumeRequest(
            outputs=_expected_outputs(paths, job),
            legacy=paths.legacy_logs(),
            input_path=item.path,
            input_sha256=digest,
            image_settings=self.image,
            model=job.transcription.name,
            summary_model=job.summary.name if job.summarize else None,
            summarize=job.summarize,
        )
        try:
            state = classify(output_dir, item.output_stem, request)
        except StoreError as exc:
            raise _refused(output_dir, exc) from exc
        except InputMismatchError as exc:
            raise DuplicateOutputError(str(exc)) from exc
        run = self.spec.run
        if run.force and state.state is not ItemState.BLOCKED:
            state = Classification(
                ItemState.NONE, item.output_stem, state.doc_id, reason="overwrite mode"
            )
        elif run.retranscribe and state.state is ItemState.REFUSED:
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
        )


def _plan_warnings(groups: Mapping[ItemState, list[PlannedItem]]) -> tuple[str, ...]:
    warnings = [
        f"Skipping '{planned.name}': {planned.reason}."
        for state in (ItemState.LEGACY, ItemState.BLOCKED)
        for planned in groups[state]
    ]
    return tuple(warnings)


def plan(
    spec: RunSpec,
    settings: Settings,
    *,
    sources: Mapping[str, Source] | None = None,
) -> RunPlan:
    """Discover, select and classify the items of a run; write nothing.

    *sources* maps spec fields to where their values came from; the plan
    reports them with the effective spec.

    Raises:
        PlanError: The run cannot proceed; the subclass names the reason and
            ``exit_code`` the exit code. A versionless, newer or unreadable
            database raises :class:`StoreRefusedError`.
    """
    job = _job(spec, settings)
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
    groups: dict[ItemState, list[PlannedItem]] = {state: [] for state in ItemState}
    for item in selected:
        try:
            planned = classifier.planned(item, _item_output_dir(item, base))
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
        warnings=warnings + _plan_warnings(groups),
        settings=settings,
        sources=dict(sources or {}),
        legacy=tuple(groups[ItemState.LEGACY]),
        blocked=tuple(groups[ItemState.BLOCKED]),
        refused=tuple(groups[ItemState.REFUSED]),
    )


@dataclass(frozen=True)
class ItemResult:
    """The outcome of one item.

    *success* is True only when every page succeeded and every configured
    output was written. *outputs* are the absolute paths written; *report*
    holds the page statistics (None when the item stopped before them).
    """

    name: str
    success: bool
    paths: ItemPaths
    outputs: tuple[str, ...] = ()
    report: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunResult:
    """The outcome of a run.

    *usage* holds the tokens of the run's model calls per role
    (transcription, summary), failed attempts included where the provider
    reported them. *skipped* names the items not processed: complete,
    legacy and blocked ones. *databases* are the absolute paths of the
    database files the run opened.
    """

    items: tuple[ItemResult, ...]
    skipped: tuple[str, ...] = ()
    seconds: float = 0.0
    usage: Mapping[str, Usage] = field(default_factory=dict)
    databases: tuple[str, ...] = ()

    @property
    def total_usage(self) -> Usage:
        """The tokens of all roles together."""
        total = Usage()
        for usage in self.usage.values():
            total = total + usage
        return total

    @property
    def complete(self) -> int:
        """The number of items that finished complete."""
        return sum(1 for item in self.items if item.success)

    @property
    def failed(self) -> int:
        """The number of items that finished incomplete."""
        return len(self.items) - self.complete

    @property
    def outputs(self) -> list[str]:
        """The absolute paths written by the run, item by item, then databases."""
        return [path for item in self.items for path in item.outputs] + list(
            self.databases
        )


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
        store = open_store(key, writer=_writer_options(self._events))
        try:
            await store.open()
        except StoreError as exc:
            raise _refused(key, exc) from exc
        self._stores[key] = store
        await begin_run(store, "run", self._plan.effective_spec())
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


@dataclass
class _ItemRun:
    """One item of a run: its store document and lease."""

    run_plan: RunPlan
    planned: PlannedItem
    stores: _Stores
    events: RunEvents

    async def lease(self) -> tuple[Store, int, Lease] | None:
        """Ensure the item's document and lease it; None when another holds it."""
        planned = self.planned
        store = await self.stores.get(planned.output_dir)
        doc_id = await store.ensure_document(
            planned.name,
            input_path=planned.item.path,
            input_sha256=planned.input_sha256,
            kind=document_kind(planned.item.kind),
        )
        lease = await store.acquire(doc_id)
        if lease is None:
            await _note(
                store,
                doc_id,
                f"Skipping '{planned.name}': another run holds its document in "
                f"{store.path}.",
                self.events,
            )
            return None
        return store, doc_id, lease

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
        """Reset the units this run works again: all of them under ``--force``
        or after an input change, the transcriptions under ``--retranscribe``.

        A reset transcription marks its page's summary stale.
        """
        planned, spec = self.planned, self.run_plan.spec
        if spec.run.force or planned.resume.reset:
            await handle.reset()
        elif spec.run.retranscribe and planned.resume.state in _RESUMABLE:
            await handle.reset(KIND_TRANSCRIPTION)


async def _run_item(
    run_plan: RunPlan,
    planned: PlannedItem,
    index: int,
    env: CallEnv,
    stores: _Stores,
) -> ItemResult | None:
    """Process one item; None when another run holds its document."""
    events = env.events
    total = len(run_plan.to_process)
    name = planned.name
    item_run = _ItemRun(run_plan, planned, stores, events)
    leased = await item_run.lease()
    if leased is None:
        return None
    store, doc_id, lease = leased
    events.item_started(ItemStarted(name, planned.item.kind, index, total))
    processor: ItemProcessor | None = None
    success = False
    try:
        if planned.resume.note:
            await _note(store, doc_id, planned.resume.note, events)
        handle = item_run.handle(store, doc_id, lease)
        await item_run.reset(handle)
        processor = ItemProcessor(
            planned.item,
            planned.output_dir,
            run_plan.job,
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
    events.item_finished(ItemFinished(name, index, total, success, outputs, report))
    return ItemResult(name, success, planned.paths, outputs, report)


async def _render_only(
    run_plan: RunPlan, planned: PlannedItem, env: CallEnv, stores: _Stores
) -> None:
    """Write a complete item's missing files from the store; no model call."""
    item_run = _ItemRun(run_plan, planned, stores, env.events)
    leased = await item_run.lease()
    if leased is None:
        return
    store, doc_id, lease = leased
    try:
        files = ItemFiles(
            planned.item,
            planned.output_dir,
            run_plan.job,
            item_run.handle(store, doc_id, lease),
            events=env.events,
        )
        await files.write()
    except (LeaseLostError, OSError) as exc:
        await _note(
            store, doc_id, f"'{planned.name}' was not rendered: {exc}", env.events
        )
    finally:
        with contextlib.suppress(sqlite3.Error):
            await store.release(lease)


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


async def run(run_plan: RunPlan, events: RunEvents | None = None) -> RunResult:
    """Process the plan's items one after another on the running event loop.

    The run owns one rate-limiter registry and one usage total, shared by
    every item, and one store per output folder, stamped with the run.
    Before any item starts, the client of every phase the run calls is
    built once; the clients are closed when the run ends. Each item's
    document is leased while its units are written and its files written
    from the store; complete items with missing files are written without
    model calls. An item whose document another run holds is skipped with a
    note. An item that crashes is logged and counted failed; the run
    continues with the next item. Refused items count as failed.
    Cancellation propagates.

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
    if run_plan.to_process:
        _check_clients(run_plan, env)
    if run_plan.output_dir is not None and (
        run_plan.to_process or run_plan.render_only
    ):
        run_plan.output_dir.mkdir(parents=True, exist_ok=True)
    stores = _Stores(run_plan, sink)
    started = time.monotonic()
    results: list[ItemResult] = []
    skipped = [item.name for item in run_plan.not_processed]
    exit_code: int = ExitCode.FAILURE
    try:
        for planned in run_plan.render_only:
            await _render_only(run_plan, planned, env, stores)
        for index, planned in enumerate(run_plan.to_process, start=1):
            result = await _run_item(run_plan, planned, index, env, stores)
            if result is None:
                skipped.append(planned.name)
            else:
                results.append(result)
        for planned in run_plan.refused:
            sink.warning(planned.reason)
            results.append(ItemResult(planned.name, False, planned.paths))
        failed = any(not result.success for result in results)
        exit_code = ExitCode.FAILURE if failed else ExitCode.OK
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
    )
