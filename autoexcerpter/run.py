"""Core entry points: discover items, plan a run, run it.

:func:`plan` turns a resolved :class:`~autoexcerpter.spec.RunSpec` and the
machine settings into a :class:`RunPlan` without API calls or writes;
:func:`run` processes the plan's items on one event loop and reports
progress through :class:`~autoexcerpter.events.RunEvents`. Neither prints,
prompts or exits: a plan that cannot run raises a :class:`PlanError` that
carries its exit code. Cancelling the task that awaits :func:`run` stops the
run, retry waits included.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoexcerpter.common.contract import ExitCode
from autoexcerpter.common.options import Source
from autoexcerpter.common.selection import select_items
from autoexcerpter.common.sqlite import SqliteWriter
from autoexcerpter.common.usage import Usage
from autoexcerpter.events import (
    ItemFinished,
    ItemStarted,
    NullEvents,
    RunEvents,
)
from autoexcerpter.llm import CallEnv
from autoexcerpter.pipeline.context import ContextError, summary_context
from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.job import ItemJob, ItemResume, job_from_spec
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.resume import (
    InputMismatchError,
    ProcessingState,
    ResumeChecker,
    ResumeResult,
)
from autoexcerpter.pipeline.scanner import scan_input_path
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.rendering.sqlite import (
    DocumentRecord,
    RunInfo,
    open_database,
    write_document,
)
from autoexcerpter.settings import Settings, SettingsError
from autoexcerpter.spec import CONTEXT_NONE, RunSpec, effective_spec

__all__ = [
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
    "discover",
    "plan",
    "run",
]

logger = logging.getLogger(__name__)

_RESUMABLE = (ProcessingState.PARTIAL, ProcessingState.TRANSCRIPTION_ONLY)


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


class NoItemsError(PlanError):
    """The input holds no PDF or image folder (exit 1)."""

    exit_code = ExitCode.FAILURE


class SelectionError(PlanError):
    """``--select`` matched no item (exit 1)."""

    exit_code = ExitCode.FAILURE


class AmbiguousSelectionError(PlanError):
    """Several items were found without ``--all`` or ``--select`` (exit 2)."""


class DuplicateOutputError(PlanError):
    """Two items, selected now or logged earlier, write the same outputs (exit 2)."""


@dataclass(frozen=True)
class PlannedItem:
    """One selected item: where it writes and what the resume check found.

    *context* is the item's summary context as it enters the prompt;
    *transcription_format* the transcription file's extension.
    """

    item: ItemSpec
    output_dir: Path
    resume: ResumeResult
    context: str | None = None
    transcription_format: str = "md"

    @property
    def name(self) -> str:
        """The item's output stem."""
        return self.item.output_stem

    @property
    def paths(self) -> ItemPaths:
        """The item's output and working paths."""
        return ItemPaths.for_item(self.name, self.output_dir, self.transcription_format)

    @property
    def resumable(self) -> bool:
        """Whether completed pages from an earlier run are reused."""
        return self.resume.state in _RESUMABLE

    @property
    def state(self) -> str:
        """The resume state: none, partial, transcription_only or complete."""
        return self.resume.state.value

    @property
    def completed_pages(self) -> int:
        """The number of pages an earlier run completed and this run reuses."""
        if not self.resumable or not self.resume.completed_page_indices:
            return 0
        return len(self.resume.completed_page_indices)

    def item_resume(self) -> ItemResume:
        """Return the working-log state handed to the item processor."""
        if not self.resumable:
            return ItemResume()
        return ItemResume(
            completed_page_indices=frozenset(self.resume.completed_page_indices or ()),
            transcription_results=self.resume.transcription_results,
            summary_results=self.resume.summary_results,
            log_header=self.resume.log_header,
        )


@dataclass(frozen=True)
class RunPlan:
    """What a run will do.

    *output_dir* None writes beside each input. *to_process* and *skipped*
    keep the discovery order; *skipped* items are already complete.
    *warnings* are notes for the user, such as ignored ``--select`` parts.
    *settings* are the machine settings the run uses. *sources* maps spec
    fields to where their values came from; fields it omits were given by
    the caller and count as flags.
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

    @property
    def force(self) -> bool:
        """Whether existing outputs are reprocessed instead of resumed."""
        return self.spec.run.force

    @property
    def dry_run(self) -> bool:
        """Whether the plan is only reported."""
        return self.spec.run.dry_run

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
    if job.summarize and not (
        job.output_markdown or job.output_docx or job.output_sqlite
    ):
        raise ConfigurationError(
            "Summaries are on but no output format is selected, so no summary "
            "would be written. Pass --no-summarize, or select summary-md, "
            "summary-docx or sqlite with --formats."
        )
    return job


def _context(item: ItemSpec, job: ItemJob) -> str | None:
    if not job.summarize or job.context == CONTEXT_NONE:
        return None
    try:
        return summary_context(item.path, job.context, default=job.fallback_context)
    except ContextError as exc:
        raise ConfigurationError(str(exc)) from exc


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
            ``exit_code`` the exit code.
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

    fmt = job.transcription_format
    checker = ResumeChecker(
        resume_mode="overwrite" if spec.run.force else "skip",
        summarize=job.summarize,
        output_docx=job.output_docx,
        output_markdown=job.output_markdown,
        retranscribe=spec.run.retranscribe,
        transcription_format=fmt,
        check_database=job.output_sqlite,
    )
    to_process: list[PlannedItem] = []
    skipped: list[PlannedItem] = []
    for item in selected:
        output_dir = _item_output_dir(item, base)
        try:
            result = checker.should_skip(item.output_stem, output_dir, item.path)
        except InputMismatchError as exc:
            raise DuplicateOutputError(str(exc), items_total=len(selected)) from exc
        if result.warning is not None:
            warnings += (result.warning,)
        if result.state is ProcessingState.COMPLETE:
            skipped.append(PlannedItem(item, output_dir, result, None, fmt))
        else:
            context = _context(item, job)
            to_process.append(PlannedItem(item, output_dir, result, context, fmt))
    logger.info(
        "%d item(s) to process, %d already complete",
        len(to_process),
        len(skipped),
    )
    return RunPlan(
        spec=spec,
        job=job,
        input_path=input_path,
        output_dir=base,
        discovered=tuple(discovered),
        to_process=tuple(to_process),
        skipped=tuple(skipped),
        warnings=warnings,
        settings=settings,
        sources=dict(sources or {}),
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
    reported them. *databases* are the absolute paths of the database files
    that received at least one item's rows.
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


def remove_working_dir(working_dir: Path) -> None:
    """Delete an item's working folder, clearing read-only flags as needed."""

    def _on_error(
        func: Callable[[str], object], path: str, _exc: BaseException
    ) -> None:
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError as exc:
            logger.warning("Could not forcibly remove %s: %s", path, exc)

    try:
        shutil.rmtree(working_dir, onexc=_on_error)
        logger.debug("Deleted temporary working directory: %s", working_dir)
    except OSError as exc:
        logger.warning("Failed to remove working directory %s: %s", working_dir, exc)


class _Databases:
    """The SQLite writers of a run: one per output folder, opened on first use."""

    def __init__(self, info: RunInfo) -> None:
        self.info = info
        self._writers: dict[Path, SqliteWriter] = {}
        self._written: dict[Path, None] = {}

    @property
    def written(self) -> tuple[str, ...]:
        """The database files written so far, in the order of their first row."""
        return tuple(str(path) for path in self._written)

    async def write(self, folder: Path, record: DocumentRecord) -> None:
        """Write *record* into the database of *folder*."""
        key = folder.resolve()
        writer = self._writers.get(key)
        if writer is None:
            writer = open_database(folder)
            await writer.open()
            self._writers[key] = writer
        await write_document(writer, record, self.info)
        self._written[writer.path.resolve()] = None

    async def close(self) -> None:
        """Close every writer."""
        writers, self._writers = list(self._writers.values()), {}
        for writer in writers:
            try:
                await writer.close()
            except sqlite3.Error as exc:
                logger.warning("Could not close %s: %s", writer.path, exc)


async def _write_record(
    databases: _Databases | None, planned: PlannedItem, processor: ItemProcessor
) -> bool:
    """Write the item's database rows; False when the write failed."""
    if databases is None or processor.record is None:
        return True
    try:
        await databases.write(planned.output_dir, processor.record)
    except (sqlite3.Error, OSError) as exc:
        logger.error("Could not write %s to the database: %s", planned.name, exc)
        return False
    return True


async def _run_item(
    run_plan: RunPlan,
    planned: PlannedItem,
    index: int,
    env: CallEnv,
    databases: _Databases | None = None,
) -> ItemResult:
    events = env.events
    total = len(run_plan.to_process)
    name = planned.name
    events.item_started(ItemStarted(name, planned.item.kind, index, total))
    planned.output_dir.mkdir(parents=True, exist_ok=True)
    processor: ItemProcessor | None = None
    success = False
    try:
        processor = ItemProcessor(
            planned.item,
            planned.output_dir,
            run_plan.job,
            planned.item_resume(),
            context=planned.context,
            events=events,
            env=env,
        )
        success = await processor.process()
        written = await _write_record(databases, planned, processor)
        success = success and written
        if not success:
            logger.error(
                "Item finished with page-level failures or missing outputs: %s",
                name,
            )
    except Exception as exc:
        logger.exception("Critical error in processing item '%s': %s", name, exc)
    finally:
        # Resume mode keeps the working logs for later runs, and an
        # incomplete item keeps them for the run that finishes it.
        working_dir = processor.paths.working_dir if processor else None
        if (
            working_dir is not None
            and success
            and run_plan.job.cleanup
            and run_plan.force
            and working_dir.exists()
        ):
            remove_working_dir(working_dir)

    outputs = (
        tuple(str(path) for path in processor.written_outputs) if processor else ()
    )
    report = processor.last_run_report if processor else None
    events.item_finished(ItemFinished(name, index, total, success, outputs, report))
    return ItemResult(name, success, planned.paths, outputs, report)


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
    every item, and with the sqlite format one database writer per output
    folder. Before any item starts, the client of every phase the run calls
    is built once. An item that crashes is logged and counted failed; the
    run continues with the next item. Cancellation propagates.

    Raises:
        ValueError: The plan is a dry run.
        ConfigurationError: A phase's client cannot be built because its key
            variable is unset or the settings cannot serve it; nothing has
            been written.
    """
    if run_plan.dry_run:
        raise ValueError("a dry-run plan is reported, not run")
    sink: RunEvents = events if events is not None else NullEvents()
    env = CallEnv.create(run_plan.settings, sink)
    if run_plan.to_process:
        _check_clients(run_plan, env)
    if run_plan.output_dir is not None:
        run_plan.output_dir.mkdir(parents=True, exist_ok=True)
    databases = (
        _Databases(RunInfo(run_plan.effective_spec()))
        if run_plan.job.output_sqlite
        else None
    )
    started = time.monotonic()
    results: list[ItemResult] = []
    try:
        for index, planned in enumerate(run_plan.to_process, start=1):
            results.append(await _run_item(run_plan, planned, index, env, databases))
    finally:
        if databases is not None:
            await databases.close()
    return RunResult(
        items=tuple(results),
        skipped=tuple(item.name for item in run_plan.skipped),
        seconds=time.monotonic() - started,
        usage={role: env.usage.get(role) for role in env.usage.roles},
        databases=databases.written if databases is not None else (),
    )
