"""Process one item (PDF or image folder) end to end, into the run's store.

The processor opens the payload source, plans one transcription unit and,
when summarizing, one summary unit per page, and works the units the store
still lacks: pending and failed pages are transcribed on an asyncio page
pool, and pages whose summary is missing, failed or stale are summarized
from their stored text. It then consolidates and enriches the citations,
stores them and renders the final files from the store. It reads no
configuration: everything comes from the
:class:`~autoexcerpter.pipeline.job.ItemJob` and the
:class:`~autoexcerpter.pipeline.units.DocumentHandle` built by the caller.
Progress goes to the :class:`~autoexcerpter.events.RunEvents` it is given.
Page images are rendered in a thread pool and the citations, OpenAlex
lookups included, are made in a worker thread; the model calls run on the
event loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from autoexcerpter.common.images import (
    IMAGE_EXTENSIONS,
    list_folder_images,
    sha256_of_file,
)
from autoexcerpter.common.sqlite import read
from autoexcerpter.common.store import PlannedUnit, UnitRow, refresh_documents
from autoexcerpter.events import NullEvents, PageDone, RunEvents
from autoexcerpter.imaging.payload import FolderPayloadSource, PdfPayloadSource
from autoexcerpter.llm import CallEnv
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.output.render import render_item
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    image_settings_record,
    read_pages,
)
from autoexcerpter.pipeline.job import ItemJob
from autoexcerpter.pipeline.managers import ItemManagers, build_managers
from autoexcerpter.pipeline.outputs import (
    OutputStatus,
    collect_citations,
    log_item_summary,
    render_formats,
)
from autoexcerpter.pipeline.pages import PageRunner
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.pipeline.units import DocumentHandle

logger = logging.getLogger(__name__)

PayloadSource = PdfPayloadSource | FolderPayloadSource

T = TypeVar("T")


async def _wait_out(worker: asyncio.Future[Any]) -> None:
    """Wait for *worker* to end, riding out further cancellations."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - the caller re-raises the cancellation
            return
    if not worker.cancelled():
        worker.exception()


def _by_index(units: Sequence[UnitRow]) -> dict[int, UnitRow]:
    return {unit.idx: unit for unit in units}


def open_source(item: ItemSpec, job: ItemJob) -> PayloadSource:
    """Open the lazy payload source of *item*; no page is rendered yet."""
    model = job.transcription
    source_cls = PdfPayloadSource if item.kind == "pdf" else FolderPayloadSource
    return source_cls(
        item.path,
        provider=model.provider,
        model_name=model.name,
        images=job.images,
        original_max_side_px=job.original_max_side_px,
    )


async def plan_document(
    document: DocumentHandle, source: PayloadSource, *, pdf: bool
) -> int:
    """Plan every page's units; drop the units of pages the input lost.

    Returns the page count.
    """
    count = len(source)
    pages = []
    for idx in range(count):
        label = source.image_name(idx)
        pages.append(PlannedUnit(idx, label=label, source_index=idx if pdf else label))
    await document.plan(pages)
    if any(unit.idx >= count for unit in document.units()):
        doc_id = document.doc_id

        def drop(connection: sqlite3.Connection) -> None:
            connection.execute(
                "DELETE FROM summaries WHERE doc_id = ? AND idx >= ?", (doc_id, count)
            )
            connection.execute(
                "DELETE FROM units WHERE doc_id = ? AND idx >= ?", (doc_id, count)
            )
            refresh_documents(connection, doc_id)

        logger.info(f"{document.name}: dropping the stored pages from {count + 1} on")
        await document.store.write(document.lease, drop)
    return count


def input_sha256(item: ItemSpec) -> str | None:
    """Return the SHA-256 of a PDF, or of an image folder's files in page order.

    None when the input cannot be read.
    """
    try:
        if item.kind == "pdf":
            return sha256_of_file(item.path)
        digest = hashlib.sha256()
        for path in list_folder_images(item.path, IMAGE_EXTENSIONS):
            digest.update(f"{path.name}\t{sha256_of_file(path)}\n".encode())
    except OSError as exc:
        logger.warning("Could not hash %s: %s", item.path, exc)
        return None
    return digest.hexdigest()


class ItemFiles:
    """Store an item's citations and render its files from the store.

    Once ``stop`` is set, OpenAlex enrichment ends and no file is rendered.
    ``written`` lists the absolute paths of the files that hold their
    rendering. *force* replaces files changed outside the tool; render
    warnings go to *events*. The files take the document's name.
    """

    def __init__(
        self,
        item: ItemSpec,
        output_dir: Path,
        job: ItemJob,
        document: DocumentHandle,
        *,
        force: bool = False,
        events: RunEvents | None = None,
    ) -> None:
        self.item = item
        self.name = document.name
        self.job = job
        self.document = document
        self.paths = ItemPaths.for_item(self.name, output_dir, job.transcription_format)
        self.type_label = "PDF" if item.kind == "pdf" else "Image Folder"
        self.stop = threading.Event()
        self.written: list[Path] = []
        self.force = force
        self.events: RunEvents = events if events is not None else NullEvents()

    async def _in_worker(self, func: Callable[[], T]) -> T:
        """Run *func* in a worker thread, off the event loop.

        When the awaiting task is cancelled, :attr:`stop` is set and the
        worker is awaited, shielded, before the cancellation propagates, so
        no file is written after it; the wait lasts at most one in-flight
        OpenAlex request.
        """
        worker = asyncio.get_running_loop().run_in_executor(None, func)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            self.stop.set()
            await _wait_out(worker)
            raise

    def _summaries(self) -> list[dict[str, Any]]:
        doc_id = self.document.doc_id

        def query(connection: sqlite3.Connection) -> list[dict[str, Any]]:
            return read_pages(connection, doc_id)[1]

        return read(self.document.store.path, query)

    def has_citations(self) -> bool:
        """Return whether the store holds citations for the document."""
        doc_id = self.document.doc_id

        def query(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                "SELECT 1 FROM citations WHERE doc_id = ? LIMIT 1", (doc_id,)
            ).fetchone()
            return row is not None

        return read(self.document.store.path, query)

    async def store_citations(self, *, offline: bool = False) -> bool:
        """Consolidate, enrich and store the citations of the stored summaries.

        *offline* takes the metadata from the OpenAlex cache alone. Returns
        False when :attr:`stop` was set during the enrichment.
        """
        if not self.job.summarize:
            return True
        summaries = self._summaries()
        if not summaries:
            return True
        citations = await self._in_worker(
            functools.partial(
                collect_citations,
                summaries,
                item_type_label=self.type_label,
                job=self.job,
                stop=self.stop,
                offline=offline,
            )
        )
        if citations is None:
            return False
        await self.document.write_citations(citations)
        return True

    async def write(self) -> OutputStatus:
        """Render the item's files from what the store holds.

        A file that could not be replaced fails its output; a file changed
        outside the tool is left as it is with a warning.

        Raises:
            LeaseLostError: The run's lease on the document was lost.
        """
        if self.stop.is_set():
            logger.warning("Item %s stopped: withholding its files.", self.name)
            return OutputStatus(transcription_ok=False, summaries_ok=False)
        report = await render_item(
            self.document.store,
            self.document.lease,
            self.paths,
            render_formats(self.job),
            force=self.force,
        )
        for message in report.warnings:
            self.events.warning(message)
        current = {*report.written, *report.unchanged}
        paths = self.paths
        for path in (paths.transcription, paths.summary_docx, paths.summary_md):
            if path in current:
                self.written.append(path.resolve())
        stale = set(report.stale)
        return OutputStatus(
            transcription_ok=paths.transcription not in stale,
            summaries_ok=not stale & {paths.summary_docx, paths.summary_md},
        )


class ItemProcessor:
    """Process a single input item (PDF or image folder) into its document.

    *managers* defaults to managers built from *job* with the item's summary
    *context* and the run's call environment *env* (rate limiters, usage
    totals, events). *force* replaces output files changed outside the tool.
    After :meth:`process`, ``written_outputs`` lists the absolute paths of
    the files that hold their rendering and ``last_run_report`` the per-item
    statistics (None when the item stopped before them). Once ``stop`` is
    set, OpenAlex enrichment ends and no output file is rendered; the item
    then counts failed and its units stay in the store.
    """

    def __init__(  # noqa: PLR0913 - the item, its document and the run's objects
        self,
        item: ItemSpec,
        output_dir: Path,
        job: ItemJob,
        document: DocumentHandle,
        *,
        managers: ItemManagers | None = None,
        context: str | None = None,
        events: RunEvents | None = None,
        env: CallEnv | None = None,
        force: bool = False,
    ) -> None:
        self.item = item
        self.name = item.output_stem
        self.job = job
        self.document = document
        self.events: RunEvents = events if events is not None else NullEvents()
        self.paths = ItemPaths.for_item(self.name, output_dir, job.transcription_format)
        self.type_label = "PDF" if item.kind == "pdf" else "Image Folder"

        if managers is None:
            env = env or CallEnv.create(events=self.events)
            managers = build_managers(job, context, env)
        self.managers = managers
        self.context = context
        self.runner = PageRunner(
            item_name=self.name,
            document=document,
            transcribe_manager=managers.transcription,
            summary_manager=managers.summary if job.summarize else None,
            transcription_provider=job.transcription.provider,
            pdf=item.kind == "pdf",
        )

        self.files = ItemFiles(
            item, output_dir, job, document, force=force, events=self.events
        )
        self.last_run_report: dict[str, Any] | None = None
        self.page_count = 0
        self.started_at = time.time()
        self._progress: PageProgress | None = None

    @property
    def stop(self) -> threading.Event:
        """The event that stops the enrichment and the file writes."""
        return self.files.stop

    @property
    def written_outputs(self) -> list[Path]:
        """The absolute paths of the files written."""
        return self.files.written

    async def process(self) -> bool:
        """Process the item end to end.

        Returns:
            True when every page succeeded (transcription and, if enabled,
            summarization) and all configured outputs were written.

        Raises:
            LeaseLostError: The run's lease on the document was lost; the
                units stored before stay.
        """
        logger.info(f"Processing {self.name} ({self.type_label})")
        self.started_at = time.time()
        try:
            source = self._open_source()
        except Exception as e:
            logger.exception(f"Failed to open input for {self.name}: {e}")
            return False
        try:
            if len(source) == 0:
                logger.info(
                    f"No images found or extracted for {self.name}. Aborting this item."
                )
                return False
            return await self._run(source)
        finally:
            source.close()

    def _open_source(self) -> PayloadSource:
        """Open the lazy payload source; no page is rendered yet."""
        return open_source(self.item, self.job)

    async def _run(self, source: PayloadSource) -> bool:
        """Plan the units, work the missing ones and write the outputs."""
        self.page_count = len(source)
        suffix = " and summarization" if self.job.summarize else ""
        logger.info(f"Prepared {self.page_count} images for transcription{suffix}.")
        await self._store_artifacts()
        await self._plan(source)
        transcriptions = _by_index(self.document.units(KIND_TRANSCRIPTION))
        pending = [
            idx
            for idx in range(self.page_count)
            if transcriptions[idx].status != "done"
        ]
        if pending:
            await self.document.set_image_settings(
                image_settings_record(source.img_cfg)
            )
        await self._summarize_stored(transcriptions, pending)
        await self._transcribe_pending(source, pending)
        if self.runner.lost is not None:
            raise self.runner.lost
        return await self._finish()

    async def _store_artifacts(self) -> None:
        """Store the prompt, schema and context texts the item's calls use."""
        texts: dict[str, Any] = {
            "transcription_prompt": getattr(
                self.managers.transcription, "system_prompt", None
            ),
            "transcription_schema": getattr(
                self.managers.transcription, "schema", None
            ),
            "context": self.context,
        }
        if self.runner.summary_manager is not None:
            summary = self.runner.summary_manager
            texts["summary_prompt"] = getattr(summary, "system_prompt", None)
            texts["summary_schema"] = getattr(summary, "schema", None)
        store = self.document.store
        for kind, value in texts.items():
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            if isinstance(value, str) and value:
                await store.store_artifact(kind, value)

    async def _plan(self, source: PayloadSource) -> None:
        """Plan every page's units; drop the units of pages the input lost."""
        await plan_document(self.document, source, pdf=self.item.kind == "pdf")

    async def _summarize_stored(
        self, transcriptions: dict[int, UnitRow], pending: Sequence[int]
    ) -> None:
        """Summarize the stored pages whose summary is missing, failed or stale."""
        if self.runner.summary_manager is None:
            return
        summaries = _by_index(self.document.units(KIND_SUMMARY))
        transcribing = set(pending)
        redo = [
            idx
            for idx in range(self.page_count)
            if idx not in transcribing
            and (summaries[idx].status, summaries[idx].stale) != ("done", 0)
        ]
        if not redo:
            return
        logger.info(f"Summarizing {len(redo)} page(s) transcribed by an earlier run")
        for idx in redo:
            unit = transcriptions[idx]
            error = unit.error if unit.status == "failed" else None
            await self.runner.summarize_page(
                idx,
                unit.label or f"page index {idx}",
                unit.content or "",
                error,
            )

    async def _transcribe_pending(
        self, source: PayloadSource, pending: list[int]
    ) -> None:
        """Run the pending pages on the page pool."""
        if not pending:
            logger.info(
                "All pages already transcribed. Skipping transcription; "
                "summaries reused or made from the stored text."
            )
            return
        already_complete = self.page_count - len(pending)
        workers = max(1, min(self.job.concurrency, len(pending)))
        resume_note = (
            f" ({already_complete} skipped via resume)" if already_complete else ""
        )
        suffix = " and summarization" if self.job.summarize else ""
        logger.info(
            f"Starting transcription{suffix} of {len(pending)} images{resume_note}..."
        )
        logger.info(f"Using {workers} concurrent workers for transcription")
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="autoexcerpter-render"
        )
        self.runner.executor = pool
        try:
            await self._run_page_pool(source, pending, already_complete, workers)
        finally:
            self.runner.executor = None
            pool.shutdown(wait=False, cancel_futures=True)

    async def _run_page_pool(
        self,
        source: PayloadSource,
        pending: list[int],
        already_complete: int,
        workers: int,
    ) -> None:
        """Run the pending pages, at most *workers* at a time.

        Results are consumed in completion order, and each finished page is
        reported as a :class:`~autoexcerpter.events.PageDone` event. A page
        error that escapes the page runner propagates after the remaining
        pages are cancelled; cancellation of the item cancels every page.
        When a phase runs on Anthropic, the first page runs alone before the
        others start, so that they read its cached system prompt.
        """
        progress = PageProgress(
            len(pending),
            already_complete=already_complete,
            workers=workers,
            started_at=self.started_at,
        )
        self._progress = progress
        semaphore = asyncio.Semaphore(workers)

        async def run_page(index: int) -> tuple[int, str]:
            async with semaphore:
                return index, await self.runner.run(index, source, progress)

        managers = (self.runner.transcribe_manager, self.runner.summary_manager)
        warm = any(getattr(m, "provider", None) == "anthropic" for m in managers)
        batches = [pending[:1], pending[1:]] if warm else [pending]
        tasks: list[asyncio.Future[tuple[int, str]]] = []
        done = 0
        try:
            for batch in batches:
                started = [asyncio.ensure_future(run_page(idx)) for idx in batch]
                tasks.extend(started)
                for finished in asyncio.as_completed(started):
                    index, status = await finished
                    done += 1
                    self.events.page_done(
                        PageDone(
                            item=self.name,
                            page=index,
                            status=status,
                            done=done,
                            total=len(pending),
                            already_complete=already_complete,
                        )
                    )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _write_outputs(self) -> OutputStatus:
        """Store the citations, then render the files from the store."""
        if not await self.files.store_citations():
            return OutputStatus(transcription_ok=False, summaries_ok=False)
        return await self.files.write()

    async def _finish(self) -> bool:
        """Write the outputs, record the run report and return the verdict."""
        elapsed_s = time.time() - self.started_at
        units = self.document.units()
        transcriptions = [u for u in units if u.kind == KIND_TRANSCRIPTION]
        # Summary units of an earlier run stay stored when this run does not
        # summarize; only the requested work decides the outcome.
        summaries = [u for u in units if u.kind == KIND_SUMMARY and self.job.summarize]
        pages_ok = sum(1 for u in transcriptions if u.status == "done")
        pages_failed = sum(1 for u in transcriptions if u.status == "failed")
        pages_deferred = sum(1 for u in transcriptions if u.status == "pending")
        summary_failures = sum(1 for u in summaries if u.status == "failed")
        summaries_pending = sum(1 for u in summaries if u.status == "pending")

        # A pending unit has no stored result. A file written now would
        # misreport the item as complete, so all final outputs are withheld;
        # a later run finishes the item from the store.
        status = OutputStatus()
        if pages_deferred or summaries_pending:
            logger.error(
                "Item %s incomplete: %d of %d page(s) and %d summary(ies) were "
                "not stored. Withholding the transcription and summary outputs "
                "(a truncated file would misreport as complete); the stored "
                "pages stay for a later run.",
                self.name,
                pages_deferred,
                self.page_count,
                summaries_pending,
            )
        else:
            status = await self._write_outputs()

        times = self._progress.transcription_times if self._progress else []
        avg_api_s = sum(times) / len(times) if times else None
        log_item_summary(
            name=self.name,
            paths=self.paths,
            job=self.job,
            pages_logged=pages_ok + pages_failed,
            pages_ok=pages_ok,
            pages_failed=pages_failed,
            elapsed_s=elapsed_s,
            avg_api_s=avg_api_s,
        )
        self.last_run_report = {
            "pages_total": self.page_count,
            "pages_attempted": pages_ok + pages_failed,
            "pages_ok": pages_ok,
            "pages_failed": pages_failed,
            "pages_deferred": pages_deferred,
            "summary_failures": summary_failures,
            "elapsed_s": elapsed_s,
            "avg_api_s": avg_api_s,
            "outputs": [str(p) for p in self.written_outputs],
        }

        write_errors = len(self.runner.write_errors)
        success = (
            pages_deferred == 0
            and summaries_pending == 0
            and pages_failed == 0
            and summary_failures == 0
            and status.summaries_ok
            and status.transcription_ok
            and write_errors == 0
        )
        if not success:
            logger.error(
                "Item %s finished incomplete: %d deferred page(s), %d failed "
                "transcription page(s), %d failed summary page(s), %d unit "
                "write(s) failed, summary files rendered ok: %s, transcription "
                "file written ok: %s",
                self.name,
                pages_deferred,
                pages_failed,
                summary_failures,
                write_errors,
                status.summaries_ok,
                status.transcription_ok,
            )
        return success


__all__ = [
    "ItemFiles",
    "ItemProcessor",
    "PayloadSource",
    "input_sha256",
    "open_source",
    "plan_document",
]
