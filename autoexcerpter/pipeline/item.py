"""Process one item (PDF or image folder) end to end.

The processor opens the payload source, restores completed pages from the
working log, runs the pending pages on an asyncio page pool and writes the
final outputs. It reads no configuration: everything comes from the
:class:`~autoexcerpter.pipeline.job.ItemJob` built by the caller. Progress
goes to the :class:`~autoexcerpter.events.RunEvents` it is given. Page images
are rendered in a thread pool and the final outputs, OpenAlex lookups
included, are written in a worker thread; the model calls run on the event
loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import hashlib
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from autoexcerpter.common.images import sha256_of_file
from autoexcerpter.events import NullEvents, PageDone, RunEvents
from autoexcerpter.imaging.payload import FolderPayloadSource, PdfPayloadSource
from autoexcerpter.llm import CallEnv
from autoexcerpter.pipeline.job import ItemJob, ItemResume
from autoexcerpter.pipeline.log import (
    LogHeader,
    append_to_log,
    finalize_log_file,
    initialize_log_or_raise,
    read_log,
)
from autoexcerpter.pipeline.managers import ItemManagers, build_managers
from autoexcerpter.pipeline.outputs import (
    ItemResults,
    OutputStatus,
    log_item_summary,
    write_outputs,
)
from autoexcerpter.pipeline.pages import PageRunner
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.resume import load_results, verify_image_settings
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.rendering.sqlite import DocumentRecord

logger = logging.getLogger(__name__)

PayloadSource = PdfPayloadSource | FolderPayloadSource


def eligible_prior_entries(
    prior_results: list[dict[str, Any]],
    completed: frozenset[int],
    page_count: int,
) -> list[dict[str, Any]]:
    """Return the prior transcription entries eligible for page-level reuse.

    An entry is eligible when it has an integer index that is marked completed
    and lies within the current page count (an index from a swapped, shorter
    input is dropped). Duplicate indices collapse to one entry, preferring the
    error-free one, so a page is never replayed twice. Ordered by index.
    """
    by_index: dict[int, dict[str, Any]] = {}
    for entry in prior_results:
        idx = entry.get("original_input_order_index")
        if not isinstance(idx, int) or idx not in completed or idx >= page_count:
            continue
        existing = by_index.get(idx)
        if existing is None or ("error" in existing and "error" not in entry):
            by_index[idx] = entry
    return [by_index[idx] for idx in sorted(by_index)]


def _summaries_by_index(
    prior_summaries: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Map page index to prior summary; a later entry for an index wins."""
    by_index: dict[int, dict[str, Any]] = {}
    for summ in prior_summaries:
        idx = summ.get("original_input_order_index")
        if isinstance(idx, int):
            by_index[idx] = summ
    return by_index


def reusable_summaries(
    eligible: list[dict[str, Any]],
    prior_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the present, error-free prior summaries of the eligible pages."""
    by_index = _summaries_by_index(prior_summaries)
    reusable = []
    for entry in eligible:
        prior = by_index.get(entry["original_input_order_index"])
        if isinstance(prior, dict) and "error" not in prior:
            reusable.append(prior)
    return reusable


async def reload_completed_pages(
    eligible: list[dict[str, Any]],
    prior_summaries: list[dict[str, Any]],
    runner: PageRunner,
    transcription_results: list[dict[str, Any]],
    summary_results: list[dict[str, Any]],
) -> None:
    """Restore completed pages and (re)attach their summaries.

    The eligible transcriptions and their reusable summaries are already on
    disk: each log was rewritten with them in one atomic write. This admits
    each page to the results; a page without a reusable summary gets one
    regenerated from its logged text, which is then appended to the summary
    log.
    """
    paths = runner.paths
    prior_by_idx = _summaries_by_index(prior_summaries)
    summarizing = runner.summary_manager is not None

    for entry in eligible:
        transcription_results.append(entry)
        if not summarizing:
            continue
        idx = entry["original_input_order_index"]
        prior = prior_by_idx.get(idx)
        if isinstance(prior, dict) and "error" not in prior:
            # Already in the rewritten summary log.
            summary_results.append(prior)
            continue
        image_name = entry.get("image") or f"page index {idx}"
        regenerated = await runner.summarize_logged(entry, idx, image_name)
        if regenerated is not None:
            append_to_log(paths.summary_log, regenerated)
            summary_results.append(regenerated)


def document_hash(source: PayloadSource, provenance: dict[str, Any]) -> str | None:
    """Return the SHA-256 of a PDF, or of an image folder's files in page order.

    None when the input cannot be read.
    """
    recorded = provenance.get("source_sha256")
    if isinstance(recorded, str):
        return recorded
    if not isinstance(source, FolderPayloadSource):
        return None
    digest = hashlib.sha256()
    try:
        for path in source.image_paths:
            digest.update(f"{path.name}\t{sha256_of_file(path)}\n".encode())
    except OSError as exc:
        logger.warning("Could not hash the images of %s: %s", source.source_path, exc)
        return None
    return digest.hexdigest()


def _by_index(entry: dict[str, Any]) -> Any:
    return entry.get("original_input_order_index", 0)


def _page_status(result: dict[str, Any] | None) -> str:
    """Return the event status of one page-runner result."""
    if result is None:
        return "deferred"
    return "failed" if "error" in result else "ok"


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


class ItemProcessor:
    """Process a single input item (PDF or image folder).

    *managers* defaults to managers built from *job* with the item's summary
    *context* and the run's call environment *env* (rate limiters, usage
    totals, events). After :meth:`process`, ``written_outputs`` lists the
    absolute paths of the files written, ``last_run_report`` holds the
    per-item statistics and ``record`` what the run produced for the
    database (both None when the item stopped before any were computed).
    Once ``stop`` is set, OpenAlex enrichment ends and no further output
    file is written; the item then counts failed and keeps its working logs.
    """

    def __init__(
        self,
        item: ItemSpec,
        output_dir: Path,
        job: ItemJob,
        resume: ItemResume | None = None,
        *,
        managers: ItemManagers | None = None,
        context: str | None = None,
        events: RunEvents | None = None,
        env: CallEnv | None = None,
    ) -> None:
        self.item = item
        self.name = item.output_stem
        self.job = job
        self.resume = resume or ItemResume()
        self.events: RunEvents = events if events is not None else NullEvents()
        self.paths = ItemPaths.for_item(self.name, output_dir, job.transcription_format)
        self.paths.working_dir.mkdir(parents=True, exist_ok=True)
        self.type_label = "PDF" if item.kind == "pdf" else "Image Folder"

        if managers is None:
            env = env or CallEnv.create(events=self.events)
            managers = build_managers(job, context, env)
        self.runner = PageRunner(
            item_name=self.name,
            paths=self.paths,
            transcribe_manager=managers.transcription,
            summary_manager=managers.summary if job.summarize else None,
            transcription_provider=job.transcription.provider,
        )

        self.stop = threading.Event()
        self.written_outputs: list[Path] = []
        self.last_run_report: dict[str, Any] | None = None
        self.record: DocumentRecord | None = None
        self.page_count = 0
        self.started_at = time.time()
        self._log_header = self.resume.log_header
        self._prior_transcriptions: list[dict[str, Any]] = []
        self._prior_summaries: list[dict[str, Any]] = []
        self._reused_pages: frozenset[int] = frozenset()
        self._source_hash: str | None = None
        # Extra metadata lines for the transcription file (e.g. a resume model
        # mismatch: logged transcriptions produced by a different model).
        self._metadata_notes: list[str] = []
        self._progress: PageProgress | None = None

    async def process(self) -> bool:
        """Process the item end to end.

        Returns:
            True when every page succeeded (transcription and, if enabled,
            summarization) and all configured outputs were written.
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
                self._remove_empty_working_dir()
                return False
            return await self._run(source)
        finally:
            source.close()

    def _open_source(self) -> PayloadSource:
        """Open the lazy payload source; no page is rendered yet."""
        model = self.job.transcription
        source_cls = (
            PdfPayloadSource if self.item.kind == "pdf" else FolderPayloadSource
        )
        return source_cls(
            self.item.path,
            provider=model.provider,
            model_name=model.name,
            images=self.job.images,
        )

    def _remove_empty_working_dir(self) -> None:
        working_dir = self.paths.working_dir
        try:
            if not any(working_dir.iterdir()):
                shutil.rmtree(working_dir)
                logger.info(f"Removed empty working directory: {working_dir}")
        except OSError as e:
            logger.warning(f"Could not remove working directory {working_dir}: {e}")

    async def _run(self, source: PayloadSource) -> bool:
        """Restore resume state, process the pages and write the outputs."""
        self.page_count = len(source)
        suffix = " and summarization" if self.job.summarize else ""
        logger.info(f"Prepared {self.page_count} images for transcription{suffix}.")

        self._load_resume_state(source)
        self._init_transcription_log(source)
        try:
            (
                transcription_results,
                summary_results,
            ) = await self._transcribe_and_summarize(source)
            return await self._finish(transcription_results, summary_results)
        finally:
            finalize_log_file(self.paths.transcription_log)
            if self.job.summarize:
                finalize_log_file(self.paths.summary_log)

    def _pages_remaining(self, source: PayloadSource) -> bool:
        completed = self.resume.completed_page_indices
        return any(idx not in completed for idx in range(len(source)))

    def _load_resume_state(self, source: PayloadSource) -> None:
        """Snapshot the prior page results before the logs are rewritten.

        Raises ValueError when pages remain to transcribe and the logged image
        settings differ from the current ones; the working log is untouched.
        """
        if not self.resume.completed_page_indices:
            return
        if self.resume.transcription_results is not None:
            self._prior_transcriptions = self.resume.transcription_results
        else:
            log = read_log(self.paths.transcription_log)
            self._prior_transcriptions = log.results if log is not None else []
            if self._log_header is None and log is not None:
                self._log_header = log.header
        if self.resume.summary_results is not None:
            self._prior_summaries = self.resume.summary_results
        else:
            self._prior_summaries = load_results(self.paths.summary_log) or []
        self._detect_resume_model_mismatch()
        # A summary-only resume reuses logged text, so changed image settings
        # matter only when pages remain to transcribe.
        if self._log_header is not None and self._pages_remaining(source):
            verify_image_settings(self._log_header, source.img_cfg)

    def _detect_resume_model_mismatch(self) -> None:
        """Warn and record when reused transcriptions came from another model."""
        header = self._log_header
        if header is None:
            log = read_log(self.paths.transcription_log)
            header = log.header if log is not None else None
        logged_model = header.get("model_name") if isinstance(header, dict) else None
        current = self.job.transcription.name
        if logged_model and logged_model != current:
            note = (
                "Summary-only resume: reusing transcriptions from model "
                f"'{logged_model}'; current transcription model is '{current}'."
            )
            self.events.warning(note)
            self._metadata_notes.append(note)

    def _eligible_entries(self, page_count: int) -> list[dict[str, Any]]:
        """Return the prior transcriptions this run reuses."""
        completed = self.resume.completed_page_indices
        if not completed:
            return []
        return eligible_prior_entries(self._prior_transcriptions, completed, page_count)

    def _init_transcription_log(self, source: PayloadSource) -> None:
        """Rewrite the transcription log: its header and the reused pages.

        Both go to disk in one atomic replace, so a failure or crash keeps
        the previous log. When every page is reused, the header keeps the
        image provenance those pages were transcribed with rather than the
        current settings.
        """
        file_provenance = source.file_provenance()
        self._source_hash = document_hash(source, file_provenance)
        recorded = (
            (self._log_header or {}).get("file_provenance")
            if self.resume.completed_page_indices
            else None
        )
        if isinstance(recorded, dict) and not self._pages_remaining(source):
            file_provenance = recorded
        target_dpi = source.target_dpi if isinstance(source, PdfPayloadSource) else None
        initialize_log_or_raise(
            self.paths.transcription_log,
            LogHeader(
                item_name=self.name,
                input_path=str(self.item.path),
                input_type=self.type_label,
                total_images=self.page_count,
                model_name=self.job.transcription.name,
                extraction_dpi=target_dpi,
                concurrency_limit=self.job.concurrency,
                file_provenance=file_provenance,
                api_timeout=self.job.timeout,
                service_tier=self.job.transcription.service_tier,
            ),
            self._eligible_entries(len(source)),
        )

    async def _transcribe_and_summarize(
        self, source: PayloadSource
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Restore completed pages, then run the pending ones on the pool."""
        transcription_results: list[dict[str, Any]] = []
        summary_results: list[dict[str, Any]] = []
        page_count = len(source)
        self.page_count = page_count
        completed = self.resume.completed_page_indices
        pending = [idx for idx in range(page_count) if idx not in completed]
        eligible = self._eligible_entries(page_count)
        self._reused_pages = frozenset(
            entry["original_input_order_index"] for entry in eligible
        )

        if self.runner.summary_manager is not None:
            initialize_log_or_raise(
                self.paths.summary_log,
                LogHeader(
                    item_name=self.name,
                    input_path=str(self.item.path),
                    input_type=self.type_label,
                    total_images=page_count,
                    model_name=self.job.summary.name,
                    concurrency_limit=self.job.concurrency,
                    log_type="summary",
                    api_timeout=self.job.timeout,
                    service_tier=self.job.transcription.service_tier,
                ),
                reusable_summaries(eligible, self._prior_summaries),
            )

        workers = max(1, min(self.job.concurrency, len(pending)))
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="autoexcerpter-render"
        )
        self.runner.executor = pool
        try:
            skipped = page_count - len(pending)
            if completed:
                await reload_completed_pages(
                    eligible,
                    self._prior_summaries,
                    self.runner,
                    transcription_results,
                    summary_results,
                )
                if skipped > 0:
                    logger.info(
                        f"Page-level resume: {skipped} page(s) already "
                        f"transcribed, {len(pending)} page(s) remaining"
                    )
            if not pending:
                logger.info(
                    "All pages already transcribed (page-level resume). "
                    "Skipping transcription; summaries reused/regenerated from logs."
                )
            else:
                suffix = " and summarization" if self.job.summarize else ""
                resume_note = f" ({skipped} skipped via resume)" if skipped else ""
                logger.info(
                    f"Starting transcription{suffix} of "
                    f"{len(pending)} images{resume_note}..."
                )
                logger.info(f"Using {workers} concurrent workers for transcription")
                await self._run_page_pool(
                    source,
                    pending,
                    skipped,
                    workers,
                    transcription_results,
                    summary_results,
                )
        finally:
            self.runner.executor = None
            pool.shutdown(wait=False, cancel_futures=True)

        transcription_results.sort(key=_by_index)
        return transcription_results, summary_results

    async def _run_page_pool(
        self,
        source: PayloadSource,
        pending: list[int],
        already_complete: int,
        workers: int,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
    ) -> None:
        """Run the pending pages, at most *workers* at a time.

        Results are consumed in completion order, and each finished page is
        reported as a :class:`~autoexcerpter.events.PageDone` event. A page
        error that escapes the page runner propagates after the remaining
        pages are cancelled; cancellation of the item cancels every page.
        """
        progress = PageProgress(
            len(pending),
            already_complete=already_complete,
            workers=workers,
            started_at=self.started_at,
        )
        self._progress = progress
        semaphore = asyncio.Semaphore(workers)

        async def run_page(index: int) -> tuple[int, dict[str, Any] | None]:
            async with semaphore:
                result = await self.runner.run(
                    index,
                    source,
                    transcription_results,
                    summary_results,
                    progress,
                )
                return index, result

        tasks = [asyncio.ensure_future(run_page(idx)) for idx in pending]
        try:
            for done, finished in enumerate(asyncio.as_completed(tasks), start=1):
                index, result = await finished
                self.events.page_done(
                    PageDone(
                        item=self.name,
                        page=index,
                        status=_page_status(result),
                        done=done,
                        total=len(tasks),
                        already_complete=already_complete,
                    )
                )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _write_outputs(
        self,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
        elapsed_s: float,
    ) -> OutputStatus:
        """Write the final files in a worker thread, off the event loop.

        When the awaiting task is cancelled, :attr:`stop` is set and the
        worker is awaited, shielded, before the cancellation propagates, so
        no file is written after it; the wait lasts at most one in-flight
        OpenAlex request.
        """
        call = functools.partial(
            write_outputs,
            name=self.name,
            item_type_label=self.type_label,
            input_path=self.item.path,
            paths=self.paths,
            job=self.job,
            results=ItemResults(
                transcriptions=transcription_results,
                summaries=summary_results,
                elapsed_s=elapsed_s,
                metadata_notes=self._metadata_notes,
            ),
            written=self.written_outputs,
            stop=self.stop,
        )
        worker = asyncio.get_running_loop().run_in_executor(None, call)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            self.stop.set()
            await _wait_out(worker)
            raise

    async def _finish(
        self,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
    ) -> bool:
        """Write the outputs, record the run report and return the verdict."""
        elapsed_s = time.time() - self.started_at
        transcription_results.sort(key=_by_index)
        pages_ok = sum(1 for r in transcription_results if "error" not in r)
        pages_failed = len(transcription_results) - pages_ok
        # Failed transcriptions get error-free placeholder summaries, so this
        # never double-counts a page already in pages_failed.
        summary_failures = sum(
            1 for s in summary_results if isinstance(s, dict) and "error" in s
        )

        # Deferred pages are absent from the results and the working log. A
        # transcription written now would misreport as complete, so all final
        # outputs are withheld; a later resume finishes the item.
        pages_deferred = self.page_count - len(transcription_results)
        status = OutputStatus()
        if pages_deferred > 0:
            logger.error(
                "Item %s incomplete: %d of %d page(s) were deferred and never "
                "transcribed. Withholding the partial transcription and summary "
                "outputs "
                "(a truncated file would misreport as complete); %d completed "
                "page(s) are retained in the working log for a later resume run.",
                self.name,
                pages_deferred,
                self.page_count,
                len(transcription_results),
            )
        else:
            status = await self._write_outputs(
                transcription_results, summary_results, elapsed_s
            )

        times = self._progress.transcription_times if self._progress else []
        avg_api_s = sum(times) / len(times) if times else None
        log_item_summary(
            name=self.name,
            paths=self.paths,
            job=self.job,
            pages_logged=len(transcription_results),
            pages_ok=pages_ok,
            pages_failed=pages_failed,
            elapsed_s=elapsed_s,
            avg_api_s=avg_api_s,
        )
        self.last_run_report = {
            "pages_total": self.page_count,
            "pages_attempted": len(transcription_results),
            "pages_ok": pages_ok,
            "pages_failed": pages_failed,
            "pages_deferred": pages_deferred,
            "summary_failures": summary_failures,
            "elapsed_s": elapsed_s,
            "avg_api_s": avg_api_s,
            "outputs": [str(p) for p in self.written_outputs],
        }

        append_failures = self.runner.log_append_failures
        success = (
            pages_deferred == 0
            and pages_failed == 0
            and summary_failures == 0
            and status.summaries_ok
            and status.transcription_ok
            and append_failures == 0
        )
        if not success:
            logger.error(
                "Item %s finished incomplete: %d deferred page(s), %d failed "
                "transcription page(s), %d failed summary page(s), %d failed "
                "working-log append(s), summary files rendered ok: %s, "
                "transcription file written ok: %s",
                self.name,
                pages_deferred,
                pages_failed,
                summary_failures,
                append_failures,
                status.summaries_ok,
                status.transcription_ok,
            )
        self.record = self._record(
            transcription_results, summary_results, status, success
        )
        return success

    def _record(
        self,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
        status: OutputStatus,
        success: bool,
    ) -> DocumentRecord:
        """Return what this run produced for the item, for the database."""
        header = self._log_header or {}
        reused_model = header.get("model_name")
        return DocumentRecord(
            name=self.name,
            path=self.item.path,
            kind=self.item.kind,
            hash=self._source_hash,
            transcription_model=self.job.transcription.name,
            summary_model=self.job.summary.name if self.job.summarize else None,
            complete=success,
            report=dict(self.last_run_report or {}),
            pages=list(transcription_results),
            summaries=list(summary_results) if self.job.summarize else None,
            citations=status.citations,
            usage=dict(self.runner.page_usage),
            reused_pages=self._reused_pages,
            reused_model=reused_model if isinstance(reused_model, str) else None,
        )


__all__ = [
    "ItemProcessor",
    "PayloadSource",
    "eligible_prior_entries",
    "reload_completed_pages",
    "reusable_summaries",
]
