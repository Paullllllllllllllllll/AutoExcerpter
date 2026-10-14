"""Batch check: finalize the batch submissions below a folder, then go on.

:func:`check_batches` finds every ``autoexcerpter.sqlite`` below a folder
and works its open submissions. A submission left ``submitting`` without a
provider batch id is marked unresolved first, and every unresolved
submission is reconciled with the provider's batch list (see
:mod:`autoexcerpter.batch.reconcile`); ``--adopt`` and ``--abandon`` settle
one by hand. Each submitted submission is checked and, once its batch has
finished, finalized into the store
(:func:`~autoexcerpter.batch.finalize.finalize_submission`) under the leases
of its documents.

Then every document with a batch job (see
:mod:`autoexcerpter.batch.jobs`) and no open or unresolved submission
advances. The summaries its finalized transcriptions call for are submitted
to the summary model's batch API, and the placeholders of failed and blank
pages are written. A document that has nothing left to submit and no
pending unit gets its citations stored and its files rendered, and its batch
job ends. The batches' input and result files are deleted from the provider
after that. ``--batch-ids`` limits the finalization to those batches.

:func:`check_until_settled` repeats the check with ``--wait`` until no
batch work a check can advance is pending, or its deadline has passed. A
:class:`WaitLedger` records what every pass saw, and :func:`wait_totals`
builds the final counts from the stores' current state of those
submissions and documents.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

from autoexcerpter.batch.bodies import ROLE_SUMMARY, build_plan
from autoexcerpter.batch.clients import BatchClients
from autoexcerpter.batch.finalize import (
    FAILED,
    FINALIZED_OUTCOME,
    PENDING,
    SKIPPED,
    SubmissionFinalization,
    delete_provider_files,
    finalize_submission,
)
from autoexcerpter.batch.item import LABEL
from autoexcerpter.batch.jobs import DocumentJob, load_jobs
from autoexcerpter.batch.reconcile import (
    abandon,
    adopt,
    count_matches,
    lease_submission,
    list_candidates,
    provider_of,
    reconcile,
)
from autoexcerpter.batch.submit import (
    BatchSubmissionError,
    submit_units,
)
from autoexcerpter.batch.work import SummaryWork, summary_work, write_placeholders
from autoexcerpter.common.batch import BatchListing
from autoexcerpter.common.polling import (
    PollResult,
    WaitLimits,
    WaitTick,
    poll_until_settled,
)
from autoexcerpter.common.retry import classify_error
from autoexcerpter.common.sqlite import read
from autoexcerpter.common.store import (
    Lease,
    LeaseLostError,
    Store,
    StoreError,
    Submission,
    SubmissionStateError,
    UnitRow,
)
from autoexcerpter.common.usage import Usage
from autoexcerpter.events import NullEvents
from autoexcerpter.output.store import (
    DATABASE_NAME,
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    BatchJob,
    delete_batch_job,
    open_store,
    read_batch_jobs,
    read_store,
)
from autoexcerpter.pipeline.item import ItemFiles
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.pipeline.units import DocumentHandle, Stamp
from autoexcerpter.settings import Settings

__all__ = [
    "ABANDONED",
    "COMPLETE",
    "NEEDS_RUN",
    "SUBMITTED",
    "UNRESOLVED",
    "WAITING",
    "CheckResult",
    "CheckStats",
    "CheckTargetError",
    "DocumentAdvance",
    "RunHint",
    "SubmissionCheck",
    "WaitLedger",
    "WaitProgress",
    "WaitTotals",
    "check_batches",
    "check_until_settled",
    "end_batch_jobs",
    "find_databases",
    "pending_work",
    "wait_totals",
]

logger = logging.getLogger(__name__)

ROLE = "batch"

UNRESOLVED: Final = "unresolved"
ABANDONED: Final = "abandoned"
SUBMITTED: Final = "submitted"
COMPLETE: Final = "complete"
WAITING: Final = "waiting"
NEEDS_RUN: Final = "needs_run"
_WORKED: Final = ("submitting", "submitted", UNRESOLVED)

_clock: Callable[[], float] = time.monotonic
"""The wait's clock; tests replace it together with :data:`_sleep`."""
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass
class CheckStats:
    """How many submissions ended in each outcome.

    ``skipped`` counts submissions that were no longer submitted;
    ``unresolved`` the submissions that wait for an adoption or an
    abandonment.
    """

    finalized: int = 0
    pending: int = 0
    failed: int = 0
    skipped: int = 0
    unresolved: int = 0
    abandoned: int = 0

    @property
    def had_failure(self) -> bool:
        """Whether something failed or waits for the user; exit code 1."""
        return self.failed > 0 or self.unresolved > 0

    def count(self, outcome: str) -> None:
        """Count one outcome."""
        if outcome == FINALIZED_OUTCOME:
            self.finalized += 1
        elif outcome == PENDING:
            self.pending += 1
        elif outcome == FAILED:
            self.failed += 1
        elif outcome == SKIPPED:
            self.skipped += 1
        elif outcome == UNRESOLVED:
            self.unresolved += 1
        elif outcome == ABANDONED:
            self.abandoned += 1


@dataclass(frozen=True)
class SubmissionCheck:
    """What one check did with one submission.

    *outcome* is ``finalized``, ``pending``, ``failed``, ``skipped``,
    ``unresolved`` or ``abandoned``; *detail* says why. *errors* counts the
    units finalized as error placeholders, *usage* holds the tokens of the
    finalized results per role and *candidates* the batches an unresolved
    submission may belong to.
    """

    database: Path
    submission_id: str
    provider: str
    outcome: str
    batch_id: str | None = None
    detail: str | None = None
    errors: int = 0
    usage: Mapping[str, Usage] = field(default_factory=dict)
    candidates: tuple[BatchListing, ...] = ()


@dataclass(frozen=True)
class DocumentAdvance:
    """What one check did with one document's batch job.

    *outcome* is ``submitted`` (its summaries went to the batch API),
    ``complete`` (its files were rendered and the job ended), ``failed``
    (rendered with failed pages, or a submission was refused), ``waiting``
    (another run holds the document) or ``needs_run`` (pages are pending
    with no batch in flight, so only a run submits them); *detail* says why.
    """

    database: Path
    document: str
    outcome: str
    submission_ids: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    detail: str | None = None


class CheckTargetError(ValueError):
    """``--adopt`` or ``--abandon`` names nothing the check can act on."""


@dataclass
class CheckResult:
    """The counts, the per-submission and the per-document results of a check.

    *refused* holds the databases the store refused (versionless, newer or
    unreadable), with the reason.
    """

    stats: CheckStats = field(default_factory=CheckStats)
    submissions: list[SubmissionCheck] = field(default_factory=list)
    documents: list[DocumentAdvance] = field(default_factory=list)
    refused: dict[Path, str] = field(default_factory=dict)

    def add(self, check: SubmissionCheck) -> None:
        """Record and count one submission's result."""
        self.submissions.append(check)
        self.stats.count(check.outcome)

    @property
    def had_failure(self) -> bool:
        """Whether a submission or a document failed or waits for the user."""
        return self.stats.had_failure or any(
            document.outcome == FAILED for document in self.documents
        )

    @property
    def outputs(self) -> list[Path]:
        """The files rendered, document by document."""
        return [path for document in self.documents for path in document.outputs]

    @property
    def usage(self) -> dict[str, Usage]:
        """The tokens of the finalized results per role."""
        total: dict[str, Usage] = {}
        for check in self.submissions:
            for role, usage in check.usage.items():
                total[role] = total.get(role, Usage()) + usage
        return total


def find_databases(directory: Path) -> list[Path]:
    """Return the tool's databases at or below *directory*, sorted."""
    return sorted(
        path for path in Path(directory).rglob(DATABASE_NAME) if path.is_file()
    )


@dataclass
class _Database:
    """A database found by the check, read before any write."""

    path: Path
    reader: Store
    submissions: list[Submission]
    jobs: list[BatchJob]

    @property
    def root(self) -> Path:
        return self.path.parent


@dataclass(frozen=True)
class _Targets:
    """The submissions ``--adopt`` and ``--abandon`` name: (database, id)."""

    adopt: tuple[Path, str, str] | None = None
    abandon: tuple[Path, str] | None = None

    def adopts(self, path: Path, submission_id: str) -> str | None:
        """Return the batch to adopt for this submission, if any."""
        if self.adopt is not None and self.adopt[:2] == (path, submission_id):
            return self.adopt[2]
        return None


def _read_databases(directory: Path, result: CheckResult) -> list[_Database]:
    found = []
    for path in find_databases(directory):
        try:
            reader = read_store(path.parent)
            jobs = read(path, read_batch_jobs) if reader is not None else []
        except StoreError as exc:
            logger.error("Batch check skips %s: %s", path, exc)
            result.refused[path] = str(exc)
            continue
        if reader is not None:
            found.append(_Database(path, reader, reader.submissions(_WORKED), jobs))
    return found


def _unsettled(submission: Submission) -> bool:
    """Whether *submission* is unresolved or was left without a batch id."""
    return submission.status == UNRESOLVED or (
        submission.status == "submitting" and submission.batch_id is None
    )


def _abandon_target(
    found: Sequence[_Database], submission_id: str, directory: Path
) -> tuple[Path, str]:
    for database in found:
        try:
            submission = database.reader.submission(submission_id)
        except KeyError:
            continue
        if not (_unsettled(submission) or submission.status == "submitted"):
            raise CheckTargetError(
                f"Submission {submission_id} is {submission.status}; only an "
                "unresolved or submitted submission can be abandoned"
            )
        return database.path, submission_id
    raise CheckTargetError(
        f"No submission {submission_id} in the databases below {directory}"
    )


async def _fitting(
    choices: Sequence[tuple[_Database, Submission]],
    batch_id: str,
    clients: BatchClients,
) -> list[tuple[_Database, Submission]]:
    """Keep the submissions whose time window holds *batch_id*.

    Those with the batch's request count win over the others.
    """
    fitting: list[tuple[_Database, Submission, bool]] = []
    for database, submission in choices:
        try:
            candidates = await list_candidates(submission, clients)
        except Exception as exc:
            raise CheckTargetError(
                f"Cannot list the provider's batches: {exc}"
            ) from exc
        listing = next((c for c in candidates if c.batch_id == batch_id), None)
        if listing is not None:
            fitting.append((database, submission, count_matches(listing, submission)))
    counted = [entry for entry in fitting if entry[2]]
    return [(database, submission) for database, submission, _ in counted or fitting]


async def _adopt_target(
    found: Sequence[_Database],
    batch_id: str,
    clients: BatchClients,
    directory: Path,
) -> tuple[Path, str, str]:
    provider = provider_of(batch_id)
    for database in found:
        for submission in database.submissions:
            if submission.batch_id == batch_id:
                raise CheckTargetError(
                    f"Batch {batch_id} already belongs to submission "
                    f"{submission.submission_id}"
                )
    choices = [
        (database, submission)
        for database in found
        for submission in database.submissions
        if submission.provider == provider and _unsettled(submission)
    ]
    if not choices:
        raise CheckTargetError(
            f"No unresolved {provider} submission below {directory} to adopt "
            f"batch {batch_id} into"
        )
    if len(choices) > 1:
        choices = await _fitting(choices, batch_id, clients)
    if len(choices) != 1:
        ids = ", ".join(submission.submission_id for _, submission in choices)
        raise CheckTargetError(
            f"Batch {batch_id} fits {len(choices)} unresolved submissions"
            + (f" ({ids})" if ids else "")
            + "; abandon the others first"
        )
    database, submission = choices[0]
    return database.path, submission.submission_id, batch_id


async def end_batch_jobs(store: Store, submission_id: str) -> None:
    """End the batch jobs of the documents of *submission_id*.

    Used when a submission is abandoned or cancelled, so that no check
    submits its documents' work again; the leased documents' jobs end. A
    document whose units another open or unresolved submission still holds
    keeps its job, which that submission's finalization reads.
    """
    held = store.held_leases()
    doc_ids = {key.doc_id for key in store.submission_requests(submission_id).values()}
    for doc_id in sorted(doc_ids):
        if doc_id in held and not store.blocked_units(doc_id):
            await delete_batch_job(store, held[doc_id])


async def _give_up(store: Store, submission: Submission, clients: BatchClients) -> str:
    """Abandon *submission*: an unresolved one as reconciliation does, a
    submitted one by marking it failed, so its pages are pending again.

    A submitted batch is not cancelled at the provider; its results are no
    longer collected.
    """
    if submission.status == "submitted":
        await store.mark_failed(
            submission.submission_id,
            "abandoned with batch check --abandon; its results are not collected",
        )
        detail = (
            "its pages are pending again; a run resubmits them (the provider "
            "batch is not cancelled)"
        )
    else:
        await abandon(store, submission, clients)
        detail = "its pages are pending again; a run resubmits them"
    await end_batch_jobs(store, submission.submission_id)
    return detail


async def _settle(
    store: Store,
    submission: Submission,
    clients: BatchClients,
    batch_id: str | None,
) -> tuple[Submission | None, str | None, tuple[BatchListing, ...]]:
    """Adopt *batch_id*, else reconcile; return the submission once submitted.

    Without a submitted submission the detail and the candidates explain
    what the user can do.
    """
    if batch_id is not None:
        adopted = await adopt(store, submission, batch_id, clients)
        return adopted, f"adopted batch {batch_id}", ()
    found = await reconcile(store, submission, clients)
    if found.adopted is None:
        return None, found.detail, found.candidates
    return found.submission, found.detail, ()


@dataclass
class _DatabaseCheck:
    """The check of one open database: its store, clients and batch jobs."""

    store: Store
    path: Path
    clients: BatchClients
    jobs: Mapping[int, DocumentJob]
    targets: _Targets
    finalized: list[Submission] = field(default_factory=list)

    async def check(self, submission: Submission) -> SubmissionCheck:
        """Settle, adopt, abandon or finalize one submission."""
        store, path = self.store, self.path
        submission_id = submission.submission_id

        def result(outcome: str, detail: str | None = None) -> SubmissionCheck:
            return SubmissionCheck(
                path,
                submission_id,
                submission.provider,
                outcome,
                submission.batch_id,
                detail,
            )

        abandoning = self.targets.abandon == (path, submission_id)
        batch_id = self.targets.adopts(path, submission_id)
        if submission.status == "submitting" and not abandoning and batch_id is None:
            return result(PENDING, "another run is submitting it")
        if not await lease_submission(store, submission_id):
            return result(
                PENDING, "a document of the submission is leased by another run"
            )
        try:
            if abandoning:
                detail = await _give_up(store, submission, self.clients)
                return result(ABANDONED, detail)
            note: str | None = None
            if submission.status != "submitted":
                try:
                    settled, note, candidates = await _settle(
                        store, submission, self.clients, batch_id
                    )
                except LeaseLostError:
                    raise
                except Exception as exc:  # noqa: BLE001 - a failure is reported
                    action = (
                        f"batch {batch_id} not adopted" if batch_id else "not settled"
                    )
                    return result(FAILED, f"{action}: {exc}")
                if settled is None:
                    return replace(result(UNRESOLVED, note), candidates=candidates)
                submission = settled
            return await self._finalize(submission, note)
        except LeaseLostError as exc:
            return result(FAILED, f"the lease was lost: {exc}")
        except SubmissionStateError as exc:
            return result(FAILED, str(exc))

    async def _finalize(
        self, submission: Submission, note: str | None
    ) -> SubmissionCheck:
        finalization: SubmissionFinalization = await finalize_submission(
            self.store, submission, self.clients, self.jobs
        )
        hint = None
        if finalization.outcome == FAILED and self._still_submitted(submission):
            hint = (
                "if its results cannot be recovered, give it up with "
                f"'autoexcerpter batch check --abandon {submission.submission_id}'"
            )
        parts = (note, finalization.detail, hint)
        detail = "; ".join(part for part in parts if part)
        if finalization.outcome == FINALIZED_OUTCOME:
            self.finalized.append(submission)
        return SubmissionCheck(
            self.path,
            submission.submission_id,
            submission.provider,
            finalization.outcome,
            submission.batch_id,
            detail or None,
            errors=finalization.errors,
            usage=finalization.usage,
        )

    def _still_submitted(self, submission: Submission) -> bool:
        try:
            current = self.store.submission(submission.submission_id)
        except KeyError:
            return False
        return current.status == "submitted"

    async def advance(self, job: DocumentJob) -> DocumentAdvance | None:
        """Submit a document's summaries or render it; None when it waits
        silently on its own open submissions."""
        store = self.store
        row = store.document(job.doc_id)
        if row is None:
            return None
        if store.blocked_units(job.doc_id):
            return None

        def advance(outcome: str, detail: str | None = None) -> DocumentAdvance:
            return DocumentAdvance(self.path, row.name, outcome, detail=detail)

        lease = store.held_leases().get(job.doc_id)
        if lease is None:
            lease = await store.acquire(job.doc_id)
        if lease is None:
            return advance(WAITING, "another run holds the document")
        try:
            return await _Advance(self, job, row.name, row.input_path, lease).run()
        except LeaseLostError as exc:
            return advance(FAILED, f"the lease was lost: {exc}")


def _summary_selected(unit: UnitRow) -> bool:
    """Whether ``batch check`` makes a summary: one not made yet, or stale.

    A failed summary waits for a run, so a page whose summary keeps failing
    does not hold the document back.
    """
    return unit.status == "pending" or bool(unit.stale)


@dataclass
class _Advance:
    """The advance of one document of a batch job under its lease."""

    database: _DatabaseCheck
    job: DocumentJob
    name: str
    input_path: str | None
    lease: Lease

    def _handle(self) -> DocumentHandle:
        item_job = self.job.job
        summary = (
            Stamp(item_job.summary.name, item_job.summary.provider)
            if item_job.summarize
            else None
        )
        return DocumentHandle(
            self.database.store,
            self.job.doc_id,
            self.name,
            self.lease,
            transcription=Stamp(
                item_job.transcription.name, item_job.transcription.provider
            ),
            summary=summary,
        )

    def _result(
        self,
        outcome: str,
        *,
        submission_ids: tuple[str, ...] = (),
        outputs: tuple[Path, ...] = (),
        warnings: tuple[str, ...] = (),
        detail: str | None = None,
    ) -> DocumentAdvance:
        return DocumentAdvance(
            self.database.path,
            self.name,
            outcome,
            submission_ids=submission_ids,
            outputs=outputs,
            warnings=warnings,
            detail=detail,
        )

    async def run(self) -> DocumentAdvance:
        """Advance the pages the submitting run's ``--pages`` selected (every
        page without it); the others stay pending and the document partial."""
        store, item_job = self.database.store, self.job.job
        pages = sum(
            1
            for unit in store.units(self.job.doc_id)
            if unit.kind == KIND_TRANSCRIPTION
        )
        chosen = set(item_job.selected_pages(pages))

        def selected(unit: UnitRow) -> bool:
            return unit.idx in chosen and _summary_selected(unit)

        if item_job.summarize:
            work = summary_work(store, self.job.doc_id, selected=selected)
            await write_placeholders(
                store,
                self.lease,
                work.placeholders,
                model=item_job.summary.name,
                provider=item_job.summary.provider,
            )
            if work.requests:
                return await self._submit(work)
        units = store.units(self.job.doc_id)
        relevant = [
            unit
            for unit in units
            if unit.idx in chosen
            and (
                unit.kind == KIND_TRANSCRIPTION
                or (item_job.summarize and unit.kind == KIND_SUMMARY)
            )
        ]
        waiting = [
            unit
            for unit in relevant
            if unit.status == "pending"
            or (unit.kind == KIND_SUMMARY and unit.stale and unit.status != "failed")
        ]
        if waiting:
            return self._result(
                NEEDS_RUN,
                detail=(
                    f"{len(waiting)} unit(s) are pending; run 'autoexcerpter run "
                    "--batch' on the input to submit them again"
                ),
            )
        return await self._render(relevant)

    async def _submit(self, work: SummaryWork) -> DocumentAdvance:
        job = self.job
        try:
            plan = build_plan(
                job.job.summary,
                ROLE_SUMMARY,
                context=job.context,
                transcription_format=job.transcription_format,
                caching=self.database.clients.settings.prompt_caching,
            )
            submitted = await submit_units(
                self.database.store,
                work.requests,
                plan=plan,
                clients=self.database.clients,
                label=LABEL,
            )
        except (BatchSubmissionError, ValueError) as exc:
            return self._result(FAILED, detail=str(exc))
        if submitted.status == "submitted":
            return self._result(SUBMITTED, submission_ids=submitted.submission_ids)
        accepted = (
            submitted.submission_ids[:-1]
            if submitted.status == "refused"
            else submitted.submission_ids
        )
        verb = "was refused" if submitted.status == "refused" else "is unresolved"
        return self._result(
            FAILED,
            submission_ids=accepted,
            detail=f"the summary batch {verb}: {submitted.detail or 'no detail'}",
        )

    async def _render(self, relevant: Sequence[UnitRow]) -> DocumentAdvance:
        job = self.job
        input_path = Path(self.input_path) if self.input_path else Path(self.name)
        kind = "image_folder" if input_path.suffix.lower() != ".pdf" else "pdf"
        events = _Warnings()
        files = ItemFiles(
            ItemSpec(kind, input_path),
            self.database.store.path.parent,
            job.job,
            self._handle(),
            force=job.spec.run.force,
            events=events,
        )
        if not await files.store_citations():
            return self._result(FAILED, detail="the citations were not stored")
        status = await files.write()
        outputs = tuple(files.written)
        warnings = tuple(events.messages)
        if not (status.transcription_ok and status.summaries_ok):
            return self._result(
                FAILED,
                outputs=outputs,
                warnings=warnings,
                detail="a file was not written",
            )
        await delete_batch_job(self.database.store, self.lease)
        failed = sum(1 for unit in relevant if unit.status == "failed")
        if failed:
            return self._result(
                FAILED,
                outputs=outputs,
                warnings=warnings,
                detail=(
                    f"{failed} unit(s) failed; a run of 'autoexcerpter run' on "
                    "the input retries them"
                ),
            )
        return self._result(COMPLETE, outputs=outputs, warnings=warnings)


class _Warnings(NullEvents):
    """Collects the render warnings of a document; drops the other events."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        """Keep the warning."""
        self.messages.append(message)


async def _check_database(
    database: _Database,
    clients: BatchClients,
    settings: Settings,
    wanted: set[str],
    targets: _Targets,
    result: CheckResult,
    formats: Sequence[str] | None = None,
) -> None:
    """Mark, reconcile and finalize the submissions of one database; then
    advance its batch jobs, rendered in *formats* when given."""
    async with open_store(database.root) as store:
        await store.begin_run(
            "batch check",
            spec={"directory": str(database.root), "batch_ids": sorted(wanted)},
        )
        failed = False
        try:
            async with store.heartbeat():
                for submission_id in await store.mark_stale_submissions():
                    logger.warning(
                        "Submission %s in %s was left without a provider batch id; "
                        "it is unresolved",
                        submission_id,
                        database.path,
                    )
                jobs = load_jobs(store, settings, formats=formats)
                work = _DatabaseCheck(store, database.path, clients, jobs, targets)
                for submission in store.submissions(_WORKED):
                    if (
                        submission.status == "submitted"
                        and wanted
                        and submission.batch_id not in wanted
                    ):
                        continue
                    check = await work.check(submission)
                    result.add(check)
                    failed = failed or check.outcome in (FAILED, UNRESOLVED)
                for row in read(store.path, read_batch_jobs):
                    job = jobs.get(row.doc_id)
                    if job is None:
                        advance = DocumentAdvance(
                            database.path,
                            _name(store, row.doc_id),
                            FAILED,
                            detail=(
                                f"the spec of run {row.run_id} cannot be rebuilt; "
                                "run 'autoexcerpter run' on the input"
                            ),
                        )
                    else:
                        found = await work.advance(job)
                        if found is None:
                            continue
                        advance = found
                    result.documents.append(advance)
                    failed = failed or advance.outcome == FAILED
                for submission in work.finalized:
                    await delete_provider_files(clients, submission)
        finally:
            await store.end_run(1 if failed else 0)


def _name(store: Store, doc_id: int) -> str:
    row = store.document(doc_id)
    return row.name if row is not None else str(doc_id)


async def check_batches(  # noqa: PLR0913 - the check options plus its objects
    directory: Path,
    settings: Settings,
    *,
    batch_ids: Sequence[str] = (),
    adopt: str | None = None,
    abandon: str | None = None,
    args: argparse.Namespace | None = None,
    clients: BatchClients | None = None,
    result: CheckResult | None = None,
    formats: Sequence[str] | None = None,
) -> CheckResult:
    """Check the open batch work of every database below *directory*.

    *formats*, when given, replaces the formats of the submitting runs
    (``batch check --formats``). *batch_ids*, when given, limit the
    finalization to those batches;
    unresolved submissions are reconciled and batch jobs advanced in every
    check. *adopt* records a provider batch for the unresolved submission it
    fits, *abandon* gives an unresolved submission, or a submitted one whose
    results are lost, up and ends the batch jobs of its documents. A
    database the store refuses is listed in :attr:`CheckResult.refused` and
    left alone. Only databases with a
    submission to work or a batch job are opened for writing. *args* holds
    the parsed command line, if any, for the extension's key variables.
    *clients* are the batch clients to use (new ones by default); *result*,
    when given, is filled as the check goes and returned, so its caller
    keeps what was checked before an exception.

    Raises:
        CheckTargetError: *adopt* or *abandon* fits no submission, or more
            than one.
    """
    directory = Path(directory)
    result = result if result is not None else CheckResult()
    if clients is None:
        clients = BatchClients(settings, ROLE, args=args)
    found = _read_databases(directory, result)
    targets = _Targets(
        adopt=(
            await _adopt_target(found, adopt, clients, directory) if adopt else None
        ),
        abandon=_abandon_target(found, abandon, directory) if abandon else None,
    )
    wanted = set(batch_ids)
    for database in found:
        if not (database.submissions or database.jobs):
            continue
        await _check_database(
            database, clients, settings, wanted, targets, result, formats
        )
    return result


# =============================================================================
# batch check --wait
# =============================================================================

type _Key = tuple[Path, str]
"""A submission or document: its database and its id or name."""


def pending_work(result: CheckResult) -> int:
    """Return the batch work of one check that a later check can advance.

    That is every pending submission (its batch runs, a provider error was
    retryable, another run submits it or holds one of its documents) and
    every document whose summaries were just submitted or that another run
    holds. Failed, unresolved and abandoned submissions and the documents
    that need a run are left to the user.
    """
    documents = sum(
        1 for advance in result.documents if advance.outcome in (SUBMITTED, WAITING)
    )
    return result.stats.pending + documents


@dataclass
class WaitLedger:
    """What the checks of a wait saw, recorded check by check.

    *submissions* and *documents* hold every (database, id or name) a check
    listed, in the order first seen; *failed* and *failed_documents* those a
    check reported failed, *abandoned* the submissions it abandoned.
    *opened* holds the summary submissions the checks opened, *outputs* the
    files written, *usage* the tokens of each finalized submission and
    *refused* the databases the store refused. *first* is the first check's
    result, the one that acts on ``--adopt`` and ``--abandon``.
    """

    submissions: dict[_Key, None] = field(default_factory=dict)
    failed: set[_Key] = field(default_factory=set)
    abandoned: set[_Key] = field(default_factory=set)
    documents: dict[_Key, None] = field(default_factory=dict)
    failed_documents: set[_Key] = field(default_factory=set)
    opened: dict[_Key, None] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)
    usage: dict[_Key, Mapping[str, Usage]] = field(default_factory=dict)
    refused: dict[Path, str] = field(default_factory=dict)
    first: CheckResult | None = None

    def record(self, result: CheckResult) -> None:
        """Add what one check reported, also when it ended early."""
        if self.first is None:
            self.first = result
        for check in result.submissions:
            key = (check.database, check.submission_id)
            self.submissions.setdefault(key, None)
            if check.outcome == FAILED:
                self.failed.add(key)
            elif check.outcome == ABANDONED:
                self.abandoned.add(key)
            if check.usage:
                self.usage[key] = check.usage
        for advance in result.documents:
            key = (advance.database, advance.document)
            self.documents.setdefault(key, None)
            if advance.outcome == FAILED:
                self.failed_documents.add(key)
            for submission_id in advance.submission_ids:
                opened = (advance.database, submission_id)
                self.submissions.setdefault(opened, None)
                if advance.outcome == SUBMITTED:
                    self.opened.setdefault(opened, None)
            for path in advance.outputs:
                if path not in self.outputs:
                    self.outputs.append(path)
        self.refused.update(result.refused)


@dataclass
class WaitProgress:
    """The state of a wait, updated after every check for its caller.

    *passes* counts the checks run, *started* is the wait clock's reading
    at the first one, *stop* says why the wait ended (``settled`` or
    ``timeout``; None while it runs) and *last* is the latest check's
    result.
    """

    ledger: WaitLedger = field(default_factory=WaitLedger)
    passes: int = 0
    started: float | None = None
    stop: str | None = None
    last: CheckResult | None = None

    def waited_seconds(self) -> int:
        """Return the whole seconds since the first check began."""
        if self.started is None:
            return 0
        return int(_clock() - self.started)


@dataclass(frozen=True)
class RunHint:
    """A document that needs a run, and the command that submits its pages."""

    database: Path
    document: str
    command: str


@dataclass
class WaitTotals:
    """The final counts of a wait, read from the stores after it ended.

    *stats* counts the submissions by their current state; *complete*,
    *failed_documents* and *needs_run* the documents. *submitted* counts the
    summary submissions the wait opened, *outputs* the files it wrote and
    *usage* the tokens of the results it stored, per role.
    """

    stats: CheckStats = field(default_factory=CheckStats)
    complete: int = 0
    failed_documents: int = 0
    needs_run: list[RunHint] = field(default_factory=list)
    submitted: int = 0
    outputs: list[Path] = field(default_factory=list)
    usage: dict[str, Usage] = field(default_factory=dict)
    refused: dict[Path, str] = field(default_factory=dict)

    @property
    def had_failure(self) -> bool:
        """Whether something failed, waits for the user or needs a run."""
        return (
            self.stats.had_failure or self.failed_documents > 0 or bool(self.needs_run)
        )


def _submission_outcome(key: _Key, status: str, ledger: WaitLedger) -> str | None:
    """Return the final outcome of a submission from its current *status*.

    A failure a check reported stays a failure unless the store shows the
    submission finalized since; a cancelled submission counts nowhere.
    """
    if status == "finalized":
        return FINALIZED_OUTCOME
    if key in ledger.abandoned:
        return ABANDONED
    if key in ledger.failed or status == FAILED:
        return FAILED
    if status == UNRESOLVED:
        return UNRESOLVED
    if status in ("submitting", "submitted"):
        return PENDING
    return None


def _run_command(database: Path, input_path: str | None, name: str) -> str:
    source = input_path or name
    return f'autoexcerpter run --input "{source}" --output "{database.parent}" --batch'


def _document_outcome(
    store: Store, key: _Key, jobs: set[int], ledger: WaitLedger
) -> tuple[str, str] | None:
    """Return the final outcome of a document and its run command, if any.

    A document whose batch job ended is complete unless units failed or are
    pending; one with a job failed if a check said so, waits on its open
    submissions, or needs a run when its pending units have none.
    """
    database, name = key
    row = store.document(name)
    if row is None:
        return None
    units = store.units(row.doc_id)
    failed = any(unit.status == FAILED for unit in units)
    waiting = any(
        unit.status == PENDING
        or (unit.kind == KIND_SUMMARY and unit.stale and unit.status != FAILED)
        for unit in units
    )
    command = _run_command(database, row.input_path, row.name)
    if row.doc_id not in jobs:
        if failed:
            return FAILED, command
        return (NEEDS_RUN, command) if waiting else (COMPLETE, command)
    if key in ledger.failed_documents:
        return FAILED, command
    if waiting and not store.blocked_units(row.doc_id):
        return NEEDS_RUN, command
    return WAITING, command


def wait_totals(ledger: WaitLedger) -> WaitTotals:
    """Build the final counts of a wait from the stores' current state.

    Every submission and document a check listed is read again, also those
    a later check no longer listed. A database the store refuses now is
    added to :attr:`WaitTotals.refused`.
    """
    totals = WaitTotals(
        submitted=len(ledger.opened),
        outputs=list(ledger.outputs),
        refused=dict(ledger.refused),
    )
    for by_role in ledger.usage.values():
        for role, usage in by_role.items():
            totals.usage[role] = totals.usage.get(role, Usage()) + usage
    stores: dict[Path, tuple[Store, set[int]] | None] = {}

    def store_of(database: Path) -> tuple[Store, set[int]] | None:
        if database not in stores:
            try:
                store = read_store(database.parent)
                jobs = read(database, read_batch_jobs) if store is not None else []
            except StoreError as exc:
                totals.refused[database] = str(exc)
                store = None
            stores[database] = (
                None if store is None else (store, {job.doc_id for job in jobs})
            )
        return stores[database]

    for key in ledger.submissions:
        found = store_of(key[0])
        if found is None:
            continue
        try:
            status = found[0].submission(key[1]).status
        except KeyError:
            continue
        outcome = _submission_outcome(key, status, ledger)
        if outcome is not None:
            totals.stats.count(outcome)
    for key in ledger.documents:
        found = store_of(key[0])
        document = (
            None
            if found is None
            else _document_outcome(found[0], key, found[1], ledger)
        )
        if document is None:
            continue
        outcome, command = document
        if outcome == COMPLETE:
            totals.complete += 1
        elif outcome == FAILED:
            totals.failed_documents += 1
        elif outcome == NEEDS_RUN:
            totals.needs_run.append(RunHint(key[0], key[1], command))
    return totals


def _ignore(*_: object) -> None:
    return None


def _retryable(exc: BaseException) -> bool:
    """Whether a check that raised *exc* is worth repeating."""
    if isinstance(exc, (CheckTargetError, StoreError)):
        return False
    return classify_error(exc).retryable


def _acted(result: CheckResult, adopt: str | None, abandon: str | None) -> bool:
    """Whether a check reached the submission ``--adopt`` or ``--abandon`` names."""
    return any(
        (adopt is not None and check.batch_id == adopt)
        or (abandon is not None and check.submission_id == abandon)
        for check in result.submissions
    )


async def check_until_settled(  # noqa: PLR0913 - the check options plus hooks
    directory: Path,
    settings: Settings,
    limits: WaitLimits,
    *,
    batch_ids: Sequence[str] = (),
    adopt: str | None = None,
    abandon: str | None = None,
    args: argparse.Namespace | None = None,
    progress: WaitProgress | None = None,
    on_pass: Callable[[int, CheckResult], None] = _ignore,
    on_wait: Callable[[WaitTick], None] = _ignore,
    on_retry: Callable[[int, Exception], None] = _ignore,
    formats: Sequence[str] | None = None,
) -> WaitProgress:
    """Check every *limits* interval until no batch work a check can advance
    is pending; start no check after the timeout.

    The options are those of :func:`check_batches`; *adopt* and *abandon*
    act in the first check that reaches their submission. *progress* is
    updated after every check, so a caller that is interrupted still reads
    what the wait saw. *on_pass* receives each check's result, *on_wait*
    each sleep and *on_retry* each check that raised a retryable error.
    The batch clients are built once and closed when the wait ends.

    Raises:
        CheckTargetError: *adopt* or *abandon* fits no submission, or more
            than one.
    """
    progress = progress if progress is not None else WaitProgress()
    clients = BatchClients(settings, ROLE, args=args)
    clock = _clock
    progress.started = clock()
    acted = False

    async def one_pass(number: int) -> CheckResult:
        nonlocal acted
        result = CheckResult()
        try:
            await check_batches(
                directory,
                settings,
                batch_ids=batch_ids,
                adopt=None if acted else adopt,
                abandon=None if acted else abandon,
                args=args,
                clients=clients,
                result=result,
                formats=formats,
            )
        finally:
            progress.ledger.record(result)
            progress.passes = number
            progress.last = result
            acted = acted or _acted(result, adopt, abandon)
        acted = True
        logger.info("Batch check %d: %d pending", number, pending_work(result))
        on_pass(number, result)
        return result

    try:
        outcome: PollResult[CheckResult] = await poll_until_settled(
            one_pass,
            pending_work,
            limits,
            on_wait=on_wait,
            on_retry=on_retry,
            retryable=_retryable,
            clock=clock,
            sleep=_sleep,
        )
    finally:
        clients.close()
    progress.stop = outcome.stop.value
    return progress
