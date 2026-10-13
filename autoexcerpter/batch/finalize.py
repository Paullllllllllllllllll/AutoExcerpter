"""Finalize submitted batches of page transcriptions and summaries.

:func:`finalize_submission` finalizes one submission of the store for
``batch check``: it asks the provider for the batch status and, once the
batch has finished, downloads the results. Every result record becomes one
``responses`` row holding the record as downloaded; then every unit of the
submission is written in a transaction that checks its document's lease.
A transcription result is read as a synchronous page ends
(:func:`~autoexcerpter.llm.transcription.answer_entry`, then
:func:`~autoexcerpter.pipeline.text_cleaner.clean_transcription`); a summary
result as a synchronous summary ends
(:func:`~autoexcerpter.llm.summary.summary_entry`), its row written to
``summaries`` with the unit. A failed request, a unit without a result and
an answer the provider's content filter stopped or the model refused fail
the unit with an error placeholder. Last, the submission is marked
finalized. A batch that ended without results marks the submission failed,
so its units are pending again. A provider error that outlasts the retries
of :class:`BatchClients` ends in an outcome and changes nothing.
:func:`delete_provider_files` deletes the batch's input and result files
once the caller has advanced the documents.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from autoexcerpter.batch.clients import (
    BatchClients,
    MissingKeyError,
    results_of,
    status_of,
)
from autoexcerpter.batch.jobs import DocumentJob
from autoexcerpter.common.batch import (
    MISSING_RESULT_ERROR,
    BatchHandle,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    supports_batch,
)
from autoexcerpter.common.responses import ResponseOutcome, inspect_response
from autoexcerpter.common.retry import ErrorKind, classify_error
from autoexcerpter.common.store import (
    LeaseLostError,
    ResponseRow,
    Store,
    Submission,
    UnitKey,
    UnitResult,
    UnitRow,
)
from autoexcerpter.common.structured import validate_json
from autoexcerpter.common.usage import Usage
from autoexcerpter.llm.summary import (
    SUMMARY_REQUIRED_KEYS,
    failed_summary,
    summary_entry,
)
from autoexcerpter.llm.transcription import answer_entry
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    transcription_outcome,
    write_summary,
)
from autoexcerpter.pipeline.pages import (
    build_summary_result,
    summary_row,
    warn_on_page_number_mismatch,
)
from autoexcerpter.pipeline.text_cleaner import clean_transcription

__all__ = [
    "FAILED",
    "FINALIZED_OUTCOME",
    "MISSING_TEXT",
    "PENDING",
    "SKIPPED",
    "SubmissionFinalization",
    "delete_provider_files",
    "finalize_submission",
    "placeholder",
    "submission_handle",
]

logger = logging.getLogger(__name__)

FINALIZED_OUTCOME = "finalized"
PENDING = "pending"
FAILED = "failed"
SKIPPED = "skipped"

MISSING_TEXT = "[transcription error: no result returned]"
_RETRYABLE = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.SERVER_ERROR,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
    }
)
_GOOGLE_STOPS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}
)


def placeholder(provider: str, item: BatchResultItem) -> str:
    """Return the error text of a failed result.

    OpenAI results name the HTTP status of an error in the output file and
    the error code; the other providers give their error message.
    """
    if item.origin == "missing":
        return MISSING_TEXT
    if provider != "openai":
        return f"[transcription error: {item.error or 'unknown error'}]"
    parts = ["[transcription error"]
    status = item.response.get("status_code") if item.origin == "output" else None
    if isinstance(status, int) and status != 200:
        parts.append(f"status {status}")
    if item.error_code:
        parts.append(f"code {item.error_code}")
    if item.error and item.error != f"HTTP {status}":
        parts.append(item.error)
    return "; ".join(parts) + "]"


def _openai_blocks(body: Mapping[str, Any]) -> list[Any]:
    blocks: list[Any] = []
    for output in body.get("output") or []:
        if isinstance(output, Mapping):
            blocks.extend(output.get("content") or [])
    return blocks


def stopped(provider: str, item: BatchResultItem) -> ResponseOutcome | None:
    """Return how a result ended when a content filter or a refusal stopped it.

    The provider's status fields are read as :func:`inspect_response` reads
    a synchronous response: OpenAI's ``incomplete_details`` and refusal
    blocks, Anthropic's ``stop_reason`` and Gemini's safety finish reasons.
    """
    response = item.response
    if provider == "google":
        candidates = response.get("candidates")
        first = candidates[0] if isinstance(candidates, list) and candidates else {}
        reason = None
        if isinstance(first, Mapping):
            reason = first.get("finish_reason") or first.get("finishReason")
        if isinstance(reason, str) and reason.upper() in _GOOGLE_STOPS:
            return ResponseOutcome("content_filter", f"finish_reason={reason}")
        return None
    if provider == "openai":
        body = response.get("body")
        meta = dict(body) if isinstance(body, Mapping) else {}
        content = _openai_blocks(meta)
    else:
        message = response.get("message")
        meta = dict(message) if isinstance(message, Mapping) else {}
        content = list(meta.get("content") or [])
    shim = SimpleNamespace(
        response_metadata=meta, content=content, additional_kwargs={}
    )
    outcome = inspect_response(shim)
    return outcome if outcome.stops_page else None


def _usage_of(items: Sequence[BatchResultItem]) -> Usage:
    total = Usage()
    for item in items:
        total = total + item.usage
    return total


def _error_suffix(info: BatchStatusInfo) -> str:
    return f": {info.error_message}" if info.error_message else ""


@dataclass(frozen=True)
class SubmissionFinalization:
    """The outcome of finalizing one submission of the store.

    *outcome* is ``finalized``, ``pending`` (still running, or a retryable
    provider error), ``failed`` (the batch ended without results, or a
    provider error that is not retryable) or ``skipped`` (the submission is
    not submitted); *detail* says why. *documents* are the ids of the
    documents whose units were written and *errors* the number of units
    written failed. *usage* holds the tokens per role.
    """

    submission_id: str
    outcome: str
    detail: str | None = None
    provider_status: str | None = None
    documents: tuple[int, ...] = ()
    units: int = 0
    errors: int = 0
    usage: Mapping[str, Usage] = field(default_factory=dict)


def submission_handle(
    submission: Submission, custom_ids: Iterable[str] = ()
) -> BatchHandle:
    """Return the backend handle of *submission*'s provider batch.

    The upload reference becomes the input file the backend deletes in its
    cleanup.
    """
    metadata: dict[str, Any] = {}
    if submission.upload_ref:
        key = "input_file_name" if submission.provider == "google" else "input_file_id"
        metadata[key] = submission.upload_ref
    return BatchHandle(
        provider=submission.provider,
        batch_id=submission.batch_id or "",
        custom_ids=list(custom_ids),
        metadata=metadata,
    )


def _counts(info: BatchStatusInfo) -> dict[str, int]:
    return {
        "total": info.total_requests,
        "completed": info.completed_requests,
        "failed": info.failed_requests,
        "pending": info.pending_requests,
    }


def _status_label(info: BatchStatusInfo) -> str:
    return info.provider_status or info.status.value


def _usage_record(usage: Usage) -> dict[str, int]:
    """Return *usage* in the shape of a unit's usage record."""
    return usage.as_dict()


def _http_status(item: BatchResultItem) -> int | None:
    status = item.response.get("status_code")
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def _missing(custom_id: str) -> BatchResultItem:
    return BatchResultItem(
        custom_id=custom_id,
        error=MISSING_RESULT_ERROR,
        error_code="missing",
        origin="missing",
    )


def _role(unit: UnitKey | None, fallback: str) -> str:
    return unit.kind if unit is not None else fallback


async def _append_records(
    store: Store,
    submission: Submission,
    requests: Mapping[str, UnitKey],
    items: Sequence[BatchResultItem],
    role: str,
) -> tuple[dict[str, BatchResultItem], dict[str, int]]:
    """Store every result record as one response row, in provider order.

    A record without a custom id (a malformed result line) is stored with
    neither a custom id nor a unit and is never attributed to a unit.
    Returns the first result of each requested custom id and the response
    id of each stored record.
    """
    results: dict[str, BatchResultItem] = {}
    response_ids: dict[str, int] = {}
    for item in items:
        if not item.custom_id:
            if item.raw is not None:
                logger.warning(
                    "Batch %s of submission %s returned a record without a "
                    "custom id; stored without a unit: %s",
                    submission.batch_id,
                    submission.submission_id,
                    item.error or "no custom id",
                )
                await store.append_response(
                    ResponseRow(
                        provider=submission.provider,
                        model=submission.model,
                        role=role,
                        attempt=1,
                        submission_id=submission.submission_id,
                        raw=item.raw,
                        raw_kind=item.raw_kind,
                        error=item.error,
                        error_type=item.error_code,
                    )
                )
            continue
        unit = requests.get(item.custom_id)
        if unit is None:
            logger.warning(
                "Batch %s returned %r, which submission %s did not request",
                submission.batch_id,
                item.custom_id,
                submission.submission_id,
            )
        if item.raw is not None:
            response_ids.setdefault(
                item.custom_id,
                await store.append_response(
                    ResponseRow(
                        provider=submission.provider,
                        model=submission.model,
                        unit=unit,
                        role=_role(unit, role),
                        attempt=1,
                        submission_id=submission.submission_id,
                        custom_id=item.custom_id,
                        raw=item.raw,
                        raw_kind=item.raw_kind,
                        usage=_usage_record(item.usage),
                        error=item.error,
                        error_type=item.error_code,
                        http_status=_http_status(item),
                    )
                ),
            )
        if unit is not None:
            results.setdefault(item.custom_id, item)
    return results, response_ids


@dataclass(frozen=True)
class _Answer:
    """What one result yields: the unit's content or why it failed."""

    text: str | None
    error: str | None = None
    error_type: str | None = None


def _answer(provider: str, item: BatchResultItem) -> _Answer:
    """Return the answer text of a result, or why the unit fails."""
    if not item.success:
        return _Answer(None, placeholder(provider, item), item.error_code)
    ended = stopped(provider, item)
    if ended is not None:
        return _Answer(None, ended.message, ended.kind)
    return _Answer(item.text or "")


@dataclass
class _Writer:
    """Writes the units of one submission under the leases the store holds."""

    store: Store
    submission: Submission
    jobs: Mapping[int, DocumentJob]
    rows: Mapping[UnitKey, UnitRow]

    def _result(
        self,
        key: UnitKey,
        item: BatchResultItem,
        response_id: int | None,
        **fields: Any,
    ) -> UnitResult:
        row = self.rows.get(key)
        return UnitResult(
            kind=key.kind,
            idx=key.idx,
            model=self.submission.model,
            provider=self.submission.provider,
            label=row.label if row is not None else None,
            source_index=row.source_index if row is not None else None,
            usage=None if item.usage.is_empty else _usage_record(item.usage),
            response_id=response_id,
            attempts=1,
            **fields,
        )

    async def transcription(
        self, key: UnitKey, item: BatchResultItem, response_id: int | None
    ) -> bool:
        """Write a transcription unit; return whether it failed."""
        row = self.rows.get(key)
        label = (row.label if row is not None else None) or f"page index {key.idx}"
        job = self.jobs.get(key.doc_id)
        plain = job is not None and job.transcription_format == "text"
        answer = _answer(self.submission.provider, item)
        if answer.text is None:
            text, error, error_type = str(answer.error), answer.error, answer.error_type
        else:
            entry = answer_entry(answer.text, label, plain=plain)
            error = entry.get("error")
            error_type = entry.get("error_type")
            text = str(entry["transcription"])
            if error is None:
                text = clean_transcription(text)
        lease = self.store.held_leases()[key.doc_id]
        await self.store.write_unit(
            lease,
            self._result(
                key,
                item,
                response_id,
                status="failed" if error is not None else "done",
                content=text,
                outcome=transcription_outcome(text, error),
                error=error,
                error_type=error_type if error is not None else None,
                provenance=row.provenance if row is not None else None,
            ),
        )
        return error is not None

    async def summary(
        self, key: UnitKey, item: BatchResultItem, response_id: int | None
    ) -> bool:
        """Write a summary unit and its ``summaries`` row; return whether it
        failed."""
        row = self.rows.get(key)
        label = (row.label if row is not None else None) or f"page index {key.idx}"
        page = key.idx + 1
        job = self.jobs.get(key.doc_id)
        response_format = job.summary_format if job is not None else "schema"
        provider = self.submission.provider
        answer = _answer(provider, item)
        if answer.text is None:
            payload = failed_summary(
                page, str(answer.error), answer.error_type or "batch_error", provider
            )
        elif not answer.text:
            payload = failed_summary(page, "empty response", "empty_response", provider)
        else:
            data, error = validate_json(answer.text, SUMMARY_REQUIRED_KEYS)
            payload = summary_entry(
                page,
                text=answer.text,
                data=data,
                error=error,
                response_format=response_format,
                provider=provider,
            )
            if "error" not in payload:
                transcription = self.rows.get(
                    UnitKey(key.doc_id, KIND_TRANSCRIPTION, key.idx)
                )
                warn_on_page_number_mismatch(
                    payload,
                    (transcription.content or "") if transcription else "",
                    label,
                )
        summary_error = payload.get("error")
        entry = build_summary_result(key.idx, label, payload, page, summary_error)
        failed = "error" in entry
        lease = self.store.held_leases()[key.doc_id]
        await write_summary(
            self.store,
            lease,
            self._result(
                key,
                item,
                response_id,
                status="failed" if failed else "done",
                content=None,
                outcome="failed" if failed else "ok",
                error=entry.get("error") if failed else None,
                error_type=entry.get("error_type") if failed else None,
            ),
            summary_row(key.idx, entry),
        )
        return failed


async def _write_units(
    store: Store,
    submission: Submission,
    requests: Mapping[str, UnitKey],
    results: Mapping[str, BatchResultItem],
    response_ids: Mapping[str, int],
    jobs: Mapping[int, DocumentJob],
) -> tuple[int, dict[str, Usage]]:
    """Write every requested unit; return the error count and usage per role."""
    doc_ids = sorted({key.doc_id for key in requests.values()})
    held = store.held_leases()
    for doc_id in doc_ids:
        if doc_id not in held:
            raise LeaseLostError(f"Document {doc_id} is not leased by this run.")
    rows: dict[UnitKey, UnitRow] = {
        row.key: row for doc_id in doc_ids for row in store.units(doc_id)
    }
    writer = _Writer(store, submission, jobs, rows)
    errors = 0
    usage: dict[str, Usage] = {}
    for custom_id, key in requests.items():
        item = results.get(custom_id) or _missing(custom_id)
        usage[key.kind] = usage.get(key.kind, Usage()) + item.usage
        response_id = response_ids.get(custom_id)
        if key.kind == KIND_SUMMARY:
            errors += await writer.summary(key, item, response_id)
        else:
            errors += await writer.transcription(key, item, response_id)
    return errors, usage


def _submission_failure(
    submission_id: str, exc: Exception, what: str
) -> SubmissionFinalization:
    """Return the outcome of a provider call that failed after its retries.

    A missing key and a non-retryable error fail the submission; a
    retryable error leaves it pending for the next check.
    """
    if isinstance(exc, MissingKeyError):
        return SubmissionFinalization(submission_id, FAILED, str(exc))
    outcome = PENDING if classify_error(exc) in _RETRYABLE else FAILED
    logger.warning("%s: %s: %s", what, type(exc).__name__, exc)
    return SubmissionFinalization(submission_id, outcome, f"{what}: {exc}")


async def finalize_submission(
    store: Store,
    submission: Submission,
    clients: BatchClients,
    jobs: Mapping[int, DocumentJob],
) -> SubmissionFinalization:
    """Finalize *submission* from its provider batch into the store.

    The store must hold the lease of every document the submission covers.
    *jobs* are the store's batch jobs; a document's job decides how its
    answers are read. Nothing is written until the batch has finished and
    its results are downloaded, apart from the provider status of each
    check.

    Raises:
        LeaseLostError: A document's lease was lost before its units were
            written; the submission stays submitted for the next check.
    """
    submission_id = submission.submission_id
    if submission.status != "submitted" or not submission.batch_id:
        return SubmissionFinalization(
            submission_id, SKIPPED, f"submission is {submission.status}"
        )
    provider = submission.provider
    if not supports_batch(provider):
        return SubmissionFinalization(
            submission_id, FAILED, f"provider {provider!r} has no batch API"
        )
    requests = store.submission_requests(submission_id)
    handle = submission_handle(submission, requests)
    try:
        info = await clients.call(
            provider,
            status_of(handle),
            model=submission.model,
            label=f"status of batch {handle.batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - classified by _submission_failure
        return _submission_failure(
            submission_id, exc, f"status of batch {handle.batch_id} not retrieved"
        )
    status = _status_label(info)
    await store.record_check(
        submission_id, provider_status=status, counts=_counts(info)
    )
    if not info.is_terminal:
        detail = f"batch {handle.batch_id} {status}{_error_suffix(info)}"
        return SubmissionFinalization(submission_id, PENDING, detail, status)
    if info.status is not BatchStatus.COMPLETED and not info.results_available:
        reason = f"batch {handle.batch_id} {info.status.value}{_error_suffix(info)}"
        await store.mark_failed(submission_id, reason)
        return SubmissionFinalization(submission_id, FAILED, reason, status)
    try:
        items = await clients.call(
            provider,
            results_of(handle),
            usage=_usage_of,
            model=submission.model,
            label=f"results of batch {handle.batch_id}",
        )
    except Exception as exc:  # noqa: BLE001 - classified by _submission_failure
        return _submission_failure(
            submission_id, exc, f"results of batch {handle.batch_id} not downloaded"
        )
    results, response_ids = await _append_records(
        store, submission, requests, items, clients.role
    )
    errors, usage = await _write_units(
        store, submission, requests, results, response_ids, jobs
    )
    await store.mark_finalized(
        submission_id, provider_status=status, counts=_counts(info)
    )
    documents = tuple(sorted({key.doc_id for key in requests.values()}))
    logger.info(
        "Finalized submission %s from batch %s: %d unit(s), %d error(s)",
        submission_id,
        handle.batch_id,
        len(requests),
        errors,
    )
    return SubmissionFinalization(
        submission_id,
        FINALIZED_OUTCOME,
        None,
        status,
        documents,
        len(requests),
        errors,
        usage,
    )


async def delete_provider_files(clients: BatchClients, submission: Submission) -> None:
    """Delete the input and result files of *submission*'s batch.

    Call it after the results are stored; the store then holds every
    response. A failure is logged.
    """
    if not submission.batch_id:
        return
    await clients.cleanup(submission_handle(submission), model=submission.model)
