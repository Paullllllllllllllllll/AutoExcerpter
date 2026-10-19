"""The report lines of ``batch check`` and ``batch cancel``.

The CLI writes them to stderr and the guided UI to its console, so both
front ends report a check or a cancel in the same words. ``[ERROR]`` and
``[WARNING]`` prefixes mark the lines that report a failure or something
the user has to act on.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from autoexcerpter.batch.cancel import CancelReport
from autoexcerpter.batch.check import (
    ABANDONED,
    COMPLETE,
    NEEDS_RUN,
    PARTIAL,
    UNRESOLVED,
    CheckResult,
    DocumentAdvance,
    SubmissionCheck,
    WaitTotals,
)
from autoexcerpter.batch.finalize import FAILED, FINALIZED_OUTCOME, PENDING, SKIPPED
from autoexcerpter.common.polling import WaitTick
from autoexcerpter.common.retry import classify_error

__all__ = [
    "cancel_lines",
    "check_lines",
    "complete_documents",
    "pending_lines",
    "retry_line",
    "submitted_batches",
    "wait_end_lines",
    "wait_line",
]


def submitted_batches(result: CheckResult) -> int:
    """Return the number of summary submissions the check opened."""
    return sum(
        len(advance.submission_ids)
        for advance in result.documents
        if advance.outcome == "submitted"
    )


def complete_documents(result: CheckResult) -> int:
    """Return the number of documents the check rendered complete."""
    return sum(1 for advance in result.documents if advance.outcome == COMPLETE)


def _partial_count(count: int) -> str:
    """Return the words of the documents ``--pages`` left partial, if any."""
    return f", {count} left partial by --pages" if count else ""


def _label(check: SubmissionCheck) -> str:
    batch = f" batch {check.batch_id}" if check.batch_id else ""
    return (
        f"{check.database.parent.name}/{check.database.name}: submission "
        f"{check.submission_id} ({check.provider}{batch})"
    )


def _submission_lines(check: SubmissionCheck) -> list[str]:
    label = _label(check)
    if check.outcome == FINALIZED_OUTCOME:
        lines = [f"Finalized {label}"]
        if check.detail:
            lines.append(f"{label}: {check.detail}")
        if check.errors:
            lines.append(
                f"[WARNING] {check.errors} unit(s) of {label} failed in the batch; "
                "a later 'autoexcerpter run' retries them."
            )
        return lines
    if check.outcome == PENDING:
        return [f"{label}: not finished yet ({check.detail})"]
    if check.outcome == FAILED:
        return [f"[ERROR] {label}: {check.detail}"]
    if check.outcome == UNRESOLVED:
        lines = [f"[WARNING] {label} is unresolved: {check.detail}"]
        for listing in check.candidates:
            created = (
                listing.created_at.isoformat() if listing.created_at else "unknown"
            )
            count = (
                "unknown" if listing.request_count is None else listing.request_count
            )
            lines.append(
                f"  candidate {listing.batch_id}: created {created}, {count} "
                f"request(s), {listing.provider_status or listing.status.value}"
            )
        return lines
    if check.outcome in (ABANDONED, SKIPPED):
        return [f"{label}: {check.outcome}: {check.detail}"]
    return [f"{label}: {check.outcome}"]


def _document_lines(advance: DocumentAdvance) -> list[str]:
    label = f"{advance.database.parent.name}/{advance.document}"
    lines = [f"[WARNING] {label}: {warning}" for warning in advance.warnings]
    if advance.outcome == "submitted":
        count = len(advance.submission_ids)
        lines.append(f"{label}: summaries submitted in {count} batch(es)")
    elif advance.outcome in (COMPLETE, PARTIAL):
        outputs = ", ".join(path.name for path in advance.outputs) or "no file"
        state = "complete" if advance.outcome == COMPLETE else "left partial by --pages"
        lines.append(f"{label}: {state} -> {outputs}")
    elif advance.outcome == FAILED:
        lines.append(f"[ERROR] {label}: {advance.detail}")
    elif advance.outcome == NEEDS_RUN:
        lines.append(f"{label}: needs a run: {advance.detail}")
    else:
        lines.append(f"{label}: {advance.outcome}: {advance.detail}")
    return lines


def check_lines(
    result: CheckResult, *, pending: bool = True, totals: bool = True
) -> list[str]:
    """Return one line per refused database, submission and document, then
    the totals.

    Without *pending* the submissions that are not finished yet are left
    out; without *totals* the totals line.
    """
    lines = [f"[ERROR] {path}: {reason}" for path, reason in result.refused.items()]
    for check in result.submissions:
        if pending or check.outcome != PENDING:
            lines += _submission_lines(check)
    for advance in result.documents:
        lines += _document_lines(advance)
    if not totals:
        return lines
    stats = result.stats
    partial = sum(1 for advance in result.documents if advance.outcome == PARTIAL)
    lines.append(
        f"Batch check: {stats.finalized} finalized, {stats.pending} pending, "
        f"{stats.failed} failed, {stats.unresolved} unresolved; "
        f"{submitted_batches(result)} summary batch(es) submitted, "
        f"{complete_documents(result)} document(s) complete"
        f"{_partial_count(partial)}."
    )
    return lines


def pending_lines(result: CheckResult) -> list[str]:
    """Return the lines of the submissions that are not finished yet."""
    return [
        line
        for check in result.submissions
        if check.outcome == PENDING
        for line in _submission_lines(check)
    ]


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}"


def wait_line(tick: WaitTick, now: datetime | None = None) -> str:
    """Return the line written before a wait sleeps.

    It begins with ``Waiting for batches``, which the job launcher reads;
    it never says "Waiting until", which marks a budget wait there.
    """
    at = (now or datetime.now()) + timedelta(seconds=tick.next_in)
    return (
        f"Waiting for batches: {tick.pending} pending; next check at "
        f"{at:%H:%M} (check {tick.passes}; {_hours(tick.elapsed)} of "
        f"{tick.timeout / 3600:g} h)."
    )


def retry_line(number: int, exc: Exception) -> str:
    """Return the warning of a check that raised a retryable error."""
    return (
        f"[WARNING] Check {number} failed ({classify_error(exc)}): {exc}; "
        "checking again."
    )


def wait_end_lines(
    totals: WaitTotals, *, passes: int, waited: float, stop: str
) -> list[str]:
    """Return the closing lines of a wait: totals, documents that need a run,
    and how the wait ended."""
    stats = totals.stats
    lines = [f"[ERROR] {path}: {reason}" for path, reason in totals.refused.items()]
    lines.append(
        f"Batch check: {stats.finalized} finalized, {stats.pending} pending, "
        f"{stats.failed} failed, {stats.unresolved} unresolved; "
        f"{totals.submitted} summary batch(es) submitted, "
        f"{totals.complete} document(s) complete{_partial_count(totals.partial)}, "
        f"{totals.failed_documents} failed, {len(totals.needs_run)} need a run."
    )
    for hint in totals.needs_run:
        lines.append(
            f"[WARNING] {hint.database.parent.name}/{hint.document} needs a run: "
            f"{hint.command}"
        )
    lines.append(f"Stopped after {passes} check(s) and {_hours(waited)} h: {stop}.")
    return lines


def cancel_lines(report: CancelReport) -> list[str]:
    """Return the lines of a cancel: listing, refusals, outcomes, totals."""
    lines: list[str] = []
    listing = report.listing
    if listing is not None:
        lines += [
            f"{provider}: key variable not set; not listed."
            for provider in listing.skipped
        ]
        lines += [
            f"[ERROR] {provider}: listing failed: {error}"
            for provider, error in listing.errors.items()
        ]
        lines.append(f"Found {len(listing.batches)} unfinished batch(es) to cancel.")
    lines += [f"[ERROR] {path}: {reason}" for path, reason in report.refused.items()]
    lines += [f"Not cancelled: {skipped}" for skipped in report.skipped]
    for outcome in report.outcomes:
        if outcome.cancelled:
            lines.append(f"Cancelled: {outcome.batch_id} ({outcome.provider})")
        else:
            lines.append(
                f"[ERROR] Failed to cancel {outcome.batch_id}: {outcome.error}"
            )
    lines.append(f"Batch cancel: {report.cancelled} cancelled, {report.failed} failed.")
    return lines
