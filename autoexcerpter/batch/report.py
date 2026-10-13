"""The report lines of ``batch check`` and ``batch cancel``.

The CLI writes them to stderr and the guided UI to its console, so both
front ends report a check or a cancel in the same words. ``[ERROR]`` and
``[WARNING]`` prefixes mark the lines that report a failure or something
the user has to act on.
"""

from __future__ import annotations

from autoexcerpter.batch.cancel import CancelReport
from autoexcerpter.batch.check import (
    ABANDONED,
    UNRESOLVED,
    CheckResult,
    DocumentAdvance,
    SubmissionCheck,
)
from autoexcerpter.batch.finalize import FAILED, FINALIZED_OUTCOME, PENDING, SKIPPED

__all__ = [
    "cancel_lines",
    "check_lines",
    "complete_documents",
    "submitted_batches",
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
    return sum(1 for advance in result.documents if advance.outcome == "complete")


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
    elif advance.outcome == "complete":
        outputs = ", ".join(path.name for path in advance.outputs) or "no file"
        lines.append(f"{label}: complete -> {outputs}")
    elif advance.outcome == FAILED:
        lines.append(f"[ERROR] {label}: {advance.detail}")
    else:
        lines.append(f"{label}: {advance.outcome}: {advance.detail}")
    return lines


def check_lines(result: CheckResult) -> list[str]:
    """Return one line per refused database, submission and document, then
    the totals."""
    lines = [f"[ERROR] {path}: {reason}" for path, reason in result.refused.items()]
    for check in result.submissions:
        lines += _submission_lines(check)
    for advance in result.documents:
        lines += _document_lines(advance)
    stats = result.stats
    lines.append(
        f"Batch check: {stats.finalized} finalized, {stats.pending} pending, "
        f"{stats.failed} failed, {stats.unresolved} unresolved; "
        f"{submitted_batches(result)} summary batch(es) submitted, "
        f"{complete_documents(result)} document(s) complete."
    )
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
