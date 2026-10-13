"""Reconcile, adopt and abandon unresolved batch submissions.

A submission is unresolved when its batch create got no confirmed response,
or when it was left ``submitting`` without a provider batch id (the
submitting process ended between the create and its record). The provider
may hold a batch for it, so its units are never resubmitted automatically.

:func:`reconcile` looks for that batch. OpenAI batches carry the submission
id in their metadata and Google batches in their display name: a single
batch that names the submission is adopted, several are reported. Anthropic
batches carry neither, so the batches created in the submission's time
window are listed as candidates, with their request counts, for the user to
adopt one (:func:`adopt`, ``batch check --adopt BATCH_ID``) or to abandon
the submission (:func:`abandon`, ``batch check --abandon SUBMISSION_ID``),
which returns its units to pending and deletes its upload.

The store must hold the lease of every document a submission covers before
any of these functions changes it (:func:`lease_submission`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from autoexcerpter.batch.clients import BatchClients, NotListed, status_of
from autoexcerpter.common.batch import (
    BatchBackend,
    BatchHandle,
    BatchListing,
    PartUpload,
    as_datetime,
)
from autoexcerpter.common.store import Store, Submission

__all__ = [
    "LISTED_BY_ID",
    "WINDOW_AFTER",
    "WINDOW_BEFORE",
    "Reconciliation",
    "abandon",
    "adopt",
    "count_matches",
    "lease_submission",
    "list_candidates",
    "provider_of",
    "reconcile",
    "window",
]

logger = logging.getLogger(__name__)

LISTED_BY_ID = frozenset({"openai", "google"})
"""Providers whose batch listing names the submission id."""

WINDOW_BEFORE = timedelta(minutes=5)
"""How long before a submission's intent time a batch may be created."""

WINDOW_AFTER = timedelta(hours=1)
"""How long after a submission's intent time a batch may be created."""


def provider_of(batch_id: str) -> str:
    """Return the provider of a batch id.

    Anthropic ids start with ``msgbatch_`` and Google ids with ``batches/``;
    every other id is an OpenAI id.
    """
    if batch_id.startswith("msgbatch_"):
        return "anthropic"
    if batch_id.startswith("batches/"):
        return "google"
    return "openai"


@dataclass(frozen=True)
class Reconciliation:
    """What a reconciliation found for one unresolved submission.

    *submission* is the submission after it; *adopted* the batch recorded
    for it, if any. *matches* are the batches that name it (OpenAI,
    Google); *candidates* the batches of its time window (Anthropic), those
    with its request count first.
    """

    submission: Submission
    adopted: str | None = None
    matches: tuple[BatchListing, ...] = ()
    candidates: tuple[BatchListing, ...] = ()
    detail: str = ""


def window(submission: Submission) -> tuple[datetime, datetime]:
    """Return the creation-time window of *submission*'s provider batch."""
    intent = as_datetime(submission.intent_at)
    if intent is None:
        raise ValueError(f"Submission {submission.submission_id} has no intent time")
    return intent - WINDOW_BEFORE, intent + WINDOW_AFTER


def count_matches(listing: BatchListing, submission: Submission) -> bool:
    """Whether *listing* reports the request count of *submission*."""
    return listing.request_count == submission.request_count


def _ordered(
    listings: Sequence[BatchListing], submission: Submission
) -> tuple[BatchListing, ...]:
    return tuple(
        sorted(listings, key=lambda listing: not count_matches(listing, submission))
    )


async def lease_submission(store: Store, submission_id: str) -> bool:
    """Lease every document of *submission_id*; False when another run holds one.

    Leases this store already holds are kept as they are.
    """
    held = store.held_leases()
    doc_ids = sorted(
        {unit.doc_id for unit in store.submission_requests(submission_id).values()}
    )
    for doc_id in doc_ids:
        if doc_id not in held and await store.acquire(doc_id) is None:
            return False
    return True


async def list_candidates(
    submission: Submission, clients: BatchClients
) -> tuple[BatchListing, ...]:
    """List the batches created in *submission*'s time window.

    Those with the submission's request count come first.
    """
    since, until = window(submission)
    listings = await clients.call(
        submission.provider,
        lambda backend: backend.list_batches(since=since, until=until),
        model=submission.model,
        label=f"listing of {submission.provider} batches",
    )
    return _ordered(listings, submission)


async def reconcile(
    store: Store, submission: Submission, clients: BatchClients
) -> Reconciliation:
    """Look for the provider batch of an unresolved *submission*.

    A single batch that names the submission is adopted; anything else
    leaves the submission unresolved with a detail for the user. A failed
    listing is reported the same way.
    """
    submission_id = submission.submission_id
    provider = submission.provider
    abandon_hint = (
        f"abandon it with 'autoexcerpter batch check --abandon {submission_id}' "
        "to submit its pages again"
    )
    try:
        if provider not in LISTED_BY_ID:
            candidates = await list_candidates(submission, clients)
            return Reconciliation(
                submission,
                candidates=candidates,
                detail=(
                    f"{provider} batches carry no submission id; "
                    f"{len(candidates)} batch(es) were created around submission "
                    f"{submission_id}. Adopt the one that holds its "
                    f"{submission.request_count} request(s) with "
                    "'autoexcerpter batch check --adopt BATCH_ID', or "
                    + abandon_hint
                    + "."
                ),
            )
        since, _until = window(submission)
        try:
            matches = tuple(
                await clients.call(
                    provider,
                    lambda backend: _listed(backend, submission_id, since),
                    model=submission.model,
                    label=f"listing of {provider} batches",
                )
            )
        except NotListed:
            matches = ()
    except Exception as exc:  # noqa: BLE001 - the submission stays unresolved
        logger.warning("Reconciliation of %s failed: %s", submission_id, exc)
        return Reconciliation(
            submission, detail=f"the provider's batch list failed: {exc}"
        )
    if len(matches) == 1:
        (match,) = matches
        adopted = await store.record_batch(
            submission_id, match.batch_id, provider_status=match.provider_status
        )
        logger.info("Adopted batch %s for submission %s", match.batch_id, submission_id)
        return Reconciliation(
            adopted,
            adopted=match.batch_id,
            matches=matches,
            detail=f"adopted batch {match.batch_id}, which names the submission",
        )
    if matches:
        ids = ", ".join(match.batch_id for match in matches)
        return Reconciliation(
            submission,
            matches=matches,
            detail=(
                f"{len(matches)} batches name submission {submission_id} ({ids}); "
                "adopt one with 'autoexcerpter batch check --adopt BATCH_ID' "
                "and cancel the others"
            ),
        )
    return Reconciliation(
        submission,
        detail=(
            f"no {provider} batch names submission {submission_id}, so its create "
            "most likely never reached the provider; " + abandon_hint
        ),
    )


def _listed(
    backend: BatchBackend, submission_id: str, since: datetime
) -> list[BatchListing]:
    """Return the batches that name *submission_id*.

    Raises:
        NotListed: None does.
    """
    listings = backend.list_batches(submission_id=submission_id, since=since)
    if not listings:
        raise NotListed(f"submission {submission_id}")
    return listings


async def adopt(
    store: Store, submission: Submission, batch_id: str, clients: BatchClients
) -> Submission:
    """Record *batch_id* as the provider batch of the unresolved *submission*.

    The batch is looked up first, so an id the provider does not know is
    never recorded.

    Raises:
        Exception: The provider's lookup error, or the store's error when
            the submission is not unresolved or another submission holds
            the batch.
    """
    info = await clients.call(
        submission.provider,
        status_of(BatchHandle(submission.provider, batch_id)),
        model=submission.model,
        label=f"status of batch {batch_id}",
    )
    adopted = await store.record_batch(
        submission.submission_id,
        batch_id,
        provider_status=info.provider_status or info.status.value,
    )
    logger.info("Adopted batch %s for submission %s", batch_id, adopted.submission_id)
    return adopted


async def abandon(
    store: Store, submission: Submission, clients: BatchClients
) -> Submission:
    """Abandon the unresolved *submission*: units pending, upload deleted.

    Raises:
        SubmissionStateError: The submission is not unresolved.
    """
    abandoned = await store.abandon_submission(submission.submission_id)
    if submission.upload_ref:
        await clients.delete_upload(
            PartUpload(submission.provider, submission.upload_ref),
            model=submission.model,
        )
    logger.info("Abandoned submission %s", submission.submission_id)
    return abandoned
