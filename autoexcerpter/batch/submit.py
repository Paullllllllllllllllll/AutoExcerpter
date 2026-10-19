"""Batch submission of units on the store.

:func:`submit_units` submits units, each with its request, to the batch API
of one model. The requests are split into parts by the provider's count and
byte limits, and every part becomes one submission of the store: the
submission and its requests, whose custom ids are scoped to the submission
id, are committed before the upload, and each request's body as submitted
(payloads redacted) is recorded next, so finalization can store it as the
request of the unit's response row; the upload reference is recorded before
the create; the provider batch id is recorded after it. Each part is
attempted once, since a retried create could duplicate a batch.

A definite refusal marks the submission failed and deletes its upload; its
units stay pending. A create whose outcome is unknown marks the submission
unresolved and keeps its upload; its units stay blocked until ``batch
check`` adopts the provider's batch or the user abandons the submission.
Provider errors end in a :class:`SubmitOutcome` and never propagate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from autoexcerpter.batch.bodies import BodyPlan
from autoexcerpter.batch.clients import BatchClients
from autoexcerpter.common.batch import (
    BatchBackend,
    BatchCreateError,
    BatchPart,
    BatchRequest,
    PartUpload,
)
from autoexcerpter.common.store import (
    Store,
    Submission,
    SubmissionRejectedError,
    UnitKey,
)

__all__ = [
    "BatchSubmissionError",
    "SubmitOutcome",
    "UnitRequest",
    "submit_units",
]

logger = logging.getLogger(__name__)

SubmitStatus = Literal["submitted", "refused", "unresolved"]

_PROBE_PREFIX = "0" * 8
"""Stands in for a submission id prefix while parts are planned."""


class BatchSubmissionError(RuntimeError):
    """The requests of a submission cannot be built."""


@dataclass(frozen=True)
class SubmitOutcome:
    """How a submission ended.

    ``submitted``: every part was accepted; the units wait for ``batch
    check``. ``refused``: a part was refused; its submission is failed, its
    upload deleted and its units pending, as are those of the parts after
    it. ``unresolved``: a create got no confirmed response; that part's
    units stay blocked. *submission_ids* lists every submission opened, in
    part order; *detail* says what went wrong.
    """

    status: SubmitStatus
    submission_ids: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class UnitRequest:
    """One unit of a submission and the request that produces it."""

    unit: UnitKey
    request: BatchRequest


def _probe_id(position: int) -> str:
    """Return a stand-in custom id as long as the store's scoped ids."""
    return f"{_PROBE_PREFIX}-{position:05d}"


def _plan_parts(
    planner: BatchBackend, units: Sequence[UnitRequest], plan: BodyPlan
) -> list[tuple[BatchPart, list[UnitRequest]]]:
    """Split the units into parts, in order.

    The requests are planned under stand-in ids of the store's id length,
    so renaming a part's requests to its scoped ids keeps the split.
    """
    probes: dict[str, UnitRequest] = {}
    requests: list[BatchRequest] = []
    for unit in units:
        probe = _probe_id(len(requests))
        probes[probe] = unit
        requests.append(replace(unit.request, custom_id=probe))
    return [
        (part, [probes[custom_id] for custom_id in part.custom_ids])
        for part in planner.plan_parts(requests, plan.body)
    ]


def _scoped_part(
    store: Store,
    submission_id: str,
    part: BatchPart,
    members: Sequence[UnitRequest],
    plan: BodyPlan,
    planner: BatchBackend,
) -> BatchPart:
    """Return *part* with its requests renamed to the submission's custom ids."""
    scoped = {
        unit: custom_id
        for custom_id, unit in store.submission_requests(submission_id).items()
    }
    requests = [
        replace(member.request, custom_id=scoped[member.unit]) for member in members
    ]
    replanned = planner.plan_parts(requests, plan.body)
    if len(replanned) != 1:
        raise BatchSubmissionError(
            f"Batch part {part.index} split again under its scoped custom ids"
        )
    return BatchPart(part.index, part.total, requests, replanned[0].entries)


async def _submit_part(
    store: Store,
    part: BatchPart,
    members: Sequence[UnitRequest],
    *,
    plan: BodyPlan,
    planner: BatchBackend,
    clients: BatchClients,
    label: str,
    opened: list[str],
) -> SubmitOutcome | None:
    """Submit one part; return the outcome that ends the submission, if any.

    The store commits the submission before :func:`upload` runs, records
    the upload reference before :func:`create` runs and the batch id after.
    """
    provider, model = plan.provider, plan.model
    stage = "open"
    scoped: list[BatchPart] = []
    uploads: list[PartUpload] = []

    async def upload(submission: Submission) -> str | None:
        nonlocal stage
        submission_id = submission.submission_id
        opened.append(submission_id)
        stage = "upload"
        final = _scoped_part(store, submission_id, part, members, plan, planner)
        scoped.append(final)
        await store.record_requests(submission_id, planner.request_bodies(final))
        uploaded = await clients.call(
            provider,
            lambda backend: backend.upload_part(
                final, submission_id=submission_id, label=label
            ),
            model=model,
            retry=False,
            label=f"upload of batch part {part.index}",
        )
        uploads.append(uploaded)
        stage = "uploaded"
        return uploaded.file_id

    async def create(submission: Submission) -> str:
        nonlocal stage
        stage = "create"
        final, uploaded = scoped[0], uploads[0]
        try:
            handle = await clients.call(
                provider,
                lambda backend: backend.create_part(
                    final, uploaded, submission_id=submission.submission_id, label=label
                ),
                model=model,
                retry=False,
                label=f"create of batch part {part.index}",
            )
        except BatchCreateError as exc:
            if exc.definite:
                raise SubmissionRejectedError(str(exc)) from exc
            raise
        except Exception as exc:
            raise SubmissionRejectedError(str(exc)) from exc
        stage = "created"
        return handle.batch_id

    try:
        submission = await store.submit(
            provider=provider,
            model=model,
            units=[member.unit for member in members],
            upload=upload,
            create=create,
            part=part.index,
        )
    except SubmissionRejectedError as exc:
        for uploaded in uploads:
            await clients.delete_upload(uploaded, model=model)
        return SubmitOutcome("refused", tuple(opened), str(exc))
    except Exception as exc:
        if stage == "upload":
            logger.warning("Upload of batch part %d failed: %s", part.index, exc)
            return SubmitOutcome("refused", tuple(opened), f"upload failed: {exc}")
        if stage != "create":
            raise
        submission_id = opened[-1]
        logger.warning(
            "Batch submission %s has no confirmed create: %s", submission_id, exc
        )
        return SubmitOutcome(
            "unresolved",
            tuple(opened),
            f"the create of batch part {part.index} got no confirmed response "
            f"({exc}); run 'autoexcerpter batch check', which finds the "
            f"batch, or abandon submission {submission_id} with "
            f"'autoexcerpter batch check --abandon {submission_id}'",
        )
    logger.info(
        "Submitted %s batch %s (submission %s, part %d of %d, %d unit(s))",
        provider,
        submission.batch_id,
        submission.submission_id,
        part.index,
        part.total,
        len(members),
    )
    return None


async def submit_units(
    store: Store,
    units: Sequence[UnitRequest],
    *,
    plan: BodyPlan,
    clients: BatchClients,
    label: str,
) -> SubmitOutcome:
    """Submit *units* to the batch API of *plan*'s model, one submission per part.

    The units must exist in the store, which must hold the lease of each
    unit's document. The parts are submitted in order; the first refused or
    unresolved part ends the submission, and the units of the parts after
    it stay pending. *label* names the batches at the provider.

    Raises:
        BatchSubmissionError: No unit is given, or the requests cannot be
            built.
    """
    if not units:
        raise BatchSubmissionError("No unit of the submission has a request to send")
    planner = clients.planner(plan.provider)
    try:
        parts = _plan_parts(planner, units, plan)
    except ValueError as exc:
        raise BatchSubmissionError(f"Cannot build the batch requests: {exc}") from exc
    opened: list[str] = []
    for part, members in parts:
        ended = await _submit_part(
            store,
            part,
            members,
            plan=plan,
            planner=planner,
            clients=clients,
            label=label,
            opened=opened,
        )
        if ended is not None:
            return ended
    return SubmitOutcome("submitted", tuple(opened))
