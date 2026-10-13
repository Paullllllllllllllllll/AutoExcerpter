"""One item of ``run --batch``: plan its units and submit them to batch APIs.

:class:`BatchItem` works one leased document. It plans the page units as a
synchronous run does, then submits the work the store still lacks: the
pages to transcribe as image requests to the transcription model's batch API,
and the summaries of pages already transcribed as text requests to the
summary model's. Pages whose transcription it submits wait for ``batch
check``, which finalizes them and submits their summaries (see
:mod:`autoexcerpter.batch.check`). A page that fails to render is stored
failed at once, and the summary of a failed or blank page is a placeholder
written without a call. The document's batch job, naming this run, is
committed before the first upload, so a check can always continue the
work. When nothing is left to submit and no unit is pending, the citations
are stored and the files rendered, as at the end of a synchronous item.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from autoexcerpter.batch.bodies import (
    ROLE_SUMMARY,
    ROLE_TRANSCRIPTION,
    BodyPlan,
    build_plan,
)
from autoexcerpter.batch.clients import BatchClients
from autoexcerpter.batch.submit import (
    BatchSubmissionError,
    SubmitOutcome,
    UnitRequest,
    submit_units,
)
from autoexcerpter.batch.work import summary_work, write_placeholders
from autoexcerpter.common.batch import BatchRequest
from autoexcerpter.common.images import PagePayload
from autoexcerpter.common.store import UnitResult
from autoexcerpter.events import NullEvents, RunEvents
from autoexcerpter.llm.options import response_format_for
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    delete_batch_job,
    image_settings_record,
    store_artifacts,
    transcription_outcome,
    write_batch_job,
)
from autoexcerpter.pipeline.item import (
    ItemFiles,
    PayloadSource,
    open_source,
    plan_document,
)
from autoexcerpter.pipeline.job import ItemJob
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.pipeline.units import DocumentHandle

__all__ = ["LABEL", "BatchItem", "BatchItemOutcome"]

logger = logging.getLogger(__name__)

LABEL = "autoexcerpter"
"""The name the batches of a run carry at the provider."""

Status = Literal["submitted", "complete", "incomplete", "failed"]


@dataclass
class BatchItemOutcome:
    """How a batch item ended.

    ``submitted``: its submissions were accepted and wait for ``batch
    check``; ``complete``: nothing was left to submit and its files were
    rendered; ``incomplete``: nothing was submitted, but units stay pending;
    ``failed``: a submission was refused or is unresolved, a page failed, or
    the item could not be read. *outputs* are the files rendered,
    *submission_ids* the submissions accepted and *detail* what went wrong.
    The caller has checked under the lease that no submission holds the
    document's units.
    """

    status: Status
    submission_ids: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    detail: str | None = None

    @property
    def success(self) -> bool:
        """Whether the item counts as done for the exit code."""
        return self.status in ("submitted", "complete")


def _request(payload: PagePayload) -> BatchRequest:
    return BatchRequest(
        custom_id=f"page-{payload.index}",
        image_base64=payload.base64,
        mime_type=payload.mime_type,
        order_index=payload.index,
    )


def _provenance(payload: PagePayload) -> dict[str, Any]:
    return {
        "source_file": payload.source_file,
        "image_provenance": payload.provenance,
    }


class BatchItem:
    """Submit the units one leased document still lacks to batch APIs.

    *context* is the item's summary context as it enters the prompt;
    *clients* the run's batch clients.
    """

    def __init__(  # noqa: PLR0913 - the item, its document and the run's objects
        self,
        item: ItemSpec,
        output_dir: Path,
        job: ItemJob,
        document: DocumentHandle,
        *,
        context: str | None,
        clients: BatchClients,
        events: RunEvents | None = None,
        force: bool = False,
    ) -> None:
        self.item = item
        self.output_dir = output_dir
        self.job = job
        self.document = document
        self.context = context
        self.clients = clients
        self.events: RunEvents = events if events is not None else NullEvents()
        self.files = ItemFiles(
            item, output_dir, job, document, force=force, events=self.events
        )
        tx = job.transcription
        self.transcription_format = response_format_for(
            tx.response_format, tx.provider, tx.name
        )
        self._job_written = False

    def _plan(self, role: str) -> BodyPlan:
        caching = self.clients.settings.prompt_caching
        if role == ROLE_TRANSCRIPTION:
            return build_plan(self.job.transcription, role, caching=caching)
        return build_plan(
            self.job.summary,
            role,
            context=self.context,
            transcription_format=self.transcription_format,
            caching=caching,
        )

    async def _store_artifacts(self, plans: Sequence[BodyPlan]) -> None:
        texts: dict[str, str | None] = {"context": self.context}
        for plan in plans:
            texts[f"{plan.role}_prompt"] = plan.system_prompt
            if plan.schema is not None:
                texts[f"{plan.role}_schema"] = json.dumps(
                    plan.schema, ensure_ascii=False, indent=2
                )
        await store_artifacts(self.document.store, texts)

    async def process(self) -> BatchItemOutcome:
        """Plan the item's units and submit the missing ones."""
        try:
            source = open_source(self.item, self.job)
        except Exception as exc:  # noqa: BLE001 - the item fails, the run goes on
            logger.exception("Failed to open input for %s: %s", self.item.path, exc)
            return BatchItemOutcome("failed", detail=f"cannot open the input: {exc}")
        try:
            if len(source) == 0:
                return BatchItemOutcome("failed", detail="no page found in the input")
            return await self._run(source)
        finally:
            source.close()

    async def _run(self, source: PayloadSource) -> BatchItemOutcome:
        document = self.document
        plans = [self._plan(ROLE_TRANSCRIPTION)]
        if self.job.summarize:
            plans.append(self._plan(ROLE_SUMMARY))
        await self._store_artifacts(plans)
        await plan_document(document, source, pdf=self.item.kind == "pdf")
        pending = [
            unit.idx
            for unit in document.units(KIND_TRANSCRIPTION)
            if unit.status != "done"
        ]
        if pending:
            await document.set_image_settings(image_settings_record(source.img_cfg))
        transcriptions = await self._render(source, pending)
        outcome = BatchItemOutcome("incomplete")
        if transcriptions:
            await self._submit(transcriptions, plans[0], outcome)
        if self.job.summarize and outcome.status != "failed":
            waiting = {request.unit.idx for request in transcriptions}
            work = summary_work(document.store, document.doc_id, waiting=waiting)
            summary = self.job.summary
            await write_placeholders(
                document.store,
                document.lease,
                work.placeholders,
                model=summary.name,
                provider=summary.provider,
            )
            if work.requests:
                await self._submit(work.requests, plans[1], outcome)
        if outcome.submission_ids:
            if outcome.status == "incomplete":
                outcome.status = "submitted"
            return outcome
        if self._job_written:
            await delete_batch_job(document.store, document.lease)
        if outcome.status == "failed":
            return outcome
        return await self._finish(outcome)

    async def _render(
        self, source: PayloadSource, pending: Sequence[int]
    ) -> list[UnitRequest]:
        """Render the pages to transcribe; store the ones that fail at once."""
        document = self.document
        stamp = document.transcription
        requests: list[UnitRequest] = []
        for idx in pending:
            label = source.image_name(idx)
            source_index: int | str = idx if self.item.kind == "pdf" else label
            try:
                payload = await asyncio.to_thread(source.build_payload, idx)
            except Exception as exc:  # noqa: BLE001 - a render failure fails one page
                logger.error("Error preparing page %s: %s", label, exc)
                text = f"[preprocessing error: {exc}]"
                await document.write_transcription(
                    UnitResult(
                        kind=KIND_TRANSCRIPTION,
                        idx=idx,
                        status="failed",
                        model=stamp.model,
                        provider=stamp.provider,
                        content=text,
                        outcome=transcription_outcome(text, str(exc)),
                        label=label,
                        source_index=source_index,
                        error=str(exc),
                        error_type="preprocessing_failure",
                    )
                )
                continue
            await document.write_transcription(
                UnitResult(
                    kind=KIND_TRANSCRIPTION,
                    idx=idx,
                    status="pending",
                    model=stamp.model,
                    provider=stamp.provider,
                    label=label,
                    source_index=source_index,
                    provenance=_provenance(payload),
                    attempts=0,
                )
            )
            requests.append(
                UnitRequest(document.key(KIND_TRANSCRIPTION, idx), _request(payload))
            )
        return requests

    async def _submit(
        self,
        requests: Sequence[UnitRequest],
        plan: BodyPlan,
        outcome: BatchItemOutcome,
    ) -> None:
        """Submit *requests*; record the batch job before the first upload."""
        document = self.document
        if not self._job_written:
            await write_batch_job(document.store, document.lease, self.context)
            self._job_written = True
        try:
            submitted: SubmitOutcome = await submit_units(
                document.store,
                requests,
                plan=plan,
                clients=self.clients,
                label=LABEL,
            )
        except BatchSubmissionError as exc:
            outcome.status = "failed"
            outcome.detail = str(exc)
            return
        accepted = (
            submitted.submission_ids
            if submitted.status != "refused"
            else submitted.submission_ids[:-1]
        )
        outcome.submission_ids.extend(accepted)
        if submitted.status == "submitted":
            return
        outcome.status = "failed"
        verb = "was refused" if submitted.status == "refused" else "is unresolved"
        outcome.detail = (
            f"the {plan.role} batch {verb}: {submitted.detail or 'no detail'}"
        )

    async def _finish(self, outcome: BatchItemOutcome) -> BatchItemOutcome:
        """Store the citations and render the files when no unit is pending."""
        units = self.document.units()
        relevant = [
            unit
            for unit in units
            if unit.kind == KIND_TRANSCRIPTION
            or (self.job.summarize and unit.kind == KIND_SUMMARY)
        ]
        if any(unit.status == "pending" for unit in relevant):
            return outcome
        if not await self.files.store_citations():
            outcome.status = "failed"
            outcome.detail = "the citations were not stored"
            return outcome
        status = await self.files.write()
        outcome.outputs = list(self.files.written)
        failed = any(unit.status == "failed" for unit in relevant)
        written = status.transcription_ok and status.summaries_ok
        outcome.status = "complete" if written and not failed else "failed"
        if failed:
            outcome.detail = "pages failed; a later run retries them"
        return outcome
