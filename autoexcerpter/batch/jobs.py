"""The batch jobs of a store: each document's governing run, rebuilt.

A document submitted by ``run --batch`` has a ``batch_jobs`` row naming that
run and the summary context it used (see
:func:`autoexcerpter.output.store.write_batch_job`). :func:`load_jobs`
rebuilds, for each row, the run's :class:`~autoexcerpter.spec.RunSpec` from
the effective spec the store recorded with the run, and the
:class:`~autoexcerpter.pipeline.job.ItemJob` from it with the current
settings. ``batch check`` finalizes and advances the documents with them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from autoexcerpter.common.sqlite import read
from autoexcerpter.common.store import Store
from autoexcerpter.common.structured import ResponseFormat
from autoexcerpter.llm.options import response_format_for
from autoexcerpter.output.store import BatchJob, read_batch_jobs
from autoexcerpter.pipeline.job import ItemJob, job_from_spec
from autoexcerpter.rendering.citations import OpenAlexClient
from autoexcerpter.settings import Settings, SettingsError
from autoexcerpter.spec import RunSpec, SpecError, spec_from_effective

__all__ = ["DocumentJob", "load_jobs"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentJob:
    """One document's batch job with the run spec and item job it follows."""

    row: BatchJob
    spec: RunSpec
    job: ItemJob

    @property
    def doc_id(self) -> int:
        """The document's id."""
        return self.row.doc_id

    @property
    def context(self) -> str | None:
        """The summary context the document's summaries are made with."""
        return self.row.context

    @property
    def transcription_format(self) -> ResponseFormat:
        """The response format of the document's transcriptions."""
        phase = self.job.transcription
        return response_format_for(phase.response_format, phase.provider, phase.name)

    @property
    def summary_format(self) -> ResponseFormat:
        """The response format of the document's summaries."""
        phase = self.job.summary
        return response_format_for(phase.response_format, phase.provider, phase.name)


def _stored_spec(store: Store, run_id: str) -> Mapping[str, Any] | None:
    row = store.run(run_id)
    raw = row.get("spec") if row is not None else None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, Mapping) else None


def load_jobs(store: Store, settings: Settings) -> dict[int, DocumentJob]:
    """Return the batch jobs of *store* by document id.

    A job whose run spec is missing or cannot be rebuilt is left out with a
    warning; its document waits for a ``run``. The jobs share one OpenAlex
    client, as the items of a run do, so its cache is written by one owner.
    """
    rows = read(store.path, read_batch_jobs)
    jobs: dict[int, DocumentJob] = {}
    shared: OpenAlexClient | None = None
    for row in rows:
        stored = _stored_spec(store, row.run_id)
        if stored is None:
            logger.warning(
                "Batch job of document %d names run %s without a stored spec; "
                "it is left for a run",
                row.doc_id,
                row.run_id,
            )
            continue
        try:
            spec = spec_from_effective(stored)
            job = job_from_spec(spec, settings)
        except (SpecError, SettingsError) as exc:
            logger.warning(
                "Batch job of document %d cannot be rebuilt: %s", row.doc_id, exc
            )
            continue
        if job.openalex is not None:
            if shared is None:
                shared = job.openalex
            else:
                job = replace(job, openalex=shared)
        jobs[row.doc_id] = DocumentJob(row, spec, job)
    return jobs
