"""Build the LLM managers of one item from its job."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from autoexcerpter.llm import CallEnv, SummaryManager, TranscriptionManager
from autoexcerpter.llm.client import resolve_provider
from autoexcerpter.llm.options import response_format_for
from autoexcerpter.pipeline.job import ItemJob
from autoexcerpter.pipeline.pages import Summarizer, Transcriber

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemManagers:
    """The LLM managers of one item; *summary* is None without summaries."""

    transcription: Transcriber
    summary: Summarizer | None


def build_managers(
    job: ItemJob, context: str | None = None, env: CallEnv | None = None
) -> ItemManagers:
    """Build the transcription and (if summarizing) summary managers.

    *context* is the item's summary context as it enters the prompt (see
    :func:`autoexcerpter.pipeline.context.summary_context`). *env* holds the
    run's rate limiters, usage totals and events; None gives the item an
    environment of its own with the default settings.
    """
    env = env or CallEnv.create()
    tx = job.transcription
    tx_format = response_format_for(tx.response_format, resolve_provider(tx), tx.name)
    logger.info("Transcription model %s, response format %s", tx.name, tx_format)
    transcription = TranscriptionManager(phase=tx, env=env)
    if not job.summarize:
        return ItemManagers(transcription, None)
    summary = SummaryManager(
        phase=job.summary,
        env=env,
        context=context,
        transcription_format=tx_format,
    )
    return ItemManagers(transcription, summary)


__all__ = ["ItemManagers", "build_managers"]
