"""The summary work of a document in batch mode, and its placeholders.

A page's summary is made from its stored transcription. A page whose
transcription failed gets the placeholder of a failed page and a blank page
the blank placeholder, both without a call, as in a synchronous run
(:class:`~autoexcerpter.pipeline.pages.PageRunner`); a page with an empty
transcription counts as blank, since a request needs text. Every other page
becomes a text request of a summary submission. :func:`summary_work` collects
both for the pages a caller selects, leaving out pages an open or
unresolved submission holds and pages whose transcription is still to come.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from autoexcerpter.batch.submit import UnitRequest
from autoexcerpter.common.batch import BatchRequest
from autoexcerpter.common.store import Lease, Store, UnitResult, UnitRow
from autoexcerpter.constants import is_blank_transcription
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    write_summary,
)
from autoexcerpter.pipeline.pages import (
    build_summary_result,
    placeholder_summary,
    summary_row,
)

__all__ = [
    "Placeholder",
    "SummaryWork",
    "needs_summary",
    "summary_work",
    "write_placeholders",
]


@dataclass(frozen=True)
class Placeholder:
    """The summary written without a call for a failed or a blank page."""

    unit: UnitRow
    entry: dict[str, Any]
    outcome: str


@dataclass
class SummaryWork:
    """The placeholders to write and the requests to submit for a document."""

    placeholders: list[Placeholder] = field(default_factory=list)
    requests: list[UnitRequest] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.placeholders or self.requests)


def needs_summary(unit: UnitRow) -> bool:
    """Whether a summary unit is to be made: missing, failed or stale."""
    return (unit.status, unit.stale) != ("done", 0)


def _placeholder(summary: UnitRow, transcription: UnitRow) -> Placeholder | None:
    idx = summary.idx
    label = summary.label or transcription.label or f"page index {idx}"
    text = transcription.content or ""
    if transcription.status == "failed":
        error = transcription.error or "failed"
        payload = placeholder_summary(
            [f"[Transcription failed: {error}]"], ["other"], error_message=error
        )
        outcome = "placeholder"
    elif is_blank_transcription(text) or not text.strip():
        payload = placeholder_summary(None, ["blank"])
        outcome = "blank"
    else:
        return None
    entry = build_summary_result(idx, label, payload, idx + 1, None)
    return Placeholder(summary, entry, outcome)


def summary_work(
    store: Store,
    doc_id: int,
    *,
    selected: Callable[[UnitRow], bool] = needs_summary,
    waiting: Collection[int] = (),
) -> SummaryWork:
    """Return the summary work of document *doc_id*.

    *selected* chooses the summary units to make; *waiting* holds the pages
    whose transcription this run submits, which wait for it. Pages held by
    an open or unresolved submission and pages without a stored
    transcription are left out.
    """
    units = store.units(doc_id)
    blocked = set(store.blocked_units(doc_id))
    transcriptions = {u.idx: u for u in units if u.kind == KIND_TRANSCRIPTION}
    work = SummaryWork()
    for summary in (u for u in units if u.kind == KIND_SUMMARY):
        idx = summary.idx
        transcription = transcriptions.get(idx)
        if (
            idx in waiting
            or summary.key in blocked
            or transcription is None
            or transcription.key in blocked
            or transcription.status == "pending"
            or not selected(summary)
        ):
            continue
        placeholder = _placeholder(summary, transcription)
        if placeholder is not None:
            work.placeholders.append(placeholder)
            continue
        work.requests.append(
            UnitRequest(
                summary.key,
                BatchRequest(
                    custom_id=f"summary-{idx}",
                    text=transcription.content or "",
                    order_index=idx,
                ),
            )
        )
    return work


async def write_placeholders(
    store: Store,
    lease: Lease,
    placeholders: Collection[Placeholder],
    *,
    model: str,
    provider: str,
) -> None:
    """Write the placeholder summaries with the summary model's stamp."""
    for placeholder in placeholders:
        unit = placeholder.unit
        result = UnitResult(
            kind=KIND_SUMMARY,
            idx=unit.idx,
            status="done",
            model=model,
            provider=provider,
            content=None,
            outcome=placeholder.outcome,
            label=unit.label,
            source_index=unit.source_index,
        )
        await write_summary(
            store, lease, result, summary_row(unit.idx, placeholder.entry)
        )
