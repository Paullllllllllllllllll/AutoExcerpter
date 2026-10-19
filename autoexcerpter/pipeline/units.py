"""One item's document in the run's store, and the unit calls made for it.

A :class:`DocumentHandle` is what an item needs to work its document: the
store, the document's id and name, the run's lease on it and the model and
provider of each role, which the units carry. It plans the expected units as
``pending``, reads them, resets them and writes one unit, a page summary or
the citations in one short transaction that checks the lease.

:func:`unit_call` wraps the provider calls of one unit: it waits for room in
the store's write queue, attributes the calls' attempts to the unit, records
them through :class:`~autoexcerpter.output.attempts.AttemptRows` and collects
their usage. :meth:`UnitCall.result` then builds the unit's row, stamped with
the attempt its content came from.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from autoexcerpter.common.capture import use_recorder
from autoexcerpter.common.store import (
    Lease,
    PlannedUnit,
    Store,
    UnitKey,
    UnitResult,
    UnitRow,
    unit_context,
)
from autoexcerpter.common.usage import UsageTotals
from autoexcerpter.llm import collect_usage
from autoexcerpter.output.attempts import AttemptRows
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    CitationRow,
    SummaryRow,
    plan_pages,
    write_citations,
    write_summary,
)

__all__ = ["DocumentHandle", "Stamp", "UnitCall", "unit_call"]


@dataclass(frozen=True)
class Stamp:
    """The model and provider a role's units carry."""

    model: str
    provider: str


@dataclass(frozen=True)
class DocumentHandle:
    """One item's document of the run's store, worked under the run's lease.

    *summary* None plans no summary units. The run core builds the handle
    after acquiring the lease; the lease keeper renews it meanwhile.
    """

    store: Store
    doc_id: int
    name: str
    lease: Lease
    transcription: Stamp
    summary: Stamp | None = None

    def key(self, kind: str, idx: int) -> UnitKey:
        """Return the key of unit *idx* of *kind*."""
        return UnitKey(self.doc_id, kind, idx)

    def stamp(self, kind: str) -> Stamp:
        """Return the stamp of *kind*'s units."""
        if kind == KIND_SUMMARY and self.summary is not None:
            return self.summary
        return self.transcription

    async def plan(self, pages: Sequence[PlannedUnit]) -> int:
        """Create the expected units as ``pending``; return the number created."""
        summary = self.summary
        return await plan_pages(
            self.store,
            self.lease,
            pages,
            transcription=(self.transcription.model, self.transcription.provider),
            summary=None if summary is None else (summary.model, summary.provider),
        )

    def units(self, kind: str | None = None) -> list[UnitRow]:
        """Return the document's units, of *kind* when given, in index order."""
        return self.store.units(self.doc_id, kind=kind)

    async def reset(
        self, kind: str | None = None, idxs: Sequence[int] | None = None
    ) -> int:
        """Return units to ``pending``, keeping their responses; return the count."""
        return await self.store.reset_units(self.lease, kind=kind, idxs=idxs)

    async def set_image_settings(self, record: Mapping[str, Any] | None) -> None:
        """Record the image settings the document's pages are transcribed with."""
        await self.store.set_image_settings(self.lease, record)

    async def write_transcription(self, result: UnitResult) -> None:
        """Commit one transcription unit.

        Raises:
            LeaseLostError: The lease expired or passed to another holder.
            sqlite3.Error: The write failed (a busy database only delays it).
        """
        await self.store.write_unit(self.lease, result)

    async def write_summary(self, result: UnitResult, row: SummaryRow | None) -> None:
        """Commit a page's ``summaries`` row, then its summary unit."""
        await write_summary(self.store, self.lease, result, row)

    async def write_citations(self, rows: Sequence[CitationRow]) -> None:
        """Replace the document's citations."""
        await write_citations(self.store, self.lease, rows)


@dataclass
class UnitCall:
    """The attempts and usage of one unit's provider calls."""

    handle: DocumentHandle
    kind: str
    idx: int
    recorder: AttemptRows
    usage: UsageTotals

    def result(  # noqa: PLR0913 - one keyword per unit field
        self,
        *,
        status: str,
        content: str | None,
        outcome: str,
        label: str | None,
        source_index: int | str | None,
        error: str | None = None,
        error_type: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> UnitResult:
        """Return the unit's row, stamped with its selected attempt.

        The selected attempt is the last one without an error, else the last
        one; without attempts the handle's stamp applies.

        Raises:
            Exception: The first attempt that could not be stored.
        """
        recorder = self.recorder
        if recorder.failures:
            raise recorder.failures[0]
        stamp = self.handle.stamp(self.kind)
        selected = recorder.selected()
        usage = self.usage.get(self.kind)
        return UnitResult(
            kind=self.kind,
            idx=self.idx,
            status=status,
            model=selected.record.model if selected else stamp.model,
            provider=selected.record.provider if selected else stamp.provider,
            content=content,
            outcome=outcome,
            label=label,
            source_index=source_index,
            usage=None if usage.is_empty else usage.as_dict(),
            error=error,
            error_type=error_type,
            provenance=provenance,
            response_id=selected.response_id if selected else None,
            attempts=len(recorder.stored) or None,
        )


@contextlib.asynccontextmanager
async def unit_call(
    handle: DocumentHandle, kind: str, idx: int
) -> AsyncIterator[UnitCall]:
    """Run the block's provider calls as the calls of unit *idx* of *kind*.

    Waits for room in the store's write queue first, so provider calls
    pause while writes wait on a busy database.
    """
    await handle.store.wait_capacity()
    stamp = handle.stamp(kind)
    recorder = AttemptRows(handle.store, provider=stamp.provider, model=stamp.model)
    with (
        unit_context(handle.key(kind, idx)),
        use_recorder(recorder),
        collect_usage() as usage,
    ):
        yield UnitCall(handle, kind, idx, recorder, usage)
