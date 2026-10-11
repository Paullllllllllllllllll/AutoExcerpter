"""Builders of item-processing objects and stored item state for the tests."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.common.store import PlannedUnit, UnitResult
from autoexcerpter.output.store import (
    DATABASE_NAME,
    KIND_TRANSCRIPTION,
    open_store,
    transcription_outcome,
)
from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.job import ItemJob, PhaseModel
from autoexcerpter.pipeline.managers import ItemManagers
from autoexcerpter.pipeline.pages import PageRunner, Summarizer, Transcriber
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.pipeline.units import DocumentHandle, Stamp

MODEL = "gpt-5-mini"
STAMP = Stamp(MODEL, "openai")


def transcriber(side_effect: Any = None, return_value: Any = None) -> MagicMock:
    """Return a transcription-manager mock whose ``transcribe_payload`` is async.

    *side_effect* may be a plain function; its result is the awaited value.
    """
    manager = MagicMock(spec=Transcriber)
    manager.transcribe_payload.side_effect = side_effect
    if return_value is not None:
        manager.transcribe_payload.return_value = return_value
    return manager


def summarizer(side_effect: Any = None, return_value: Any = None) -> MagicMock:
    """Return a summary-manager mock whose ``generate_summary`` is async."""
    manager = MagicMock(spec=Summarizer)
    manager.generate_summary.side_effect = side_effect
    if return_value is not None:
        manager.generate_summary.return_value = return_value
    return manager


def make_job(**overrides: Any) -> ItemJob:
    """Return an item job for the OpenAI test model; summaries off by default."""
    values: dict[str, Any] = {
        "transcription": PhaseModel(MODEL, "openai"),
        "summary": PhaseModel(MODEL, "openai"),
        "summarize": False,
        "output_docx": True,
        "output_markdown": True,
        "concurrency": 2,
        "timeout": 60,
    }
    values.update(overrides)
    return ItemJob(**values)


def document_stub(name: str = "doc") -> MagicMock:
    """Return a document handle for page runners that store nothing."""
    document = MagicMock(spec=DocumentHandle)
    document.name = name
    return document


@contextlib.asynccontextmanager
async def stored_document(
    output_dir: Path, name: str = "doc", *, summarize: bool = False, kind: str = "pdf"
) -> AsyncIterator[DocumentHandle]:
    """Yield document *name* of *output_dir*'s store, leased by a test run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    async with open_store(output_dir) as store:
        await store.begin_run("test")
        doc_id = await store.ensure_document(name, kind=kind)
        lease = await store.acquire(doc_id)
        assert lease is not None
        try:
            yield DocumentHandle(
                store, doc_id, name, lease, STAMP, STAMP if summarize else None
            )
        finally:
            await store.end_run(0)


async def store_pages(
    document: DocumentHandle, pages: int, texts: dict[int, str]
) -> None:
    """Plan *pages* pages of *document* and store the transcriptions in *texts*."""
    await document.plan(
        [
            PlannedUnit(idx, label=f"page_{idx + 1:04d}.jpg", source_index=idx)
            for idx in range(pages)
        ]
    )
    for idx, text in texts.items():
        await document.write_transcription(
            UnitResult(
                kind=KIND_TRANSCRIPTION,
                idx=idx,
                status="done",
                model=MODEL,
                provider="openai",
                content=text,
                outcome=transcription_outcome(text, None),
            )
        )


def make_runner(
    tmp_path: Path,
    *,
    transcribe_manager: Any = None,
    summary_manager: Any = None,
    document: Any = None,
) -> PageRunner:
    """Return a page runner; without *document* it stores nothing."""
    return PageRunner(
        item_name="doc",
        document=document if document is not None else document_stub(),
        transcribe_manager=transcribe_manager or transcriber(),
        summary_manager=summary_manager,
        transcription_provider="openai",
    )


def clean_page_runner(document: Any, summary_manager: Any = None) -> PageRunner:
    """Return a page runner whose transcription call returns one clean page."""
    transcribe_manager = transcriber(
        return_value={
            "image": "page_0001.jpg",
            "transcription": "Hello world",
            "processing_time": 0.01,
            "provider": "openai",
        }
    )
    return make_runner(
        Path(),
        transcribe_manager=transcribe_manager,
        summary_manager=summary_manager,
        document=document,
    )


def content_summary() -> dict[str, Any]:
    """Return an error-free summary answer for content page 1."""
    return {
        "page": 1,
        "page_information": {
            "page_number_integer": 1,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": ["a point"],
        "references": [],
    }


def make_progress(total: int = 1, **kwargs: Any) -> PageProgress:
    """Return a page counter for *total* pending pages."""
    return PageProgress(total, **kwargs)


def make_processor(
    path: Path,
    output_dir: Path,
    document: Any = None,
    *,
    kind: str = "pdf",
    transcribe_manager: Any = None,
    summary_manager: Any = None,
    **job_overrides: Any,
) -> ItemProcessor:
    """Return an item processor with injected (mock) managers.

    Without *document* the processor stores nothing.
    """
    return ItemProcessor(
        ItemSpec(kind=kind, path=path),
        output_dir,
        make_job(**job_overrides),
        document if document is not None else document_stub(path.stem),
        managers=ItemManagers(transcribe_manager or transcriber(), summary_manager),
    )


def process(
    path: Path,
    output_dir: Path,
    *,
    before: Callable[[DocumentHandle], Awaitable[None]] | None = None,
    prepare: Callable[[ItemProcessor], None] | None = None,
    kind: str = "pdf",
    **options: Any,
) -> tuple[ItemProcessor, bool]:
    """Process one item into *output_dir*'s store; return it and its verdict.

    *before* seeds the document first; *prepare* may patch the processor
    before it runs. *options* go to :func:`make_processor`.
    """
    name = ItemSpec(kind=kind, path=path).output_stem
    summarize = bool(options.get("summarize", False))

    async def main() -> tuple[ItemProcessor, bool]:
        async with stored_document(
            output_dir, name, summarize=summarize, kind=kind
        ) as document:
            if before is not None:
                await before(document)
            processor = make_processor(path, output_dir, document, kind=kind, **options)
            if prepare is not None:
                prepare(processor)
            return processor, await processor.process()

    return asyncio.run(main())


def stored_units(
    output_dir: Path, kind: str = KIND_TRANSCRIPTION
) -> list[dict[str, Any]]:
    """Return the stored units of *kind* in page order."""
    rows = read_rows(output_dir / DATABASE_NAME, "units", order_by=["doc_id", "idx"])
    return [row for row in rows if row["kind"] == kind]


def statuses(output_dir: Path, kind: str = KIND_TRANSCRIPTION) -> list[str]:
    """Return the status of every stored unit of *kind* in page order."""
    return [str(row["status"]) for row in stored_units(output_dir, kind)]


__all__ = [
    "MODEL",
    "STAMP",
    "clean_page_runner",
    "content_summary",
    "document_stub",
    "make_job",
    "make_processor",
    "make_progress",
    "make_runner",
    "process",
    "statuses",
    "store_pages",
    "stored_document",
    "stored_units",
    "summarizer",
    "transcriber",
]
