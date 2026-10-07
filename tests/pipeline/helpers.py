"""Builders of item-processing objects and on-disk item state for the tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from autoexcerpter.constants import LOG_FORMAT_VERSION
from autoexcerpter.pipeline.item import ItemProcessor
from autoexcerpter.pipeline.job import ItemJob, ItemResume, PhaseModel
from autoexcerpter.pipeline.managers import ItemManagers
from autoexcerpter.pipeline.pages import PageRunner, Summarizer, Transcriber
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.types import ItemSpec

MODEL = "gpt-5-mini"
ALL_OUTPUTS = ("transcription", "summary_docx", "summary_md")


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


def make_runner(
    tmp_path: Path,
    *,
    transcribe_manager: Any = None,
    summary_manager: Any = None,
    name: str = "doc",
) -> PageRunner:
    """Return a page runner whose logs live in a working dir under *tmp_path*."""
    paths = ItemPaths.for_item(name, tmp_path)
    paths.working_dir.mkdir(parents=True, exist_ok=True)
    return PageRunner(
        item_name=name,
        paths=paths,
        transcribe_manager=transcribe_manager or transcriber(),
        summary_manager=summary_manager,
        transcription_provider="openai",
    )


def clean_page_runner(tmp_path: Path, summary_manager: Any = None) -> PageRunner:
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
        tmp_path,
        transcribe_manager=transcribe_manager,
        summary_manager=summary_manager,
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
    *,
    kind: str = "pdf",
    transcribe_manager: Any = None,
    summary_manager: Any = None,
    completed: set[int] | None = None,
    resume: ItemResume | None = None,
    **job_overrides: Any,
) -> ItemProcessor:
    """Return an item processor with injected (mock) managers.

    *completed* builds an :class:`ItemResume` that reads the working logs from
    disk; pass *resume* for one carrying parsed log data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and completed is not None:
        resume = ItemResume(completed_page_indices=frozenset(completed))
    return ItemProcessor(
        ItemSpec(kind=kind, path=path),
        output_dir,
        make_job(**job_overrides),
        resume,
        managers=ItemManagers(transcribe_manager or transcriber(), summary_manager),
    )


def jsonl_log(
    entries: list[dict[str, Any]],
    total_images: int | None = None,
    model_name: str | None = None,
) -> str:
    """Return JSONL working-log content with the current versioned header."""
    header: dict[str, Any] = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": "test",
        "input_item_path": "/fake/test.pdf",
        "input_type": "PDF",
        "total_images": total_images if total_images is not None else len(entries),
    }
    if model_name is not None:
        header["model_name"] = model_name

    lines = [json.dumps(header)]
    lines.extend(json.dumps(entry) for entry in entries)
    return "\n".join(lines) + "\n"


def create_outputs(
    output_dir: Path,
    item_name: str,
    fields: tuple[str, ...] = ALL_OUTPUTS,
    content: str = "content",
) -> None:
    """Create the item's outputs named by ``ItemPaths`` *fields*."""
    paths = ItemPaths.for_item(item_name, output_dir)
    for field in fields:
        path: Path = getattr(paths, field)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def create_log_with_entries(
    output_dir: Path,
    item_name: str,
    entries: list[dict[str, Any]],
) -> Path:
    """Create the item's transcription log holding *entries*."""
    paths = ItemPaths.for_item(item_name, output_dir)
    paths.working_dir.mkdir(parents=True, exist_ok=True)
    paths.transcription_log.write_text(jsonl_log(entries), encoding="utf-8")
    return paths.transcription_log


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return every parsed JSONL object in *path* (header included)."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def page_entries(path: Path, index: int) -> list[dict[str, Any]]:
    """Return the per-page entries in *path* carrying *index*."""
    return [e for e in read_jsonl(path) if e.get("original_input_order_index") == index]


def write_working_logs(
    processor: ItemProcessor,
    tx_entries: list[dict[str, Any]],
    summary_entries: list[dict[str, Any]],
    total_images: int,
) -> None:
    """Write the transcription and summary logs an interrupted run leaves."""

    def _write(path: Path, log_type: str, entries: list[dict[str, Any]]) -> None:
        header = {
            "_format_version": LOG_FORMAT_VERSION,
            "log_type": log_type,
            "input_item_name": processor.name,
            "input_item_path": str(processor.item.path),
            "input_type": "PDF",
            "total_images": total_images,
            "model_name": processor.job.transcription.name,
        }
        lines = [json.dumps(header)] + [json.dumps(e) for e in entries]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write(processor.paths.transcription_log, "transcription", tx_entries)
    _write(processor.paths.summary_log, "summary", summary_entries)


__all__ = [
    "ALL_OUTPUTS",
    "MODEL",
    "clean_page_runner",
    "content_summary",
    "create_log_with_entries",
    "create_outputs",
    "jsonl_log",
    "make_job",
    "make_processor",
    "make_progress",
    "make_runner",
    "page_entries",
    "read_jsonl",
    "summarizer",
    "transcriber",
    "write_working_logs",
]
