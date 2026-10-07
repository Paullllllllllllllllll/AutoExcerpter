"""Resume reload of logged pages and summaries (pipeline/item.py).

``eligible_prior_entries`` picks the reusable logged pages. The resumed log
is rewritten with its header and those pages in one atomic replace, and a
failed rewrite keeps the previous log. ``reload_completed_pages`` admits the
pages and regenerates summaries only where missing. A working log without the
image-settings fingerprint is refused before any page is reused.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import autoexcerpter.pipeline.item as item_module
import autoexcerpter.pipeline.log as log_mod
from autoexcerpter.constants import LOG_FORMAT_VERSION
from autoexcerpter.pipeline.item import (
    ItemProcessor,
    eligible_prior_entries,
    reload_completed_pages,
)
from autoexcerpter.pipeline.job import ItemResume
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.resume import (
    ProcessingState,
    ResumeChecker,
    verify_image_settings,
)
from tests.pipeline.helpers import (
    clean_page_runner,
    content_summary,
    make_processor,
    make_runner,
    page_entries,
    read_jsonl,
    summarizer,
    write_working_logs,
)


@pytest.fixture(autouse=True)
def _release_cached_log_handles() -> Generator[None]:
    """Close every cached log handle after each test.

    A leaked descriptor blocks Windows tmp_path cleanup.
    """
    yield
    log_mod.release_logs()


def _entry(idx: int, text: str = "content") -> dict[str, Any]:
    return {
        "original_input_order_index": idx,
        "image": f"p{idx}",
        "transcription": text,
    }


def _other_page_summarizer() -> MagicMock:
    return summarizer(
        return_value={
            "page": 1,
            "page_information": {"page_types": ["other"]},
            "bullet_points": ["point"],
            "references": [],
        }
    )


def _read_log_indices(log_path: Path) -> list[int]:
    """Return the original_input_order_index of every per-page log line."""
    return [
        e["original_input_order_index"]
        for e in read_jsonl(log_path)
        if "original_input_order_index" in e
    ]


class TestEligiblePriorEntries:
    """A duplicated index is replayed once, as the error-free entry."""

    def test_error_entry_first_is_dropped(self) -> None:
        bad = {"original_input_order_index": 0, "error": "api", "transcription": "x"}
        good = {"original_input_order_index": 0, "transcription": "real text"}

        eligible = eligible_prior_entries([bad, good], frozenset({0}), 2)

        assert eligible == [good]

    def test_error_entry_last_is_dropped(self) -> None:
        good = {"original_input_order_index": 0, "transcription": "real text"}
        bad = {"original_input_order_index": 0, "error": "api", "transcription": "x"}

        eligible = eligible_prior_entries([good, bad], frozenset({0}), 2)

        assert eligible == [good]

    def test_distinct_indices_are_all_kept_in_index_order(self) -> None:
        one = {"original_input_order_index": 1, "transcription": "b"}
        zero = {"original_input_order_index": 0, "transcription": "a"}

        eligible = eligible_prior_entries([one, zero], frozenset({0, 1}), 2)

        assert eligible == [zero, one]


class TestReloadPhantomIndex:
    def test_phantom_index_beyond_page_count_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stale log entry whose idx exceeds the current page count is
        # skipped entirely (neither logged nor counted). The page count is 1:
        # the input was swapped for a shorter file.
        prior = [
            {"original_input_order_index": 5, "image": "p5", "transcription": "stale"}
        ]
        eligible = eligible_prior_entries(prior, frozenset({0, 5}), 1)

        logged: list[tuple[Path, dict[str, Any]]] = []
        monkeypatch.setattr(
            item_module,
            "append_to_log",
            lambda path, entry: logged.append((path, entry)),
        )

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        asyncio.run(
            reload_completed_pages(
                eligible, [], make_runner(tmp_path), t_results, s_results
            )
        )

        assert t_results == []
        assert logged == []


class TestResumeLogRewrite:
    """The resumed log is rewritten with its reused pages in one replace."""

    def _resumed(self, tmp_path: Path) -> ItemProcessor:
        processor = make_processor(
            tmp_path / "doc.pdf", tmp_path / "out", completed={0, 1}
        )
        write_working_logs(
            processor, [_entry(0, "zero"), _entry(1, "one")], [], total_images=2
        )
        return processor

    def test_header_and_reused_pages_written_in_one_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        processor = self._resumed(tmp_path)
        source: Any = _TwoPageSource()
        replaced: list[Path] = []
        original = os.replace

        def counting_replace(src: Any, dst: Any) -> None:
            replaced.append(Path(dst))
            original(src, dst)

        monkeypatch.setattr("autoexcerpter.common.jsonl.os.replace", counting_replace)

        processor._load_resume_state(source)
        processor._init_transcription_log(source)

        log_path = processor.paths.transcription_log
        assert replaced == [log_path]
        entries = read_jsonl(log_path)
        assert entries[0]["_format_version"] == LOG_FORMAT_VERSION
        assert entries[0]["log_type"] == "transcription"
        assert _read_log_indices(log_path) == [0, 1]

    def test_failed_rewrite_keeps_the_previous_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        processor = self._resumed(tmp_path)
        source: Any = _TwoPageSource()
        log_path = processor.paths.transcription_log
        before = log_path.read_bytes()

        def boom(_src: Any, _dst: Any) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr("autoexcerpter.common.jsonl.os.replace", boom)

        processor._load_resume_state(source)
        with pytest.raises(RuntimeError, match="Could not initialize"):
            processor._init_transcription_log(source)

        assert log_path.read_bytes() == before


class TestReloadReusedSummaries:
    def test_reusable_summary_not_appended_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _other_page_summarizer()
        runner = make_runner(tmp_path, summary_manager=manager)
        summary_log = runner.paths.summary_log
        eligible = eligible_prior_entries([_entry(0), _entry(1)], frozenset({0, 1}), 2)
        # Page 0 has a reusable error-free prior summary; page 1 has none.
        prior_summaries = [
            {
                "original_input_order_index": 0,
                "page_information": {"page_types": ["other"]},
                "bullet_points": ["reused"],
                "references": [],
            }
        ]

        events: list[tuple[str, Any, Any]] = []

        def fake_append(path: Path, entry: dict[str, Any]) -> None:
            events.append(("append", path, entry.get("original_input_order_index")))

        monkeypatch.setattr(item_module, "append_to_log", fake_append)

        def fake_generate(text: str, page: Any) -> dict[str, Any]:
            events.append(("generate", None, None))
            return {
                "page": page,
                "page_information": {"page_types": ["other"]},
                "bullet_points": ["new"],
                "references": [],
            }

        manager.generate_summary.side_effect = fake_generate

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        asyncio.run(
            reload_completed_pages(
                eligible, prior_summaries, runner, t_results, s_results
            )
        )

        # Page 0's reused summary is already in the rewritten summary log, so
        # the reload appends only page 1's regenerated summary.
        appends = [e for e in events if e[0] == "append"]
        assert [(e[1], e[2]) for e in appends] == [(summary_log, 1)]

        # Both pages admitted; page 1 regenerated exactly once.
        assert sorted(r["original_input_order_index"] for r in t_results) == [0, 1]
        assert len(s_results) == 2
        assert manager.generate_summary.call_count == 1


class TestReloadBlankPage:
    def test_blank_page_summary_is_a_placeholder(self, tmp_path: Path) -> None:
        manager = _other_page_summarizer()
        runner = make_runner(tmp_path, summary_manager=manager)
        eligible = [_entry(0, text="[p0: no transcribable text]")]

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        asyncio.run(reload_completed_pages(eligible, [], runner, t_results, s_results))

        # Blank page completed with a placeholder and no LLM summary call.
        assert manager.generate_summary.call_count == 0
        assert len(t_results) == 1
        assert len(s_results) == 1
        assert s_results[0]["page_information"]["page_types"] == ["blank"]


def _write_crash_state(tmp_path: Path, item_name: str = "doc") -> tuple[Path, Path]:
    """Write the on-disk state left by a crash during the summary call.

    Transcription log: header + one error-free page entry. Summary log: header
    only (the page's summary never made it).
    """
    paths = ItemPaths.for_item(item_name, tmp_path / "out")
    paths.working_dir.mkdir(parents=True)
    t_log = paths.transcription_log
    s_log = paths.summary_log

    def _header(log_type: str) -> dict[str, Any]:
        return {
            "_format_version": LOG_FORMAT_VERSION,
            "log_type": log_type,
            "input_item_name": item_name,
            "total_images": 1,
            "model_name": "test-model",
        }

    page = {
        "image": "page_0000.jpg",
        "transcription": "Hello world",
        "processing_time": 0.01,
        "provider": "openai",
        "original_input_order_index": 0,
    }
    t_log.write_text(
        json.dumps(_header("transcription"), ensure_ascii=False)
        + "\n"
        + json.dumps(page, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    s_log.write_text(
        json.dumps(_header("summary"), ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return t_log, s_log


class TestCrashMidSummaryResumes:
    """The logged transcription is reused; only the summary is regenerated."""

    def test_resume_checker_reports_the_page_completed(self, tmp_path: Path) -> None:
        _write_crash_state(tmp_path)

        result = ResumeChecker(resume_mode="skip", summarize=True)._check_item(
            "doc", tmp_path / "out"
        )

        assert result.state is not ProcessingState.COMPLETE
        assert result.completed_page_indices == {0}

    def test_reload_regenerates_only_the_missing_summary(self, tmp_path: Path) -> None:
        t_log, s_log = _write_crash_state(tmp_path)
        manager = summarizer(return_value=content_summary())
        runner = clean_page_runner(tmp_path / "out", summary_manager=manager)
        assert runner.paths.transcription_log == t_log
        prior = [e for e in read_jsonl(t_log) if "original_input_order_index" in e]

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        asyncio.run(
            reload_completed_pages(
                eligible_prior_entries(prior, frozenset({0}), 1),
                [],
                runner,
                t_results,
                s_results,
            )
        )

        # The summary was regenerated from the logged transcription ...
        assert manager.generate_summary.call_count == 1
        # ... and no page was re-transcribed.
        runner.transcribe_manager.transcribe_payload.assert_not_called()  # type: ignore[attr-defined]
        assert len(t_results) == 1
        assert len(s_results) == 1
        assert len(page_entries(s_log, 0)) == 1


def _fake_transcribe(payload: Any) -> dict[str, Any]:
    return {
        "image": payload.image_name,
        "sequence_number": payload.number,
        "transcription": f"Fresh transcription of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _fake_summary(transcription: str, page_num: int) -> dict[str, Any]:
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": [f"REGEN summary for page {page_num}"],
        "references": None,
        "provider": "openai",
    }


@pytest.fixture
def summarizing_env(mock_image_processing: dict[str, Any]) -> dict[str, MagicMock]:
    """Return mocks of both LLM managers; the jobs make no OpenAlex calls."""
    mock_tx = MagicMock()
    mock_tx.transcribe_payload = AsyncMock(side_effect=_fake_transcribe)
    mock_sum = MagicMock()
    mock_sum.generate_summary = AsyncMock(side_effect=_fake_summary)
    return {"tx": mock_tx, "sum": mock_sum}


def test_resume_reuses_prior_summaries_and_skips_transcription(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    summarizing_env: dict[str, MagicMock],
) -> None:
    """A resumed item's Markdown keeps the logged summaries; nothing is re-transcribed.

    Page 0 has both a logged transcription and a logged summary (reused).
    Page 1 has a logged transcription but no summary (regenerated from the log,
    with no transcription API call).
    """
    pdf_path = make_pdf("Resume.pdf", num_pages=2)
    processor: ItemProcessor = make_processor(
        pdf_path,
        tmp_path / "out",
        transcribe_manager=summarizing_env["tx"],
        summary_manager=summarizing_env["sum"],
        completed={0, 1},
        summarize=True,
        openalex_enabled=False,
    )
    write_working_logs(
        processor,
        tx_entries=[
            {
                "image": "page_0001.jpg",
                "original_input_order_index": 0,
                "transcription": "prior text page zero",
            },
            {
                "image": "page_0002.jpg",
                "original_input_order_index": 1,
                "transcription": "prior text page one",
            },
        ],
        summary_entries=[
            {
                "original_input_order_index": 0,
                "image_filename": "page_0001.jpg",
                "page": 1,
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["PRIOR pre-crash summary of page zero"],
                "references": None,
            }
        ],
        total_images=2,
    )

    asyncio.run(processor.process())

    # No transcription API call: both pages are reused from the log.
    assert summarizing_env["tx"].transcribe_payload.call_count == 0
    # Page 0 summary reused (not regenerated); page 1 summary regenerated once.
    assert summarizing_env["sum"].generate_summary.call_count == 1

    # The logged summary of page 0 reaches the rendered Markdown, and the
    # regenerated page-1 summary is present too.
    md = processor.paths.summary_md.read_text(encoding="utf-8")
    assert "PRIOR pre-crash summary of page zero" in md
    # Page at index 1 has no logged "page" key, so its regenerated summary uses
    # the fallback numbering (index + 1 = 2).
    assert "REGEN summary for page 2" in md
    # The summary log holds each page's summary once.
    summary_log = processor.paths.summary_log
    assert [len(page_entries(summary_log, idx)) for idx in (0, 1)] == [1, 1]


_NO_FINGERPRINT = [
    pytest.param(None, id="no-provenance"),
    pytest.param(
        {"source_file": "doc.pdf", "image_config": {}},
        id="provenance-without-fingerprint",
    ),
]


def _fingerprint_header(provenance: dict[str, Any] | None) -> dict[str, Any]:
    header: dict[str, Any] = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": "doc",
        "total_images": 2,
        "model_name": "test-model",
    }
    if provenance is not None:
        header["file_provenance"] = provenance
    return header


def _write_half_done_log(output_dir: Path, provenance: dict[str, Any] | None) -> Path:
    """Write a transcription log with page 0 done and page 1 still pending."""
    paths = ItemPaths.for_item("doc", output_dir)
    paths.working_dir.mkdir(parents=True)
    log_path = paths.transcription_log
    page = {"original_input_order_index": 0, "transcription": "Page zero."}
    log_path.write_text(
        json.dumps(_fingerprint_header(provenance)) + "\n" + json.dumps(page) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return log_path


class _TwoPageSource:
    """Payload source stub with two pages and no I/O."""

    img_cfg: dict[str, Any] = {"payload_format": "jpeg"}

    def __len__(self) -> int:
        return 2

    def file_provenance(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


@pytest.mark.parametrize("provenance", _NO_FINGERPRINT)
def test_verify_image_settings_requires_fingerprint(
    provenance: dict[str, Any] | None,
) -> None:
    with pytest.raises(ValueError, match="image_settings_fingerprint"):
        verify_image_settings(
            _fingerprint_header(provenance), {"payload_format": "jpeg"}
        )


@pytest.mark.parametrize("provenance", _NO_FINGERPRINT)
def test_resume_refuses_log_without_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provenance: dict[str, Any] | None,
) -> None:
    output_dir = tmp_path / "out"
    log_path = _write_half_done_log(output_dir, provenance)
    before = log_path.read_bytes()

    result = ResumeChecker("skip", summarize=False).should_skip("doc", output_dir)
    assert result.state is not ProcessingState.COMPLETE

    processor = make_processor(
        tmp_path / "doc.pdf",
        output_dir,
        resume=ItemResume(
            completed_page_indices=frozenset(result.completed_page_indices or ()),
            transcription_results=result.transcription_results,
            summary_results=result.summary_results,
            log_header=result.log_header,
        ),
    )
    assert processor.paths.transcription_log == log_path

    async def _must_not_run(_source: Any) -> Any:
        raise AssertionError("a refused log must not reach page processing")

    monkeypatch.setattr(processor, "_transcribe_and_summarize", _must_not_run)

    with pytest.raises(ValueError, match="image_settings_fingerprint"):
        asyncio.run(processor._run(_TwoPageSource()))  # type: ignore[arg-type]

    # No logged page was reused and the working log was not rewritten.
    assert log_path.read_bytes() == before
