"""Regression tests for hardening round 2 (state-integrity / resume holes).

Each test targets one fix applied in this round:

- fix 1: the truncate-then-reappend loss window is closed. ``_reload_completed_
  pages`` re-logs EVERY completed transcription in a first pass (no LLM calls)
  before the slow summary work, so a crash mid-summary can no longer destroy a
  not-yet-reappended completed transcription.
- fix 2: the final ``.txt`` and ``.md`` writers are atomic (temp file +
  ``os.replace``), so a crash mid-write never leaves a truncated output that
  resume classifies as COMPLETE.
- fix 3: on resume, a cheap ``file_provenance`` mismatch (byte size) refuses
  page-level reuse so two documents are never spliced into one output.
- fix 4: a repeated log-header init failure raises instead of silently leaving a
  headerless log.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import pipeline.transcriber as transcriber_module
import rendering.markdown as markdown_module
from config import app as app_config
from config.constants import LOG_FORMAT_VERSION
from pipeline.log import finalize_log_file
from pipeline.paths import create_safe_directory_name, create_safe_log_filename
from pipeline.resume import ProcessingState, ResumeChecker
from pipeline.transcriber import ItemTranscriber
from rendering.markdown import create_markdown_summary
from rendering.text import write_transcription_to_text


def _bare_transcriber() -> ItemTranscriber:
    """Construct an ItemTranscriber without running __init__."""
    return ItemTranscriber.__new__(ItemTranscriber)


def _read_log_indices(log_path: Path) -> list[int]:
    """Return the original_input_order_index of every per-page log line."""
    indices: list[int] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and "original_input_order_index" in obj:
            indices.append(obj["original_input_order_index"])
    return indices


# ---------------------------------------------------------------------------
# fix 1: crash-window — pass 1 re-logs every completed page before summaries
# ---------------------------------------------------------------------------
class TestReloadCrashWindowClosed:
    def test_all_transcriptions_relogged_before_summary_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj.completed_page_indices = {0, 1, 2}
        obj.total_items_to_transcribe = 3
        obj.log_path = tmp_path / "trans.log"
        obj.summary_log_path = tmp_path / "sum.log"
        obj._prior_transcription_results = [
            {"original_input_order_index": i, "image": f"p{i}", "transcription": "t"}
            for i in range(3)
        ]
        obj._prior_summary_results = []  # every page needs a fresh summary
        obj.summary_manager = MagicMock()
        obj._budget_exhausted = threading.Event()  # not set: not budget-deferred
        obj._token_tracker = MagicMock()
        obj._token_tracker.try_reserve.return_value = 123

        monkeypatch.setattr(app_config, "SUMMARIZE", True, raising=False)

        # Pass 2's very first LLM call blows up, simulating a crash / Ctrl+C
        # partway through the slow summary loop.
        def _boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("summary regeneration crashed")

        monkeypatch.setattr(obj, "_summarize_transcription", _boom)

        t_results: list[dict[str, Any]] = []
        s_results: list[dict[str, Any]] = []
        with pytest.raises(RuntimeError, match="crashed"):
            obj._reload_completed_pages(t_results, s_results)

        finalize_log_file(obj.log_path)

        # Pass 1 persisted ALL completed transcriptions BEFORE the crash, so the
        # working log still proves every page complete.
        assert sorted(_read_log_indices(obj.log_path)) == [0, 1, 2]


# ---------------------------------------------------------------------------
# fix 2: atomic final-output writes (.txt and .md)
# ---------------------------------------------------------------------------
class TestAtomicTextWrite:
    def test_happy_path_writes_content(self, tmp_path: Path) -> None:
        out = tmp_path / "doc.txt"
        results = [
            {"transcription": "Page one text."},
            {"transcription": "Page two text."},
        ]
        ok = write_transcription_to_text(
            results, out, "doc", "PDF", 1.0, tmp_path / "doc.pdf"
        )
        assert ok is True
        content = out.read_text(encoding="utf-8")
        assert "Page one text." in content
        assert "Page two text." in content
        # No temp artifact left behind.
        assert not out.with_name(out.name + ".tmp").exists()

    def test_replace_failure_leaves_target_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "doc.txt"
        out.write_text("PRIOR COMPLETE OUTPUT", encoding="utf-8")

        def _boom(_src: Any, _dst: Any) -> None:
            raise OSError("replace failed mid-swap")

        monkeypatch.setattr("rendering.text.os.replace", _boom)

        ok = write_transcription_to_text(
            [{"transcription": "new text"}],
            out,
            "doc",
            "PDF",
            1.0,
            tmp_path / "doc.pdf",
        )
        assert ok is False
        # The pre-existing complete output is untouched (not truncated).
        assert out.read_text(encoding="utf-8") == "PRIOR COMPLETE OUTPUT"
        # The temp file was cleaned up.
        assert not out.with_name(out.name + ".tmp").exists()


class TestAtomicMarkdownWrite:
    def test_happy_path_writes_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(markdown_module, "enrich_if_enabled", lambda _cm: None)
        out = tmp_path / "doc.md"
        create_markdown_summary([], out, "My Document")
        content = out.read_text(encoding="utf-8")
        assert "# My Document" in content
        assert not out.with_name(out.name + ".tmp").exists()

    def test_replace_failure_leaves_target_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(markdown_module, "enrich_if_enabled", lambda _cm: None)
        out = tmp_path / "doc.md"
        out.write_text("PRIOR COMPLETE MD", encoding="utf-8")

        def _boom(_src: Any, _dst: Any) -> None:
            raise OSError("replace failed mid-swap")

        monkeypatch.setattr("rendering.markdown.os.replace", _boom)

        with pytest.raises(OSError, match="replace failed"):
            create_markdown_summary([], out, "My Document")

        assert out.read_text(encoding="utf-8") == "PRIOR COMPLETE MD"
        assert not out.with_name(out.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# fix 3: input-identity validation on resume (cheap provenance mismatch)
# ---------------------------------------------------------------------------
def _write_working_log(
    output_dir: Path,
    item_name: str,
    entries: list[dict[str, Any]],
    file_provenance: dict[str, Any] | None,
) -> Path:
    """Create a versioned transcription working log with an optional provenance."""
    working_dir = output_dir / create_safe_directory_name(item_name, "_working_files")
    working_dir.mkdir(parents=True, exist_ok=True)
    log_path = working_dir / create_safe_log_filename(item_name, "transcription")

    header: dict[str, Any] = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": item_name,
        "input_type": "PDF",
        "total_images": len(entries),
        "model_name": "gpt-5-mini",
    }
    if file_provenance is not None:
        header["file_provenance"] = file_provenance

    lines = [json.dumps(header)] + [json.dumps(e) for e in entries]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


class TestProvenanceIdentityGuard:
    @staticmethod
    def _entries() -> list[dict[str, Any]]:
        return [
            {"original_input_order_index": 0, "transcription": "page zero"},
            {"original_input_order_index": 1, "transcription": "page one"},
        ]

    def _checker(self) -> ResumeChecker:
        return ResumeChecker(
            resume_mode="skip",
            summarize=True,
            output_docx=True,
            output_markdown=True,
        )

    def test_size_mismatch_refuses_page_reuse(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        input_file = tmp_path / "TestDoc.pdf"
        input_file.write_bytes(b"the real current input bytes")

        _write_working_log(
            output_dir,
            "TestDoc",
            self._entries(),
            file_provenance={
                "source_file": str(input_file),
                "size": input_file.stat().st_size + 100,  # stale, different file
            },
        )

        result = self._checker().should_skip("TestDoc", output_dir)
        # Page-level reuse refused: no completed indices threaded, full reprocess.
        assert result.completed_page_indices is None
        assert result.state == ProcessingState.NONE

    def test_size_match_resumes_normally(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        input_file = tmp_path / "TestDoc.pdf"
        input_file.write_bytes(b"the real current input bytes")

        _write_working_log(
            output_dir,
            "TestDoc",
            self._entries(),
            file_provenance={
                "source_file": str(input_file),
                "size": input_file.stat().st_size,  # matches: same file
            },
        )

        result = self._checker().should_skip("TestDoc", output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0, 1}

    def test_legacy_header_without_provenance_resumes(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        _write_working_log(
            output_dir,
            "TestDoc",
            self._entries(),
            file_provenance=None,  # header predates the field
        )

        result = self._checker().should_skip("TestDoc", output_dir)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0, 1}


# ---------------------------------------------------------------------------
# fix 4: repeated log-init failure raises
# ---------------------------------------------------------------------------
class TestInitializeLogOrRaise:
    def test_persistent_failure_raises_after_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        calls: list[int] = []

        def _always_false(*_args: Any, **_kwargs: Any) -> bool:
            calls.append(1)
            return False

        monkeypatch.setattr(transcriber_module, "initialize_log_file", _always_false)

        with pytest.raises(RuntimeError, match="Could not initialize"):
            obj._initialize_log_or_raise(
                tmp_path / "trans.log",
                "doc",
                str(tmp_path / "doc.pdf"),
                "PDF",
                1,
                "gpt-5-mini",
            )
        # Attempted the header write twice (initial + one retry) before raising.
        assert len(calls) == 2

    def test_second_attempt_succeeds_no_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        outcomes = iter([False, True])

        def _flaky(*_args: Any, **_kwargs: Any) -> bool:
            return next(outcomes)

        monkeypatch.setattr(transcriber_module, "initialize_log_file", _flaky)

        # Must NOT raise: the retry succeeds.
        obj._initialize_log_or_raise(
            tmp_path / "trans.log",
            "doc",
            str(tmp_path / "doc.pdf"),
            "PDF",
            1,
            "gpt-5-mini",
        )
