"""The JSONL working logs in pipeline/log.py.

Covers the versioned header and its recorded defaults, appends that reach disk
before finalize, the full init-append-finalize cycle, the atomic header write
that raises after a failed retry, and appends racing a finalize or arriving
after it, which reach disk and leave no open handle.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.pipeline.log as log_mod
from autoexcerpter.pipeline.log import (
    LogHeader,
    append_to_log,
    finalize_log_file,
    initialize_log_file,
    initialize_log_or_raise,
)
from tests.pipeline.helpers import read_jsonl


@pytest.fixture(autouse=True)
def _release_cached_log_handles() -> Generator[None]:
    """Close every cached log handle after each test.

    A leaked descriptor blocks Windows tmp_path cleanup.
    """
    yield
    log_mod.release_logs()


def _header(total_images: int = 1, model_name: str = "gpt-5-mini") -> LogHeader:
    return LogHeader(
        item_name="doc.pdf",
        input_path="/path",
        input_type="pdf",
        total_images=total_images,
        model_name=model_name,
    )


def _first_line(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(
        path.read_text(encoding="utf-8").splitlines()[0]
    )
    return payload


# ============================================================================
# Header, appends and finalize
# ============================================================================
class TestInitializeLogFile:
    """Tests for initialize_log_file."""

    def test_creates_log_file_with_header(self, tmp_path: Path) -> None:
        """The first line is the versioned header with the run values."""
        log_path = tmp_path / "test.log.json"

        result = initialize_log_file(
            log_path,
            LogHeader(
                item_name="test_doc.pdf",
                input_path="/path/to/test_doc.pdf",
                input_type="pdf",
                total_images=10,
                model_name="gpt-5-mini",
                extraction_dpi=300,
                concurrency_limit=8,
                api_timeout=300,
                service_tier="priority",
            ),
        )

        assert result is True
        payload = _first_line(log_path)
        assert payload["_format_version"] == 2
        assert payload["input_item_name"] == "test_doc.pdf"
        assert payload["total_images"] == 10
        assert payload["configuration"] == {
            "concurrent_requests": 8,
            "api_timeout_seconds": 300,
            "model_name": "gpt-5-mini",
            "extraction_dpi": 300,
            "service_tier": "priority",
        }

    def test_header_defaults(self, tmp_path: Path) -> None:
        """Without run values the header records the code defaults."""
        log_path = tmp_path / "test.log.json"

        initialize_log_file(log_path, LogHeader("doc", "/path", "PDF", 1, "gpt-5-mini"))

        configuration = _first_line(log_path)["configuration"]
        assert configuration["concurrent_requests"] == 16
        assert configuration["api_timeout_seconds"] == 900
        assert configuration["service_tier"] == "flex"

    def test_non_openai_model_service_tier_na(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.log.json"

        initialize_log_file(log_path, _header(5, model_name="claude-3-opus"))

        assert _first_line(log_path)["configuration"]["service_tier"] == "N/A"

    def test_returns_false_on_error(self, tmp_path: Path) -> None:
        """A header that cannot be written returns False."""
        bad_path = tmp_path / "nonexistent_dir" / "sub" / "test.log.json"

        assert initialize_log_file(bad_path, _header(5)) is False


class TestAppendToLog:
    """Tests for append_to_log."""

    def test_appends_entry_as_json(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.log.json"
        initialize_log_file(log_path, _header(2))

        result = append_to_log(log_path, {"page": 1, "status": "success"})

        assert result is True
        finalize_log_file(log_path)
        assert '"page": 1' in log_path.read_text(encoding="utf-8")

    def test_entry_is_on_disk_before_finalize(self, tmp_path: Path) -> None:
        """Each appended line is flushed at once: resume tolerates only a
        truncated final line, so completed page records must not wait in a
        buffer until finalize."""
        log_path = tmp_path / "test.log.json"
        initialize_log_file(log_path, _header())
        append_to_log(log_path, {"page": 1, "status": "ok"})

        # Read while the handle is still open and cached.
        content = log_path.read_text(encoding="utf-8")
        assert json.loads(content.splitlines()[-1]) == {"page": 1, "status": "ok"}

    def test_returns_false_on_error(self, tmp_path: Path) -> None:
        """An append into a missing folder returns False."""
        bad_path = tmp_path / "nonexistent_dir" / "log.json"

        assert append_to_log(bad_path, {"test": True}) is False


class TestFinalizeLogFile:
    """Tests for finalize_log_file."""

    def test_finalize_keeps_last_entry(self, tmp_path: Path) -> None:
        """Finalize releases the handle; the last line is a complete object."""
        log_path = tmp_path / "test.log.json"
        initialize_log_file(log_path, _header())
        append_to_log(log_path, {"page": 1, "status": "ok"})

        assert finalize_log_file(log_path) is True

        content = log_path.read_text(encoding="utf-8")
        assert json.loads(content.splitlines()[-1]) == {"page": 1, "status": "ok"}

    def test_full_lifecycle_produces_valid_json(self, tmp_path: Path) -> None:
        """Init, two appends and finalize give one parseable object per line."""
        log_path = tmp_path / "test.log.json"
        initialize_log_file(log_path, _header(2))
        append_to_log(log_path, {"page": 1, "status": "ok"})
        append_to_log(log_path, {"page": 2, "status": "ok"})
        finalize_log_file(log_path)

        content = log_path.read_text(encoding="utf-8")
        lines = [json.loads(ln) for ln in content.splitlines() if ln.strip()]
        assert len(lines) == 3
        assert lines[0]["_format_version"] == 2
        assert [line["page"] for line in lines[1:]] == [1, 2]


# ============================================================================
# Header retry and atomic write
# ============================================================================
class TestInitializeLogOrRaise:
    def test_persistent_failure_raises_after_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def _always_false(*_args: Any, **_kwargs: Any) -> bool:
            calls.append(1)
            return False

        monkeypatch.setattr(log_mod, "initialize_log_file", _always_false)

        with pytest.raises(RuntimeError, match="Could not initialize"):
            initialize_log_or_raise(
                tmp_path / "trans.log",
                LogHeader("doc", str(tmp_path / "doc.pdf"), "PDF", 1, "gpt-5-mini"),
            )
        # The header write ran twice (initial attempt and one retry).
        assert len(calls) == 2

    def test_second_attempt_succeeds_no_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcomes = iter([False, True])

        def _flaky(*_args: Any, **_kwargs: Any) -> bool:
            return next(outcomes)

        monkeypatch.setattr(log_mod, "initialize_log_file", _flaky)

        # The retry succeeds, so nothing is raised.
        initialize_log_or_raise(
            tmp_path / "trans.log",
            LogHeader("doc", str(tmp_path / "doc.pdf"), "PDF", 1, "gpt-5-mini"),
        )


class TestInitializeLogAtomicWrite:
    def test_header_write_failure_preserves_prior_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = tmp_path / "trans.log"
        prior = '{"_format_version": 1, "completed": "page"}\n'
        log_path.write_text(prior, encoding="utf-8")

        def boom(_src: Any, _dst: Any) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr("autoexcerpter.common.jsonl.os.replace", boom)

        ok = log_mod.initialize_log_file(
            log_path,
            log_mod.LogHeader("item", str(tmp_path / "in.pdf"), "PDF", 3, "gpt-5-mini"),
        )

        assert ok is False
        # The prior completed-page entries survive the failed re-init.
        assert log_path.read_text(encoding="utf-8") == prior
        # No temp file is left behind.
        assert list(tmp_path.glob("*.tmp")) == []

    def test_entries_written_with_header(self, tmp_path: Path) -> None:
        """Retained entries land after the header; later appends follow them."""
        log_path = tmp_path / "trans.log"
        log_path.write_text('{"_format_version": 2}\n{"page": 9}\n', encoding="utf-8")

        assert initialize_log_file(log_path, _header(3), [{"page": 0}, {"page": 1}])
        assert append_to_log(log_path, {"page": 2}) is True
        finalize_log_file(log_path)

        lines = read_jsonl(log_path)
        assert lines[0]["input_item_name"] == "doc.pdf"
        assert [line["page"] for line in lines[1:]] == [0, 1, 2]


# ============================================================================
# Appends around finalize
# ============================================================================
class TestFinalizeAppendRace:
    """A page entry reaches disk whichever side of a finalize it lands on."""

    def test_append_after_finalize_still_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "finalized.jsonl"
        assert log_mod.append_to_log(path, {"page": 0}) is True
        log_mod.finalize_log_file(path)

        assert log_mod.append_to_log(path, {"page": 1}) is True

        assert [e["page"] for e in read_jsonl(path)] == [0, 1]

    def test_concurrent_append_and_finalize_lose_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "concurrent.jsonl"
        rounds = 40
        for i in range(rounds):
            appender = threading.Thread(
                target=log_mod.append_to_log, args=(path, {"page": i})
            )
            finalizer = threading.Thread(target=log_mod.finalize_log_file, args=(path,))
            appender.start()
            finalizer.start()
            appender.join()
            finalizer.join()

        assert sorted(e["page"] for e in read_jsonl(path)) == list(range(rounds))


class TestLogHandleReleasedAfterFinalize:
    """No log handle stays open once a file is finalized."""

    def test_one_shot_append_after_finalize(self, tmp_path: Path) -> None:
        path = tmp_path / "work.jsonl"
        assert log_mod.append_to_log(path, {"a": 1}) is True
        assert log_mod.finalize_log_file(path) is True

        # A late worker append is written and leaves no handle open.
        assert log_mod.append_to_log(path, {"b": 2}) is True

        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines == ['{"a": 1}', '{"b": 2}']
        path.unlink()

    def test_no_handle_survives_a_racing_finalize(self, tmp_path: Path) -> None:
        """Either order is safe; neither leaves an open handle on the file."""
        paths = [tmp_path / f"race_{i}.jsonl" for i in range(40)]
        for index, path in enumerate(paths):
            appender = threading.Thread(
                target=log_mod.append_to_log, args=(path, {"page": index})
            )
            finalizer = threading.Thread(target=log_mod.finalize_log_file, args=(path,))
            appender.start()
            finalizer.start()
            appender.join()
            finalizer.join()
            assert path.read_text(encoding="utf-8") == f'{{"page": {index}}}\n'
            path.unlink()
