"""Resume and run-control flows, recorded through ``run_tool``.

Each test runs the tool more than once over the same output folder. Goldens
live under ``golden/resume/<case>/<run>/``; each run's snapshot holds the whole
output folder as it stands after that run, so files a resumed run rewrites
appear as well as files it creates. All runs use the default OpenAI schema
path on a three-page PDF unless a test says otherwise.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.rendering.sqlite import DATABASE_NAME
from tests.characterization.adapter import RunResult, item_paths, run_tool
from tests.characterization.fakes import (
    SUMMARY,
    TRANSCRIPTION,
    FakeLLM,
    FakeOpenAlex,
    Injection,
)
from tests.characterization.golden import GOLDEN_DIR, check_golden, snapshot
from tests.characterization.inputs import make_pdf
from tests.conftest import WriteGuard

Golden = Callable[[str, RunResult], None]

PAGES = 3
FAILED_PAGE = 1
REFUSAL = Injection(refusal="I cannot help with this page.")
CHANGED_JPEG_QUALITY = 80
# Paths relative to the output folder, as the snapshot keys them.
DOC = item_paths("doc", Path())
WORKING_DIR = DOC.working_dir.as_posix()
WORKING_LOGS = {DOC.transcription_log.as_posix(), DOC.summary_log.as_posix()}
FINAL_OUTPUTS = (
    DOC.transcription.as_posix(),
    DOC.summary_md.as_posix(),
    f"{DOC.summary_docx.as_posix()}.dump.txt",
)


# ============================================================================
# Helpers
# ============================================================================
class Runner:
    """Run the tool repeatedly over one input and one output folder."""

    def __init__(self, root: Path, llm: FakeLLM, golden: Golden) -> None:
        self.root = root
        self.llm = llm
        self.golden = golden
        self.output_dir = root / "out"

    def run(self, input_path: Path, **kwargs: Any) -> RunResult:
        """Forget earlier requests, then run the tool once."""
        self.llm.reset_records()
        return run_tool(
            input_path,
            llm=self.llm,
            root=self.root,
            output_dir=self.output_dir,
            **kwargs,
        )

    def check(self, case: str, result: RunResult) -> None:
        """Compare the run and the output folder after it with the golden."""
        self.golden(f"resume/{case}", with_output_state(result))


def with_output_state(result: RunResult) -> RunResult:
    """Return *result* with every file now in the output folder as created."""
    files = sorted(path for path in result.output_dir.rglob("*") if path.is_file())
    return dataclasses.replace(result, created=files)


def output_files(result: RunResult) -> dict[str, str]:
    """Return the normalized snapshot of the output folder's files."""
    return {
        name.removeprefix("files/out/"): text
        for name, text in snapshot(with_output_state(result)).items()
        if name.startswith("files/")
    }


def final_outputs(result: RunResult) -> dict[str, str]:
    """Return the normalized transcription and summary files only."""
    files = output_files(result)
    return {name: files[name] for name in FINAL_OUTPUTS if name in files}


def folder_bytes(folder: Path) -> dict[str, bytes]:
    """Return the raw bytes of every file under *folder*."""
    if not folder.is_dir():
        return {}
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def pages(result: RunResult, role: str) -> list[int | None]:
    """Return the page of every request of *role*, in page order."""
    return [call["page"] for call in result.calls(role)]


def refuse(llm: FakeLLM, page: int = FAILED_PAGE) -> None:
    """Make the first transcription request for *page* a refusal."""
    llm.inject(page, TRANSCRIPTION, 1, REFUSAL)


def stop_refusing(llm: FakeLLM, page: int = FAILED_PAGE) -> None:
    """Remove every scripted failure from *page*."""
    llm.page(page).injections.clear()


def assert_json_last_line(result: RunResult) -> dict[str, Any]:
    """Assert that the last stdout line is the JSON run summary; return it."""
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.stderr
    parsed: dict[str, Any] = json.loads(lines[-1])
    assert parsed == result.summary
    return parsed


def plan_summary(result: RunResult) -> dict[str, Any]:
    """Return a dry run's JSON summary without its effective spec.

    The spec must carry every run option with its value and source.
    """
    assert result.summary is not None, result.stdout
    summary = dict(result.summary)
    spec = summary.pop("spec")
    assert spec["run.dry_run"] == {"value": True, "source": "flag"}
    assert spec["run.concurrency"] == {"value": 16, "source": "default"}
    assert all(set(entry) == {"value", "source"} for entry in spec.values())
    return summary


def assert_counts(result: RunResult, **expected: int) -> None:
    """Assert the JSON summary's item counts."""
    assert result.summary is not None, result.stdout
    counts = {key: result.summary[f"items_{key}"] for key in expected}
    assert counts == expected


@pytest.fixture
def golden(fake_openalex: FakeOpenAlex, write_guard: WriteGuard) -> Golden:
    """Return ``check(case, result)``; updates may write anywhere in resume/."""

    def check(case: str, result: RunResult) -> None:
        check_golden(
            case,
            result,
            openalex=fake_openalex,
            on_write=lambda _case_dir: write_guard.consume(GOLDEN_DIR / "resume"),
        )

    return check


@pytest.fixture
def runner(tmp_path: Path, fake_llm: FakeLLM, golden: Golden) -> Runner:
    return Runner(tmp_path, fake_llm, golden)


@pytest.fixture
def pdf(docs_dir: Path) -> Path:
    return make_pdf(docs_dir / "doc.pdf", pages=PAGES)


def clean_outputs(tmp_path: Path, fake_llm: FakeLLM, golden: Golden) -> RunResult:
    """Run a clean, complete pass in a sibling root; check its golden."""
    root = tmp_path / "clean"
    clean_pdf = make_pdf(root / "docs" / "doc.pdf", pages=PAGES)
    clean = Runner(root, fake_llm, golden)
    result = clean.run(clean_pdf)
    assert result.exit_code == 0, result.stderr
    clean.check("clean", result)
    return result


def failed_first_run(runner: Runner, input_path: Path, **kwargs: Any) -> RunResult:
    """Run once with page ``FAILED_PAGE`` refused; return the failed run."""
    refuse(runner.llm)
    result = runner.run(input_path, **kwargs)
    stop_refusing(runner.llm)
    return result


# ============================================================================
# (1) A complete item is skipped
# ============================================================================
def test_complete_item_is_skipped(runner: Runner, pdf: Path) -> None:
    first = runner.run(pdf)
    assert first.exit_code == 0, first.stderr
    runner.check("complete_skipped/run1", first)
    before = folder_bytes(runner.output_dir)

    second = runner.run(pdf)

    assert second.exit_code == 0, second.stderr
    assert_counts(second, total=0, complete=0, failed=0, skipped=1)
    assert second.summary is not None
    assert second.summary["outputs"] == []
    assert second.calls() == []
    assert second.llm.models == []
    assert second.created == []
    assert folder_bytes(runner.output_dir) == before
    runner.check("complete_skipped/run2", second)


# ============================================================================
# (2) Partial failure, then a resume that requests only the missing page
# ============================================================================
def test_partial_failure_then_resume(
    tmp_path: Path, runner: Runner, pdf: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    clean = clean_outputs(tmp_path, fake_llm, golden)

    first = failed_first_run(runner, pdf)

    assert first.exit_code == 1, first.stderr
    assert_counts(first, total=1, complete=0, failed=1, skipped=0)
    assert_json_last_line(first)
    assert "finished INCOMPLETE" in first.stderr
    assert pages(first, TRANSCRIPTION) == [0, 1, 2]
    assert pages(first, SUMMARY) == [0, 2]
    assert WORKING_LOGS.issubset(output_files(first))
    runner.check("partial_failure/run1", first)

    second = runner.run(pdf)

    assert second.exit_code == 0, second.stderr
    assert_counts(second, total=1, complete=1, failed=0, skipped=0)
    assert pages(second, TRANSCRIPTION) == [FAILED_PAGE]
    assert pages(second, SUMMARY) == [FAILED_PAGE]
    runner.check("partial_failure/run2", second)
    assert output_files(second) == output_files(clean)


# ============================================================================
# (3) --force reprocesses every page
# ============================================================================
def test_force_reprocesses_every_page(
    tmp_path: Path, runner: Runner, pdf: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    clean = clean_outputs(tmp_path, fake_llm, golden)
    first = runner.run(pdf)
    assert first.exit_code == 0, first.stderr

    second = runner.run(pdf, force=True)

    assert second.exit_code == 0, second.stderr
    assert_counts(second, total=1, complete=1, failed=0, skipped=0)
    assert pages(second, TRANSCRIPTION) == [0, 1, 2]
    assert pages(second, SUMMARY) == [0, 1, 2]
    assert final_outputs(second) == final_outputs(clean)
    assert not (runner.output_dir / WORKING_DIR).exists()
    runner.check("force/run2", second)


# ============================================================================
# (4) --retranscribe from each resumable start state
# ============================================================================
def _start_transcription_only(runner: Runner, pdf: Path) -> RunResult:
    return runner.run(pdf, summarize=False)


def _start_page_failed(runner: Runner, pdf: Path) -> RunResult:
    return failed_first_run(runner, pdf)


def _start_complete(runner: Runner, pdf: Path) -> RunResult:
    return runner.run(pdf)


RETRANSCRIBE_STARTS: dict[str, Callable[[Runner, Path], RunResult]] = {
    "transcription_only": _start_transcription_only,
    "page_failed": _start_page_failed,
    "complete": _start_complete,
}


@pytest.mark.parametrize("start", list(RETRANSCRIBE_STARTS))
def test_retranscribe(runner: Runner, pdf: Path, start: str) -> None:
    first = RETRANSCRIBE_STARTS[start](runner, pdf)
    runner.check(f"retranscribe_{start}/run1", first)

    second = runner.run(pdf, retranscribe=True)

    phases = {
        TRANSCRIPTION: pages(second, TRANSCRIPTION),
        SUMMARY: pages(second, SUMMARY),
    }
    if start == "complete":
        assert phases == {TRANSCRIPTION: [], SUMMARY: []}
        assert_counts(second, total=0, complete=0, failed=0, skipped=1)
    else:
        assert phases == {TRANSCRIPTION: [0, 1, 2], SUMMARY: [0, 1, 2]}
        assert_counts(second, total=1, complete=1, failed=0, skipped=0)
    assert second.exit_code == 0, second.stderr
    runner.check(f"retranscribe_{start}/run2", second)


# ============================================================================
# (5) Transcription only, then summaries without re-transcription
# ============================================================================
def test_transcription_only_then_summary(runner: Runner, pdf: Path) -> None:
    first = runner.run(pdf, summarize=False)
    assert first.exit_code == 0, first.stderr
    assert pages(first, SUMMARY) == []
    runner.check("transcription_only_then_summary/run1", first)

    plan = runner.run(pdf, dry_run=True)
    assert plan_summary(plan) == {
        "dry_run": True,
        "to_process": [
            {"name": "doc", "state": "transcription_only", "completed_pages": 3}
        ],
        "skipped": [],
    }

    second = runner.run(pdf, summarize=True)

    assert second.exit_code == 0, second.stderr
    assert_counts(second, total=1, complete=1, failed=0, skipped=0)
    assert pages(second, TRANSCRIPTION) == []
    assert pages(second, SUMMARY) == [0, 1, 2]
    runner.check("transcription_only_then_summary/run2", second)


# ============================================================================
# (6) Dry run over a fresh, a partial and a complete item
# ============================================================================
def test_dry_run_plan(tmp_path: Path, runner: Runner) -> None:
    tree = tmp_path / "in" / "tree"
    for stem in ("alpha", "beta", "gamma"):
        make_pdf(tree / f"{stem}.pdf", pages=PAGES)
    complete = runner.run(tree, selection="gamma")
    assert complete.exit_code == 0, complete.stderr
    failed = failed_first_run(runner, tree, selection="beta")
    assert failed.exit_code == 1, failed.stderr
    # A crash mid-item leaves the working logs but no final outputs.
    beta = item_paths("beta", runner.output_dir)
    for path in (beta.transcription, beta.summary_md, beta.summary_docx):
        path.unlink()
    before = folder_bytes(runner.output_dir)

    result = runner.run(tree, dry_run=True)

    assert result.exit_code == 0, result.stderr
    assert plan_summary(result) == {
        "dry_run": True,
        "to_process": [
            {"name": "alpha", "state": "none", "completed_pages": 0},
            {"name": "beta", "state": "partial", "completed_pages": 2},
        ],
        "skipped": ["gamma"],
    }
    assert_json_last_line(result)
    assert result.calls() == []
    assert result.llm.models == []
    assert result.created == []
    assert folder_bytes(runner.output_dir) == before
    runner.check("dry_run_plan", result)


def test_dry_run_fresh_output_folder_is_not_created(runner: Runner, pdf: Path) -> None:
    result = runner.run(pdf, dry_run=True)

    assert result.exit_code == 0, result.stderr
    assert plan_summary(result) == {
        "dry_run": True,
        "to_process": [{"name": "doc", "state": "none", "completed_pages": 0}],
        "skipped": [],
    }
    assert not runner.output_dir.exists()
    assert result.calls() == []


# ============================================================================
# (7) A rerun after changing a fingerprinted image setting
# ============================================================================
@pytest.mark.parametrize("start", ["page_failed", "complete"])
def test_changed_image_setting(runner: Runner, pdf: Path, start: str) -> None:
    first = RETRANSCRIBE_STARTS[start](runner, pdf)
    runner.check(f"image_setting_changed_{start}/run1", first)
    before = folder_bytes(runner.output_dir)

    second = runner.run(pdf, jpeg_quality=CHANGED_JPEG_QUALITY)

    assert_json_last_line(second)
    if start == "complete":
        assert second.exit_code == 0, second.stderr
        assert_counts(second, total=0, complete=0, failed=0, skipped=1)
    else:
        assert second.exit_code == 1, second.stderr
        assert_counts(second, total=1, complete=0, failed=1, skipped=0)
        assert "Image settings changed: jpeg_quality" in second.stderr
    assert second.calls() == []
    assert folder_bytes(runner.output_dir) == before
    runner.check(f"image_setting_changed_{start}/run2", second)


# ============================================================================
# (7b) A rerun of the failed page with another transcription model
# ============================================================================
FIRST_MODEL = "gpt-5.6-terra"
SWITCHED = {
    "transcription_provider": "openrouter",
    "transcription_model": "z-ai/glm-5.3-flash",
}


def page_models(output_dir: Path) -> dict[int, str | None]:
    """Return the ``pages.model`` of each page in the output database."""
    with contextlib.closing(sqlite3.connect(output_dir / DATABASE_NAME)) as db:
        rows = db.execute("SELECT page_index, model FROM pages").fetchall()
    return dict(rows)


def strip_page_models(log_path: Path) -> None:
    """Rewrite a working log as a log without per-page models."""
    entries = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
    for entry in entries:
        entry.pop("model", None)
    log_path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )


def test_model_switch_on_resume_proceeds_and_warns(runner: Runner, pdf: Path) -> None:
    first = failed_first_run(runner, pdf)
    assert first.exit_code == 1, first.stderr

    second = runner.run(pdf, **SWITCHED)

    assert second.exit_code == 0, second.stderr
    assert_counts(second, total=1, complete=1, failed=0, skipped=0)
    assert pages(second, TRANSCRIPTION) == [FAILED_PAGE]
    assert "Resume with another model" in second.stderr
    assert "Image settings changed" not in second.stderr
    transcript = (runner.output_dir / DOC.transcription).read_text(encoding="utf-8")
    assert "Resume with another model" in transcript
    assert page_models(runner.output_dir) == {
        0: FIRST_MODEL,
        1: SWITCHED["transcription_model"],
        2: FIRST_MODEL,
    }


@pytest.mark.parametrize("switch", [False, True], ids=["same_model", "switched"])
def test_log_without_page_models_keeps_resuming(
    runner: Runner, pdf: Path, switch: bool
) -> None:
    first = failed_first_run(runner, pdf)
    assert first.exit_code == 1, first.stderr
    strip_page_models(runner.output_dir / DOC.transcription_log)

    second = runner.run(pdf, **(SWITCHED if switch else {}))

    assert second.exit_code == 0, second.stderr
    assert pages(second, TRANSCRIPTION) == [FAILED_PAGE]
    resumed = SWITCHED["transcription_model"] if switch else FIRST_MODEL
    # Reused pages fall back to the model of the log header.
    assert page_models(runner.output_dir) == {
        0: FIRST_MODEL,
        1: resumed,
        2: FIRST_MODEL,
    }


def test_model_switch_with_changed_image_setting_stops(
    runner: Runner, pdf: Path
) -> None:
    first = failed_first_run(runner, pdf)
    assert first.exit_code == 1, first.stderr
    before = folder_bytes(runner.output_dir)

    second = runner.run(pdf, jpeg_quality=CHANGED_JPEG_QUALITY, **SWITCHED)

    assert second.exit_code == 1, second.stderr
    assert "Image settings changed: jpeg_quality." in second.stderr
    assert second.calls() == []
    assert folder_bytes(runner.output_dir) == before


# ============================================================================
# (8) Exit codes, each with the JSON summary as the last stdout line
# ============================================================================
def _exit_0(runner: Runner, pdf: Path) -> RunResult:
    return runner.run(pdf)


def _exit_1(runner: Runner, pdf: Path) -> RunResult:
    return failed_first_run(runner, pdf)


def _exit_2(runner: Runner, pdf: Path) -> RunResult:
    make_pdf(pdf.parent / "other.pdf", pages=1)
    return runner.run(pdf.parent, selection=None)


EXIT_SCENARIOS: dict[int, Callable[[Runner, Path], RunResult]] = {
    0: _exit_0,
    1: _exit_1,
    2: _exit_2,
}


@pytest.mark.parametrize("code", list(EXIT_SCENARIOS))
def test_exit_code_with_json_last_line(runner: Runner, pdf: Path, code: int) -> None:
    result = EXIT_SCENARIOS[code](runner, pdf)

    assert result.exit_code == code, result.stderr
    summary = assert_json_last_line(result)
    assert summary["dry_run"] is False
    runner.check(f"exit_{code}", result)
