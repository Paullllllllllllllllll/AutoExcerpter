"""The progress file the launcher names, written by a real run and a wait."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.batch import check as check_module
from autoexcerpter.common.heartbeat import ENV_VAR
from tests.characterization.adapter import run_tool
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_batch import FakeBatchService
from tests.characterization.inputs import make_pdf
from tests.characterization.test_batch import Case
from tests.characterization.test_batch_wait import WaitClock


def read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_a_run_leaves_its_last_item_finished_in_the_file(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "job.heartbeat.json"
    monkeypatch.setenv(ENV_VAR, str(path))
    pdf = make_pdf(docs_dir / "doc.pdf", pages=2)

    result = run_tool(pdf, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out")

    assert result.exit_code == 0, result.stderr
    data = read(path)
    assert data["unit_label"] == "pages processed, item 1/1"
    assert (data["units_done"], data["units_total"], data["eta_seconds"]) == (2, 2, 0)


def test_a_wait_writes_the_batches_before_it_sleeps(
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = WaitClock()
    monkeypatch.setattr(check_module, "_clock", clock)
    monkeypatch.setattr(check_module, "_sleep", clock.sleep)
    run = Case(tmp_path, fake_llm, fake_batch, "openai")
    fake_batch.polls_until_complete = 2
    run.submit()
    path = tmp_path / "job.heartbeat.json"
    monkeypatch.setenv(ENV_VAR, str(path))

    waited = run.check("wait", "--wait")

    assert waited.result.exit_code == 0, waited.result.stderr
    data = read(path)
    assert (data["unit_label"], data["eta_seconds"]) == ("batches", None)
    # The last sleep came after the summary batch was submitted: the
    # transcription batch had finished, the summary batch had not.
    assert (data["units_done"], data["units_total"]) == (1, 2)
