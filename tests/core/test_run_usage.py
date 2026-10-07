"""Model calls of a whole run: attempt contexts and per-role token totals."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import autoexcerpter.run as core
from autoexcerpter.common.testing import FakeRequest
from autoexcerpter.common.usage import Usage
from autoexcerpter.settings import RetrySettings
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import (
    SUMMARY_ANSWER,
    TRANSCRIPTION_ANSWER,
    RecordingExt,
    StatusError,
)

MakePdf = Callable[..., Path]
RECOVERED = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
ANSWER = Usage(input_tokens=100, output_tokens=20, total_tokens=120)


def _is_transcription(request: FakeRequest) -> bool:
    content = request.messages[-1].content
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in content
    )


def _responder() -> Callable[[FakeRequest], Any]:
    failed: list[bool] = []

    def answer(request: FakeRequest) -> Any:
        if not _is_transcription(request):
            return SUMMARY_ANSWER
        if not failed:
            failed.append(True)
            return StatusError(408, RECOVERED)
        return TRANSCRIPTION_ANSWER

    return answer


@pytest.fixture
def scripted_run(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    mock_image_processing: dict[str, Any],
) -> tuple[core.RunResult, RecordingExt]:
    """Run two pages with summaries; the first transcription attempt fails."""
    recorder = RecordingExt().install(monkeypatch)
    helpers.install(monkeypatch, [_responder()])
    retry = RetrySettings(backoff_base=0.0, jitter_min=0.0, jitter_max=0.0)
    settings = make_settings(
        retry=retry, rate_limits=((100_000, 1),), state_dir=tmp_path / "state"
    )
    spec = make_spec(
        make_pdf("doc.pdf", num_pages=2),
        output=tmp_path / "out",
        openalex=False,
        formats="summary-md",
    )
    result = asyncio.run(core.run(core.plan(spec, settings)))
    return result, recorder


def test_every_model_call_runs_in_an_attempt_context(
    scripted_run: tuple[core.RunResult, RecordingExt],
) -> None:
    result, recorder = scripted_run

    assert result.complete == 1
    assert sorted(recorder.roles) == ["summary"] * 2 + ["transcription"] * 3
    assert all(attempt.released for attempt in recorder.attempts)
    assert [type(error) for error in recorder.errors] == [StatusError]
    assert recorder.commits == [ANSWER] * 4
    failed = next(a for a in recorder.attempts if a.errors)
    assert failed.identity.role == "transcription"
    assert failed.recovered == Usage(8, 2, 10)


def test_the_run_result_holds_token_totals_per_role(
    scripted_run: tuple[core.RunResult, RecordingExt],
) -> None:
    result, _ = scripted_run

    assert result.usage == {
        "transcription": ANSWER + ANSWER + Usage(8, 2, 10),
        "summary": ANSWER + ANSWER,
    }
    assert result.total_usage.total_tokens == 490
