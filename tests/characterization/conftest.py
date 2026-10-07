"""Fixtures for the characterization tests: fake LLM, fake OpenAlex, goldens."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

# Imported at module load: google.auth subclasses requests.Session at import,
# and the autouse fake_openalex fixture replaces requests.Session.
import langchain_anthropic
import langchain_google_genai
import langchain_openai
import pytest
import requests

from tests.characterization.adapter import RunResult
from tests.characterization.fakes import FakeLLM, FakeOpenAlex
from tests.characterization.golden import check_golden
from tests.conftest import WriteGuard

PROVIDER_CLASSES = (
    (langchain_openai, "ChatOpenAI"),
    (langchain_anthropic, "ChatAnthropic"),
    (langchain_google_genai, "ChatGoogleGenerativeAI"),
)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Replace the provider classes with ``FakeLLM`` stand-ins and return it."""
    controller = FakeLLM()
    for module, name in PROVIDER_CLASSES:
        monkeypatch.setattr(module, name, controller.constructor(name))
    return controller


@pytest.fixture(autouse=True)
def fake_openalex(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAlex:
    """Serve OpenAlex requests from fixtures instead of the network."""
    service = FakeOpenAlex()
    monkeypatch.setattr(requests, "Session", service.session)
    monkeypatch.setattr(requests, "get", service.get)
    return service


@pytest.fixture
def golden(
    fake_openalex: FakeOpenAlex, write_guard: WriteGuard
) -> Callable[[str, RunResult], None]:
    """Return ``check(case, result)`` comparing a run with its golden files."""

    def check(case: str, result: RunResult) -> None:
        check_golden(
            case,
            result,
            openalex=fake_openalex,
            on_write=lambda case_dir: write_guard.consume(case_dir.parent),
        )

    return check


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Return the input folder (outputs beside the input land here too)."""
    folder = tmp_path / "docs"
    folder.mkdir()
    return folder
