"""The real client factory in ``llm.client``, per provider, through ``run_tool``.

The autouse ``fake_llm`` fixture replaces the provider classes the factory
imports (``ChatOpenAI``, ``ChatAnthropic``, ``ChatGoogleGenerativeAI``;
OpenRouter and custom endpoints build ``ChatOpenAI``) with stand-ins that
record the constructor kwargs and return the fake chat model, so no provider
client is built and the run completes. Goldens hold the kwargs with each API
key replaced by the name of the environment variable it came from, under
``golden/client_factory/<case>/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.characterization.adapter import CUSTOM_BASE_URL, CUSTOM_MODEL, run_tool
from tests.characterization.fakes import (
    CUSTOM_KEY_ENV,
    DUMMY_KEYS,
    OPENROUTER_BASE_URL,
    ROLE_ORDER,
    FakeLLM,
)
from tests.characterization.golden import GOLDEN_DIR, check_files
from tests.characterization.inputs import make_pdf
from tests.conftest import WriteGuard

PAGES = 2
API_TIMEOUT = 900
PER_PHASE_TIMEOUT = {"connect": 10.0, "read": 900.0, "write": 30.0, "pool": 30.0}


@dataclass(frozen=True)
class FactoryCase:
    """One provider and the constructor call the factory makes for it."""

    provider: str
    model: str
    constructor: str
    key_param: str
    key_env: str
    base_url: str | None
    responses_api: bool
    per_phase_timeout: bool


CASES = [
    FactoryCase(
        "openai",
        "gpt-5.6-terra",
        "ChatOpenAI",
        "api_key",
        "OPENAI_API_KEY",
        base_url=None,
        responses_api=True,
        per_phase_timeout=True,
    ),
    FactoryCase(
        "anthropic",
        "claude-sonnet-4-5",
        "ChatAnthropic",
        "api_key",
        "ANTHROPIC_API_KEY",
        base_url=None,
        responses_api=False,
        per_phase_timeout=False,
    ),
    FactoryCase(
        "google",
        "gemini-2.5-flash-lite",
        "ChatGoogleGenerativeAI",
        "google_api_key",
        "GOOGLE_API_KEY",
        base_url=None,
        responses_api=False,
        per_phase_timeout=False,
    ),
    FactoryCase(
        "openrouter",
        "anthropic/claude-sonnet-4.5",
        "ChatOpenAI",
        "api_key",
        "OPENROUTER_API_KEY",
        base_url=OPENROUTER_BASE_URL,
        responses_api=False,
        per_phase_timeout=True,
    ),
    FactoryCase(
        "custom",
        CUSTOM_MODEL,
        "ChatOpenAI",
        "api_key",
        CUSTOM_KEY_ENV,
        base_url=CUSTOM_BASE_URL,
        responses_api=False,
        per_phase_timeout=True,
    ),
]


# ============================================================================
# Golden
# ============================================================================
CheckConstructors = Callable[[str, FakeLLM], None]


def _golden_files(llm: FakeLLM) -> dict[str, str]:
    records = [c.record() for c in llm.constructions_by_role()]
    text = json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False)
    return {"constructors.json": text + "\n"}


@pytest.fixture
def check_constructors(write_guard: WriteGuard) -> CheckConstructors:
    """Return ``check(case, llm)`` comparing constructions with the golden."""

    def check(case: str, llm: FakeLLM) -> None:
        check_files(
            f"client_factory/{case}",
            _golden_files(llm),
            on_write=lambda _dir: write_guard.consume(GOLDEN_DIR / "client_factory"),
        )

    return check


# ============================================================================
# Cases
# ============================================================================
def _assert_timeout(case: FactoryCase, timeout: Any) -> None:
    if case.per_phase_timeout:
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.as_dict() == PER_PHASE_TIMEOUT
    else:
        assert not isinstance(timeout, httpx.Timeout)
        assert timeout == API_TIMEOUT


@pytest.mark.parametrize("case", [pytest.param(c, id=c.provider) for c in CASES])
def test_client_factory(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    check_constructors: CheckConstructors,
    case: FactoryCase,
) -> None:
    pdf = make_pdf(docs_dir / "doc.pdf", pages=PAGES)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        transcription_provider=case.provider,
        transcription_model=case.model,
        summary_provider=case.provider,
        summary_model=case.model,
    )

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 1
    constructions = fake_llm.constructions_by_role()
    assert [c.role for c in constructions] == list(ROLE_ORDER)
    for construction in constructions:
        kwargs = construction.kwargs
        assert construction.constructor == case.constructor
        assert construction.provider == case.provider
        assert kwargs["model"] == case.model
        assert kwargs[case.key_param] == DUMMY_KEYS[case.key_env]
        assert kwargs.get("base_url") == case.base_url
        assert kwargs.get("use_responses_api") is (True if case.responses_api else None)
        assert "service_tier" not in kwargs
        assert kwargs["max_retries"] == 0
        _assert_timeout(case, kwargs["timeout"])
    assert {call["provider"] for call in result.calls()} == {case.provider}
    check_constructors(case.provider, fake_llm)
