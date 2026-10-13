"""Planning a ``--batch`` run: providers, the extension's consent, the keys."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter import ext
from autoexcerpter import run as core
from autoexcerpter.settings import Settings, SettingsError
from autoexcerpter.spec import resolve_spec


def _plan(pdf: Path, **flags: Any) -> core.RunPlan:
    spec = resolve_spec({"input": pdf, "batch": True, **flags}).spec
    return core.plan(spec, Settings())


@pytest.fixture
def pdf(make_pdf: Callable[..., Path]) -> Path:
    return make_pdf("book.pdf", num_pages=1)


def test_a_batch_plan_needs_each_phase_key(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _plan(pdf).batch

    with pytest.raises(core.ConfigurationError, match="ANTHROPIC_API_KEY") as caught:
        _plan(pdf, summary_model="claude-sonnet-4-5")
    assert caught.value.exit_code == 2
    assert "summary batches" in str(caught.value)


def test_the_extension_can_refuse_batch_submission(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    def refuse(*args: Any, **kwargs: Any) -> bool:
        raise SettingsError("batch is not available under this access")

    monkeypatch.setattr(ext, "batch_allowed", refuse)
    with pytest.raises(core.ConfigurationError, match="not available under"):
        _plan(pdf)

    monkeypatch.setattr(ext, "batch_allowed", lambda *args, **kwargs: False)
    with pytest.raises(core.ConfigurationError, match="not available for this run"):
        _plan(pdf)


def test_only_the_summarized_phases_need_a_batch_api(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    with pytest.raises(core.ConfigurationError, match="the summary model"):
        _plan(pdf, summary_model="anthropic/claude-sonnet-4.5")
    assert _plan(pdf, summary_model="anthropic/claude-sonnet-4.5", summarize=False)


def test_a_direct_plan_asks_nothing_of_the_batch_seams(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("a direct run asked for batch consent")

    monkeypatch.setattr(ext, "batch_allowed", unexpected)
    spec = resolve_spec({"input": pdf, "model": "anthropic/claude-sonnet-4.5"}).spec
    assert not core.plan(spec, Settings()).batch
