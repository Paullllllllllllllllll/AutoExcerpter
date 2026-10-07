"""Conditional options: warnings when a given option does not apply."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter import __main__ as entry
from autoexcerpter.cli.command import condition_warnings
from tests.conftest import make_spec

MakePdf = Callable[..., Path]


def _dry_run(pdf: Path, *extra: str) -> list[str]:
    return ["run", "--input", str(pdf), *extra, "--dry-run", "--json"]


def _warnings(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("[WARNING]")]


def test_service_tier_with_another_provider_warns_and_runs(
    make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_pdf("doc.pdf")
    argv = _dry_run(pdf, "--provider", "anthropic", "--service-tier", "priority")

    assert entry.main(argv) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    json.loads(lines[0])
    (warning,) = _warnings(captured.err)
    assert "--service-tier" in warning
    assert "only for provider openai" in warning
    assert "anthropic" in warning


def test_without_the_conditional_flag_nothing_warns(
    make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_pdf("doc.pdf")

    assert entry.main(_dry_run(pdf, "--provider", "anthropic")) == 0
    assert _warnings(capsys.readouterr().err) == []


def test_unprefixed_flag_warns_once_and_names_each_phase() -> None:
    flags = {"provider": "anthropic", "service_tier": "priority"}
    spec = make_spec(**flags)

    (warning,) = condition_warnings(flags, {}, spec)

    assert warning.startswith("--service-tier ")
    assert "transcription provider is anthropic" in warning
    assert "summary provider is anthropic" in warning


def test_unprefixed_flag_names_only_the_failing_phase() -> None:
    flags = {"summary_provider": "google", "service_tier": "priority"}
    spec = make_spec(**flags)

    (warning,) = condition_warnings(flags, {}, spec)

    assert "summary provider is google" in warning
    assert "transcription provider" not in warning


def test_phase_flag_names_the_phase_flag() -> None:
    flags: dict[str, Any] = {
        "summary_provider": "anthropic",
        "summary_service_tier": "priority",
    }

    (warning,) = condition_warnings(flags, {}, make_spec(**flags))

    assert warning.startswith("--summary-service-tier ")
    assert "only for summary-provider openai" in warning


def test_openai_provider_does_not_warn() -> None:
    flags = {"service_tier": "priority"}
    assert condition_warnings(flags, {}, make_spec(**flags)) == []


def test_code_defaults_never_warn() -> None:
    flags = {"provider": "anthropic"}
    assert condition_warnings(flags, {}, make_spec(**flags)) == []


def test_settings_default_warns_with_its_name() -> None:
    flags = {"provider": "anthropic"}
    defaults = {"service_tier": "priority"}

    (warning,) = condition_warnings(flags, defaults, make_spec(**flags))

    assert warning.startswith("the settings default service_tier (--service-tier)")


def test_settings_default_warns_through_the_cli(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("defaults:\n  service_tier: priority\n", encoding="utf-8")
    pdf = make_pdf("doc.pdf")
    argv = _dry_run(pdf, "--provider", "anthropic", "--settings", str(settings))

    assert entry.main(argv) == 0

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    (warning,) = _warnings(captured.err)
    assert "settings default service_tier" in warning


def test_flag_wins_over_a_settings_default_in_the_warning() -> None:
    flags = {"provider": "anthropic", "service_tier": "auto"}
    defaults = {"service_tier": "priority"}

    (warning,) = condition_warnings(flags, defaults, make_spec(**flags))

    assert warning.startswith("--service-tier ")
