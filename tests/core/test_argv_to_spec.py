"""Hand-written command lines parsed into the run spec, as the CLI parses them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.common.options import Source
from autoexcerpter.spec import (
    OPTIONS,
    CitationSpec,
    ImageSpec,
    InputSpec,
    ModelSpec,
    OutputSpec,
    Resolution,
    RunControl,
    RunSpec,
    SummarySpec,
    resolve_spec,
)

FLAG = Source.FLAG
SETTINGS = Source.SETTINGS
DEFAULT = Source.DEFAULT

CODE_DEFAULTS = RunSpec(
    input=InputSpec(path=Path("in.pdf"), select=None, all=False),
    output=OutputSpec(
        directory=None,
        transcription_format="md",
        formats=("summary-md", "summary-docx", "sqlite"),
        keep_working_files=False,
    ),
    summary=SummarySpec(enabled=True, context=None, fallback_context=None),
    citations=CitationSpec(openalex=True),
    transcription_model=ModelSpec(
        provider="openai",
        model="gpt-5.6-luna",
        endpoint=None,
        reasoning_effort="high",
        verbosity="medium",
        max_output_tokens=None,
        temperature=1.0,
        top_p=None,
        service_tier="flex",
        response_format=None,
    ),
    summary_model=ModelSpec(
        provider="openai",
        model="gpt-5.6-luna",
        endpoint=None,
        reasoning_effort="high",
        verbosity="low",
        max_output_tokens=None,
        temperature=1.0,
        top_p=None,
        service_tier="flex",
        response_format=None,
    ),
    images=ImageSpec(
        dpi=None, detail=None, format=None, jpeg_quality=None, grayscale=None
    ),
    run=RunControl(
        force=False, retranscribe=False, concurrency=16, dry_run=False, json=False
    ),
)


def _resolve(argv: list[str], defaults: Mapping[str, Any] | None = None) -> Resolution:
    return resolve_spec(OPTIONS.parse(argv), OPTIONS.check_defaults(defaults or {}))


def _with(spec: RunSpec, changes: Mapping[str, Any]) -> RunSpec:
    for path, value in changes.items():
        part, name = path.split(".")
        spec = replace(spec, **{part: replace(getattr(spec, part), **{name: value})})
    return spec


def _both(name: str, value: Any) -> dict[str, Any]:
    return {f"transcription_model.{name}": value, f"summary_model.{name}": value}


@dataclass(frozen=True)
class Row:
    """One option on the command line and the spec fields it sets."""

    option: str
    argv: tuple[str, ...]
    sets: Mapping[str, Any]
    defaults: Mapping[str, Any] = field(default_factory=dict)


def _row(
    option: str,
    *words: str,
    sets: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
) -> Row:
    return Row(option, ("--input", "in.pdf", *words), sets, defaults or {})


ROWS = [
    Row("input", ("--input", "a b/c.pdf"), {"input.path": Path("a b/c.pdf")}),
    _row(
        "select",
        "--select",
        "1-3",
        sets={"input.select": "1-3", "input.all": False},
    ),
    _row("all", "--all", sets={"input.all": True, "input.select": None}),
    _row("output", "--output", "x y", sets={"output.directory": Path("x y")}),
    _row(
        "beside_input",
        "--beside-input",
        sets={"output.directory": None},
        defaults={"output": "out"},
    ),
    _row(
        "transcription_format",
        "--transcription-format",
        "txt",
        sets={"output.transcription_format": "txt"},
    ),
    _row(
        "formats",
        "--formats",
        "sqlite,summary-md",
        sets={"output.formats": ("summary-md", "sqlite")},
    ),
    _row(
        "keep_working_files",
        "--keep-working-files",
        sets={"output.keep_working_files": True},
    ),
    _row("summarize", "--no-summarize", sets={"summary.enabled": False}),
    _row(
        "context",
        "--context",
        "Food History, Wages",
        sets={"summary.context": "Food History, Wages"},
    ),
    _row("context", "--context", "none", sets={"summary.context": "none"}),
    _row(
        "context",
        "--context",
        "auto",
        sets={"summary.context": None},
        defaults={"context": "Topics"},
    ),
    _row(
        "fallback_context",
        "--fallback-context",
        "fallback.txt",
        sets={"summary.fallback_context": "fallback.txt"},
    ),
    _row(
        "fallback_context",
        "--fallback-context",
        "none",
        sets={"summary.fallback_context": None},
        defaults={"fallback_context": "Topics"},
    ),
    _row("openalex", "--no-openalex", sets={"citations.openalex": False}),
    _row("provider", "--provider", "anthropic", sets=_both("provider", "anthropic")),
    _row(
        "model",
        "--model",
        "gpt-5.6-terra",
        sets={**_both("model", "gpt-5.6-terra"), **_both("provider", "openai")},
    ),
    _row(
        "model",
        "--model",
        "claude-sonnet-5",
        sets={**_both("model", "claude-sonnet-5"), **_both("provider", "anthropic")},
    ),
    _row(
        "model",
        "--model",
        "openrouter:anthropic/claude-sonnet-5",
        sets={
            **_both("model", "anthropic/claude-sonnet-5"),
            **_both("provider", "openrouter"),
        },
    ),
    _row(
        "endpoint",
        "--endpoint",
        "local",
        sets={**_both("endpoint", "local"), **_both("provider", "custom")},
    ),
    _row(
        "reasoning_effort",
        "--reasoning-effort",
        "low",
        sets=_both("reasoning_effort", "low"),
    ),
    _row("verbosity", "--verbosity", "high", sets=_both("verbosity", "high")),
    _row(
        "max_output_tokens",
        "--max-output-tokens",
        "8000",
        sets=_both("max_output_tokens", 8000),
    ),
    _row("temperature", "--temperature", "0.4", sets=_both("temperature", 0.4)),
    _row("top_p", "--top-p", "0.9", sets=_both("top_p", 0.9)),
    _row(
        "service_tier",
        "--service-tier",
        "priority",
        sets=_both("service_tier", "priority"),
    ),
    _row(
        "response_format",
        "--response-format",
        "json",
        sets=_both("response_format", "json"),
    ),
    _row(
        "transcription_provider",
        "--transcription-provider",
        "google",
        sets={"transcription_model.provider": "google"},
    ),
    _row(
        "transcription_model",
        "--transcription-model",
        "gemini-3.1-pro",
        sets={
            "transcription_model.model": "gemini-3.1-pro",
            "transcription_model.provider": "google",
        },
    ),
    _row(
        "transcription_endpoint",
        "--transcription-endpoint",
        "local",
        sets={
            "transcription_model.endpoint": "local",
            "transcription_model.provider": "custom",
        },
    ),
    _row(
        "transcription_reasoning_effort",
        "--transcription-reasoning-effort",
        "xhigh",
        sets={"transcription_model.reasoning_effort": "xhigh"},
    ),
    _row(
        "transcription_verbosity",
        "--transcription-verbosity",
        "low",
        sets={"transcription_model.verbosity": "low"},
    ),
    _row(
        "transcription_max_output_tokens",
        "--transcription-max-output-tokens",
        "4096",
        sets={"transcription_model.max_output_tokens": 4096},
    ),
    _row(
        "transcription_temperature",
        "--transcription-temperature",
        "0.5",
        sets={"transcription_model.temperature": 0.5},
    ),
    _row(
        "transcription_top_p",
        "--transcription-top-p",
        "0.5",
        sets={"transcription_model.top_p": 0.5},
    ),
    _row(
        "transcription_service_tier",
        "--transcription-service-tier",
        "default",
        sets={"transcription_model.service_tier": "default"},
    ),
    _row(
        "transcription_response_format",
        "--transcription-response-format",
        "text",
        sets={"transcription_model.response_format": "text"},
    ),
    _row(
        "summary_provider",
        "--summary-provider",
        "openrouter",
        sets={"summary_model.provider": "openrouter"},
    ),
    _row(
        "summary_model",
        "--summary-model",
        "gpt-5.5",
        sets={"summary_model.model": "gpt-5.5", "summary_model.provider": "openai"},
    ),
    _row(
        "summary_endpoint",
        "--summary-endpoint",
        "remote",
        sets={"summary_model.endpoint": "remote", "summary_model.provider": "custom"},
    ),
    _row(
        "summary_reasoning_effort",
        "--summary-reasoning-effort",
        "none",
        sets={"summary_model.reasoning_effort": "none"},
    ),
    _row(
        "summary_verbosity",
        "--summary-verbosity",
        "medium",
        sets={"summary_model.verbosity": "medium"},
    ),
    _row(
        "summary_max_output_tokens",
        "--summary-max-output-tokens",
        "1000",
        sets={"summary_model.max_output_tokens": 1000},
    ),
    _row(
        "summary_temperature",
        "--summary-temperature",
        "2.0",
        sets={"summary_model.temperature": 2.0},
    ),
    _row(
        "summary_top_p",
        "--summary-top-p",
        "1.0",
        sets={"summary_model.top_p": 1.0},
    ),
    _row(
        "summary_service_tier",
        "--summary-service-tier",
        "auto",
        sets={"summary_model.service_tier": "auto"},
    ),
    _row(
        "summary_response_format",
        "--summary-response-format",
        "schema",
        sets={"summary_model.response_format": "schema"},
    ),
    _row("dpi", "--dpi", "native", sets={"images.dpi": "native"}),
    _row("dpi", "--dpi", "300", sets={"images.dpi": 300}),
    _row("image_detail", "--image-detail", "high", sets={"images.detail": "high"}),
    _row("image_format", "--image-format", "png", sets={"images.format": "png"}),
    _row("jpeg_quality", "--jpeg-quality", "80", sets={"images.jpeg_quality": 80}),
    _row("grayscale", "--no-grayscale", sets={"images.grayscale": False}),
    _row("grayscale", "--grayscale", sets={"images.grayscale": True}),
    _row("force", "--force", sets={"run.force": True}),
    _row("retranscribe", "--retranscribe", sets={"run.retranscribe": True}),
    _row("concurrency", "--concurrency", "4", sets={"run.concurrency": 4}),
    _row("dry_run", "--dry-run", sets={"run.dry_run": True}),
    _row("json", "--json", sets={"run.json": True}),
    _row("settings", "--settings", "other.yaml", sets={}),
]


def test_every_option_has_a_row() -> None:
    assert {row.option for row in ROWS} == set(OPTIONS.names)


@pytest.mark.parametrize(
    "row", ROWS, ids=[f"{row.option}:{' '.join(row.argv[2:])}" for row in ROWS]
)
def test_one_option_sets_its_fields(row: Row) -> None:
    argv = list(row.argv)
    given = {"input"} if row.option == "input" else {"input", row.option}
    assert set(OPTIONS.parse(argv)) == given

    resolution = _resolve(argv, row.defaults)

    base = replace(
        CODE_DEFAULTS, input=replace(CODE_DEFAULTS.input, path=Path(argv[1]))
    )
    assert resolution.spec == _with(base, row.sets)
    expected_sources = {
        path: FLAG if path in row.sets or path == "input.path" else DEFAULT
        for path in resolution.sources
    }
    assert dict(resolution.sources) == expected_sources


def test_the_settings_option_sets_the_settings_path() -> None:
    flags = OPTIONS.parse(["--input", "in.pdf", "--settings", "other.yaml"])
    assert flags["settings"] == Path("other.yaml")


COMMAND_LINES: list[tuple[list[str], RunSpec]] = [
    (["--input", "in.pdf"], CODE_DEFAULTS),
    (
        [
            "--input",
            "scans",
            "--all",
            "--model",
            "gpt-5.6-terra",
            "--summary-model",
            "claude-sonnet-5",
            "--formats",
            "summary-md,sqlite",
            "--context",
            "Food History, Wages",
            "--dry-run",
            "--json",
        ],
        RunSpec(
            input=InputSpec(path=Path("scans"), select=None, all=True),
            output=OutputSpec(
                directory=None,
                transcription_format="md",
                formats=("summary-md", "sqlite"),
                keep_working_files=False,
            ),
            summary=SummarySpec(
                enabled=True, context="Food History, Wages", fallback_context=None
            ),
            citations=CitationSpec(openalex=True),
            transcription_model=ModelSpec(
                provider="openai",
                model="gpt-5.6-terra",
                endpoint=None,
                reasoning_effort="high",
                verbosity="medium",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="flex",
                response_format=None,
            ),
            summary_model=ModelSpec(
                provider="anthropic",
                model="claude-sonnet-5",
                endpoint=None,
                reasoning_effort="high",
                verbosity="low",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="flex",
                response_format=None,
            ),
            images=ImageSpec(
                dpi=None, detail=None, format=None, jpeg_quality=None, grayscale=None
            ),
            run=RunControl(
                force=False, retranscribe=False, concurrency=16, dry_run=True, json=True
            ),
        ),
    ),
    (
        [
            "--input",
            "a b/c.pdf",
            "--select",
            "1-3",
            "--output",
            "out",
            "--transcription-model",
            "gpt-5.6-sol",
            "--transcription-reasoning-effort",
            "xhigh",
            "--summary-model",
            "gpt-5.6-luna",
            "--summary-verbosity",
            "medium",
            "--fallback-context",
            "fallback.txt",
            "--force",
            "--retranscribe",
            "--concurrency",
            "4",
        ],
        RunSpec(
            input=InputSpec(path=Path("a b/c.pdf"), select="1-3", all=False),
            output=OutputSpec(
                directory=Path("out"),
                transcription_format="md",
                formats=("summary-md", "summary-docx", "sqlite"),
                keep_working_files=False,
            ),
            summary=SummarySpec(
                enabled=True, context=None, fallback_context="fallback.txt"
            ),
            citations=CitationSpec(openalex=True),
            transcription_model=ModelSpec(
                provider="openai",
                model="gpt-5.6-sol",
                endpoint=None,
                reasoning_effort="xhigh",
                verbosity="medium",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="flex",
                response_format=None,
            ),
            summary_model=ModelSpec(
                provider="openai",
                model="gpt-5.6-luna",
                endpoint=None,
                reasoning_effort="high",
                verbosity="medium",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="flex",
                response_format=None,
            ),
            images=ImageSpec(
                dpi=None, detail=None, format=None, jpeg_quality=None, grayscale=None
            ),
            run=RunControl(
                force=True, retranscribe=True, concurrency=4, dry_run=False, json=False
            ),
        ),
    ),
    (
        [
            "--input",
            "scans",
            "--select",
            "Mennell",
            "--provider",
            "google",
            "--model",
            "gemini-3.1-pro",
            "--reasoning-effort",
            "low",
            "--response-format",
            "json",
            "--context",
            "none",
            "--no-openalex",
            "--dpi",
            "300",
            "--image-detail",
            "high",
            "--json",
        ],
        RunSpec(
            input=InputSpec(path=Path("scans"), select="Mennell", all=False),
            output=OutputSpec(
                directory=None,
                transcription_format="md",
                formats=("summary-md", "summary-docx", "sqlite"),
                keep_working_files=False,
            ),
            summary=SummarySpec(enabled=True, context="none", fallback_context=None),
            citations=CitationSpec(openalex=False),
            transcription_model=ModelSpec(
                provider="google",
                model="gemini-3.1-pro",
                endpoint=None,
                reasoning_effort="low",
                verbosity="medium",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="flex",
                response_format="json",
            ),
            summary_model=ModelSpec(
                provider="google",
                model="gemini-3.1-pro",
                endpoint=None,
                reasoning_effort="low",
                verbosity="low",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="flex",
                response_format="json",
            ),
            images=ImageSpec(
                dpi=300, detail="high", format=None, jpeg_quality=None, grayscale=None
            ),
            run=RunControl(
                force=False,
                retranscribe=False,
                concurrency=16,
                dry_run=False,
                json=True,
            ),
        ),
    ),
    (
        [
            "--input",
            "scans",
            "--all",
            "--endpoint",
            "local",
            "--model",
            "qwen-vl",
            "--transcription-max-output-tokens",
            "8000",
            "--summary-temperature",
            "0.3",
            "--top-p",
            "0.9",
            "--transcription-format",
            "txt",
            "--formats",
            "summary-docx",
            "--no-summarize",
            "--image-format",
            "png",
            "--grayscale",
            "--force",
            "--dry-run",
            "--json",
        ],
        RunSpec(
            input=InputSpec(path=Path("scans"), select=None, all=True),
            output=OutputSpec(
                directory=None,
                transcription_format="txt",
                formats=("summary-docx",),
                keep_working_files=False,
            ),
            summary=SummarySpec(enabled=False, context=None, fallback_context=None),
            citations=CitationSpec(openalex=True),
            transcription_model=ModelSpec(
                provider="custom",
                model="qwen-vl",
                endpoint="local",
                reasoning_effort="high",
                verbosity="medium",
                max_output_tokens=8000,
                temperature=1.0,
                top_p=0.9,
                service_tier="flex",
                response_format=None,
            ),
            summary_model=ModelSpec(
                provider="custom",
                model="qwen-vl",
                endpoint="local",
                reasoning_effort="high",
                verbosity="low",
                max_output_tokens=None,
                temperature=0.3,
                top_p=0.9,
                service_tier="flex",
                response_format=None,
            ),
            images=ImageSpec(
                dpi=None, detail=None, format="png", jpeg_quality=None, grayscale=True
            ),
            run=RunControl(
                force=True, retranscribe=False, concurrency=16, dry_run=True, json=True
            ),
        ),
    ),
    (
        [
            "--input",
            "in.pdf",
            "--model",
            "openrouter:anthropic/claude-sonnet-5",
            "--summary-provider",
            "openai",
            "--summary-model",
            "gpt-5.5",
            "--service-tier",
            "priority",
            "--beside-input",
            "--keep-working-files",
            "--context",
            "auto",
            "--fallback-context",
            "Prices, Wages",
            "--retranscribe",
            "--jpeg-quality",
            "70",
        ],
        RunSpec(
            input=InputSpec(path=Path("in.pdf"), select=None, all=False),
            output=OutputSpec(
                directory=None,
                transcription_format="md",
                formats=("summary-md", "summary-docx", "sqlite"),
                keep_working_files=True,
            ),
            summary=SummarySpec(
                enabled=True, context=None, fallback_context="Prices, Wages"
            ),
            citations=CitationSpec(openalex=True),
            transcription_model=ModelSpec(
                provider="openrouter",
                model="anthropic/claude-sonnet-5",
                endpoint=None,
                reasoning_effort="high",
                verbosity="medium",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="priority",
                response_format=None,
            ),
            summary_model=ModelSpec(
                provider="openai",
                model="gpt-5.5",
                endpoint=None,
                reasoning_effort="high",
                verbosity="low",
                max_output_tokens=None,
                temperature=1.0,
                top_p=None,
                service_tier="priority",
                response_format=None,
            ),
            images=ImageSpec(
                dpi=None, detail=None, format=None, jpeg_quality=70, grayscale=None
            ),
            run=RunControl(
                force=False,
                retranscribe=True,
                concurrency=16,
                dry_run=False,
                json=False,
            ),
        ),
    ),
]


@pytest.mark.parametrize(
    ("argv", "expected"), COMMAND_LINES, ids=[" ".join(c[0]) for c in COMMAND_LINES]
)
def test_a_full_command_line_gives_the_written_spec(
    argv: list[str], expected: RunSpec
) -> None:
    assert _resolve(argv).spec == expected


PRECEDENCE: list[tuple[list[str], dict[str, Any], str, Any, Source]] = [
    ([], {}, "run.concurrency", 16, DEFAULT),
    ([], {"concurrency": 4}, "run.concurrency", 4, SETTINGS),
    (["--concurrency", "8"], {"concurrency": 4}, "run.concurrency", 8, FLAG),
    ([], {"verbosity": "high"}, "summary_model.verbosity", "high", SETTINGS),
    (
        ["--summary-verbosity", "medium"],
        {"verbosity": "high"},
        "summary_model.verbosity",
        "medium",
        FLAG,
    ),
    (
        ["--summary-verbosity", "medium"],
        {"verbosity": "high"},
        "transcription_model.verbosity",
        "high",
        SETTINGS,
    ),
    (
        ["--verbosity", "medium"],
        {"summary_verbosity": "high"},
        "summary_model.verbosity",
        "medium",
        FLAG,
    ),
    ([], {"context": "Topics"}, "summary.context", "Topics", SETTINGS),
    (["--context", "auto"], {"context": "Topics"}, "summary.context", None, FLAG),
    ([], {"output": "out"}, "output.directory", Path("out"), SETTINGS),
    (["--beside-input"], {"output": "out"}, "output.directory", None, FLAG),
    ([], {"all": True}, "input.all", True, SETTINGS),
    (["--select", "2"], {"all": True}, "input.select", "2", FLAG),
    (["--select", "2"], {"all": True}, "input.all", False, FLAG),
    ([], {"endpoint": "local"}, "transcription_model.provider", "custom", SETTINGS),
    (
        ["--provider", "anthropic"],
        {"endpoint": "local"},
        "transcription_model.provider",
        "anthropic",
        FLAG,
    ),
    (
        ["--provider", "anthropic"],
        {"endpoint": "local"},
        "transcription_model.endpoint",
        None,
        FLAG,
    ),
    ([], {"model": "claude-sonnet-5"}, "summary_model.provider", "anthropic", SETTINGS),
    (
        ["--model", "gpt-5.6-terra"],
        {"provider": "anthropic"},
        "summary_model.provider",
        "openai",
        FLAG,
    ),
    ([], {"formats": ["summary-md"]}, "output.formats", ("summary-md",), SETTINGS),
    ([], {"dpi": 300}, "images.dpi", 300, SETTINGS),
    (["--dpi", "native"], {"dpi": 300}, "images.dpi", "native", FLAG),
    ([], {"summarize": False}, "summary.enabled", False, SETTINGS),
    (["--summarize"], {"summarize": False}, "summary.enabled", True, FLAG),
]


@pytest.mark.parametrize(("argv", "defaults", "path", "value", "source"), PRECEDENCE)
def test_precedence_of_flag_settings_default_and_code_default(
    argv: list[str],
    defaults: dict[str, Any],
    path: str,
    value: Any,
    source: Source,
) -> None:
    resolution = _resolve(["--input", "in.pdf", *argv], defaults)
    part, name = path.split(".")
    assert getattr(getattr(resolution.spec, part), name) == value
    assert resolution.sources[path] is source
