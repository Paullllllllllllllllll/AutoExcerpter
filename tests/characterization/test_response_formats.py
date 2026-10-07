"""Every response format, per provider, recorded through ``run_tool``.

Goldens live under ``golden/formats/<case>/``. ``schema`` is API-enforced
JSON, ``json`` is prompted JSON without enforcement, ``text`` is plain text
(custom endpoint Mode B). A format of None leaves the choice to the registry
model, which yields prompted JSON for a Google model with thinking
enabled and for a registry model without structured-output support. The
image-folder input covers the providers whose image blocks differ from the
OpenAI ``image_url`` block with ``detail``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pytest

from tests.characterization.adapter import (
    CUSTOM_MODEL,
    ResponseFormat,
    RunResult,
    run_tool,
)
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM, Injection
from tests.characterization.inputs import make_image_folder, make_pdf

Golden = Callable[[str, RunResult], None]
PromptVariant = Literal["standard", "plain"]

PAGES = 3
OPENAI_MODEL = "gpt-5.6-terra"
OPENAI_UNSTRUCTURED_MODEL = "gpt-5.2-pro"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GOOGLE_MODEL = "gemini-2.5-flash-lite"
GOOGLE_THINKING_MODEL = "gemini-3.5-flash"
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"

TEXT_FORMAT = "text.format"
RESPONSE_FORMAT = "response_format"
GOOGLE_SCHEMA = "response_schema"
GOOGLE_MIME = "response_mime_type"
STRUCTURED_JSON_SCHEMA = "structured:json_schema"
# The fake's default method: the caller passed no ``method`` argument.
STRUCTURED_DEFAULT = "structured:function_calling"

FENCED_PAGE = {TRANSCRIPTION: 0, SUMMARY: 1}
INVALID_THEN_VALID_PAGE = {TRANSCRIPTION: 1, SUMMARY: 2}
INVALID_REPLY = "I could not produce JSON for this page."


@dataclass(frozen=True)
class Phase:
    """Provider, model and requested response format of one phase."""

    provider: str
    model: str
    response_format: ResponseFormat | None


@dataclass(frozen=True)
class FormatCase:
    """One cell of the response-format matrix and what its requests carry."""

    case_id: str
    transcription: Phase
    summary: Phase
    images: bool
    transcription_enforcement: frozenset[str]
    summary_enforcement: frozenset[str]
    transcription_prompt: PromptVariant = "standard"
    summary_prompt: PromptVariant = "standard"
    malformed_replies: bool = False
    required_kwargs: frozenset[str] = frozenset()


def _phase(provider: str, model: str, fmt: ResponseFormat | None) -> Phase:
    return Phase(provider, model, fmt)


CASES = [
    FormatCase(
        "schema_openai_pdf",
        _phase("openai", OPENAI_MODEL, "schema"),
        _phase("openai", OPENAI_MODEL, "schema"),
        images=False,
        transcription_enforcement=frozenset({TEXT_FORMAT}),
        summary_enforcement=frozenset({TEXT_FORMAT}),
    ),
    FormatCase(
        "schema_anthropic_images",
        _phase("anthropic", ANTHROPIC_MODEL, "schema"),
        _phase("anthropic", ANTHROPIC_MODEL, "schema"),
        images=True,
        transcription_enforcement=frozenset({STRUCTURED_JSON_SCHEMA}),
        summary_enforcement=frozenset({STRUCTURED_JSON_SCHEMA}),
    ),
    FormatCase(
        "schema_google_images",
        _phase("google", GOOGLE_MODEL, "schema"),
        _phase("google", GOOGLE_MODEL, "schema"),
        images=True,
        transcription_enforcement=frozenset({GOOGLE_MIME, GOOGLE_SCHEMA}),
        summary_enforcement=frozenset({GOOGLE_MIME, GOOGLE_SCHEMA}),
    ),
    FormatCase(
        "schema_openrouter_pdf",
        _phase("openrouter", OPENROUTER_MODEL, "schema"),
        _phase("openrouter", OPENROUTER_MODEL, "schema"),
        images=False,
        transcription_enforcement=frozenset({STRUCTURED_DEFAULT}),
        summary_enforcement=frozenset({STRUCTURED_DEFAULT}),
    ),
    FormatCase(
        "schema_custom_pdf",
        _phase("custom", CUSTOM_MODEL, "schema"),
        _phase("custom", CUSTOM_MODEL, "schema"),
        images=False,
        transcription_enforcement=frozenset({RESPONSE_FORMAT}),
        summary_enforcement=frozenset({RESPONSE_FORMAT}),
    ),
    FormatCase(
        "json_custom_pdf",
        _phase("custom", CUSTOM_MODEL, "json"),
        _phase("custom", CUSTOM_MODEL, "json"),
        images=False,
        transcription_enforcement=frozenset(),
        summary_enforcement=frozenset(),
        malformed_replies=True,
    ),
    FormatCase(
        "json_google_thinking_images",
        _phase("google", GOOGLE_THINKING_MODEL, None),
        _phase("google", GOOGLE_THINKING_MODEL, None),
        images=True,
        transcription_enforcement=frozenset(),
        summary_enforcement=frozenset(),
        malformed_replies=True,
        required_kwargs=frozenset({"thinking_config"}),
    ),
    FormatCase(
        "json_openai_without_structured_output_pdf",
        _phase("openai", OPENAI_UNSTRUCTURED_MODEL, None),
        _phase("openai", OPENAI_UNSTRUCTURED_MODEL, None),
        images=False,
        transcription_enforcement=frozenset(),
        summary_enforcement=frozenset(),
        malformed_replies=True,
    ),
    FormatCase(
        "text_custom_pdf",
        _phase("custom", CUSTOM_MODEL, "text"),
        _phase("custom", CUSTOM_MODEL, "text"),
        images=False,
        transcription_enforcement=frozenset(),
        summary_enforcement=frozenset(),
        transcription_prompt="plain",
        summary_prompt="plain",
    ),
    FormatCase(
        "mixed_text_transcription_schema_summary_pdf",
        _phase("custom", CUSTOM_MODEL, "text"),
        _phase("openai", OPENAI_MODEL, "schema"),
        images=False,
        transcription_enforcement=frozenset(),
        summary_enforcement=frozenset({TEXT_FORMAT}),
        transcription_prompt="plain",
        summary_prompt="plain",
    ),
]


# ============================================================================
# Expectations
# ============================================================================
def _schema(name: str) -> dict[str, Any]:
    from autoexcerpter.llm.prompts import SCHEMAS_DIR

    data: dict[str, Any] = json.loads((SCHEMAS_DIR / name).read_text(encoding="utf-8"))
    return data


@cache
def _expected_prompt(role: str, variant: PromptVariant) -> str:
    """Return the system prompt the pipeline builds for *role* and *variant*."""
    from autoexcerpter.llm.prompts import PROMPTS_DIR, render_prompt_with_schema

    if role == TRANSCRIPTION:
        if variant == "plain":
            path = PROMPTS_DIR / "transcription_plain_text_prompt.txt"
            return path.read_text(encoding="utf-8")
        raw = (PROMPTS_DIR / "transcription_system_prompt.txt").read_text(
            encoding="utf-8"
        )
        return render_prompt_with_schema(
            raw, _schema("transcription_schema.json")["schema"]
        )
    name = (
        "summary_plain_text_prompt.txt"
        if variant == "plain"
        else ("summary_system_prompt.txt")
    )
    raw = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return render_prompt_with_schema(
        raw, _schema("summary_schema.json")["schema"], context=None
    )


def _schema_title(role: str) -> str:
    """Return the title the structured-output paths give the schema."""
    if role == TRANSCRIPTION:
        wrapper = _schema("transcription_schema.json")
        return str(wrapper["schema"].get("title", wrapper["name"]))
    wrapper = _schema("summary_schema.json")
    return str(wrapper["schema"].get("title", wrapper["name"]))


def _schema_name(role: str) -> str:
    file = (
        "transcription_schema.json"
        if role == TRANSCRIPTION
        else ("summary_schema.json")
    )
    return str(_schema(file)["name"])


def _enforcement(call: dict[str, Any]) -> frozenset[str]:
    """Return the schema-enforcement mechanisms one recorded request carries."""
    found: set[str] = set()
    kwargs = call["kwargs"]
    text = kwargs.get("text")
    if isinstance(text, dict) and "format" in text:
        found.add(TEXT_FORMAT)
    for key in (RESPONSE_FORMAT, GOOGLE_SCHEMA, GOOGLE_MIME):
        if key in kwargs:
            found.add(key)
    structured = call["structured"]
    if structured:
        found.add(f"structured:{structured['method']}")
    return frozenset(found)


def _assert_enforcement(
    result: RunResult, role: str, expected: frozenset[str], phase: Phase
) -> None:
    calls = result.calls(role)
    assert calls, f"no {role} requests"
    for call in calls:
        assert call["provider"] == phase.provider
        assert _enforcement(call) == expected, call
        if TEXT_FORMAT in expected:
            text_format = call["kwargs"]["text"]["format"]
            assert text_format["type"] == "json_schema"
            assert text_format["name"] == _schema_name(role)
            assert text_format["strict"] is True
        if RESPONSE_FORMAT in expected:
            response_format = call["kwargs"][RESPONSE_FORMAT]
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"]["name"] == _schema_name(role)
        if GOOGLE_MIME in expected:
            assert call["kwargs"][GOOGLE_MIME] == "application/json"
        structured = call["structured"]
        if structured:
            assert structured["include_raw"] is True
            assert structured["schema_title"] == _schema_title(role)


def _assert_prompts(fake_llm: FakeLLM, role: str, variant: PromptVariant) -> None:
    prompts = set(fake_llm.system_prompts_for(role))
    assert prompts == {_expected_prompt(role, variant)}
    other: PromptVariant = "standard" if variant == "plain" else "plain"
    assert _expected_prompt(role, other) not in prompts


def _assert_message_shapes(result: RunResult, case: FormatCase) -> None:
    for call in result.calls(TRANSCRIPTION):
        user = call["messages"][1]
        assert user["type"] == "human"
        (block,) = user["blocks"]
        provider = case.transcription.provider
        if provider == "anthropic":
            assert block["type"] == "image"
            assert block["source_type"] == "base64"
        else:
            assert block["type"] == "image_url"
            assert ("detail" in block) is (provider == "openai")
    for call in result.calls(SUMMARY):
        user = call["messages"][1]
        if case.summary.provider in ("anthropic", "google"):
            assert "text" in user and "blocks" not in user
        else:
            assert [block["type"] for block in user["blocks"]] == ["text"]


def _inject_malformed_replies(fake_llm: FakeLLM) -> None:
    """Fence one reply per role in markdown; make another invalid once."""
    for role, page in FENCED_PAGE.items():
        payload = json.dumps(fake_llm.page(page).json_for(role), ensure_ascii=False)
        fake_llm.inject(page, role, 1, Injection(text=f"```json\n{payload}\n```"))
    for role, page in INVALID_THEN_VALID_PAGE.items():
        fake_llm.inject(page, role, 1, Injection(text=INVALID_REPLY))


def _attempts_per_page(result: RunResult, role: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    for call in result.calls(role):
        counts[call["page"]] = counts.get(call["page"], 0) + 1
    return counts


# ============================================================================
# The matrix
# ============================================================================
@pytest.mark.parametrize("case", [pytest.param(c, id=c.case_id) for c in CASES])
def test_response_format(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden, case: FormatCase
) -> None:
    source = tmp_path / "in"
    if case.images:
        input_path = make_image_folder(source / "book", pages=PAGES)
    else:
        input_path = make_pdf(source / "doc.pdf", pages=PAGES)
    if case.malformed_replies:
        _inject_malformed_replies(fake_llm)

    result = run_tool(
        input_path,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        keep_working_files=True,
        transcription_provider=case.transcription.provider,
        transcription_model=case.transcription.model,
        transcription_format=case.transcription.response_format,
        summary_provider=case.summary.provider,
        summary_model=case.summary.model,
        summary_format=case.summary.response_format,
    )

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 1
    _assert_enforcement(
        result, TRANSCRIPTION, case.transcription_enforcement, case.transcription
    )
    _assert_enforcement(result, SUMMARY, case.summary_enforcement, case.summary)
    for call in result.calls():
        assert case.required_kwargs <= set(call["kwargs"]), call
    _assert_prompts(fake_llm, TRANSCRIPTION, case.transcription_prompt)
    _assert_prompts(fake_llm, SUMMARY, case.summary_prompt)
    _assert_message_shapes(result, case)

    transcript = result.paths(input_path.stem).transcription.read_text(encoding="utf-8")
    for page in range(PAGES):
        assert fake_llm.page(page).key in transcript
    assert "```" not in transcript
    assert INVALID_REPLY not in transcript

    transcription_attempts = _attempts_per_page(result, TRANSCRIPTION)
    summary_attempts = _attempts_per_page(result, SUMMARY)
    if case.malformed_replies:
        retried = INVALID_THEN_VALID_PAGE
        assert transcription_attempts == {
            page: 2 if page == retried[TRANSCRIPTION] else 1 for page in range(PAGES)
        }
        assert summary_attempts == {
            page: 2 if page == retried[SUMMARY] else 1 for page in range(PAGES)
        }
    elif case.summary.response_format == "text":
        # Plain-text summaries fail validation until the retries run out and
        # are then wrapped into the summary structure.
        assert transcription_attempts == dict.fromkeys(range(PAGES), 1)
        assert summary_attempts == dict.fromkeys(range(PAGES), 4)
    else:
        assert transcription_attempts == dict.fromkeys(range(PAGES), 1)
        assert summary_attempts == dict.fromkeys(range(PAGES), 1)

    golden(f"formats/{case.case_id}", result)
