"""The json and text response formats on every registry provider.

Each case keeps its provider and requests the format with
``--<phase>-response-format`` (no custom endpoint). ``json`` sends the
schema prompt without enforcement and retries invalid answers; ``text``
sends the plain-text prompts, takes the transcription as returned and wraps
summaries that never parse. Goldens live under ``golden/formats/<case>/``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM
from tests.characterization.inputs import make_image_folder, make_pdf
from tests.characterization.test_response_formats import (
    ANTHROPIC_MODEL,
    GOOGLE_MODEL,
    INVALID_REPLY,
    INVALID_THEN_VALID_PAGE,
    OPENAI_MODEL,
    OPENROUTER_MODEL,
    PAGES,
    FormatCase,
    Phase,
    _assert_enforcement,
    _assert_message_shapes,
    _assert_prompts,
    _attempts_per_page,
    _inject_malformed_replies,
)

Golden = Callable[[str, RunResult], None]

PROVIDERS = (
    ("openai", OPENAI_MODEL, False),
    ("anthropic", ANTHROPIC_MODEL, True),
    ("google", GOOGLE_MODEL, True),
    ("openrouter", OPENROUTER_MODEL, False),
)


def _case(fmt: str, provider: str, model: str, images: bool) -> FormatCase:
    phase = Phase(provider, model, fmt)  # type: ignore[arg-type]
    plain = fmt == "text"
    return FormatCase(
        f"{fmt}_{provider}_{'images' if images else 'pdf'}",
        phase,
        phase,
        images=images,
        transcription_enforcement=frozenset(),
        summary_enforcement=frozenset(),
        transcription_prompt="plain" if plain else "standard",
        summary_prompt="plain" if plain else "standard",
        malformed_replies=not plain,
    )


CASES = [
    _case(fmt, provider, model, images)
    for fmt in ("json", "text")
    for provider, model, images in PROVIDERS
]


@pytest.mark.parametrize("case", [pytest.param(c, id=c.case_id) for c in CASES])
def test_response_format_on_a_registry_provider(
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
        reroute_formats=False,
    )

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 1
    _assert_enforcement(result, TRANSCRIPTION, frozenset(), case.transcription)
    _assert_enforcement(result, SUMMARY, frozenset(), case.summary)
    _assert_prompts(fake_llm, TRANSCRIPTION, case.transcription_prompt)
    _assert_prompts(fake_llm, SUMMARY, case.summary_prompt)
    _assert_message_shapes(result, case)
    for construction in fake_llm.constructions:
        assert construction.provider == case.transcription.provider

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
    else:
        assert transcription_attempts == dict.fromkeys(range(PAGES), 1)
        assert summary_attempts == dict.fromkeys(range(PAGES), 4)

    golden(f"formats/{case.case_id}", result)
