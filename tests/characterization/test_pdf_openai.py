"""PDF input through an OpenAI registry model with enforced structured output."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM
from tests.characterization.inputs import make_pdf

Golden = Callable[[str, RunResult], None]


def _format_names(result: RunResult, role: str) -> set[str]:
    return {call["kwargs"]["text"]["format"]["name"] for call in result.calls(role)}


def test_pdf_openai_schema_with_summary(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    pdf = make_pdf(docs_dir / "doc.pdf", pages=3)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        summarize=True,
        docx=True,
        markdown=True,
        openalex=True,
    )

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 1
    assert [call["page"] for call in result.calls(TRANSCRIPTION)] == [0, 1, 2]
    assert [call["page"] for call in result.calls(SUMMARY)] == [0, 1, 2]
    assert _format_names(result, TRANSCRIPTION) == {"markdown_transcription_schema"}
    assert _format_names(result, SUMMARY) == {"article_page_summary"}
    golden("pdf_openai_schema_with_summary", result)


def test_pdf_openai_schema_without_summary(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    pdf = make_pdf(docs_dir / "doc.pdf", pages=3)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        summarize=False,
    )

    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 1
    assert result.calls(SUMMARY) == []
    assert _format_names(result, TRANSCRIPTION) == {"markdown_transcription_schema"}
    golden("pdf_openai_schema_without_summary", result)
