"""Summary context: the per-item prompt context from sidecars and arguments."""

import logging
from pathlib import Path

import pytest

from autoexcerpter.common.context import resolve_context
from autoexcerpter.pipeline.context import (
    CONTEXT_SIZE_THRESHOLD,
    CONTEXT_SUFFIX,
    ContextError,
    summary_context,
)


class TestSidecarReading:
    def test_strips_whitespace(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        (tmp_path / "doc_summary_context.txt").write_text(
            "  Food History  \n", encoding="utf-8"
        )

        assert summary_context(pdf) == "Food History"

    def test_warns_for_a_large_sidecar(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        large = "x" * (CONTEXT_SIZE_THRESHOLD + 100)
        (tmp_path / "doc_summary_context.txt").write_text(large, encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            content = summary_context(pdf)

        assert content == large
        assert "large" in caplog.text.lower()

    def test_dotted_folder_finds_its_context_file(self, tmp_path: Path) -> None:
        """An image folder's sidecar is named after the full folder name."""
        folder = tmp_path / "photos.2023"
        folder.mkdir()
        ctx = tmp_path / "photos.2023_summary_context.txt"
        ctx.write_text("Context for the 2023 photo set.", encoding="utf-8")
        # The sidecar of a document named after the folder's Path.stem.
        wrong = tmp_path / "photos_summary_context.txt"
        wrong.write_text("Context for photos.pdf, not the folder.", encoding="utf-8")

        resolved = resolve_context(folder, CONTEXT_SUFFIX)

        assert resolved is not None
        assert resolved.path == ctx
        assert resolved.text == "Context for the 2023 photo set."

    def test_pdf_input_uses_its_stem(self, tmp_path: Path) -> None:
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        ctx = tmp_path / "book_summary_context.txt"
        ctx.write_text("Context for the book.", encoding="utf-8")

        resolved = resolve_context(pdf, CONTEXT_SUFFIX)

        assert resolved is not None
        assert resolved.path == ctx
        assert resolved.text == "Context for the book."


class TestSummaryContext:
    def test_text_enters_the_prompt_as_given(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        (tmp_path / "doc_summary_context.txt").write_text("Sidecar", encoding="utf-8")

        assert summary_context(pdf, "Wages\nPrices") == "Wages\nPrices"

    def test_a_given_file_is_read_and_joined(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        topics = tmp_path / "topics.txt"
        topics.write_text("Wages\nPrices\n", encoding="utf-8")

        assert summary_context(pdf, str(topics)) == "Wages, Prices"

    def test_sidecars_are_joined(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        (tmp_path / "doc_summary_context.txt").write_text(
            "Wages\n\nPrices\n", encoding="utf-8"
        )

        assert summary_context(pdf) == "Wages, Prices"

    def test_no_context(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()

        assert summary_context(pdf) is None

    def test_an_empty_given_file_is_an_error(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        topics = tmp_path / "topics.txt"
        topics.write_text("\n", encoding="utf-8")

        with pytest.raises(ContextError):
            summary_context(pdf, str(topics))

    def test_a_sidecar_beats_the_default(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        (tmp_path / "doc_summary_context.txt").write_text("Sidecar", encoding="utf-8")

        assert summary_context(pdf, default="Default topics") == "Sidecar"

    def test_the_default_applies_without_a_sidecar(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        topics = tmp_path / "topics.txt"
        topics.write_text("Wages\nPrices\n", encoding="utf-8")

        assert summary_context(pdf, default="Wages\nPrices") == "Wages\nPrices"
        assert summary_context(pdf, default=str(topics)) == "Wages, Prices"

    def test_given_context_beats_the_default(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.touch()

        assert summary_context(pdf, "Given", default="Default") == "Given"


class TestIntegrationWithPromptUtils:
    """Integration tests with prompt_utils module."""

    def test_context_injection_in_prompt(self) -> None:
        """Context should be properly injected into prompts."""
        from autoexcerpter.llm.prompts import render_prompt_with_schema

        prompt = (
            "Instructions:\n- Rule 1\n- Pay attention to: {{CONTEXT}}"
            "\n\nThe JSON schema:\n{{SCHEMA}}"
        )
        schema = {"type": "object"}
        context = "Food History, Wages"

        result = render_prompt_with_schema(prompt, schema, context=context)

        assert "Food History, Wages" in result
        assert "{{CONTEXT}}" not in result

    def test_context_line_removed_when_no_context(self) -> None:
        """Context placeholder line should be removed when no context provided."""
        from autoexcerpter.llm.prompts import render_prompt_with_schema

        prompt = (
            "Instructions:\n- Rule 1\n- Pay attention to: {{CONTEXT}}"
            "\n\nThe JSON schema:\n{{SCHEMA}}"
        )
        schema = {"type": "object"}

        result = render_prompt_with_schema(prompt, schema, context=None)

        assert "{{CONTEXT}}" not in result
        assert "Pay attention to:" not in result

    def test_context_line_removed_when_empty_context(self) -> None:
        """Context placeholder line should be removed when context is empty string."""
        from autoexcerpter.llm.prompts import render_prompt_with_schema

        prompt = (
            "Instructions:\n- Rule 1\n- Pay attention to: {{CONTEXT}}"
            "\n\nThe JSON schema:\n{{SCHEMA}}"
        )
        schema = {"type": "object"}

        result = render_prompt_with_schema(prompt, schema, context="")

        assert "{{CONTEXT}}" not in result
        assert "Pay attention to:" not in result
