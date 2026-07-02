"""Regression tests for the Tier-1 hardening fixes (production composition).

Each test exercises the actual pipeline composition and fails on the pre-fix
code:

- B1: a resumed item's rendered summary output contains pre-crash summaries.
- B2: the summary-only (TRANSCRIPTION_ONLY) resume path makes zero transcription
  API calls and regenerates summaries from the logged transcriptions.
- B3: outputs that exist but whose log shows missing pages are NOT marked
  COMPLETE (completeness contract).
- B4: both writers receive the one shared, enriched CitationManager.
- Citation dedup policy: fold-merge; volume/year differences never merge.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import pipeline.transcriber as transcriber_module
from config import app as app_config
from config.constants import LOG_FORMAT_VERSION
from pipeline.resume import ProcessingState, ResumeChecker
from pipeline.transcriber import ItemTranscriber
from rendering.citations import CitationManager


def _fake_transcribe(payload: Any, max_schema_retries: int = 3) -> dict[str, Any]:
    return {
        "image": payload.image_name,
        "sequence_number": payload.sequence_number,
        "transcription": f"Fresh transcription of {payload.image_name}.",
        "processing_time": 0.01,
        "provider": "openai",
    }


def _fake_summary(transcription: str, page_num: int) -> dict[str, Any]:
    return {
        "page": page_num,
        "page_information": {
            "page_number_integer": page_num,
            "page_number_type": "arabic",
            "page_types": ["content"],
        },
        "bullet_points": [f"REGEN summary for page {page_num}"],
        "references": None,
        "provider": "openai",
    }


@pytest.fixture
def summarizing_env(
    monkeypatch: pytest.MonkeyPatch, mock_config_loader: MagicMock
) -> dict[str, MagicMock]:
    """Mock both LLM managers and enable summarization + both writers."""
    mock_tx = MagicMock()
    mock_tx.transcribe_payload.side_effect = _fake_transcribe
    mock_sum = MagicMock()
    mock_sum.generate_summary.side_effect = _fake_summary

    monkeypatch.setattr(
        transcriber_module, "TranscriptionManager", MagicMock(return_value=mock_tx)
    )
    monkeypatch.setattr(
        transcriber_module, "SummaryManager", MagicMock(return_value=mock_sum)
    )
    monkeypatch.setattr(
        transcriber_module, "get_config_loader", lambda: mock_config_loader
    )
    monkeypatch.setattr("imaging.payload.get_config_loader", lambda: mock_config_loader)
    monkeypatch.setattr(app_config, "SUMMARIZE", True)
    monkeypatch.setattr(app_config, "OUTPUT_DOCX", True)
    monkeypatch.setattr(app_config, "OUTPUT_MARKDOWN", True)
    # No live OpenAlex calls.
    monkeypatch.setattr(app_config, "CITATION_ENABLE_OPENALEX", False)
    return {"tx": mock_tx, "sum": mock_sum}


def _seed_logs(
    transcriber: ItemTranscriber,
    tx_entries: list[dict[str, Any]],
    summary_entries: list[dict[str, Any]],
    total_images: int,
) -> None:
    """Write versioned JSONL transcription and summary logs (crashed run)."""

    def _write(path: Path, log_type: str, entries: list[dict[str, Any]]) -> None:
        header = {
            "_format_version": LOG_FORMAT_VERSION,
            "log_type": log_type,
            "input_item_name": transcriber.name,
            "input_item_path": str(transcriber.input_path),
            "input_type": "PDF",
            "total_images": total_images,
            "model_name": transcriber.transcription_model,
        }
        lines = [json.dumps(header)] + [json.dumps(e) for e in entries]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write(transcriber.log_path, "transcription", tx_entries)
    _write(transcriber.summary_log_path, "summary", summary_entries)


def test_resume_reuses_prior_summaries_and_skips_transcription(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    summarizing_env: dict[str, MagicMock],
) -> None:
    """B1 + B2: resumed item's .md carries pre-crash summaries; no re-transcribe.

    Page 0 has both a logged transcription and a logged summary (reused).
    Page 1 has a logged transcription but no summary (regenerated from the log,
    with no transcription API call).
    """
    pdf_path = make_pdf("Resume.pdf", num_pages=2)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    transcriber = ItemTranscriber(
        input_path=pdf_path,
        input_type="pdf",
        base_output_dir=output_dir,
        completed_page_indices={0, 1},
    )
    _seed_logs(
        transcriber,
        tx_entries=[
            {
                "image": "page_0001.jpg",
                "original_input_order_index": 0,
                "transcription": "prior text page zero",
            },
            {
                "image": "page_0002.jpg",
                "original_input_order_index": 1,
                "transcription": "prior text page one",
            },
        ],
        summary_entries=[
            {
                "original_input_order_index": 0,
                "image_filename": "page_0001.jpg",
                "page": 1,
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["PRIOR pre-crash summary of page zero"],
                "references": None,
            }
        ],
        total_images=2,
    )

    transcriber.process_item()

    # B2: no transcription API call — both pages reused from the log.
    assert summarizing_env["tx"].transcribe_payload.call_count == 0
    # Page 0 summary reused (not regenerated); page 1 summary regenerated once.
    assert summarizing_env["sum"].generate_summary.call_count == 1

    # B1: the pre-crash summary of page 0 survives into the rendered Markdown,
    # and the regenerated page-1 summary is present too.
    md = transcriber.output_summary_md_path.read_text(encoding="utf-8")
    assert "PRIOR pre-crash summary of page zero" in md
    # Page at index 1 has no logged "page" key, so its regenerated summary uses
    # the fallback numbering (index + 1 = 2).
    assert "REGEN summary for page 2" in md


def test_resume_single_citation_manager_shared_by_both_writers(
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    summarizing_env: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B4: DOCX and Markdown writers receive the same CitationManager instance."""
    captured: dict[str, Any] = {}

    def spy_docx(results, path, name, citation_manager=None, data=None):
        captured["docx"] = citation_manager

    def spy_md(results, path, name, citation_manager=None, data=None):
        captured["md"] = citation_manager

    monkeypatch.setattr(transcriber_module, "create_docx_summary", spy_docx)
    monkeypatch.setattr(transcriber_module, "create_markdown_summary", spy_md)

    pdf_path = make_pdf("Cite.pdf", num_pages=1)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    transcriber = ItemTranscriber(
        input_path=pdf_path,
        input_type="pdf",
        base_output_dir=output_dir,
        completed_page_indices={0},
    )
    _seed_logs(
        transcriber,
        tx_entries=[
            {
                "image": "page_0001.jpg",
                "original_input_order_index": 0,
                "transcription": "text",
            }
        ],
        summary_entries=[
            {
                "original_input_order_index": 0,
                "page": 1,
                "page_information": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                    "page_types": ["content"],
                },
                "bullet_points": ["a point"],
                "references": ["Smith, J. (2020). A Title. Journal, 1, 1-2."],
            }
        ],
        total_images=1,
    )

    transcriber.process_item()

    assert captured["docx"] is not None
    assert captured["docx"] is captured["md"]
    assert isinstance(captured["docx"], CitationManager)


def test_completeness_gate_blocks_complete_when_pages_missing(
    tmp_path: Path,
) -> None:
    """B3: outputs exist but the log proves missing pages -> not COMPLETE."""
    from pipeline.paths import create_safe_directory_name, create_safe_log_filename

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    name = "Doc"
    for ext in (".txt", ".docx", ".md"):
        (output_dir / f"{name}{ext}").write_text("x", encoding="utf-8")

    working = output_dir / create_safe_directory_name(name, "_working_files")
    working.mkdir()
    log_path = working / create_safe_log_filename(name, "transcription")
    header = {
        "_format_version": LOG_FORMAT_VERSION,
        "log_type": "transcription",
        "input_item_name": name,
        "total_images": 5,  # expected 5 pages ...
        "model_name": "gpt-5-mini",
    }
    # ... but only 3 pages were logged (budget stall wrote partial outputs).
    entries = [
        {"original_input_order_index": i, "transcription": "t"} for i in range(3)
    ]
    log_path.write_text(
        "\n".join([json.dumps(header)] + [json.dumps(e) for e in entries]) + "\n",
        encoding="utf-8",
    )

    checker = ResumeChecker(resume_mode="skip", summarize=True)
    result = checker.should_skip(name, output_dir)

    assert result.state != ProcessingState.COMPLETE
    assert result.completed_page_indices == {0, 1, 2}


def test_completeness_gate_allows_complete_when_log_matches(tmp_path: Path) -> None:
    """A full log (or no log) leaves an all-outputs-present item COMPLETE."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    name = "Full"
    for ext in (".txt", ".docx", ".md"):
        (output_dir / f"{name}{ext}").write_text("x", encoding="utf-8")

    # No working-dir log: cannot prove a shortfall, so outputs are trusted.
    checker = ResumeChecker(resume_mode="skip", summarize=True)
    assert checker.should_skip(name, output_dir).state == ProcessingState.COMPLETE


# ============================================================================
# Citation deduplication policy
# ============================================================================


class TestCitationConsolidatePolicy:
    def test_accent_variants_merge(self) -> None:
        """Muller / Müller fold to one key before consolidation even runs."""
        manager = CitationManager()
        manager.add_citations(
            ["Muller, S. (1985). The All-Consuming Nature of Taste. Blackwell."], 1
        )
        manager.add_citations(
            ["Müller, S. (1985). The All-Consuming Nature of Taste. Blackwell."], 2
        )
        # Accent folding collapses them without needing the fuzzy pass.
        assert len(manager.citations) == 1
        merged = next(iter(manager.citations.values()))
        assert merged.pages == {1, 2}

    def test_fuzzy_variants_merge(self) -> None:
        """Same-author/year abbreviated vs full first name merge via fuzzy pass."""
        manager = CitationManager()
        manager.add_citations(
            ["Mennell, S. (1985). All Manners of Food. Blackwell."], 1
        )
        manager.add_citations(
            ["Mennell, Stephen (1985). All Manners of Food. Blackwell."], 2
        )
        # Distinct structured keys (S. vs Stephen), so both survive insertion.
        assert len(manager.citations) == 2
        manager.consolidate()
        assert len(manager.citations) == 1
        merged = next(iter(manager.citations.values()))
        assert merged.pages == {1, 2}

    def test_different_years_never_merge(self) -> None:
        """1985 vs 1987 (same author/title) are different works."""
        manager = CitationManager()
        manager.add_citations(["Smith, J. (1985). A Study of Things. Press."], 1)
        manager.add_citations(["Smith, J. (1987). A Study of Things. Press."], 2)
        manager.consolidate()
        assert len(manager.citations) == 2

    def test_different_volumes_never_merge(self) -> None:
        """Vol. 1 vs Vol. 2 (same author/year/title) are different works."""
        manager = CitationManager()
        manager.add_citations(
            ["Braudel, F. (1979). Civilization and Capitalism, Vol. 1. Harper."], 1
        )
        manager.add_citations(
            ["Braudel, F. (1979). Civilization and Capitalism, Vol. 2. Harper."], 2
        )
        manager.consolidate()
        assert len(manager.citations) == 2

    def test_unnumbered_page_citation_rendered(self) -> None:
        """A citation only on an unnumbered page renders 'unnumbered'."""
        manager = CitationManager()
        manager.add_citations(["Doe, J. (2001). Untitled."], None)
        citation = next(iter(manager.citations.values()))
        assert citation.get_page_range_str() == "unnumbered"


def test_enrichment_counts_every_request_against_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B9: misses count against max_api_requests, not only successes."""
    manager = CitationManager()
    manager.add_citations(["First, A. (2001). One. Press."], 1)
    manager.add_citations(["Second, B. (2002). Two. Press."], 2)
    manager.add_citations(["Third, C. (2003). Three. Press."], 3)

    calls = {"n": 0}

    def fake_fetch(text: str) -> None:
        calls["n"] += 1
        return None  # every lookup misses

    monkeypatch.setattr(manager, "_fetch_metadata_from_openalex", fake_fetch)

    manager.enrich_with_metadata(max_requests=2)

    # Exactly two requests were spent even though all missed (old code would
    # have kept going because misses did not count).
    assert calls["n"] == 2


# ============================================================================
# Decision 16: shared rate limiter + temperature wiring
# ============================================================================


def test_shared_rate_limiter_is_per_provider() -> None:
    """One RateLimiter instance is shared per provider across managers."""
    from llm.rate_limit import get_shared_rate_limiter

    a = get_shared_rate_limiter("openai")
    b = get_shared_rate_limiter("openai")
    c = get_shared_rate_limiter("anthropic")
    assert a is b
    assert a is not c


def test_temperature_wired_when_supported() -> None:
    """B5: temperature reaches invoke kwargs for a temperature-capable model."""
    from llm.base import LLMClientBase

    obj = LLMClientBase.__new__(LLMClientBase)
    obj.provider = "openai"
    obj.model_name = "gpt-4o"
    obj.model_config = {"temperature": 0.5}
    obj.service_tier = "auto"

    kwargs = obj._build_invoke_kwargs()
    assert kwargs.get("temperature") == 0.5


def test_temperature_skipped_for_reasoning_model() -> None:
    """B5: temperature is skipped for reasoning models / active thinking."""
    from llm.base import LLMClientBase

    obj = LLMClientBase.__new__(LLMClientBase)
    obj.provider = "openai"
    obj.model_name = "gpt-5-mini"
    obj.model_config = {"temperature": 0.5, "reasoning": {"effort": "low"}}
    obj.service_tier = "auto"

    kwargs = obj._build_invoke_kwargs()
    assert "temperature" not in kwargs
