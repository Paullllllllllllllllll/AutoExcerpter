"""Regression tests for the round-6 maintenance sweep fixes.

Pins, one class per fix:

1.  ``parse_latex_in_text`` resumes the inline scan just past a rejected
    display-overlapping candidate's opening ``$``, so a currency ``$`` before
    a ``$$...$$`` block no longer desynchronizes every later dollar pairing.
2.  ``balance_dollar_signs`` treats a trailing escaped ``\\$`` as a literal
    dollar and closes the genuine opener instead of deleting it.
3.  Payload sources tolerate ``provider: null`` / a null model name from
    model.yaml instead of crashing on ``None.lower()`` before any page runs.
4.  The resume shortfall gate counts UNIQUE logged page indices, so duplicate
    entries for one index cannot mask a genuinely missing page.
5.  The duplicate-output guard runs BEFORE resume filtering, so a same-stem
    sibling cannot be misclassified COMPLETE off another document's outputs.
6.  The ``--json`` summary line survives a legacy-codepage stdout by falling
    back to ASCII-escaped JSON instead of raising ``UnicodeEncodeError``.
7.  ``_load_model_config`` returns ``{}`` for a bare ``transcription_model:``
    / ``summary_model:`` header (YAML ``None``), not ``None``.
8.  The pre-run overview survives a partially-nulled concurrency.yaml
    (``api_requests:`` / ``transcription:`` / ``retry:`` as bare headers).
9.  ``find_targets`` sniffs the transcription header on raw bytes, so a
    legacy-encoded transcription is a target whose decode failure surfaces in
    the reported ``read_failures`` channel instead of being silently skipped.
10. The repair tool's image-block scan is bounded, so a deviant opener that
    never closes with ``]`` cannot swallow the rest of the file.
11. The interactive ALL sentinel (N+1) is honored inside a comma list, as the
    out-of-range message already promises.
12. ``_extract_doi`` strips unbalanced trailing ``]`` / ``>`` (and mixed
    tails) while preserving balanced internal pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import main.excerpt as main_mod
from cli import display
from config import app as app_config
from imaging.payload import FolderPayloadSource, _PayloadSourceBase
from llm.base import LLMClientBase
from pipeline.resume import ProcessingState, ResumeChecker
from pipeline.text_cleaner import balance_dollar_signs
from pipeline.types import ItemSpec
from rendering.citations import CitationManager
from rendering.docx import parse_latex_in_text
from scripts.repair_layout.repair import repair_text
from scripts.repair_layout.run_repair import find_targets, process_file
from tests.test_resume import _create_file, _create_log_with_entries  # noqa: F401


# ---------------------------------------------------------------------------
# Fix 1: inline-math pairing survives a currency $ before a $$...$$ block
# ---------------------------------------------------------------------------
class TestInlineMathAfterDisplayBlock:
    """A rejected display-overlapping candidate must not consume the opening
    ``$`` of the first genuine formula after the block."""

    def test_two_inline_formulas_after_display(self) -> None:
        segments = parse_latex_in_text(
            "Cost $5 rose. $$E=mc^2$$ Later $x_1$ and $y_2$ done."
        )
        assert ("x_1", "latex_inline") in segments
        assert ("y_2", "latex_inline") in segments
        # The prose word between them must stay text, not become a formula.
        assert (" and ", "text") in segments

    def test_single_inline_formula_after_display(self) -> None:
        segments = parse_latex_in_text("price $5 up. $$a=b$$ then $x^2$ end.")
        assert ("x^2", "latex_inline") in segments

    def test_control_without_currency_dollar(self) -> None:
        """Negative control: the fix must not disturb the clean case."""
        segments = parse_latex_in_text("Text $$E=mc^2$$ then $x_1$ end.")
        assert ("E=mc^2", "latex_display") in segments
        assert ("x_1", "latex_inline") in segments


# ---------------------------------------------------------------------------
# Fix 2: a trailing escaped \$ is not an orphan closing dollar
# ---------------------------------------------------------------------------
class TestBalanceDollarEscapedTail:
    def test_escaped_trailing_dollar_keeps_the_opener(self) -> None:
        assert (
            balance_dollar_signs("The sum $x+y equals \\$")
            == "The sum $x+y equals \\$$"
        )

    def test_true_orphan_at_line_end_still_removed(self) -> None:
        """Negative control: a genuinely bare trailing ``$`` is still dropped."""
        assert balance_dollar_signs("Orphan at end $") == "Orphan at end "

    def test_balanced_line_untouched(self) -> None:
        assert balance_dollar_signs("Sum $x+y$ done.") == "Sum $x+y$ done."


# ---------------------------------------------------------------------------
# Fix 3: payload sources tolerate provider: null / model name null
# ---------------------------------------------------------------------------
class TestPayloadSourceNullProvider:
    def test_folder_source_accepts_null_provider_and_model(
        self, tmp_path: Path
    ) -> None:
        source = FolderPayloadSource(tmp_path, provider=None, model_name=None)
        assert source.model_type == "openai"

    def test_model_name_detection_still_works_without_provider(
        self, tmp_path: Path
    ) -> None:
        base = _PayloadSourceBase(tmp_path, None, "gemini-2.5-flash")
        assert base.model_type == "google"


# ---------------------------------------------------------------------------
# Fix 4: shortfall gate counts unique page indices, not raw entries
# ---------------------------------------------------------------------------
class TestResumeShortfallUniqueCount:
    def test_duplicate_indices_do_not_mask_a_missing_page(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        item_name = "TestDocument"
        for suffix in (".txt", ".docx", ".md"):
            _create_file(output_dir / f"{item_name}{suffix}")
        # Three entries, three expected pages -- but only TWO unique indices:
        # page 2 never reached the log and must be resumed, not skipped.
        entries = [
            {"original_input_order_index": 0, "page": 1, "transcription": "a"},
            {"original_input_order_index": 0, "page": 1, "transcription": "a"},
            {"original_input_order_index": 1, "page": 2, "transcription": "b"},
        ]
        _create_log_with_entries(output_dir, item_name, entries)

        checker = ResumeChecker(
            resume_mode="skip",
            summarize=True,
            output_docx=True,
            output_markdown=True,
        )
        result = checker.should_skip(item_name, output_dir)

        assert result.state != ProcessingState.COMPLETE
        assert result.completed_page_indices == {0, 1}

    def test_control_full_log_is_still_complete(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        item_name = "TestDocument"
        for suffix in (".txt", ".docx", ".md"):
            _create_file(output_dir / f"{item_name}{suffix}")
        entries = [
            {"original_input_order_index": i, "page": i + 1, "transcription": "t"}
            for i in range(3)
        ]
        _create_log_with_entries(output_dir, item_name, entries)

        checker = ResumeChecker(
            resume_mode="skip",
            summarize=True,
            output_docx=True,
            output_markdown=True,
        )
        result = checker.should_skip(item_name, output_dir)

        assert result.state == ProcessingState.COMPLETE


# ---------------------------------------------------------------------------
# Fix 5: the duplicate-output guard runs before resume filtering
# ---------------------------------------------------------------------------
class TestGuardRunsBeforeResumeFiltering:
    def test_collision_aborts_before_filtering(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item_a = ItemSpec(kind="pdf", path=tmp_path / "dirA" / "report.pdf")
        item_b = ItemSpec(kind="pdf", path=tmp_path / "dirB" / "report.pdf")

        monkeypatch.setattr(main_mod, "setup_argparse", lambda: argparse.Namespace())
        # main/excerpt.py binds ``from config import app as config``; patching
        # the app-config module patches the same object main.excerpt reads.
        monkeypatch.setattr(app_config, "CLI_MODE", True)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)
        monkeypatch.setattr(
            main_mod,
            "_setup_and_scan",
            lambda args: ([item_a, item_b], tmp_path / "out", None, "skip"),
        )

        def _must_not_run(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "resume filtering ran before the duplicate-output guard"
            )

        monkeypatch.setattr(main_mod, "_apply_resume_filtering", _must_not_run)

        with pytest.raises(SystemExit) as exc:
            main_mod.main()
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Fix 6: the --json line survives a legacy-codepage stdout
# ---------------------------------------------------------------------------
class TestJsonLineSurvivesCodepage:
    @staticmethod
    def _codepage_print(printed: list[str]) -> Any:
        def fake_print(arg: str) -> None:
            if any(ord(ch) > 127 for ch in arg):
                raise UnicodeEncodeError(
                    "charmap", arg, 0, 1, "character maps to <undefined>"
                )
            printed.append(arg)

        return fake_print

    def test_non_ascii_object_falls_back_to_escaped_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        printed: list[str] = []
        monkeypatch.setattr("builtins.print", self._codepage_print(printed))

        obj = {"outputs": ["Kraków_łan.txt"], "items_complete": 1}
        main_mod._print_json_line(obj)

        assert len(printed) == 1
        # Lossless: the escaped line parses back to the identical object.
        assert json.loads(printed[0]) == obj

    def test_ascii_object_prints_directly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        printed: list[str] = []
        monkeypatch.setattr("builtins.print", self._codepage_print(printed))

        main_mod._print_json_line({"items_complete": 2})

        assert printed == ['{"items_complete": 2}']


# ---------------------------------------------------------------------------
# Fix 7: a bare transcription_model:/summary_model: header yields {}
# ---------------------------------------------------------------------------
class TestLoadModelConfigNullSection:
    def test_null_section_returns_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader = MagicMock()
        loader.get_model_config.return_value = {"transcription_model": None}
        monkeypatch.setattr("llm.base.get_config_loader", lambda: loader)

        result = LLMClientBase._load_model_config(
            cast(LLMClientBase, object()), "transcription_model"
        )

        assert result == {}

    def test_real_section_returned_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loader = MagicMock()
        loader.get_model_config.return_value = {
            "transcription_model": {"name": "gpt-5-mini"}
        }
        monkeypatch.setattr("llm.base.get_config_loader", lambda: loader)

        result = LLMClientBase._load_model_config(
            cast(LLMClientBase, object()), "transcription_model"
        )

        assert result == {"name": "gpt-5-mini"}


# ---------------------------------------------------------------------------
# Fix 8: the overview survives a partially-nulled concurrency.yaml
# ---------------------------------------------------------------------------
class TestOverviewSurvivesNulledConcurrency:
    @pytest.mark.parametrize(
        "concurrency_cfg",
        [
            {"api_requests": None, "retry": None},
            {"api_requests": {"transcription": None, "summary": None}, "retry": {}},
        ],
        ids=["null-sections", "null-subsections"],
    )
    def test_overview_renders_without_attribute_error(
        self,
        concurrency_cfg: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(app_config, "SUMMARIZE", True)
        monkeypatch.setattr(app_config, "DAILY_TOKEN_LIMIT_ENABLED", False)
        monkeypatch.setattr(app_config, "INPUT_PATHS_IS_OUTPUT_PATH", False)

        loader = MagicMock()
        loader.get_concurrency_config.return_value = concurrency_cfg
        loader.get_model_config.return_value = {
            "transcription_model": {"name": "gpt-5-mini"},
            "summary_model": {"name": "gpt-5-mini"},
        }
        loader.get_image_processing_config.return_value = {}
        monkeypatch.setattr("config.loader.get_config_loader", lambda: loader)
        monkeypatch.setattr(display, "prompt_yes_no", lambda *a, **k: True)

        result = display._display_processing_summary(
            [ItemSpec(kind="pdf", path=Path("/tmp/A.pdf"))], Path("/out"), None
        )

        assert result is True
        out = capsys.readouterr().out
        assert "Transcription API" in out


# ---------------------------------------------------------------------------
# Fix 9: legacy-encoded transcriptions are targets, reported not silent
# ---------------------------------------------------------------------------
class TestFindTargetsLegacyEncoding:
    def test_cp1252_transcription_is_a_target(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.txt"
        legacy.write_bytes(b"# Transcription of: caf\xe9.pdf\r\nBody text\r\n")

        assert legacy in find_targets(tmp_path)

    def test_cp1252_target_surfaces_as_read_failure(self, tmp_path: Path) -> None:
        """The decode failure reaches the reported channel via process_file."""
        legacy = tmp_path / "legacy.txt"
        legacy.write_bytes(b"# Transcription of: caf\xe9.pdf\r\nBody text\r\n")

        with pytest.raises(UnicodeDecodeError):
            process_file(legacy, tmp_path, dry_run=True)

    def test_non_transcription_stays_excluded(self, tmp_path: Path) -> None:
        """Negative control: unrelated legacy-encoded .txt is not a target."""
        other = tmp_path / "notes.txt"
        other.write_bytes(b"random caf\xe9 notes\n")

        assert other not in find_targets(tmp_path)

    def test_bom_utf8_transcription_still_found(self, tmp_path: Path) -> None:
        bom = tmp_path / "bom.txt"
        bom.write_bytes(b"\xef\xbb\xbf# Transcription of: x.pdf\nBody\n")

        assert bom in find_targets(tmp_path)


# ---------------------------------------------------------------------------
# Fix 10: a deviant image opener cannot swallow the rest of the file
# ---------------------------------------------------------------------------
class TestImageBlockScanBounded:
    def test_repairs_still_run_after_a_markdown_image(self) -> None:
        body = "\n".join(
            [
                "# Transcription of: test.pdf",
                "",
                "<page_number>1</page_number>",
                "![alt](fig.png)",
                "",
                "This is a long line of ordinary prose that was wrapped by the greedy",
                "wrapper at a fixed width and continues on and on for quite a while",
                "here to give the estimator several full-length lines to work with in",
                "this region so the width can be estimated with reasonable confidence",
                "okay.",
                "",
                "<page_number>2</page_number>",
            ]
        )
        repaired, _audit = repair_text(body)

        # The orphan remainder is rejoined; pre-fix the unbounded block scan
        # swallowed everything after the opener and no repair ran.
        assert "reasonable confidence okay." in repaired
        assert "\nokay." not in repaired

    def test_control_proper_image_block_passes_through(self) -> None:
        body = "\n".join(
            [
                "# Transcription of: test.pdf",
                "",
                "![Image: a woodcut of a kitchen scene]",
                "",
                "Prose follows.",
            ]
        )
        repaired, _audit = repair_text(body)
        assert "![Image: a woodcut of a kitchen scene]" in repaired


# ---------------------------------------------------------------------------
# Fix 11: the ALL sentinel is honored inside a comma list
# ---------------------------------------------------------------------------
class TestAllSentinelInCommaList:
    def test_sentinel_in_comma_list_selects_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cli.interaction as inter

        monkeypatch.setattr("builtins.input", lambda _p: "2,4")
        result = inter.prompt_selection(["a", "b", "c"], display_func=str)

        assert result == ["a", "b", "c"]

    def test_control_plain_comma_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cli.interaction as inter

        monkeypatch.setattr("builtins.input", lambda _p: "1,3")
        result = inter.prompt_selection(["a", "b", "c"], display_func=str)

        assert result == ["a", "c"]


# ---------------------------------------------------------------------------
# Fix 12: _extract_doi strips unbalanced trailing ] and >
# ---------------------------------------------------------------------------
class TestDoiBracketAngleBalance:
    def test_bracketed_doi_loses_trailing_bracket(self) -> None:
        manager = CitationManager()
        assert manager._extract_doi("[doi:10.1234/abc123]") == "10.1234/abc123"

    def test_angle_wrapped_url_loses_trailing_angle(self) -> None:
        manager = CitationManager()
        assert manager._extract_doi("<https://doi.org/10.5555/xyz9>") == "10.5555/xyz9"

    def test_mixed_tail_fully_stripped(self) -> None:
        manager = CitationManager()
        assert manager._extract_doi("see (doi:10.1234/abc)]") == "10.1234/abc"

    def test_balanced_internal_angles_preserved(self) -> None:
        """Negative control: a Wiley-style DOI keeps its balanced angles."""
        manager = CitationManager()
        doi = "10.1002/(SICI)1097-0257(199804)17:8<873::AID-SIM779>3.0.CO"
        assert manager._extract_doi(f"doi:{doi}") == doi

    def test_balanced_internal_parens_preserved(self) -> None:
        manager = CitationManager()
        assert (
            manager._extract_doi("https://doi.org/10.1016/S0140-6736(00)57123-X")
            == "10.1016/S0140-6736(00)57123-X"
        )
