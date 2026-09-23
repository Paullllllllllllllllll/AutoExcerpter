"""Round 4: maintenance-sweep regression tests.

One focused regression per verified fix:

1. A phase without its own ``service_tier`` inherits the transcription tier
   before falling back to ``"flex"`` (the pre-run overview already showed the
   inherited value, but the accessor returned ``"flex"``).
2. The four "config missing / failed to load -> using bundled defaults"
   notices are logged at WARNING; at INFO the console handler discarded them,
   so the user never learned they were running on the shipped template.
3. ``_dehyphenate_lines`` merges hyphenation to a fixpoint, so a word split
   across three printed lines is rejoined instead of keeping an interior
   hyphen.
4. ``balance_dollar_signs`` recognizes the European postfix currency style
   ``"100 $"`` (whitespace between the digit and the sign) as currency.
5. A digits-only escaped bracket pair (``\\[42\\]``) is a citation marker, not
   display math, and stays literal.
6. Failed (``other``) and blank pages fold into the content flow instead of
   minting singleton pseudo-sections that the section-median sort relocates
   to the end of the document.
7. ``_run_dry_run`` disarms the exit hook after emitting its one JSON object,
   so a later interrupt cannot print a second, contradictory summary.
8. An escaped dollar inside a currency span no longer defeats the currency
   guard in ``parse_latex_in_text`` (the placeholder contains ``_``, a math
   indicator).
9. Malformed OpenAlex payloads (non-object 200 bodies, non-dict candidates,
   null ``authorships``, null ``primary_location``) are tolerated instead of
   crashing the enrichment run.
10. NaN / Infinity debris in the shared ledger coerces to 0 rather than
    raising out of ``int()``.
11. ``_build_invoke_kwargs``: ``output_config`` counts as active reasoning
    (so no temperature is sent), adaptive-thinking Claude models declare
    ``supports_temperature=False``, and Gemini 3+ takes a discrete
    ``thinking_level`` while the 2.5 family keeps ``thinking_budget``.
12. ``model_copy(update=...)`` survives ``with_structured_output``: the SDK
    merges the copied ``output_config.effort`` with the wrapper's
    ``output_config.format`` (the merge ``_get_structured_chat_model``
    relies on).
13. Cooperative abort: the rate limiter abandons a saturated window early
    without recording a request, and ``_invoke_with_retry`` raises before
    the API call when the abort event is set.
14. Plain-text sentinel exhaustion returns a success dict with placeholder
    text, not an error-marked dict (no schema exists to validate against).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

import config.accessors as accessors
import config.app as app_config
import config.loader as config_loader
import llm.shared_ledger as shared_ledger
from llm.base import LLMClientBase, clear_abort, request_abort
from llm.capabilities import detect_capabilities
from llm.rate_limit import RateLimiter
from llm.transcription import TranscriptionManager
from pipeline.page_numbering import PageNumberProcessor
from pipeline.text_cleaner import balance_dollar_signs, normalize_math_delimiters
from rendering.citations import CitationManager, _reset_budget_state_for_tests
from rendering.docx import parse_latex_in_text
from scripts.repair_layout.repair import _dehyphenate_lines


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _loader_with(cfg: dict[str, Any]) -> MagicMock:
    """Return a mock ConfigLoader serving *cfg* as the concurrency config."""
    loader = MagicMock()
    loader.get_concurrency_config.return_value = cfg
    return loader


def _propagate(monkeypatch: pytest.MonkeyPatch, logger_name: str) -> logging.Logger:
    """Let *logger_name* propagate so pytest's caplog handler sees its records.

    ``setup_logger`` sets ``propagate = False`` to avoid duplicate console
    output; caplog installs its handler on the root logger.
    """
    logger = logging.getLogger(logger_name)
    monkeypatch.setattr(logger, "propagate", True)
    return logger


def _make_client(**overrides: Any) -> LLMClientBase:
    """Create a bare LLMClientBase bypassing __init__ for isolated unit tests."""
    client = LLMClientBase.__new__(LLMClientBase)
    defaults: dict[str, Any] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "timeout": 300,
        "rate_limiter": None,
        "max_retries": 5,
        "max_elapsed": 0.0,
        "chat_model": MagicMock(),
        "_stats_lock": threading.Lock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "model_config": {},
        "service_tier": "auto",
        "schema_retry_config": {},
        "_output_schema": None,
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(client, attr, val)
    return client


@pytest.fixture(autouse=True)
def _reset_openalex_budget_latch() -> Generator[None]:
    """Clear the module-level OpenAlex budget latch around every test."""
    _reset_budget_state_for_tests()
    yield
    _reset_budget_state_for_tests()


# ---------------------------------------------------------------------------
# Fix 1: a phase without its own service_tier inherits the transcription tier
# ---------------------------------------------------------------------------
class TestServiceTierPhaseFallback:
    """A missing per-phase tier inherits transcription, then "flex"."""

    def test_summary_inherits_transcription_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg: dict[str, Any] = {
            "api_requests": {"transcription": {"service_tier": "priority"}}
        }
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with(cfg))

        assert accessors.get_service_tier("summary") == "priority"

    def test_explicit_summary_tier_still_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg: dict[str, Any] = {
            "api_requests": {
                "transcription": {"service_tier": "priority"},
                "summary": {"service_tier": "flex"},
            }
        }
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with(cfg))

        assert accessors.get_service_tier("summary") == "flex"

    def test_nothing_configured_falls_back_to_flex(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback must terminate rather than recurse forever."""
        monkeypatch.setattr(accessors, "get_config_loader", lambda: _loader_with({}))

        assert accessors.get_service_tier("summary") == "flex"
        assert accessors.get_service_tier("transcription") == "flex"


# ---------------------------------------------------------------------------
# Fix 2: bundled-defaults notices are WARNING, not (invisible) INFO
# ---------------------------------------------------------------------------
class TestBundledDefaultsWarnings:
    """The console handler drops records below WARNING; these must be seen."""

    def test_loader_missing_config_warns_about_example(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (tmp_path / "demo.example.yaml").write_text("key: value", encoding="utf-8")
        monkeypatch.setattr(config_loader, "CONFIG_DIR", tmp_path)
        _propagate(monkeypatch, config_loader.logger.name)

        with caplog.at_level(logging.WARNING, logger=config_loader.logger.name):
            result = config_loader.ConfigLoader()._load_yaml_config("demo.yaml")

        assert result == {"key": "value"}
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("demo.example.yaml" in r.getMessage() for r in warnings)

    def test_loader_unparsable_config_warns_about_example(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (tmp_path / "demo.example.yaml").write_text("key: value", encoding="utf-8")
        (tmp_path / "demo.yaml").write_text("key: [unclosed", encoding="utf-8")
        monkeypatch.setattr(config_loader, "CONFIG_DIR", tmp_path)
        _propagate(monkeypatch, config_loader.logger.name)

        with caplog.at_level(logging.WARNING, logger=config_loader.logger.name):
            result = config_loader.ConfigLoader()._load_yaml_config("demo.yaml")

        assert result == {"key": "value"}
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "failed to load" in r.getMessage() and "demo.example.yaml" in r.getMessage()
            for r in warnings
        )

    def test_app_config_missing_warns_about_example(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (tmp_path / "app.example.yaml").write_text("cli_mode: false", encoding="utf-8")
        monkeypatch.setattr(app_config, "_APP_CONFIG_PATH", tmp_path / "app.yaml")
        _propagate(monkeypatch, app_config.logger.name)

        with caplog.at_level(logging.WARNING, logger=app_config.logger.name):
            result = app_config._load_yaml_app_config()

        assert result == {"cli_mode": False}
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("app.example.yaml" in r.getMessage() for r in warnings)

    def test_app_config_unparsable_warns_about_example(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (tmp_path / "app.example.yaml").write_text("cli_mode: false", encoding="utf-8")
        (tmp_path / "app.yaml").write_text("cli_mode: [unclosed", encoding="utf-8")
        monkeypatch.setattr(app_config, "_APP_CONFIG_PATH", tmp_path / "app.yaml")
        _propagate(monkeypatch, app_config.logger.name)

        with caplog.at_level(logging.WARNING, logger=app_config.logger.name):
            result = app_config._load_yaml_app_config()

        assert result == {"cli_mode": False}
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "failed to load" in r.getMessage() and "app.example.yaml" in r.getMessage()
            for r in warnings
        )


# ---------------------------------------------------------------------------
# Fix 3: de-hyphenation runs to a fixpoint
# ---------------------------------------------------------------------------
class TestDehyphenationFixpoint:
    """A word broken across three printed lines must rejoin completely."""

    def test_triple_split_merges_into_one_word(self) -> None:
        lines, _ = _dehyphenate_lines(["Manage-", "ment-", "system des Staates"])

        assert lines == ["Managementsystem", "des Staates"]

    def test_triple_split_records_two_decisions(self) -> None:
        _, decisions = _dehyphenate_lines(["Manage-", "ment-", "system des Staates"])

        assert [(d.left, d.right) for d in decisions] == [
            ("Manage", "ment"),
            ("Management", "system"),
        ]

    def test_two_line_split_unchanged(self) -> None:
        lines, decisions = _dehyphenate_lines(["Manage-", "ment des Staates"])

        assert lines == ["Management", "des Staates"]
        assert len(decisions) == 1


# ---------------------------------------------------------------------------
# Fix 4: space-separated line-final currency survives
# ---------------------------------------------------------------------------
class TestSpacedCurrencySurvives:
    """The European postfix style "100 $" is currency, not orphan math."""

    def test_spaced_postfix_currency_unchanged(self) -> None:
        text = "Der Preis betrug 100 $"

        assert balance_dollar_signs(text) == text

    def test_tight_postfix_currency_unchanged(self) -> None:
        assert balance_dollar_signs("100$") == "100$"

    def test_prefix_currency_unchanged(self) -> None:
        assert balance_dollar_signs("$ 30") == "$ 30"


# ---------------------------------------------------------------------------
# Fix 5: digits-only escaped bracket pairs stay literal
# ---------------------------------------------------------------------------
class TestDigitsOnlyBracketStaysLiteral:
    """ "\\[42\\]" is an escaped citation marker, not a display formula."""

    def test_escaped_citation_marker_unchanged(self) -> None:
        text = r"see note \[42\] for details"

        assert normalize_math_delimiters(text) == text

    def test_bare_digit_pair_unchanged(self) -> None:
        assert normalize_math_delimiters(r"\[5\]") == r"\[5\]"

    def test_real_formula_still_converts(self) -> None:
        assert normalize_math_delimiters(r"\[x^2\]") == "$$x^2$$"


# ---------------------------------------------------------------------------
# Fix 6: "other" / "blank" pages fold into the content flow
# ---------------------------------------------------------------------------
def _summary(index: int, page_types: list[str], number: int | None) -> dict[str, Any]:
    """Build one summary_result carrying flat page_information."""
    return {
        "original_input_order_index": index,
        "page_information": {
            "page_number_integer": number,
            "page_number_type": "arabic" if number is not None else "none",
            "page_types": page_types,
        },
    }


class TestPlaceholderPagesFoldIntoContent:
    """Unnumbered placeholders must not become relocatable pseudo-sections."""

    @pytest.fixture
    def processor(self) -> PageNumberProcessor:
        return PageNumberProcessor()

    def test_other_maps_to_content(self, processor: PageNumberProcessor) -> None:
        assert processor._get_primary_section_type(["other"]) == "content"

    def test_blank_maps_to_content(self, processor: PageNumberProcessor) -> None:
        assert processor._get_primary_section_type(["blank"]) == "content"

    def test_genuine_unknown_section_keeps_its_type(
        self, processor: PageNumberProcessor
    ) -> None:
        assert processor._get_primary_section_type(["toc"]) == "toc"

    def test_failed_page_keeps_its_physical_position(
        self, processor: PageNumberProcessor
    ) -> None:
        """A failed page at index 2 stays at index 2 in the render order.

        As its own section its median index tied with content's, and the
        min-index tiebreak pushed the whole singleton section past every
        content page, so the placeholder rendered at the end of the document.
        """
        summaries = [
            _summary(0, ["content"], 1),
            _summary(1, ["content"], 2),
            _summary(2, ["other"], None),
            _summary(3, ["content"], 4),
            _summary(4, ["content"], 5),
        ]

        ordered = processor.adjust_and_sort_page_numbers(summaries)

        assert [r["original_input_order_index"] for r in ordered] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Fix 7: the dry run disarms the exit hook after emitting its JSON object
# ---------------------------------------------------------------------------
class TestDryRunDisarmsExitHook:
    """Only one JSON object may reach stdout in --json dry-run mode."""

    def test_exit_hook_cleared_after_json_emission(self) -> None:
        import main.excerpt as main_module

        with patch.object(main_module, "set_exit_hook") as mock_hook:
            main_module._run_dry_run([], {}, [], emit_json=True)

        mock_hook.assert_called_once_with(None)

    def test_exit_hook_untouched_without_json(self) -> None:
        import main.excerpt as main_module

        with patch.object(main_module, "set_exit_hook") as mock_hook:
            main_module._run_dry_run([], {}, [], emit_json=False)

        mock_hook.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 8: an escaped dollar inside a currency span keeps the span as prose
# ---------------------------------------------------------------------------
class TestEscapedDollarInCurrencySpan:
    """The escaped-dollar placeholder must not read as a math indicator."""

    def test_currency_span_with_escaped_dollar_stays_text(self) -> None:
        segments = parse_latex_in_text("cost $5 \\$ each and $x$ math")

        assert segments == [
            ("cost $5 $ each and ", "text"),
            ("x", "latex_inline"),
            (" math", "text"),
        ]

    def test_formula_with_escaped_dollar_stays_math(self) -> None:
        """A math indicator in the span still wins over the currency guard."""
        segments = parse_latex_in_text("total $5 = \\$x^2$ end")

        assert segments == [
            ("total ", "text"),
            ("5 = $x^2", "latex_inline"),
            (" end", "text"),
        ]


# ---------------------------------------------------------------------------
# Fix 9: malformed OpenAlex payloads never crash the enrichment run
# ---------------------------------------------------------------------------
def _mock_200(body: Any) -> MagicMock:
    """Build a MagicMock 200 response whose .json() yields *body*."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    return response


class TestMalformedOpenAlexPayloads:
    """A proxy, captive portal, or API change must degrade to a miss."""

    def test_non_object_200_body_returns_none(self) -> None:
        manager = CitationManager()

        with patch(
            "rendering.citations.requests.get", return_value=_mock_200([1, 2, 3])
        ):
            result = manager._make_openalex_request(
                "https://api.openalex.org/works", {"search": "test"}, "test query"
            )

        assert result is None

    def test_non_dict_candidates_are_skipped(self) -> None:
        manager = CitationManager()
        citation = "Smith, John. *Introduction to Modern Testing*. Test Press, 2020."
        body = {"results": [None, "junk", 42]}

        with patch("rendering.citations.requests.get", return_value=_mock_200(body)):
            result = manager._query_openalex_by_text(citation)

        assert result is None

    def test_valid_candidate_after_junk_still_matches(self) -> None:
        manager = CitationManager()
        citation = "Smith, John. *Introduction to Modern Testing*. Test Press, 2020."
        body = {
            "results": [
                None,
                "junk",
                {
                    "title": "Introduction to Modern Testing",
                    "publication_year": 2020,
                    "doi": "https://doi.org/10.1234/test",
                },
            ]
        }

        with patch("rendering.citations.requests.get", return_value=_mock_200(body)):
            result = manager._query_openalex_by_text(citation)

        assert result is not None
        assert result["doi"] == "10.1234/test"

    def test_verify_match_tolerates_null_authorships(self) -> None:
        manager = CitationManager()
        work = {"title": "Introduction to Modern Testing", "authorships": None}

        assert (
            manager._verify_citation_match(
                "Smith, John. *Introduction to Modern Testing*. Test Press.", work
            )
            is False
        )

    def test_extract_metadata_tolerates_null_fields(self) -> None:
        manager = CitationManager()
        work: dict[str, Any] = {
            "title": "A Title",
            "doi": None,
            "authorships": None,
            "primary_location": None,
        }

        metadata = manager._extract_metadata_from_response(work)

        assert metadata["doi"] is None
        assert metadata["authors"] == []
        assert metadata.get("venue") is None

    def test_extract_metadata_tolerates_wrong_types(self) -> None:
        manager = CitationManager()
        work: dict[str, Any] = {
            "title": "A Title",
            "doi": 12345,
            "authorships": [None, "junk", {"author": None}, {"author": {}}],
            "primary_location": {"source": None},
        }

        metadata = manager._extract_metadata_from_response(work)

        assert metadata["doi"] is None
        assert metadata["authors"] == []
        assert metadata.get("venue") is None


# ---------------------------------------------------------------------------
# Fix 10: NaN / Infinity debris in the shared ledger coerces to 0
# ---------------------------------------------------------------------------
def _write_ledger(tmp_path: Path, tools: dict[str, Any]) -> None:
    """Write a same-day ledger file whose "tools" map holds *tools*."""
    payload = {
        "schema_version": shared_ledger.LEDGER_SCHEMA_VERSION,
        "date": shared_ledger._today(),
        "tools": tools,
        "usage": [],
    }
    # json.dumps emits bare NaN / Infinity tokens, which json.loads accepts.
    (tmp_path / shared_ledger.LEDGER_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


class TestLedgerNaNHardening:
    """Non-finite floats pass the isinstance check but blow up ``int()``."""

    def test_module_version_bumped(self) -> None:
        assert shared_ledger.LEDGER_MODULE_VERSION == "2.1.4"

    def test_coerce_int_neutralizes_debris(self) -> None:
        assert shared_ledger._coerce_int(float("nan")) == 0
        assert shared_ledger._coerce_int(float("inf")) == 0
        assert shared_ledger._coerce_int(float("-inf")) == 0
        # A JSON boolean is not a token count either.
        assert shared_ledger._coerce_int(True) == 0

    def test_readers_survive_non_finite_tools(self, tmp_path: Path) -> None:
        _write_ledger(
            tmp_path,
            {
                "chronominer": float("nan"),
                "chronotranscriber": float("inf"),
                "autoexcerpter": 500,
            },
        )
        ledger = shared_ledger.SharedTokenLedger("autoexcerpter", ledger_dir=tmp_path)

        combined = ledger.read_combined()
        breakdown = ledger.read_breakdown()
        usage = ledger.read_usage()

        assert combined == 500
        assert breakdown == {
            "chronominer": 0,
            "chronotranscriber": 0,
            "autoexcerpter": 500,
        }
        assert usage is not None
        assert usage.combined == 500
        assert usage.own_total == 500
        assert not math.isnan(float(usage.combined))


# ---------------------------------------------------------------------------
# Fix 11a/b: output_config suppresses temperature; adaptive models declare it
# ---------------------------------------------------------------------------
_ADAPTIVE_CAPS: dict[str, bool] = {
    "max_tokens": True,
    "reasoning": False,
    "extended_thinking": False,
    "adaptive_thinking": True,
    "thinking": False,
    "text_verbosity": False,
    # Deliberately True: without the output_config guard the temperature
    # would sail through and the API would reject the request.
    "temperature": True,
}


class TestOutputConfigSuppressesTemperature:
    """``output_config`` marks reasoning as active, like ``thinking``."""

    @patch("llm.base.get_model_capabilities", return_value=_ADAPTIVE_CAPS)
    def test_adaptive_effort_drops_temperature(self, _: Any) -> None:
        client = _make_client(
            provider="anthropic",
            model_name="claude-fable-5",
            model_config={"reasoning": {"effort": "high"}, "temperature": 0.2},
            service_tier="",
        )

        kwargs = client._build_invoke_kwargs()

        assert kwargs["output_config"] == {"effort": "high"}
        assert "temperature" not in kwargs

    @patch("llm.base.get_model_capabilities", return_value=_ADAPTIVE_CAPS)
    def test_temperature_survives_without_output_config(self, _: Any) -> None:
        """Control: with no reasoning configured the temperature is sent."""
        client = _make_client(
            provider="anthropic",
            model_name="claude-fable-5",
            model_config={"temperature": 0.2},
            service_tier="",
        )

        kwargs = client._build_invoke_kwargs()

        assert "output_config" not in kwargs
        assert kwargs["temperature"] == 0.2

    def test_adaptive_registry_entries_disable_temperature(self) -> None:
        assert detect_capabilities("claude-fable-5").supports_temperature is False
        assert detect_capabilities("claude-sonnet-4-6").supports_temperature is False


# ---------------------------------------------------------------------------
# Fix 11c: Gemini 3+ takes thinking_level; the 2.5 family keeps the budget
# ---------------------------------------------------------------------------
_GOOGLE_CAPS: dict[str, bool] = {
    "max_tokens": True,
    "reasoning": False,
    "extended_thinking": False,
    "adaptive_thinking": False,
    "thinking": True,
    "text_verbosity": False,
    "temperature": False,
}


class TestGoogleThinkingRouting:
    """``thinking_budget`` is deprecated for Gemini 3+; levels replace it."""

    @staticmethod
    def _google_client(model_name: str, effort: str) -> LLMClientBase:
        return _make_client(
            provider="google",
            model_name=model_name,
            model_config={"reasoning": {"effort": effort}},
            service_tier="",
        )

    @patch("llm.base.get_model_capabilities", return_value=_GOOGLE_CAPS)
    def test_gemini_3_uses_thinking_level(self, _: Any) -> None:
        client = self._google_client("gemini-3.5-flash", "medium")
        kwargs = client._build_invoke_kwargs()

        assert kwargs["thinking_config"] == {"thinking_level": "medium"}

    @patch("llm.base.get_model_capabilities", return_value=_GOOGLE_CAPS)
    def test_gemini_3_clamps_xhigh_to_high(self, _: Any) -> None:
        kwargs = self._google_client("gemini-3-pro", "xhigh")._build_invoke_kwargs()

        assert kwargs["thinking_config"] == {"thinking_level": "high"}

    @patch("llm.base.get_model_capabilities", return_value=_GOOGLE_CAPS)
    def test_gemini_25_keeps_numeric_budget(self, _: Any) -> None:
        client = self._google_client("gemini-2.5-flash", "medium")
        kwargs = client._build_invoke_kwargs()

        assert kwargs["thinking_config"] == {"thinking_budget": 4096}

    @patch("llm.base.get_model_capabilities", return_value=_GOOGLE_CAPS)
    def test_gemini_25_minimal_maps_to_smallest_budget(self, _: Any) -> None:
        """ "minimal" was absent from the budget table and sent nothing."""
        client = self._google_client("gemini-2.5-flash", "minimal")
        kwargs = client._build_invoke_kwargs()

        assert kwargs["thinking_config"] == {"thinking_budget": 512}


# ---------------------------------------------------------------------------
# Fix 12: model_copy(update=...) survives with_structured_output
# ---------------------------------------------------------------------------
class TestAnthropicOutputConfigMerge:
    """``bind()`` kwargs are dropped by ``with_structured_output``; copies are not.

    This pins the SDK behavior ``_get_structured_chat_model`` relies on: a
    field copied onto the model merges with the structured-output wrapper's
    own ``output_config`` rather than replacing it.
    """

    def test_effort_and_format_both_reach_the_payload(self) -> None:
        model = ChatAnthropic(
            api_key="test-key",  # type: ignore[arg-type]
            model="claude-sonnet-4-5",  # type: ignore[call-arg]
            max_tokens=1024,
        )
        copied = model.model_copy(
            update={"max_tokens": 32000, "output_config": {"effort": "medium"}}
        )

        payload = copied._get_request_payload(
            [HumanMessage(content="hi")],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "properties": {}},
                }
            },
        )

        assert payload["max_tokens"] == 32000
        assert payload["output_config"]["effort"] == "medium"
        assert payload["output_config"]["format"]["type"] == "json_schema"


# ---------------------------------------------------------------------------
# Fix 13: cooperative abort short-circuits both the wait and the call
# ---------------------------------------------------------------------------
class TestAbortHandling:
    """After Ctrl+C no worker may sit out a window or fire a fresh call."""

    def test_saturated_window_returns_promptly_on_abort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        limiter = RateLimiter([(1, 60)])
        limiter.wait_for_capacity()  # saturate the one-request window

        def _no_sleep(seconds: float) -> None:
            raise AssertionError(f"slept {seconds}s despite an abort")

        monkeypatch.setattr(time, "sleep", _no_sleep)

        start = time.monotonic()
        waited = limiter.wait_for_capacity(should_abort=lambda: True)

        assert time.monotonic() - start < 1.0
        assert waited >= 0.0
        # The aborted wait records nothing: no timestamp, no request count.
        assert limiter.total_requests == 1
        assert len(limiter.request_timestamps[0]) == 1

    def test_invoke_with_retry_raises_before_calling_the_model(self) -> None:
        client = _make_client()
        structured_model = MagicMock()

        request_abort()
        try:
            with pytest.raises(RuntimeError, match="Abort requested"):
                client._invoke_with_retry(
                    structured_model, [], {}, "Transcription for page_001.png"
                )
        finally:
            clear_abort()

        structured_model.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 14: plain-text sentinel exhaustion is a success, not a failure
# ---------------------------------------------------------------------------
def _make_plain_text_tm(**overrides: Any) -> TranscriptionManager:
    """Bare plain-text-mode TranscriptionManager bypassing __init__."""
    tm = TranscriptionManager.__new__(TranscriptionManager)
    caps = MagicMock()
    caps.use_plain_text_prompt = True
    defaults: dict[str, Any] = {
        "model_name": "gpt-5-mini",
        "provider": "openai",
        "custom_capabilities": caps,
        "_stats_lock": threading.Lock(),
        "successful_requests": 0,
        "failed_requests": 0,
        "processing_times": deque(maxlen=50),
        "rate_limiter": None,
        "model_config": {},
        "system_prompt": "sys",
        "_output_schema": None,
        "schema_retry_config": {
            "no_transcribable_text": {
                "enabled": True,
                # Deliberately larger than the shared attempt budget so the
                # loop runs out of iterations while retries remain allowed.
                "max_attempts": 99,
                "backoff_base": 0.0,
                "backoff_multiplier": 1.0,
            }
        },
    }
    defaults.update(overrides)
    for attr, val in defaults.items():
        setattr(tm, attr, val)
    return tm


def _wire_tm(tm: TranscriptionManager, extract: Any) -> None:
    """Stub the API boundary; *extract* is the return_value or side_effect."""
    tm._build_model_inputs = MagicMock(return_value=([], {}))  # type: ignore[method-assign]
    tm._get_structured_chat_model = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    tm._invoke_with_retry = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    tm._report_token_usage = MagicMock()  # type: ignore[method-assign]
    mock = MagicMock()
    if isinstance(extract, list):
        mock.side_effect = extract
    else:
        mock.return_value = extract
    tm._extract_output_text = mock  # type: ignore[method-assign]


def _payload(name: str = "page_001.png") -> Any:
    payload = MagicMock()
    payload.image_name = name
    payload.sequence_number = 1
    payload.base64 = "AAAA"
    return payload


class TestPlainTextSentinelExhaustion:
    """Plain-text mode has no schema, so exhaustion is a placeholder success."""

    def test_exhausted_sentinel_returns_success_placeholder(self) -> None:
        tm = _make_plain_text_tm()
        _wire_tm(tm, "[no transcribable text]")

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload(), max_schema_retries=2)

        assert "error" not in result
        assert "error_type" not in result
        assert result["transcription"] == "[page_001.png: no transcribable text]"
        assert result["schema_retries"]["no_transcribable_text"] > 0

    def test_json_mode_exhaustion_still_marks_the_page_failed(self) -> None:
        """Control: with a schema, unrecoverable exhaustion stays an error."""
        caps = MagicMock()
        caps.use_plain_text_prompt = False
        tm = _make_plain_text_tm(
            custom_capabilities=caps,
            schema_retry_config={
                "validation_failure": {
                    "enabled": True,
                    "max_attempts": 99,
                    "backoff_base": 0.0,
                    "backoff_multiplier": 1.0,
                }
            },
        )
        _wire_tm(tm, "not json at all")

        with patch("llm.transcription.time.sleep"):
            result = tm.transcribe_payload(_payload(), max_schema_retries=2)

        assert result["error_type"] == "schema_validation_exhausted"
