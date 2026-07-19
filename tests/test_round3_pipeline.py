"""Round-3 hardening regressions for the pipeline and imaging layers.

One focused test per fix:

1.  balance_dollar_signs no longer mangles postfix / space-separated currency.
2.  adjust_and_sort_page_numbers keeps an inferred number on the no-anchor path.
3.  Folder provenance carries an image-set identity the resume guard compares.
4.  A password-protected PDF aborts construction cleanly.
5.  A late append after finalize does not re-cache (leak) the log handle.
6.  Run progress/ETA denominator is this run's pending page count.
7.  A failed page reports its error_type (no phantom "retries").
8.  A hopeless budget page fails fast (stall decided before the reset wait).
10. The transcription token stamp uses the manager's resolved provider.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import pipeline.transcriber as transcriber_module
from config import app as app_config
from imaging.payload import FolderPayloadSource, PdfPayloadSource
from pipeline import log as log_mod
from pipeline.page_numbering import PageNumberProcessor
from pipeline.resume import _input_changed_since_log
from pipeline.text_cleaner import balance_dollar_signs
from pipeline.transcriber import ItemTranscriber


def _bare_transcriber() -> ItemTranscriber:
    """An ItemTranscriber shell (no __init__) for method-level tests."""
    return ItemTranscriber.__new__(ItemTranscriber)


# ---------------------------------------------------------------------------
# Fix 1: currency is not mistaken for unclosed inline math
# ---------------------------------------------------------------------------
class TestBalanceDollarCurrency:
    @pytest.mark.parametrize(
        "text",
        [
            "The price was 100$.",
            "It cost 100$ at the market.",
            "A price of $ 30 was normal.",
            "Er zahlte 12$ in bar.",
        ],
    )
    def test_currency_round_trips_unchanged(self, text: str) -> None:
        assert balance_dollar_signs(text) == text

    def test_genuine_unclosed_math_is_still_closed(self) -> None:
        out = balance_dollar_signs("The value $x^2 + 1 is large")
        assert out.count("$") == 2
        assert out.startswith("The value $x^2 + 1 is large")

    def test_prefix_price_still_untouched(self) -> None:
        assert balance_dollar_signs("It was $3 in 1850") == "It was $3 in 1850"


# ---------------------------------------------------------------------------
# Fix 2: inferred page number survives the no-anchor fallback branch
# ---------------------------------------------------------------------------
class TestInferredPageOnNoAnchorPath:
    @staticmethod
    def _summary(
        idx: int, page: int | None, ptype: str, section: str
    ) -> dict[str, Any]:
        return {
            "original_input_order_index": idx,
            "page_information": {
                "page_number_integer": page,
                "page_number_type": ptype,
                "page_types": [section],
            },
        }

    def test_inferred_number_not_discarded(self) -> None:
        # Page 5 (content) -> unnumbered page misclassified into a lone
        # "appendix" section -> page 7 (content). The middle page has no section
        # anchor (its section holds only itself, and it was unnumbered when
        # anchors were computed), so it takes the no-anchor branch. Inference
        # writes 6 into model_page_number_int; the branch must use it, not the
        # virtual-position fallback (which would label it 2).
        results = [
            self._summary(0, 5, "arabic", "content"),
            self._summary(1, None, "none", "appendix"),
            self._summary(2, 7, "arabic", "content"),
        ]
        adjusted = PageNumberProcessor().adjust_and_sort_page_numbers(results)
        by_idx = {r["original_input_order_index"]: r for r in adjusted}
        middle = by_idx[1]["page_information"]
        assert middle["page_number_integer"] == 6


# ---------------------------------------------------------------------------
# Fix 3: folder provenance identity + resume folder-change guard
# ---------------------------------------------------------------------------
class TestFolderProvenanceGuard:
    def _make_folder(self, tmp_path: Path, count: int) -> Path:
        from PIL import Image

        folder = tmp_path / "imgs"
        folder.mkdir(exist_ok=True)
        for i in range(count):
            Image.new("RGB", (20, 20), color=(i * 10, 0, 0)).save(
                folder / f"page_{i:03d}.jpg", "JPEG"
            )
        return folder

    def test_provenance_records_identity_fields(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_config_loader: MagicMock,
    ) -> None:
        monkeypatch.setattr(
            "imaging.payload.get_config_loader", lambda: mock_config_loader
        )
        folder = self._make_folder(tmp_path, 3)
        source = FolderPayloadSource(folder, provider="openai", model_name="gpt-5")
        prov = source.file_provenance()
        assert prov["image_count"] == 3
        expected = sum(p.stat().st_size for p in folder.glob("*.jpg"))
        assert prov["total_image_bytes"] == expected

    def test_resume_guard_detects_folder_change(self, tmp_path: Path) -> None:
        folder = self._make_folder(tmp_path, 2)
        total = sum(p.stat().st_size for p in folder.glob("*.jpg"))
        header = {
            "file_provenance": {
                "source_file": str(folder),
                "image_count": 2,
                "total_image_bytes": total,
            }
        }
        assert _input_changed_since_log(header) is False

        # Add a page: the count no longer matches -> changed.
        from PIL import Image

        Image.new("RGB", (20, 20), color=(9, 9, 9)).save(
            folder / "page_099.jpg", "JPEG"
        )
        assert _input_changed_since_log(header) is True

    def test_legacy_header_without_fields_not_changed(self, tmp_path: Path) -> None:
        # Backward compatibility: a header lacking the identity fields must not
        # be treated as changed (folder path present but no image_count).
        folder = self._make_folder(tmp_path, 1)
        header = {"file_provenance": {"source_file": str(folder)}}
        assert _input_changed_since_log(header) is False


# ---------------------------------------------------------------------------
# Fix 4: password-protected PDF aborts construction
# ---------------------------------------------------------------------------
class TestEncryptedPdfAborts:
    def test_needs_pass_raises_valueerror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_config_loader: MagicMock,
    ) -> None:
        import fitz

        monkeypatch.setattr(
            "imaging.payload.get_config_loader", lambda: mock_config_loader
        )
        pdf_path = tmp_path / "locked.pdf"
        doc = fitz.open()
        doc.new_page(width=200, height=300)
        doc.save(
            pdf_path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
        doc.close()

        with pytest.raises(ValueError, match="password-protected"):
            PdfPayloadSource(pdf_path, provider="openai", model_name="gpt-5")


# ---------------------------------------------------------------------------
# Fix 5: a late append after finalize does not leak (re-cache) the handle
# ---------------------------------------------------------------------------
class TestLogHandleNoLeakAfterFinalize:
    def test_one_shot_append_after_finalize(self, tmp_path: Path) -> None:
        path = tmp_path / "work.jsonl"
        try:
            assert log_mod.append_to_log(path, {"a": 1}) is True
            assert path in log_mod._LOG_HANDLES  # cached live handle

            assert log_mod.finalize_log_file(path) is True
            assert path not in log_mod._LOG_HANDLES  # closed
            assert path in log_mod._FINALIZED_LOGS

            # Late worker append: written but NOT re-cached (no fd leak).
            assert log_mod.append_to_log(path, {"b": 2}) is True
            assert path not in log_mod._LOG_HANDLES

            lines = path.read_text(encoding="utf-8").splitlines()
            assert lines == ['{"a": 1}', '{"b": 2}']
        finally:
            log_mod.finalize_log_file(path)
            with log_mod._LOG_HANDLES_GUARD:
                log_mod._FINALIZED_LOGS.discard(path)


# ---------------------------------------------------------------------------
# Fixes 6/7/8: transcriber run-loop behaviour
# ---------------------------------------------------------------------------
class _FakeSource:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


class TestProgressAndBudgetLoop:
    def test_progress_denominator_is_pending_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        obj.summary_manager = None
        obj.completed_page_indices = {0, 1}
        obj.total_items_to_transcribe = 0

        # Pretend pages 0 and 1 were completed on a prior run.
        def fake_reload(
            t_results: list[dict[str, Any]], _s: list[dict[str, Any]]
        ) -> None:
            t_results.extend({"original_input_order_index": i} for i in (0, 1))

        obj._reload_completed_pages = fake_reload  # type: ignore[method-assign,assignment]

        captured: list[tuple[int, int]] = []

        def spy(idx, source, t, s, progress_total, count_ref, already_complete=0):
            captured.append((progress_total, already_complete))
            t.append({"original_input_order_index": idx})
            return {"original_input_order_index": idx}

        obj._process_single_page = spy  # type: ignore[method-assign]
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber_module, "get_transcription_concurrency", lambda: (2, None)
        )

        obj._transcribe_and_summarize(_FakeSource(5))  # type: ignore[arg-type]

        # 5 total, 2 already complete -> 3 pending pages this run.
        assert captured, "spy was never called"
        assert all(total == 3 for total, _ in captured)
        assert all(done == 2 for _, done in captured)

    def test_failed_page_reports_error_type(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from llm.token_tracker import DailyTokenTracker

        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        obj._count_lock = threading.Lock()
        obj._token_tracker = DailyTokenTracker(
            daily_limit=1, enabled=False, state_file=tmp_path / "s.json"
        )
        obj.transcription_times = []
        obj.transcription_provider = "openai"
        obj.summary_manager = None
        obj.log_path = tmp_path / "log.jsonl"
        obj.summary_log_path = tmp_path / "slog.jsonl"
        obj.start_time_processing = 0.0
        obj._transcription_stamp = {}
        obj._summary_stamp = {}

        mgr = MagicMock()
        mgr.transcribe_payload.return_value = {
            "image": "page_0001.jpg",
            "transcription": "[transcription error]",
            "processing_time": 0.01,
            "error": "API error after retries",
            "error_type": "api_failure",
            "schema_retries": {"gpt-5": 2},
            "provider": "openai",
        }
        obj.transcribe_manager = mgr

        class _Src:
            def image_name(self, idx: int) -> str:
                return "page_0001.jpg"

            def build_payload(self, idx: int) -> Any:
                from types import SimpleNamespace

                return SimpleNamespace(source_file="f", provenance={}, page_index=None)

        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(transcriber_module, "append_to_log", lambda *a, **k: None)

        with caplog.at_level(logging.DEBUG, logger="pipeline.transcriber"):
            obj._process_single_page(0, _Src(), [], [], 1, [0])  # type: ignore[arg-type]

        text = caplog.text
        assert "FAILED (api_failure" in text
        assert "2 schema retries" in text
        assert "0 retries" not in text

    def test_hopeless_budget_page_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj = _bare_transcriber()
        obj._budget_exhausted = threading.Event()
        obj.summary_manager = None
        obj.completed_page_indices = set()
        obj.total_items_to_transcribe = 0
        obj._transcription_stamp = {}
        obj._summary_stamp = {}

        # Every page always defers: no progress is ever made.
        def always_defer(idx, *a, **k):
            obj._budget_exhausted.set()
            return None

        obj._process_single_page = always_defer  # type: ignore[method-assign,assignment]

        wait_calls = [0]

        def fake_wait(**_k) -> bool:
            wait_calls[0] += 1
            return True  # simulate a successful daily reset each time

        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)
        monkeypatch.setattr(
            transcriber_module, "get_transcription_concurrency", lambda: (2, None)
        )
        monkeypatch.setattr(transcriber_module, "wait_for_token_reset", fake_wait)

        obj._transcribe_and_summarize(_FakeSource(4))  # type: ignore[arg-type]

        # The stall verdict is reached BEFORE the second reset wait, so a
        # hopeless page waits at most once (previously twice, ~48 h).
        assert wait_calls[0] == 1


# ---------------------------------------------------------------------------
# Fix 10: transcription stamp uses the manager's resolved provider
# ---------------------------------------------------------------------------
class TestTranscriptionStampProvider:
    def test_stamp_reads_resolved_provider(
        self,
        make_pdf: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_config_loader: MagicMock,
    ) -> None:
        # model.yaml with provider: null for transcription; the manager resolves
        # it to a real provider. The stamp must follow the manager, not the raw
        # (None) config value that would land in the UNATTRIBUTED bucket.
        mock_config_loader.get_model_config.return_value = {
            "transcription_model": {"name": "gpt-5-mini", "provider": None},
            "summary_model": {"name": "gpt-5-mini", "provider": "openai"},
        }
        mgr = MagicMock()
        mgr.provider = "openai"
        mgr.key_env = "OPENAI_API_KEY"
        monkeypatch.setattr(
            transcriber_module, "TranscriptionManager", MagicMock(return_value=mgr)
        )
        monkeypatch.setattr(
            transcriber_module, "get_config_loader", lambda: mock_config_loader
        )
        monkeypatch.setattr(app_config, "SUMMARIZE", False, raising=False)

        pdf = make_pdf("Stamp.pdf", 1)
        obj = ItemTranscriber(
            input_path=pdf,
            input_type="pdf",
            base_output_dir=tmp_path / "out",
        )
        assert obj._transcription_stamp["provider"] == "openai"
        assert obj._transcription_stamp["provider"] is not None
