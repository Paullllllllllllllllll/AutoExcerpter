from __future__ import annotations

import concurrent.futures
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import app as config
from config.accessors import (
    get_api_timeout,
    get_transcription_concurrency,
)
from config.constants import (
    DEFAULT_MODEL,
    ETA_BLEND_WEIGHT_OVERALL,
    ETA_BLEND_WEIGHT_RECENT,
    MIN_SAMPLES_FOR_ETA,
    RECENT_SAMPLES_FOR_ETA,
    is_blank_transcription,
)
from config.loader import get_config_loader
from config.logger import setup_logger
from imaging.payload import FolderPayloadSource, PagePayload, PdfPayloadSource
from llm import SummaryManager, TranscriptionManager
from llm.token_tracker import get_token_tracker, wait_for_token_reset
from llm.types import CustomEndpointCapabilities
from pipeline.context import format_context_for_prompt, resolve_summary_context
from pipeline.log import append_to_log, finalize_log_file, initialize_log_file
from pipeline.page_numbering import PageNumberProcessor
from pipeline.paths import create_safe_directory_name, create_safe_log_filename
from pipeline.resume import (
    _header_from_entries,
    _read_and_parse_log,
    _results_from_entries,
    load_log_header,
    load_transcription_results_from_log,
)
from pipeline.text_cleaner import clean_transcription
from rendering import (
    create_docx_summary,
    create_markdown_summary,
    write_transcription_to_text,
)
from rendering.citations import enrich_if_enabled
from rendering.summary import build_render_context

logger = setup_logger(__name__)


class ItemTranscriber:
    """Process a single input item (PDF or image folder).

    Attributes:
        input_path: Source path of the item to process.
        input_type: Either "pdf" or "image_folder".
        base_output_dir: Base directory where outputs and working files are written.
        working_dir: Item-specific working directory containing logs.
        transcribe_manager: Manages image transcription via LLM API.
        summary_manager: Manages summarization via LLM API (if enabled).
        summary_context: Optional context string for guiding summarization focus.
    """

    def __init__(
        self,
        input_path: Path,
        input_type: str,
        base_output_dir: Path,
        summary_context: str | None = None,
        resume_mode: str = "skip",
        completed_page_indices: set[int] | None = None,
        prior_transcription_results: list[dict[str, Any]] | None = None,
        prior_summary_results: list[dict[str, Any]] | None = None,
        logged_log_header: dict[str, Any] | None = None,
    ) -> None:
        self.input_path = input_path
        self.input_type = input_type  # "pdf" or "image_folder"
        self.name = self.input_path.stem
        self.resume_mode = resume_mode
        self.completed_page_indices = completed_page_indices or set()
        # Working-log data already parsed by ResumeChecker during the resume
        # check, threaded in so process_item need not re-read the same files.
        # Any left as None falls back to a disk read (see process_item).
        self._resume_transcription_results = prior_transcription_results
        self._resume_summary_results = prior_summary_results
        self._resume_log_header = logged_log_header

        # Page-level token-budget gate state, consulted by _process_single_page
        # and driven by the re-pass loop in _transcribe_and_summarize.
        self._token_tracker = get_token_tracker()
        self._budget_exhausted = threading.Event()
        # Guards the shared processed-page counter against lost-update races
        # between worker threads (a bare ``+= 1`` is not atomic).
        self._count_lock = threading.Lock()

        self.base_output_dir = base_output_dir
        self.output_txt_path = self.base_output_dir / f"{self.name}.txt"
        self.output_summary_docx_path = self.base_output_dir / f"{self.name}.docx"
        self.output_summary_md_path = self.base_output_dir / f"{self.name}.md"

        # Item-specific working directory for logs and temporary images
        # Use safe directory name to avoid Windows MAX_PATH (260 char) limitations
        safe_working_dir_name = create_safe_directory_name(self.name, "_working_files")
        self.working_dir = self.base_output_dir / safe_working_dir_name
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Use safe log filenames to avoid path length issues
        safe_transcription_log_name = create_safe_log_filename(
            self.name, "transcription"
        )
        safe_summary_log_name = create_safe_log_filename(self.name, "summary")
        self.log_path = self.working_dir / safe_transcription_log_name
        self.summary_log_path = self.working_dir / safe_summary_log_name

        self.total_items_to_transcribe = 0
        self.start_time_processing: float | None = None
        self.transcription_times: list[float] = []  # For successful transcriptions

        # Snapshot of prior page results for page-level resume, loaded by
        # process_item before the log files are reinitialized (truncated).
        self._prior_transcription_results: list[dict[str, Any]] = []
        self._prior_summary_results: list[dict[str, Any]] = []
        # Extra metadata lines recorded in the .txt output (e.g. a resume model
        # mismatch: logged transcriptions produced by a different model).
        self._output_metadata_notes: list[str] = []

        # Absolute paths of output files actually written by process_item
        # (transcription .txt plus summary .docx/.md), consumed by the CLI
        # loop for the --json run summary.
        self.written_outputs: list[Path] = []

        # Per-item run report populated at the end of process_item (normal,
        # partial, and failure verdicts). Consumed by the CLI overview layer;
        # remains None when the item aborts before any stats are computed.
        self.last_run_report: dict[str, Any] | None = None

        # Load model configuration from model.yaml
        config_loader = get_config_loader()
        model_cfg = config_loader.get_model_config()

        # Get transcription model configuration (centralized in model.yaml)
        transcription_cfg = model_cfg.get("transcription_model", {})
        self.transcription_model = transcription_cfg.get("name", DEFAULT_MODEL)
        self.transcription_provider = transcription_cfg.get("provider", "openai")

        # Get summary model configuration (centralized in model.yaml)
        summary_cfg = model_cfg.get("summary_model", {})
        self.summary_model = summary_cfg.get("name", DEFAULT_MODEL)
        summary_provider = summary_cfg.get("provider")

        # Load custom endpoint capabilities (if applicable)
        transcription_custom_caps: CustomEndpointCapabilities | None = None
        if self.transcription_provider == "custom":
            custom_endpoint_cfg = transcription_cfg.get("custom_endpoint", {})
            caps_dict = custom_endpoint_cfg.get("capabilities", {})
            transcription_custom_caps = CustomEndpointCapabilities.from_dict(caps_dict)
            logger.info(
                "Custom transcription endpoint: "
                "structured_output="
                f"{transcription_custom_caps.supports_structured_output}, "
                f"plain_text={transcription_custom_caps.use_plain_text_prompt}, "
                f"vision={transcription_custom_caps.supports_vision}"
            )

        summary_custom_caps: CustomEndpointCapabilities | None = None
        if summary_provider == "custom":
            summary_endpoint_cfg = summary_cfg.get("custom_endpoint", {})
            summary_caps_dict = summary_endpoint_cfg.get("capabilities", {})
            summary_custom_caps = CustomEndpointCapabilities.from_dict(
                summary_caps_dict
            )

        transcription_was_plain_text = (
            transcription_custom_caps is not None
            and transcription_custom_caps.use_plain_text_prompt
        )

        # Share ONE rate limiter per provider across transcription and the
        # inline summary phase so their combined request rate honors the
        # configured caps (previously each manager built its own limiter,
        # doubling the effective rate for a same-provider setup).
        from llm.client import get_provider_for_model
        from llm.rate_limit import get_shared_rate_limiter

        transcription_rate_limiter = get_shared_rate_limiter(
            self.transcription_provider
        )
        self.transcribe_manager = TranscriptionManager(
            model_name=self.transcription_model,
            provider=self.transcription_provider,
            timeout=get_api_timeout(),
            custom_capabilities=transcription_custom_caps,
            rate_limiter=transcription_rate_limiter,
        )

        # Only initialize summary manager if summarization is enabled
        self.summary_manager = None
        self.summary_context = None
        if config.SUMMARIZE:
            # Resolve summary context: CLI/interactive context takes precedence,
            # then file-specific, folder-specific, or general context
            if summary_context:
                self.summary_context = summary_context
                logger.info(
                    f"Using user-provided summary context: {summary_context[:50]}..."
                    if len(summary_context) > 50
                    else f"Using user-provided summary context: {summary_context}"
                )
            else:
                # Try hierarchical context resolution
                resolved_context, context_path = resolve_summary_context(
                    input_file=input_path
                )
                if resolved_context:
                    self.summary_context = format_context_for_prompt(resolved_context)
                    logger.info(f"Resolved summary context from: {context_path}")

            resolved_summary_provider = summary_provider or get_provider_for_model(
                self.summary_model
            )
            summary_rate_limiter = (
                transcription_rate_limiter
                if resolved_summary_provider == self.transcription_provider
                else get_shared_rate_limiter(resolved_summary_provider)
            )
            self.summary_manager = SummaryManager(
                model_name=self.summary_model,
                provider=summary_provider,
                summary_context=self.summary_context,
                custom_capabilities=summary_custom_caps,
                transcription_was_plain_text=transcription_was_plain_text,
                rate_limiter=summary_rate_limiter,
            )

        # Per-role token-tracker stamps (provider, key_env NAME, model) so each
        # role's usage and reservations land on their own per-key-pool bucket.
        # Resolved once here from the constructed managers; a free/local
        # transcription endpoint (pool None) is never blocked by a paid summary
        # key's exhaustion because they occupy independent buckets.
        self._transcription_stamp: dict[str, str | None] = {
            # Use the manager's RESOLVED provider (mirrors the summary stamp
            # below): with ``provider: null`` in model.yaml the raw config value
            # is None, which would land reservations in the UNATTRIBUTED bucket
            # while commits land in the real one.
            "provider": getattr(self.transcribe_manager, "provider", None),
            "key_env": getattr(self.transcribe_manager, "key_env", None),
            "model": self.transcription_model,
        }
        self._summary_stamp: dict[str, str | None] = (
            {
                "provider": getattr(self.summary_manager, "provider", None),
                "key_env": getattr(self.summary_manager, "key_env", None),
                "model": self.summary_model,
            }
            if self.summary_manager is not None
            else {}
        )

        # Page number processor for adjusting and sorting summary page numbers
        self.page_number_processor = PageNumberProcessor()

        # Image preprocessing is handled within imaging.preprocessing inside the
        # transcription manager.

    def _create_payload_source(self) -> PdfPayloadSource | FolderPayloadSource:
        """Create the lazy in-memory payload source for this item."""
        if self.input_type == "pdf":
            return PdfPayloadSource(
                self.input_path,
                provider=self.transcription_provider,
                model_name=self.transcription_model,
            )
        return FolderPayloadSource(
            self.input_path,
            provider=self.transcription_provider,
            model_name=self.transcription_model,
        )

    def _build_summary_result(
        self,
        original_index: int,
        image_name: str,
        summary_payload: dict[str, Any],
        page_number: int | None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Create a consistent flat summary result structure for downstream consumers.

        Merges API result fields with metadata at the same level (no nesting).
        """
        # Start with metadata fields
        result = {
            "original_input_order_index": original_index,
            "image_filename": image_name,
        }

        # Add page number (prefer explicit, then from payload)
        if page_number is not None:
            result["page"] = page_number
        elif isinstance(summary_payload, dict) and "page" in summary_payload:
            result["page"] = summary_payload.get("page")

        # Merge content fields from summary_payload at top level
        if isinstance(summary_payload, dict):
            # Core content fields from API
            result["page_information"] = summary_payload.get("page_information")
            result["bullet_points"] = summary_payload.get("bullet_points")
            result["references"] = summary_payload.get("references")

            # Metadata fields from API
            if "processing_time" in summary_payload:
                result["processing_time"] = summary_payload["processing_time"]
            if "provider" in summary_payload:
                result["provider"] = summary_payload["provider"]
            if "api_response" in summary_payload:
                result["api_response"] = summary_payload["api_response"]
            if "schema_retries" in summary_payload:
                result["schema_retries"] = summary_payload["schema_retries"]

        if error_message:
            result["error"] = error_message

        return result

    def _create_placeholder_summary(
        self,
        page_number: int | None,
        page_number_type: str,
        bullet_points: list[str] | None,
        references: list[str] | None = None,
        page_types: list[str] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Create a flat placeholder summary structure."""
        if page_types is None:
            page_types = ["other"]

        # Flat structure - all fields at top level
        payload = {
            "page": page_number if page_number is not None else 0,
            "page_information": {
                "page_number_integer": page_number,
                "is_two_page_spread": False,
                "page_number_integer_end": None,
                "page_number_type": page_number_type,
                "page_types": page_types,
            },
            "bullet_points": bullet_points,
            "references": references,
        }
        if error_message:
            payload["error"] = error_message
        return payload

    @staticmethod
    def _compute_page_num(page_num_model: Any, fallback_index: int) -> tuple[bool, int]:
        """Return (has_valid_model_page_num, page_num_to_use)."""
        has_valid = isinstance(page_num_model, int)
        return has_valid, page_num_model if has_valid else fallback_index + 1

    def _summarize_transcription(
        self,
        transcription_result: dict[str, Any],
        original_index: int,
        img_name: str,
    ) -> dict[str, Any] | None:
        """Generate or create placeholder summary for a single transcription.

        Returns the summary result dict, or None if summarization is disabled.
        """
        if not config.SUMMARIZE or not self.summary_manager:
            return None

        page_num_model = transcription_result.get("page")
        has_valid, page_num_to_use = self._compute_page_num(
            page_num_model, original_index
        )

        if "error" not in transcription_result:
            transcription_text = transcription_result.get("transcription", "")

            if is_blank_transcription(transcription_text):
                summary_data = self._create_placeholder_summary(
                    page_num_to_use if has_valid else None,
                    "arabic" if has_valid else "none",
                    None,
                    references=None,
                    page_types=["blank"],
                )
            else:
                summary_data = self.summary_manager.generate_summary(
                    transcription_text, page_num_to_use
                )
            summary_error = (
                summary_data.get("error") if isinstance(summary_data, dict) else None
            )
        else:
            error_msg = transcription_result.get("error", "Unknown error")
            summary_data = self._create_placeholder_summary(
                page_num_to_use if has_valid else None,
                "arabic" if has_valid else "none",
                [f"[Transcription failed: {error_msg}]"],
                references=None,
                page_types=["other"],
                error_message=error_msg,
            )
            summary_error = None

        return self._build_summary_result(
            original_index,
            img_name,
            summary_data,
            page_num_to_use,
            summary_error,
        )

    def _stamps_share_bucket(
        self,
        stamp_a: dict[str, str | None],
        stamp_b: dict[str, str | None],
    ) -> bool:
        """Whether two token-tracker stamps resolve to the same accounting bucket.

        Two roles on the same key and pool must not double-reserve one page,
        while distinct buckets must each pass their own admission gate.
        """
        resolve = self._token_tracker._bucket_for
        return resolve(
            stamp_a.get("provider"), stamp_a.get("key_env"), stamp_a.get("model")
        ) == resolve(
            stamp_b.get("provider"), stamp_b.get("key_env"), stamp_b.get("model")
        )

    def _process_single_page(
        self,
        original_input_order_index: int,
        source: PdfPayloadSource | FolderPayloadSource,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
        progress_total: int,
        processed_count_ref: list[int],
        already_complete: int = 0,
    ) -> dict[str, Any] | None:
        """Render/load, transcribe, and optionally summarize a single page.

        Appends results to the shared lists and increments the processed counter.
        Returns None without touching the shared lists or logs when the page is
        deferred by the token-budget gate, so the resume path re-runs it later.

        *progress_total* is the number of pending pages for THIS run (not the
        source page count), so progress and ETA are not skewed by pages already
        completed on a prior run; *already_complete* is that prior-run count,
        reported in the progress line for context.
        """
        # Whole-page budget gate: reserve a combined transcription + summary
        # estimate up front. If it cannot fit the remaining daily budget, defer
        # the entire page (no API call, no log, no result) so resume re-runs it
        # after the daily reset. try_reserve returns 0 when limiting is disabled.
        if self._budget_exhausted.is_set():
            return None
        # Stamp the page-level reservation with the TRANSCRIPTION key-pool bucket
        # (the page's primary call). A free/local transcription endpoint (pool
        # None) is never blocked here even when a paid summary key is exhausted;
        # the summary call's own usage is stamped and accounted separately on
        # its own bucket (see _summarize_transcription -> summary manager).
        trans_stamp = getattr(self, "_transcription_stamp", {})
        reserved = self._token_tracker.try_reserve(**trans_stamp)
        if reserved is None:
            self._budget_exhausted.set()
            return None
        # The summary call commits its usage to the SUMMARY bucket, so when
        # that bucket differs from the transcription one its own gates (per-key
        # pool cap, combined budget) must admit the page too -- otherwise the
        # summary key's cap would never be enforced on the fresh-page path.
        summ_stamp = getattr(self, "_summary_stamp", {})
        summ_reserved: int | None = None
        if summ_stamp and not self._stamps_share_bucket(trans_stamp, summ_stamp):
            summ_reserved = self._token_tracker.try_reserve(**summ_stamp)
            if summ_reserved is None:
                self._token_tracker.release(reserved, **trans_stamp)
                self._budget_exhausted.set()
                return None

        image_name = f"page index {original_input_order_index}"
        try:
            image_name = source.image_name(original_input_order_index)

            payload: PagePayload | None = None
            try:
                payload = source.build_payload(original_input_order_index)
            except Exception as e:
                # Render/preprocess failure: record an error page, skip the API
                logger.error(f"Error preparing page {image_name}: {e}")
                transcription_result: dict[str, Any] = {
                    "image": image_name,
                    "sequence_number": original_input_order_index + 1,
                    "transcription": f"[preprocessing error: {e}]",
                    "processing_time": 0.0,
                    "error": str(e),
                    "error_type": "preprocessing_failure",
                    "provider": self.transcription_provider,
                    "original_input_order_index": original_input_order_index,
                }
            else:
                transcription_result = {
                    **self.transcribe_manager.transcribe_payload(payload),
                    "original_input_order_index": original_input_order_index,
                    "source_file": payload.source_file,
                    "image_provenance": payload.provenance,
                }
                if payload.page_index is not None:
                    transcription_result["page_index"] = payload.page_index

            if "error" not in transcription_result:
                raw_text = transcription_result.get("transcription", "")
                transcription_result["transcription"] = clean_transcription(raw_text)

            summary_result = self._summarize_transcription(
                transcription_result, original_input_order_index, image_name
            )
            if summary_result is not None:
                append_to_log(self.summary_log_path, summary_result)
                summary_results.append(summary_result)

            append_to_log(self.log_path, transcription_result)
            # Lock-guarded increment + read: worker threads otherwise lose
            # updates on a bare ``+= 1`` (read-modify-write is not atomic).
            with self._count_lock:
                processed_count_ref[0] += 1
                processed_now = processed_count_ref[0]

            if (
                "processing_time" in transcription_result
                and "error" not in transcription_result
            ):
                self.transcription_times.append(transcription_result["processing_time"])

            if "error" not in transcription_result:
                status = "SUCCESS"
            else:
                # Report what producers actually set: the error_type and, when
                # present, the schema-validation retry count (a per-model dict).
                # No producer ever set "retries", so the old read always
                # reported "0 retries".
                error_type = transcription_result.get("error_type", "unknown")
                schema_retries = transcription_result.get("schema_retries")
                if isinstance(schema_retries, dict) and schema_retries:
                    total_schema_retries = sum(schema_retries.values())
                    status = (
                        f"FAILED ({error_type}, {total_schema_retries} schema retries)"
                    )
                else:
                    status = f"FAILED ({error_type})"
            item_num_str = str(original_input_order_index + 1)
            eta_str = self._calculate_eta(processed_now, progress_total)

            logger.debug(
                f"Processed {processed_now}/{progress_total} pending page(s) "
                f"({already_complete} already complete) "
                f"- Item {item_num_str} - Status: {status} - {eta_str}"
            )
            transcription_results.append(transcription_result)
            return transcription_result

        except Exception as e:
            logger.exception(f"Critical error during task for {image_name}: {e}")
            seq_num = original_input_order_index + 1
            error_result: dict[str, Any] = {
                "page": seq_num,
                "image": image_name,
                "transcription": f"[CRITICAL ERROR] Unhandled in task: {e}",
                "error": str(e),
                "original_input_order_index": original_input_order_index,
            }
            transcription_results.append(error_result)
            with self._count_lock:
                processed_count_ref[0] += 1
            return error_result
        finally:
            # Release the reservations (same stamps as the reserves); actual
            # usage was committed via add_tokens inside the provider layer
            # during the calls above.
            self._token_tracker.release(reserved, **trans_stamp)
            if summ_reserved:
                self._token_tracker.release(summ_reserved, **summ_stamp)
            # Feed this page's total actual usage (transcription + summary +
            # retries, accumulated per-thread by add_tokens) to the reservation
            # EWMA as ONE per-page observation, so the gate reserves per-page
            # cost rather than per-call cost.
            self._token_tracker.record_page_usage()

    def _detect_resume_model_mismatch(self) -> None:
        """Warn (and record) when reusing transcriptions from a different model.

        Summary-only resume reuses logged transcriptions rather than re-running
        the vision model. If the logged transcription model differs from the
        currently configured one, the reused text and any freshly generated
        summaries were produced by different models; surface that so the mixed
        provenance is visible in both the log and the .txt metadata.
        """
        header = self._resume_log_header
        if header is None:
            header = load_log_header(self.log_path)
        logged_model = header.get("model_name") if isinstance(header, dict) else None
        if logged_model and logged_model != self.transcription_model:
            note = (
                "Summary-only resume: reusing transcriptions from model "
                f"'{logged_model}'; current transcription model is "
                f"'{self.transcription_model}'."
            )
            logger.warning(note)
            self._output_metadata_notes.append(note)

    def _initialize_log_or_raise(
        self,
        log_path: Path,
        item_name: str,
        input_path: str,
        input_type: str,
        total_images: int,
        model_name: str,
        extraction_dpi: int | None = None,
        *,
        concurrency_limit: int | None = None,
        file_provenance: dict[str, Any] | None = None,
        log_type: str = "transcription",
    ) -> None:
        """Write the log header, retrying once, and raise if it cannot be written.

        A transient header-write failure would otherwise be swallowed, leaving a
        headerless log that a later resume refuses wholesale (the item's
        completed pages become unrecoverable). Raising instead lets the caller
        (``_process_single_item``) convert it into a clean item-level failure so
        the working log is preserved for a genuine retry.
        """
        for _attempt in range(2):
            if initialize_log_file(
                log_path,
                item_name,
                input_path,
                input_type,
                total_images,
                model_name,
                extraction_dpi,
                concurrency_limit=concurrency_limit,
                file_provenance=file_provenance,
                log_type=log_type,
            ):
                return
        raise RuntimeError(
            f"Could not initialize {log_type} log header at {log_path} "
            "after a retry; aborting this item to protect the working log."
        )

    def _reload_completed_pages(
        self,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
    ) -> None:
        """Reload completed pages' transcriptions and (re)attach their summaries.

        Runs in TWO passes over the same eligible-entry list so the resume path
        never opens a data-loss window:

        Pass 1 immediately re-appends EVERY eligible completed transcription to
        the freshly initialized transcription log (``initialize_log_file`` just
        truncated it, and the log was these pages' only on-disk copy). This is
        pure file I/O with no LLM calls, so the truncate-then-reappend window is
        closed in milliseconds — a crash or Ctrl+C during the (potentially slow)
        summary work in pass 2 can no longer destroy a completed transcription.

        Pass 2 admits each completed page to the in-memory results and handles
        its summary. It does NOT touch the transcription log again (pass 1
        already persisted every eligible entry). If summarizing, it reuses the
        logged summary when present AND error-free; otherwise it regenerates it
        from the logged transcription text (the summary-only resume path) —
        never dropping a completed page's summary, and never reusing a summary
        that previously errored (AE-2, so a re-run repairs it).

        Regenerating a summary calls the LLM, so it is gated by the same daily
        token budget as ``_process_single_page`` (AE-6): if the remaining budget
        cannot cover a page, the WHOLE page is deferred — its completed
        transcription is already re-logged (pass 1) so a later run can resume it
        without re-transcribing, but nothing is appended to the in-memory
        results, so ``pages_deferred`` withholds the truncated outputs and a
        later run finishes it — rather than blowing past the daily cap on a
        large resume.
        """
        prior_results = self._prior_transcription_results or (
            load_transcription_results_from_log(self.log_path) or []
        )
        prior_summary_by_idx: dict[int, dict[str, Any]] = {}
        for summ in self._prior_summary_results:
            idx = summ.get("original_input_order_index")
            if isinstance(idx, int):
                prior_summary_by_idx[idx] = summ

        summarizing = bool(config.SUMMARIZE and self.summary_manager)

        # Eligibility filter computed once and shared by both passes: a real
        # per-page index, marked completed, and within the current page count
        # (a stale phantom index from a swapped, shorter input is dropped).
        eligible = [
            entry
            for entry in prior_results
            if isinstance(entry.get("original_input_order_index"), int)
            and entry["original_input_order_index"] in self.completed_page_indices
            and entry["original_input_order_index"] < self.total_items_to_transcribe
        ]

        # --- Pass 1: persist every completed transcription NOW (no LLM calls),
        # closing the truncate-then-reappend loss window. ---
        for entry in eligible:
            append_to_log(self.log_path, entry)

        # --- Pass 2: reuse/regenerate summaries and admit pages to results.
        # The transcription log is intentionally NOT written here again. ---
        for entry in eligible:
            idx = entry["original_input_order_index"]

            if not summarizing:
                transcription_results.append(entry)
                continue

            # Reuse a logged summary only when present and error-free; anything
            # missing or previously errored is regenerated.
            summary_result = prior_summary_by_idx.get(idx)
            needs_generation = summary_result is None or (
                isinstance(summary_result, dict) and "error" in summary_result
            )

            if needs_generation:
                # Token-budget gate, mirroring _process_single_page: defer the
                # whole page (append nothing to results) when the daily budget
                # cannot cover it. The transcription is already re-logged by
                # pass 1, so a later run resumes it without re-transcribing.
                if self._budget_exhausted.is_set():
                    continue
                # Summary-only regeneration: stamp the SUMMARY key-pool bucket
                # (no transcription call happens on this path).
                summ_stamp = getattr(self, "_summary_stamp", {})
                reserved = self._token_tracker.try_reserve(**summ_stamp)
                if reserved is None:
                    self._budget_exhausted.set()
                    continue
                try:
                    image_name = entry.get("image") or f"page index {idx}"
                    summary_result = self._summarize_transcription(
                        entry, idx, image_name
                    )
                finally:
                    self._token_tracker.release(reserved, **summ_stamp)
                    self._token_tracker.record_page_usage()

            transcription_results.append(entry)
            if summary_result is not None:
                append_to_log(self.summary_log_path, summary_result)
                summary_results.append(summary_result)

    def _transcribe_and_summarize(
        self, source: PdfPayloadSource | FolderPayloadSource
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        transcription_results: list[dict[str, Any]] = []
        summary_results: list[dict[str, Any]] = []
        total_images = len(source)
        # Keep the current page count on the instance so the phantom-index
        # guard in _reload_completed_pages is correct even when this method is
        # entered without _run_processing having set it first.
        self.total_items_to_transcribe = total_images

        # Initialize the summary log ONCE up front (if summarizing) so that both
        # reused and freshly generated summaries are appended to the same file.
        if config.SUMMARIZE and self.summary_manager:
            max_workers, _ = get_transcription_concurrency()
            self._initialize_log_or_raise(
                self.summary_log_path,
                self.name,
                str(self.input_path),
                "PDF" if self.input_type == "pdf" else "Image Folder",
                total_images,
                self.summary_model,
                concurrency_limit=max_workers,
                log_type="summary",
            )

        # --- Page-level resume: reload completed transcriptions, reuse their
        # summaries where logged, and regenerate summaries for completed pages
        # that lack one (the summary-only / TRANSCRIPTION_ONLY resume path).
        # This runs BEFORE any page is rendered so no completed page is lost. ---
        pending_indices = list(range(total_images))
        skipped_page_count = 0

        if self.completed_page_indices:
            self._reload_completed_pages(transcription_results, summary_results)
            pending_indices = [
                idx for idx in pending_indices if idx not in self.completed_page_indices
            ]
            skipped_page_count = total_images - len(pending_indices)

            if skipped_page_count > 0:
                logger.info(
                    f"Page-level resume: {skipped_page_count} page(s) already "
                    f"transcribed, {len(pending_indices)} page(s) remaining"
                )

        if not pending_indices:
            logger.info(
                "All pages already transcribed (page-level resume). "
                "Skipping transcription; summaries reused/regenerated from logs."
            )
            transcription_results.sort(
                key=lambda x: x.get("original_input_order_index", 0)
            )
            return transcription_results, summary_results

        suffix = " and summarization" if config.SUMMARIZE else ""
        resume_note = (
            f" ({skipped_page_count} skipped via resume)" if skipped_page_count else ""
        )
        logger.info(
            f"Starting transcription{suffix} of "
            f"{len(pending_indices)} images{resume_note}..."
        )

        # Mutable counter shared across threads (replaces nonlocal)
        processed_count_ref = [0]

        # One ThreadPoolExecutor per ITEM (sized once from config), reused
        # across budget re-passes so exhaustion + reset does not churn a fresh
        # pool each pass. Sized to the initial pending count; later, smaller
        # passes simply leave surplus workers idle.
        max_workers, _ = get_transcription_concurrency()
        max_workers = min(max_workers, len(pending_indices))
        if max_workers <= 0:
            max_workers = 1
        logger.info(f"Using {max_workers} concurrent workers for transcription")

        # Page-level token-budget loop: each pass submits the pending pages and
        # consumes results in COMPLETION order (so tqdm reflects real progress);
        # the gate in _process_single_page sets _budget_exhausted and defers the
        # rest once the daily budget cannot fit a page. Deferred pages write no
        # log/result, so resume re-runs exactly them. On KeyboardInterrupt or a
        # fatal error the pool is shut down with cancel_futures=True so queued
        # pages are cancelled rather than draining for up to api_timeout each.
        pending = list(pending_indices)
        stalled_resets = 0
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            while pending:
                self._budget_exhausted.clear()

                futures = [
                    executor.submit(
                        self._process_single_page,
                        idx,
                        source,
                        transcription_results,
                        summary_results,
                        len(pending_indices),
                        processed_count_ref,
                        already_complete=skipped_page_count,
                    )
                    for idx in pending
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Processing images",
                ):
                    # _process_single_page handles its own errors; result()
                    # only re-raises a truly unexpected failure, which we let
                    # propagate to the finally (deterministic shutdown).
                    future.result()

                if not self._budget_exhausted.is_set():
                    break

                # Recompute the still-pending pages: those not yet recorded in
                # the transcription results (deferred pages appended nothing).
                done = {
                    r.get("original_input_order_index")
                    for r in transcription_results
                    if isinstance(r, dict)
                }
                remaining = [idx for idx in pending if idx not in done]
                made_progress = len(remaining) < len(pending)
                pending = remaining
                if not pending:
                    break

                # Safeguard evaluated BEFORE the wait: if a reset still yields no
                # progress twice running, a page cannot fit the available daily
                # budget. Deciding here (rather than after wait_for_token_reset)
                # lets a hopeless page fail fast instead of waiting through up to
                # two daily resets (~48 h) first. A page that CAN fit after a
                # reset makes progress and resets the counter, so its semantics
                # are unchanged.
                if not made_progress:
                    stalled_resets += 1
                    if stalled_resets >= 2:
                        logger.warning(
                            "A page cannot fit the daily token budget; stopping. "
                            "Raise daily_tokens to process the remaining %d page(s).",
                            len(pending),
                        )
                        break
                else:
                    stalled_resets = 0

                logger.warning(
                    "Daily token budget reached; %d page(s) deferred. "
                    "Waiting for daily reset...",
                    len(pending),
                )
                # Stamp the wait with the transcription bucket so a per-key pool
                # cap (not just the combined budget) is what the wait polls on;
                # otherwise a still-open combined budget would spin the re-pass.
                # A distinct summary bucket is gated in _process_single_page
                # too, so wait until BOTH buckets can admit a page again.
                resumed = wait_for_token_reset(
                    **getattr(self, "_transcription_stamp", {})
                )
                summ_stamp = getattr(self, "_summary_stamp", {})
                if resumed and summ_stamp:
                    resumed = wait_for_token_reset(**summ_stamp)
                if not resumed:
                    logger.info(
                        "Token-limit wait cancelled; %d page(s) left for a later run.",
                        len(pending),
                    )
                    break
        finally:
            # Deterministic teardown. cancel_futures cancels any pages still
            # queued (e.g. after a KeyboardInterrupt) instead of letting them
            # drain; wait=False returns without blocking on in-flight calls.
            executor.shutdown(wait=False, cancel_futures=True)

        transcription_results.sort(key=lambda x: x.get("original_input_order_index", 0))
        return transcription_results, summary_results

    def _calculate_eta(self, processed_count: int, total_images: int) -> str:
        """
        Calculate estimated time of arrival for remaining items.

        Args:
            processed_count: Number of items processed so far.
            total_images: Total number of images to process.

        Returns:
            Formatted ETA string.
        """
        if processed_count <= MIN_SAMPLES_FOR_ETA or not self.start_time_processing:
            return "ETA: N/A"

        elapsed_total = time.time() - self.start_time_processing
        items_per_sec_overall = processed_count / elapsed_total

        if items_per_sec_overall <= 0:
            return "ETA: N/A"

        # Blend overall and recent rates for more accurate estimates
        blended_rate = self._calculate_blended_processing_rate(items_per_sec_overall)

        if blended_rate <= 0:
            return "ETA: N/A"

        remaining_items = total_images - processed_count
        eta_seconds = remaining_items / blended_rate
        return f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"

    def _calculate_blended_processing_rate(self, overall_rate: float) -> float:
        """Calculate blended processing rate from overall and recent samples."""
        recent_samples = self.transcription_times[-RECENT_SAMPLES_FOR_ETA:]
        if not recent_samples:
            return overall_rate

        recent_avg_time = sum(recent_samples) / len(recent_samples)
        recent_rate = 1.0 / recent_avg_time if recent_avg_time > 0 else overall_rate

        return (
            ETA_BLEND_WEIGHT_OVERALL * overall_rate
            + ETA_BLEND_WEIGHT_RECENT * recent_rate
        )

    def process_item(self) -> bool:
        """Process the item end to end.

        Returns:
            True when every page succeeded (transcription and, if enabled,
            summarization) and all configured outputs were written; False when
            the item aborted early, at least one page failed, or a summary
            output file could not be produced.
        """
        item_type_str = "PDF" if self.input_type == "pdf" else "Image Folder"
        logger.info(f"Processing {self.name} ({item_type_str})")
        self.start_time_processing = time.time()

        # Open the lazy payload source (PDF document or image folder listing);
        # no page is rendered until a transcription worker requests it.
        try:
            source = self._create_payload_source()
        except Exception as e:
            logger.exception(f"Failed to open input for {self.name}: {e}")
            return False

        if len(source) == 0:
            source.close()
            logger.info(
                f"No images found or extracted for {self.name}. Aborting this item."
            )
            # Clean up empty working directory if it was created for this item
            try:
                if not any(self.working_dir.iterdir()):  # Check if empty
                    shutil.rmtree(self.working_dir)
                    logger.info(f"Removed empty working directory: {self.working_dir}")
            except Exception as e:
                logger.warning(
                    f"Could not remove working directory {self.working_dir}: {e}"
                )
            return False

        # From here the payload source holds an open PDF handle. Guard the
        # setup phase (resume snapshotting, provenance read, log init) so any
        # failure closes the source rather than leaking the handle; the main
        # processing loop below has its own try/finally that also closes it.
        try:
            return self._run_processing(source, item_type_str)
        except Exception:
            source.close()
            raise

    def _run_processing(
        self, source: PdfPayloadSource | FolderPayloadSource, item_type_str: str
    ) -> bool:
        """Run resume setup, transcription/summary, and output rendering.

        Assumes *source* is open; the caller closes it on setup failure and the
        finally block below closes it on the normal/processing path.
        """
        self.total_items_to_transcribe = len(source)
        suffix = " and summarization" if config.SUMMARIZE else ""
        logger.info(
            f"Prepared {self.total_items_to_transcribe} images "
            f"for transcription{suffix}."
        )

        # Snapshot prior page results BEFORE the log files are reinitialized
        # below (initialize_log_file truncates them). Also detect a model
        # mismatch so summary-only resume can warn and record both models.
        if self.completed_page_indices:
            # Prefer data parsed once by ResumeChecker; otherwise read the
            # transcription log a single time and derive both the per-page
            # results and the header from it (the header feeds the resume
            # model-mismatch check, avoiding a second read of the same file).
            if self._resume_transcription_results is not None:
                self._prior_transcription_results = self._resume_transcription_results
            else:
                log_entries = _read_and_parse_log(self.log_path)
                self._prior_transcription_results = (
                    _results_from_entries(log_entries) or []
                )
                if self._resume_log_header is None:
                    self._resume_log_header = _header_from_entries(log_entries)
            self._prior_summary_results = (
                self._resume_summary_results
                if self._resume_summary_results is not None
                else load_transcription_results_from_log(self.summary_log_path) or []
            )
            self._detect_resume_model_mismatch()

        # Initialize log file with a header (incl. file-level provenance)
        target_dpi = source.target_dpi if isinstance(source, PdfPayloadSource) else None
        actual_concurrency, _ = get_transcription_concurrency()
        self._initialize_log_or_raise(
            self.log_path,
            self.name,
            str(self.input_path),
            item_type_str,
            self.total_items_to_transcribe,
            self.transcription_model,
            target_dpi,
            concurrency_limit=actual_concurrency,
            file_provenance=source.file_provenance(),
        )

        transcription_results: list[dict[str, Any]] = []
        summary_results: list[dict[str, Any]] = []

        try:
            # Process all images - transcribe and summarize
            transcription_results, summary_results = self._transcribe_and_summarize(
                source
            )

            # Final processing and output
            total_elapsed_time = time.time() - (
                self.start_time_processing or time.time()
            )
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed_time))

            # Order strictly by original input order. (A prior sort by the
            # optional "page" key relocated critical-error pages, which lack it,
            # to the front and scrambled the output.)
            transcription_results.sort(
                key=lambda x: x.get("original_input_order_index", 0)
            )

            final_success_count = sum(
                1 for r in transcription_results if "error" not in r
            )
            final_failure_count = len(transcription_results) - final_success_count

            # Summary-phase page failures (API errors after retries). Failed
            # transcriptions get error-free placeholder summaries, so this
            # never double-counts a page already in final_failure_count.
            summary_failure_count = sum(
                1 for s in summary_results if isinstance(s, dict) and "error" in s
            )
            summary_render_ok = True
            txt_write_ok = True

            # A page-budget stall (or a cancelled budget wait) can leave pages
            # deferred: they appended nothing to transcription_results and are
            # absent from the working log. total_items_to_transcribe is the
            # source page count, so a positive difference means real pages are
            # missing. Writing the .txt/summary now would emit a truncated file
            # that self-reports as complete, so withhold ALL final outputs and
            # register nothing in written_outputs. The working-log JSONL is still
            # finalized in the finally block below, so the resume completeness
            # gate reconstructs and finishes the missing pages on a later run.
            pages_deferred = self.total_items_to_transcribe - len(transcription_results)

            if pages_deferred > 0:
                logger.error(
                    "Item %s incomplete: %d of %d page(s) were deferred by the "
                    "token budget and never transcribed. Withholding the partial "
                    ".txt and summary outputs (a truncated file would misreport "
                    "as complete); %d completed page(s) are retained in the "
                    "working log for a later resume run.",
                    self.name,
                    pages_deferred,
                    self.total_items_to_transcribe,
                    len(transcription_results),
                )
            else:
                # Save transcription to text file
                logger.info(
                    f"Writing final transcription output to: {self.output_txt_path}"
                )
                txt_write_ok = write_transcription_to_text(
                    transcription_results,
                    self.output_txt_path,
                    self.name,
                    item_type_str,
                    total_elapsed_time,
                    self.input_path,
                    metadata_notes=self._output_metadata_notes or None,
                )
                if txt_write_ok:
                    self.written_outputs.append(self.output_txt_path.resolve())
                else:
                    # The write failed (e.g. disk full, file locked): the .txt
                    # does not exist, so it must neither be advertised in
                    # written_outputs nor let the item count as complete. The
                    # working log is retained, so a re-run rewrites it.
                    logger.error(
                        "Item %s: transcription .txt could not be written; "
                        "marking the item failed (working log retained for "
                        "a re-run).",
                        self.name,
                    )

                # Save summaries to output files if summarization is enabled
                if config.SUMMARIZE and summary_results:
                    # Step 1: Adjust page numbers and sort
                    # (uses page_number_type for unnumbered detection)
                    adjusted_summary_results = (
                        self.page_number_processor.adjust_and_sort_page_numbers(
                            summary_results
                        )
                    )

                    # Build ONE citation manager per item: citations are
                    # collected, consolidated, and OpenAlex-enriched exactly
                    # once, then both writers render from the same instance
                    # (prevents duplicate API lookups and diverging
                    # bibliographies). A failure here blocks BOTH writers.
                    render_context = None
                    try:
                        render_context = build_render_context(
                            adjusted_summary_results,
                            polite_pool_email=config.CITATION_OPENALEX_EMAIL,
                        )
                        enrich_if_enabled(render_context[0])
                    except Exception as e:
                        summary_render_ok = False
                        logger.error(f"Error building summary render context: {e}")

                    if render_context is not None:
                        citation_manager, render_data = render_context
                        # Each writer gets its own try/except so a DOCX failure
                        # does not skip the (independent) Markdown writer.
                        if config.OUTPUT_DOCX:
                            try:
                                create_docx_summary(
                                    adjusted_summary_results,
                                    self.output_summary_docx_path,
                                    self.name,
                                    citation_manager=citation_manager,
                                    data=render_data,
                                )
                                self.written_outputs.append(
                                    self.output_summary_docx_path.resolve()
                                )
                            except Exception as e:
                                summary_render_ok = False
                                logger.error(f"Error creating DOCX summary: {e}")

                        if config.OUTPUT_MARKDOWN:
                            try:
                                create_markdown_summary(
                                    adjusted_summary_results,
                                    self.output_summary_md_path,
                                    self.name,
                                    citation_manager=citation_manager,
                                    data=render_data,
                                )
                                self.written_outputs.append(
                                    self.output_summary_md_path.resolve()
                                )
                            except Exception as e:
                                summary_render_ok = False
                                logger.error(f"Error creating Markdown summary: {e}")

            logger.info(f"PROCESSING COMPLETE for item: {self.name}")
            logger.info(f"  Total images for this item: {len(transcription_results)}")
            logger.info(f"  Successfully transcribed: {final_success_count}")
            logger.info(f"  Failed items: {final_failure_count}")
            logger.info(f"  Total time for this item: {elapsed_str}")
            if self.transcription_times:  # Based on successful API calls
                avg_api_time = sum(self.transcription_times) / len(
                    self.transcription_times
                )
                logger.info(
                    "  Average API processing time per successful image: "
                    f"{avg_api_time:.2f}s"
                )
            if total_elapsed_time > 0 and final_success_count > 0:
                throughput_iph = (final_success_count / total_elapsed_time) * 3600
                logger.info(
                    f"  Overall throughput for this item: "
                    f"{throughput_iph:.1f} successful images/hour"
                )
            logger.info(f"  Final transcription output: {self.output_txt_path}")
            if config.SUMMARIZE:
                summary_outputs = []
                if config.OUTPUT_DOCX:
                    summary_outputs.append(str(self.output_summary_docx_path))
                if config.OUTPUT_MARKDOWN:
                    summary_outputs.append(str(self.output_summary_md_path))
                if summary_outputs:
                    logger.info(
                        f"  Final summary outputs: {', '.join(summary_outputs)}"
                    )
            detail_suffix = (
                " and " + str(self.summary_log_path) if config.SUMMARIZE else ""
            )
            logger.info(f"  Detailed logs: {self.log_path}{detail_suffix}")

            # Structured per-item report for the CLI overview layer, built from
            # values already computed above. Populated on every verdict that
            # reaches this point (complete, partial/deferred, failed).
            avg_api_s = (
                sum(self.transcription_times) / len(self.transcription_times)
                if self.transcription_times
                else None
            )
            self.last_run_report = {
                "pages_total": self.total_items_to_transcribe,
                "pages_attempted": len(transcription_results),
                "pages_ok": final_success_count,
                "pages_failed": final_failure_count,
                "pages_deferred": pages_deferred,
                "summary_failures": summary_failure_count,
                "elapsed_s": total_elapsed_time,
                "avg_api_s": avg_api_s,
                "outputs": [str(p) for p in self.written_outputs],
            }

            # Item verdict: a budget-deferred page, any failed page
            # (transcription or summary), or a failed summary-file render means
            # the item is NOT complete, so the caller counts it failed and the
            # run exits non-zero.
            item_success = (
                pages_deferred == 0
                and final_failure_count == 0
                and summary_failure_count == 0
                and summary_render_ok
                and txt_write_ok
            )
            if not item_success:
                logger.error(
                    "Item %s finished incomplete: %d deferred page(s), %d failed "
                    "transcription page(s), %d failed summary page(s), summary "
                    "files rendered ok: %s, transcription .txt written ok: %s",
                    self.name,
                    pages_deferred,
                    final_failure_count,
                    summary_failure_count,
                    summary_render_ok,
                    txt_write_ok,
                )
            return item_success

        finally:
            # Always finalize log files by closing JSON arrays, even if errors occurred
            finalize_log_file(self.log_path)
            if config.SUMMARIZE:
                finalize_log_file(self.summary_log_path)

            source.close()
            # Managers are built per item; signal end-of-item teardown. The
            # provider httpx clients are process-shared and must NOT be closed
            # here (see LLMClientBase.close), so this is now effectively a
            # debug-logging hook kept for lifecycle symmetry.
            self._close_managers()

    def _close_managers(self) -> None:
        """Invoke each per-item manager's ``close`` (a documented no-op).

        Retained for lifecycle symmetry and API compatibility;
        ``LLMClientBase.close`` no longer touches the shared httpx clients.
        """
        for manager in (
            getattr(self, "transcribe_manager", None),
            getattr(self, "summary_manager", None),
        ):
            if manager is None:
                continue
            close = getattr(manager, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # defensive: teardown must not raise
                    logger.debug("Manager close failed: %s", exc)
