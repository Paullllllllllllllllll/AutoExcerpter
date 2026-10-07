"""One page of an item: render, transcribe, clean, summarize and log it.

Rendering runs in an executor; the model calls are awaited on the event
loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import logging
from typing import Any, Protocol

from autoexcerpter.common.images import PagePayload
from autoexcerpter.common.usage import UsageTotals
from autoexcerpter.constants import is_blank_transcription
from autoexcerpter.llm import collect_usage
from autoexcerpter.pipeline.log import append_to_log
from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.text_cleaner import clean_transcription
from autoexcerpter.rendering.page_tags import printed_page_numbers

logger = logging.getLogger(__name__)


class PageSource(Protocol):
    """The part of a payload source that the page runner uses."""

    def image_name(self, index: int) -> str: ...

    def build_payload(self, index: int) -> PagePayload: ...


class Transcriber(Protocol):
    """The transcription manager interface used per page."""

    async def transcribe_payload(self, payload: PagePayload) -> dict[str, Any]: ...


class Summarizer(Protocol):
    """The summary manager interface used per page."""

    async def generate_summary(
        self, transcription: str, page_num: int
    ) -> dict[str, Any]: ...


def warn_on_page_number_mismatch(
    summary_data: dict[str, Any], transcription_text: str, image_name: str
) -> bool:
    """Warn when the summary's page number disagrees with the page's tags.

    Compares the returned ``page_information.page_number_integer`` with the
    ``<page_number>`` tags in the page's transcription (Roman numerals
    converted). A mismatch is an integer not among the tagged values, a null
    despite tags, or an integer despite no tags. The value is never changed.

    Returns:
        True when a mismatch was logged.
    """
    page_info = summary_data.get("page_information")
    returned = (
        page_info.get("page_number_integer") if isinstance(page_info, dict) else None
    )
    if isinstance(returned, bool) or not isinstance(returned, int):
        returned = None
    tagged = printed_page_numbers(transcription_text)

    if returned is None and not tagged:
        return False
    if returned is not None and returned in tagged:
        return False
    if returned is None:
        problem = f"returned no page number, but the page is tagged {tagged}"
    elif not tagged:
        problem = f"returned page {returned}, but the page has no <page_number> tag"
    else:
        problem = f"returned page {returned}, but the page is tagged {tagged}"
    logger.warning(f"Summary for {image_name} {problem}; keeping the returned value.")
    return True


def build_summary_result(
    original_index: int,
    image_name: str,
    summary_payload: dict[str, Any],
    page_number: int | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Merge the API summary fields and the page metadata into one flat entry."""
    result: dict[str, Any] = {
        "original_input_order_index": original_index,
        "image_filename": image_name,
    }

    if page_number is not None:
        result["page"] = page_number
    elif isinstance(summary_payload, dict) and "page" in summary_payload:
        result["page"] = summary_payload.get("page")

    if isinstance(summary_payload, dict):
        result["page_information"] = summary_payload.get("page_information")
        result["bullet_points"] = summary_payload.get("bullet_points")
        result["references"] = summary_payload.get("references")

        if "processing_time" in summary_payload:
            result["processing_time"] = summary_payload["processing_time"]
        if "provider" in summary_payload:
            result["provider"] = summary_payload["provider"]
        if "api_response" in summary_payload:
            result["api_response"] = summary_payload["api_response"]
        if "schema_retries" in summary_payload:
            result["schema_retries"] = summary_payload["schema_retries"]
        if error_message and "error_type" in summary_payload:
            result["error_type"] = summary_payload["error_type"]

    if error_message:
        result["error"] = error_message

    return result


def placeholder_summary(
    bullet_points: list[str] | None,
    page_types: list[str],
    error_message: str | None = None,
) -> dict[str, Any]:
    """Return a summary payload that records no printed page number."""
    payload: dict[str, Any] = {
        "page": 0,
        "page_information": {
            "page_number_integer": None,
            "is_two_page_spread": False,
            "page_number_integer_end": None,
            "page_number_type": "none",
            "page_types": page_types,
        },
        "bullet_points": bullet_points,
        "references": None,
    }
    if error_message:
        payload["error"] = error_message
    return payload


def _position(page_value: Any, fallback_index: int) -> int:
    """Return a stored integer "page" value, else the 1-based position."""
    return page_value if isinstance(page_value, int) else fallback_index + 1


class PageRunner:
    """Process single pages of one item on the event loop.

    *summary_manager* is None when the run does not summarize. *executor*
    renders the page images; None uses the loop's default executor.
    ``page_usage`` maps each page index run here to the usage of its model
    calls, per role.
    """

    def __init__(
        self,
        *,
        item_name: str,
        paths: ItemPaths,
        transcribe_manager: Transcriber,
        summary_manager: Summarizer | None,
        transcription_provider: str | None,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self.item_name = item_name
        self.paths = paths
        self.transcribe_manager = transcribe_manager
        self.summary_manager = summary_manager
        self.transcription_provider = transcription_provider
        self.executor = executor
        # Working-log appends that failed on the page path. Such a page exists
        # only in memory, so the item must not be reported complete.
        self.log_append_failures = 0
        self.page_usage: dict[int, UsageTotals] = {}

    async def summarize(
        self,
        transcription_result: dict[str, Any],
        original_index: int,
        image_name: str,
    ) -> dict[str, Any] | None:
        """Generate or build a placeholder summary for one transcription.

        Returns None when the run does not summarize.
        """
        if self.summary_manager is None:
            return None

        # A position (or a stored "page" value), never a printed page number:
        # the placeholders below record none.
        page_num = _position(transcription_result.get("page"), original_index)

        if "error" not in transcription_result:
            transcription_text = transcription_result.get("transcription", "")
            if is_blank_transcription(transcription_text):
                summary_data = placeholder_summary(None, ["blank"])
            else:
                summary_data = await self.summary_manager.generate_summary(
                    transcription_text, page_num
                )
                if isinstance(summary_data, dict) and "error" not in summary_data:
                    warn_on_page_number_mismatch(
                        summary_data, transcription_text, image_name
                    )
            summary_error = (
                summary_data.get("error") if isinstance(summary_data, dict) else None
            )
        else:
            error_msg = transcription_result.get("error", "Unknown error")
            summary_data = placeholder_summary(
                [f"[Transcription failed: {error_msg}]"],
                ["other"],
                error_message=error_msg,
            )
            summary_error = None

        return build_summary_result(
            original_index, image_name, summary_data, page_num, summary_error
        )

    def note_log_append_failure(self, log_path: Any) -> None:
        """Count (and log) a page-path working-log append that failed."""
        self.log_append_failures += 1
        failures = self.log_append_failures
        logger.error(
            "Item %s: failed to append an entry to %s (%d hot-path append "
            "failure(s) so far); that page is not recoverable from the working "
            "log on a later resume.",
            self.item_name,
            log_path,
            failures,
        )

    async def run(
        self,
        index: int,
        source: PageSource,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
        progress: PageProgress,
    ) -> dict[str, Any]:
        """Render, transcribe and optionally summarize the page at *index*.

        Appends to the shared result lists, advances *progress* and records
        the page's usage in ``page_usage``. A page cancelled before its
        transcription is logged leaves no trace, so a later resume runs it.
        """
        with collect_usage() as usage:
            self.page_usage[index] = usage
            return await self._run(
                index, source, transcription_results, summary_results, progress
            )

    async def summarize_logged(
        self, entry: dict[str, Any], index: int, image_name: str
    ) -> dict[str, Any] | None:
        """Summarize a page transcribed by an earlier run; record its usage."""
        with collect_usage() as usage:
            self.page_usage[index] = usage
            return await self.summarize(entry, index, image_name)

    async def _run(
        self,
        index: int,
        source: PageSource,
        transcription_results: list[dict[str, Any]],
        summary_results: list[dict[str, Any]],
        progress: PageProgress,
    ) -> dict[str, Any]:
        image_name = f"page index {index}"
        # Set once the paid transcription entry is on disk; the except handler
        # must not log a second entry for this index.
        transcription_logged = False
        transcription_result: dict[str, Any] = {}
        try:
            image_name = source.image_name(index)
            transcription_result = await self._transcribe(index, source, image_name)

            if "error" not in transcription_result:
                raw_text = transcription_result.get("transcription", "")
                transcription_result["transcription"] = clean_transcription(raw_text)

            # Persist the paid transcription before the summary call, so a crash
            # during that call leaves the supported summary-only resume state.
            transcription_logged = append_to_log(
                self.paths.transcription_log, transcription_result
            )
            if not transcription_logged:
                self.note_log_append_failure(self.paths.transcription_log)

            summary_result = await self.summarize(
                transcription_result, index, image_name
            )
            if summary_result is not None:
                if not append_to_log(self.paths.summary_log, summary_result):
                    self.note_log_append_failure(self.paths.summary_log)
                summary_results.append(summary_result)

            processed_now = progress.advance()
            if (
                "processing_time" in transcription_result
                and "error" not in transcription_result
            ):
                progress.record_time(transcription_result["processing_time"])

            logger.debug(
                f"Processed {processed_now}/{progress.total} pending page(s) "
                f"({progress.already_complete} already complete) "
                f"- Item {index + 1} - Status: {_status(transcription_result)} "
                f"- {progress.eta(processed_now)}"
            )
            transcription_results.append(transcription_result)
            return transcription_result

        except Exception as e:
            logger.exception(f"Critical error during task for {image_name}: {e}")
            error_result: dict[str, Any] = {
                "page": index + 1,
                "image": image_name,
                "transcription": f"[CRITICAL ERROR] Unhandled in task: {e}",
                "error": str(e),
                "original_input_order_index": index,
            }
            # Keep the working log and the summaries consistent with the normal
            # failure path. When the transcription entry is already on disk,
            # a second entry would duplicate the page on every later resume.
            if not transcription_logged:
                append_to_log(self.paths.transcription_log, error_result)
            try:
                summary_result = await self.summarize(error_result, index, image_name)
            except Exception:
                logger.exception(
                    f"Failed to build placeholder summary for {image_name}"
                )
                summary_result = None
            if summary_result is not None:
                if transcription_logged:
                    # The logged transcription is error-free, so the failure
                    # must live on the summary entry, which resume regenerates.
                    summary_result["error"] = str(e)
                append_to_log(self.paths.summary_log, summary_result)
                summary_results.append(summary_result)
            # A logged transcription means "transcribed, summary pending":
            # keep the real text in memory rather than the crash placeholder.
            transcription_results.append(
                transcription_result if transcription_logged else error_result
            )
            progress.advance()
            return error_result

    async def _render(self, source: PageSource, index: int) -> PagePayload:
        """Build the page payload in the executor."""
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        call = functools.partial(context.run, source.build_payload, index)
        return await loop.run_in_executor(self.executor, call)

    async def _transcribe(
        self, index: int, source: PageSource, image_name: str
    ) -> dict[str, Any]:
        """Build the page payload and transcribe it, or record the failure."""
        try:
            payload = await self._render(source, index)
        except Exception as e:  # noqa: BLE001 - a render failure fails one page
            logger.error(f"Error preparing page {image_name}: {e}")
            return {
                "image": image_name,
                "sequence_number": index + 1,
                "transcription": f"[preprocessing error: {e}]",
                "processing_time": 0.0,
                "error": str(e),
                "error_type": "preprocessing_failure",
                "provider": self.transcription_provider,
                "original_input_order_index": index,
            }
        result = {
            **(await self.transcribe_manager.transcribe_payload(payload)),
            "original_input_order_index": index,
            "source_file": payload.source_file,
            "image_provenance": payload.provenance,
        }
        if payload.page_index is not None:
            result["page_index"] = payload.page_index
        return result


def _status(transcription_result: dict[str, Any]) -> str:
    """Return the progress-line status of one transcription result."""
    if "error" not in transcription_result:
        return "SUCCESS"
    error_type = transcription_result.get("error_type", "unknown")
    schema_retries = transcription_result.get("schema_retries")
    if isinstance(schema_retries, dict) and schema_retries:
        return f"FAILED ({error_type}, {sum(schema_retries.values())} schema retries)"
    return f"FAILED ({error_type})"


__all__ = [
    "PageRunner",
    "PageSource",
    "Summarizer",
    "Transcriber",
    "build_summary_result",
    "placeholder_summary",
    "warn_on_page_number_mismatch",
]
