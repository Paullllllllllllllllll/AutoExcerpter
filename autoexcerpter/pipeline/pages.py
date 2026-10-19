"""One page of an item: render, transcribe, clean and summarize it into the store.

Rendering runs in an executor; the model calls are awaited on the event
loop. Each page has a transcription unit and, when the run summarizes, a
summary unit (see :mod:`autoexcerpter.output.store`). The transcription unit
is committed before the summary call, so a crash during that call leaves the
page transcribed and its summary pending. A unit that cannot be written is
counted and the page stays pending; a lost lease stops every page that has
not started its unit yet.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from autoexcerpter.common.images import PagePayload
from autoexcerpter.common.store import LeaseLostError
from autoexcerpter.constants import is_blank_transcription
from autoexcerpter.llm.summary import placeholder_summary as failed_summary
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    SummaryRow,
    transcription_outcome,
)
from autoexcerpter.pipeline.progress import PageProgress
from autoexcerpter.pipeline.text_cleaner import clean_transcription
from autoexcerpter.pipeline.units import DocumentHandle, UnitCall, unit_call
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


def _summary_outcome(
    entry: dict[str, Any], transcription_failed: bool, text: str
) -> str:
    """Return the outcome of a page's summary unit."""
    if transcription_failed:
        return "placeholder"
    if "error" in entry:
        return "failed"
    return "blank" if is_blank_transcription(text) or not text.strip() else "ok"


def summary_row(index: int, entry: dict[str, Any]) -> SummaryRow:
    """Return the ``summaries`` row of a page's summary entry."""
    page = entry.get("page")
    info = entry.get("page_information")
    bullets = entry.get("bullet_points")
    references = entry.get("references")
    return SummaryRow(
        index,
        page=page if isinstance(page, int) and not isinstance(page, bool) else None,
        page_information=info if isinstance(info, dict) else None,
        bullet_points=bullets if isinstance(bullets, list) else None,
        references=references if isinstance(references, list) else None,
    )


def _provenance(result: dict[str, Any]) -> dict[str, Any] | None:
    provenance: dict[str, Any] = {
        key: result[key] for key in ("source_file", "image_provenance") if key in result
    }
    return provenance or None


class PageRunner:
    """Process single pages of one item on the event loop.

    *summary_manager* is None when the run does not summarize. *executor*
    renders the page images; None uses the loop's default executor. *pdf*
    says whether a page's source index is its PDF page (else its image
    name). After the pages, ``write_errors`` holds the unit writes that
    failed and ``lost`` the lost lease, if any.
    """

    def __init__(
        self,
        *,
        item_name: str,
        document: DocumentHandle,
        transcribe_manager: Transcriber,
        summary_manager: Summarizer | None,
        transcription_provider: str | None,
        pdf: bool = True,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self.item_name = item_name
        self.document = document
        self.transcribe_manager = transcribe_manager
        self.summary_manager = summary_manager
        self.transcription_provider = transcription_provider
        self.pdf = pdf
        self.executor = executor
        self.write_errors: list[BaseException] = []
        self.lost: LeaseLostError | None = None

    def source_index(self, index: int, image_name: str) -> int | str:
        """Return a page's source index: its PDF page, else its image name."""
        return index if self.pdf else image_name

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
            if (
                is_blank_transcription(transcription_text)
                or not transcription_text.strip()
            ):
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

    async def run(self, index: int, source: PageSource, progress: PageProgress) -> str:
        """Render, transcribe and optionally summarize the page at *index*.

        Advances *progress* and returns the page's event status: ``ok``,
        ``failed``, or ``deferred`` when the page was not stored (a lost
        lease or a failed write).
        """
        image_name = f"page index {index}"
        try:
            image_name = source.image_name(index)
        except Exception as exc:  # noqa: BLE001 - the transcription records it
            logger.error("Could not name page %d: %s", index, exc)
        stored = await self._transcribe_page(index, source, image_name, progress)
        if stored is None:
            progress.advance()
            return "deferred"
        text, error = stored
        if self.summary_manager is not None:
            await self.summarize_page(index, image_name, text, error)
        processed = progress.advance()
        logger.debug(
            f"Processed {processed}/{progress.total} pending page(s) "
            f"({progress.already_complete} already complete) - Item {index + 1} "
            f"- {progress.eta(processed)}"
        )
        return "failed" if error is not None else "ok"

    async def summarize_page(
        self, index: int, image_name: str, text: str, error: str | None
    ) -> bool:
        """Summarize a stored transcription into the page's summary unit.

        *error* is the transcription's error: a failed page gets a
        placeholder summary without a call. Returns whether the unit was
        stored.
        """
        if self.lost is not None or self.summary_manager is None:
            return False
        source_index = self.source_index(index, image_name)
        transcription: dict[str, Any] = {"transcription": text}
        if error is not None:
            transcription["error"] = error
        async with unit_call(self.document, KIND_SUMMARY, index) as call:
            try:
                entry = await self.summarize(transcription, index, image_name)
            except Exception as exc:  # noqa: BLE001 - a crash fails one summary
                logger.exception(f"Failed to summarize {image_name}: {exc}")
                page = index + 1
                entry = build_summary_result(
                    index, image_name, failed_summary(page, str(exc)), page, str(exc)
                )
                entry["error_type"] = type(exc).__name__
        if entry is None:
            return False
        failed = "error" in entry and error is None
        row = summary_row(index, entry)
        return await self._write(
            call,
            lambda result: self.document.write_summary(result, row),
            status="failed" if failed else "done",
            content=None,
            outcome=_summary_outcome(entry, error is not None, text),
            label=image_name,
            source_index=source_index,
            error=entry.get("error") if failed else None,
            error_type=entry.get("error_type") if failed else None,
        )

    async def _transcribe_page(
        self, index: int, source: PageSource, image_name: str, progress: PageProgress
    ) -> tuple[str, str | None] | None:
        """Transcribe and store one page; return its text and error.

        None when the unit was not stored.
        """
        if self.lost is not None:
            return None
        async with unit_call(self.document, KIND_TRANSCRIPTION, index) as call:
            try:
                result = await self._transcribe(index, source, image_name)
                if "error" not in result:
                    raw_text = result.get("transcription", "")
                    result["transcription"] = clean_transcription(raw_text)
            except Exception as exc:  # noqa: BLE001 - a crash fails one page
                logger.exception(f"Critical error during task for {image_name}: {exc}")
                result = {
                    "transcription": f"[CRITICAL ERROR] Unhandled in task: {exc}",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
        text = result.get("transcription", "")
        if not isinstance(text, str):
            text = str(text)
        error = result.get("error")
        error = None if error is None else str(error)
        stored = await self._write(
            call,
            self.document.write_transcription,
            status="failed" if error is not None else "done",
            content=text,
            outcome=transcription_outcome(text, error),
            label=image_name,
            source_index=self.source_index(index, image_name),
            error=error,
            error_type=result.get("error_type") if error is not None else None,
            provenance=_provenance(result),
        )
        if not stored:
            return None
        if error is None and "processing_time" in result:
            progress.record_time(result["processing_time"])
        logger.debug(f"Page {index + 1} - Status: {_status(result)}")
        return text, error

    async def _write(
        self,
        call: UnitCall,
        write: Callable[[Any], Awaitable[None]],
        **fields: Any,
    ) -> bool:
        """Build the unit's row from *call* and commit it through *write*.

        A lost lease stops the item; any other failure is counted. Returns
        whether the unit was stored.
        """
        try:
            await write(call.result(**fields))
        except LeaseLostError as exc:
            logger.error(
                "Stopping '%s': page %d was not stored: %s",
                self.item_name,
                call.idx + 1,
                exc,
            )
            self.lost = exc
            return False
        except Exception as exc:  # noqa: BLE001 - tallied; the item fails
            logger.error(
                "The %s of page %d of '%s' could not be stored in %s: %s",
                call.kind,
                call.idx + 1,
                self.item_name,
                self.document.store.path,
                exc,
            )
            self.write_errors.append(exc)
            return False
        return True

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
            }
        return {
            **(await self.transcribe_manager.transcribe_payload(payload)),
            "source_file": payload.source_file,
            "image_provenance": payload.provenance,
        }


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
    "summary_row",
    "warn_on_page_number_mismatch",
]
