"""Page summaries: one page transcription in, a structured summary out.

The summary always asks for the summary schema's JSON. Its prompt is the
plain-text variant when the transcription was plain text. ``schema``
enforces the schema where the provider can; ``json`` and ``text`` only ask
for it. Invalid answers are retried under the settings' rule; when the
retries run out, a ``text`` answer is kept as one bullet point and any other
answer leaves an error placeholder. An answer cut at the output-token limit
gets no retry, since a retry with the same limit would be cut again: it
fails the page in every response format.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

from autoexcerpter.common.provider_requests import (
    Part,
    message_content,
    request_params,
    system_content,
)
from autoexcerpter.common.responses import get_response_metadata
from autoexcerpter.common.structured import (
    ResponseFormat,
    StructuredRequest,
    StructuredResult,
)
from autoexcerpter.llm.caller import CallEnv
from autoexcerpter.llm.options import request_plan, response_format_for
from autoexcerpter.llm.prompts import (
    PROMPTS_DIR,
    SCHEMAS_DIR,
    render_prompt_with_schema,
    strip_markdown_code_block,
)
from autoexcerpter.llm.transcription import (
    EMPTY_RESPONSE_BACKOFF_S,
    EMPTY_RESPONSE_RETRIES,
    MAX_PAGE_CALLS,
)
from autoexcerpter.llm.types import PhaseModel

__all__ = [
    "SUMMARY_REQUIRED_KEYS",
    "SummaryManager",
    "ensure_page_information",
    "failed_summary",
    "placeholder_summary",
    "summary_entry",
    "summary_prompt",
    "summary_request_format",
]

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA_FILE = "summary_schema.json"
SUMMARY_PROMPT_FILE = "summary_system_prompt.txt"
PLAIN_TEXT_SUMMARY_PROMPT_FILE = "summary_plain_text_prompt.txt"

SUMMARY_REQUIRED_KEYS = ("page_information", "bullet_points", "references")


def placeholder_summary(
    page_num: int, error_message: str = "", page_types: list[str] | None = None
) -> dict[str, Any]:
    """Return a summary for a page that could not be summarized.

    *page_num* is the page's position and stays in ``page``; no printed page
    number is known, so ``page_information`` records none.
    """
    bullet = (
        f"[Error generating summary: {error_message}]"
        if error_message
        else "[Summary generation failed]"
    )
    result: dict[str, Any] = {
        "page": page_num,
        "page_information": {
            "page_number_integer": None,
            "is_two_page_spread": False,
            "page_number_integer_end": None,
            "page_number_type": "none",
            "page_types": page_types or ["other"],
        },
        "bullet_points": [bullet],
        "references": None,
    }
    if error_message:
        result["error"] = error_message
    return result


def ensure_page_information(summary: dict[str, Any]) -> None:
    """Fill missing ``page_information`` fields without inventing a page number.

    A missing ``page_number_type`` becomes ``arabic`` only when the model
    returned an integer page number, else ``none``.
    """
    info = summary.get("page_information")
    if not isinstance(info, dict):
        summary["page_information"] = {
            "page_number_integer": None,
            "is_two_page_spread": False,
            "page_number_integer_end": None,
            "page_number_type": "none",
            "page_types": ["content"],
        }
        return
    info.setdefault("page_number_integer", None)
    if "page_number_type" not in info:
        number = info.get("page_number_integer")
        is_int = isinstance(number, int) and not isinstance(number, bool)
        info["page_number_type"] = "arabic" if is_int else "none"
    info.setdefault("page_types", ["content"])
    info.setdefault("is_two_page_spread", False)
    info.setdefault("page_number_integer_end", None)


def _wrapped(text: str) -> dict[str, Any]:
    """Return a plain answer as the only bullet point of a summary."""
    return {
        "page_information": {
            "page_number_integer": None,
            "page_number_type": "none",
            "page_types": ["content"],
        },
        "bullet_points": [strip_markdown_code_block(text).strip()],
        "references": None,
    }


def summary_prompt(
    context: str | None, transcription_format: str
) -> tuple[dict[str, Any], str]:
    """Return the summary schema and the system prompt of a summary call.

    *context* is the item's summary context as it enters the prompt;
    *transcription_format* is the transcription's response format, whose
    ``text`` value selects the plain-text prompt.
    """
    schema: dict[str, Any] = json.loads(
        (SCHEMAS_DIR / SUMMARY_SCHEMA_FILE).read_text(encoding="utf-8")
    )
    prompt_file = (
        PLAIN_TEXT_SUMMARY_PROMPT_FILE
        if transcription_format == "text"
        else SUMMARY_PROMPT_FILE
    )
    prompt = (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
    system_prompt = render_prompt_with_schema(
        prompt, schema.get("schema", schema), context=context
    )
    return schema, system_prompt


def failed_summary(
    page_num: int,
    message: str,
    error_type: str,
    provider: str,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return the error placeholder of a summary that failed."""
    placeholder = placeholder_summary(page_num, message)
    placeholder["error_type"] = error_type
    placeholder["provider"] = provider
    if counts and sum(counts.values()) > 0:
        placeholder["schema_retries"] = dict(counts)
    return placeholder


def summary_entry(
    page_num: int,
    *,
    text: str,
    data: dict[str, Any] | None,
    error: str | None,
    response_format: str,
    provider: str,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return a page's summary from a final answer, or its error placeholder.

    *data* and *error* are the answer's parsed JSON and validation failure.
    A valid answer is kept; a ``text``-format answer that failed validation
    becomes one bullet point; any other failure gives the placeholder. The
    synchronous and the batch path share this rule.
    """
    if error is None and isinstance(data, dict):
        summary = dict(data)
    elif response_format == "text":
        logger.warning(
            "Summary for page %d: wrapping a plain answer into the summary structure.",
            page_num,
        )
        summary = _wrapped(text)
    else:
        logger.error(
            "Invalid summary JSON for page %d after validation retries: %s",
            page_num,
            error,
        )
        return failed_summary(
            page_num,
            f"Invalid summary JSON: {error}",
            "schema_validation",
            provider,
            counts,
        )
    ensure_page_information(summary)
    entry: dict[str, Any] = {
        "page": page_num,
        "page_information": summary.get("page_information"),
        "bullet_points": summary.get("bullet_points"),
        "references": summary.get("references"),
        "provider": provider,
    }
    if counts and sum(counts.values()) > 0:
        entry["schema_retries"] = dict(counts)
    return entry


def summary_request_format(response_format: ResponseFormat) -> ResponseFormat:
    """Return the format a summary is requested in; ``text`` asks for JSON.

    A summary is always parsed as JSON; text only changes what happens when
    no valid answer arrives.
    """
    return "json" if response_format == "text" else response_format


class SummaryManager:
    """Summarize page transcriptions with the summary model of one item.

    *context* is the item's summary context as it enters the prompt;
    *transcription_format* is the response format of the transcription.
    """

    def __init__(
        self,
        phase: PhaseModel,
        env: CallEnv,
        *,
        context: str | None = None,
        transcription_format: str = "schema",
    ) -> None:
        self.caller = env.caller("summary", phase)
        self.provider = self.caller.provider
        self.model_name = phase.name
        self.response_format: ResponseFormat = response_format_for(
            phase.response_format, self.provider, phase.name
        )
        retry = env.settings.retry
        self.rule = retry.schema_retries.get("summary", {}).get("validation_failure")
        self._jitter = (retry.jitter_min, retry.jitter_max)
        self.summary_context = context
        self.schema, self.system_prompt = summary_prompt(context, transcription_format)
        self.caching = env.settings.prompt_caching
        self.plan = request_plan(
            self.provider, phase.name, phase.options, phase.service_tier, self.caching
        )
        self.invoke_kwargs = request_params(self.plan)

    def messages(self, transcription: str) -> list[Any]:
        """Return the system and user messages for one page's text."""
        from langchain_core.messages import HumanMessage, SystemMessage

        return [
            SystemMessage(content=system_content(self.plan, self.system_prompt)),
            HumanMessage(
                content=message_content(self.plan, [Part(text=transcription)])
            ),
        ]

    def _request(self, used: int, calls_left: int) -> StructuredRequest:
        rule = self.rule
        retries = 0
        if rule is not None and rule.enabled:
            retries = max(0, min(rule.max_attempts - used, calls_left - 1))

        def delay(retry: int) -> float:
            if rule is None:
                return 0.0
            step = used + retry - 1
            return rule.backoff_base * rule.backoff_multiplier**step + random.uniform(
                *self._jitter
            )

        return StructuredRequest(
            provider=self.provider,
            response_format=summary_request_format(self.response_format),
            schema=self.schema,
            invoke_kwargs=self.invoke_kwargs,
            required=SUMMARY_REQUIRED_KEYS,
            validation_retries=retries,
            validation_delay=delay,
        )

    async def generate_summary(
        self, transcription: str, page_num: int
    ) -> dict[str, Any]:
        """Summarize one page; failures return a placeholder with ``error``."""
        started = time.time()
        counts = {"validation_failure": 0}
        try:
            result = await self._answer(transcription, page_num, counts)
        except Exception as exc:  # noqa: BLE001 - one page must not fail the item
            logger.error(
                "Summary API error for page %d after retries: %s - %s",
                page_num,
                type(exc).__name__,
                exc,
            )
            return self._failed(page_num, str(exc), "api_failure")
        if result is None:
            logger.error(
                "Summary schema retries exhausted for page %d without a valid "
                "response.",
                page_num,
            )
            return self._failed(
                page_num,
                "Summary schema validation retries exhausted",
                "schema_validation",
                counts,
            )
        if result.stopped:
            outcome = result.outcome
            logger.warning(
                "Summary for page %d stopped (%s): %s",
                page_num,
                outcome.kind,
                outcome.reason or "no reason given",
            )
            return self._failed(page_num, outcome.message, outcome.kind)
        entry = summary_entry(
            page_num,
            text=result.text,
            data=result.data,
            error=result.error,
            response_format=self.response_format,
            provider=self.provider,
            counts=counts,
        )
        if "error" in entry:
            return entry
        entry["processing_time"] = round(time.time() - started, 2)
        entry["api_response"] = get_response_metadata(result.response)
        return entry

    async def _answer(
        self, transcription: str, page_num: int, counts: dict[str, int]
    ) -> StructuredResult | None:
        """Return the final answer, or None when the request limit is reached."""
        label = f"Summary for page {page_num}"
        messages = self.messages(transcription)
        empty = 0
        calls = 0
        while calls < MAX_PAGE_CALLS:
            result = await self.caller.call(
                messages,
                self._request(counts["validation_failure"], MAX_PAGE_CALLS - calls),
                label=label,
            )
            calls += 1 + result.validation_retries
            counts["validation_failure"] += result.validation_retries
            if result.stopped or result.text:
                return result
            if empty >= EMPTY_RESPONSE_RETRIES:
                raise ValueError("LLM API returned empty content for summary.")
            empty += 1
            logger.warning(
                "Empty content for page %d. Retrying (%d/%d)...",
                page_num,
                empty,
                EMPTY_RESPONSE_RETRIES,
            )
            await asyncio.sleep(EMPTY_RESPONSE_BACKOFF_S)
        return None

    def _failed(
        self,
        page_num: int,
        message: str,
        error_type: str,
        counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return failed_summary(page_num, message, error_type, self.provider, counts)
