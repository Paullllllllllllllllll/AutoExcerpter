"""Page transcription: one preprocessed page image in, the page text out.

The response format decides the prompt: ``schema`` and ``json`` use the
system prompt with the transcription schema and parse a JSON answer, ``text``
uses the plain-text prompt and takes the answer as the transcription. Within
one page, invalid JSON answers, an empty answer and the model's
``no_transcribable_text`` and ``transcription_not_possible`` flags (or their
plain-text sentinels) are retried under the settings' rules; a content filter
or refusal ends the page at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from autoexcerpter.common.images import PagePayload
from autoexcerpter.common.responses import inspect_response
from autoexcerpter.common.structured import (
    ResponseFormat,
    StructuredRequest,
    StructuredResult,
)
from autoexcerpter.llm.caller import CallEnv
from autoexcerpter.llm.capabilities import detect_capabilities
from autoexcerpter.llm.image_detail import resolve_request_detail
from autoexcerpter.llm.options import build_invoke_kwargs, response_format_for
from autoexcerpter.llm.prompts import (
    PROMPTS_DIR,
    SCHEMAS_DIR,
    render_prompt_with_schema,
    strip_markdown_code_block,
)
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.settings import RetryRule

__all__ = [
    "TRANSCRIPTION_REQUIRED_KEYS",
    "TranscriptionManager",
    "page_text",
    "stops_page",
]

logger = logging.getLogger(__name__)

TRANSCRIPTION_SCHEMA_FILE = "transcription_schema.json"
SYSTEM_PROMPT_FILE = "transcription_system_prompt.txt"
PLAIN_TEXT_PROMPT_FILE = "transcription_plain_text_prompt.txt"

TRANSCRIPTION_REQUIRED_KEYS = (
    "image_analysis",
    "transcription",
    "no_transcribable_text",
    "transcription_not_possible",
)
FLAGS = ("no_transcribable_text", "transcription_not_possible")
SENTINELS: Mapping[str, str] = {
    "[no transcribable text]": "no_transcribable_text",
    "[transcription not possible]": "transcription_not_possible",
}

EMPTY_RESPONSE_RETRIES = 2
EMPTY_RESPONSE_BACKOFF_S = 0.5
# Upper bound on the requests one page may make for validation, flag and
# empty-response retries together (transport retries not counted).
MAX_PAGE_CALLS = 6


def stops_page(response: Any) -> bool:
    """Whether a response ends the page at once (content filter, refusal)."""
    return inspect_response(response).stops_page


def _truncate(text: Any, max_chars: int = 100) -> str:
    """Shorten an image analysis to about *max_chars* at a word boundary."""
    if not text:
        return "no details available"
    value = str(text).strip()
    if len(value) <= max_chars:
        return value
    truncated = value[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
    return truncated.rstrip(".,;:") + "..."


def _salvage(text: str) -> Any:
    """Parse *text*, else the last JSON object that ends at its final brace."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    last_close = text.rfind("}")
    for index in range(last_close, -1, -1):
        if text[index] != "{":
            continue
        try:
            return json.loads(text[index : last_close + 1])
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def page_text(
    text: str, image_name: str, data: Any = None, *, plain: bool = False
) -> str:
    """Return the page text of an answer, or a bracketed status message.

    *plain* answers are the text itself, minus a code fence; the two
    sentinels become status messages. JSON answers use *data* when given,
    else the parsed text; flags and a null transcription become status
    messages, and an answer without a transcription stays as given.
    """
    stripped = strip_markdown_code_block(text)
    name = image_name or "unknown_image"
    if plain:
        if stripped == "[no transcribable text]":
            return f"[{name}: no transcribable text]"
        if stripped == "[transcription not possible]":
            return f"[{name}: transcription not possible]"
        return stripped
    obj = data
    if not isinstance(obj, dict):
        if not stripped.startswith("{"):
            return stripped
        obj = _salvage(stripped)
        if not isinstance(obj, dict):
            return stripped
    if obj.get("no_transcribable_text") is True:
        reason = _truncate(obj.get("image_analysis", ""))
        return f"[{name}: no transcribable text — {reason}]"
    if obj.get("transcription_not_possible") is True:
        reason = _truncate(obj.get("image_analysis", ""))
        return f"[{name}: transcription not possible — {reason}]"
    transcription = obj.get("transcription")
    if isinstance(transcription, str):
        return transcription
    if "transcription" in obj and transcription is None:
        return f"[{name}: no transcribable text]"
    return stripped


def _recoverable(data: Any) -> bool:
    """Whether an answer that failed validation still carries a usable page."""
    if not isinstance(data, dict):
        return False
    if data.get("no_transcribable_text") is True:
        return True
    if data.get("transcription_not_possible") is True:
        return True
    return isinstance(data.get("transcription"), str)


def _load_prompt(plain: bool) -> tuple[dict[str, Any] | None, str]:
    """Return the transcription schema (None for text) and the system prompt."""
    if plain:
        return None, (PROMPTS_DIR / PLAIN_TEXT_PROMPT_FILE).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(
        (SCHEMAS_DIR / TRANSCRIPTION_SCHEMA_FILE).read_text(encoding="utf-8")
    )
    raw = (PROMPTS_DIR / SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
    return schema, render_prompt_with_schema(raw, schema.get("schema", schema))


class TranscriptionManager:
    """Transcribe page payloads with the transcription model of one item."""

    def __init__(self, phase: PhaseModel, env: CallEnv) -> None:
        self.caller = env.caller("transcription", phase)
        self.provider = self.caller.provider
        self.model_name = phase.name
        self.options = phase.options
        self.response_format: ResponseFormat = response_format_for(
            phase.response_format, self.provider, phase.name
        )
        retry = env.settings.retry
        self.rules: Mapping[str, RetryRule] = retry.schema_retries.get(
            "transcription", {}
        )
        self._jitter = (retry.jitter_min, retry.jitter_max)
        vision = phase.supports_vision
        if vision is None:
            vision = detect_capabilities(phase.name).supports_vision
        if not vision:
            logger.warning(
                "Model '%s' may not support image input; transcription may fail.",
                phase.name,
            )
        self.schema, self.system_prompt = _load_prompt(self.plain_text)
        self.invoke_kwargs = build_invoke_kwargs(
            self.provider, phase.name, phase.options, phase.service_tier
        )

    @property
    def plain_text(self) -> bool:
        """Whether the model answers in plain text (the text format)."""
        return self.response_format == "text"

    def messages(self, payload: PagePayload) -> list[Any]:
        """Return the system and user messages for one page image."""
        data = payload.base64
        mime = payload.mime_type
        block: dict[str, Any]
        if self.provider == "anthropic":
            block = {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            }
        else:
            image_url: dict[str, Any] = {"url": f"data:{mime};base64,{data}"}
            detail = resolve_request_detail(
                dict(self.options), self.provider, self.model_name, logger
            )
            if detail is not None:
                image_url["detail"] = detail
            block = {"type": "image_url", "image_url": image_url}
        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=[block]),
        ]

    def _jittered(self, rule: RetryRule, step: int) -> float:
        return rule.backoff_base * rule.backoff_multiplier**step + random.uniform(
            *self._jitter
        )

    def _request(self, counts: Mapping[str, int], calls_left: int) -> StructuredRequest:
        rule = self.rules.get("validation_failure")
        used = counts["validation_failure"]
        retries = 0
        if rule is not None and rule.enabled:
            retries = max(0, min(rule.max_attempts - used, calls_left - 1))

        def delay(retry: int) -> float:
            return self._jittered(rule, used + retry - 1) if rule else 0.0

        return StructuredRequest(
            provider=self.provider,
            response_format=self.response_format,
            schema=self.schema,
            invoke_kwargs=self.invoke_kwargs,
            required=None if self.plain_text else TRANSCRIPTION_REQUIRED_KEYS,
            validation_retries=retries,
            validation_delay=delay,
        )

    def _flags(self, text: str, data: Any) -> list[str]:
        if self.plain_text:
            flag = SENTINELS.get(strip_markdown_code_block(text))
            return [flag] if flag else []
        if not isinstance(data, dict):
            return []
        return [flag for flag in FLAGS if data.get(flag) is True]

    async def _retry_flag(
        self, flags: list[str], counts: dict[str, int], image_name: str
    ) -> bool:
        """Sleep before a retry for the first flag whose rule allows one."""
        for flag in flags:
            rule = self.rules.get(flag)
            if rule is None or not rule.enabled or counts[flag] >= rule.max_attempts:
                continue
            delay = self._jittered(rule, counts[flag])
            counts[flag] += 1
            logger.warning(
                "Flag '%s' for %s. Retrying (%d/%d) in %.2fs...",
                flag,
                image_name,
                counts[flag],
                rule.max_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            return True
        return False

    async def transcribe_payload(self, payload: PagePayload) -> dict[str, Any]:
        """Transcribe one page; failures return an entry with ``error``."""
        page = _Page(self, payload)
        try:
            return await page.run()
        except Exception as exc:  # noqa: BLE001 - one page must not fail the item
            logger.error(
                "Transcription API error for %s after retries: %s - %s",
                payload.image_name,
                type(exc).__name__,
                exc,
            )
            return page.entry(
                f"[transcription error: {exc}]",
                error=str(exc),
                error_type="api_failure",
                counts=False,
            )


class _Page:
    """The request loop of one page."""

    def __init__(self, manager: TranscriptionManager, payload: PagePayload) -> None:
        self.manager = manager
        self.payload = payload
        self.started = time.time()
        self.counts = dict.fromkeys(("validation_failure", *FLAGS), 0)

    def entry(
        self,
        transcription: str,
        *,
        error: str | None = None,
        error_type: str | None = None,
        counts: bool | None = None,
    ) -> dict[str, Any]:
        """Return the page's result entry.

        *counts* True always adds the retry counts, False never, None when
        any retry happened.
        """
        result: dict[str, Any] = {
            "image": self.payload.image_name,
            "sequence_number": self.payload.number,
            "transcription": transcription,
            "processing_time": round(time.time() - self.started, 2),
        }
        if counts or (counts is None and sum(self.counts.values()) > 0):
            result["schema_retries"] = dict(self.counts)
        if error is not None:
            result["error"] = error
            result["error_type"] = error_type
        result["provider"] = self.manager.provider
        return result

    def text_of(self, result: StructuredResult) -> str:
        return page_text(
            result.text,
            self.payload.image_name,
            result.data,
            plain=self.manager.plain_text,
        )

    async def run(self) -> dict[str, Any]:
        manager = self.manager
        name = self.payload.image_name
        label = f"Transcription for {name}"
        messages = manager.messages(self.payload)
        empty = 0
        calls = 0
        last: StructuredResult | None = None
        while calls < MAX_PAGE_CALLS:
            result = await manager.caller.call(
                messages,
                manager._request(self.counts, MAX_PAGE_CALLS - calls),
                label=label,
                stop=stops_page,
            )
            calls += 1 + result.validation_retries
            self.counts["validation_failure"] += result.validation_retries
            if result.stopped:
                outcome = inspect_response(result.response)
                logger.warning(
                    "%s stopped (%s): %s",
                    label,
                    outcome.kind,
                    outcome.reason or "no reason given",
                )
                return self.entry(
                    f"[transcription error: {outcome.message}]",
                    error=outcome.message,
                    error_type=outcome.kind,
                    counts=False,
                )
            if not result.text:
                if empty < EMPTY_RESPONSE_RETRIES:
                    empty += 1
                    logger.warning(
                        "Empty content for %s. Retrying (%d/%d)...",
                        name,
                        empty,
                        EMPTY_RESPONSE_RETRIES,
                    )
                    await asyncio.sleep(EMPTY_RESPONSE_BACKOFF_S)
                    continue
                raise ValueError("LLM API returned empty content for transcription.")
            last = result
            if result.error is not None and not _recoverable(result.data):
                logger.error(
                    "Validation retries exhausted for %s: %s. Marking page failed.",
                    name,
                    result.error,
                )
                return self.entry(
                    self.text_of(result),
                    error=f"schema validation retries exhausted: {result.error}",
                    error_type="schema_validation_exhausted",
                    counts=True,
                )
            flags = manager._flags(result.text, result.data)
            if flags and await manager._retry_flag(flags, self.counts, name):
                continue
            return self.entry(self.text_of(result))
        return self._exhausted(last)

    def _exhausted(self, last: StructuredResult | None) -> dict[str, Any]:
        """Return the entry of a page that reached the request limit."""
        if last is None:
            raise ValueError("LLM API returned empty content for transcription.")
        text = self.text_of(last)
        if self.manager.plain_text or _recoverable(last.data):
            return self.entry(text, counts=True)
        return self.entry(
            text,
            error="schema validation retries exhausted",
            error_type="schema_validation_exhausted",
            counts=True,
        )
