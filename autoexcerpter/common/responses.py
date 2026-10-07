"""How a model response ended, read from its provider status fields.

``inspect_response`` reads every shape a LangChain response arrives in: an
``AIMessage``, or the ``with_structured_output(include_raw=True)`` dict with
its ``raw`` message and ``parsing_error``. Content-filter stops and refusals
end a page at once, because a retry would get the same answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "ResponseKind",
    "ResponseOutcome",
    "get_response_metadata",
    "inspect_response",
    "unwrap_raw_response",
]

ResponseKind = Literal["ok", "content_filter", "refusal", "max_output_tokens"]

_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "content_filter": "Response stopped by the provider's content filter",
        "refusal": "Model refused the request",
        "max_output_tokens": "Response truncated at max_output_tokens",
        "ok": "Response completed",
    }
)


@dataclass(frozen=True)
class ResponseOutcome:
    """The kind of ending and, when known, the reason the provider gave.

    ``content_filter`` and ``refusal`` stop the page; ``max_output_tokens`` and
    ``ok`` leave the response to the caller's empty-content and schema checks.
    """

    kind: ResponseKind
    reason: str | None = None

    @property
    def stops_page(self) -> bool:
        """Whether this outcome fails the page without further retries."""
        return self.kind in ("content_filter", "refusal")

    @property
    def message(self) -> str:
        """A readable error message including the reason."""
        label = _LABELS[self.kind]
        return f"{label}: {self.reason}" if self.reason else label


def unwrap_raw_response(response: Any) -> Any:
    """Return the raw message inside a ``with_structured_output`` wrapper dict."""
    if isinstance(response, dict) and "raw" in response:
        return response["raw"]
    return response


def get_response_metadata(response: Any) -> dict[str, Any]:
    """Return the ``response_metadata`` dict of a possibly wrapped response.

    Returns ``{}`` when the response carries no metadata dict.
    """
    meta = getattr(unwrap_raw_response(response), "response_metadata", None)
    return meta if isinstance(meta, dict) else {}


def _refusal_text(target: Any) -> str | None:
    """Return the refusal text of a message, ``""`` for an empty one, else None.

    Chat Completions put it in ``additional_kwargs["refusal"]``; the Responses
    API puts it in a content block of type ``refusal``.
    """
    kwargs = getattr(target, "additional_kwargs", None)
    if isinstance(kwargs, dict):
        refusal = kwargs.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()
    content = getattr(target, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "refusal":
                text = block.get("refusal")
                return text.strip() if isinstance(text, str) else ""
    return None


def _incomplete_reason(meta: dict[str, Any]) -> str | None:
    """Return ``incomplete_details.reason`` of an incomplete Responses API call."""
    if meta.get("status") != "incomplete":
        return None
    details = meta.get("incomplete_details")
    if isinstance(details, dict) and isinstance(details.get("reason"), str):
        return str(details["reason"])
    return None


def _truncation_reason(meta: dict[str, Any], incomplete: str | None) -> str | None:
    """Return the field that reports a stop at the output-token limit."""
    if incomplete == "max_output_tokens":
        return "incomplete_details.reason=max_output_tokens"
    if meta.get("stop_reason") == "max_tokens":
        return "stop_reason=max_tokens"
    if meta.get("finish_reason") == "MAX_TOKENS":
        return "finish_reason=MAX_TOKENS"
    return None


def inspect_response(response: Any) -> ResponseOutcome:
    """Classify how a LangChain model response ended.

    Reads the Responses API ``status`` and ``incomplete_details.reason``, the
    Chat Completions ``finish_reason``, the Anthropic ``stop_reason``, a
    ``parsing_error`` of class ``OpenAIRefusalError`` in an ``include_raw``
    dict, ``additional_kwargs["refusal"]`` and refusal content blocks. A Chat
    Completions ``finish_reason`` of ``length`` counts as ``ok``. Anything
    else is ``ok``.
    """
    meta = get_response_metadata(response)
    incomplete = _incomplete_reason(meta)

    if incomplete == "content_filter":
        return ResponseOutcome(
            "content_filter", "incomplete_details.reason=content_filter"
        )
    if meta.get("finish_reason") == "content_filter":
        return ResponseOutcome("content_filter", "finish_reason=content_filter")
    if meta.get("stop_reason") == "refusal":
        return ResponseOutcome("refusal", "stop_reason=refusal")

    if isinstance(response, dict) and "raw" in response:
        parsing_error = response.get("parsing_error")
        if (
            parsing_error is not None
            and type(parsing_error).__name__ == "OpenAIRefusalError"
        ):
            return ResponseOutcome("refusal", str(parsing_error).strip() or None)

    refusal = _refusal_text(unwrap_raw_response(response))
    if refusal is not None:
        return ResponseOutcome("refusal", refusal or None)

    truncation = _truncation_reason(meta, incomplete)
    if truncation is not None:
        return ResponseOutcome("max_output_tokens", truncation)
    return ResponseOutcome("ok")
