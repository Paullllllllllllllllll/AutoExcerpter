"""Request one model answer in a response format: schema, json or text.

``schema`` has the provider enforce a JSON schema, ``json`` asks for JSON in the
prompt only and parses the answer, ``text`` returns plain text. Both JSON
formats check required keys and retry invalid answers within their own budget.
The caller builds the messages and wraps each call (retries for transport
errors, rate limits, accounting) through the ``invoke`` hook, which receives
the prepared request and can rebuild it for another client instance.

A schema spec is either a bare JSON schema or a mapping with ``name``,
``strict`` and ``schema`` keys.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

ResponseFormat = Literal["schema", "json", "text"]
RESPONSE_FORMATS: tuple[ResponseFormat, ...] = ("schema", "json", "text")

Invoker = Callable[["PreparedRequest", Sequence[Any]], Awaitable[Any]]

_ANTHROPIC_SCHEMA_LIMITS = (
    "Tool schema contains too many conditional branches",
    "reduce the use of anyOf constructs",
    "anyOf constructs (limit: 8)",
    "too many parameters with union types",
    "parameters with unions",
)

_FENCED_BLOCK_RE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)\n\s*```", re.DOTALL)
_MAX_SALVAGE_CANDIDATES = 1000


def schema_body(spec: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the bare JSON schema of *spec*, or None when it has none."""
    if not isinstance(spec, Mapping):
        return None
    body = spec.get("schema", spec)
    if not isinstance(body, Mapping) or not body:
        return None
    return dict(body)


def required_keys(spec: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the top-level ``required`` keys of *spec*'s schema."""
    body = schema_body(spec)
    required = body.get("required") if body else None
    if not isinstance(required, list):
        return ()
    return tuple(key for key in required if isinstance(key, str))


def build_text_format(
    spec: Mapping[str, Any] | None, default_name: str = "json_schema"
) -> dict[str, Any] | None:
    """Return the OpenAI Responses ``text.format`` entry for *spec*."""
    body = schema_body(spec)
    if spec is None or body is None:
        return None
    return {
        "type": "json_schema",
        "name": spec.get("name", default_name),
        "schema": body,
        "strict": bool(spec.get("strict", True)),
    }


def build_response_format(spec: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the Chat Completions ``response_format`` entry for *spec*."""
    text_format = build_text_format(spec)
    if text_format is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {k: v for k, v in text_format.items() if k != "type"},
    }


def sanitize_schema_for_anthropic(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *schema* with union types reduced to their first non-null
    type, which Anthropic's ``json_schema`` method requires."""
    result: dict[str, Any] = copy.deepcopy(dict(schema))

    def _walk(node: dict[str, Any]) -> None:
        if isinstance(node.get("type"), list):
            non_null = [t for t in node["type"] if t != "null"]
            node["type"] = non_null[0] if non_null else "string"
        for value in node.values():
            if isinstance(value, dict):
                _walk(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _walk(item)

    _walk(result)
    return result


@dataclass(frozen=True)
class PreparedRequest:
    """What to invoke for one call: the runnable, its kwargs, and whether the
    provider enforces the schema.

    The remaining fields are the inputs of :func:`prepare_request`, so
    :meth:`for_model` can build the same request for another client.
    """

    runnable: Any
    kwargs: dict[str, Any]
    enforced: bool
    model: Any = None
    provider: str = ""
    response_format: ResponseFormat = "text"
    schema: Mapping[str, Any] | None = None
    invoke_kwargs: Mapping[str, Any] | None = None

    def for_model(self, model: Any) -> PreparedRequest:
        """Return this request built for *model*; itself when it is the same."""
        if model is self.model:
            return self
        return prepare_request(
            model, self.provider, self.response_format, self.schema, self.invoke_kwargs
        )


def _bind(model: Any, kwargs: Mapping[str, Any]) -> Any:
    """Copy *kwargs* onto the model's fields; a structured-output wrapper does
    not forward invoke-time kwargs to the inner model."""
    if not kwargs:
        return model
    return model.model_copy(update=dict(kwargs))


def _titled(body: dict[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    if "title" in body:
        return body
    return {**body, "title": spec.get("name", "structured_output")}


def prepare_request(
    model: Any,
    provider: str,
    response_format: ResponseFormat,
    schema: Mapping[str, Any] | None,
    invoke_kwargs: Mapping[str, Any] | None = None,
) -> PreparedRequest:
    """Build the runnable and kwargs for one call in *response_format*.

    Only ``schema`` changes the request: OpenAI gets ``text.format``, custom
    OpenAI-compatible endpoints ``response_format``, Google
    ``response_mime_type`` and ``response_schema`` (left out while
    ``thinking_config`` is set), Anthropic a ``json_schema`` structured-output
    wrapper on the sanitized schema, OpenRouter a structured-output wrapper
    with the default method.
    """
    kwargs = dict(invoke_kwargs or {})

    def prepared(runnable: Any, enforced: bool) -> PreparedRequest:
        return PreparedRequest(
            runnable,
            kwargs,
            enforced,
            model=model,
            provider=provider,
            response_format=response_format,
            schema=schema,
            invoke_kwargs=invoke_kwargs,
        )

    body = schema_body(schema)
    if response_format != "schema" or schema is None or body is None:
        return prepared(model, False)
    if provider == "openai":
        text = dict(kwargs.get("text") or {})
        text["format"] = build_text_format(schema)
        kwargs["text"] = text
        return prepared(model, True)
    if provider == "custom":
        kwargs["response_format"] = build_response_format(schema)
        return prepared(model, True)
    if provider == "google":
        if "thinking_config" in kwargs:
            logger.debug("Google thinking is active; response_schema is not sent")
            return prepared(model, False)
        kwargs.setdefault("response_mime_type", "application/json")
        kwargs.setdefault("response_schema", body)
        return prepared(model, True)
    if provider == "anthropic":
        anthropic_schema = _titled(sanitize_schema_for_anthropic(body), schema)
        runnable = _bind(model, kwargs).with_structured_output(
            anthropic_schema, method="json_schema", include_raw=True
        )
        return prepared(runnable, True)
    if provider == "openrouter":
        runnable = _bind(model, kwargs).with_structured_output(
            _titled(body, schema), include_raw=True
        )
        return prepared(runnable, True)
    logger.debug("No schema enforcement for provider %r", provider)
    return prepared(model, False)


def is_schema_rejection(exc: BaseException) -> bool:
    """Whether Anthropic rejected a structured-output schema as too complex."""
    message = str(exc)
    return any(marker in message for marker in _ANTHROPIC_SCHEMA_LIMITS)


def _from_aimessage(data: Any) -> str | None:
    if not isinstance(data, AIMessage):
        return None
    parts: list[str] = []
    content = data.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("output_text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
    elif isinstance(content, str) and content.strip():
        parts.append(content)
    result = "".join(parts).strip()
    if not result:
        logger.warning("Empty content in the model response message")
    return result


def _from_output_attribute(data: Any) -> str | None:
    try:
        text = getattr(data, "output_text", None)
    except Exception as exc:  # noqa: BLE001 - exotic SDK objects may raise
        logger.debug("output_text attribute access failed: %s", exc)
        return None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _dump_json(value: Any) -> str | None:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _from_structured_wrapper(data: Any) -> str | None:
    if not isinstance(data, dict) or "raw" not in data:
        return None
    parsed = data.get("parsed")
    if isinstance(parsed, dict):
        dumped = _dump_json(parsed)
        if dumped is not None:
            return dumped
    elif parsed is not None:
        model_dump = getattr(parsed, "model_dump", None)
        if callable(model_dump):
            dumped = _dump_json(model_dump())
            if dumped is not None:
                return dumped
    raw = data.get("raw")
    if raw is None:
        return None
    text = _from_aimessage(raw)
    if text:
        return text
    tool_calls = getattr(raw, "tool_calls", None)
    if tool_calls:
        first = tool_calls[0]
        args = first.get("args") if isinstance(first, dict) else None
        if args is not None:
            dumped = _dump_json(args)
            if dumped is not None:
                return dumped
    return text


def _from_dict(data: Any) -> str | None:
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        text: str = data["output_text"].strip()
        if text:
            return text
    return None


def _from_nested_output(data: Any) -> str | None:
    obj = data
    if not isinstance(obj, dict):
        convert = getattr(data, "to_dict", None) or getattr(data, "model_dump", None)
        if callable(convert):
            try:
                obj = convert()
            except Exception as exc:  # noqa: BLE001 - best-effort conversion
                logger.warning("Could not convert the response to a dict: %s", exc)
                return None
    output = obj.get("output") if isinstance(obj, dict) else None
    parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") in (
                    "output_text",
                    "text",
                ):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
    result = "".join(parts).strip()
    if not result:
        logger.warning("Empty content in the model response output list")
    return result


_EXTRACTORS: tuple[Callable[[Any], str | None], ...] = (
    _from_aimessage,
    _from_output_attribute,
    _from_structured_wrapper,
    _from_dict,
    _from_nested_output,
)


def extract_output_text(response: Any) -> str:
    """Return the answer text of a model response in any supported shape.

    Handles chat messages, ``with_structured_output(include_raw=True)``
    wrappers (parsed value, raw text, or first tool-call arguments), and
    Responses API objects or dicts.
    """
    for extractor in _EXTRACTORS:
        result = extractor(response)
        if result is not None:
            return result
    logger.warning("Could not extract output text from the response")
    return ""


def strip_code_fence(text: str) -> str:
    """Remove a Markdown code fence that wraps the whole of *text*."""
    stripped = text.strip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        elif stripped[:7].lower() == "```json":
            stripped = stripped[7:]
        else:
            stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return None


def _loads_object(text: str) -> dict[str, Any] | None:
    value = _loads(text)
    return value if isinstance(value, dict) else None


def _salvage_last_object(text: str) -> dict[str, Any] | None:
    """Return the last parseable JSON object that ends at the final brace."""
    last_close = text.rfind("}")
    attempts = 0
    for index in range(last_close, -1, -1):
        if text[index] != "{":
            continue
        obj = _loads_object(text[index : last_close + 1])
        if obj is not None:
            return obj
        attempts += 1
        if attempts >= _MAX_SALVAGE_CANDIDATES:
            return None
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a prompted answer.

    Tries the text as is, without its wrapping fence, the last fenced block,
    the span from the first ``{`` to the last ``}``, and finally the last
    parseable object before the closing brace.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    unfenced = strip_code_fence(stripped)
    candidates = [stripped, unfenced]
    blocks = _FENCED_BLOCK_RE.findall(stripped)
    if blocks:
        candidates.append(str(blocks[-1]).strip())
    for candidate in candidates:
        obj = _loads_object(candidate)
        if obj is not None:
            return obj
    first, last = unfenced.find("{"), unfenced.rfind("}")
    if 0 <= first < last:
        obj = _loads_object(unfenced[first : last + 1])
        if obj is not None:
            return obj
    return _salvage_last_object(unfenced)


def validate_json(
    text: str, required: Sequence[str] = ()
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse *text* and check *required* keys.

    Returns the parsed object (or None) and the failure reason (or None).
    """
    obj = parse_json_object(text)
    if obj is None:
        value = _loads(strip_code_fence(text or ""))
        if value is not None:
            return None, f"expected JSON object, got {type(value).__name__}"
        return None, "invalid JSON"
    missing = sorted(set(required) - obj.keys())
    if missing:
        return obj, f"missing keys: {', '.join(missing)}"
    return obj, None


@dataclass(frozen=True)
class StructuredRequest:
    """One request in a response format.

    ``required`` defaults to the schema's top-level ``required`` keys.
    ``validation_retries`` bounds the extra calls made for answers that fail
    JSON validation; ``validation_delay`` maps the retry number (from 1) to
    the seconds to wait before it.
    """

    provider: str
    response_format: ResponseFormat
    schema: Mapping[str, Any] | None = None
    invoke_kwargs: Mapping[str, Any] | None = None
    required: tuple[str, ...] | None = None
    validation_retries: int = 0
    validation_delay: Callable[[int], float] | None = None

    def __post_init__(self) -> None:
        if self.response_format not in RESPONSE_FORMATS:
            raise ValueError(f"Unknown response format: {self.response_format!r}")
        if self.response_format == "schema" and schema_body(self.schema) is None:
            raise ValueError("Response format 'schema' needs a JSON schema")
        if self.validation_retries < 0:
            raise ValueError("validation_retries must not be negative")

    @property
    def required_keys(self) -> tuple[str, ...]:
        """The keys a JSON answer must carry."""
        if self.required is not None:
            return self.required
        return required_keys(self.schema)


@dataclass(frozen=True)
class StructuredResult:
    """The outcome of a structured call.

    ``text`` is the answer text (a JSON string for the JSON formats); ``data``
    the parsed object; ``error`` the validation failure of the last answer.
    ``responses`` holds every response received, for usage totals.
    ``fallback`` marks an Anthropic schema that was rejected as too complex
    and replaced by prompted JSON; ``stopped`` an answer the caller's stop
    predicate ended.
    """

    response_format: ResponseFormat
    text: str
    data: dict[str, Any] | None
    error: str | None
    response: Any
    responses: tuple[Any, ...]
    enforced: bool
    fallback: bool = False
    stopped: bool = False
    validation_retries: int = 0

    @property
    def ok(self) -> bool:
        """Whether the answer is usable."""
        return self.error is None and not self.stopped


async def default_invoke(request: PreparedRequest, messages: Sequence[Any]) -> Any:
    """Call the request's runnable asynchronously, in a worker thread if it is
    sync-only."""
    runnable, kwargs = request.runnable, request.kwargs
    ainvoke = getattr(runnable, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(list(messages), **kwargs)
    return await asyncio.to_thread(runnable.invoke, list(messages), **kwargs)


async def _invoke_once(
    model: Any,
    messages: Sequence[Any],
    request: StructuredRequest,
    call: Invoker,
    fallback: bool,
) -> tuple[Any, PreparedRequest, bool]:
    """Make one call, switching to prompted JSON on an Anthropic schema
    rejection."""
    response_format: ResponseFormat = "json" if fallback else request.response_format
    prepared = prepare_request(
        model, request.provider, response_format, request.schema, request.invoke_kwargs
    )
    try:
        response = await call(prepared, messages)
    except Exception as exc:
        if not (
            prepared.enforced
            and request.provider == "anthropic"
            and is_schema_rejection(exc)
        ):
            raise
        logger.warning(
            "Anthropic rejected the schema as too complex; using prompted JSON: %s",
            exc,
        )
        return await _invoke_once(model, messages, request, call, True)
    return response, prepared, fallback


async def structured_call(
    model: Any,
    messages: Sequence[Any],
    request: StructuredRequest,
    *,
    invoke: Invoker | None = None,
    stop: Callable[[Any], bool] | None = None,
) -> StructuredResult:
    """Request one answer from *model* in ``request.response_format``.

    An empty answer, an answer for which *stop* returns True, and a ``text``
    answer return at once. A JSON answer that fails validation is requested
    again up to ``request.validation_retries`` times. Exceptions from
    *invoke* propagate, except an Anthropic schema rejection, which falls
    back to prompted JSON.
    """
    call = invoke or default_invoke
    responses: list[Any] = []
    fallback = False
    retries = 0
    while True:
        response, prepared, fallback = await _invoke_once(
            model, messages, request, call, fallback
        )
        responses.append(response)
        text = extract_output_text(response)
        stopped = stop is not None and stop(response)
        data: dict[str, Any] | None = None
        error: str | None = None
        if not text:
            error = "empty response"
        elif request.response_format != "text" and not stopped:
            data, error = validate_json(text, request.required_keys)
        done = (
            error is None
            or stopped
            or not text
            or retries >= request.validation_retries
        )
        if done:
            return StructuredResult(
                response_format=request.response_format,
                text=text,
                data=data,
                error=error,
                response=response,
                responses=tuple(responses),
                enforced=prepared.enforced,
                fallback=fallback,
                stopped=stopped,
                validation_retries=retries,
            )
        retries += 1
        delay = request.validation_delay(retries) if request.validation_delay else 0
        logger.warning(
            "Invalid JSON answer (%s); retrying (%d/%d)",
            error,
            retries,
            request.validation_retries,
        )
        if delay > 0:
            await asyncio.sleep(delay)


__all__ = [
    "RESPONSE_FORMATS",
    "Invoker",
    "PreparedRequest",
    "ResponseFormat",
    "StructuredRequest",
    "StructuredResult",
    "build_response_format",
    "build_text_format",
    "default_invoke",
    "extract_output_text",
    "is_schema_rejection",
    "parse_json_object",
    "prepare_request",
    "required_keys",
    "sanitize_schema_for_anthropic",
    "schema_body",
    "strip_code_fence",
    "structured_call",
    "validate_json",
]
