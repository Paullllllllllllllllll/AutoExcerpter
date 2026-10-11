"""Verbatim provider responses captured at the HTTP transport.

A pipeline opens one :func:`capture_attempt` scope per provider attempt. The
response hooks installed on every httpx client hand each response body, as it
came over the wire, to the scope of the current attempt, which a context
variable holds; outside a scope nothing is captured. Scopes stay separate per
asyncio task and follow ``asyncio.to_thread``, which copies the context.

The hooks read the whole body before the SDK parses it, so a streamed
response inside a scope arrives buffered.

The module also defines the attempt record a pipeline builds from a scope and
the recorder protocol through which the record is stored; a pipeline installs
its recorder with :func:`use_recorder` for the calls it makes.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

RawKind = Literal["http", "sdk"]

_PAYLOAD_MIN_CHARS = 1024


@dataclass(frozen=True)
class CapturedResponse:
    """One HTTP response seen inside a capture scope."""

    status: int
    method: str
    url: str
    body: bytes
    elapsed: float | None

    @property
    def text(self) -> str:
        """Return the body decoded as UTF-8 (invalid bytes replaced)."""
        return self.body.decode("utf-8", errors="replace")


class CaptureScope:
    """The responses captured during one provider attempt, in arrival order."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._responses: list[CapturedResponse] = []

    def add(self, response: CapturedResponse) -> None:
        """Append *response*; safe from any thread."""
        with self._lock:
            self._responses.append(response)

    @property
    def responses(self) -> list[CapturedResponse]:
        """Return a copy of the captured responses."""
        with self._lock:
            return list(self._responses)

    @property
    def last(self) -> CapturedResponse | None:
        """Return the latest captured response, or None."""
        with self._lock:
            return self._responses[-1] if self._responses else None


_SCOPE: ContextVar[CaptureScope | None] = ContextVar("capture_scope", default=None)


def current_scope() -> CaptureScope | None:
    """Return the capture scope of the current context, or None."""
    return _SCOPE.get()


@contextlib.contextmanager
def capture_attempt() -> Iterator[CaptureScope]:
    """Open a capture scope for one provider attempt."""
    scope = CaptureScope()
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def _elapsed(response: httpx.Response) -> float | None:
    try:
        return response.elapsed.total_seconds()
    except RuntimeError:
        return None


def _store(scope: CaptureScope, response: httpx.Response) -> None:
    scope.add(
        CapturedResponse(
            status=response.status_code,
            method=response.request.method,
            url=str(response.request.url),
            body=response.content,
            elapsed=_elapsed(response),
        )
    )


def capture_response(response: httpx.Response) -> None:
    """Response hook for ``httpx.Client``: read the body into the current scope."""
    scope = _SCOPE.get()
    if scope is None:
        return
    response.read()
    _store(scope, response)


async def acapture_response(response: httpx.Response) -> None:
    """Response hook for ``httpx.AsyncClient``: read the body into the scope."""
    scope = _SCOPE.get()
    if scope is None:
        return
    await response.aread()
    _store(scope, response)


def capture_hooks() -> dict[str, list[Callable[[httpx.Response], None]]]:
    """Return ``event_hooks`` for an ``httpx.Client`` that capture responses."""
    return {"response": [capture_response]}


def async_capture_hooks() -> dict[
    str, list[Callable[[httpx.Response], Awaitable[None]]]
]:
    """Return ``event_hooks`` for an ``httpx.AsyncClient`` that capture responses."""
    return {"response": [acapture_response]}


def install_capture(client: httpx.Client | httpx.AsyncClient) -> None:
    """Add the matching capture hook to *client* once."""
    hook: Any = (
        acapture_response if isinstance(client, httpx.AsyncClient) else capture_response
    )
    hooks = client.event_hooks
    responses = list(hooks.get("response", []))
    if hook not in responses:
        responses.append(hook)
        client.event_hooks = {**hooks, "response": responses}


@dataclass(frozen=True)
class AttemptRecord:
    """One provider attempt as stored: request, verbatim response and outcome.

    ``request`` holds the call parameters without image or audio payloads
    (see :func:`redact_payloads`). ``raw`` is the captured body as bytes,
    unaltered, with ``raw_kind`` ``'http'``, or a dump of the SDK object with
    ``'sdk'`` where no hook could see the body. ``latency`` is in seconds.
    """

    provider: str
    model: str
    role: str
    attempt: int
    label: str | None = None
    request: Mapping[str, Any] = field(default_factory=dict)
    raw: str | bytes | None = None
    raw_kind: RawKind | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    latency: float | None = None


class AttemptRecorder(Protocol):
    """Stores attempt records; returns the stored response id, or None."""

    async def record(self, record: AttemptRecord) -> int | None: ...


_RECORDER: ContextVar[AttemptRecorder | None] = ContextVar(
    "attempt_recorder", default=None
)


def current_recorder() -> AttemptRecorder | None:
    """Return the recorder installed for the current context, or None."""
    return _RECORDER.get()


@contextlib.contextmanager
def use_recorder(recorder: AttemptRecorder | None) -> Iterator[None]:
    """Install *recorder* for the calls made inside the block."""
    token = _RECORDER.set(recorder)
    try:
        yield
    finally:
        _RECORDER.reset(token)


async def record_attempt(record: AttemptRecord) -> int | None:
    """Store *record* through the current recorder; None without one."""
    recorder = _RECORDER.get()
    if recorder is None:
        return None
    return await recorder.record(record)


def sdk_dump(obj: Any) -> str:
    """Return a JSON dump of an SDK response object."""
    dump = getattr(obj, "model_dump_json", None)
    if callable(dump):
        return str(dump())
    to_json = getattr(obj, "to_json", None)
    if callable(to_json):
        return str(to_json())
    return json.dumps(obj, ensure_ascii=False, default=str)


def _is_base64(text: str) -> bool:
    if len(text) < _PAYLOAD_MIN_CHARS:
        return False
    try:
        base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def redact_payloads(value: Any) -> Any:
    """Return *value* with binary payloads replaced by a short placeholder.

    Data URLs, base64 strings of at least 1,024 characters and bytes are
    replaced, recursively through mappings, lists and tuples.
    """
    if isinstance(value, bytes | bytearray):
        return f"<omitted {len(value)} bytes>"
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:256]:
            head = value.split(",", 1)[0]
            return f"<omitted {head}, {len(value)} chars>"
        if _is_base64(value):
            return f"<omitted base64, {len(value)} chars>"
        return value
    if isinstance(value, Mapping):
        return {key: redact_payloads(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact_payloads(item) for item in value]
    return value
