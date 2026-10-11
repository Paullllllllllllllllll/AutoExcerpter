"""Model calls of one role: structured calls, the retry ladder and attempts.

A :class:`ModelCaller` builds its chat model once and sends every request
through :func:`~autoexcerpter.common.structured.structured_call`. Each HTTP
attempt, retries included, runs inside :func:`autoexcerpter.ext.attempt`:
the attempt's identity supplies the key variable, ``commit`` records the
usage of a response, ``on_error`` records usage recovered from a failure,
on entry or of the request, and decides what the ladder of
:func:`~autoexcerpter.common.retry.call_with_retry` does next. Committed and
recovered usage is summed per role in :attr:`CallEnv.usage` and in every open
:func:`collect_usage` block; retry waits go to :meth:`RunEvents.waiting`.

With a recorder installed (:func:`~autoexcerpter.common.capture.use_recorder`),
each HTTP attempt runs in its own capture scope, and its
:class:`~autoexcerpter.common.capture.AttemptRecord`, named by the caller's
role (``transcription`` or ``summary``), is stored through the recorder
before the call returns, raises or retries.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel

from autoexcerpter import ext
from autoexcerpter.common.capture import (
    AttemptRecord,
    CaptureScope,
    RawKind,
    capture_attempt,
    current_recorder,
    record_attempt,
    redact_payloads,
    sdk_dump,
)
from autoexcerpter.common.rate_limit import RateLimiterRegistry
from autoexcerpter.common.responses import unwrap_raw_response
from autoexcerpter.common.retry import (
    Decision,
    RetryEvent,
    RetryHooks,
    call_with_retry,
    default_decision,
)
from autoexcerpter.common.structured import (
    PreparedRequest,
    StructuredRequest,
    StructuredResult,
    default_invoke,
    structured_call,
)
from autoexcerpter.common.usage import Usage, UsageTotals, usage_from_response
from autoexcerpter.events import NullEvents, RunEvents, Waiting
from autoexcerpter.llm.client import build_chat_model, resolve_provider
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.settings import Settings

__all__ = ["CallEnv", "HttpAttempt", "ModelCaller", "collect_usage"]

logger = logging.getLogger(__name__)

_CLIENT_ATTRIBUTES = (
    "root_async_client",
    "root_client",
    "async_client",
    "client",
    "_async_client",
    "_client",
)
"""Attributes under which LangChain chat models keep their SDK clients."""


async def _close_quietly(client: Any, method: str = "close") -> None:
    """Call ``client.<method>()``, awaiting an async closer; never raises."""
    close = getattr(client, method, None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - closing is best effort
        logger.debug("Error closing client %r: %s", type(client).__name__, exc)


class HttpAttempt:
    """The capture scope and the stored record of one HTTP attempt.

    Without a recorder in the context it captures and stores nothing.
    *request* returns the request parameters, without media payloads.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        role: str,
        attempt: int,
        label: str | None,
        request: Callable[[], Mapping[str, Any]],
    ) -> None:
        self.active = current_recorder() is not None
        self._identity = (provider, model, role, attempt, label)
        self._request = request
        self._scope: CaptureScope | None = None
        self._started = time.monotonic()

    @contextlib.contextmanager
    def capturing(self) -> Iterator[None]:
        """Capture the responses of the block into this attempt's scope."""
        self._started = time.monotonic()
        if not self.active:
            yield
            return
        with capture_attempt() as scope:
            self._scope = scope
            yield

    async def store(
        self,
        *,
        response: Any = None,
        usage: Usage | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Build the attempt's record and await the recorder with it.

        The raw body is the last response captured; without one, the SDK
        object *response* is dumped and marked ``sdk``.
        """
        if not self.active:
            return
        latency = time.monotonic() - self._started
        captured = self._scope.last if self._scope is not None else None
        raw: str | bytes | None = None
        raw_kind: RawKind | None = None
        status = getattr(error, "status_code", None)
        if captured is not None:
            raw, raw_kind, status = captured.body, "http", captured.status
        elif response is not None:
            raw, raw_kind = sdk_dump(unwrap_raw_response(response)), "sdk"
        provider, model, role, attempt, label = self._identity
        await record_attempt(
            AttemptRecord(
                provider=provider,
                model=model,
                role=role,
                attempt=attempt,
                label=label,
                request=self._request(),
                raw=raw,
                raw_kind=raw_kind,
                usage=None if usage is None else usage.as_dict(),
                error=None if error is None else str(error),
                error_type=None if error is None else type(error).__name__,
                http_status=status if isinstance(status, int) else None,
                latency=latency,
            )
        )


_SCOPE: ContextVar[tuple[UsageTotals, ...]] = ContextVar("usage_scope", default=())


@contextlib.contextmanager
def collect_usage() -> Iterator[UsageTotals]:
    """Collect, per role, the usage of the calls made inside the block.

    The block's task and the tasks it starts share the collector. An
    enclosing block also counts the calls of blocks nested in it, including
    those in child tasks; the run's totals in :attr:`CallEnv.usage` count the
    same calls as well.
    """
    totals = UsageTotals()
    token = _SCOPE.set(_SCOPE.get() + (totals,))
    try:
        yield totals
    finally:
        _SCOPE.reset(token)


@dataclass(frozen=True)
class CallEnv:
    """What the model calls of one run share.

    *limiters* holds one rate limiter per provider, *usage* the token totals
    per role, and *events* receives the retry waits. *prepared* holds callers
    built ahead of the items, one per role (see :meth:`prepare`); *callers*
    every caller built, which :meth:`aclose` closes at the end of the run.
    """

    settings: Settings = field(default_factory=Settings)
    limiters: RateLimiterRegistry = field(default_factory=RateLimiterRegistry)
    usage: UsageTotals = field(default_factory=UsageTotals)
    events: RunEvents = field(default_factory=NullEvents)
    prepared: dict[str, ModelCaller] = field(default_factory=dict)
    callers: list[ModelCaller] = field(default_factory=list)

    @classmethod
    def create(
        cls, settings: Settings | None = None, events: RunEvents | None = None
    ) -> CallEnv:
        """Return a fresh environment with the rate limits of *settings*."""
        settings = settings or Settings()
        return cls(
            settings=settings,
            limiters=RateLimiterRegistry(settings.rate_limits),
            usage=UsageTotals(),
            events=events or NullEvents(),
        )

    def prepare(self, role: str, phase: PhaseModel) -> None:
        """Build the caller of *role* now; :meth:`caller` hands it out once.

        Raises what :class:`ModelCaller` raises, for example ``OSError`` when
        the key variable is unset.
        """
        self.prepared[role] = self._build(role, phase)

    def caller(self, role: str, phase: PhaseModel) -> ModelCaller:
        """Return the prepared caller of *role* for *phase*, else a new one."""
        prepared = self.prepared.pop(role, None)
        if prepared is not None and prepared.phase == phase:
            return prepared
        return self._build(role, phase)

    def _build(self, role: str, phase: PhaseModel) -> ModelCaller:
        caller = ModelCaller(role, phase, self)
        self.callers.append(caller)
        return caller

    async def aclose(self) -> None:
        """Close the HTTP clients of every caller built; never raises."""
        callers = list(self.callers)
        self.callers.clear()
        self.prepared.clear()
        for caller in callers:
            await caller.aclose()


class ModelCaller:
    """Send the requests of one role (transcription or summary) of one item.

    The chat model is built once, here, through an attempt that is not
    entered; an attempt whose identity names another key variable gets a
    client of its own.
    """

    def __init__(self, role: str, phase: PhaseModel, env: CallEnv) -> None:
        self.role = role
        self.phase = phase
        self.env = env
        self.provider: str = resolve_provider(phase)
        self._policy = env.settings.retry.policy()
        self._clients: dict[str, BaseChatModel] = {}
        self.model: BaseChatModel = self._attempt().build()

    def _attempt(self) -> ext.Attempt[BaseChatModel]:
        return ext.attempt(
            self.role,
            self.provider,
            None,
            settings=self.env.settings,
            build_client=self.client_for,
            endpoint=self.phase.endpoint,
            model=self.phase.name,
        )

    def client_for(self, identity: ext.Identity) -> BaseChatModel:
        """Return the client for an attempt identity, building it if new."""
        client = self._clients.get(identity.api_key_env)
        if client is None:
            client = build_chat_model(
                self.phase,
                self.env.settings.timeouts,
                api_key_env=identity.api_key_env,
            )
            self._clients[identity.api_key_env] = client
        return client

    async def call(
        self,
        messages: Sequence[Any],
        request: StructuredRequest,
        *,
        label: str,
        stop: Callable[[Any], bool] | None = None,
    ) -> StructuredResult:
        """Request one answer in ``request.response_format``.

        Transient errors are retried by the ladder; the last one propagates.
        """

        numbers = itertools.count(1)

        async def invoke(prepared: PreparedRequest, sent: Sequence[Any]) -> Any:
            return await self._with_retries(prepared, sent, label, numbers)

        return await structured_call(
            self.model, messages, request, invoke=invoke, stop=stop
        )

    def _record(self, usage: Usage) -> None:
        if usage.is_empty:
            return
        self.env.usage.add(self.role, usage)
        for scope in _SCOPE.get():
            scope.add(self.role, usage)

    def _request_params(self, prepared: PreparedRequest) -> dict[str, Any]:
        """Return the request parameters of an attempt record."""
        phase = self.phase
        schema = prepared.schema or {}
        return {
            "model": phase.name,
            "provider": phase.provider,
            "endpoint": phase.endpoint,
            "base_url": phase.base_url,
            "service_tier": phase.service_tier,
            "options": dict(phase.options),
            "response_format": prepared.response_format,
            "schema": schema.get("name"),
            "enforced": prepared.enforced,
            "invoke_kwargs": redact_payloads(dict(prepared.kwargs)),
        }

    async def _with_retries(
        self,
        prepared: PreparedRequest,
        messages: Sequence[Any],
        label: str,
        numbers: Iterator[int] | None = None,
    ) -> Any:
        decisions: list[Decision] = []
        counter = numbers if numbers is not None else itertools.count(1)

        async def one_attempt() -> Any:
            attempt = self._attempt()
            async with contextlib.AsyncExitStack() as stack:
                try:
                    call = await stack.enter_async_context(attempt)
                except Exception as exc:
                    decisions.append(attempt.on_error(exc))
                    self._record(attempt.usage)
                    raise
                http = HttpAttempt(
                    provider=self.provider,
                    model=self.phase.name,
                    role=self.role,
                    attempt=next(counter),
                    label=label,
                    request=lambda: self._request_params(prepared),
                )
                try:
                    with http.capturing():
                        response = await default_invoke(
                            prepared.for_model(call.client), messages
                        )
                except Exception as exc:
                    decisions.append(call.on_error(exc))
                    self._record(call.usage)
                    await http.store(usage=call.usage, error=exc)
                    raise
                usage = usage_from_response(response)
                call.commit(usage)
                self._record(call.usage)
                await http.store(response=response, usage=usage)
                return response

        def decide(exc: BaseException) -> Decision:
            return decisions.pop() if decisions else default_decision(exc)

        hooks = RetryHooks(
            on_error=decide,
            on_retry=self._report_retry,
            gate=self.env.limiters.get(self.provider),
        )
        return await call_with_retry(
            one_attempt, policy=self._policy, hooks=hooks, label=label
        )

    async def aclose(self) -> None:
        """Close the SDK clients, and so the hooked httpx clients, of every
        chat model built; never raises."""
        models, self._clients = list(self._clients.values()), {}
        for model in models:
            for attr in _CLIENT_ATTRIBUTES:
                await _close_quietly(getattr(model, attr, None))
            # A google-genai client closes its async httpx client apart.
            aio = getattr(getattr(model, "client", None), "aio", None)
            await _close_quietly(aio, "aclose")

    def _report_retry(self, event: RetryEvent) -> None:
        self.env.events.waiting(
            Waiting(
                reason=(
                    f"{event.label or self.role}: {event.kind.value} on attempt "
                    f"{event.attempt}, {event.decision.value}"
                ),
                seconds=event.delay,
            )
        )
