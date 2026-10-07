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
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel

from autoexcerpter import ext
from autoexcerpter.common.rate_limit import RateLimiterRegistry
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

__all__ = ["CallEnv", "ModelCaller", "collect_usage"]

logger = logging.getLogger(__name__)

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
    built ahead of the items, one per role (see :meth:`prepare`).
    """

    settings: Settings = field(default_factory=Settings)
    limiters: RateLimiterRegistry = field(default_factory=RateLimiterRegistry)
    usage: UsageTotals = field(default_factory=UsageTotals)
    events: RunEvents = field(default_factory=NullEvents)
    prepared: dict[str, ModelCaller] = field(default_factory=dict)

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
        self.prepared[role] = ModelCaller(role, phase, self)

    def caller(self, role: str, phase: PhaseModel) -> ModelCaller:
        """Return the prepared caller of *role* for *phase*, else a new one."""
        prepared = self.prepared.pop(role, None)
        if prepared is not None and prepared.phase == phase:
            return prepared
        return ModelCaller(role, phase, self)


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

        async def invoke(prepared: PreparedRequest, sent: Sequence[Any]) -> Any:
            return await self._with_retries(prepared, sent, label)

        return await structured_call(
            self.model, messages, request, invoke=invoke, stop=stop
        )

    def _record(self, usage: Usage) -> None:
        if usage.is_empty:
            return
        self.env.usage.add(self.role, usage)
        for scope in _SCOPE.get():
            scope.add(self.role, usage)

    async def _with_retries(
        self, prepared: PreparedRequest, messages: Sequence[Any], label: str
    ) -> Any:
        decisions: list[Decision] = []

        async def one_attempt() -> Any:
            attempt = self._attempt()
            async with contextlib.AsyncExitStack() as stack:
                try:
                    call = await stack.enter_async_context(attempt)
                except Exception as exc:
                    decisions.append(attempt.on_error(exc))
                    self._record(attempt.usage)
                    raise
                try:
                    response = await default_invoke(
                        prepared.for_model(call.client), messages
                    )
                except Exception as exc:
                    decisions.append(call.on_error(exc))
                    self._record(call.usage)
                    raise
                call.commit(usage_from_response(response))
                self._record(call.usage)
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
