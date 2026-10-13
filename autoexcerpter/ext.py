"""Extension seam: startup hooks and the per-call attempt context.

These hooks pass everything through: no extra arguments (and so no argv
words for a printed command), settings blocks or wizard steps, no startup
work, nothing reserved, the provider's key
variable from the settings, no client options, and retry decisions from the
shared error classifier. The core wraps every model call, each retry and
batch call included, in :func:`attempt` and builds every client through it.
Plan-time key checks ask :func:`key_variables` which variables serve a call.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Final

from autoexcerpter.common.retry import Decision, default_decision
from autoexcerpter.common.usage import Usage, usage_from_exception
from autoexcerpter.common.wizard import Step
from autoexcerpter.settings import BlockParser, Settings
from autoexcerpter.spec import CommandSpec, RunSpec

__all__ = [
    "CLIENT_KINDS",
    "Attempt",
    "ClientOptions",
    "Decision",
    "Identity",
    "attempt",
    "batch_allowed",
    "client_options",
    "extra_argv",
    "key_variables",
    "register_args",
    "settings_blocks",
    "startup",
    "wizard_steps",
]


def register_args(parser: argparse.ArgumentParser) -> None:
    """Add extension arguments to a command parser; none here.

    The commands that call providers register them
    (:data:`autoexcerpter.spec.EXTENSION_COMMANDS`).
    """


def extra_argv(args: argparse.Namespace | None) -> tuple[str, ...]:
    """Return the argv words of the extension's options given in *args*.

    Only options given explicitly count, so a printed command reproduces
    them; none here. The UI appends them to the commands that take them.
    """
    return ()


def settings_blocks() -> Mapping[str, BlockParser]:
    """Return the parsers of extra top-level settings blocks; none here.

    Each parser receives the block's YAML value and returns what
    ``Settings.extensions`` stores under the block name; it raises when the
    value is invalid.
    """
    return MappingProxyType({})


def wizard_steps() -> Sequence[Step]:
    """Return extra wizard steps, asked after the tool's own; none here.

    Each step names the options it sets; they must be options of
    :data:`autoexcerpter.spec.OPTIONS`.
    """
    return ()


def startup(spec: CommandSpec, *, settings: Settings, args: argparse.Namespace) -> None:
    """Prepare a command before its first model call; nothing to do here.

    *spec* is a run, a batch check or a batch cancel. The CLI and the UI
    call this once per command, after planning and before any model or
    batch call; a dry run calls it too, a real run with nothing to process
    does not. *args* holds the parsed command line, extension arguments
    included. Under ``spec.run.dry_run`` an extension may only validate: it
    makes no calls out and writes nothing. A
    :class:`~autoexcerpter.settings.SettingsError` stops the command as a
    configuration error.
    """


def batch_allowed(
    spec: RunSpec,
    *,
    settings: Settings,
    args: argparse.Namespace | None = None,
) -> bool:
    """Whether batch submission is allowed for ``spec``; always here.

    *args* holds the parsed command line where the front end has one. An
    extension may refuse with a reason instead of returning False: the
    message of a :class:`~autoexcerpter.settings.SettingsError` it raises
    is shown to the user.
    """
    return True


CLIENT_KINDS: Final = ("chat", "batch")
"""The client kinds: chat models and batch API clients."""


def key_variables(
    provider: str,
    endpoint: str | None = None,
    *,
    kind: str = "chat",
    settings: Settings,
    args: argparse.Namespace | None = None,
) -> tuple[str, ...]:
    """Return the key variables any one of which serves these calls.

    The calls are those of *provider* (or of the named *endpoint*) with a
    client of *kind* (:data:`CLIENT_KINDS`). An empty tuple means no
    variable is required. Here: the settings' key variable of the provider.

    Raises:
        SettingsError: The settings name no key variable for the provider.
    """
    return (settings.api_key_env(provider, endpoint),)


@dataclass(frozen=True)
class Identity:
    """Who a call is made as, fixed when the attempt starts.

    ``group`` is an optional accounting group; ``role`` is the call's role,
    such as transcription or summary; ``model`` names the model called;
    ``kind`` is the kind of client the call uses (:data:`CLIENT_KINDS`).
    """

    provider: str
    api_key_env: str
    role: str
    group: str | None = None
    model: str | None = None
    kind: str = "chat"


@dataclass(frozen=True)
class ClientOptions:
    """How a client builder adjusts the client of an identity.

    ``api_key`` replaces the key read from the identity's key variable;
    batch clients honor it.
    """

    api_key: str | None = None


def client_options(identity: Identity) -> ClientOptions:
    """Return the client options of *identity*; none here."""
    return ClientOptions()


class Attempt[C]:
    """One model call attempt: its identity, client and recorded usage."""

    def __init__(
        self,
        identity: Identity,
        estimate: int | None,
        build_client: Callable[[Identity], C],
    ) -> None:
        self._identity = identity
        self.estimate = estimate
        self._build_client = build_client
        self._client: C | None = None
        self.committed = Usage()
        self.recovered = Usage()
        self.errors: list[BaseException] = []
        self.released = False

    @property
    def identity(self) -> Identity:
        """The identity fixed at entry."""
        return self._identity

    @property
    def client(self) -> C:
        """The client built for the identity."""
        if self._client is None:
            raise RuntimeError("the attempt has no client; enter or build it first")
        return self._client

    def build(self) -> C:
        """Build and keep the client for the identity, without entering."""
        client = self._build_client(self._identity)
        self._client = client
        return client

    async def refresh(self) -> C:
        """Build the client for the identity again and return it."""
        return self.build()

    def commit(self, usage: Usage | None) -> None:
        """Record the usage of a successful call."""
        if usage is not None:
            self.committed = self.committed + usage

    def on_error(self, exc: BaseException) -> Decision:
        """Record usage recovered from a failed call and decide what follows."""
        self.errors.append(exc)
        usage = usage_from_exception(exc)
        if usage is not None:
            self.recovered = self.recovered + usage
        return default_decision(exc)

    @property
    def usage(self) -> Usage:
        """Committed and recovered usage together."""
        return self.committed + self.recovered

    async def __aenter__(self) -> Attempt[C]:
        try:
            await self.refresh()
        except BaseException:
            self._release()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._release()

    def _release(self) -> None:
        self.released = True


def attempt[C](
    role: str,
    provider: str,
    estimate: int | None = None,
    *,
    settings: Settings,
    build_client: Callable[[Identity], C],
    endpoint: str | None = None,
    model: str | None = None,
    kind: str = "chat",
) -> Attempt[C]:
    """Open an attempt context for one model call.

    Use as ``async with attempt(...) as call:``. The identity takes the key
    variable of ``provider`` (or of the named ``endpoint``) from
    ``settings`` and the client *kind* (:data:`CLIENT_KINDS`);
    ``build_client`` builds the client for that identity.
    """
    if kind not in CLIENT_KINDS:
        raise ValueError(f"Unknown client kind {kind!r}; expected {CLIENT_KINDS}")
    identity = Identity(
        provider=provider,
        api_key_env=settings.api_key_env(provider, endpoint),
        role=role,
        model=model,
        kind=kind,
    )
    return Attempt(identity, estimate, build_client)
