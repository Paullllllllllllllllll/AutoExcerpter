"""The client factory: LangChain chat models for each provider.

Supports OpenAI (Responses API), Anthropic, Google, OpenRouter and named
OpenAI-compatible custom endpoints. The run spec has already settled each
phase's provider and stripped any ``provider:`` prefix from the model name.
Provider classes are imported inside the factory functions. SDK retries are
off (``max_retries=0``): the retry ladder of :mod:`autoexcerpter.llm.caller`
sees every attempt.

Every chat model sends its requests through httpx clients that carry the
capture hooks of :mod:`autoexcerpter.common.capture`, so an attempt's
request and response bodies reach its capture scope as they went over HTTP:
``ChatOpenAI`` takes them as ``http_client`` and ``http_async_client``;
``ChatAnthropic`` gets copies of its cached SDK clients on hooked httpx
clients; ``ChatGoogleGenerativeAI`` passes ``client_args`` to both httpx
clients of google-genai, with a transport that keeps async calls on httpx.
"""

from __future__ import annotations

import functools
import logging
import os
import ssl
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from autoexcerpter.common.capture import (
    acapture_response,
    async_capture_hooks,
    capture_hooks,
    capture_request,
    capture_response,
)
from autoexcerpter.common.http_timeouts import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    build_httpx_timeout,
)
from autoexcerpter.llm.types import PhaseModel
from autoexcerpter.providers import PROVIDERS
from autoexcerpter.settings import DEFAULT_API_KEYS, Timeouts

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class LLMConfig:
    """What the factory needs to build one chat model.

    *api_key_env* None takes the provider's default variable; a ``custom``
    endpoint needs one, and its *base_url*. *timeout* is the read timeout;
    the OpenAI-family clients also get the connect, write and pool timeouts.
    """

    model: str
    provider: str
    api_key_env: str | None = None
    base_url: str | None = None
    timeout: float = 900
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    write_timeout: float = DEFAULT_WRITE_TIMEOUT
    pool_timeout: float = DEFAULT_POOL_TIMEOUT


def default_key_env(provider: str) -> str | None:
    """Return the default key variable of *provider* (None for ``custom``)."""
    return DEFAULT_API_KEYS.get(provider)


def _get_api_key(provider: str, api_key_env: str | None = None) -> str:
    """Return the API key from *api_key_env* or the provider's default variable.

    Raises:
        OSError: No variable is known, or it is unset.
    """
    env_key = api_key_env or default_key_env(provider)
    if env_key is None:
        raise OSError(
            "Custom endpoint API key not found: the endpoint names no "
            "api_key_env in the settings."
        )
    api_key = os.environ.get(env_key)
    if not api_key:
        raise OSError(
            f"API key not found for provider '{provider}'. "
            f"Please set the {env_key} environment variable."
        )
    return api_key


def get_chat_model(config: LLMConfig) -> BaseChatModel:
    """Create the LangChain chat model of *config*, without SDK retries.

    Raises:
        ImportError: The provider package is not installed.
        OSError: The API key is not found.
        ValueError: The provider is unknown or a custom endpoint has no URL.
    """
    provider = config.provider
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    api_key = _get_api_key(provider, config.api_key_env)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "timeout": config.timeout,
        "max_retries": 0,
    }
    phases = {
        "connect": config.connect_timeout,
        "write": config.write_timeout,
        "pool": config.pool_timeout,
    }
    if provider == "openai":
        return _create_openai_model(api_key, kwargs, phases)
    if provider == "anthropic":
        return _create_anthropic_model(api_key, kwargs)
    if provider == "google":
        return _create_google_model(api_key, kwargs)
    if provider == "openrouter":
        return _create_openrouter_model(api_key, kwargs, phases)
    if not config.base_url:
        raise ValueError("provider custom requires the endpoint's base_url")
    return _create_custom_model(api_key, kwargs, config.base_url, phases)


def _apply_per_phase_timeout(kwargs: dict[str, Any], phases: dict[str, Any]) -> None:
    """Rewrite the scalar ``timeout`` in *kwargs* as per-phase httpx budgets.

    The scalar stays the read budget; *phases* holds the ``connect``,
    ``write`` and ``pool`` budgets. A scalar alone would set connect too.
    """
    kwargs["timeout"] = build_httpx_timeout(kwargs["timeout"], **phases)


@functools.cache
def tls_context() -> ssl.SSLContext:
    """Return the TLS context of every hooked httpx client, built once.

    It is the context httpx builds by default; building one loads the CA
    bundle, which takes about a quarter second, so the clients share it.
    """
    return httpx.create_ssl_context()


def _hooked_openai_clients(kwargs: dict[str, Any]) -> None:
    """Add httpx clients with the capture hooks and the kwargs' timeout."""
    import openai

    timeout = kwargs["timeout"]
    kwargs["http_client"] = openai.DefaultHttpxClient(
        event_hooks=capture_hooks(), timeout=timeout, verify=tls_context()
    )
    kwargs["http_async_client"] = openai.DefaultAsyncHttpxClient(
        event_hooks=async_capture_hooks(), timeout=timeout, verify=tls_context()
    )


def _create_openai_model(
    api_key: str, kwargs: dict[str, Any], phases: dict[str, Any]
) -> BaseChatModel:
    """Create an OpenAI chat model instance (Responses API)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for OpenAI models. "
            "Install with: pip install langchain-openai"
        ) from None

    kwargs["api_key"] = api_key
    _apply_per_phase_timeout(kwargs, phases)
    _hooked_openai_clients(kwargs)
    kwargs["use_responses_api"] = True
    logger.debug("Creating OpenAI model: %s", kwargs.get("model"))
    return ChatOpenAI(**kwargs)


def _create_anthropic_model(api_key: str, kwargs: dict[str, Any]) -> BaseChatModel:
    """Create an Anthropic chat model instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic package is required for Anthropic models. "
            "Install with: pip install langchain-anthropic"
        ) from None

    kwargs["api_key"] = api_key
    logger.debug("Creating Anthropic model: %s", kwargs.get("model"))
    model = ChatAnthropic(**kwargs)
    _hook_anthropic(model)
    return model


def _hook_anthropic(model: Any) -> None:
    """Replace the SDK clients a ``ChatAnthropic`` caches with hooked copies.

    langchain-anthropic builds them in the cached properties ``_client`` and
    ``_async_client`` on httpx clients shared between instances; the copies
    keep every SDK option and take httpx clients of their own. A model
    without these properties (a test stand-in) is left as it is.
    """
    import anthropic

    sync_client = getattr(model, "_client", None)
    async_client = getattr(model, "_async_client", None)
    if not isinstance(sync_client, anthropic.Client) or not isinstance(
        async_client, anthropic.AsyncClient
    ):
        return
    model.__dict__["_client"] = sync_client.with_options(
        http_client=anthropic.DefaultHttpxClient(
            timeout=sync_client.timeout,
            event_hooks=capture_hooks(),
            verify=tls_context(),
        )
    )
    model.__dict__["_async_client"] = async_client.with_options(
        http_client=anthropic.DefaultAsyncHttpxClient(
            timeout=async_client.timeout,
            event_hooks=async_capture_hooks(),
            verify=tls_context(),
        )
    )


def capture_either(response: httpx.Response) -> Awaitable[None] | None:
    """Capture hook for an ``httpx.Client`` and an ``httpx.AsyncClient`` alike.

    An async client awaits the returned coroutine; a sync client reads the
    body at once.
    """
    if isinstance(response.stream, httpx.AsyncByteStream):
        return acapture_response(response)
    capture_response(response)
    return None


class EitherTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """An HTTP transport that serves a sync and an async httpx client.

    google-genai builds its sync and async httpx clients from one set of
    ``client_args``; a transport among them also keeps its async calls on
    httpx where aiohttp is installed. It hands each request body to the
    capture scope, which one request hook cannot do for both clients.
    """

    def __init__(self) -> None:
        self._sync = httpx.HTTPTransport(verify=tls_context())
        self._async = httpx.AsyncHTTPTransport(verify=tls_context())

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        capture_request(request)
        return self._sync.handle_request(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        capture_request(request)
        return await self._async.handle_async_request(request)

    def close(self) -> None:
        self._sync.close()

    async def aclose(self) -> None:
        await self._async.aclose()


def _create_google_model(api_key: str, kwargs: dict[str, Any]) -> BaseChatModel:
    """Create a Google Generative AI chat model whose httpx clients capture bodies."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai package is required for Google models. "
            "Install with: pip install langchain-google-genai"
        ) from None

    kwargs["google_api_key"] = api_key
    kwargs["client_args"] = {
        "transport": EitherTransport(),
        "event_hooks": {"response": [capture_either]},
        "verify": tls_context(),
    }
    logger.debug("Creating Google model: %s", kwargs.get("model"))
    return ChatGoogleGenerativeAI(**kwargs)


def _create_openrouter_model(
    api_key: str, kwargs: dict[str, Any], phases: dict[str, Any]
) -> BaseChatModel:
    """Create an OpenRouter chat model instance (OpenAI-compatible API)."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for OpenRouter models. "
            "Install with: pip install langchain-openai"
        ) from None

    kwargs["api_key"] = api_key
    kwargs["base_url"] = OPENROUTER_BASE_URL
    _apply_per_phase_timeout(kwargs, phases)
    _hooked_openai_clients(kwargs)
    kwargs["default_headers"] = {
        "HTTP-Referer": "https://github.com/autoexcerpter",
        "X-Title": "AutoExcerpter",
    }
    logger.debug("Creating OpenRouter model: %s", kwargs.get("model"))
    return ChatOpenAI(**kwargs)


def _create_custom_model(
    api_key: str,
    kwargs: dict[str, Any],
    base_url: str,
    phases: dict[str, Any],
) -> BaseChatModel:
    """Create a model for a custom OpenAI-compatible endpoint."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai is required for custom endpoints. "
            "Install with: pip install langchain-openai"
        ) from None

    kwargs["api_key"] = api_key
    kwargs["base_url"] = base_url
    _apply_per_phase_timeout(kwargs, phases)
    _hooked_openai_clients(kwargs)
    logger.debug("Creating custom endpoint model: %s at %s", kwargs["model"], base_url)
    return ChatOpenAI(**kwargs)


def _seconds(value: float) -> float:
    """Return *value* as an int when it is whole, so clients get ``900``."""
    number = float(value)
    return int(number) if number.is_integer() else number


def resolve_provider(phase: PhaseModel) -> str:
    """Return the phase's provider."""
    return phase.provider


def build_chat_model(
    phase: PhaseModel,
    timeouts: Timeouts,
    *,
    api_key_env: str | None = None,
) -> BaseChatModel:
    """Build the chat model of one phase, without SDK retries.

    *api_key_env* overrides the phase's key variable (the attempt identity
    may name another one).

    Raises:
        OSError: The key variable is unset.
        ImportError: The provider package is missing.
        ValueError: The provider is unknown or a custom endpoint has no URL.
    """
    config = LLMConfig(
        model=phase.name,
        provider=phase.provider,
        api_key_env=api_key_env or phase.api_key_env,
        base_url=phase.base_url,
        timeout=_seconds(timeouts.request),
        connect_timeout=timeouts.connect,
        write_timeout=timeouts.write,
        pool_timeout=timeouts.pool,
    )
    return get_chat_model(config)


__all__ = [
    "OPENROUTER_BASE_URL",
    "LLMConfig",
    "build_chat_model",
    "default_key_env",
    "get_chat_model",
    "resolve_provider",
]
