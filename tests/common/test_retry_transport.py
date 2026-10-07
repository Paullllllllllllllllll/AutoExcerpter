"""Classification of raw transport errors and message-only server errors."""

from __future__ import annotations

import ssl

import httpx
import pytest

from autoexcerpter.common.retry import (
    ErrorKind,
    classify_error,
    is_connection_error,
    is_server_error_message,
)

TRANSPORT_ERRORS: list[type[Exception]] = [
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.CloseError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
    ssl.SSLError,
]

TIMEOUT_ERRORS: list[type[Exception]] = [
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
]

CLIENT_TRANSPORT_ERRORS: list[type[Exception]] = [
    httpx.UnsupportedProtocol,
    httpx.LocalProtocolError,
]


def _wrapped(cause: BaseException) -> Exception:
    """Return a provider-style wrapper raised from ``cause``."""
    try:
        try:
            raise cause
        except BaseException as inner:
            raise RuntimeError("request failed") from inner
    except RuntimeError as wrapper:
        return wrapper


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("error_type", TRANSPORT_ERRORS)
def test_transport_error_is_connection(error_type: type[Exception]) -> None:
    exc = error_type("peer went away")
    assert is_connection_error(exc)
    assert classify_error(exc) is ErrorKind.CONNECTION


@pytest.mark.parametrize("error_type", TRANSPORT_ERRORS)
def test_wrapped_transport_error_is_connection(error_type: type[Exception]) -> None:
    wrapper = _wrapped(error_type("peer went away"))
    assert wrapper.__cause__ is not None
    assert classify_error(wrapper) is ErrorKind.CONNECTION


@pytest.mark.parametrize("error_type", TIMEOUT_ERRORS)
def test_timeout_is_decided_first(error_type: type[Exception]) -> None:
    assert classify_error(error_type("slow")) is ErrorKind.TIMEOUT
    assert classify_error(_wrapped(error_type("slow"))) is ErrorKind.TIMEOUT


@pytest.mark.parametrize("error_type", CLIENT_TRANSPORT_ERRORS)
def test_local_transport_error_is_client_error(error_type: type[Exception]) -> None:
    exc = error_type("Request URL has an unsupported protocol 'ftp://'.")
    assert not is_connection_error(exc)
    assert classify_error(exc) is ErrorKind.CLIENT_ERROR
    assert not classify_error(exc).retryable


def test_status_400_beats_connection_message() -> None:
    exc = _StatusError("Connection error.", 400)
    assert classify_error(exc) is ErrorKind.CLIENT_ERROR


@pytest.mark.parametrize("status", [408, 409])
def test_408_and_409_stay_retryable(status: int) -> None:
    kind = classify_error(_StatusError("Connection error.", status))
    assert kind is ErrorKind.OTHER
    assert kind.retryable


def test_openai_message_only_500_is_server_error() -> None:
    message = "The server had an error while processing your request."
    assert is_server_error_message(message)
    assert classify_error(Exception(message)) is ErrorKind.SERVER_ERROR
