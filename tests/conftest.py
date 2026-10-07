"""Pytest fixtures and configuration for AutoExcerpter tests.

The session is hermetic. ``pytest_configure`` strips provider keys from the
environment and points the home directories at a temp dir. The autouse
``_hermetic`` fixture then redirects every per-user path into ``tmp_path``
(the default settings file included, so a real ``config/settings.yaml`` is
never read), blocks non-loopback connections and launches of the tool itself,
and fails any test that writes outside the pytest base temp directory.
AutoExcerpter writes no log files, so there is no logs directory to redirect.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator, Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import fitz  # PyMuPDF
import pytest

from autoexcerpter.common import testing
from autoexcerpter.common.testing import (
    XDG_VARS,
    LaunchEntry,
    WriteGuard,
    home_env,
    is_loopback_address,
)
from autoexcerpter.settings import Settings
from autoexcerpter.spec import RunSpec, resolve_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECRET_PREFIXES = ("OPENALEX_",)
LAUNCH_ENTRY = LaunchEntry(
    programs=("autoexcerpter",),
    scripts=("main/excerpt.py",),
    modules=("autoexcerpter", "main.excerpt"),
)

_SESSION_ENV_BACKUP: dict[str, str | None] = {}

__all__ = [
    "PROJECT_ROOT",
    "NetworkBlockedError",
    "ToolLaunchRefusedError",
    "WriteGuard",
    "home_env",
    "image_config",
    "is_loopback_address",
    "is_secret_env_var",
    "launches_tool",
    "make_settings",
    "make_spec",
]


class NetworkBlockedError(RuntimeError):
    """Raised when a test connects to a non-loopback address."""


class ToolLaunchRefusedError(RuntimeError):
    """Raised when a test launches the AutoExcerpter entry point."""


# ============================================================================
# Session setup
# ============================================================================
def is_secret_env_var(name: str) -> bool:
    """Return True for variables that may hold a provider or OpenAlex key."""
    return testing.is_secret_env_var(name, SECRET_PREFIXES)


def _session_home(config: pytest.Config) -> Path:
    factory: pytest.TempPathFactory | None = getattr(config, "_tmp_path_factory", None)
    if factory is not None:
        return factory.mktemp("session-home")
    return Path(tempfile.mkdtemp(prefix="autoexcerpter-tests-"))


def _set_session_env(name: str, value: str | None) -> None:
    _SESSION_ENV_BACKUP.setdefault(name, os.environ.get(name))
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Drop key variables and redirect the home dirs."""
    for name in [name for name in os.environ if is_secret_env_var(name)]:
        _set_session_env(name, None)
    session_home = _session_home(config)
    env = home_env(session_home)
    xdg_names = {*XDG_VARS, *(n for n in os.environ if n.startswith("XDG_"))}
    env.update(dict.fromkeys(xdg_names, str(session_home)))
    for name, value in env.items():
        _set_session_env(name, value)
    _install_write_guard()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the environment changed in ``pytest_configure``."""
    for name, value in _SESSION_ENV_BACKUP.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _SESSION_ENV_BACKUP.clear()


# ============================================================================
# Write guard
# ============================================================================
_WRITE_GUARD = WriteGuard()
_WRITE_GUARD_INSTALLED = False


def _install_write_guard() -> None:
    global _WRITE_GUARD_INSTALLED
    if not _WRITE_GUARD_INSTALLED:
        sys.addaudithook(_WRITE_GUARD.audit)
        _WRITE_GUARD_INSTALLED = True


# ============================================================================
# Network and launch guards
# ============================================================================
def launches_tool(args: Any, executable: Any = None) -> bool:
    """Return True when a Popen argv would start the AutoExcerpter entry point."""
    return testing.launches_entry(args, executable, LAUNCH_ENTRY)


def _guard_network_and_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    real_popen_init = subprocess.Popen.__init__

    def getaddrinfo(host: Any, port: Any, *a: Any, **kw: Any) -> Any:
        # asyncio's Windows proactor loop connects through ConnectEx, which
        # bypasses socket.connect; every connection still resolves its host here.
        name = host.decode() if isinstance(host, bytes) else host
        if name not in (None, "localhost") and not is_loopback_address((name, 0)):
            raise NetworkBlockedError(f"Network access blocked in tests: {host}")
        return real_getaddrinfo(host, port, *a, **kw)

    def connect(sock: socket.socket, address: Any) -> None:
        if not is_loopback_address(address):
            raise NetworkBlockedError(f"Network access blocked in tests: {address}")
        real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        if not is_loopback_address(address):
            raise NetworkBlockedError(f"Network access blocked in tests: {address}")
        return real_connect_ex(sock, address)

    def popen_init(self: subprocess.Popen[Any], args: Any, *a: Any, **kw: Any) -> None:
        executable = kw.get("executable", a[1] if len(a) > 1 else None)
        if launches_tool(args, executable):
            raise ToolLaunchRefusedError(
                f"Tests must not launch AutoExcerpter as a subprocess: {args!r}"
            )
        real_popen_init(self, args, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(subprocess.Popen, "__init__", popen_init)


def _restore_root_logger(handlers: list[logging.Handler], level: int) -> None:
    """Drop root handlers a test added and reset the root level.

    Pytest's own capture handlers are swapped per test phase and left alone.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler in handlers or type(handler).__module__.startswith("_pytest"):
            continue
        root.removeHandler(handler)
    root.setLevel(level)


# ============================================================================
# Per-test isolation (autouse)
# ============================================================================
@pytest.fixture(autouse=True)
def _hermetic(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[WriteGuard]:
    """Redirect per-user state into ``tmp_path`` and guard writes and network."""
    import autoexcerpter.settings as settings_module

    home = tmp_path / "home"
    temp = tmp_path / "tmp"
    for directory in (home, temp):
        directory.mkdir()
    for name, value in home_env(home).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    monkeypatch.setattr(
        settings_module,
        "DEFAULT_SETTINGS_PATH",
        tmp_path / "config" / "settings.yaml",
    )

    _guard_network_and_launches(monkeypatch)

    root_logger = logging.getLogger()
    root_handlers = list(root_logger.handlers)
    root_level = root_logger.level

    _WRITE_GUARD.start([tmp_path_factory.getbasetemp()])
    try:
        yield _WRITE_GUARD
    finally:
        leaked = _WRITE_GUARD.stop()
        _restore_root_logger(root_handlers, root_level)
    if leaked:
        listing = "\n  ".join(leaked)
        pytest.fail(
            f"Test wrote outside the pytest temp directory:\n  {listing}",
            pytrace=False,
        )


@pytest.fixture
def write_guard(_hermetic: WriteGuard) -> WriteGuard:
    """Return the write guard so a test can consume expected violations."""
    return _hermetic


# ============================================================================
# Path Fixtures
# ============================================================================
@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path]:
    """Return a directory under ``tmp_path`` for test outputs."""
    out_dir = tmp_path / "temp_dir"
    out_dir.mkdir()
    yield out_dir


# ============================================================================
# PDF Fixtures
# ============================================================================
@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes a tiny PDF with the given page count."""

    def _make(name: str = "doc.pdf", num_pages: int = 1) -> Path:
        pdf_path = tmp_path / name
        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page(width=200, height=300)
            page.insert_text((50, 100), f"Page {i + 1}")
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    return _make


# ============================================================================
# Settings and spec builders
# ============================================================================
def make_spec(input_path: Path | str = "doc.pdf", **flags: Any) -> RunSpec:
    """Return a run spec from option values keyed by option name."""
    return resolve_spec({"input": Path(input_path), **flags}).spec


def make_settings(**changes: Any) -> Settings:
    """Return the code-default settings with *changes* applied."""
    return dataclasses.replace(Settings(), **changes)


@pytest.fixture
def mock_image_processing_config() -> dict[str, Any]:
    """Return small image sections at 300 DPI with a high detail profile."""
    return {
        "api_image_processing": {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "llm_detail": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
        },
        "google_image_processing": {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "media_resolution": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 768],
        },
        "anthropic_image_processing": {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "resize_profile": "auto",
            "low_max_side_px": 512,
            "high_max_side_px": 1568,
        },
    }


@contextlib.contextmanager
def image_config(
    sections: Mapping[str, Any], image_size: str | None = None
) -> Iterator[None]:
    """Use *sections* as the image settings while the context is open.

    *image_size* is the OpenAI image detail option of the transcription
    model; None sends no detail.
    """
    import autoexcerpter.imaging.settings as image_settings

    def detail_option(provider: str | None, images: Any = None) -> str | None:
        return image_size

    with (
        patch.object(image_settings, "IMAGE_PROCESSING", sections),
        patch.object(image_settings, "request_detail_option", detail_option),
    ):
        yield


@pytest.fixture
def mock_image_processing(
    mock_image_processing_config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Install the small image sections (no image detail) for one test."""
    with image_config(mock_image_processing_config):
        yield mock_image_processing_config


# ============================================================================
# Citation Fixtures
# ============================================================================
@pytest.fixture
def sample_citations() -> list[str]:
    """Return sample citation strings for testing."""
    return [
        "Smith, J. (2020). Introduction to Testing. New York: Test Press.",
        "Johnson, A., & Williams, B. (2019). Advanced Python Testing. "
        "London: Code Publishers.",
        "Brown, C. (2021). Modern Software Development. Cambridge University Press.",
        "Smith, J. (2020). Introduction to Testing. New York: Test Press.",  # Duplicate
    ]


@pytest.fixture
def mock_openalex_response() -> dict[str, Any]:
    """Return a mock OpenAlex API response."""
    return {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/test.2020.001",
        "title": "Introduction to Testing",
        "publication_year": 2020,
        "authorships": [
            {"author": {"display_name": "John Smith"}},
        ],
        "primary_location": {
            "source": {"display_name": "Test Press"},
        },
    }


# ============================================================================
# Environment Fixtures
# ============================================================================
@pytest.fixture
def mock_api_keys() -> Generator[None]:
    """Stub the provider API key variables with test values."""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GOOGLE_API_KEY": "test-google-key",
            "OPENROUTER_API_KEY": "test-openrouter-key",
        },
    ):
        yield
