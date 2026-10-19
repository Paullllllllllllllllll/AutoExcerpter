"""Tests for the hermetic harness in tests/conftest.py."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from tests.conftest import (
    PROJECT_ROOT,
    NetworkBlockedError,
    ToolLaunchRefusedError,
    WriteGuard,
    is_loopback_address,
    is_secret_env_var,
    launches_tool,
)


def _under(path: Path | str, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


class TestPathRedirection:
    def test_home_is_under_tmp_path(self, tmp_path: Path) -> None:
        assert _under(Path.home(), tmp_path)
        assert _under(os.path.expanduser("~"), tmp_path)

    def test_default_state_dir_is_under_tmp_home(self, tmp_path: Path) -> None:
        from autoexcerpter.settings import Settings
        from autoexcerpter.state import get_state_dir

        assert _under(get_state_dir(), tmp_path / "home")
        assert _under(Settings().state_directory(), tmp_path / "home")

    def test_tempdir_is_under_tmp_path(self, tmp_path: Path) -> None:
        assert _under(tempfile.gettempdir(), tmp_path)


class TestSettingsRedirection:
    def test_default_settings_path_is_under_tmp_path(self, tmp_path: Path) -> None:
        from autoexcerpter import settings

        assert _under(settings.DEFAULT_SETTINGS_PATH, tmp_path)

    def test_load_reads_the_tracked_example(self) -> None:
        from autoexcerpter import settings

        loaded = settings.load()
        assert loaded.source == settings.EXAMPLE_SETTINGS_PATH
        assert loaded.source.name == "settings.example.yaml"

    def test_redirected_settings_file_is_read(self, tmp_path: Path) -> None:
        from autoexcerpter import settings

        real = tmp_path / "config" / "settings.yaml"
        real.parent.mkdir()
        real.write_text("openalex:\n  max_requests: 7\n", encoding="utf-8")

        loaded = settings.load()
        assert loaded.source == real
        assert loaded.openalex.max_requests == 7


class TestNetworkGuard:
    def test_non_loopback_connect_raises(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with pytest.raises(NetworkBlockedError):
                sock.connect(("192.0.2.1", 443))
            with pytest.raises(NetworkBlockedError):
                sock.connect_ex(("192.0.2.1", 443))

    def test_async_http_client_is_blocked(self) -> None:
        """The Windows proactor loop bypasses socket.connect; resolution does not."""
        import asyncio

        import httpx

        async def fetch() -> None:
            async with httpx.AsyncClient() as client:
                await client.get("https://api.openai.com/v1/models")

        with pytest.raises(NetworkBlockedError):
            asyncio.run(fetch())

    def test_loopback_socketpair_works(self) -> None:
        left, right = socket.socketpair()
        with left, right:
            left.sendall(b"ping")
            assert right.recv(4) == b"ping"

    def test_loopback_classification(self) -> None:
        assert is_loopback_address(("127.0.0.1", 80))
        assert is_loopback_address(("::1", 80, 0, 0))
        assert is_loopback_address(("localhost", 80))
        assert not is_loopback_address(("api.example.org", 443))
        assert not is_loopback_address(("10.0.0.1", 443))


class TestWriteGuard:
    def test_write_to_repo_root_is_recorded(self, write_guard: WriteGuard) -> None:
        probe = PROJECT_ROOT / f".hermetic-probe-{uuid.uuid4().hex}.txt"
        try:
            probe.write_text("probe", encoding="utf-8")
        finally:
            probe.unlink(missing_ok=True)

        recorded = write_guard.consume(probe)

        expected = os.path.normcase(str(probe))
        assert expected in recorded
        assert write_guard.violations == []

    def test_write_under_tmp_path_is_not_recorded(
        self, tmp_path: Path, write_guard: WriteGuard
    ) -> None:
        target = tmp_path / "sub" / "out.txt"
        target.parent.mkdir()
        target.write_text("ok", encoding="utf-8")
        target.rename(tmp_path / "moved.txt")

        assert write_guard.violations == []


class TestEnvironment:
    def test_no_real_key_values(self) -> None:
        leaked = [
            name
            for name, value in os.environ.items()
            if is_secret_env_var(name) and not value.startswith("test-")
        ]
        assert leaked == []

    @pytest.mark.usefixtures("mock_api_keys")
    def test_stubbed_keys_are_test_values(self) -> None:
        assert os.environ["OPENAI_API_KEY"].startswith("test-")


class TestLaunchGuard:
    def test_entry_point_launch_is_refused(self) -> None:
        script = PROJECT_ROOT / "autoexcerpter" / "cli" / "excerpt.py"
        with pytest.raises(ToolLaunchRefusedError):
            subprocess.run([sys.executable, str(script), "--help"], check=False)

    def test_argv_classification(self) -> None:
        assert launches_tool([sys.executable, "main\\excerpt.py"])
        assert launches_tool([sys.executable, "-m", "autoexcerpter", "--cli"])
        assert launches_tool(["autoexcerpter.exe", "--cli"])
        assert launches_tool("python main/excerpt.py --cli")
        assert launches_tool([sys.executable, "-m", "main.excerpt", "--cli"])
        assert launches_tool([sys.executable, "excerpt.py", "--cli"])
        assert launches_tool(
            [sys.executable, "-c", "import runpy; runpy.run_path('main/excerpt.py')"]
        )
        assert not launches_tool(
            [
                sys.executable,
                "-c",
                "print(1)",
                str(PROJECT_ROOT / "autoexcerpter" / "llm" / "x.py"),
            ]
        )
