"""The single entry point through which characterization tests run the tool."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from autoexcerpter.output.paths import ItemPaths
from tests.characterization.fakes import (
    CUSTOM_KEY_ENV,
    DUMMY_KEYS,
    SUMMARY,
    TRANSCRIPTION,
    FakeLLM,
)

ResponseFormat = Literal["schema", "json", "text"]

CUSTOM_MODEL = "org/fake-model"
CUSTOM_BASE_URL = "https://fake-endpoint.invalid/v1/"
CUSTOM_ENDPOINT = "fake"

_ASYNC_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


@dataclass
class RunResult:
    """Outcome of one ``run_tool`` call."""

    exit_code: int
    summary: dict[str, Any] | None
    stdout: str
    stderr: str
    root: Path
    output_dir: Path
    scan_dirs: tuple[Path, ...]
    created: list[Path]
    llm: FakeLLM

    def calls(self, role: str | None = None) -> list[dict[str, Any]]:
        """Return the recorded request digests, optionally for one role."""
        if role is None:
            return self.llm.sorted_calls()
        return self.llm.calls_for(role)

    def paths(self, name: str, output_dir: Path | None = None) -> ItemPaths:
        """Return the output and working paths of item *name*.

        *output_dir* defaults to the run's output folder.
        """
        return item_paths(name, self.output_dir if output_dir is None else output_dir)


def item_paths(
    name: str, output_dir: Path, transcription_format: str = "md"
) -> ItemPaths:
    """Return the output and working-log paths of item *name* in *output_dir*.

    With ``Path()`` as *output_dir* the paths are relative names inside the
    output folder.
    """
    return ItemPaths.for_item(name, output_dir, transcription_format)


def _no_sleep(_seconds: float) -> None:
    """Stand-in for ``time.sleep`` that returns at once."""


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Stand-in for ``asyncio.sleep`` that only yields to the event loop.

    The store's lease heartbeat keeps its real interval, so it does not spin.
    """
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _ASYNC_SLEEP(delay, result)
    await _ASYNC_SLEEP(0)
    return result


def _phase_args(
    phase: str,
    provider: str,
    model: str,
    response_format: ResponseFormat | None,
    reroute: bool = True,
) -> list[str]:
    """Return the per-phase provider, model, endpoint and format flags.

    A registry model keeps its provider; with *reroute* the json and text
    formats, and any custom provider, go to the named fake endpoint with
    that format. Without *reroute* a registry model keeps its provider and
    gets the format flag itself.
    """
    if response_format is None and provider == "custom":
        raise ValueError("a custom endpoint needs an explicit response format")
    if not reroute and provider != "custom":
        args = [f"--{phase}-provider", provider, f"--{phase}-model", model]
        if response_format is not None:
            args += [f"--{phase}-response-format", response_format]
        return args
    if response_format is None or (
        response_format == "schema" and provider != "custom"
    ):
        return [f"--{phase}-provider", provider, f"--{phase}-model", model]
    name = model if provider == "custom" else CUSTOM_MODEL
    return [
        f"--{phase}-provider",
        "custom",
        f"--{phase}-endpoint",
        CUSTOM_ENDPOINT,
        f"--{phase}-model",
        name,
        f"--{phase}-response-format",
        response_format,
    ]


def write_settings(folder: Path) -> Path:
    """Write the adapter's settings file into *folder* and return its path.

    It names the fake custom endpoint, sets rate limits that never wait,
    leaves the OpenAlex contact empty and puts the state dir in
    ``folder.parent / "state"``.
    """
    data = {
        "endpoints": {
            CUSTOM_ENDPOINT: {
                "base_url": CUSTOM_BASE_URL,
                "api_key_env": CUSTOM_KEY_ENV,
                "supports_vision": True,
                "supports_schema": True,
            }
        },
        "rate_limits": [[10000, 1]],
        "state_dir": str(folder.parent / "state"),
        "openalex": {"email": "", "api_key_env": ""},
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8", newline="\n")
    return path


def _files_under(dirs: Sequence[Path]) -> set[Path]:
    found: set[Path] = set()
    for directory in dirs:
        if directory.is_dir():
            found.update(path for path in directory.rglob("*") if path.is_file())
    return found


def _last_json_line(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def run_tool(
    input_path: Path,
    *,
    llm: FakeLLM,
    root: Path,
    output_dir: Path | None = None,
    beside_input: bool = False,
    selection: str | None = "all",
    summarize: bool = True,
    docx: bool = True,
    markdown: bool = True,
    context: str | None = None,
    openalex: bool = False,
    transcription_provider: str = "openai",
    transcription_model: str = "gpt-5.6-terra",
    summary_provider: str = "openai",
    summary_model: str = "gpt-5.6-terra",
    transcription_format: ResponseFormat | None = "schema",
    summary_format: ResponseFormat | None = "schema",
    force: bool = False,
    retranscribe: bool = False,
    dry_run: bool = False,
    jpeg_quality: int | None = None,
    reroute_formats: bool = True,
    transcription_file_format: str | None = None,
    batch: bool = False,
) -> RunResult:
    """Run AutoExcerpter once, in process, as a fresh CLI invocation would.

    This is the only function that knows how the tool is invoked. It builds
    the argv of ``autoexcerpter run --input <input> ... --json`` with a
    temporary settings file (see :func:`write_settings`), sets the dummy
    provider keys in ``DUMMY_KEYS``, and calls ``autoexcerpter.__main__.main``.

    *selection* is ``"all"``, a ``--select`` expression, or None for neither.
    *beside_input* writes beside the input; otherwise the outputs go to
    *output_dir* (``root / "out"`` when None). *transcription_format* and
    *summary_format* pick the response format per phase: ``"schema"`` keeps
    the given registry model (or the fake endpoint in schema mode when the
    provider is ``custom``); ``"json"`` and ``"text"`` route the phase to the
    fake endpoint with that format. None requests no format and keeps the
    registry model, whose capabilities then decide between enforced and
    prompted JSON (a custom endpoint rejects None). With *reroute_formats*
    False a registry model keeps its provider for every format and the
    format is passed as ``--<phase>-response-format``. *jpeg_quality* is
    passed as ``--jpeg-quality`` and *transcription_file_format* (``md`` or
    ``txt``) as ``--transcription-format``; *batch* adds ``--batch``.
    During the run ``time.sleep`` returns at once and ``asyncio.sleep`` only
    yields; neither shapes the outputs.
    The fake LLM must already be installed (see ``conftest.fake_llm``).
    """
    output = output_dir if output_dir is not None else root / "out"
    settings_path = write_settings(root / "config")
    formats = [
        name
        for name, wanted in (("summary-md", markdown), ("summary-docx", docx))
        if wanted
    ]

    argv = [
        "run",
        "--input",
        str(input_path),
        *(["--beside-input"] if beside_input else ["--output", str(output)]),
        "--json",
        "--settings",
        str(settings_path),
        "--summarize" if summarize else "--no-summarize",
        *(["--formats", ",".join(formats)] if formats else []),
        "--openalex" if openalex else "--no-openalex",
        *_phase_args(
            "transcription",
            transcription_provider,
            transcription_model,
            transcription_format,
            reroute_formats,
        ),
        *_phase_args(
            "summary", summary_provider, summary_model, summary_format, reroute_formats
        ),
    ]
    if selection == "all":
        argv.append("--all")
    elif selection is not None:
        argv.extend(["--select", selection])
    if context is not None:
        argv.extend(["--context", context])
    if force:
        argv.append("--force")
    if retranscribe:
        argv.append("--retranscribe")
    if dry_run:
        argv.append("--dry-run")
    if jpeg_quality is not None:
        argv.extend(["--jpeg-quality", str(jpeg_quality)])
    if transcription_file_format is not None:
        argv.extend(["--transcription-format", transcription_file_format])
    if batch:
        argv.append("--batch")

    llm.text_roles = {
        role
        for role, fmt in (
            (TRANSCRIPTION, transcription_format),
            (SUMMARY, summary_format),
        )
        if fmt == "text"
    }
    scan_dirs = tuple(dict.fromkeys([input_path.parent, output]))
    return _invoke(argv, llm=llm, root=root, output=output, scan_dirs=scan_dirs)


def run_batch_command(
    command: Literal["check", "cancel"],
    *,
    llm: FakeLLM,
    root: Path,
    directory: Path | None = None,
    extra: Sequence[str] = (),
) -> RunResult:
    """Run ``autoexcerpter batch <command>`` once, in process.

    *directory* (``root / "out"`` when None) is passed as ``--input``, with
    ``--json``, the adapter's settings file and *extra*; the environment is
    that of :func:`run_tool`. The fake batch service must already be
    installed.
    """
    folder = directory if directory is not None else root / "out"
    settings_path = write_settings(root / "config")
    argv = [
        "batch",
        command,
        "--input",
        str(folder),
        "--json",
        "--settings",
        str(settings_path),
        *extra,
    ]
    return _invoke(argv, llm=llm, root=root, output=folder, scan_dirs=(folder,))


def _invoke(
    argv: list[str],
    *,
    llm: FakeLLM,
    root: Path,
    output: Path,
    scan_dirs: tuple[Path, ...],
) -> RunResult:
    """Call the entry point with *argv* under the adapter's environment."""
    from autoexcerpter import __main__ as entry

    before = _files_under(scan_dirs)
    stdout, stderr = io.StringIO(), io.StringIO()
    exit_code: int
    with pytest.MonkeyPatch.context() as mp:
        for name, value in DUMMY_KEYS.items():
            mp.setenv(name, value)
        mp.setattr(time, "sleep", _no_sleep)
        mp.setattr(asyncio, "sleep", _yield_only)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code: Any = entry.main(argv)
            except SystemExit as exc:
                code = exc.code
        exit_code = code if isinstance(code, int) else (0 if code is None else 1)

    created = sorted(_files_under(scan_dirs) - before)
    return RunResult(
        exit_code=exit_code,
        summary=_last_json_line(stdout.getvalue()),
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        root=root,
        output_dir=output,
        scan_dirs=scan_dirs,
        created=created,
        llm=llm,
    )


__all__ = [
    "RunResult",
    "ResponseFormat",
    "item_paths",
    "run_batch_command",
    "run_tool",
    "write_settings",
]
