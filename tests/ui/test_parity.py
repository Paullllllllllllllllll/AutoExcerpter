"""Parity between the CLI and the wizard, computed from the option table.

Every option must be an argument of ``autoexcerpter run`` and must be set by
some scripted wizard session, apart from the exemptions below. Each session's
printed command must resolve to the spec its answers resolve to, with the
session's settings and with an empty defaults block. The random round trip
``parse(to_argv(spec)) == spec`` lives in ``tests/core/test_spec.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pytest

import autoexcerpter.run as core
from autoexcerpter import ext
from autoexcerpter.cli.parser import build_parser
from autoexcerpter.common.options import Option, OptionTable
from autoexcerpter.common.testing import KEEP, ScriptedPrompter
from autoexcerpter.common.wizard import (
    BACK,
    Decision,
    Outcome,
    State,
    Step,
    StepContext,
    command_display,
    command_line,
)
from autoexcerpter.pipeline.resume import ProcessingState, ResumeResult
from autoexcerpter.settings import Settings, load, resolve
from autoexcerpter.spec import OPTIONS, PHASES, RunSpec
from autoexcerpter.ui.review import FORCE_ACTION, PROGRAM
from autoexcerpter.ui.session import Session, resolution
from autoexcerpter.ui.wizard import build_steps, build_wizard

EXEMPT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "json": "agent-contract output (one JSON line on stdout); no use in the UI",
        "settings": "the ui subcommand's own flag, checked against the ui parser",
    }
)
WIDTH: Final = 100
_MISSING: Final = object()


def unprefixed_model_options(names: Collection[str]) -> set[str]:
    """Return the options that every phase repeats with its prefix."""
    return {
        name for name in names if all(f"{phase}_{name}" in names for phase in PHASES)
    }


def unreached(names: Collection[str], reached: Collection[str]) -> set[str]:
    """Return the options of *names* that no wizard answer reached.

    Exempt options are passed over; an unprefixed model option counts as
    reached when every per-phase option is reached.
    """
    unprefixed = unprefixed_model_options(names)
    missing: set[str] = set()
    for name in names:
        if name in EXEMPT:
            continue
        if name in unprefixed:
            if not all(f"{phase}_{name}" in reached for phase in PHASES):
                missing.add(name)
        elif name not in reached:
            missing.add(name)
    return missing


@dataclasses.dataclass(frozen=True)
class Script:
    """A wizard session: the settings it starts with and the answers."""

    session: Session
    answers: Sequence[Any]


@dataclasses.dataclass
class Played:
    """A finished session: outcome, transcript and what each step set."""

    script: Script
    outcome: Outcome
    prompter: ScriptedPrompter
    changed: dict[str, set[str]]
    reached: dict[str, set[str]]

    @property
    def session(self) -> Session:
        return self.script.session

    @property
    def command(self) -> tuple[str, ...]:
        command = self.outcome.command
        assert command is not None
        return command

    def spec(self) -> RunSpec:
        state = State(OPTIONS, self.session.settings.defaults, self.outcome.answers)
        return resolution(state, self.session).spec


class Workspace:
    """Inputs, settings files and environment of one test."""

    def __init__(
        self,
        tmp_path: Path,
        make_pdf: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.tmp = tmp_path
        self.make_pdf = make_pdf
        self.monkeypatch = monkeypatch

    def pdf(self, name: str = "book.pdf", pages: int = 2) -> Path:
        (self.tmp / name).parent.mkdir(parents=True, exist_ok=True)
        return self.make_pdf(name, num_pages=pages)

    def folder(self, name: str, *items: str) -> Path:
        for item in items:
            self.pdf(f"{name}/{item}.pdf")
        return self.tmp / name

    def text(self, name: str, content: str) -> Path:
        path = self.tmp / name
        path.write_text(content, encoding="utf-8")
        return path

    def session(self, yaml: str | None = None) -> Session:
        """Return a session over a settings file written from *yaml*."""
        if yaml is None:
            return Session(Settings())
        path = self.text("settings.yaml", yaml)
        return Session(load(path, blocks=ext.settings_blocks()), path)

    def keys(self, *names: str) -> None:
        for name in names:
            self.monkeypatch.setenv(name, "test-key")


def _quoted(path: Path) -> str:
    return json.dumps(str(path))


def _base(input_path: Path, *rest: Any) -> list[Any]:
    """Answers for one PDF with summary and recommended values, then *rest*."""
    return [
        str(input_path),  # input
        "beside",  # output
        True,  # summarize
        "md",  # transcription format
        KEEP,  # output formats
        "auto",  # context
        "none",  # fallback context
        "openai",  # transcription provider
        "gpt-5.6-luna",  # transcription model
        KEEP,  # reasoning effort
        KEEP,  # response format
        "recommended",  # model parameters
        "same",  # summary model
        "recommended",  # page images
        KEEP,  # run options
        KEEP,  # concurrency
        *rest,
    ]


def settings_defaults(ws: Workspace) -> Script:
    """Model, service tier, fallback context and output from settings."""
    ws.keys("OPENAI_API_KEY")
    pdf = ws.pdf()
    session = ws.session(
        "defaults:\n"
        "  model: gpt-5.6-terra\n"
        "  service_tier: priority\n"
        "  fallback_context: regional cookery\n"
        f"  output: {_quoted(ws.tmp / 'excerpts')}\n"
    )
    answers = [
        str(pdf),
        "folder",
        KEEP,  # output folder from settings
        True,
        "md",
        KEEP,
        "auto",
        KEEP,  # fallback mode: text, from settings
        KEEP,  # fallback topics from settings
        KEEP,  # provider
        KEEP,  # model from settings
        KEEP,
        KEEP,
        "recommended",
        "same",
        "recommended",
        KEEP,
        KEEP,
        "run",
    ]
    return Script(session, answers)


def back_to_code_defaults(ws: Workspace) -> Script:
    """Values chosen back to their code default against settings defaults."""
    pdf = ws.pdf()
    session = ws.session(
        "defaults:\n"
        "  reasoning_effort: medium\n"
        "  concurrency: 4\n"
        "  transcription_format: txt\n"
        "  openalex: false\n"
        "  context: none\n"
        "  fallback_context: regional cookery\n"
        f"  output: {_quoted(ws.tmp / 'excerpts')}\n"
    )
    answers = _base(pdf, "copy")
    answers[9] = "high"
    answers[14:16] = [["openalex"], "16"]
    return Script(session, answers)


def named_endpoints(ws: Workspace) -> Script:
    """A named endpoint per phase and a context file."""
    pdf = ws.pdf()
    topics = ws.text("topics.txt", "trade; prices\n")
    session = ws.session(
        "endpoints:\n"
        "  local:\n"
        "    base_url: http://127.0.0.1:8000/v1\n"
        "    api_key_env: LOCAL_KEY\n"
        "    supports_schema: true\n"
        "  lab:\n"
        "    base_url: http://127.0.0.1:9000/v1\n"
        "    api_key_env: LAB_KEY\n"
    )
    answers = [
        str(pdf),
        "beside",
        True,
        "md",
        KEEP,
        "file",
        str(topics),
        "endpoint:local",
        "qwen-vl",
        "schema",
        "recommended",
        "own",
        "endpoint:lab",
        "llava-next",
        "text",
        "recommended",
        "recommended",
        KEEP,
        KEEP,
        "copy",
    ]
    return Script(session, answers)


def settings_endpoint_overridden(ws: Workspace) -> Script:
    """A settings endpoint and summary model replaced by a provider model."""
    pdf = ws.pdf()
    session = ws.session(
        "endpoints:\n"
        "  local:\n"
        "    base_url: http://127.0.0.1:8000/v1\n"
        "    api_key_env: LOCAL_KEY\n"
        "defaults:\n"
        "  endpoint: local\n"
        "  model: qwen-vl\n"
        "  summary_model: claude-sonnet-5\n"
        "  summary_reasoning_effort: low\n"
    )
    return Script(session, _base(pdf, "copy"))


def own_summary_model(ws: Workspace) -> Script:
    """A separate summary model, both phases with custom parameters."""
    pdf = ws.pdf()
    answers = _base(pdf, "copy")
    answers[7:12] = [
        "openai",
        "<other>",
        "gpt-4o",
        "json",
        "customize",
        "8000",
        "0.4",
        "0.9",
        "priority",
    ]
    answers[16:17] = [
        "own",
        "openai",
        "gpt-5.6-terra",
        "medium",
        "text",
        "customize",
        "high",
        "4000",
        "auto",
    ]
    return Script(ws.session(), answers)


def custom_images_and_parameters(ws: Workspace) -> Script:
    """Customized page images and model parameters."""
    pdf = ws.pdf()
    answers = _base(pdf, "copy")
    answers[7:12] = [
        "openai",
        "gpt-5.6-terra",
        "low",
        "text",
        "customize",
        "high",
        "12000",
        "priority",
    ]
    answers[15:17] = [
        "own",
        "openai",
        "<other>",
        "gpt-4o",
        "json",
        "customize",
        "3000",
        "0.2",
        "0.5",
        "default",
        "customize",
        "200",
        "high",
        "jpeg",
        "80",
        False,
    ]
    return Script(ws.session(), answers)


def items_by_number(ws: Workspace) -> Script:
    """Several items selected by number and range, in a folder with a space."""
    folder = ws.folder("my scans", "alpha", "beta", "gamma", "delta")
    answers = _base(folder, "copy")
    answers[1:1] = ["numbers", "1,3-4"]
    return Script(ws.session(), answers)


def items_by_name(ws: Workspace) -> Script:
    """A number selection replaced by a name search from the review."""
    folder = ws.folder("scans", "alpha", "beta", "gamma")
    answers = _base(folder, "edit", "input", KEEP, "name", "amm", "copy")
    answers[1:1] = ["numbers", "2"]
    return Script(ws.session(), answers)


def all_items_without_summary(ws: Workspace) -> Script:
    """Every item into one folder, transcription only."""
    folder = ws.folder("scans", "alpha", "beta")
    output = ws.tmp / "out"
    answers = [
        str(folder),
        "all",
        "folder",
        str(output),
        False,
        "txt",
        [],
        "openai",
        "gpt-5.6-luna",
        KEEP,
        KEEP,
        "recommended",
        "recommended",
        ["retranscribe"],
        "2",
        "copy",
    ]
    return Script(ws.session(), answers)


def back_navigation(ws: Workspace) -> Script:
    """Back within steps, across steps and from the review."""
    pdf = ws.pdf()
    topics = ws.text("fallback.txt", "pottery\n")
    answers = [
        str(pdf),
        "folder",
        str(ws.tmp / "out"),
        BACK,  # summary: back to the output step
        "beside",
        True,
        "md",
        BACK,  # formats: back to the transcription format
        "txt",
        KEEP,
        "auto",
        "text",
        BACK,  # topics: back to the fallback mode
        "file",
        str(topics),
        "anthropic",
        BACK,  # model name: back to the provider
        "openai",
        "gpt-5.6-luna",
        KEEP,
        KEEP,
        "recommended",
        "same",
        "recommended",
        KEEP,
        KEEP,
        BACK,  # review: back to the run options
        KEEP,
        "8",
        "quit",
    ]
    return Script(ws.session(), answers)


def force_review_action(ws: Workspace) -> Script:
    """The review's force action for complete items, then Run."""
    ws.keys("OPENAI_API_KEY")
    real = core.plan

    def plan(spec: RunSpec, settings: Settings, **kwargs: Any) -> core.RunPlan:
        result = real(spec, settings, **kwargs)
        if spec.run.force:
            return result
        complete = tuple(
            dataclasses.replace(
                item, resume=ResumeResult(item.name, ProcessingState.COMPLETE)
            )
            for item in result.to_process
        )
        return dataclasses.replace(result, to_process=(), skipped=complete)

    ws.monkeypatch.setattr(core, "plan", plan)
    pdf = ws.pdf()
    answers = _base(pdf, FORCE_ACTION, "run")
    answers[4:7] = [["summary-md", "sqlite"], "none"]
    return Script(ws.session(), answers)


def dry_run(ws: Workspace) -> Script:
    """A dry run with every run flag and quoted context text."""
    pdf = ws.pdf("dir with space/book.pdf")
    answers = _base(pdf, "run")
    answers[5:7] = ["text", 'early modern "spice" trade, 1600\\1800']
    answers[14:16] = [["keep_working_files", "force", "retranscribe", "dry_run"], "3"]
    return Script(ws.session(), answers)


SESSIONS: Final[Mapping[str, Callable[[Workspace], Script]]] = MappingProxyType(
    {
        builder.__name__: builder
        for builder in (
            settings_defaults,
            back_to_code_defaults,
            named_endpoints,
            settings_endpoint_overridden,
            own_summary_model,
            custom_images_and_parameters,
            items_by_number,
            items_by_name,
            all_items_without_summary,
            back_navigation,
            force_review_action,
            dry_run,
        )
    }
)


def _recording(
    steps: Iterable[Step],
    changed: dict[str, set[str]],
    reached: dict[str, set[str]],
) -> list[Step]:
    """Wrap *steps* to record the answers each one changes and reaches."""

    def wrap(step: Step) -> Step:
        def ask(ctx: StepContext) -> Any:
            before = ctx.state.answers
            result = step.ask(ctx)
            after = ctx.state.answers
            names = {
                name
                for name in before.keys() | after.keys()
                if before.get(name, _MISSING) != after.get(name, _MISSING)
            }
            changed.setdefault(step.name, set()).update(names)
            reached.setdefault(step.name, set()).update(
                name
                for name in names
                if name in after and after[name] != OPTIONS[name].default
            )
            return result

        return dataclasses.replace(step, ask=ask)

    return [wrap(step) for step in steps]


def play(script: Script) -> Played:
    """Run the wizard through *script* and return what it produced."""
    prompter = ScriptedPrompter(script.answers)
    changed: dict[str, set[str]] = {}
    reached: dict[str, set[str]] = {}
    steps = _recording(build_steps(script.session), changed, reached)
    wizard = build_wizard(script.session, prompter, steps=steps, width=WIDTH)
    with prompter:
        outcome = wizard.run()
    return Played(script, outcome, prompter, changed, reached)


@pytest.fixture
def workspace(
    tmp_path: Path, make_pdf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> Workspace:
    return Workspace(tmp_path, make_pdf, monkeypatch)


def split_windows(line: str) -> list[str]:
    """Split *line* by the rules ``subprocess.list2cmdline`` quotes for."""
    words: list[str] = []
    word: list[str] = []
    in_word = quoted = False
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\":
            end = index
            while end < len(line) and line[end] == "\\":
                end += 1
            count = end - index
            if end < len(line) and line[end] == '"':
                word.append("\\" * (count // 2))
                if count % 2:
                    word.append('"')
                    end += 1
            else:
                word.append("\\" * count)
            in_word = True
            index = end
            continue
        if char == '"':
            quoted = not quoted
            in_word = True
        elif char in " \t" and not quoted:
            if in_word:
                words.append("".join(word))
                word, in_word = [], False
        else:
            word.append(char)
            in_word = True
        index += 1
    if in_word:
        words.append("".join(word))
    return words


def split_line(line: str, *, windows: bool) -> list[str]:
    """Split a command line with the quoting rules of the platform."""
    return split_windows(line) if windows else shlex.split(line)


def _windows_api_split(line: str) -> list[str]:
    """Split *line* with ``CommandLineToArgvW``; Windows only."""
    if sys.platform != "win32":
        raise RuntimeError("CommandLineToArgvW exists only on Windows")
    import ctypes
    from ctypes import wintypes

    to_argv = ctypes.windll.shell32.CommandLineToArgvW
    to_argv.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    to_argv.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    argv = to_argv(line, ctypes.byref(count))
    try:
        return [str(argv[index]) for index in range(count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv)


def _subparser(name: str) -> argparse.ArgumentParser:
    parser = build_parser()
    (commands,) = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    sub: argparse.ArgumentParser = commands.choices[name]
    return sub


def _arguments(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction)
    }


def _extension_dests() -> set[str]:
    parser = argparse.ArgumentParser(add_help=False)
    ext.register_args(parser)
    return {action.dest for action in parser._actions}


def _command_spec(words: Sequence[str], settings: Settings) -> RunSpec:
    """Parse *words* with the real CLI parser and resolve them with *settings*."""
    assert tuple(words[: len(PROGRAM)]) == PROGRAM
    args = build_parser().parse_args(list(words[1:]))
    flags = OPTIONS.values(args)
    flags.pop("settings", None)
    return resolve(settings, flags).spec


def test_every_option_is_an_argument_of_the_run_parser() -> None:
    arguments = _arguments(_subparser("run"))
    assert set(arguments) == {*OPTIONS.names, *_extension_dests()}
    for option in OPTIONS:
        strings = set(arguments[option.name].option_strings)
        expected = {option.flag}
        if option.kind.value == "switch":
            expected.add(option.negative_flag)
        assert strings == expected, option.name


def test_the_ui_parser_accepts_the_settings_option(tmp_path: Path) -> None:
    arguments = _arguments(_subparser("ui"))
    assert set(arguments) == {"settings", *_extension_dests()}
    path = tmp_path / "other.yaml"
    args = build_parser().parse_args(["ui", "--settings", str(path)])
    assert args.command == "ui"
    assert args.settings == path


def test_the_exemptions_are_options_with_reasons() -> None:
    assert set(EXEMPT) == {"json", "settings"}
    assert set(EXEMPT) <= set(OPTIONS.names)
    assert all(reason.strip() for reason in EXEMPT.values())
    declared = {name for step in build_steps(Session(Settings())) for name in step.sets}
    assert not declared & set(EXEMPT)


def test_an_option_missing_from_every_step_is_reported() -> None:
    extra = Option("extra", "--extra", "an option no step sets")
    table = OptionTable([*OPTIONS, extra])
    reached = set(OPTIONS.names) - unprefixed_model_options(OPTIONS.names)
    assert unreached(OPTIONS.names, reached) == set()
    assert unreached(table.names, reached) == {"extra"}
    assert unreached(OPTIONS.names, reached - {"summary_top_p"}) == {
        "summary_top_p",
        "top_p",
    }


def test_the_sessions_reach_every_option_from_their_steps(
    workspace: Workspace,
) -> None:
    changed: dict[str, set[str]] = {}
    reached: dict[str, set[str]] = {}
    for build in SESSIONS.values():
        played = play(build(workspace))
        for name, names in played.changed.items():
            changed.setdefault(name, set()).update(names)
        for name, names in played.reached.items():
            reached.setdefault(name, set()).update(names)
    every = {name for names in reached.values() for name in names}
    assert unreached(OPTIONS.names, every) == set()
    for step in build_steps(workspace.session()):
        assert changed.get(step.name, set()) <= set(step.sets), step.name
        assert set(step.sets) <= reached.get(step.name, set()), step.name


def _check_lines(played: Played) -> None:
    """The printed and displayed command lines split back into the words."""
    words = list(played.command)
    for windows in (False, True):
        line = command_line(words, windows=windows)
        assert split_line(line, windows=windows) == words
        display = command_display(words, width=WIDTH, windows=windows)
        joined = " ".join(part.strip() for part in display)
        assert split_line(joined, windows=windows) == words
        if windows and sys.platform == "win32":
            assert _windows_api_split(line) == words
    shown = command_display(words, width=WIDTH)
    lines = played.prompter.lines
    start = len(lines) - lines[::-1].index("Equivalent command:")
    assert lines[start : start + len(shown)] == shown


@pytest.mark.parametrize("name", list(SESSIONS))
def test_the_printed_command_yields_the_wizard_spec(
    workspace: Workspace, name: str
) -> None:
    played = play(SESSIONS[name](workspace))
    spec = played.spec()
    settings = played.session.settings
    words = played.command
    if played.session.settings_path is not None:
        assert words[-2:] == ("--settings", str(played.session.settings_path))
        args = build_parser().parse_args(list(words[1:]))
        loaded = load(args.settings, blocks=ext.settings_blocks())
        assert loaded.defaults == settings.defaults
        assert loaded.endpoints == settings.endpoints
    assert "--json" not in words
    assert _command_spec(words, settings) == spec
    empty = dataclasses.replace(settings, defaults=MappingProxyType({}))
    assert _command_spec(words, empty) == spec
    _check_lines(played)


def _flag_value(words: Sequence[str], flag: str) -> str:
    return words[list(words).index(flag) + 1]


def test_settings_defaults_are_kept_and_spelled_out(workspace: Workspace) -> None:
    played = play(settings_defaults(workspace))
    assert played.outcome.action is Decision.RUN
    assert played.outcome.answers == {"input": workspace.tmp / "book.pdf"}
    spec = played.spec()
    assert spec.output.directory == workspace.tmp / "excerpts"
    assert spec.summary.fallback_context == "regional cookery"
    for phase in PHASES:
        model = spec.model(phase)
        assert (model.model, model.service_tier) == ("gpt-5.6-terra", "priority")
    words = played.command
    assert _flag_value(words, "--model") == "gpt-5.6-terra"
    assert _flag_value(words, "--service-tier") == "priority"
    assert _flag_value(words, "--fallback-context") == "regional cookery"


def test_code_defaults_chosen_against_settings_are_spelled_out(
    workspace: Workspace,
) -> None:
    played = play(back_to_code_defaults(workspace))
    answers = played.outcome.answers
    assert answers["beside_input"] is True
    assert answers["transcription_format"] == "md"
    assert answers["concurrency"] == 16
    assert answers["openalex"] is True
    words = played.command
    for flag, value in (
        ("--reasoning-effort", "high"),
        ("--transcription-format", "md"),
        ("--context", "auto"),
        ("--fallback-context", "none"),
        ("--concurrency", "16"),
    ):
        assert _flag_value(words, flag) == value
    assert {"--beside-input", "--openalex"} <= set(words)
    assert "--output" not in words


def test_a_provider_model_replaces_settings_endpoint_and_summary_model(
    workspace: Workspace,
) -> None:
    played = play(settings_endpoint_overridden(workspace))
    spec = played.spec()
    for phase in PHASES:
        model = spec.model(phase)
        assert (model.provider, model.endpoint, model.model) == (
            "openai",
            None,
            "gpt-5.6-luna",
        )
        assert model.reasoning_effort == "high"
    assert _flag_value(played.command, "--provider") == "openai"


def test_back_navigation_keeps_the_last_answers(workspace: Workspace) -> None:
    played = play(back_navigation(workspace))
    assert played.outcome.action is Decision.QUIT
    spec = played.spec()
    assert spec.output.directory is None
    assert spec.output.transcription_format == "txt"
    assert spec.summary.fallback_context == str(workspace.tmp / "fallback.txt")
    assert spec.transcription_model.provider == "openai"
    assert spec.run.concurrency == 8


def test_the_force_action_and_the_dry_run_reach_the_command(
    workspace: Workspace,
) -> None:
    forced = play(force_review_action(workspace))
    assert forced.outcome.action is Decision.RUN
    assert forced.outcome.answers["force"] is True
    assert "--force" in forced.command
    dry = play(dry_run(workspace))
    assert dry.outcome.action is Decision.RUN
    run = dry.spec().run
    assert (run.dry_run, run.force, run.retranscribe) == (True, True, True)
    assert not dry.spec().citations.openalex
