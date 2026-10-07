"""Scripted sessions through AutoExcerpter's wizard steps and review screen."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import autoexcerpter.run as core
from autoexcerpter import ext
from autoexcerpter.common.testing import KEEP, ScriptedPrompter, ScriptError
from autoexcerpter.common.wizard import (
    BACK,
    SETTINGS_LEGEND,
    Decision,
    Outcome,
    State,
    Step,
    StepContext,
)
from autoexcerpter.pipeline.resume import ProcessingState, ResumeResult
from autoexcerpter.settings import Endpoint, Settings
from autoexcerpter.spec import OPTIONS, RunSpec, parse
from autoexcerpter.ui.review import FORCE_ACTION, PROGRAM
from autoexcerpter.ui.session import Session, resolution
from autoexcerpter.ui.wizard import STEP_NAMES, build_steps, build_wizard
from tests.conftest import make_settings

KEYS = {"OPENAI_API_KEY": "test-key"}
_MISSING = object()


def _settings(**defaults: Any) -> Settings:
    checked = OPTIONS.check_defaults(defaults)
    return make_settings(defaults=MappingProxyType(checked))


def _checked(steps: Sequence[Step], extra: dict[str, set[str]]) -> list[Step]:
    """Wrap *steps* to record answers a step changes outside its ``sets``."""

    def wrap(step: Step) -> Step:
        def ask(ctx: StepContext) -> Any:
            before = ctx.state.answers
            result = step.ask(ctx)
            after = ctx.state.answers
            changed = {
                name
                for name in before.keys() | after.keys()
                if before.get(name, _MISSING) != after.get(name, _MISSING)
            }
            extra.setdefault(step.name, set()).update(changed - set(step.sets))
            return result

        return dataclasses.replace(step, ask=ask)

    return [wrap(step) for step in steps]


@dataclasses.dataclass
class Run:
    outcome: Outcome
    prompter: ScriptedPrompter
    session: Session

    @property
    def transcript(self) -> str:
        return self.prompter.transcript

    def spec(self) -> RunSpec:
        state = State(OPTIONS, self.session.settings.defaults, self.outcome.answers)
        return resolution(state, self.session).spec


def _session(
    settings: Settings | None = None,
    environ: Mapping[str, str] = KEYS,
    settings_path: Path | None = None,
) -> Session:
    return Session(settings or Settings(), settings_path, environ)


def _run(answers: Sequence[Any], session: Session | None = None) -> Run:
    session = session or _session()
    prompter = ScriptedPrompter(answers)
    extra: dict[str, set[str]] = {}
    wizard = build_wizard(
        session, prompter, steps=_checked(build_steps(session), extra), width=100
    )
    with prompter:
        outcome = wizard.run()
    assert {name: names for name, names in extra.items() if names} == {}
    return Run(outcome, prompter, session)


def _pdf_answers(pdf: Path, *review: Any) -> list[Any]:
    """Answers for one PDF with summary and every recommended value."""
    return [
        str(pdf),  # input
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
        *review,
    ]


def _assert_round_trip(run: Run) -> None:
    command = run.outcome.command
    assert command is not None
    assert tuple(command[:2]) == PROGRAM
    assert "--json" not in command
    argv = list(command[2:])
    spec = run.spec()
    assert parse(argv, run.session.settings.defaults) == spec
    assert parse(argv) == spec


def _headers(run: Run) -> list[str]:
    return [line for line in run.prompter.lines if line.startswith("AutoExcerpter |")]


@pytest.fixture
def pdf(make_pdf: Callable[..., Path]) -> Path:
    return make_pdf("book.pdf", num_pages=3)


def test_every_option_but_json_and_settings_belongs_to_a_step() -> None:
    declared = {name for step in build_steps(_session()) for name in step.sets}
    unprefixed = {
        name
        for name in OPTIONS.names
        if any(f"{phase}_{name}" in OPTIONS for phase in ("transcription", "summary"))
    }
    assert set(OPTIONS.names) - declared == {"json", "settings", *unprefixed}
    assert [step.name for step in build_steps(_session())] == list(STEP_NAMES)


def test_pdf_session_with_summary_runs_with_recommended_values(pdf: Path) -> None:
    run = _run(_pdf_answers(pdf, "run"))
    assert run.outcome.action is Decision.RUN
    assert "  found 1 PDF, 3 pages" in run.prompter.lines
    assert len(_headers(run)) == 9
    assert _headers(run)[0].endswith("step 1 of 9")
    spec = run.spec()
    assert spec.input.path == pdf
    assert spec.summary.enabled
    assert spec.summary_model.verbosity == "low"
    same = dataclasses.replace(spec.summary_model, verbosity="medium")
    assert same == spec.transcription_model
    assert run.outcome.command == ("autoexcerpter", "run", "--input", str(pdf))
    assert "Plan: 1 new, 0 partial, 0 complete" in run.prompter.lines
    assert "  book: new" in run.prompter.lines
    _assert_round_trip(run)


def test_back_returns_to_the_previous_question_and_step(pdf: Path) -> None:
    answers = _pdf_answers(pdf, "quit")
    answers[2:2] = [BACK, "folder", BACK, "beside"]
    answers[11:11] = ["anthropic", BACK]
    run = _run(answers)
    assert run.outcome.action is Decision.QUIT
    headers = _headers(run)
    assert [header.split("|")[1].strip() for header in headers[1:5]] == [
        "new run   step 2 of 9",
        "new run   step 3 of 9",
        "new run   step 2 of 9",
        "new run   step 3 of 9",
    ]
    assert run.prompter.lines.count("? Transcription provider:") == 2
    assert run.spec().transcription_model.provider == "openai"


def test_summary_off_skips_context_and_summary_model(pdf: Path) -> None:
    run = _run(
        [
            str(pdf),
            "beside",
            False,
            "txt",
            [],
            "openai",
            "gpt-5.6-luna",
            KEEP,
            KEEP,
            "recommended",
            "recommended",
            KEEP,
            KEEP,
            "run",
        ]
    )
    transcript = run.transcript
    assert "Summary context:" not in transcript
    assert "? Summary model:" not in transcript
    assert "OpenAlex" not in transcript
    assert "Summary as Markdown" not in transcript
    assert _headers(run)[-1].endswith("step 7 of 7")
    spec = run.spec()
    assert not spec.summary.enabled
    assert spec.output.transcription_format == "txt"
    assert spec.output.formats == ("summary-md", "summary-docx")
    _assert_round_trip(run)


def test_folder_selection_by_name_and_numbers(
    tmp_path: Path, make_pdf: Callable[..., Path]
) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    for name in ("alpha", "beta", "gamma"):
        make_pdf(f"scans/{name}.pdf", num_pages=2)
    answers = _pdf_answers(folder, "edit", "input", str(folder), "numbers", "1,3")
    answers[1:1] = ["name", "amm"]
    run = _run([*answers, "run"])
    assert "  found 3 PDFs, 6 pages" in run.prompter.lines
    assert "     2  beta.pdf  (PDF, 2 pages)" in run.prompter.lines
    assert "  selected 1 of 3 items:" in run.prompter.lines
    assert "  selected 2 of 3 items:" in run.prompter.lines
    spec = run.spec()
    assert spec.input.select == "1,3"
    assert not spec.input.all
    _assert_round_trip(run)


def test_invalid_and_empty_inputs_re_ask(tmp_path: Path, pdf: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ScriptError, match="no such file"):
        _run([str(tmp_path / "missing.pdf")])
    run = _run([str(empty), *_pdf_answers(pdf, "quit")])
    assert f"  found no PDF or image folder in {empty}" in run.prompter.lines
    assert run.spec().input.path == pdf


def test_settings_defaults_are_preselected_marked_and_spelled_out(pdf: Path) -> None:
    settings = _settings(
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        formats=["summary-md", "sqlite"],
        dpi=300,
        concurrency=4,
    )
    run = _run(
        [
            str(pdf),
            "beside",
            True,
            "md",
            KEEP,
            "auto",
            "none",
            "openai",
            KEEP,
            KEEP,
            KEEP,
            "recommended",
            "same",
            "recommended",
            KEEP,
            KEEP,
            "run",
        ],
        _session(settings),
    )
    lines = run.prompter.lines
    assert " > gpt-5.6-terra (s)" in lines
    assert " > medium (s)" in lines
    assert "   [x] Summary as Markdown (X_summary.md) (s)" in lines
    assert "? Parallel page requests: [4 (s)] 4" in lines
    assert any("300 DPI (s)" in line for line in lines)
    review = run.transcript.split("\nReview")[-1]
    assert SETTINGS_LEGEND in review
    assert "openai / gpt-5.6-terra (s)" in review
    assert "reasoning medium (s)" in review
    assert "formats summary-md, sqlite (s)" in review
    assert "300 DPI (s)" in review
    command = run.outcome.command
    assert command is not None
    for words in (
        ("--model", "gpt-5.6-terra"),
        ("--reasoning-effort", "medium"),
        ("--formats", "summary-md,sqlite"),
        ("--dpi", "300"),
        ("--concurrency", "4"),
    ):
        index = command.index(words[0])
        assert tuple(command[index : index + 2]) == words
    _assert_round_trip(run)


def test_resets_are_offered_only_without_a_settings_default(pdf: Path) -> None:
    free = _run(_pdf_answers(pdf, "quit"))
    assert "Automatic  schema for this model" in free.transcript
    settings = _settings(response_format="json", image_format="png")
    answers = _pdf_answers(pdf, "quit")
    answers[13] = "customize"
    answers[14:14] = ["", KEEP, KEEP, KEEP]
    run = _run(answers, _session(settings))
    transcript = run.transcript
    assert "Automatic" not in transcript
    assert " > JSON (prompted, validated) (s)" in transcript
    assert "Recommended (jpeg)" not in transcript
    assert " > PNG (s)" in transcript
    assert "Recommended (original)" in transcript
    assert "JPEG quality" not in transcript
    spec = run.spec()
    assert spec.transcription_model.response_format == "json"
    assert spec.images.format == "png"
    assert spec.images.dpi is None


def _tuned(pdf: Path, block: Sequence[Any]) -> list[Any]:
    """Answers with gpt-4o and customized parameters: tokens, temperature, top p."""
    answers = _pdf_answers(pdf, "run")
    answers[8:12] = ["<other>", "gpt-4o", KEEP, "customize", *block, KEEP]
    return answers


def _response(pdf: Path, block: Sequence[Any]) -> list[Any]:
    answers = _pdf_answers(pdf, "run")
    answers[10:11] = block
    return answers


def _images(pdf: Path, block: Sequence[Any]) -> list[Any]:
    """Answers with customized images: DPI, detail, format, quality, grayscale."""
    answers = _pdf_answers(pdf, "run")
    answers[13:14] = ["customize", *block]
    return answers


@dataclasses.dataclass(frozen=True)
class ResetCase:
    """A question that offers a reset to the unset value only without a default.

    ``shown`` is the question's line under the settings default, ``hint`` the
    reset text offered without one, and ``words`` the command words of
    ``new``.
    """

    option: str
    question: str
    default: Any
    build: Callable[[Path, Sequence[Any]], list[Any]]
    size: int
    index: int
    new: Any
    reset: Any
    shown: str
    hint: str
    words: tuple[str, ...]
    stem: str
    field: Callable[[RunSpec], Any]
    value: Any

    def answers(self, pdf: Path, answer: Any) -> list[Any]:
        block: list[Any] = [KEEP] * self.size
        block[self.index] = answer
        return self.build(pdf, block)


RESET_CASES = [
    ResetCase(
        option="response_format",
        question="Response format:",
        default="json",
        build=_response,
        size=1,
        index=0,
        new="text",
        reset=None,
        shown=" > JSON (prompted, validated) (s)",
        hint="Automatic",
        words=("--response-format", "text"),
        stem="response-format",
        field=lambda spec: spec.transcription_model.response_format,
        value="text",
    ),
    ResetCase(
        option="max_output_tokens",
        question="Max output tokens:",
        default=4000,
        build=_tuned,
        size=3,
        index=0,
        new="8000",
        reset="",
        shown="? Max output tokens: [4000 (s)] 8000",
        hint="Max output tokens (empty = the model's limit):",
        words=("--transcription-max-output-tokens", "8000"),
        stem="max-output-tokens",
        field=lambda spec: spec.transcription_model.max_output_tokens,
        value=8000,
    ),
    ResetCase(
        option="top_p",
        question="Top p (0.0 to 1.0):",
        default=0.5,
        build=_tuned,
        size=3,
        index=2,
        new="0.9",
        reset="",
        shown="? Top p (0.0 to 1.0): [0.5 (s)] 0.9",
        hint="Top p (0.0 to 1.0, empty = not sent):",
        words=("--transcription-top-p", "0.9"),
        stem="top-p",
        field=lambda spec: spec.transcription_model.top_p,
        value=0.9,
    ),
    ResetCase(
        option="dpi",
        question="DPI (native or a number):",
        default=300,
        build=_images,
        size=5,
        index=0,
        new="200",
        reset="",
        shown="? DPI (native or a number): [300 (s)] 200",
        hint="DPI (native or a number, empty = native):",
        words=("--dpi", "200"),
        stem="--dpi",
        field=lambda spec: spec.images.dpi,
        value=200,
    ),
    ResetCase(
        option="image_detail",
        question="Image detail:",
        default="low",
        build=_images,
        size=5,
        index=1,
        new="high",
        reset=None,
        shown=" > low (s)",
        hint="Recommended (original)",
        words=("--image-detail", "high"),
        stem="image-detail",
        field=lambda spec: spec.images.detail,
        value="high",
    ),
    ResetCase(
        option="image_format",
        question="Image format:",
        default="png",
        build=_images,
        size=5,
        index=2,
        new="jpeg",
        reset=None,
        shown=" > PNG (s)",
        hint="Recommended (jpeg)",
        words=("--image-format", "jpeg"),
        stem="image-format",
        field=lambda spec: spec.images.format,
        value="jpeg",
    ),
    ResetCase(
        option="jpeg_quality",
        question="JPEG quality (1 to 100):",
        default=80,
        build=_images,
        size=5,
        index=3,
        new="90",
        reset="",
        shown="? JPEG quality (1 to 100): [80 (s)] 90",
        hint="JPEG quality (1 to 100, empty = 95):",
        words=("--jpeg-quality", "90"),
        stem="jpeg-quality",
        field=lambda spec: spec.images.jpeg_quality,
        value=90,
    ),
    ResetCase(
        option="grayscale",
        question="Grayscale:",
        default=False,
        build=_images,
        size=5,
        index=4,
        new=True,
        reset=None,
        shown=" > No (s)",
        hint="Recommended (yes)",
        words=("--grayscale",),
        stem="grayscale",
        field=lambda spec: spec.images.grayscale,
        value=True,
    ),
]


def _has_words(command: Sequence[str], words: tuple[str, ...]) -> bool:
    return any(
        tuple(command[index : index + len(words)]) == words
        for index in range(len(command))
    )


@pytest.mark.parametrize("case", RESET_CASES, ids=[c.option for c in RESET_CASES])
def test_a_settings_default_withholds_the_reset_and_a_new_answer_is_spelled_out(
    case: ResetCase, pdf: Path
) -> None:
    session = _session(_settings(**{case.option: case.default}))
    refused = f"^'{re.escape(case.question)}': .*(rejected|is not one of)"
    with pytest.raises(ScriptError, match=refused):
        _run(case.answers(pdf, case.reset), session)
    run = _run(case.answers(pdf, case.new), session)
    assert case.shown in run.prompter.lines
    assert case.hint not in run.transcript
    assert case.field(run.spec()) == case.value
    command = run.outcome.command
    assert command is not None
    assert _has_words(command, case.words)
    assert "Equivalent command:" in run.prompter.lines
    _assert_round_trip(run)


@pytest.mark.parametrize("case", RESET_CASES, ids=[c.option for c in RESET_CASES])
def test_without_a_settings_default_the_reset_leaves_the_option_unset(
    case: ResetCase, pdf: Path
) -> None:
    run = _run(case.answers(pdf, case.reset))
    assert case.hint in run.transcript
    assert case.field(run.spec()) is None
    command = run.outcome.command
    assert command is not None
    assert not any(case.stem in word for word in command)
    _assert_round_trip(run)


def test_same_summary_model_keeps_a_summary_settings_default(pdf: Path) -> None:
    run = _run(
        _pdf_answers(pdf, "run"), _session(_settings(summary_response_format="json"))
    )
    spec = run.spec()
    assert spec.transcription_model.response_format is None
    assert spec.summary_model.response_format == "json"
    command = run.outcome.command
    assert command is not None
    assert _has_words(command, ("--summary-response-format", "json"))
    _assert_round_trip(run)


def test_same_summary_model_copies_an_unset_value_without_a_default(
    pdf: Path,
) -> None:
    run = _run(_pdf_answers(pdf, "run"))
    spec = run.spec()
    assert spec.transcription_model.response_format is None
    assert spec.summary_model.response_format is None
    command = run.outcome.command
    assert command is not None
    assert not any("response-format" in word for word in command)


def test_custom_parameters_and_own_summary_model(pdf: Path) -> None:
    answers = _pdf_answers(pdf)
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
        "anthropic",
        "<other>",
        "claude-sonnet-4-5",
        KEEP,
        "text",
        "recommended",
    ]
    answers.append("copy")
    run = _run(answers, _session(environ={**KEYS, "ANTHROPIC_API_KEY": "x"}))
    spec = run.spec()
    transcription = spec.transcription_model
    assert (transcription.model, transcription.max_output_tokens) == ("gpt-4o", 8000)
    assert (transcription.temperature, transcription.top_p) == (0.4, 0.9)
    assert transcription.service_tier == "priority"
    assert transcription.response_format == "json"
    summary = spec.summary_model
    assert (summary.provider, summary.model) == ("anthropic", "claude-sonnet-4-5")
    assert summary.response_format == "text"
    assert run.outcome.action is Decision.COPY
    _assert_round_trip(run)


def test_model_name_implies_its_provider(pdf: Path) -> None:
    answers = _pdf_answers(pdf, "quit")
    answers[8:10] = ["<other>", "gemini-3.1-pro", KEEP]
    run = _run(answers, _session(environ={"GOOGLE_API_KEY": "x"}))
    assert "  provider google, as the model name implies" in run.prompter.lines
    model = run.spec().transcription_model
    assert (model.provider, model.model) == ("google", "gemini-3.1-pro")
    _assert_round_trip(run)


def test_named_endpoints_from_the_settings(pdf: Path) -> None:
    endpoint = Endpoint(
        "local", "http://127.0.0.1:8000/v1", "LOCAL_KEY", supports_schema=True
    )
    settings = make_settings(endpoints=MappingProxyType({"local": endpoint}))
    answers = _pdf_answers(pdf, "run")
    answers[7:12] = ["endpoint:local", "qwen-vl", "schema", "recommended"]
    run = _run(answers, _session(settings, environ={"LOCAL_KEY": "x"}))
    assert "   Endpoint local  http://127.0.0.1:8000/v1" in run.prompter.lines
    spec = run.spec()
    for phase in ("transcription", "summary"):
        model = spec.model(phase)
        assert (model.provider, model.endpoint, model.model) == (
            "custom",
            "local",
            "qwen-vl",
        )
    assert "endpoint local / qwen-vl" in run.transcript
    command = run.outcome.command
    assert command is not None
    assert "--endpoint" in command
    _assert_round_trip(run)


def test_context_text_and_fallback_file(tmp_path: Path, pdf: Path) -> None:
    topics = tmp_path / "topics.txt"
    topics.write_text("trade; prices\n", encoding="utf-8")
    answers = _pdf_answers(pdf, "run")
    answers[5:7] = ["auto", "file", str(topics)]
    run = _run(answers)
    assert run.spec().summary.fallback_context == str(topics)
    _assert_round_trip(run)
    answers = _pdf_answers(pdf, "run")
    answers[5:7] = ["text", "early modern spice trade"]
    run = _run(answers)
    assert run.spec().summary.context == "early modern spice trade"
    assert "context: early modern spice trade" in run.transcript
    _assert_round_trip(run)


def test_extension_steps_follow_the_tool_steps(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    def ask(ctx: StepContext) -> Any:
        return ctx.text_option("concurrency", "Extension concurrency:")

    step = Step("extension", "Extension", ask=ask, sets=("concurrency",))
    monkeypatch.setattr(ext, "wizard_steps", lambda: (step,))
    answers = _pdf_answers(pdf, "run")
    answers.insert(16, "3")
    run = _run(answers)
    assert _headers(run)[-1].endswith("step 10 of 10")
    assert run.spec().run.concurrency == 3
    _assert_round_trip(run)


def test_settings_file_joins_the_equivalent_command(tmp_path: Path, pdf: Path) -> None:
    settings_path = tmp_path / "other.yaml"
    run = _run(_pdf_answers(pdf, "copy"), _session(settings_path=settings_path))
    assert run.outcome.command is not None
    assert run.outcome.command[-2:] == ("--settings", str(settings_path))


def _complete_unless_forced(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(core, "plan", plan)


def test_force_is_a_review_action_for_complete_items(
    monkeypatch: pytest.MonkeyPatch, pdf: Path
) -> None:
    _complete_unless_forced(monkeypatch)
    run = _run(_pdf_answers(pdf, FORCE_ACTION, "run"))
    lines = run.prompter.lines
    assert "   Reprocess 1 complete item(s) (--force)" in lines
    assert "  book: complete (skipped)" in lines
    assert run.outcome.answers["force"] is True
    assert run.outcome.command is not None
    assert "--force" in run.outcome.command
    assert run.prompter.lines.count("Plan: 1 new, 0 partial, 0 complete") == 1
    _assert_round_trip(run)


def test_a_plan_error_withholds_run_until_a_step_is_edited(
    tmp_path: Path, pdf: Path
) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    answers = _pdf_answers(pdf)
    answers[5:7] = ["file", str(empty)]
    with pytest.raises(ScriptError, match="'run' is not one of"):
        _run([*answers, "run"])
    run = _run([*answers, "edit", "context", "auto", "none", "run"])
    lines = run.prompter.lines
    assert any(line.startswith("Plan error: Context file is empty") for line in lines)
    assert "Run unavailable: the plan failed; edit a step" in lines
    assert run.outcome.action is Decision.RUN
    assert run.spec().summary.context is None


def test_a_missing_key_variable_withholds_run(pdf: Path) -> None:
    run = _run(_pdf_answers(pdf, "quit"), _session(environ={}))
    lines = run.prompter.lines
    assert (
        "Missing API key: OPENAI_API_KEY is not set (transcription, summary)" in lines
    )
    assert "Run unavailable: set OPENAI_API_KEY in the environment" in lines
    assert "   Run" not in lines and " > Run" not in lines
    dry = _pdf_answers(pdf, "run")
    dry[14] = ["openalex", "dry_run"]
    run = _run(dry, _session(environ={}))
    assert run.spec().run.dry_run
    assert "Missing API key" not in run.transcript
