"""Tests for common/wizard.py and the scripted prompter in common/testing.py."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.common.options import Kind, Option, OptionError, OptionTable, Source
from autoexcerpter.common.testing import KEEP, ScriptedPrompter, ScriptError
from autoexcerpter.common.wizard import (
    BACK,
    DONE,
    Choice,
    Decision,
    Line,
    Nav,
    Part,
    Review,
    ReviewAction,
    State,
    Step,
    StepContext,
    Wizard,
    command_display,
    command_line,
    copy_to_clipboard,
    option_validator,
    quote_word,
    render_review,
)


def _table() -> OptionTable:
    return OptionTable(
        [
            Option("input", "--input", "input path", type=Path),
            Option("output", "--output", "output folder", type=Path, exclusive="where"),
            Option(
                "beside",
                "--beside-input",
                "write beside the input",
                kind=Kind.FLAG,
                default=False,
                exclusive="where",
            ),
            Option(
                "summarize", "--summarize", "summarize", kind=Kind.SWITCH, default=True
            ),
            Option(
                "formats",
                "--formats",
                "output formats",
                kind=Kind.LIST,
                choices=("md", "docx", "sqlite"),
                default=("md", "sqlite"),
            ),
            Option("model", "--model", "model", default="m1"),
            Option(
                "summary_model", "--summary-model", "summary model", fallback="model"
            ),
            Option("workers", "--workers", "workers", type=int, default=4),
        ]
    )


def _ask_input(ctx: StepContext) -> Nav:
    return ctx.path_option(
        "input",
        "Input (file or folder):",
        validate=lambda text: None if text.strip() else "enter a path",
    )


def _ask_output(ctx: StepContext) -> Nav:
    state = ctx.state

    def where() -> Nav:
        current = "folder" if state.value("output") is not None else "beside"
        marker = state.marker("output") if current == "folder" else None
        answer = ctx.prompter.select(
            "Output:",
            [Choice("beside", "Beside the input"), Choice("folder", "A folder")],
            default=current,
            marker=marker,
        )
        if answer is BACK:
            return BACK
        if answer == "beside":
            state.set("beside", True)
        else:
            state.clear("beside")
        return DONE

    def folder() -> Nav | None:
        if state.value("beside"):
            return None
        return ctx.path_option("output", "Output folder:", only_directories=True)

    return ctx.sequence([where, folder])


def _summarizing(state: State) -> bool:
    return bool(state.value("summarize"))


STEPS = [
    Step("input", "Input", ask=_ask_input, sets=("input",)),
    Step("output", "Output", ask=_ask_output, sets=("output", "beside")),
    Step(
        "summary",
        "Summary",
        ask=lambda ctx: ctx.select_option(
            "summarize", "Summarize?", [Choice(True, "Yes"), Choice(False, "No")]
        ),
        sets=("summarize",),
    ),
    Step(
        "summary_model",
        "Summary model",
        ask=lambda ctx: ctx.text_option("summary_model", "Summary model:", empty=None),
        sets=("summary_model",),
        applies=_summarizing,
    ),
    Step(
        "formats",
        "Formats",
        ask=lambda ctx: ctx.checkbox_option(
            "formats",
            "Formats:",
            [
                Choice("md", "Markdown"),
                Choice("docx", "Word"),
                Choice("sqlite", "SQLite"),
            ],
        ),
        sets=("formats",),
    ),
]


def _review(state: State) -> Review:
    output = "beside input" if state.value("beside") else str(state.value("output"))
    return Review(
        lines=[
            Line("input", [str(state.value("input"))]),
            Line("output", [state.part("output", output)]),
            Line(
                "model",
                [
                    state.part("model", str(state.value("model"))),
                    state.part("workers", f"workers {state.value('workers')}"),
                ],
            ),
        ]
    )


def _command(state: State) -> list[str]:
    return ["tool", "run", *state.table.to_argv(state.answers)]


def _wizard(
    answers: Sequence[Any],
    *,
    defaults: dict[str, Any] | None = None,
    review: Callable[[State], Review] = _review,
    command: Callable[[State], Sequence[str]] | None = _command,
    extensions: Sequence[Step] = (),
) -> tuple[Wizard, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    wizard = Wizard(
        State(_table(), defaults),
        prompter,
        [*STEPS, *extensions],
        title="Tool",
        task="new run",
        review=review,
        command=command,
    )
    return wizard, prompter


def _headers(prompter: ScriptedPrompter) -> list[str]:
    return [line for line in prompter.lines if line.startswith("Tool | ")]


class TestNavigation:
    def test_a_full_session_runs(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "folder", "out", True, "s1", ["md", "docx"], "run"]
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.action is Decision.RUN
        assert outcome.answers == {
            "input": Path("a.pdf"),
            "output": Path("out"),
            "summary_model": "s1",
            "formats": ("md", "docx"),
        }
        assert outcome.command == (
            "tool",
            "run",
            "--input",
            "a.pdf",
            "--output",
            "out",
            "--formats",
            "md,docx",
            "--summary-model",
            "s1",
        )

    def test_step_counter_follows_applicable_steps(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, BACK, True, "s", KEEP, "quit"]
        )
        with prompter:
            wizard.run()

        assert _headers(prompter) == [
            "Tool | new run   step 1 of 5",
            "Tool | new run   step 2 of 5",
            "Tool | new run   step 3 of 5",
            "Tool | new run   step 4 of 4",
            "Tool | new run   step 3 of 4",
            "Tool | new run   step 4 of 5",
            "Tool | new run   step 5 of 5",
        ]

    def test_back_passes_over_skipped_steps(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, BACK, BACK, "folder", "o", KEEP, KEEP, "quit"]
        )
        with prompter:
            outcome = wizard.run()

        assert [h.rsplit("   ", 1)[1] for h in _headers(prompter)] == [
            "step 1 of 5",
            "step 2 of 5",
            "step 3 of 5",
            "step 4 of 4",
            "step 3 of 4",
            "step 2 of 4",
            "step 3 of 4",
            "step 4 of 4",
        ]
        assert outcome.answers["output"] == Path("o")
        assert "beside" not in outcome.answers

    def test_back_inside_a_step_returns_to_its_previous_question(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "folder", BACK, "beside", KEEP, KEEP, KEEP, "quit"]
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.answers["beside"] is True

    def test_back_on_the_first_step_offers_to_quit(self) -> None:
        wizard, prompter = _wizard([BACK, "quit"])
        with prompter:
            outcome = wizard.run()

        assert outcome.action is Decision.QUIT
        assert "? Quit without running?" in prompter.lines

    def test_staying_asks_the_first_step_again(self) -> None:
        wizard, prompter = _wizard(
            [BACK, BACK, "a.pdf", "beside", KEEP, KEEP, KEEP, "quit"]
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.answers["input"] == Path("a.pdf")

    def test_back_from_the_review_returns_to_the_last_step(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, KEEP, BACK, ["docx"], "run"]
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.answers["formats"] == ("docx",)
        assert _headers(prompter)[-1] == "Tool | new run   step 4 of 4"

    def test_skipped_step_answers_are_cleared(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "beside", True, "s1", BACK, BACK, False, KEEP, "run"]
        )
        with prompter:
            outcome = wizard.run()

        assert "summary_model" not in outcome.answers
        assert outcome.answers["summarize"] is False

    def test_extension_steps_come_before_the_review(self) -> None:
        extra = Step(
            "workers",
            "Workers",
            ask=lambda ctx: ctx.text_option("workers", "Workers:"),
            sets=("workers",),
        )
        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, KEEP, "8", "run"], extensions=[extra]
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.answers["workers"] == 8
        assert _headers(prompter)[-1] == "Tool | new run   step 5 of 5"

    def test_step_validation(self) -> None:
        bad = Step("x", "X", ask=lambda ctx: DONE, sets=("missing",))
        with pytest.raises(OptionError):
            Wizard(
                State(_table()),
                ScriptedPrompter([]),
                [bad],
                title="T",
                task="t",
                review=_review,
            )
        with pytest.raises(ValueError, match="duplicate"):
            Wizard(
                State(_table()),
                ScriptedPrompter([]),
                [STEPS[0], STEPS[0]],
                title="T",
                task="t",
                review=_review,
            )

    def test_sequence_skips_none_on_back(self) -> None:
        calls: list[str] = []
        script = iter([DONE, BACK, DONE, DONE])

        def first() -> Nav:
            calls.append("first")
            return next(script)

        def skipped() -> None:
            calls.append("skipped")

        def last() -> Nav:
            calls.append("last")
            return next(script)

        assert StepContext.sequence([first, skipped, last]) is DONE
        assert calls == ["first", "skipped", "last", "first", "skipped", "last"]
        assert StepContext.sequence([lambda: BACK]) is BACK


class TestState:
    def test_unchanged_settings_default_keeps_its_source(self) -> None:
        state = State(_table(), {"workers": 8})
        state.set("workers", 8)

        assert state.answers == {}
        assert state.source("workers") is Source.SETTINGS
        assert state.marker("workers") == "(s)"

    def test_code_default_against_a_settings_default_is_stored(self) -> None:
        state = State(_table(), {"workers": 8})
        state.set("workers", 4)

        assert state.answers == {"workers": 4}
        assert state.source("workers") is Source.FLAG
        assert state.marker("workers") is None

    def test_setting_the_resolved_value_removes_the_answer(self) -> None:
        state = State(_table(), answers={"workers": 2})
        state.set("workers", 4)

        assert state.answers == {}
        assert state.source("workers") is Source.DEFAULT

    def test_exclusive_members_clear_each_other(self) -> None:
        state = State(_table(), {"output": Path("o")})
        assert state.from_settings("output")

        state.set("beside", True)
        assert state.answers == {"beside": True}
        assert state.value("output") is None
        assert state.source("output") is Source.FLAG

        state.set("output", Path("o"))
        assert state.answers == {}
        assert state.value("output") == Path("o")
        assert state.from_settings("output")

    def test_unset_member_does_not_clear_the_others(self) -> None:
        state = State(_table(), answers={"output": Path("x")})
        state.set("beside", False)

        assert state.answers == {"output": Path("x")}

    def test_fallback_chain(self) -> None:
        state = State(_table())
        state.set("model", "x")
        assert state.value("summary_model") == "x"

        state.set("summary_model", "x")
        assert "summary_model" not in state.answers
        state.set("summary_model", "y")
        assert state.answers["summary_model"] == "y"

        state.clear("summary_model")
        assert state.value("summary_model") == "x"

    def test_lists_are_stored_as_tuples(self) -> None:
        state = State(_table())
        state.set("formats", ["md", "sqlite"])
        assert state.answers == {}
        state.set("formats", ["docx"])
        assert state.answers == {"formats": ("docx",)}

    def test_unknown_names_are_rejected(self) -> None:
        with pytest.raises(OptionError):
            State(_table()).set("nope", 1)
        with pytest.raises(OptionError):
            State(_table(), answers={"nope": 1})

    def test_part_flags_settings_defaults(self) -> None:
        state = State(_table(), {"model": "big"})
        assert state.part("model", "big") == Part("big", True)
        assert state.part("workers", "4") == Part("4", False)

    def test_option_validator(self) -> None:
        validate = option_validator(_table()["workers"])
        assert validate("3") is None
        assert "--workers" in str(validate("x"))
        assert option_validator(_table()["workers"], empty_ok=True)(" ") is None


class TestMarkers:
    def test_settings_defaults_are_preselected_and_marked(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", KEEP, KEEP, KEEP, KEEP, "quit"],
            defaults={"output": Path("o"), "summarize": False, "model": "big"},
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.answers == {"input": Path("a.pdf")}
        assert " > A folder (s)" in prompter.lines
        assert " > No (s)" in prompter.lines
        assert any(
            line.startswith("? Output folder: [o (s)]") for line in prompter.lines
        )
        assert "  model    big (s) | workers 4" in prompter.lines

    def test_a_changed_value_loses_its_marker(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "beside", True, KEEP, KEEP, "quit"],
            defaults={"output": Path("o")},
        )
        with prompter:
            wizard.run()

        assert "  output   beside input" in prompter.lines

    def test_checked_settings_items_are_marked(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, ["sqlite"], "quit"],
            defaults={"formats": ("docx",)},
        )
        with prompter:
            outcome = wizard.run()

        assert "   [x] Word (s)" in prompter.lines
        assert "   [ ] SQLite" in prompter.lines
        assert outcome.answers["formats"] == ("sqlite",)


class TestReview:
    def test_render_review_layout(self) -> None:
        review = Review(
            lines=[
                Line("input", ["C:\\scans\\a.pdf  (1 PDF, 212 pages)"]),
                Line("model", [Part("openai / big", True), "reasoning high"]),
            ],
            notes=["Plan:", "  a.pdf  new"],
            withheld="no API key",
        )
        text = render_review(review, command=["  tool run --input a.pdf"], width=60)

        assert text.split("\n") == [
            "Review" + " " * 26 + "(s) = from settings defaults",
            "  input   C:\\scans\\a.pdf  (1 PDF, 212 pages)",
            "  model   openai / big (s) | reasoning high",
            "",
            "Plan:",
            "  a.pdf  new",
            "",
            "Run unavailable: no API key",
            "",
            "Equivalent command:",
            "  tool run --input a.pdf",
        ]

    def test_legend_only_with_settings_values(self) -> None:
        text = render_review(Review(lines=[Line("a", ["b"])]), command_error="bad")
        assert text.split("\n") == [
            "Review",
            "  a   b",
            "",
            "Equivalent command unavailable: bad",
        ]

    def test_copy_returns_the_command(self) -> None:
        wizard, prompter = _wizard(["a.pdf", "beside", False, KEEP, "copy"])
        with prompter:
            outcome = wizard.run()

        assert outcome.action is Decision.COPY
        assert outcome.command == (
            "tool",
            "run",
            "--input",
            "a.pdf",
            "--beside-input",
            "--no-summarize",
        )
        assert "Equivalent command:" in prompter.lines

    def test_command_error_is_shown_and_withholds_copy(self) -> None:
        def broken(state: State) -> list[str]:
            raise ValueError("cannot spell the image settings")

        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, KEEP, "copy"], command=broken
        )
        with pytest.raises(ScriptError, match="'copy' is not one of"):
            wizard.run()
        assert (
            "Equivalent command unavailable: cannot spell the image settings"
            in prompter.lines
        )

    def test_withheld_run_is_not_offered(self) -> None:
        def withheld(state: State) -> Review:
            return Review(lines=[], withheld="plan error: nothing found")

        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, KEEP, "run"], review=withheld
        )
        with pytest.raises(ScriptError, match="'run' is not one of"):
            wizard.run()
        assert "Run unavailable: plan error: nothing found" in prompter.lines

    def test_a_tool_action_changes_answers_and_returns_to_the_review(self) -> None:
        def with_action(state: State) -> Review:
            actions = []
            if not state.is_answered("workers"):
                actions.append(
                    ReviewAction("more", "Use 9 workers", lambda s: s.set("workers", 9))
                )
            return Review(lines=[], actions=actions)

        wizard, prompter = _wizard(
            ["a.pdf", "beside", False, KEEP, "more", "run"], review=with_action
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.answers["workers"] == 9
        assert prompter.lines.count("? Start:") == 2
        assert "   Use 9 workers" in prompter.lines

    def test_reserved_action_keys_are_rejected(self) -> None:
        def clash(state: State) -> Review:
            return Review(lines=[], actions=[ReviewAction("run", "x", lambda s: None)])

        wizard, _ = _wizard(["a.pdf", "beside", False, KEEP], review=clash)
        with pytest.raises(ValueError, match="reserved"):
            wizard.run()

    def test_edit_a_step_and_ask_newly_applicable_steps(self) -> None:
        wizard, prompter = _wizard(
            [
                "a.pdf",
                "beside",
                False,
                KEEP,
                "edit",
                "summary",
                True,
                "s2",
                "run",
            ]
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.answers["summary_model"] == "s2"
        assert "summarize" not in outcome.answers
        assert _headers(prompter)[-2:] == [
            "Tool | new run   step 3 of 4",
            "Tool | new run   step 4 of 5",
        ]

    def test_back_while_editing_returns_to_the_review(self) -> None:
        wizard, prompter = _wizard(
            [
                "a.pdf",
                "beside",
                False,
                KEEP,
                "edit",
                BACK,
                "edit",
                "input",
                BACK,
                "quit",
            ]
        )
        with prompter:
            outcome = wizard.run()

        assert outcome.action is Decision.QUIT
        assert prompter.lines.count("? Start:") == 3

    def test_quit(self) -> None:
        wizard, prompter = _wizard(["a.pdf", "beside", False, KEEP, "quit"])
        with prompter:
            outcome = wizard.run()

        assert outcome.action is Decision.QUIT
        assert outcome.answers["input"] == Path("a.pdf")


class TestScriptedPrompter:
    def test_transcript_layout(self) -> None:
        wizard, prompter = _wizard(
            ["a.pdf", "beside", BACK, "beside", KEEP, KEEP, KEEP, "run"],
            defaults={"model": "big"},
        )
        with prompter:
            wizard.run()

        assert prompter.transcript == "\n".join(
            [
                "",
                "Tool | new run   step 1 of 5",
                "? Input (file or folder): a.pdf",
                "",
                "Tool | new run   step 2 of 5",
                "? Output:",
                " > Beside the input",
                "   A folder",
                "   Back",
                "  answer: Beside the input",
                "",
                "Tool | new run   step 3 of 5",
                "? Summarize?",
                " > Yes",
                "   No",
                "   Back",
                "  answer: Back",
                "",
                "Tool | new run   step 2 of 5",
                "? Output:",
                " > Beside the input",
                "   A folder",
                "   Back",
                "  answer: Beside the input",
                "",
                "Tool | new run   step 3 of 5",
                "? Summarize?",
                " > Yes",
                "   No",
                "   Back",
                "  answer: Yes",
                "",
                "Tool | new run   step 4 of 5",
                "? Summary model: [big (s)] big",
                "",
                "Tool | new run   step 5 of 5",
                "? Formats:",
                "   [x] Markdown",
                "   [ ] Word",
                "   [x] SQLite",
                "   [ ] Back",
                "  answer: Markdown, SQLite",
                "",
                "Review" + " " * 46 + "(s) = from settings defaults",
                "  input    a.pdf",
                "  output   beside input",
                "  model    big (s) | workers 4",
                "",
                "Equivalent command:",
                "  tool run --input a.pdf --beside-input",
                "? Start:",
                " > Run",
                "   Edit a step",
                "   Copy command and quit",
                "   Quit",
                "   Back",
                "  answer: Run",
                "",
            ]
        )

    def test_an_answer_not_offered_names_the_question(self) -> None:
        prompter = ScriptedPrompter(["z"])
        with pytest.raises(ScriptError, match="'Pick'.*'z'"):
            prompter.select("Pick", [Choice("a", "A")])

    def test_running_out_of_answers_fails(self) -> None:
        with pytest.raises(ScriptError, match="no answer left for 'Name'"):
            ScriptedPrompter([]).text("Name")

    def test_unused_answers_fail(self) -> None:
        with pytest.raises(ScriptError, match="unused"), ScriptedPrompter(["x"]):
            pass

    def test_validator_rejection_fails(self) -> None:
        prompter = ScriptedPrompter(["bad"])
        with pytest.raises(ScriptError, match="rejected: no"):
            prompter.path("Input", validate=lambda text: "no")

    def test_text_back_and_keep(self) -> None:
        prompter = ScriptedPrompter(["<", KEEP, BACK])
        assert prompter.text("A", validate=lambda text: "never") is BACK
        assert prompter.text("B", default="d") == "d"
        assert prompter.text("C") is BACK

    def test_checkbox_checks_values(self) -> None:
        prompter = ScriptedPrompter([["z"], "a", BACK, KEEP])
        choices = [Choice("a", "A"), Choice("b", "B")]
        with pytest.raises(ScriptError, match="'z' is not one of"):
            prompter.checkbox("Pick", choices)
        with pytest.raises(ScriptError, match="expected a list"):
            prompter.checkbox("Pick", choices)
        assert prompter.checkbox("Pick", choices) is BACK
        assert prompter.checkbox("Pick", choices, checked=["b"]) == ["b"]


class TestCommand:
    @pytest.mark.parametrize(
        ("word", "windows", "quoted"),
        [
            ("plain", True, "plain"),
            ("with space", True, '"with space"'),
            ('say "hi"', True, '"say \\"hi\\""'),
            ("C:\\dir with space\\", True, '"C:\\dir with space\\\\"'),
            ("C:\\plain\\path", True, "C:\\plain\\path"),
            ("", True, '""'),
            ("plain", False, "plain"),
            ("with space", False, "'with space'"),
            ("it's", False, "'it'\"'\"'s'"),
            ("a\\b", False, "'a\\b'"),
        ],
    )
    def test_quote_word(self, word: str, windows: bool, quoted: str) -> None:
        assert quote_word(word, windows=windows) == quoted

    def test_command_line(self) -> None:
        words = ["tool", "run", "--input", "C:\\a b\\x.pdf"]
        assert command_line(words, windows=True) == 'tool run --input "C:\\a b\\x.pdf"'
        assert command_line(words, windows=False) == (
            "tool run --input 'C:\\a b\\x.pdf'"
        )

    def test_display_keeps_flags_with_their_values(self) -> None:
        words = [
            "tool",
            "run",
            "--input",
            "C:\\scans\\cookbook 1791",
            "--model",
            "gpt-5.6-terra",
            "--beside-input",
            "--reasoning-effort",
            "high",
            "--formats",
            "md,sqlite",
        ]
        lines = command_display(words, width=50, windows=True)

        assert lines == [
            '  tool run --input "C:\\scans\\cookbook 1791"',
            "    --model gpt-5.6-terra --beside-input",
            "    --reasoning-effort high --formats md,sqlite",
        ]

    def test_display_of_a_short_command_is_one_line(self) -> None:
        assert command_display(["tool", "ui"], windows=False) == ["  tool ui"]


class TestClipboard:
    def test_copy_runs_the_program_with_encoded_text(self) -> None:
        calls: list[tuple[list[str], bytes]] = []

        def runner(argv: Sequence[str], data: bytes) -> None:
            calls.append((list(argv), data))

        assert copy_to_clipboard("tool run", runner=runner, platform="darwin")
        assert copy_to_clipboard("é", runner=runner, platform="win32")
        assert calls == [(["pbcopy"], b"tool run"), (["clip"], "é".encode("utf-16"))]

    def test_copy_never_raises(self) -> None:
        def failing(argv: Sequence[str], data: bytes) -> None:
            raise OSError("no clipboard")

        assert not copy_to_clipboard("x", runner=failing, platform="win32")

    def test_unknown_platform_runs_nothing(self) -> None:
        def unexpected(argv: Sequence[str], data: bytes) -> None:
            raise AssertionError("runner called")

        assert not copy_to_clipboard("x", runner=unexpected, platform="linux")
