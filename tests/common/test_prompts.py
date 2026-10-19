"""Tests for common/prompts.py: questionary prompts driven through a pipe."""

from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from autoexcerpter.common.prompts import QuestionaryPrompter
from autoexcerpter.common.wizard import BACK, Choice

DOWN = "\x1b[B"
UP = "\x1b[A"
ENTER = "\r"
BACKSPACE = "\x7f"

CHOICES = [Choice("a", "Alpha"), Choice(None, "Nothing"), Choice("c", "Gamma")]


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    with create_pipe_input() as pipe_input:
        yield pipe_input


@pytest.fixture
def screen() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def prompter(pipe: PipeInput, screen: io.StringIO) -> QuestionaryPrompter:
    return QuestionaryPrompter(input=pipe, output=DummyOutput(), stream=screen)


class TestSelect:
    def test_enter_takes_the_preselected_value(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text(ENTER)
        assert prompter.select("Pick", CHOICES, default=None) is None

    def test_arrow_keys_move_from_the_first_entry(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text(DOWN + DOWN + ENTER)
        assert prompter.select("Pick", CHOICES) == "c"

    def test_back_is_the_last_entry(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text(DOWN * 3 + ENTER)
        assert prompter.select("Pick", CHOICES, default="a", marker="(s)") is BACK

    def test_keyboard_interrupt_propagates(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text("\x03")
        with pytest.raises(KeyboardInterrupt):
            prompter.select("Pick", CHOICES)


class TestCheckbox:
    def test_space_toggles_and_checked_values_stay(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text(" " + ENTER)
        assert prompter.checkbox("Formats", CHOICES, checked=["c"]) == ["a", "c"]

    def test_checking_back_goes_back(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text(UP + " " + ENTER)
        assert prompter.checkbox("Formats", CHOICES, checked=["a"]) is BACK


class TestText:
    def test_less_than_alone_means_back(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text("<" + ENTER)
        assert prompter.text("Name", validate=lambda text: "never") is BACK

    def test_validator_re_asks(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text("x" + ENTER + BACKSPACE + "y" + ENTER)
        answer = prompter.text(
            "Name", validate=lambda text: "not x" if text == "x" else None
        )
        assert answer == "y"

    def test_enter_keeps_the_default(
        self, pipe: PipeInput, prompter: QuestionaryPrompter
    ) -> None:
        pipe.send_text(ENTER)
        assert prompter.text("Name", default="kept", marker="(s)") == "kept"


class TestPath:
    def test_path_answer(self, pipe: PipeInput, prompter: QuestionaryPrompter) -> None:
        pipe.send_text("notes.txt" + ENTER)
        assert prompter.path("Input") == "notes.txt"

    def test_path_back(self, pipe: PipeInput, prompter: QuestionaryPrompter) -> None:
        pipe.send_text("<" + ENTER)
        assert prompter.path("Input", only_directories=True) is BACK


def test_show_writes_to_the_stream(
    prompter: QuestionaryPrompter, screen: io.StringIO
) -> None:
    prompter.show("Review\n  input  a.pdf")
    assert screen.getvalue() == "Review\n  input  a.pdf\n"
