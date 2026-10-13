"""What a guided session shares: settings, tasks, discovered items, the spec.

Every wizard task (a new run, batch check and cancel, render, migrate) is a
:class:`TaskFlow`.
:func:`build_wizard` puts the task step first, so the user can switch
tasks, then the task's own steps. The run task's steps and its review read
the answers through :func:`resolution`, which resolves them like
command-line flags over the settings defaults. :class:`Inventory` holds what
the input step found.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import fitz

from autoexcerpter import ext
from autoexcerpter import run as core
from autoexcerpter.common.options import OptionTable, Source
from autoexcerpter.common.selection import ItemSelection, select_items
from autoexcerpter.common.wizard import (
    BACK,
    DONE,
    Choice,
    Nav,
    Prompter,
    Review,
    State,
    Step,
    StepContext,
    Wizard,
)
from autoexcerpter.llm.capabilities import detect_capabilities
from autoexcerpter.llm.options import build_invoke_kwargs
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.settings import Settings, SettingsError, resolve
from autoexcerpter.spec import OPTIONS, ModelSpec, Resolution

if TYPE_CHECKING:
    from rich.console import Console

__all__ = [
    "INVENTORY",
    "TASKS",
    "TASK_STEP",
    "TITLE",
    "Inventory",
    "RunRequest",
    "Session",
    "Support",
    "TaskFlow",
    "TaskSwitch",
    "absolute",
    "build_wizard",
    "can_reset",
    "command_words",
    "current_model",
    "has_settings_default",
    "inventory_of",
    "new_state",
    "plural",
    "resolution",
    "selection_of",
    "source_of",
    "support_of",
    "take_inventory",
    "task_step",
]

logger = logging.getLogger(__name__)

TITLE: Final = "AutoExcerpter"
INVENTORY: Final = "inventory"
TASK_STEP: Final = "task"

TASKS: Mapping[str, str] = MappingProxyType(
    {
        "run": "new run",
        "batch-check": "check batches",
        "batch-cancel": "cancel batches",
        "render": "render files again",
        "migrate": "import version 1.0 outputs (migrate)",
    }
)
"""The wizard tasks in menu order, by name, with their titles."""


@dataclass(frozen=True)
class Session:
    """The settings of a guided session and where they came from.

    ``settings_path`` is the ``--settings`` file the ui was started with;
    ``environ`` is where the review looks up API key variables; ``args``
    holds the parsed ``ui`` arguments, which the review's plan passes to the
    extension's batch consent.
    """

    settings: Settings
    settings_path: Path | None = None
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    args: argparse.Namespace | None = None


def new_state(session: Session, table: OptionTable = OPTIONS) -> State:
    """Return an empty answer state over *table* and its settings defaults."""
    defaults = {
        name: value
        for name, value in session.settings.defaults.items()
        if name in table and table[name].settable
    }
    return State(table, defaults)


def absolute(path: Path | str) -> Path:
    """Expand ``~`` and make *path* absolute against the current directory."""
    return Path(os.path.abspath(os.path.expanduser(str(path).strip())))


def command_words(
    program: Sequence[str],
    argv: Sequence[str],
    session: Session,
    table: OptionTable,
) -> list[str]:
    """Return the full command: *program*, *argv* and the settings file.

    ``--settings`` is added when the ui was started with one and *table*
    has the option.
    """
    words = [*program, *argv]
    if session.settings_path is not None and "settings" in table:
        words += ["--settings", str(absolute(session.settings_path))]
    return words


class TaskSwitch(Exception):
    """The user chose another task on the task step."""

    def __init__(self, task: str) -> None:
        super().__init__(task)
        self.task = task


def task_step(current: str) -> Step:
    """Return the task step: DONE for *current*, TaskSwitch for another task."""

    def ask(ctx: StepContext) -> Nav:
        answer = ctx.prompter.select(
            "Task:",
            [Choice(name, title) for name, title in TASKS.items()],
            default=current,
        )
        if answer is BACK:
            return BACK
        if answer != current:
            raise TaskSwitch(answer)
        return DONE

    return Step(TASK_STEP, "Task", ask=ask)


class _TaskWizard(Wizard):
    """A wizard whose task step counts as answered on its first visit.

    It serves a task the user chose on an earlier wizard's task step: the
    first pass skips the task step without showing it, and Back from the
    task's first step returns to it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_chosen = True

    def _ask(self, step: Step) -> Nav:
        if self._task_chosen and step.name == TASK_STEP:
            self._task_chosen = False
            self.visited.add(step.name)
            return DONE
        return super()._ask(step)


@dataclass(frozen=True)
class RunRequest:
    """What a task runs with after the wizard: answers, console and arguments.

    ``args`` holds the parsed ``ui`` arguments for the extension.
    """

    session: Session
    answers: Mapping[str, Any]
    console: Console
    args: argparse.Namespace


@dataclass(frozen=True)
class TaskFlow:
    """One wizard task: its steps, review, command and run.

    ``program`` holds the leading command words, such as
    ``("autoexcerpter", "render")``; ``steps`` returns the task's steps
    without the task step; ``command`` returns the full equivalent command,
    program first; ``run`` returns the exit code.
    """

    name: str
    title: str
    program: tuple[str, ...]
    table: OptionTable
    steps: Callable[[Session], Sequence[Step]]
    review: Callable[[State, Session], Review]
    command: Callable[[State, Session], list[str]]
    run: Callable[[RunRequest], int]


def build_wizard(
    flow: TaskFlow,
    session: Session,
    prompter: Prompter,
    *,
    width: int = 80,
    chosen: bool = False,
) -> Wizard:
    """Return the wizard of *flow*: the task step, then the task's steps.

    The extension steps follow only for the run options. *chosen* means the
    user chose the task on an earlier wizard's task step: the wizard then
    starts at the task's first step, and Back from it returns to the task
    step.
    """
    steps = [task_step(flow.name), *flow.steps(session)]
    if flow.table is OPTIONS:
        steps += ext.wizard_steps()
    wizard = _TaskWizard if chosen else Wizard
    return wizard(
        new_state(session, flow.table),
        prompter,
        steps,
        title=TITLE,
        task=flow.title,
        review=lambda state: flow.review(state, session),
        command=lambda state: flow.command(state, session),
        width=width,
    )


def plural(count: int, word: str, many: str | None = None) -> str:
    """Return ``1 page`` or ``2 pages``."""
    return f"{count} {word if count == 1 else (many or word + 's')}"


def resolution(state: State, session: Session) -> Resolution:
    """Resolve the answers into the effective spec with each value's source.

    Raises:
        SpecError: The answers do not form a valid spec.
        SettingsError: A named endpoint is not in the settings.
    """
    flags: dict[str, Any] = state.answers
    if state.value("input") is None:
        flags["input"] = Path()
    return resolve(session.settings, flags)


def current_model(state: State, session: Session, phase: str) -> ModelSpec | None:
    """Return the effective model of *phase*, or None when unresolvable."""
    try:
        return resolution(state, session).spec.model(phase)
    except ValueError:
        return None


def source_of(state: State, session: Session, target: str) -> Source | None:
    """Return where the value of spec field *target* comes from."""
    try:
        return resolution(state, session).sources.get(target)
    except ValueError:
        return None


_REASONING_KEYS = ("reasoning", "thinking", "thinking_config", "output_config")


@dataclass(frozen=True)
class Support:
    """Which model options the core sends for one phase's model.

    ``schema`` is whether API-enforced schemas are available: the
    endpoint's declaration for a custom endpoint, else the capability
    table.
    """

    reasoning: bool
    verbosity: bool
    temperature: bool
    top_p: bool
    service_tier: bool
    schema: bool

    @property
    def automatic_format(self) -> str:
        """The response format an unset ``--response-format`` takes."""
        return "schema" if self.schema else "json"


def support_of(model: ModelSpec, settings: Settings) -> Support:
    """Return what the core would send for *model*, probed with its options."""
    probe = build_invoke_kwargs(
        model.provider, model.model, {"reasoning": {"effort": "high"}}
    )
    sent = build_invoke_kwargs(
        model.provider,
        model.model,
        {
            "reasoning": {"effort": model.reasoning_effort},
            "text": {"verbosity": model.verbosity},
            "temperature": model.temperature,
            "top_p": 0.5,
        },
    )
    if model.provider == "custom" and model.endpoint is not None:
        try:
            schema = settings.endpoint(model.endpoint).supports_schema
        except SettingsError:
            schema = False
    else:
        schema = detect_capabilities(model.model).supports_structured_output
    return Support(
        reasoning=any(key in probe for key in _REASONING_KEYS),
        verbosity="text" in sent,
        temperature="temperature" in sent,
        top_p="top_p" in sent,
        service_tier=model.provider == "openai",
        schema=schema,
    )


def has_settings_default(state: State, name: str) -> bool:
    """Whether the settings defaults give option *name* a value."""
    current: str | None = name
    while current is not None:
        if current in state.defaults:
            return True
        current = OPTIONS[current].fallback
    return False


def can_reset(state: State, name: str) -> bool:
    """Whether *name* may be reset to its unset code default.

    An unset value has no command-line spelling, so it cannot override a
    settings default; the wizard offers the reset only without one.
    """
    return not has_settings_default(state, name)


def _page_count(path: Path) -> int | None:
    try:
        with fitz.open(path) as document:
            return int(document.page_count)
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF has no count
        logger.debug("Could not count the pages of %s: %s", path, exc)
        return None


@dataclass(frozen=True)
class Inventory:
    """The items found at an input path, with page or image counts."""

    path: Path
    items: tuple[ItemSpec, ...]
    counts: tuple[int | None, ...]

    @property
    def names(self) -> list[str]:
        """The item names that selection expressions match against."""
        return [item.path.name for item in self.items]

    def describe(self) -> str:
        """Return ``1 PDF, 212 pages`` or a summary of several kinds."""
        parts: list[str] = []
        for kind, label, unit in (
            ("pdf", "PDF", "page"),
            ("image_folder", "image folder", "image"),
        ):
            counts = [
                count
                for item, count in zip(self.items, self.counts, strict=True)
                if item.kind == kind
            ]
            if not counts:
                continue
            total = sum(count or 0 for count in counts)
            unknown = " or more" if any(count is None for count in counts) else ""
            parts.append(
                f"{plural(len(counts), label)}, {plural(total, unit)}{unknown}"
            )
        return "; ".join(parts) if parts else "nothing to process"

    def label(self, index: int) -> str:
        """Return one numbered list line for the item at 0-based *index*."""
        item, count = self.items[index], self.counts[index]
        if item.kind == "pdf":
            detail = "PDF" if count is None else f"PDF, {plural(count, 'page')}"
        else:
            detail = "image folder"
            if count is not None:
                detail += f", {plural(count, 'image')}"
        return f"{index + 1:>4}  {item.path.name}  ({detail})"


def take_inventory(path: Path) -> Inventory:
    """Discover the items at *path* and count their pages or images."""
    items = tuple(core.discover(path))
    counts = tuple(
        _page_count(item.path) if item.kind == "pdf" else item.image_count
        for item in items
    )
    return Inventory(path, items, counts)


def inventory_of(state: State, path: Path) -> Inventory:
    """Return the stored inventory of *path*, taking it when missing."""
    stored = state.data.get(INVENTORY)
    if isinstance(stored, Inventory) and stored.path == path:
        return stored
    inventory = take_inventory(path)
    state.data[INVENTORY] = inventory
    return inventory


def selection_of(inventory: Inventory, expr: str) -> ItemSelection:
    """Return what the selection expression *expr* picks."""
    return select_items(inventory.names, expr)
