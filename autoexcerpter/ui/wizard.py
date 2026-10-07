"""AutoExcerpter's wizard: the steps of a guided run on the shared framework.

Steps in order: input and item selection, output, summary, formats,
context, transcription model, summary model, images and run options, then
the extension steps and the review. Each :class:`Step` names the options it
may set. Answers act like command-line flags over the settings defaults; a
reset to an unset code default is offered only where no settings default
applies, since the equivalent command could not express it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from autoexcerpter import ext
from autoexcerpter.common.options import OptionError, Source
from autoexcerpter.common.selection import is_numeric_selection
from autoexcerpter.common.wizard import (
    BACK,
    DONE,
    NO_DEFAULT,
    Choice,
    Nav,
    Prompter,
    State,
    Step,
    StepContext,
    Validator,
    Wizard,
)
from autoexcerpter.providers import implied_provider, split_provider_prefix
from autoexcerpter.spec import (
    CONTEXT_NONE,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    OPTIONS,
    OUTPUT_FORMATS,
    REASONING_EFFORTS,
    SERVICE_TIERS,
    VERBOSITIES,
    ImageSpec,
    ModelSpec,
    model_suggestions,
    recommended_images,
)
from autoexcerpter.ui.review import (
    IMAGE_OPTIONS,
    IMAGE_TEXT_INDEX,
    build_review,
    describe_images,
    equivalent_command,
    image_family,
)
from autoexcerpter.ui.session import (
    INVENTORY,
    Inventory,
    Session,
    absolute,
    can_reset,
    current_model,
    has_settings_default,
    inventory_of,
    new_state,
    plural,
    selection_of,
    source_of,
    support_of,
    take_inventory,
)

__all__ = [
    "IMAGE_OPTIONS",
    "MODEL_FIELDS",
    "OTHER_MODEL",
    "STEP_NAMES",
    "TASK",
    "TITLE",
    "build_steps",
    "build_wizard",
    "phase_options",
    "provider_key",
]

TITLE: Final = "AutoExcerpter"
TASK: Final = "new run"

MODEL_FIELDS: Final = (
    "provider",
    "model",
    "endpoint",
    "reasoning_effort",
    "verbosity",
    "max_output_tokens",
    "temperature",
    "top_p",
    "service_tier",
    "response_format",
)
TUNING_FIELDS: Final = (
    "verbosity",
    "max_output_tokens",
    "temperature",
    "top_p",
    "service_tier",
)
SAME_MODEL_FIELDS: Final = ("reasoning_effort", "service_tier", "response_format")
RUN_FLAGS: Final = (
    "openalex",
    "keep_working_files",
    "force",
    "retranscribe",
    "dry_run",
)
SUMMARY_FORMATS: Final = ("summary-md", "summary-docx")

OTHER_MODEL: Final = "<other>"
ENDPOINT_PREFIX: Final = "endpoint:"
LIST_LIMIT: Final = 20

STEP_NAMES: Final = (
    "input",
    "output",
    "summary",
    "formats",
    "context",
    "transcription_model",
    "summary_model",
    "images",
    "run",
)

_PROVIDER_TITLES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "openrouter": "OpenRouter",
}
_PHASE_TITLES = {"transcription": "Transcription", "summary": "Summary"}

Question = Callable[[], Nav | None]


def phase_options(phase: str) -> tuple[str, ...]:
    """Return the per-phase model option names of *phase*."""
    return tuple(f"{phase}_{field}" for field in MODEL_FIELDS)


def provider_key(provider: str, endpoint: str | None) -> str:
    """Return the provider menu value: a provider name or ``endpoint:NAME``."""
    if provider == "custom" and endpoint is not None:
        return ENDPOINT_PREFIX + endpoint
    return provider


def _split_key(key: str) -> tuple[str, str | None]:
    if key.startswith(ENDPOINT_PREFIX):
        return "custom", key.removeprefix(ENDPOINT_PREFIX)
    return key, None


def _marker(state: State, source: Source | None) -> str | None:
    return state.settings_marker if source is Source.SETTINGS else None


def _existing_input(text: str) -> str | None:
    if not text.strip():
        return "enter a PDF, an image folder or a folder holding them"
    if not absolute(text).exists():
        return "no such file or folder"
    return None


def _preview(ctx: StepContext, inventory: Inventory, indices: Sequence[int]) -> None:
    total = len(inventory.items)
    ctx.prompter.show(f"  selected {len(indices)} of {plural(total, 'item')}:")
    for index in list(indices)[:LIST_LIMIT]:
        ctx.prompter.show("  " + inventory.label(index))
    if len(indices) > LIST_LIMIT:
        ctx.prompter.show(f"        ... and {len(indices) - LIST_LIMIT} more")


def _selection_validator(inventory: Inventory, by_number: bool) -> Validator:
    def validate(text: str) -> str | None:
        expr = text.strip()
        if not expr:
            return "enter a selection"
        if by_number != is_numeric_selection(expr):
            if by_number:
                return "enter numbers or ranges, such as 1,3-5"
            return "a name search needs letters; choose 'By number or range'"
        if not selection_of(inventory, expr).indices:
            return f"no item matches {expr!r}"
        return None

    return validate


class _InputQuestions:
    """The questions of the input step: path, selection mode, expression."""

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.how_answer: str | None = None

    def questions(self) -> list[Question]:
        return [self.path, self.how, self.expression]

    def path(self) -> Nav:
        state, prompter = self.state, self.ctx.prompter
        while True:
            current = state.value("input")
            answer = prompter.path(
                "Input (PDF, image folder, or a folder of them):",
                default="" if current is None else str(current),
                validate=_existing_input,
                marker=state.marker("input"),
            )
            if answer is BACK:
                return BACK
            location = absolute(answer)
            inventory = take_inventory(location)
            state.data[INVENTORY] = inventory
            if not inventory.items:
                prompter.show(f"  found no PDF or image folder in {location}")
                continue
            state.set("input", location)
            prompter.show(f"  found {inventory.describe()}")
            return DONE

    def _list_items(self, inventory: Inventory) -> None:
        count = len(inventory.items)
        for index in range(min(count, LIST_LIMIT)):
            self.ctx.prompter.show("  " + inventory.label(index))
        if count > LIST_LIMIT:
            self.ctx.prompter.show(f"        ... and {count - LIST_LIMIT} more")

    def _current_how(self) -> str:
        select = self.state.value("select")
        if not self.state.value("all") and select:
            return "numbers" if is_numeric_selection(select) else "name"
        return "all"

    def how(self) -> Nav | None:
        state = self.state
        inventory = inventory_of(state, state.value("input"))
        count = len(inventory.items)
        if count < 2:
            state.clear("select")
            state.clear("all")
            return None
        self._list_items(inventory)
        current = self._current_how()
        option = "all" if current == "all" else "select"
        answer = self.ctx.prompter.select(
            "Which items?",
            [
                Choice("all", f"All {count} items"),
                Choice("numbers", "By number or range", "such as 1,3-5"),
                Choice("name", "By name search"),
            ],
            default=current,
            marker=state.marker(option),
        )
        if answer is BACK:
            return BACK
        self.how_answer = answer
        if answer == "all":
            state.set("all", True)
            _preview(self.ctx, inventory, range(count))
        return DONE

    def expression(self) -> Nav | None:
        if self.how_answer not in ("numbers", "name"):
            return None
        state, prompter = self.state, self.ctx.prompter
        by_number = self.how_answer == "numbers"
        inventory = inventory_of(state, state.value("input"))
        current = state.value("select") or ""
        if current and is_numeric_selection(current) != by_number:
            current = ""
        answer = prompter.text(
            "Item numbers or ranges:" if by_number else "Name contains:",
            default=current,
            validate=_selection_validator(inventory, by_number),
            marker=state.marker("select") if current else None,
        )
        if answer is BACK:
            return BACK
        expr = answer.strip()
        state.set("select", expr)
        picked = selection_of(inventory, expr)
        if picked.unmatched:
            prompter.show(
                f"  ignored {', '.join(picked.unmatched)} "
                f"(valid: 1-{len(inventory.items)})"
            )
        _preview(self.ctx, inventory, picked.indices)
        return DONE


def _ask_input(ctx: StepContext) -> Nav:
    return StepContext.sequence(_InputQuestions(ctx).questions())


def _folder(text: str) -> str | None:
    if not text.strip():
        return "enter a folder"
    if absolute(text).is_file():
        return "this is a file; enter a folder"
    return None


def _ask_output(ctx: StepContext) -> Nav:
    state = ctx.state
    chosen: dict[str, str] = {}

    def where() -> Nav:
        folder = state.value("output") is not None
        answer = ctx.prompter.select(
            "Where should the outputs go?",
            [
                Choice("beside", "Beside each input"),
                Choice("folder", "Into one folder", "created when missing"),
            ],
            default="folder" if folder else "beside",
            marker=state.marker("output") if folder else None,
        )
        if answer is BACK:
            return BACK
        chosen["where"] = answer
        if answer == "folder":
            state.clear("beside_input")
        elif has_settings_default(state, "output"):
            state.set("beside_input", True)
        else:
            state.clear("output")
            state.clear("beside_input")
        return DONE

    def folder() -> Nav | None:
        if chosen.get("where") != "folder":
            return None
        current = state.value("output")
        answer = ctx.prompter.path(
            "Output folder:",
            default="" if current is None else str(current),
            validate=_folder,
            only_directories=True,
            marker=state.marker("output"),
        )
        if answer is BACK:
            return BACK
        state.set("output", absolute(answer))
        return DONE

    return StepContext.sequence([where, folder])


def _ask_summary(ctx: StepContext) -> Nav:
    return ctx.select_option(
        "summarize",
        "Summarize after transcribing?",
        [
            Choice(True, "Yes, transcribe and summarize"),
            Choice(False, "No, transcribe only"),
        ],
    )


_FORMAT_TITLES = {
    "summary-md": "Summary as Markdown (X_summary.md)",
    "summary-docx": "Summary as Word (X_summary.docx)",
    "sqlite": "SQLite database (autoexcerpter.sqlite)",
}


def _ask_formats(ctx: StepContext) -> Nav:
    state = ctx.state

    def transcription_format() -> Nav:
        return ctx.select_option(
            "transcription_format",
            "Transcription file format:",
            [Choice("md", "Markdown (.md)"), Choice("txt", "Plain text (.txt)")],
        )

    def formats() -> Nav:
        summarizing = bool(state.value("summarize"))
        offered = [
            Choice(name, _FORMAT_TITLES[name])
            for name in OUTPUT_FORMATS
            if summarizing or name not in SUMMARY_FORMATS
        ]
        current = tuple(state.value("formats") or ())
        while True:
            answer = ctx.prompter.checkbox(
                "Output formats:",
                offered,
                checked=current,
                marker=state.marker("formats"),
            )
            if answer is BACK:
                return BACK
            if summarizing and not answer:
                ctx.prompter.show(
                    "  summaries need summary-md, summary-docx or sqlite; "
                    "select at least one"
                )
                continue
            shown = {choice.value for choice in offered}
            hidden = [name for name in current if name not in shown]
            state.set("formats", OPTIONS["formats"].convert([*hidden, *answer]))
            return DONE

    return StepContext.sequence([transcription_format, formats])


_CONTEXT_CHOICES = (
    Choice("auto", "Sidecar files (auto)", "X_summary_context.txt beside the input"),
    Choice("text", "Topics as text"),
    Choice("file", "A text file"),
    Choice("none", "No context"),
)
_FALLBACK_CHOICES = (
    Choice("none", "No fallback"),
    Choice("text", "Topics as text"),
    Choice("file", "A text file"),
)
_CONTEXT_KEYWORDS: dict[str, dict[str, str | None]] = {
    "context": {"auto": None, "none": CONTEXT_NONE},
    "fallback_context": {"none": None},
}


def _context_mode(name: str, value: str | None) -> str:
    for mode, keyword in _CONTEXT_KEYWORDS[name].items():
        if value == keyword:
            return mode
    if value is not None and Path(value).expanduser().is_file():
        return "file"
    return "text"


def _topics(name: str) -> Validator:
    def validate(text: str) -> str | None:
        if not text.strip():
            return "enter the topics"
        try:
            value = OPTIONS[name].convert(text.strip())
        except OptionError as exc:
            return str(exc)
        if value is None or value == CONTEXT_NONE:
            return "choose the menu entry instead of typing a keyword"
        return None

    return validate


def _text_file(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    if not absolute(text).is_file():
        return "no such file"
    return None


def _context_questions(
    ctx: StepContext,
    name: str,
    message: str,
    choices: Sequence[Choice[str]],
    applies: Callable[[], bool],
) -> list[Question]:
    state = ctx.state
    chosen: dict[str, str] = {}

    def mode() -> Nav | None:
        if not applies():
            return None
        answer = ctx.prompter.select(
            message,
            choices,
            default=_context_mode(name, state.value(name)),
            marker=state.marker(name),
        )
        if answer is BACK:
            return BACK
        chosen["mode"] = answer
        keywords = _CONTEXT_KEYWORDS[name]
        if answer in keywords:
            state.set(name, keywords[answer])
        if name == "context" and answer != "auto":
            state.clear("fallback_context")
        return DONE

    def value() -> Nav | None:
        current_mode = chosen.get("mode")
        if not applies() or current_mode not in ("text", "file"):
            return None
        current = state.value(name)
        if _context_mode(name, current) != current_mode:
            current = None
        default = "" if current is None else str(current)
        marker = state.marker(name) if current is not None else None
        answer: str | Nav
        if current_mode == "text":
            answer = ctx.prompter.text(
                "Topics:", default=default, validate=_topics(name), marker=marker
            )
            if answer is BACK:
                return BACK
            state.set(name, answer.strip())
        else:
            answer = ctx.prompter.path(
                "Context file:", default=default, validate=_text_file, marker=marker
            )
            if answer is BACK:
                return BACK
            state.set(name, str(absolute(answer)))
        return DONE

    return [mode, value]


def _ask_context(ctx: StepContext) -> Nav:
    state = ctx.state
    questions = [
        *_context_questions(
            ctx, "context", "Summary context:", _CONTEXT_CHOICES, lambda: True
        ),
        *_context_questions(
            ctx,
            "fallback_context",
            "Context for items without a sidecar:",
            _FALLBACK_CHOICES,
            lambda: state.value("context") is None,
        ),
    ]
    return StepContext.sequence(questions)


def _provider_choices(session: Session) -> list[Choice[str]]:
    choices = [Choice(name, title) for name, title in _PROVIDER_TITLES.items()]
    for name, endpoint in session.settings.endpoints.items():
        choices.append(
            Choice(provider_key("custom", name), f"Endpoint {name}", endpoint.base_url)
        )
    return choices


def _model_name(provider: str) -> Validator:
    def validate(text: str) -> str | None:
        if not text.strip():
            return "enter a model name"
        prefixed, _name = split_provider_prefix(text.strip())
        if provider == "custom" and prefixed is not None:
            return "an endpoint model name takes no provider prefix"
        return None

    return validate


def _compatible(provider: str, model: str) -> bool:
    implied = implied_provider(model)
    return implied is None or implied == provider


def _commit_model(
    ctx: StepContext,
    session: Session,
    phase: str,
    key: str,
    text: str,
) -> None:
    """Answer the model of *phase*; the provider a model name implies wins."""
    state = ctx.state
    provider, endpoint = _split_key(key)
    prefixed, name = split_provider_prefix(text.strip())
    if provider != "custom":
        implied = prefixed or implied_provider(name)
        if implied is not None and implied != provider:
            ctx.prompter.show(f"  provider {implied}, as the model name implies")
            provider = implied
    state.set(f"{phase}_model", name)
    state.clear(f"{phase}_provider")
    state.clear(f"{phase}_endpoint")
    model = current_model(state, session, phase)
    if model is None or (model.provider, model.endpoint) != (provider, endpoint):
        state.set(f"{phase}_provider", provider)
        state.set(f"{phase}_endpoint", endpoint)


def _tuning_note(state: State, session: Session, phase: str) -> str:
    """Describe the tuning values an answer-free phase takes."""
    kept = {
        name: value
        for name, value in state.answers.items()
        if name not in {f"{phase}_{field}" for field in TUNING_FIELDS}
    }
    trial = State(state.table, state.defaults, kept)
    model = current_model(state, session, phase)
    if model is None:
        return ""
    support = support_of(model, session.settings)
    marker = state.settings_marker

    def shown(field: str, text: str) -> str:
        name = f"{phase}_{field}"
        return f"{text} {marker}" if trial.from_settings(name) else text

    def value(field: str) -> Any:
        return trial.value(f"{phase}_{field}")

    parts = []
    if support.verbosity:
        parts.append(shown("verbosity", f"verbosity {value('verbosity')}"))
    limit = value("max_output_tokens")
    parts.append(
        shown(
            "max_output_tokens",
            "model's output limit" if limit is None else f"max {limit} tokens",
        )
    )
    if support.temperature:
        parts.append(shown("temperature", f"temperature {value('temperature')}"))
    if support.top_p and value("top_p") is not None:
        parts.append(shown("top_p", f"top p {value('top_p')}"))
    if support.service_tier:
        parts.append(shown("service_tier", f"tier {value('service_tier')}"))
    return ", ".join(parts)


class _ModelQuestions:
    """The questions that choose and tune the model of one phase."""

    def __init__(
        self,
        ctx: StepContext,
        session: Session,
        phase: str,
        skip: Callable[[], bool],
    ) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.session = session
        self.phase = phase
        self.skip = skip
        self.title = _PHASE_TITLES[phase]
        self.key = ""
        self.tuning: str | None = None

    def questions(self) -> list[Question]:
        return [
            self.provider,
            self.name,
            self.reasoning,
            self.response_format,
            self.tune,
            self.verbosity,
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.service_tier,
        ]

    def _option(self, field: str) -> str:
        return f"{self.phase}_{field}"

    def _model(self) -> ModelSpec | None:
        return current_model(self.state, self.session, self.phase)

    def _commit(self, text: str) -> None:
        _commit_model(self.ctx, self.session, self.phase, self.key, text)

    def provider(self) -> Nav | None:
        if self.skip():
            return None
        current = self._model()
        key: Any = NO_DEFAULT
        if current is not None:
            key = provider_key(current.provider, current.endpoint)
        source = source_of(self.state, self.session, f"{self.phase}_model.provider")
        answer = self.ctx.prompter.select(
            f"{self.title} provider:",
            _provider_choices(self.session),
            default=key,
            marker=_marker(self.state, source),
        )
        if answer is BACK:
            return BACK
        self.key = answer
        return DONE

    def name(self) -> Nav | None:
        if self.skip():
            return None
        chosen_provider, endpoint = _split_key(self.key)
        current = self._model()
        source = source_of(self.state, self.session, f"{self.phase}_model.model")
        if chosen_provider == "custom":
            same = current is not None and provider_key(
                current.provider, current.endpoint
            ) == provider_key(chosen_provider, endpoint)
            return self._endpoint_model(endpoint, current if same else None, source)
        return self._listed_model(chosen_provider, current, source)

    def _endpoint_model(
        self, endpoint: str | None, same: ModelSpec | None, source: Source | None
    ) -> Nav:
        answer = self.ctx.prompter.text(
            f"{self.title} model at endpoint {endpoint}:",
            default=same.model if same is not None else "",
            validate=_model_name("custom"),
            marker=_marker(self.state, source) if same is not None else None,
        )
        if answer is BACK:
            return BACK
        self._commit(answer)
        return DONE

    def _listed_model(
        self, provider: str, current: ModelSpec | None, source: Source | None
    ) -> Nav:
        names = list(model_suggestions(provider))
        if (
            current is not None
            and current.model not in names
            and _compatible(provider, current.model)
        ):
            names.insert(0, current.model)
        choices = [Choice(value, value) for value in names]
        choices.append(Choice(OTHER_MODEL, "Other (type a name)"))
        while True:
            default = current.model if current is not None else NO_DEFAULT
            answer = self.ctx.prompter.select(
                f"{self.title} model:",
                choices,
                default=default,
                marker=_marker(self.state, source),
            )
            if answer is BACK:
                return BACK
            if answer != OTHER_MODEL:
                self._commit(answer)
                return DONE
            typed = self.ctx.prompter.text(
                "Model name:", validate=_model_name(provider)
            )
            if typed is not BACK:
                self._commit(typed)
                return DONE

    def reasoning(self) -> Nav | None:
        current = self._model()
        if self.skip() or current is None:
            return None
        if not support_of(current, self.session.settings).reasoning:
            return None
        return self.ctx.select_option(
            self._option("reasoning_effort"),
            "Reasoning effort:",
            [Choice(value, value) for value in REASONING_EFFORTS],
        )

    def response_format(self) -> Nav | None:
        current = self._model()
        if self.skip() or current is None:
            return None
        support = support_of(current, self.session.settings)
        name = self._option("response_format")
        choices: list[Choice[str | None]] = []
        if can_reset(self.state, name):
            choices.append(
                Choice(None, "Automatic", f"{support.automatic_format} for this model")
            )
        if support.schema:
            choices.append(Choice("schema", "Schema (API-enforced)"))
        choices.append(Choice("json", "JSON (prompted, validated)"))
        choices.append(Choice("text", "Plain text"))
        return self.ctx.select_option(name, "Response format:", choices)

    def tune(self) -> Nav | None:
        if self.skip():
            return None
        state = self.state
        names = {self._option(field) for field in TUNING_FIELDS}
        customized = any(name in names for name in state.answers)
        note = _tuning_note(state, self.session, self.phase)
        answer = self.ctx.prompter.select(
            "Model parameters:",
            [
                Choice("recommended", "Recommended", note),
                Choice("customize", "Customize"),
            ],
            default="customize" if customized else "recommended",
        )
        if answer is BACK:
            return BACK
        self.tuning = answer
        if answer == "recommended":
            for name in names:
                state.clear(name)
        return DONE

    def _customizing(self) -> ModelSpec | None:
        if self.skip() or self.tuning != "customize":
            return None
        return self._model()

    def verbosity(self) -> Nav | None:
        current = self._customizing()
        if current is None or not support_of(current, self.session.settings).verbosity:
            return None
        return self.ctx.select_option(
            self._option("verbosity"),
            "Verbosity:",
            [Choice(value, value) for value in VERBOSITIES],
        )

    def max_tokens(self) -> Nav | None:
        if self._customizing() is None:
            return None
        name = self._option("max_output_tokens")
        if can_reset(self.state, name):
            return self.ctx.text_option(
                name, "Max output tokens (empty = the model's limit):", empty=None
            )
        return self.ctx.text_option(name, "Max output tokens:")

    def temperature(self) -> Nav | None:
        current = self._customizing()
        settings = self.session.settings
        if current is None or not support_of(current, settings).temperature:
            return None
        return self.ctx.text_option(
            self._option("temperature"), "Temperature (0.0 to 2.0):"
        )

    def top_p(self) -> Nav | None:
        current = self._customizing()
        if current is None or not support_of(current, self.session.settings).top_p:
            return None
        name = self._option("top_p")
        if can_reset(self.state, name):
            return self.ctx.text_option(
                name, "Top p (0.0 to 1.0, empty = not sent):", empty=None
            )
        return self.ctx.text_option(name, "Top p (0.0 to 1.0):")

    def service_tier(self) -> Nav | None:
        current = self._customizing()
        if current is None or current.provider != "openai":
            return None
        return self.ctx.select_option(
            self._option("service_tier"),
            "Service tier:",
            [Choice(value, value) for value in SERVICE_TIERS],
        )


def _summarizing(state: State) -> bool:
    return bool(state.value("summarize"))


def _same_model(first: ModelSpec, second: ModelSpec) -> bool:
    fields = ("provider", "endpoint", "model", *SAME_MODEL_FIELDS)
    return all(getattr(first, f) == getattr(second, f) for f in fields)


def _transcription_model_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        questions = _ModelQuestions(ctx, session, "transcription", lambda: False)
        return StepContext.sequence(questions.questions())

    return Step(
        "transcription_model",
        "Transcription model",
        ask=ask,
        sets=phase_options("transcription"),
    )


def _summary_model_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        state = ctx.state
        chosen: dict[str, bool] = {}

        def same() -> Nav:
            first = current_model(state, session, "transcription")
            second = current_model(state, session, "summary")
            identical = (
                first is not None and second is not None and _same_model(first, second)
            )
            note = f"{first.provider} / {first.model}" if first is not None else None
            answer = ctx.prompter.select(
                "Summary model:",
                [
                    Choice("same", "Same as transcription", note),
                    Choice("own", "Choose a summary model"),
                ],
                default="same" if identical else "own",
            )
            if answer is BACK:
                return BACK
            chosen["same"] = answer == "same"
            if answer == "same" and first is not None:
                _copy_model(ctx, session, first)
            return DONE

        questions = _ModelQuestions(
            ctx, session, "summary", lambda: chosen.get("same", True)
        )
        return StepContext.sequence([same, *questions.questions()])

    return Step(
        "summary_model",
        "Summary model",
        ask=ask,
        sets=phase_options("summary"),
        applies=_summarizing,
    )


def _copy_model(ctx: StepContext, session: Session, source: ModelSpec) -> None:
    """Give the summary phase the transcription model and its main options."""
    state = ctx.state
    for name in phase_options("summary"):
        state.clear(name)
    key = provider_key(source.provider, source.endpoint)
    _commit_model(ctx, session, "summary", key, source.model)
    for field in SAME_MODEL_FIELDS:
        value = getattr(source, field)
        name = f"summary_{field}"
        if value is None and not can_reset(state, name):
            continue
        state.set(name, value)


def _images_note(state: State, recommended: ImageSpec) -> str:
    kept = {k: v for k, v in state.answers.items() if k not in IMAGE_OPTIONS}
    trial = State(state.table, state.defaults, kept)
    effective = ImageSpec(*(trial.value(name) for name in IMAGE_OPTIONS))
    texts = describe_images(effective.over(recommended))
    flagged = {
        IMAGE_TEXT_INDEX[name] for name in IMAGE_OPTIONS if trial.from_settings(name)
    }
    return ", ".join(
        f"{text} {state.settings_marker}" if index in flagged else text
        for index, text in enumerate(texts)
    )


class _ImageQuestions:
    """The questions of the images step: recommended or customized values."""

    def __init__(self, ctx: StepContext, session: Session) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.session = session
        self.mode_answer: str | None = None

    def questions(self) -> list[Question]:
        return [
            self.mode,
            self.dpi,
            self.detail,
            self.image_format,
            self.quality,
            self.grayscale,
        ]

    def _family(self) -> str:
        return image_family(current_model(self.state, self.session, "transcription"))

    def _recommended(self) -> ImageSpec:
        return recommended_images(self._family())

    def _customizing(self) -> bool:
        return self.mode_answer == "customize"

    def _reset_choice(self, name: str, text: str) -> list[Choice[Any]]:
        if can_reset(self.state, name):
            return [Choice(None, f"Recommended ({text})")]
        return []

    def mode(self) -> Nav:
        state = self.state
        family = self._family()
        customized = any(name in state.answers for name in IMAGE_OPTIONS)
        answer = self.ctx.prompter.select(
            "Page images:",
            [
                Choice(
                    "recommended",
                    f"Recommended for {family} models",
                    _images_note(state, self._recommended()),
                ),
                Choice("customize", "Customize"),
            ],
            default="customize" if customized else "recommended",
        )
        if answer is BACK:
            return BACK
        self.mode_answer = answer
        if answer == "recommended":
            for name in IMAGE_OPTIONS:
                state.clear(name)
        return DONE

    def dpi(self) -> Nav | None:
        if not self._customizing():
            return None
        if can_reset(self.state, "dpi"):
            return self.ctx.text_option(
                "dpi",
                f"DPI (native or a number, empty = {self._recommended().dpi}):",
                empty=None,
            )
        return self.ctx.text_option("dpi", "DPI (native or a number):")

    def detail(self) -> Nav | None:
        if not self._customizing():
            return None
        choices = [
            *self._reset_choice("image_detail", str(self._recommended().detail)),
            *(Choice(value, value) for value in IMAGE_DETAILS),
        ]
        return self.ctx.select_option("image_detail", "Image detail:", choices)

    def image_format(self) -> Nav | None:
        if not self._customizing():
            return None
        choices = [
            *self._reset_choice("image_format", str(self._recommended().format)),
            *(Choice(value, value.upper()) for value in IMAGE_FORMATS),
        ]
        return self.ctx.select_option("image_format", "Image format:", choices)

    def quality(self) -> Nav | None:
        if not self._customizing():
            return None
        recommended = self._recommended()
        fmt = self.state.value("image_format") or recommended.format
        if fmt != "jpeg":
            return None
        if can_reset(self.state, "jpeg_quality"):
            return self.ctx.text_option(
                "jpeg_quality",
                f"JPEG quality (1 to 100, empty = {recommended.jpeg_quality}):",
                empty=None,
            )
        return self.ctx.text_option("jpeg_quality", "JPEG quality (1 to 100):")

    def grayscale(self) -> Nav | None:
        if not self._customizing():
            return None
        default = "yes" if self._recommended().grayscale else "no"
        choices = [
            *self._reset_choice("grayscale", default),
            Choice(True, "Yes"),
            Choice(False, "No"),
        ]
        return self.ctx.select_option("grayscale", "Grayscale:", choices)


def _images_step(session: Session) -> Step:
    def ask(ctx: StepContext) -> Nav:
        return StepContext.sequence(_ImageQuestions(ctx, session).questions())

    return Step("images", "Images", ask=ask, sets=IMAGE_OPTIONS)


_RUN_FLAG_TITLES = {
    "openalex": "Look up citations in OpenAlex",
    "keep_working_files": "Keep working files after a forced run",
    "force": "Reprocess items whose outputs exist (--force)",
    "retranscribe": "Transcribe again instead of reusing logged pages",
    "dry_run": "Dry run: show the plan, no API calls or writes",
}


def _ask_run(ctx: StepContext) -> Nav:
    state = ctx.state

    def flags() -> Nav:
        names = [
            name
            for name in RUN_FLAGS
            if name != "openalex" or bool(state.value("summarize"))
        ]
        choices = [
            Choice(name, _RUN_FLAG_TITLES[name], state.marker(name)) for name in names
        ]
        answer = ctx.prompter.checkbox(
            "Run options:",
            choices,
            checked=[name for name in names if state.value(name)],
        )
        if answer is BACK:
            return BACK
        for name in names:
            state.set(name, name in answer)
        return DONE

    def concurrency() -> Nav:
        return ctx.text_option("concurrency", "Parallel page requests:")

    return StepContext.sequence([flags, concurrency])


def build_steps(session: Session) -> list[Step]:
    """Return the tool's steps in order, extension steps excluded."""
    return [
        Step("input", "Input", ask=_ask_input, sets=("input", "select", "all")),
        Step("output", "Output", ask=_ask_output, sets=("output", "beside_input")),
        Step("summary", "Summary", ask=_ask_summary, sets=("summarize",)),
        Step(
            "formats",
            "Formats",
            ask=_ask_formats,
            sets=("transcription_format", "formats"),
        ),
        Step(
            "context",
            "Context",
            ask=_ask_context,
            sets=("context", "fallback_context"),
            applies=_summarizing,
        ),
        _transcription_model_step(session),
        _summary_model_step(session),
        _images_step(session),
        Step("run", "Run options", ask=_ask_run, sets=(*RUN_FLAGS, "concurrency")),
    ]


def build_wizard(
    session: Session,
    prompter: Prompter,
    *,
    state: State | None = None,
    steps: Sequence[Step] | None = None,
    width: int = 80,
) -> Wizard:
    """Return the wizard of a guided run, extension steps appended."""
    return Wizard(
        state if state is not None else new_state(session),
        prompter,
        [
            *(steps if steps is not None else build_steps(session)),
            *ext.wizard_steps(),
        ],
        title=TITLE,
        task=TASK,
        review=lambda current: build_review(current, session),
        command=lambda current: equivalent_command(current, session),
        width=width,
    )
