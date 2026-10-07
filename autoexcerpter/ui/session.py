"""What a guided session shares: settings, discovered items, effective spec.

The wizard steps and the review screen both read the answers through
:func:`resolution`, which resolves them like command-line flags over the
settings defaults. :class:`Inventory` holds what the input step found.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from autoexcerpter import run as core
from autoexcerpter.common.options import Source
from autoexcerpter.common.selection import ItemSelection, select_items
from autoexcerpter.common.wizard import State
from autoexcerpter.llm.capabilities import detect_capabilities
from autoexcerpter.llm.options import build_invoke_kwargs
from autoexcerpter.pipeline.types import ItemSpec
from autoexcerpter.settings import Settings, SettingsError, resolve
from autoexcerpter.spec import OPTIONS, ModelSpec, Resolution

__all__ = [
    "INVENTORY",
    "Inventory",
    "Session",
    "Support",
    "absolute",
    "can_reset",
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
]

logger = logging.getLogger(__name__)

INVENTORY = "inventory"


@dataclass(frozen=True)
class Session:
    """The settings of a guided session and where they came from.

    ``settings_path`` is the ``--settings`` file the ui was started with;
    ``environ`` is where the review looks up API key variables.
    """

    settings: Settings
    settings_path: Path | None = None
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)


def new_state(session: Session) -> State:
    """Return an empty answer state over the option table and the defaults."""
    return State(OPTIONS, session.settings.defaults)


def absolute(path: Path | str) -> Path:
    """Expand ``~`` and make *path* absolute against the current directory."""
    return Path(os.path.abspath(os.path.expanduser(str(path).strip())))


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
