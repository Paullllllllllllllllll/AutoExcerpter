"""Summary context: AutoExcerpter's sidecar names over the shared resolver.

An item's summary context comes from ``--context`` (text, or a path to a text
file), else from ``{name}_summary_context.txt`` beside the input, else from
``{folder}_summary_context.txt`` beside the input's folder, else from
``--fallback-context``. File context is joined into one prompt line; context
given as text enters the prompt as is.
"""

from __future__ import annotations

from pathlib import Path

from autoexcerpter.common.context import (
    ContextError,
    format_context_for_prompt,
    resolve_context,
)

__all__ = [
    "CONTEXT_SIZE_THRESHOLD",
    "CONTEXT_SUFFIX",
    "ContextError",
    "format_context_for_prompt",
    "summary_context",
]

CONTEXT_SUFFIX = "_summary_context"
CONTEXT_SIZE_THRESHOLD = 4000


def _text_or_file(value: str | None) -> str | Path | None:
    """Return *value* as a path when it names an existing file, else as text."""
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_file() else value


def summary_context(
    input_path: Path, given: str | None = None, *, default: str | None = None
) -> str | None:
    """Return the summary context of one item as it enters the prompt.

    *given* is the run's ``--context`` text or path and *default* its
    ``--fallback-context``, which applies only when no sidecar exists. For both, a
    path to an existing file is read and anything else is context text.

    Raises:
        ContextError: The given context file is empty or unreadable.
    """
    resolved = resolve_context(
        input_path,
        CONTEXT_SUFFIX,
        explicit=_text_or_file(given),
        default=_text_or_file(default),
        size_threshold=CONTEXT_SIZE_THRESHOLD,
    )
    if resolved is None:
        return None
    if resolved.path is None:
        return resolved.text
    return format_context_for_prompt(resolved.text)
