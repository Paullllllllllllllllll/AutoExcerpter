"""Provider names and the provider a model name implies.

A ``provider:`` prefix naming a supported provider selects it; a
``vendor/model`` id selects openrouter; ``claude``, ``gemini`` and the OpenAI
name prefixes select their providers. Any other name implies no provider and
runs on openai.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "implied_provider",
    "infer_provider",
    "split_provider_prefix",
]

PROVIDERS = ("openai", "anthropic", "google", "openrouter", "custom")
DEFAULT_PROVIDER = "openai"

_NAME_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("claude",)),
    ("google", ("gemini",)),
    ("openai", ("gpt", "chatgpt", "o1", "o3", "o4", "text-")),
)


def split_provider_prefix(model: str) -> tuple[str | None, str]:
    """Return the provider a ``provider:`` prefix names and the bare model name.

    Only a prefix naming a supported provider counts, so model ids that
    contain a colon (OpenAI fine-tunes, ``ft:gpt-4o:org:id``) stay whole.
    """
    prefix, separator, rest = model.partition(":")
    provider = prefix.strip().lower()
    if separator and provider in PROVIDERS:
        return provider, rest.strip()
    return None, model


def implied_provider(model: str) -> str | None:
    """Return the provider *model* implies, or None when the name implies none."""
    prefixed, name = split_provider_prefix(model)
    if prefixed is not None:
        return prefixed
    lowered = name.strip().lower()
    if "/" in lowered:
        return "openrouter"
    for provider, prefixes in _NAME_PREFIXES:
        if lowered.startswith(prefixes):
            return provider
    return None


def infer_provider(model: str) -> str:
    """Return the provider of *model*: the implied one, else openai."""
    return implied_provider(model) or DEFAULT_PROVIDER
