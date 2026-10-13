"""AutoExcerpter's view of the shared model capability table.

The family rules and their sources are in
:mod:`autoexcerpter.common.capability_table`; this tool adds no extension.
Ids are matched after a ``provider:`` or ``models/`` prefix is removed, and
``vendor/model`` ids resolve through the OpenRouter rules. Unmatched ids get
the conservative profile: no image input and no structured output.
"""

from __future__ import annotations

from typing import Literal, cast

from autoexcerpter.common import capabilities as engine
from autoexcerpter.common.capabilities import Capabilities, CapabilityError
from autoexcerpter.common.capability_table import ORIGINAL_MAX_SIDE, TABLE

ImageDetail = Literal["auto", "high", "low"]
MediaResolution = Literal["low", "medium", "high", "ultra_high", "auto"]
ProviderType = Literal[
    "openai", "anthropic", "google", "openrouter", "custom", "unknown"
]

ProviderCapabilities = Capabilities


def detect_provider(model_name: str) -> ProviderType:
    """Return the provider a model id names, or ``unknown``."""
    return cast(ProviderType, engine.detect_provider(model_name, TABLE))


def detect_capabilities(model_name: str) -> ProviderCapabilities:
    """Return the capabilities of *model_name* from the shared table."""
    return engine.resolve_capabilities(model_name, TABLE)


def ensure_image_support(model_name: str, capabilities: ProviderCapabilities) -> None:
    """Raise ``CapabilityError`` unless the model accepts image input."""
    if not capabilities.supports_vision:
        raise CapabilityError(
            f"Model '{model_name}' does not support image inputs. "
            "Choose an image-capable model (e.g., gpt-5, gpt-4o, claude, gemini) "
            "or use a text-only flow."
        )


__all__ = [
    "ORIGINAL_MAX_SIDE",
    "TABLE",
    "CapabilityError",
    "ImageDetail",
    "MediaResolution",
    "ProviderCapabilities",
    "ProviderType",
    "detect_capabilities",
    "detect_provider",
    "ensure_image_support",
]
