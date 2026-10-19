"""The model of one phase, as the LLM layer receives it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PhaseModel"]


@dataclass(frozen=True)
class PhaseModel:
    """The model of one phase (transcription or summary).

    *provider* is the provider the run spec resolved. *response_format*
    None takes schema when the model supports it, else json. *options* holds the
    request options (``max_output_tokens`` when set,
    ``reasoning.effort``, ``text.verbosity``, ``temperature``, ``top_p``,
    ``image_size``). *api_key_env* None takes the provider's default key
    variable. *endpoint* names the custom endpoint behind *base_url*;
    *supports_vision* is that endpoint's declared image support.
    """

    name: str
    provider: str
    response_format: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    service_tier: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    supports_vision: bool | None = None
