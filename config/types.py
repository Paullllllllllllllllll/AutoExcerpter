"""Frozen dataclass value objects for concurrency and model configuration.

These are typed projections of the YAML config sections loaded by
:class:`config.loader.ConfigLoader`. Currently not used in production code
paths but covered by the test suite; provided here as a stable contract
for future typed-config consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Configuration for concurrent processing."""

    image_processing_limit: int = 24
    transcription_limit: int = 150
    summary_limit: int = 150
    transcription_delay: float = 0.05
    summary_delay: float = 0.05
    transcription_service_tier: str = "flex"
    summary_service_tier: str = "flex"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ConcurrencyConfig:
        """Create ConcurrencyConfig from configuration dictionary."""
        img_proc = config.get("image_processing", {})
        api_req = config.get("api_requests", {})
        trans_cfg = api_req.get("transcription", {})
        summ_cfg = api_req.get("summary", {})

        return cls(
            image_processing_limit=img_proc.get("concurrency_limit", 24),
            transcription_limit=trans_cfg.get("concurrency_limit", 150),
            summary_limit=summ_cfg.get("concurrency_limit", 150),
            transcription_delay=trans_cfg.get("delay_between_tasks", 0.05),
            summary_delay=summ_cfg.get("delay_between_tasks", 0.05),
            transcription_service_tier=trans_cfg.get("service_tier", "flex"),
            summary_service_tier=summ_cfg.get("service_tier", "flex"),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model parameters."""

    name: str
    max_output_tokens: int
    reasoning_effort: str | None = None
    text_verbosity: str | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from configuration dictionary."""
        reasoning = config.get("reasoning", {})
        text = config.get("text", {})

        return cls(
            name=config.get("name", "gpt-5-mini"),
            max_output_tokens=config.get("max_output_tokens", 16384),
            reasoning_effort=(
                reasoning.get("effort") if isinstance(reasoning, dict) else None
            ),
            text_verbosity=text.get("verbosity") if isinstance(text, dict) else None,
        )


__all__ = ["ConcurrencyConfig", "ModelConfig"]
