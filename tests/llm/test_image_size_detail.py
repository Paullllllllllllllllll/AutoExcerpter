"""The OpenAI per-image ``detail`` a transcription request sends."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from autoexcerpter.llm.image_detail import resolve_request_detail

LOG = logging.getLogger("tests.image_detail")


@pytest.mark.parametrize(
    ("model", "options", "provider", "expected"),
    [
        ("gpt-5.4-mini", {"image_size": "high"}, "openai", "high"),
        ("gpt-5.6-sol", {"image_size": "original"}, "openai", "original"),
        ("gpt-5.4-mini", {}, "openai", None),
        ("gpt-5.4-mini", {"image_size": "gigantic"}, "openai", None),
        ("openai/gpt-5.4-mini", {"image_size": "high"}, "openrouter", None),
        ("claude-sonnet-4-5", {"image_size": "high"}, "anthropic", None),
    ],
)
def test_resolve_request_detail(
    model: str, options: dict[str, Any], provider: str, expected: str | None
) -> None:
    assert resolve_request_detail(options, provider, model, LOG) == expected


def test_original_falls_back_to_high_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        detail = resolve_request_detail(
            {"image_size": "original"}, "openai", "gpt-5.4-mini", LOG
        )
    assert detail == "high"
    assert "falling back to 'high'" in caplog.text
