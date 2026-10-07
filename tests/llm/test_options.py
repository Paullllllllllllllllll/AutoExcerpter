"""Request options per provider and the response format of a phase."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from autoexcerpter.llm.options import (
    build_invoke_kwargs,
    request_service_tier,
    response_format_for,
)

FULL: dict[str, Any] = {
    "max_output_tokens": 128000,
    "reasoning": {"effort": "high"},
    "text": {"verbosity": "medium"},
    "temperature": 1.0,
}


def _options(**changes: Any) -> dict[str, Any]:
    return {**FULL, **changes}


# ============================================================================
# Invoke kwargs per provider
# ============================================================================
@pytest.mark.parametrize(
    ("provider", "model", "options", "expected"),
    [
        pytest.param(
            "openai",
            "gpt-5.6-terra",
            FULL,
            {
                "max_output_tokens": 128000,
                "service_tier": "flex",
                "reasoning": {"effort": "high"},
                "text": {"verbosity": "medium"},
            },
            id="openai-reasoning",
        ),
        pytest.param(
            "openai",
            "gpt-4o",
            _options(temperature=0.5, top_p=0.9),
            {
                "max_output_tokens": 128000,
                "service_tier": "flex",
                "temperature": 0.5,
                "top_p": 0.9,
            },
            id="openai-sampling",
        ),
        pytest.param(
            "anthropic",
            "claude-sonnet-4-5",
            FULL,
            {
                "max_tokens": 128000,
                "thinking": {"type": "enabled", "budget_tokens": 8192},
            },
            id="anthropic-budget",
        ),
        pytest.param(
            "anthropic",
            "claude-fable-5",
            _options(reasoning={"effort": "minimal"}),
            {"max_tokens": 128000, "output_config": {"effort": "low"}},
            id="anthropic-adaptive",
        ),
        pytest.param(
            "anthropic",
            "claude-fable-5",
            _options(reasoning={"effort": "none"}),
            {"max_tokens": 128000},
            id="anthropic-adaptive-none",
        ),
        pytest.param(
            "google",
            "gemini-2.5-flash",
            _options(reasoning={"effort": "minimal"}),
            {
                "max_output_tokens": 128000,
                "thinking_config": {"thinking_budget": 512},
            },
            id="google-budget",
        ),
        pytest.param(
            "google",
            "gemini-3.5-flash",
            _options(reasoning={"effort": "xhigh"}),
            {
                "max_output_tokens": 128000,
                "thinking_config": {"thinking_level": "high"},
            },
            id="google-level",
        ),
        pytest.param(
            "google",
            "gemini-2.0-flash",
            FULL,
            {"max_output_tokens": 128000, "temperature": 1.0},
            id="google-without-thinking",
        ),
        pytest.param(
            "openrouter",
            "anthropic/claude-sonnet-4.5",
            FULL,
            {"max_tokens": 128000, "temperature": 1.0},
            id="openrouter",
        ),
        pytest.param(
            "custom",
            "org/fake-model",
            FULL,
            {"max_tokens": 128000, "temperature": 1.0},
            id="custom",
        ),
    ],
)
def test_invoke_kwargs_per_provider(
    provider: str, model: str, options: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert build_invoke_kwargs(provider, model, options) == expected


def test_anthropic_without_an_output_limit_takes_the_model_maximum() -> None:
    kwargs = build_invoke_kwargs("anthropic", "claude-sonnet-4-5", {})
    assert kwargs == {"max_tokens": 64000}


def test_openai_sends_the_requested_service_tier() -> None:
    kwargs = build_invoke_kwargs("openai", "gpt-5.6-terra", {}, "priority")

    assert kwargs["service_tier"] == "priority"
    assert request_service_tier("openai", None) == "flex"
    assert request_service_tier("anthropic", "priority") == "auto"


def test_reasoning_and_verbosity_need_a_mapping_with_the_key() -> None:
    kwargs = build_invoke_kwargs(
        "openai", "gpt-5.6-terra", {"reasoning": "high", "text": {"other": 1}}
    )
    assert "reasoning" not in kwargs
    assert "text" not in kwargs


def test_top_p_is_sent_only_when_given_and_supported() -> None:
    assert "top_p" not in build_invoke_kwargs("openai", "gpt-4o", {})
    supported = build_invoke_kwargs("openai", "gpt-4o", {"top_p": 0.5})
    assert supported["top_p"] == 0.5
    rejected = build_invoke_kwargs(
        "anthropic", "claude-haiku-4-5", {"top_p": 0.5, "reasoning": {"effort": "none"}}
    )
    assert "top_p" not in rejected


def test_anthropic_effort_copied_onto_the_model_merges_with_the_format() -> None:
    """The structured-output path copies kwargs onto the model; the copy's
    ``output_config`` must merge with the wrapper's schema format."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    model = ChatAnthropic(
        api_key="test-key",  # type: ignore[arg-type]
        model="claude-sonnet-4-5",  # type: ignore[call-arg]
        max_tokens=1024,
    )
    copied = model.model_copy(
        update={"max_tokens": 32000, "output_config": {"effort": "medium"}}
    )

    payload = copied._get_request_payload(
        [HumanMessage(content="hi")],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {}},
            }
        },
    )

    assert payload["max_tokens"] == 32000
    assert payload["output_config"]["effort"] == "medium"
    assert payload["output_config"]["format"]["type"] == "json_schema"


# ============================================================================
# Response format
# ============================================================================
@pytest.mark.parametrize(
    ("requested", "provider", "model", "expected"),
    [
        (None, "openai", "gpt-5.6-terra", "schema"),
        (None, "openai", "gpt-5.2-pro", "json"),
        (None, "anthropic", "claude-sonnet-4-5", "schema"),
        (None, "custom", "org/model", "json"),
        ("schema", "custom", "org/model", "schema"),
        ("json", "openai", "gpt-5.6-terra", "json"),
        ("text", "google", "gemini-2.5-flash", "text"),
    ],
)
def test_response_format_for(
    requested: str | None, provider: str, model: str, expected: str
) -> None:
    assert response_format_for(requested, provider, model) == expected


def test_schema_without_structured_output_falls_back_to_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert response_format_for("schema", "openai", "gpt-5.2-pro") == "json"
    assert "prompted JSON" in caplog.text


def test_unknown_response_format_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unknown response format"):
        response_format_for("xml", "openai", "gpt-5.6-terra")
