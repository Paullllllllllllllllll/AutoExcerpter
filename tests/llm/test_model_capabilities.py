"""Capability edge cases the recorded family table cannot express.

Per-model values live in ``data/capability_expectations.json`` and are checked
by test_capability_families.py.
"""

import dataclasses

import pytest

from autoexcerpter.llm.capabilities import (
    CapabilityError,
    ProviderCapabilities,
    detect_capabilities,
    ensure_image_support,
)

# The vendors' maximum output tokens per model id (OpenAI model pages, Google
# "Gemini models").
VENDOR_LIMITS: dict[str, int] = {
    "gpt-5-chat-latest": 16384,
    "gpt-5.1-chat-latest": 16384,
    "gpt-5.2-chat-latest": 16384,
    "gpt-5.3-chat-latest": 16384,
    "gemini-3-pro-image": 32768,
    "gemini-3-pro-image-preview": 32768,
    "gemini-3.1-flash-image": 32768,
    "gemini-3.1-flash-image-preview": 32768,
    "gpt-4.1-mini": 32768,
    "gemini-2.5-flash-lite": 65536,
}


@pytest.mark.parametrize(("model", "limit"), VENDOR_LIMITS.items())
def test_vendor_output_limits(model: str, limit: int) -> None:
    assert detect_capabilities(model).max_output_tokens == limit


@pytest.mark.parametrize(
    ("model", "base"),
    [
        ("gpt-5-chat-latest", "gpt-5"),
        ("gpt-5.1-chat-latest", "gpt-5.1"),
        ("gpt-5.2-chat-latest", "gpt-5.2"),
        ("gpt-5.3-chat-latest", "gpt-5.3"),
        ("gemini-3-pro-image-preview", "gemini-3-pro"),
        ("gemini-3.1-flash-image-preview", "gemini-3"),
    ],
)
def test_vendor_limit_changes_only_the_output_limit(model: str, base: str) -> None:
    """The specific rule keeps every other field of the version prefix."""
    skip = {"model_name", "max_output_tokens"}
    assert _fields(model, skip) == _fields(base, skip)


def test_gpt_6_1_sol_has_the_gpt_6_sol_profile() -> None:
    assert detect_capabilities("gpt-6.1-sol").family == "gpt-6.1-sol"
    skip = {"model_name", "family"}
    assert _fields("gpt-6.1-sol", skip) == _fields("gpt-6-sol", skip)


def _fields(model: str, skip: set[str]) -> dict[str, object]:
    caps = detect_capabilities(model)
    return {
        f.name: getattr(caps, f.name)
        for f in dataclasses.fields(caps)
        if f.name not in skip
    }


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro"])
def test_longer_prefix_wins_over_shorter_family(model: str) -> None:
    """A variant id matches its own family, not the 'gpt-5.4' prefix."""
    assert detect_capabilities(model).family == model
    assert detect_capabilities("gpt-5.4").family == "gpt-5.4"


@pytest.mark.parametrize("model", ["z-ai/glm-5.3-flash", "glm-5.3-flash"])
def test_glm_5_3_flash_takes_its_openrouter_profile(
    model: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The OpenRouter id and the bare model segment both match, silently."""
    with caplog.at_level("WARNING"):
        caps = detect_capabilities(model)

    assert caps.family == "glm-5.3-flash"
    assert caps.supports_vision and caps.supports_structured_output
    assert caps.supports_reasoning_effort
    assert (caps.max_context_tokens, caps.max_output_tokens) == (1048576, 943718)
    assert "No exact capability profile" not in caplog.text


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("z-ai/glm-5.3-flashx", "glm-5.3-flashx"),
        ("z-ai/glm-5v-turbo", "glm-5v-turbo"),
        ("z-ai/glm-4.6v", "glm-4.6v"),
        ("z-ai/glm-4.5v", "glm-4.5v"),
    ],
)
def test_glm_vision_siblings_take_prompted_json(model: str, family: str) -> None:
    """OpenRouter lists no structured outputs for the other GLM vision ids."""
    caps = detect_capabilities(model)

    assert caps.family == family
    assert caps.supports_vision and caps.supports_json_mode
    assert not caps.supports_structured_output


class TestCapabilityError:
    """Test CapabilityError and ensure_image_support."""

    def test_ensure_image_support_raises(self) -> None:
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="text-only",
            supports_vision=False,
        )
        with pytest.raises(CapabilityError, match="does not support image inputs"):
            ensure_image_support("text-only", caps)

    def test_ensure_image_support_passes(self) -> None:
        caps = ProviderCapabilities(
            provider_name="openai",
            model_name="gpt-4o",
            supports_vision=True,
        )
        ensure_image_support("gpt-4o", caps)
