"""Tests for the vendored capability resolver engine."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from autoexcerpter.common.capabilities import (
    Capabilities,
    CapabilityError,
    CapabilityTable,
    FamilyRule,
    ProviderRule,
    detect_provider,
    ensure_image_support,
    resolve_capabilities,
    split_model_id,
)

OPENAI: dict[str, Any] = {
    "provider_name": "openai",
    "supports_vision": True,
    "supports_structured_output": True,
    "is_reasoning_model": True,
    "supports_temperature": False,
}
OPENROUTER: dict[str, Any] = {
    "provider_name": "openrouter",
    "supports_vision": True,
    "supports_structured_output": True,
}

TABLE = CapabilityTable(
    families=(
        FamilyRule(
            "gpt-5.6-sol", {**OPENAI, "max_output_tokens": 128000}, ("gpt-5.6",)
        ),
        FamilyRule("gpt-5", {**OPENAI, "supports_text_verbosity": True}, ("gpt-5",)),
        FamilyRule("o3-mini", {**OPENAI, "supports_vision": False}, ("o3-mini",)),
        FamilyRule("o3", OPENAI, prefixes=("o3-",), exact=("o3",), exclude=("o3-pro",)),
        FamilyRule(
            "gemini-2.5-flash",
            {"provider_name": "google", "supports_vision": True},
            ("gemini-2.5-flash",),
        ),
        FamilyRule(
            "openrouter-llama-vision",
            {**OPENROUTER, "extras": {"supports_audio": False}},
            contains=("llama",),
            provider="openrouter",
            predicate=lambda name: "vision" in name or "llama-3.2" in name,
        ),
        FamilyRule(
            "openrouter-llama",
            {**OPENROUTER, "supports_vision": False},
            contains=("llama",),
            provider="openrouter",
        ),
        FamilyRule("openrouter", OPENROUTER, provider="openrouter", warn=True),
    ),
    providers=(
        ProviderRule("openai", prefixes=("gpt", "o1", "o3", "o4")),
        ProviderRule("anthropic", prefixes=("claude",), contains=("anthropic",)),
        ProviderRule("google", prefixes=("gemini",)),
    ),
)


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("gpt-5.6-terra", "openai"),
        ("o3", "openai"),
        ("claude-sonnet-4-5", "anthropic"),
        ("my-anthropic-model", "anthropic"),
        ("Gemini-2.5-Flash", "google"),
        ("models/gemini-2.5-flash", "google"),
        ("models/gemma-3", "google"),
        ("anthropic/claude-sonnet-4.5", "openrouter"),
        ("openrouter/auto", "openrouter"),
        ("anthropic:claude-opus-4-5", "anthropic"),
        ("custom:org/model", "custom"),
        ("mystery", "unknown"),
    ],
)
def test_detect_provider(model: str, provider: str) -> None:
    assert detect_provider(model, TABLE) == provider


def test_default_provider_is_configurable() -> None:
    table = CapabilityTable(families=(), default_provider="openai")

    assert detect_provider("mystery", table) == "openai"


def test_unknown_prefix_before_colon_is_kept() -> None:
    assert split_model_id("ft:gpt-4o:org:x", TABLE) == (None, "ft:gpt-4o:org:x")
    assert split_model_id(" OpenAI:GPT-5 ", TABLE) == ("openai", "gpt-5")


def test_first_matching_rule_wins() -> None:
    caps = resolve_capabilities("gpt-5.6-terra", TABLE)

    assert caps.family == "gpt-5.6-sol"
    assert caps.max_output_tokens == 128000
    assert not caps.supports_text_verbosity
    assert caps.model_name == "gpt-5.6-terra"
    assert caps.provider_name == "openai"


def test_exact_prefix_and_exclude_matchers() -> None:
    assert resolve_capabilities("o3", TABLE).family == "o3"
    assert resolve_capabilities("o3-2025-04-16", TABLE).family == "o3"
    assert resolve_capabilities("o3-mini", TABLE).family == "o3-mini"
    assert resolve_capabilities("o3-pro", TABLE).family == "unknown"
    assert resolve_capabilities("o3x", TABLE).family == "unknown"


def test_prefixed_ids_match_their_family() -> None:
    assert resolve_capabilities("openai:gpt-5", TABLE).family == "gpt-5"
    google = resolve_capabilities("models/gemini-2.5-flash", TABLE)
    assert google.family == "gemini-2.5-flash"
    assert google.model_name == "models/gemini-2.5-flash"


def test_table_can_keep_prefixes() -> None:
    table = CapabilityTable(
        families=TABLE.families, providers=TABLE.providers, strip_prefixes=False
    )

    assert split_model_id("OpenAI:GPT-5", table) == (None, "openai:gpt-5")
    assert detect_provider("openai:gpt-5", table) == "unknown"
    assert detect_provider("anthropic:claude-x", table) == "anthropic"
    assert detect_provider("models/gemini-2.5-flash", table) == "openrouter"
    assert resolve_capabilities("openai:gpt-5", table).family == "unknown"
    assert resolve_capabilities("models/gemini-2.5-flash", table).family == (
        "openrouter"
    )


def test_openrouter_rules_apply_only_to_openrouter() -> None:
    vision = resolve_capabilities("meta-llama/llama-3.2-90b-vision", TABLE)
    text_only = resolve_capabilities("meta-llama/llama-3.1-70b", TABLE)

    assert vision.family == "openrouter-llama-vision"
    assert vision.supports_vision
    assert vision.extra("supports_audio") is False
    assert text_only.family == "openrouter-llama"
    assert not text_only.supports_vision
    assert text_only.extra("supports_audio", "absent") == "absent"
    assert resolve_capabilities("llama-3.2", TABLE).family == "unknown"


def test_catch_all_rule_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        first = resolve_capabilities("vendor/brand-new-model", TABLE)
        resolve_capabilities("vendor/brand-new-model", TABLE)

    assert first.family == "openrouter"
    assert first.supports_structured_output
    warnings = [r for r in caplog.records if "vendor/brand-new-model" in r.message]
    assert len(warnings) == 1


def test_unknown_model_gets_the_conservative_profile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        caps = resolve_capabilities("never-seen-model-x1", TABLE)
        resolve_capabilities("never-seen-model-x1", TABLE)

    assert caps.family == "unknown"
    assert caps.provider_name == "unknown"
    assert not caps.supports_vision
    assert not caps.supports_structured_output
    assert caps.supports_temperature
    assert caps.max_output_tokens == 4096
    warnings = [r for r in caplog.records if "never-seen-model-x1" in r.message]
    assert len(warnings) == 1


def test_unknown_profile_can_be_overridden() -> None:
    table = CapabilityTable(families=(), unknown={"supports_structured_output": True})

    assert resolve_capabilities("never-seen-model-x2", table).supports_structured_output


def test_provider_argument_overrides_detection() -> None:
    table = CapabilityTable(
        families=(
            FamilyRule("custom", {"supports_vision": True}, provider="custom"),
            *TABLE.families,
        ),
        providers=TABLE.providers,
    )

    custom = resolve_capabilities("org/fake-model", table, provider="custom")
    routed = resolve_capabilities("org/fake-model", table)

    assert custom.family == "custom"
    assert custom.provider_name == "custom"
    assert routed.family == "openrouter"


def test_provider_name_defaults_to_the_resolved_provider() -> None:
    table = CapabilityTable(
        families=(FamilyRule("claude", {"supports_vision": True}, ("claude",)),),
        providers=TABLE.providers,
    )

    assert resolve_capabilities("claude-x", table).provider_name == "anthropic"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="supports_vison"):
        FamilyRule("typo", {"supports_vison": True}, ("x",))
    with pytest.raises(ValueError, match="family"):
        FamilyRule("bad", {"family": "other"}, ("x",))
    with pytest.raises(ValueError, match="supports_vison"):
        CapabilityTable(families=(), unknown={"supports_vison": True})


def test_ensure_image_support() -> None:
    ensure_image_support(resolve_capabilities("gpt-5", TABLE))
    blind = Capabilities(provider_name="openai", model_name="o3-mini")

    with pytest.raises(CapabilityError, match="o3-mini"):
        ensure_image_support(blind)
    assert issubclass(CapabilityError, ValueError)
