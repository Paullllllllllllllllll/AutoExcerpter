"""AutoExcerpter's model capability table on the shared resolver engine.

Families are matched in order against the lowercased model id; the first
match wins, so specific prefixes precede general ones. Ids keep their
``provider:`` and ``models/`` prefixes, and ``vendor/model`` ids resolve
through the OpenRouter rules. The output limits of the GPT-5 chat-latest ids,
the Gemini 3 image models, gpt-4.1-mini and gemini-2.5-flash-lite are the
vendors' published maximums. Unmatched ids get the conservative profile:
no image input and no structured output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from autoexcerpter.common import capabilities as engine
from autoexcerpter.common.capabilities import (
    Capabilities,
    CapabilityError,
    CapabilityTable,
    FamilyRule,
    ProviderRule,
)

ImageDetail = Literal["auto", "high", "low"]
MediaResolution = Literal["low", "medium", "high", "ultra_high", "auto"]
ProviderType = Literal[
    "openai", "anthropic", "google", "openrouter", "custom", "unknown"
]

ProviderCapabilities = Capabilities

_Fields = Mapping[str, Any]

_TEXT_AND_IMAGES: _Fields = {
    "supports_vision": True,
    "supports_structured_output": True,
    "supports_json_mode": True,
}
_REASONING: _Fields = {"is_reasoning_model": True, "supports_reasoning_effort": True}
_NO_SAMPLING: _Fields = {"supports_temperature": False, "supports_top_p": False}
_PENALTIES: _Fields = {
    "supports_frequency_penalty": True,
    "supports_presence_penalty": True,
}
_DETAIL: _Fields = {"supports_image_detail": True, "default_image_detail": "high"}
_ORIGINAL: _Fields = {"supports_original_image_detail": True}
_PATCH_CAP: _Fields = {**_ORIGINAL, "image_original_patch_cap_30k": True}

_OPENAI: _Fields = {"provider_name": "openai", **_TEXT_AND_IMAGES, **_DETAIL}
_O_SERIES: _Fields = {
    **_OPENAI,
    **_REASONING,
    **_NO_SAMPLING,
    "max_context_tokens": 200000,
    "max_output_tokens": 100000,
}
_GPT4: _Fields = {**_OPENAI, **_PENALTIES, "max_output_tokens": 16384}
_ANTHROPIC: _Fields = {
    "provider_name": "anthropic",
    **_TEXT_AND_IMAGES,
    "max_context_tokens": 200000,
    "max_output_tokens": 8192,
}
_GOOGLE: _Fields = {
    "provider_name": "google",
    **_TEXT_AND_IMAGES,
    "supports_media_resolution": True,
    "max_context_tokens": 1000000,
    "max_output_tokens": 8192,
}
_OPENROUTER: _Fields = {"provider_name": "openrouter", **_TEXT_AND_IMAGES}


def _gpt5(context: int, **extra: Any) -> _Fields:
    return {
        **_O_SERIES,
        "supports_text_verbosity": True,
        "max_context_tokens": context,
        "max_output_tokens": 128000,
        **extra,
    }


def _chat_latest(family: str, context: int) -> FamilyRule:
    """The ``chat-latest`` id of a GPT-5 version: its family, 16,384 output."""
    fields = {**_gpt5(context), "max_output_tokens": 16384}
    return FamilyRule(family, fields, (f"{family}-chat-latest",))


def _claude(output: int, context: int = 200000, **extra: Any) -> _Fields:
    return {
        **_ANTHROPIC,
        **_REASONING,
        "max_context_tokens": context,
        "max_output_tokens": output,
        **extra,
    }


def _gemini(context: int = 1000000, **extra: Any) -> _Fields:
    return {
        **_GOOGLE,
        **_REASONING,
        "max_context_tokens": context,
        "max_output_tokens": 65536,
        **extra,
    }


_GPT56 = _gpt5(1050000, **_PATCH_CAP)
_ADAPTIVE = _claude(128000, 1000000, uses_adaptive_thinking=True, **_NO_SAMPLING)
_ADAPTIVE_HIGH_RES = {**_ADAPTIVE, "image_high_res_tier": True}
_CLAUDE_45 = _claude(64000, supports_top_p=False)


def _same(fields: _Fields, *families: str) -> tuple[FamilyRule, ...]:
    """One rule per family name, matched as a prefix."""
    return tuple(FamilyRule(family, fields, (family,)) for family in families)


def _dotted(fields: _Fields, *families: str) -> tuple[FamilyRule, ...]:
    """Rules that also match the dashed spelling (``4-5`` for ``4.5``)."""
    return tuple(
        FamilyRule(
            family, fields, tuple(dict.fromkeys((family, family.replace(".", "-"))))
        )
        for family in families
    )


def _last_segment_or(part: str, vendor: str) -> Callable[[str], bool]:
    """Match *part* in the model segment of an id or *vendor* anywhere."""
    return lambda name: part in name.rsplit("/", 1)[-1] or vendor in name


def _openrouter(
    family: str, fields: _Fields, *contains: str, **options: Any
) -> FamilyRule:
    """A rule for OpenRouter ids that contain one of *contains*."""
    return FamilyRule(
        family,
        {**_OPENROUTER, **fields},
        contains=contains,
        provider=engine.OPENROUTER,
        **options,
    )


def _has(*parts: str) -> Callable[[str], bool]:
    """Match ids that contain any of *parts*."""
    return lambda name: any(part in name for part in parts)


_OPENROUTER_OPENAI_REASONING = {**_DETAIL, **_REASONING, **_NO_SAMPLING}
_OPENROUTER_GEMINI = {
    "supports_media_resolution": True,
    "supports_reasoning_effort": True,
    "max_context_tokens": 1000000,
    "max_output_tokens": 8192,
}
_CLAUDE_TAIL = _last_segment_or("claude", "anthropic/")
_GEMINI_TAIL = _last_segment_or("gemini", "google/")
_LLAMA_TAIL = _last_segment_or("llama", "meta/")

_FAMILIES: tuple[FamilyRule, ...] = (
    *_same(_GPT56, "gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"),
    *_same(_GPT56, "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
    FamilyRule("gpt-5.6-sol", _GPT56, ("gpt-5.6",)),
    *_same(_gpt5(1050000), "gpt-5.5-pro"),
    *_same(_gpt5(1050000, **_ORIGINAL), "gpt-5.5", "gpt-5.4-pro"),
    *_same(_gpt5(400000), "gpt-5.4-mini", "gpt-5.4-nano"),
    *_same(_gpt5(1050000, **_ORIGINAL), "gpt-5.4"),
    _chat_latest("gpt-5.3", 400000),
    *_same(_gpt5(400000), "gpt-5.3-codex", "gpt-5.3"),
    _chat_latest("gpt-5.2", 400000),
    *_same(
        _gpt5(400000, supports_structured_output=False, supports_json_mode=False),
        "gpt-5.2-pro",
    ),
    *_same(_gpt5(400000), "gpt-5.2"),
    _chat_latest("gpt-5.1", 256000),
    _chat_latest("gpt-5", 256000),
    *_same(_gpt5(256000), "gpt-5.1", "gpt-5"),
    *_same(_O_SERIES, "o4-mini-deep-research", "o4-mini", "o4"),
    *_same(_O_SERIES, "o3-deep-research", "o3-pro"),
    *_same(
        {**_O_SERIES, "supports_vision": False, "supports_image_detail": False},
        "o3-mini",
    ),
    *_same(
        {
            **_O_SERIES,
            "supports_vision": False,
            "supports_image_detail": False,
            "supports_structured_output": False,
            "supports_json_mode": False,
            "max_context_tokens": 128000,
            "max_output_tokens": 65536,
        },
        "o1-mini",
    ),
    FamilyRule("o3", _O_SERIES, ("o3-",), exact=("o3",)),
    *_same({**_O_SERIES, "supports_structured_output": False}, "o1-pro"),
    *_same(_O_SERIES, "o1"),
    *_same(_GPT4, "gpt-4o"),
    *_same(
        {**_GPT4, "max_context_tokens": 1000000, "max_output_tokens": 32768},
        "gpt-4.1-mini",
    ),
    *_same({**_GPT4, "max_context_tokens": 1000000}, "gpt-4.1-nano"),
    *_same(
        {**_GPT4, "max_context_tokens": 1000000, "max_output_tokens": 32768},
        "gpt-4.1",
    ),
    *_same(_GPT4, "gpt-4.5"),
    *_same({**_GPT4, "max_output_tokens": 4096}, "gpt-4-turbo"),
    *_same({**_GPT4, "supports_vision": False, "max_output_tokens": 4096}, "gpt-4"),
    *_dotted(
        _ADAPTIVE_HIGH_RES,
        "claude-opus-5.5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4.8",
        "claude-opus-4.7",
    ),
    *_dotted(_ADAPTIVE, "claude-opus-4.6", "claude-sonnet-4.6"),
    *_dotted(_CLAUDE_45, "claude-opus-4.5"),
    *_dotted({**_CLAUDE_45, "max_context_tokens": 1000000}, "claude-sonnet-4.5"),
    *_dotted(_CLAUDE_45, "claude-haiku-4.5"),
    *_dotted(_claude(16384), "claude-opus-4.1", "claude-opus-4"),
    *_dotted(
        _ANTHROPIC,
        "claude-sonnet-4",
        "claude-3.7-sonnet",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
    ),
    *_same(
        {**_ANTHROPIC, "max_output_tokens": 4096},
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ),
    *_same(_ANTHROPIC, "claude"),
    FamilyRule(
        "gemini-3-pro",
        {**_gemini(2000000), "max_output_tokens": 32768},
        ("gemini-3-pro-image",),
    ),
    FamilyRule(
        "gemini-3",
        {**_gemini(), "max_output_tokens": 32768},
        ("gemini-3.1-flash-image", "gemini-3-1-flash-image"),
    ),
    *_dotted(_gemini(**_NO_SAMPLING), "gemini-3.7-flash", "gemini-3.6-flash"),
    *_dotted(
        _gemini(1048576),
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash-lite",
    ),
    FamilyRule(
        "gemini-3-flash", _gemini(1048576), ("gemini-3-flash", "gemini-3.0-flash")
    ),
    FamilyRule("gemini-3-pro", _gemini(2000000), ("gemini-3-pro", "gemini-3.0-pro")),
    *_same(_gemini(), "gemini-3"),
    *_dotted(_gemini(2000000), "gemini-2.5-pro"),
    *_dotted(
        {**_GOOGLE, "max_context_tokens": 1048576, "max_output_tokens": 65536},
        "gemini-2.5-flash-lite",
    ),
    *_dotted(_gemini(), "gemini-2.5-flash"),
    *_dotted(_GOOGLE, "gemini-2.0"),
    *_dotted({**_GOOGLE, "max_context_tokens": 2000000}, "gemini-1.5-pro"),
    *_dotted(_GOOGLE, "gemini-1.5-flash"),
    *_same(_GOOGLE, "gemini"),
    _openrouter(
        "openrouter-deepseek", _REASONING, "deepseek", predicate=_has("r1", "terminus")
    ),
    _openrouter("openrouter-deepseek", {}, "deepseek"),
    _openrouter(
        "openrouter-gpt-oss", {**_DETAIL, **_REASONING, **_PENALTIES}, "gpt-oss"
    ),
    _openrouter("openrouter-gpt5", _OPENROUTER_OPENAI_REASONING, "gpt-5"),
    _openrouter(
        "openrouter-o-series", _OPENROUTER_OPENAI_REASONING, "/o1", "/o3", "/o4"
    ),
    _openrouter("openrouter-openai", {**_DETAIL, **_PENALTIES}, "openai/", "gpt-4"),
    _openrouter(
        "openrouter-claude",
        {**_REASONING, "max_context_tokens": 200000},
        predicate=_CLAUDE_TAIL,
    ),
    _openrouter(
        "openrouter-gemini",
        {**_OPENROUTER_GEMINI, "is_reasoning_model": True},
        predicate=lambda name: (
            _GEMINI_TAIL(name) and _has("gemini-2.5", "gemini-3", "gemini-2-5")(name)
        ),
    ),
    _openrouter("openrouter-gemini", _OPENROUTER_GEMINI, predicate=_GEMINI_TAIL),
    _openrouter(
        "openrouter-llama",
        _PENALTIES,
        predicate=lambda name: _LLAMA_TAIL(name) and _has("vision", "llama-3.2")(name),
    ),
    _openrouter(
        "openrouter-llama",
        {**_PENALTIES, "supports_vision": False},
        predicate=_LLAMA_TAIL,
    ),
    _openrouter(
        "openrouter-mistral", {}, "pixtral", predicate=_has("mistral", "mixtral")
    ),
    _openrouter("openrouter-mistral", {"supports_vision": False}, "mistral", "mixtral"),
    _openrouter("openrouter", {}),
)

TABLE = CapabilityTable(
    families=_FAMILIES,
    providers=(
        ProviderRule("anthropic", prefixes=("claude",), contains=("anthropic",)),
        ProviderRule("google", prefixes=("gemini",), contains=("google",)),
        ProviderRule("openai", prefixes=("gpt", "o1", "o3", "o4", "text-", "chatgpt")),
    ),
    strip_prefixes=False,
)


def detect_provider(model_name: str) -> ProviderType:
    """Return the provider a model id names, or ``unknown``."""
    return cast(ProviderType, engine.detect_provider(model_name, TABLE))


def detect_capabilities(model_name: str) -> ProviderCapabilities:
    """Return the capabilities of *model_name* from AutoExcerpter's table."""
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
