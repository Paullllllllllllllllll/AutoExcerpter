"""The model capability table the tools share.

Every value follows the documentation of the lab that makes the model, read on
13.10.2026: OpenAI's model pages, reasoning, flex-processing and
prompt-caching guides; Anthropic's model, thinking, effort and
structured-output pages; Google's Gemini and Gemma model pages; Z.ai's GLM
pages; OpenRouter's model list for routed ids. Where a lab states nothing,
the value is the conservative one: no request field the API might reject, no
output limit above the documented one. Ids no rule matches, and ids no lab
documents (``gpt-5.3``, ``gpt-5.3-chat``, ``glm-flash-latest``), get the
conservative profile of the engine.

Rules are matched in order against the lowercased id after a ``provider:`` or
``models/`` prefix is removed; the first match wins, so specific prefixes
precede general ones. ``vendor/model`` ids resolve through the OpenRouter
rules. :func:`build_table` adds a tool's :class:`TableExtension`: rules
matched before the shared ones and extra keys for the ``extras`` of named
families (ChronoTranscriber's audio flags use it). :data:`AUDIO_FAMILIES`
lists the families that take audio input and :data:`SPEECH_TO_TEXT_FAMILIES`
those served only by OpenAI's transcription endpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from .capabilities import OPENROUTER, CapabilityTable, FamilyRule, ProviderRule

__all__ = [
    "AUDIO_FAMILIES",
    "FAMILIES",
    "ORIGINAL_MAX_SIDE",
    "PROVIDERS",
    "SPEECH_TO_TEXT_FAMILIES",
    "TABLE",
    "TableExtension",
    "build_table",
]

ORIGINAL_MAX_SIDE = "image_original_max_side_px"
"""The ``Capabilities.extras`` key of the default long-side limit in pixels of
OpenAI pages sent at ``original`` detail."""

_Fields = Mapping[str, Any]

_REASONING: _Fields = {"is_reasoning_model": True, "supports_reasoning_effort": True}
_NO_SAMPLING: _Fields = {"supports_temperature": False, "supports_top_p": False}
_PENALTIES: _Fields = {
    "supports_frequency_penalty": True,
    "supports_presence_penalty": True,
}
_TEXT_ONLY: _Fields = {"supports_vision": False, "supports_image_detail": False}
_NO_STRUCTURE: _Fields = {
    "supports_structured_output": False,
    "supports_json_mode": False,
}
_ORIGINAL: _Fields = {"supports_original_image_detail": True}
_PATCH_CAP: _Fields = {**_ORIGINAL, "image_original_patch_cap_30k": True}
_VERBOSITY: _Fields = {"supports_text_verbosity": True}
_FLEX: _Fields = {"supports_flex_tier": True}
_CACHE_TTL: _Fields = {"prompt_cache_parameter": "prompt_cache_options"}
_CACHE_RETENTION: _Fields = {"prompt_cache_parameter": "prompt_cache_retention"}

# Reasoning efforts per OpenAI model page and the reasoning guide.
_NONE_TO_MAX = ("none", "low", "medium", "high", "xhigh", "max")
_LOW_TO_MAX = ("low", "medium", "high", "xhigh", "max")
_NONE_TO_XHIGH = ("none", "low", "medium", "high", "xhigh")
_NONE_TO_HIGH = ("none", "low", "medium", "high")
_MINIMAL_TO_HIGH = ("minimal", "low", "medium", "high")
_LOW_TO_XHIGH = ("low", "medium", "high", "xhigh")
_PRO = ("medium", "high", "xhigh")
# Thinking levels per Gemini and Gemma model page.
_LOW_TO_HIGH = ("low", "medium", "high")
_GEMMA_LEVELS = ("minimal", "high")


def _efforts(levels: tuple[str, ...]) -> _Fields:
    return {"reasoning_efforts": levels}


def _rules(fields: _Fields, *families: str) -> tuple[FamilyRule, ...]:
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


# OpenAI: model pages, the reasoning guide, the flex-processing guide (models
# with a Flex price) and the prompt-caching guide (prompt_cache_options.ttl on
# GPT-5.6 and later; prompt_cache_retention on the models it lists).
_OPENAI: _Fields = {
    "provider_name": "openai",
    "supports_vision": True,
    "supports_image_detail": True,
    "default_image_detail": "high",
    "supports_structured_output": True,
    "supports_json_mode": True,
}
_O_SERIES: _Fields = {
    **_OPENAI,
    **_REASONING,
    **_NO_SAMPLING,
    "max_context_tokens": 200000,
    "max_output_tokens": 100000,
}


def _gpt(context: int, efforts: tuple[str, ...], **extra: Any) -> _Fields:
    """A GPT-5 or GPT-6 reasoning model with 128,000 output tokens."""
    return {
        **_O_SERIES,
        **_efforts(efforts),
        "max_context_tokens": context,
        "max_output_tokens": 128000,
        **extra,
    }


_GPT6 = _gpt(1050000, _NONE_TO_MAX, **_PATCH_CAP, **_VERBOSITY, **_FLEX, **_CACHE_TTL)
_CHAT_LATEST: _Fields = {
    **_OPENAI,
    **_NO_SAMPLING,
    "max_context_tokens": 128000,
    "max_output_tokens": 16384,
}
_CODEX = _gpt(400000, _LOW_TO_XHIGH)
_GPT4: _Fields = {**_OPENAI, "max_output_tokens": 16384}
_GPT41: _Fields = {**_GPT4, "max_context_tokens": 1047576, "max_output_tokens": 32768}
_TRANSCRIBE: _Fields = {
    "provider_name": "openai",
    "default_image_detail": "high",
    "supports_top_p": False,
    "max_context_tokens": 16000,
    "max_output_tokens": 2000,
}

_OPENAI_RULES: tuple[FamilyRule, ...] = (
    *_rules({**_GPT6, **_efforts(_LOW_TO_MAX)}, "gpt-6.1-sol", "gpt-6-astra"),
    *_rules(_GPT6, "gpt-6-sol"),
    *_rules({**_GPT6, "extras": {ORIGINAL_MAX_SIDE: 3000}}, "gpt-6-luna"),
    *_rules(_GPT6, "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"),
    # The gpt-5.6 alias routes to GPT-5.6 Sol.
    FamilyRule("gpt-5.6-sol", _GPT6, ("gpt-5.6",)),
    *_rules(_gpt(1050000, _PRO, **_FLEX, **_CACHE_RETENTION), "gpt-5.5-pro"),
    *_rules(
        _gpt(
            1050000,
            _NONE_TO_XHIGH,
            **_ORIGINAL,
            **_VERBOSITY,
            **_FLEX,
            **_CACHE_RETENTION,
        ),
        "gpt-5.5",
    ),
    *_rules(_gpt(1050000, _PRO, **_ORIGINAL, **_NO_STRUCTURE, **_FLEX), "gpt-5.4-pro"),
    *_rules(
        _gpt(400000, _NONE_TO_XHIGH, **_ORIGINAL, **_VERBOSITY, **_FLEX),
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ),
    *_rules(
        _gpt(
            1050000,
            _NONE_TO_XHIGH,
            **_ORIGINAL,
            **_VERBOSITY,
            **_FLEX,
            **_CACHE_RETENTION,
        ),
        "gpt-5.4",
    ),
    *_rules(_CODEX, "gpt-5.3-codex", "gpt-5.2-codex"),
    *_rules(_CHAT_LATEST, "gpt-5.3-chat-latest", "gpt-5.2-chat-latest"),
    *_rules({**_CHAT_LATEST, **_CACHE_RETENTION}, "gpt-5.1-chat-latest"),
    *_rules(_CHAT_LATEST, "gpt-5-chat-latest"),
    *_rules(_gpt(400000, _PRO, **_NO_STRUCTURE), "gpt-5.2-pro"),
    *_rules(
        _gpt(400000, _NONE_TO_XHIGH, **_VERBOSITY, **_FLEX, **_CACHE_RETENTION),
        "gpt-5.2",
    ),
    *_rules(
        _gpt(400000, _NONE_TO_HIGH, **_VERBOSITY, **_FLEX, **_CACHE_RETENTION),
        "gpt-5.1",
    ),
    *_rules(
        _gpt(400000, _MINIMAL_TO_HIGH, **_VERBOSITY, **_FLEX),
        "gpt-5-mini",
        "gpt-5-nano",
    ),
    FamilyRule(
        "gpt-5",
        _gpt(400000, _MINIMAL_TO_HIGH, **_VERBOSITY, **_FLEX, **_CACHE_RETENTION),
        ("gpt-5-",),
        exact=("gpt-5",),
    ),
    *_rules(
        {**_O_SERIES, "supports_structured_output": False}, "o4-mini-deep-research"
    ),
    *_rules({**_O_SERIES, **_FLEX}, "o4-mini"),
    *_rules(_O_SERIES, "o4"),
    *_rules({**_O_SERIES, "supports_structured_output": False}, "o3-deep-research"),
    *_rules(_O_SERIES, "o3-pro"),
    *_rules({**_O_SERIES, **_TEXT_ONLY}, "o3-mini"),
    *_rules(
        {
            **_O_SERIES,
            **_TEXT_ONLY,
            **_NO_STRUCTURE,
            "supports_reasoning_effort": False,
            "max_context_tokens": 128000,
            "max_output_tokens": 65536,
        },
        "o1-mini",
    ),
    FamilyRule("o3", {**_O_SERIES, **_FLEX}, ("o3-",), exact=("o3",)),
    *_rules(_O_SERIES, "o1-pro"),
    *_rules({**_O_SERIES, "supports_reasoning_effort": False}, "o1"),
    FamilyRule(
        "openai-audio",
        _TRANSCRIBE,
        ("gpt-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"),
    ),
    *_rules(_GPT4, "gpt-4o"),
    *_rules(_GPT41, "gpt-4.1-mini", "gpt-4.1-nano"),
    *_rules({**_GPT41, **_CACHE_RETENTION}, "gpt-4.1"),
    *_rules(_GPT4, "gpt-4.5"),
    *_rules({**_GPT4, **_NO_STRUCTURE, "max_output_tokens": 4096}, "gpt-4-turbo"),
    *_rules(
        {
            **_GPT4,
            **_TEXT_ONLY,
            **_NO_STRUCTURE,
            "max_context_tokens": 8192,
            "max_output_tokens": 8192,
        },
        "gpt-4",
    ),
)

# Anthropic: models overview, thinking, effort and structured-output pages.
_ANTHROPIC: _Fields = {
    "provider_name": "anthropic",
    "supports_vision": True,
    "supports_structured_output": True,
    "supports_json_mode": True,
    "max_context_tokens": 200000,
    "max_output_tokens": 8192,
}
_ADAPTIVE: _Fields = {
    **_ANTHROPIC,
    **_REASONING,
    **_NO_SAMPLING,
    "uses_adaptive_thinking": True,
    "max_context_tokens": 1000000,
    "max_output_tokens": 128000,
}
_CLAUDE_45: _Fields = {
    **_ANTHROPIC,
    **_REASONING,
    "supports_top_p": False,
    "max_output_tokens": 64000,
}
_CLAUDE_4: _Fields = {
    **_ANTHROPIC,
    **_REASONING,
    "supports_structured_output": False,
}

_ANTHROPIC_RULES: tuple[FamilyRule, ...] = (
    *_dotted(
        {**_ADAPTIVE, "image_high_res_tier": True},
        "claude-opus-5.5",
        "claude-sonnet-5.5",
        "claude-haiku-5.5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4.8",
        "claude-opus-4.7",
    ),
    *_dotted(_ADAPTIVE, "claude-opus-4.6", "claude-sonnet-4.6"),
    *_dotted(_CLAUDE_45, "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"),
    *_dotted(
        {**_CLAUDE_4, "max_output_tokens": 32000}, "claude-opus-4.1", "claude-opus-4"
    ),
    *_dotted({**_CLAUDE_4, "max_output_tokens": 64000}, "claude-sonnet-4"),
    *_dotted(
        {**_ANTHROPIC, "supports_structured_output": False},
        "claude-3.7-sonnet",
        "claude-3.5-sonnet",
    ),
    *_dotted({**_ANTHROPIC, **_NO_STRUCTURE}, "claude-3.5-haiku"),
    *_rules(
        {**_ANTHROPIC, **_NO_STRUCTURE, "max_output_tokens": 4096},
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ),
    *_rules(_ANTHROPIC, "claude"),
)

# Google: Gemini model pages, the Gemini 3 guide, the Gemma on Gemini API page.
_GOOGLE: _Fields = {
    "provider_name": "google",
    "supports_vision": True,
    "supports_media_resolution": True,
    "supports_structured_output": True,
    "supports_json_mode": True,
    "max_context_tokens": 1000000,
    "max_output_tokens": 8192,
}
_THINKING: _Fields = {**_GOOGLE, **_REASONING, "max_output_tokens": 65536}
# Gemini 3.6 Flash and later ignore samplers; 3.5 Flash routes to 3.6 Flash
# and 3.7 Flash to 3.8 Flash.
_FLASH_36: _Fields = {**_THINKING, **_NO_SAMPLING}
_IMAGE_MODEL: _Fields = {
    **_GOOGLE,
    **_NO_STRUCTURE,
    "is_reasoning_model": True,
    "max_output_tokens": 32768,
}
_GEMMA: _Fields = {
    **_GOOGLE,
    **_REASONING,
    **_NO_STRUCTURE,
    **_efforts(_GEMMA_LEVELS),
    "supports_media_resolution": False,
    "max_context_tokens": 262144,
}

_GOOGLE_RULES: tuple[FamilyRule, ...] = (
    *_rules(_GEMMA, "gemma-4-31b-it", "gemma-4-26b-a4b-it"),
    *_dotted(
        {**_FLASH_36, **_efforts(_LOW_TO_HIGH)}, "gemini-3.8-flash", "gemini-3.7-flash"
    ),
    *_dotted(_FLASH_36, "gemini-3.6-flash", "gemini-3.5-flash"),
    *_dotted(
        {**_THINKING, **_efforts(_LOW_TO_HIGH), "max_context_tokens": 1048576},
        "gemini-3.1-pro",
    ),
    *_dotted({**_THINKING, "max_context_tokens": 1048576}, "gemini-3.1-flash-lite"),
    FamilyRule(
        "gemini-3.1-flash-image",
        {**_IMAGE_MODEL, "max_context_tokens": 131072},
        ("gemini-3.1-flash-image", "gemini-3-1-flash-image"),
    ),
    FamilyRule(
        "gemini-3-pro-image",
        {**_IMAGE_MODEL, "max_context_tokens": 65536},
        ("gemini-3-pro-image",),
    ),
    FamilyRule(
        "gemini-3-flash",
        {**_THINKING, "max_context_tokens": 1048576},
        ("gemini-3-flash", "gemini-3.0-flash"),
    ),
    FamilyRule(
        "gemini-3-pro",
        {**_THINKING, "max_context_tokens": 2000000},
        ("gemini-3-pro", "gemini-3.0-pro"),
    ),
    FamilyRule(
        "gemini-3-preview", _THINKING, ("gemini-3-preview", "gemini-3.0-preview")
    ),
    *_rules(_THINKING, "gemini-3"),
    *_dotted({**_THINKING, "max_context_tokens": 1048576}, "gemini-2.5-pro"),
    *_dotted(
        {
            **_GOOGLE,
            "is_reasoning_model": True,
            "max_context_tokens": 1048576,
            "max_output_tokens": 65536,
        },
        "gemini-2.5-flash-lite",
    ),
    *_dotted(_THINKING, "gemini-2.5-flash"),
    FamilyRule("gemini-2.0", _GOOGLE, ("gemini-2.0", "gemini-2-0", "gemini-2-flash")),
    *_dotted({**_GOOGLE, "max_context_tokens": 2000000}, "gemini-1.5-pro"),
    *_dotted(_GOOGLE, "gemini-1.5-flash"),
    *_rules(_GOOGLE, "gemini"),
)

# Z.ai GLM ids called directly (Z.ai's GLM pages): JSON mode, no strict
# schema, no penalties. glm-5.3-flash:batch and glm-flash-latest are not
# documented and take the conservative profile.
_GLM: _Fields = {
    "supports_vision": True,
    "supports_json_mode": True,
    "is_reasoning_model": True,
}
_GLM_FLASH: _Fields = {
    **_GLM,
    "supports_reasoning_effort": True,
    "max_context_tokens": 1048576,
    "max_output_tokens": 131072,
}
_GLM_5V: _Fields = {**_GLM, "max_context_tokens": 202752, "max_output_tokens": 131072}
_GLM_46V: _Fields = {**_GLM, "max_context_tokens": 131072, "max_output_tokens": 32768}
_GLM_45V: _Fields = {**_GLM, "max_context_tokens": 65536, "max_output_tokens": 16384}

_GLM_RULES: tuple[FamilyRule, ...] = (
    *_rules(_GLM_FLASH, "glm-5.3-flashx"),
    FamilyRule(
        "glm-5.3-flash", _GLM_FLASH, ("glm-5.3-flash-",), exact=("glm-5.3-flash",)
    ),
    *_rules(_GLM_5V, "glm-5v-turbo"),
    *_rules(_GLM_46V, "glm-4.6v"),
    *_rules(_GLM_45V, "glm-4.5v"),
)

# OpenRouter: the parameters and modalities of its public model list. Image
# detail is not documented there, so routed OpenAI models never take original.
_ROUTED: _Fields = {
    "provider_name": "openrouter",
    "supports_vision": True,
    "supports_structured_output": True,
    "supports_json_mode": True,
}
_ROUTED_DETAIL: _Fields = {
    "supports_image_detail": True,
    "default_image_detail": "high",
}
_ROUTED_OPENAI: _Fields = {**_ROUTED_DETAIL, **_REASONING, **_NO_SAMPLING}
_ROUTED_GEMINI: _Fields = {
    "supports_reasoning_effort": True,
    "supports_media_resolution": True,
    "max_context_tokens": 1000000,
    "max_output_tokens": 8192,
}
_ROUTED_CLAUDE_NO_SAMPLING = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-5",
    "claude-fable-5",
    "claude-opus-4.8",
    "claude-opus-4-8",
    "claude-opus-4.7",
    "claude-opus-4-7",
    "claude-opus-4.6",
    "claude-opus-4-6",
    "claude-sonnet-4.6",
    "claude-sonnet-4-6",
)
"""Routed Claude models that reject samplers, as they do on Anthropic's API."""


def _has(*parts: str) -> Callable[[str], bool]:
    """Match ids that contain any of *parts*."""
    return lambda name: any(part in name for part in parts)


def _tail(part: str, vendor: str) -> Callable[[str], bool]:
    """Match *part* in the model segment of an id or *vendor* anywhere."""
    return lambda name: part in name.rsplit("/", 1)[-1] or vendor in name


def _routed(
    family: str, fields: _Fields, match: Callable[[str], bool] | None = None
) -> FamilyRule:
    """A rule for OpenRouter ``vendor/model`` ids that *match* selects."""

    def predicate(name: str) -> bool:
        return "/" in name and (match is None or match(name))

    return FamilyRule(
        family, {**_ROUTED, **fields}, provider=OPENROUTER, predicate=predicate
    )


def _mistral(name: str) -> bool:
    return "mistral" in name.rsplit("/", 1)[-1] or "mixtral" in name


_CLAUDE_TAIL = _tail("claude", "anthropic/")
_GEMINI_TAIL = _tail("gemini", "google/")
_LLAMA_TAIL = _tail("llama", "meta/")
_ROUTED_O_SERIES = _has("/o1", "/o3", "/o4")

_NO_SCHEMA: _Fields = {"supports_structured_output": False}

_OPENROUTER_RULES: tuple[FamilyRule, ...] = (
    _routed("glm-5.3-flashx", {**_GLM_FLASH, **_NO_SCHEMA}, _has("glm-5.3-flashx")),
    _routed(
        "glm-5.3-flash",
        {**_GLM_FLASH, **_PENALTIES},
        lambda name: "glm-5.3-flash" in name and ":batch" not in name,
    ),
    _routed("glm-5v-turbo", {**_GLM_5V, **_NO_SCHEMA}, _has("glm-5v-turbo")),
    _routed("glm-4.6v", {**_GLM_46V, **_PENALTIES, **_NO_SCHEMA}, _has("glm-4.6v")),
    _routed("glm-4.5v", {**_GLM_45V, **_PENALTIES, **_NO_SCHEMA}, _has("glm-4.5v")),
    _routed("openrouter-deepseek", _REASONING, _has("deepseek-r1")),
    _routed(
        "openrouter-deepseek",
        _REASONING,
        lambda name: "deepseek" in name and "terminus" in name,
    ),
    _routed(
        "openrouter-deepseek", {"supports_reasoning_effort": True}, _has("deepseek")
    ),
    _routed(
        "openrouter-gpt-oss",
        {**_ROUTED_DETAIL, **_REASONING, **_PENALTIES},
        _has("gpt-oss"),
    ),
    _routed("openrouter-gpt6", {**_REASONING, **_NO_SAMPLING}, _has("/gpt-6")),
    _routed(
        "openrouter-gpt5",
        _ROUTED_OPENAI,
        lambda name: "gpt-5" in name and "gpt-5.3-chat" not in name,
    ),
    _routed(
        "openrouter-o-series",
        {**_ROUTED_OPENAI, **_TEXT_ONLY},
        _has("/o1-mini", "/o3-mini"),
    ),
    _routed("openrouter-o-series", _ROUTED_OPENAI, _ROUTED_O_SERIES),
    _routed("openrouter-gpt-4o", _PENALTIES, _has("/gpt-4o")),
    _routed(
        "openrouter-claude",
        {**_REASONING, **_NO_SAMPLING, "max_context_tokens": 200000},
        lambda name: _CLAUDE_TAIL(name) and _has(*_ROUTED_CLAUDE_NO_SAMPLING)(name),
    ),
    _routed(
        "openrouter-claude", {**_REASONING, "max_context_tokens": 200000}, _CLAUDE_TAIL
    ),
    _routed(
        "openrouter-gemini",
        {**_ROUTED_GEMINI, "is_reasoning_model": True},
        lambda name: (
            _GEMINI_TAIL(name) and _has("gemini-2.5", "gemini-3", "gemini-2-5")(name)
        ),
    ),
    _routed("openrouter-gemini", _ROUTED_GEMINI, _GEMINI_TAIL),
    _routed(
        "openrouter-llama",
        _PENALTIES,
        lambda name: _LLAMA_TAIL(name) and _has("vision", "llama-3.2")(name),
    ),
    _routed("openrouter-llama", {**_PENALTIES, "supports_vision": False}, _LLAMA_TAIL),
    _routed(
        "openrouter-mistral", {}, lambda name: _mistral(name) and "pixtral" in name
    ),
    _routed("openrouter-mistral", {"supports_vision": False}, _mistral),
    _routed(
        "openrouter-qwen",
        {"max_context_tokens": 262144, "max_output_tokens": 32768},
        _has("qwen3-vl-235b-a22b-instruct"),
    ),
    _routed(
        "openrouter-qwen", {**_REASONING, "max_context_tokens": 131072}, _has("qwen")
    ),
    _routed("openrouter", {}),
)

FAMILIES: tuple[FamilyRule, ...] = (
    *_OPENAI_RULES,
    *_ANTHROPIC_RULES,
    *_GOOGLE_RULES,
    *_GLM_RULES,
    *_OPENROUTER_RULES,
)
"""The shared family rules, in match order."""

PROVIDERS: tuple[ProviderRule, ...] = (
    ProviderRule("anthropic", prefixes=("claude",), contains=("anthropic",)),
    ProviderRule("google", prefixes=("gemini", "gemma"), contains=("google",)),
    ProviderRule(
        "openai", prefixes=("gpt", "o1", "o3", "o4", "chatgpt", "text-", "whisper")
    ),
    ProviderRule(
        "openrouter", contains=("llama", "mistral", "mixtral", "qwen", "deepseek")
    ),
)
"""Provider detection for ids without a ``provider:`` prefix or a vendor."""

AUDIO_FAMILIES: frozenset[str] = frozenset(
    {
        "openai-audio",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash-lite",
        "gemini-3-flash",
        "gemini-3-pro",
        "gemini-3-preview",
        "gemini-3",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini",
    }
)
"""Families whose models take audio input; the Gemini image models do not."""

SPEECH_TO_TEXT_FAMILIES: frozenset[str] = frozenset({"openai-audio"})
"""Families served only by OpenAI's transcription endpoint."""


@dataclass(frozen=True)
class TableExtension:
    """A tool's additions to the shared table.

    *rules* are matched before the shared rules. *family_extras* maps a
    family name to keys merged into the ``extras`` of every shared rule of
    that family.
    """

    rules: tuple[FamilyRule, ...] = ()
    family_extras: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _with_extras(rule: FamilyRule, extras: Mapping[str, Any] | None) -> FamilyRule:
    if not extras:
        return rule
    merged = {**dict(rule.fields.get("extras") or {}), **extras}
    return replace(rule, fields={**rule.fields, "extras": merged})


def build_table(extension: TableExtension | None = None) -> CapabilityTable:
    """Return the shared table with *extension* applied.

    Raises:
        ValueError: *extension* names a family no shared rule has.
    """
    if extension is None:
        return TABLE
    known = {rule.family for rule in FAMILIES}
    unknown = sorted(set(extension.family_extras) - known)
    if unknown:
        raise ValueError(f"Unknown capability families: {', '.join(unknown)}")
    families = tuple(
        _with_extras(rule, extension.family_extras.get(rule.family))
        for rule in FAMILIES
    )
    return CapabilityTable(families=(*extension.rules, *families), providers=PROVIDERS)


TABLE = CapabilityTable(families=FAMILIES, providers=PROVIDERS)
"""The shared table without a tool's extension."""
