"""Typed settings: machine values and optional run defaults.

``config/settings.yaml`` (gitignored) holds this machine's values; the
tracked ``config/settings.example.yaml`` is read when it is missing. Every
key is optional and falls back to the code value. The ``defaults`` block sets
run options by option name (see :mod:`autoexcerpter.spec`); a flag overrides
it, and it overrides the code default. Further top-level blocks are accepted
only when the caller passes a parser for them; :attr:`Settings.extensions`
holds their parsed values.

The shared blocks (``api_keys``, ``endpoints``, ``timeouts``,
``rate_limits``, ``images``, ``state_dir``) are read by
:mod:`autoexcerpter.common.settings_schema`; ``retry`` and ``openalex`` are
this tool's own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from autoexcerpter.common.retry import DEFAULT_BACKOFF_MULTIPLIERS, RetryPolicy
from autoexcerpter.common.settings_loader import SettingsError, load_settings
from autoexcerpter.common.settings_schema import (
    DEFAULT_API_KEYS,
    BlockParser,
    Endpoint,
    Images,
    Reader,
    Timeouts,
    check_blocks,
    read_api_keys,
    read_endpoints,
    read_extensions,
    read_images,
    read_rate_limits,
    read_state_dir,
    read_timeouts,
    state_directory,
)
from autoexcerpter.spec import OPTIONS, Resolution, SpecError, resolve_spec
from autoexcerpter.state import DEFAULT_DIR_NAME

__all__ = [
    "BlockParser",
    "DEFAULT_API_KEYS",
    "DEFAULT_SETTINGS_PATH",
    "EXAMPLE_SETTINGS_PATH",
    "Endpoint",
    "Images",
    "OpenAlexSettings",
    "RetryRule",
    "RetrySettings",
    "Settings",
    "SettingsError",
    "SpecError",
    "Timeouts",
    "load",
    "resolve",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
EXAMPLE_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.example.yaml"

RATE_LIMITS: tuple[tuple[int, int], ...] = ((10, 1), (600, 60), (600, 3600))
"""The code default of ``rate_limits``: ``[requests, seconds]`` windows."""


@dataclass(frozen=True)
class RetryRule:
    """Retries after a validation failure or a flag in the model output."""

    enabled: bool
    max_attempts: int
    backoff_base: float
    backoff_multiplier: float


def _schema_retries() -> Mapping[str, Mapping[str, RetryRule]]:
    return MappingProxyType(
        {
            "transcription": MappingProxyType(
                {
                    "validation_failure": RetryRule(True, 3, 0.5, 1.5),
                    "no_transcribable_text": RetryRule(False, 0, 0.1, 1.5),
                    "transcription_not_possible": RetryRule(True, 3, 0.1, 1.5),
                }
            ),
            "summary": MappingProxyType(
                {"validation_failure": RetryRule(True, 3, 0.5, 1.5)}
            ),
        }
    )


@dataclass(frozen=True)
class RetrySettings:
    """The transient-error ladder and the per-phase output retries."""

    max_attempts: int = 8
    backoff_base: float = 0.5
    backoff_cap: float = 120.0
    backoff_multipliers: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_BACKOFF_MULTIPLIERS
    )
    jitter_min: float = 0.5
    jitter_max: float = 1.0
    schema_retries: Mapping[str, Mapping[str, RetryRule]] = field(
        default_factory=_schema_retries
    )

    def policy(self) -> RetryPolicy:
        """Return the retry policy of the transient-error ladder."""
        return RetryPolicy(
            max_attempts=self.max_attempts,
            backoff_base=self.backoff_base,
            backoff_cap=self.backoff_cap,
            multipliers=self.backoff_multipliers,
            jitter_min=self.jitter_min,
            jitter_max=self.jitter_max,
        )


@dataclass(frozen=True)
class OpenAlexSettings:
    """OpenAlex contact and request limit per document."""

    email: str = ""
    api_key_env: str = "OPENALEX_API_KEY"
    max_requests: int = 300


@dataclass(frozen=True)
class Settings:
    """Machine values, the optional run defaults and any extra blocks.

    *extensions* maps the name of each extra top-level block in the file to
    the value its parser returned.
    """

    api_keys: Mapping[str, str] = field(default_factory=lambda: DEFAULT_API_KEYS)
    endpoints: Mapping[str, Endpoint] = field(
        default_factory=lambda: MappingProxyType({})
    )
    timeouts: Timeouts = field(default_factory=Timeouts)
    rate_limits: tuple[tuple[int, int], ...] = RATE_LIMITS
    retry: RetrySettings = field(default_factory=RetrySettings)
    state_dir: Path | None = None
    openalex: OpenAlexSettings = field(default_factory=OpenAlexSettings)
    images: Images = field(default_factory=Images)
    defaults: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    source: Path | None = None
    extensions: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def api_key_env(self, provider: str, endpoint: str | None = None) -> str:
        """Return the environment variable that holds the key of a provider.

        Raises:
            SettingsError: An unknown endpoint, or a provider without one.
        """
        if endpoint is not None:
            return self.endpoint(endpoint).api_key_env
        try:
            return self.api_keys[provider]
        except KeyError:
            raise SettingsError(
                f"no API key variable for provider {provider!r}"
            ) from None

    def endpoint(self, name: str) -> Endpoint:
        """Return a named endpoint.

        Raises:
            SettingsError: No endpoint of that name.
        """
        try:
            return self.endpoints[name]
        except KeyError:
            known = ", ".join(sorted(self.endpoints)) or "none"
            raise SettingsError(
                f"unknown endpoint {name!r}; the settings define: {known}"
            ) from None

    def state_directory(self) -> Path:
        """Return the state directory (not created here)."""
        return state_directory(self.state_dir, DEFAULT_DIR_NAME)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
        source: Path | None = None,
        blocks: Mapping[str, BlockParser] | None = None,
    ) -> Settings:
        """Build settings from parsed YAML; missing keys keep code values.

        *blocks* names the extra top-level blocks to accept, each with its
        parser; a block present in *data* is parsed into :attr:`extensions`.

        Raises:
            SettingsError: An unknown key, a value of the wrong type, or a
                block its parser rejects.
            ValueError: A block name is a built-in key.
        """
        blocks = dict(blocks or {})
        check_blocks(_TOP_KEYS, blocks)
        where = str(source) if source is not None else "settings"
        reader = Reader(where)
        reader.keys(data, _TOP_KEYS | frozenset(blocks), "")
        state_dir = read_state_dir(reader, data)
        return cls(
            api_keys=read_api_keys(reader, data.get("api_keys")),
            endpoints=read_endpoints(reader, data.get("endpoints")),
            timeouts=read_timeouts(reader, data.get("timeouts")),
            rate_limits=read_rate_limits(reader, data.get("rate_limits"), RATE_LIMITS),
            retry=_retry(reader, data.get("retry")),
            state_dir=state_dir,
            openalex=_openalex(reader, data.get("openalex")),
            images=read_images(reader, data.get("images")),
            defaults=MappingProxyType(dict(defaults or {})),
            source=source,
            extensions=read_extensions(reader, data, blocks),
        )


_TOP_KEYS = frozenset(
    {
        "api_keys",
        "endpoints",
        "timeouts",
        "rate_limits",
        "retry",
        "state_dir",
        "openalex",
        "images",
    }
)

_RETRY_KEYS = frozenset(
    {
        "max_attempts",
        "backoff_base",
        "backoff_cap",
        "backoff_multipliers",
        "jitter",
        "schema_retries",
    }
)
_RULE_KEYS = frozenset(
    {"enabled", "max_attempts", "backoff_base", "backoff_multiplier"}
)


def _retry(reader: Reader, raw: Any) -> RetrySettings:
    section = reader.section(raw, "retry", _RETRY_KEYS)
    base = RetrySettings()
    multipliers = dict(base.backoff_multipliers)
    raw_multipliers = reader.section(
        section.get("backoff_multipliers"),
        "retry.backoff_multipliers",
        frozenset(multipliers),
    )
    for kind in raw_multipliers:
        multipliers[kind] = reader.number(
            raw_multipliers, kind, multipliers[kind], "retry.backoff_multipliers."
        )
    jitter = reader.section(
        section.get("jitter"), "retry.jitter", frozenset({"min", "max"})
    )
    return RetrySettings(
        max_attempts=reader.positive(
            section, "max_attempts", base.max_attempts, "retry."
        ),
        backoff_base=reader.number(
            section, "backoff_base", base.backoff_base, "retry."
        ),
        backoff_cap=reader.number(section, "backoff_cap", base.backoff_cap, "retry."),
        backoff_multipliers=MappingProxyType(multipliers),
        jitter_min=reader.number(jitter, "min", base.jitter_min, "retry.jitter."),
        jitter_max=reader.number(jitter, "max", base.jitter_max, "retry.jitter."),
        schema_retries=_schema_rules(
            reader, section.get("schema_retries"), base.schema_retries
        ),
    )


def _schema_rules(
    reader: Reader, raw: Any, base: Mapping[str, Mapping[str, RetryRule]]
) -> Mapping[str, Mapping[str, RetryRule]]:
    section = reader.section(raw, "retry.schema_retries", frozenset(base))
    phases: dict[str, Mapping[str, RetryRule]] = {}
    for phase, rules in base.items():
        path = f"retry.schema_retries.{phase}"
        raw_rules = reader.section(section.get(phase), path, frozenset(rules))
        merged: dict[str, RetryRule] = {}
        for name, rule in rules.items():
            entry = reader.section(raw_rules.get(name), f"{path}.{name}", _RULE_KEYS)
            prefix = f"{path}.{name}."
            merged[name] = RetryRule(
                enabled=reader.flag(entry, "enabled", rule.enabled, prefix),
                max_attempts=reader.integer(
                    entry, "max_attempts", rule.max_attempts, prefix
                ),
                backoff_base=reader.number(
                    entry, "backoff_base", rule.backoff_base, prefix
                ),
                backoff_multiplier=reader.number(
                    entry, "backoff_multiplier", rule.backoff_multiplier, prefix
                ),
            )
        phases[phase] = MappingProxyType(merged)
    return MappingProxyType(phases)


def _openalex(reader: Reader, raw: Any) -> OpenAlexSettings:
    section = reader.section(
        raw, "openalex", frozenset({"email", "api_key_env", "max_requests"})
    )
    base = OpenAlexSettings()
    return OpenAlexSettings(
        email=reader.text(section, "email", base.email, "openalex."),
        api_key_env=reader.text(section, "api_key_env", base.api_key_env, "openalex."),
        max_requests=reader.integer(
            section, "max_requests", base.max_requests, "openalex."
        ),
    )


def load(
    path: Path | None = None, *, blocks: Mapping[str, BlockParser] | None = None
) -> Settings:
    """Load the settings file; an explicit ``path`` (``--settings``) wins.

    *blocks* maps extra top-level block names to their parsers (see
    :meth:`Settings.from_mapping`).

    Raises:
        SettingsError: The file is missing or invalid.
    """
    loaded = load_settings(
        DEFAULT_SETTINGS_PATH,
        example=EXAMPLE_SETTINGS_PATH,
        explicit=path,
        options=OPTIONS,
    )
    return Settings.from_mapping(
        loaded.data, defaults=loaded.defaults, source=loaded.path, blocks=blocks
    )


def resolve(settings: Settings, flags: Mapping[str, Any]) -> Resolution:
    """Return the effective spec and the source of each value.

    Precedence: flag, then settings default, then code default. Named
    endpoints must exist in the settings.

    Raises:
        SpecError: The options do not form a valid spec.
        SettingsError: A named endpoint is not defined.
    """
    resolution = resolve_spec(flags, settings.defaults)
    for phase in ("transcription", "summary"):
        endpoint = resolution.spec.model(phase).endpoint
        if endpoint is not None:
            settings.endpoint(endpoint)
    return resolution
