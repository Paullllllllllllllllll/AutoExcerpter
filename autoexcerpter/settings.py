"""Typed settings: machine values and optional run defaults.

``config/settings.yaml`` (gitignored) holds this machine's values; the
tracked ``config/settings.example.yaml`` is read when it is missing. Every
key is optional and falls back to the code value. The ``defaults`` block sets
run options by option name (see :mod:`autoexcerpter.spec`); a flag overrides
it, and it overrides the code default. Further top-level blocks are accepted
only when the caller passes a parser for them; :attr:`Settings.extensions`
holds their parsed values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from autoexcerpter.common.retry import DEFAULT_BACKOFF_MULTIPLIERS, RetryPolicy
from autoexcerpter.common.settings_loader import SettingsError, load_settings
from autoexcerpter.spec import OPTIONS, Resolution, SpecError, resolve_spec
from autoexcerpter.state import DEFAULT_DIR_NAME

__all__ = [
    "BlockParser",
    "DEFAULT_API_KEYS",
    "DEFAULT_SETTINGS_PATH",
    "EXAMPLE_SETTINGS_PATH",
    "Endpoint",
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

DEFAULT_API_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
)

type BlockParser = Callable[[Any], Any]
"""Turns the parsed YAML value of an extra top-level block into its setting."""


@dataclass(frozen=True)
class Endpoint:
    """A named OpenAI-compatible endpoint."""

    name: str
    base_url: str
    api_key_env: str
    supports_vision: bool = True
    supports_schema: bool = False


@dataclass(frozen=True)
class Timeouts:
    """HTTP timeouts in seconds; ``request`` is the read timeout."""

    request: float = 900.0
    connect: float = 10.0
    write: float = 30.0
    pool: float = 30.0


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
    rate_limits: tuple[tuple[int, int], ...] = ((10, 1), (600, 60), (600, 3600))
    retry: RetrySettings = field(default_factory=RetrySettings)
    state_dir: Path | None = None
    openalex: OpenAlexSettings = field(default_factory=OpenAlexSettings)
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
        if self.state_dir is not None:
            return self.state_dir.expanduser()
        return Path.home() / DEFAULT_DIR_NAME

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
        clashes = sorted(_TOP_KEYS.intersection(blocks))
        if clashes:
            raise ValueError(f"built-in settings keys: {', '.join(clashes)}")
        where = str(source) if source is not None else "settings"
        reader = _Reader(where)
        reader.keys(data, _TOP_KEYS | frozenset(blocks), "")
        base = cls()
        state_dir = reader.text(data, "state_dir", "", "")
        return cls(
            api_keys=_api_keys(reader, data.get("api_keys")),
            endpoints=_endpoints(reader, data.get("endpoints")),
            timeouts=_timeouts(reader, data.get("timeouts")),
            rate_limits=_rate_limits(reader, data.get("rate_limits"), base),
            retry=_retry(reader, data.get("retry")),
            state_dir=Path(state_dir) if state_dir else None,
            openalex=_openalex(reader, data.get("openalex")),
            defaults=MappingProxyType(dict(defaults or {})),
            source=source,
            extensions=_extensions(reader, data, blocks),
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
    }
)


class _Reader:
    """Typed reads from a parsed settings mapping with errors naming the key."""

    def __init__(self, where: str) -> None:
        self.where = where

    def fail(self, path: str, message: str) -> SettingsError:
        return SettingsError(f"{self.where}: {path}: {message}")

    def section(self, raw: Any, path: str) -> Mapping[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise self.fail(path, "must be a mapping")
        return raw

    def keys(self, raw: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise self.fail(path or "top level", f"unknown keys {', '.join(unknown)}")

    def text(self, raw: Mapping[str, Any], key: str, default: str, path: str) -> str:
        value = raw.get(key)
        if value is None:
            return default
        if not isinstance(value, str):
            raise self.fail(path + key, "must be a string")
        return value.strip()

    def number(
        self, raw: Mapping[str, Any], key: str, default: float, path: str
    ) -> float:
        value = raw.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self.fail(path + key, "must be a number")
        if value < 0:
            raise self.fail(path + key, "must not be negative")
        return float(value)

    def integer(self, raw: Mapping[str, Any], key: str, default: int, path: str) -> int:
        value = raw.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise self.fail(path + key, "must be a non-negative integer")
        return value

    def flag(self, raw: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
        value = raw.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise self.fail(path + key, "must be true or false")
        return value


def _api_keys(reader: _Reader, raw: Any) -> Mapping[str, str]:
    section = reader.section(raw, "api_keys")
    reader.keys(section, frozenset(DEFAULT_API_KEYS), "api_keys")
    return MappingProxyType(
        {
            provider: reader.text(section, provider, default, "api_keys.") or default
            for provider, default in DEFAULT_API_KEYS.items()
        }
    )


_ENDPOINT_KEYS = frozenset(
    {"base_url", "api_key_env", "supports_vision", "supports_schema"}
)


def _endpoints(reader: _Reader, raw: Any) -> Mapping[str, Endpoint]:
    section = reader.section(raw, "endpoints")
    endpoints: dict[str, Endpoint] = {}
    for name, value in section.items():
        path = f"endpoints.{name}."
        entry = reader.section(value, path.rstrip("."))
        reader.keys(entry, _ENDPOINT_KEYS, path.rstrip("."))
        base_url = reader.text(entry, "base_url", "", path)
        key_env = reader.text(entry, "api_key_env", "", path)
        if not base_url or not key_env:
            raise reader.fail(path.rstrip("."), "needs base_url and api_key_env")
        endpoints[str(name)] = Endpoint(
            name=str(name),
            base_url=base_url,
            api_key_env=key_env,
            supports_vision=reader.flag(entry, "supports_vision", True, path),
            supports_schema=reader.flag(entry, "supports_schema", False, path),
        )
    return MappingProxyType(endpoints)


def _timeouts(reader: _Reader, raw: Any) -> Timeouts:
    section = reader.section(raw, "timeouts")
    reader.keys(section, frozenset({"request", "connect", "write", "pool"}), "timeouts")
    base = Timeouts()
    return Timeouts(
        *(
            reader.number(section, name, getattr(base, name), "timeouts.")
            for name in ("request", "connect", "write", "pool")
        )
    )


def _rate_limits(
    reader: _Reader, raw: Any, base: Settings
) -> tuple[tuple[int, int], ...]:
    if raw is None:
        return base.rate_limits
    if not isinstance(raw, list) or not raw:
        raise reader.fail("rate_limits", "must be a list of [requests, seconds]")
    limits: list[tuple[int, int]] = []
    for item in raw:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in item)
            or min(item) <= 0
        ):
            raise reader.fail("rate_limits", f"invalid entry {item!r}")
        limits.append((item[0], item[1]))
    return tuple(limits)


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


def _retry(reader: _Reader, raw: Any) -> RetrySettings:
    section = reader.section(raw, "retry")
    reader.keys(section, _RETRY_KEYS, "retry")
    base = RetrySettings()
    multipliers = dict(base.backoff_multipliers)
    raw_multipliers = reader.section(
        section.get("backoff_multipliers"), "retry.backoff_multipliers"
    )
    reader.keys(raw_multipliers, frozenset(multipliers), "retry.backoff_multipliers")
    for kind in raw_multipliers:
        multipliers[kind] = reader.number(
            raw_multipliers, kind, multipliers[kind], "retry.backoff_multipliers."
        )
    jitter = reader.section(section.get("jitter"), "retry.jitter")
    reader.keys(jitter, frozenset({"min", "max"}), "retry.jitter")
    max_attempts = reader.integer(section, "max_attempts", base.max_attempts, "retry.")
    if max_attempts < 1:
        raise reader.fail("retry.max_attempts", "must be at least 1")
    return RetrySettings(
        max_attempts=max_attempts,
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
    reader: _Reader, raw: Any, base: Mapping[str, Mapping[str, RetryRule]]
) -> Mapping[str, Mapping[str, RetryRule]]:
    section = reader.section(raw, "retry.schema_retries")
    reader.keys(section, frozenset(base), "retry.schema_retries")
    phases: dict[str, Mapping[str, RetryRule]] = {}
    for phase, rules in base.items():
        path = f"retry.schema_retries.{phase}"
        raw_rules = reader.section(section.get(phase), path)
        reader.keys(raw_rules, frozenset(rules), path)
        merged: dict[str, RetryRule] = {}
        for name, rule in rules.items():
            entry = reader.section(raw_rules.get(name), f"{path}.{name}")
            reader.keys(entry, _RULE_KEYS, f"{path}.{name}")
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


def _openalex(reader: _Reader, raw: Any) -> OpenAlexSettings:
    section = reader.section(raw, "openalex")
    reader.keys(
        section, frozenset({"email", "api_key_env", "max_requests"}), "openalex"
    )
    base = OpenAlexSettings()
    return OpenAlexSettings(
        email=reader.text(section, "email", base.email, "openalex."),
        api_key_env=reader.text(section, "api_key_env", base.api_key_env, "openalex."),
        max_requests=reader.integer(
            section, "max_requests", base.max_requests, "openalex."
        ),
    )


def _extensions(
    reader: _Reader, data: Mapping[str, Any], blocks: Mapping[str, BlockParser]
) -> Mapping[str, Any]:
    parsed: dict[str, Any] = {}
    for name, parser in blocks.items():
        if name not in data:
            continue
        try:
            parsed[name] = parser(data[name])
        except Exception as exc:
            raise reader.fail(name, str(exc) or type(exc).__name__) from exc
    return MappingProxyType(parsed)


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
