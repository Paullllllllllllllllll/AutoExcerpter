"""The capability table reproduces the recorded capability of every model id.

``data/capability_expectations.json`` records the shared table of
``common/capability_table.py`` (regenerated when the tables of the three
tools merged and the vendor facts were settled). It covers every family
prefix, dated variants, and unknown ids: bare, ``provider:``-prefixed,
``vendor/model`` and ``models/gemini-*``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.common.capabilities import split_model_id
from autoexcerpter.llm.capabilities import (
    TABLE,
    detect_capabilities,
    detect_provider,
)

_DATA = Path(__file__).parent / "data" / "capability_expectations.json"
_ROWS: list[dict[str, Any]] = json.loads(_DATA.read_text(encoding="utf-8"))


def _recorded(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


@pytest.mark.parametrize("row", _ROWS, ids=[repr(row["model"]) for row in _ROWS])
def test_family_table_matches_recorded_capabilities(row: dict[str, Any]) -> None:
    expected = dict(row)
    model = expected.pop("model")
    provider = expected.pop("provider")
    extras = expected.pop("extras", {})
    caps = detect_capabilities(model)

    assert detect_provider(model) == provider
    assert caps.model_name == model
    assert {name: _recorded(getattr(caps, name)) for name in expected} == expected
    assert dict(caps.extras) == extras


def test_expectations_cover_every_kind_of_id() -> None:
    models = [row["model"] for row in _ROWS]

    assert any(m.startswith("openai:") for m in models)
    assert any(m.startswith("models/gemini") for m in models)
    assert sum("/" in m for m in models) > 20
    assert any(row["family"] == "unknown" for row in _ROWS)


def test_ids_lose_their_prefixes() -> None:
    assert split_model_id("OpenAI:GPT-5", TABLE) == ("openai", "gpt-5")
    assert split_model_id("models/gemini-2.5-flash", TABLE) == (
        "google",
        "gemini-2.5-flash",
    )


def test_unknown_model_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        detect_capabilities("never-seen-family-model-x9")
        detect_capabilities("never-seen-family-model-x9")

    hits = [r for r in caplog.records if "never-seen-family-model-x9" in r.message]
    assert len(hits) == 1
