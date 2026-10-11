"""Fixtures of the migrate tests: an OpenAlex cache and a guard against lookups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.rendering.citations import (
    LookupOutcome,
    LookupResult,
    OpenAlexCache,
    OpenAlexClient,
)
from autoexcerpter.rendering.citations.model import Citation
from tests.migrate.legacy import CITATION

METADATA = {"doi": "10.1234/cited", "url": "https://doi.org/10.1234/cited"}


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a state folder whose OpenAlex cache knows :data:`CITATION`.

    Any OpenAlex lookup fails the test: migrate takes metadata from the
    cache alone.
    """

    def lookup(self: OpenAlexClient, citation_text: str) -> Any:
        raise AssertionError(f"OpenAlex was asked about {citation_text!r}")

    monkeypatch.setattr(OpenAlexClient, "lookup", lookup)
    folder = tmp_path / "state"
    folder.mkdir()
    cache = OpenAlexCache(folder)
    key = Citation(raw_text=CITATION).normalized_key
    cache.record(key, LookupResult(LookupOutcome.FOUND, METADATA))
    cache.checkpoint()
    return folder
