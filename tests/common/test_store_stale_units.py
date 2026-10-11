"""Store.set_units_stale flags units whose inputs changed.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_store_stale_units.py -q
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autoexcerpter.common.sqlite import read_rows
from autoexcerpter.common.store import (
    LEASE_TTL,
    LeaseLostError,
    PlannedUnit,
    Store,
)
from tests.common.store_child import FakeClock


def _stale(db: Path) -> dict[tuple[str, int], int]:
    return {(row["kind"], row["idx"]): row["stale"] for row in read_rows(db, "units")}


def test_set_units_stale_flags_only_the_given_units(tmp_path: Path) -> None:
    db = tmp_path / "tool.sqlite"
    clock = FakeClock()
    seen: list[dict[tuple[str, int], int]] = []

    async def main() -> tuple[int, int, int]:
        async with Store(db, "tool", clock=clock) as store:
            await store.begin_run("run")
            doc_id = await store.ensure_document("doc")
            lease = await store.acquire(doc_id)
            assert lease is not None
            planned = [PlannedUnit(idx) for idx in range(4)]
            for kind in ("page", "chunk"):
                await store.plan_units(
                    lease, kind, planned, model="model-a", provider="openai"
                )
            flagged = await store.set_units_stale(lease, "page", [1, 2, 9])
            seen.append(_stale(db))
            cleared = await store.set_units_stale(lease, "page", [1], stale=False)
            seen.append(_stale(db))
            clock.advance(LEASE_TTL + 1.0)
            with pytest.raises(LeaseLostError, match="expired"):
                await store.set_units_stale(lease, "page", [0])
            missing = await _other_kind(store)
            return flagged, cleared, missing

    async def _other_kind(store: Store) -> int:
        doc_id = await store.ensure_document("other")
        lease = await store.acquire(doc_id)
        assert lease is not None
        return await store.set_units_stale(lease, "page", [0])

    flagged, cleared, missing = asyncio.run(main())

    assert flagged == 2
    assert cleared == 1
    assert missing == 0
    expected = {(kind, idx): 0 for kind in ("page", "chunk") for idx in range(4)}
    assert seen[0] == {**expected, ("page", 1): 1, ("page", 2): 1}
    assert seen[1] == {**expected, ("page", 2): 1}
    assert _stale(db) == seen[1]
