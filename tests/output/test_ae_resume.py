"""Resume classification from the store, with the D22 image-settings cases.

Each case seeds an output folder through the store API (raw SQL only where a
case needs a state no run leaves: a live foreign lease, a stale output, a
newer schema), classifies the item and checks the state, the units to
process, the render and reset flags, the reason and the note. Every case also
checks that classification left the folder's bytes unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.common.store import (
    Lease,
    NewerSchemaError,
    Store,
    StoreError,
    UnitKey,
    UnreadableDatabaseError,
    VersionlessDatabaseError,
)
from autoexcerpter.imaging.settings import resolve_image_settings
from autoexcerpter.output.store import (
    DATABASE_NAME,
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    InputMismatchError,
    ItemState,
    ResumeRequest,
    classify,
    image_settings_record,
    open_store,
    write_summary,
)
from autoexcerpter.spec import ImageSpec
from tests.output.test_ae_store import (
    MODEL,
    NAME,
    SUMMARY_MODEL,
    Page,
    page,
    seed_document,
    summary,
)

HOLDER = "this-run"
OTHER_MODEL = "gpt-5.6-terra"
OTHER_SUMMARY_MODEL = "gemini-3.5-pro"
INPUT_SHA = "a" * 64


def resolved(provider: str, model: str, **images: Any) -> dict[str, Any]:
    """Return the image-settings record of *model* with *images* options."""
    settings, _, _ = resolve_image_settings(provider, model, ImageSpec(**images))
    return image_settings_record(settings)


SETTINGS = resolved("openai", MODEL)

After = Callable[[Store, Lease], Awaitable[None]]
Setup = Callable[[Path], None]


def transcription_file(root: Path) -> Path:
    return root / "doc_transcription.md"


def summary_file(root: Path) -> Path:
    return root / "doc_summary.md"


def _write_outputs(root: Path) -> None:
    for path in (transcription_file(root), summary_file(root)):
        path.write_text("rendered\n", encoding="utf-8")


def seeded(
    pages: Sequence[Page],
    *,
    summaries: Sequence[int] = (),
    summarize: bool = True,
    render: bool = True,
    after: After | None = None,
    sql: str | None = None,
) -> Setup:
    """Return a setup that stores *pages*, writes the outputs and runs *after*.

    The outputs are registered with their hashes, as a render records them.
    *sql* runs on the closed database, for states no run leaves.
    """

    def setup(root: Path) -> None:
        async def main() -> None:
            async with open_store(root) as store:
                await store.begin_run("run")
                lease = await seed_document(
                    store,
                    pages,
                    summaries=summaries,
                    summarize=summarize,
                    image_settings=SETTINGS,
                    input_path=str(root / "doc.pdf"),
                    input_sha256=INPUT_SHA,
                )
                if render:
                    _write_outputs(root)
                    for path in (transcription_file(root), summary_file(root)):
                        await store.register_output(lease, path, format="md")
                if after is not None:
                    await after(store, lease)
                await store.end_run(0)

        asyncio.run(main())
        if sql is not None:
            _execute(root, sql)

    return setup


def _execute(root: Path, sql: str) -> None:
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        connection.execute(sql)
        connection.commit()
    finally:
        connection.close()


def legacy_output(root: Path) -> None:
    transcription_file(root).write_text("Rendered by v1.0\n", encoding="utf-8")


def legacy_log(root: Path) -> None:
    log = root / "doc_autoexcerpter" / "transcription.jsonl"
    log.parent.mkdir()
    log.write_text('{"_format_version": 2}\n', encoding="utf-8")


def other_document_and_legacy_output(root: Path) -> None:
    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("run")
            await seed_document(store, [page(0)], name="other")

    asyncio.run(main())
    legacy_output(root)


def moved_input(root: Path) -> None:
    seeded(PENDING)(root)
    _execute(root, f"UPDATE documents SET input_path = '{root / 'gone.pdf'}'")


async def failed_summary(store: Store, lease: Lease) -> None:
    await write_summary(store, lease, summary(1, error="refused"), None)


async def open_submission(store: Store, lease: Lease) -> None:
    await store.open_submission(
        provider="openai",
        model=MODEL,
        units=[UnitKey(lease.doc_id, "transcription", 1)],
    )


def _lease(holder: str, offset: float) -> str:
    return (
        f"UPDATE documents SET lease_run_id = '{holder}', "
        f"lease_until = {time.time() + offset}, "
        "lease_generation = lease_generation + 1"
    )


ALL = (0, 1, 2)
DONE = [page(0), page(1), page(2)]
WITH_FAILED = [page(0), page(1, "failed"), page(2)]
PENDING = [page(0, "pending"), page(1), page(2), page(3, "pending")]


def keys(kind: str, *idxs: int) -> tuple[tuple[str, int], ...]:
    return tuple((kind, idx) for idx in idxs)


T = KIND_TRANSCRIPTION
S = KIND_SUMMARY


@dataclass(frozen=True)
class Case:
    """One classification: the setup, the request and what it must return."""

    setup: Setup | None
    state: ItemState
    units: tuple[tuple[str, int], ...] = ()
    render: bool = False
    reset: bool = False
    request: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    note: tuple[str, ...] = ()


CASES = {
    "no_database": Case(None, ItemState.NONE),
    "all_done": Case(seeded(DONE, summaries=ALL), ItemState.COMPLETE),
    "all_done_without_summaries": Case(
        seeded(DONE, summarize=False), ItemState.COMPLETE,
        request={"summarize": False},
    ),
    "failed_transcription_retried": Case(
        seeded(WITH_FAILED, summaries=ALL), ItemState.PARTIAL,
        keys(T, 1) + keys(S, 1),
    ),
    "failed_summary_retried": Case(
        seeded(DONE, summaries=(0, 2), after=failed_summary),
        ItemState.TRANSCRIPTION_ONLY, keys(S, 1),
    ),
    "pending_summary_made": Case(
        seeded(DONE, summaries=(0, 2)), ItemState.TRANSCRIPTION_ONLY, keys(S, 1),
    ),
    "pending_pages": Case(
        seeded(PENDING, summaries=(1, 2)), ItemState.PARTIAL,
        keys(T, 0, 3) + keys(S, 0, 3), reason="2 of 4 pages",
    ),
    "pending_pages_without_summaries": Case(
        seeded(PENDING, summaries=(1, 2)), ItemState.PARTIAL, keys(T, 0, 3),
        request={"summarize": False},
    ),
    "transcribed_then_summary_requested": Case(
        seeded(DONE, summarize=False), ItemState.TRANSCRIPTION_ONLY,
        keys(S, 0, 1, 2), reason="3 of 3 page summaries",
    ),
    "stale_summary_made_again": Case(
        seeded(DONE, summaries=ALL, sql=(
            "UPDATE units SET stale = 1 WHERE kind = 'summary' AND idx = 2"
        )),
        ItemState.TRANSCRIPTION_ONLY, keys(S, 2),
    ),
    "stale_summary_ignored_without_summaries": Case(
        seeded(DONE, summaries=ALL, sql=(
            "UPDATE units SET stale = 1 WHERE kind = 'summary' AND idx = 2"
        )),
        ItemState.COMPLETE, request={"summarize": False},
    ),
    "legacy_output_without_database": Case(
        legacy_output, ItemState.LEGACY, reason="migrate"
    ),
    "legacy_log_without_database": Case(legacy_log, ItemState.LEGACY, reason="migrate"),
    "legacy_output_without_row": Case(
        other_document_and_legacy_output, ItemState.LEGACY, reason="migrate"
    ),
    "missing_file_renders_again": Case(
        seeded(DONE, summaries=ALL, render=False), ItemState.COMPLETE, render=True
    ),
    "stale_output_renders_again": Case(
        seeded(DONE, summaries=ALL, sql="UPDATE outputs SET stale = 1"),
        ItemState.COMPLETE, render=True,
    ),
    "open_submission": Case(
        seeded(PENDING, after=open_submission), ItemState.BLOCKED, reason="batch"
    ),
    "leased_by_another_live_holder": Case(
        seeded(PENDING, sql=_lease("other-run", 600)), ItemState.BLOCKED,
        reason="other-run",
    ),
    "expired_foreign_lease": Case(
        seeded(PENDING, summarize=False, sql=_lease("other-run", -1)),
        ItemState.PARTIAL, keys(T, 0, 3), request={"summarize": False},
    ),
    "own_lease": Case(
        seeded(PENDING, summarize=False, sql=_lease(HOLDER, 600)),
        ItemState.PARTIAL, keys(T, 0, 3), request={"summarize": False},
    ),
    "same_input": Case(
        seeded(DONE, summaries=ALL), ItemState.COMPLETE,
        request={"input_sha256": INPUT_SHA},
    ),
    "changed_input_hash_resets": Case(
        seeded(DONE, summaries=ALL), ItemState.NONE, reset=True,
        request={"input_sha256": "b" * 64}, reason="input changed",
    ),
    "moved_input_resumes": Case(
        moved_input, ItemState.PARTIAL, keys(T, 0, 3) + keys(S, 0, 1, 2, 3),
    ),
}  # fmt: skip

# D22: the image settings are compared key by key without the model's identity
# and the keys derived from it; a model change alone resumes with a note.
D22 = {
    "same_settings": Case(
        seeded(PENDING), ItemState.PARTIAL, keys(T, 0, 3) + keys(S, 0, 1, 2, 3),
        request={"image_settings": SETTINGS, "model": MODEL},
    ),
    "changed_user_setting_refused": Case(
        seeded(PENDING), ItemState.REFUSED,
        request={"image_settings": resolved("openai", MODEL, dpi=200), "model": MODEL},
        reason="target_dpi",
    ),
    "changed_detail_same_model_refused": Case(
        seeded(PENDING), ItemState.REFUSED,
        request={
            "image_settings": resolved("openai", MODEL, detail="high"),
            "model": MODEL,
        },
        reason="llm_detail",
    ),
    "changed_setting_with_summaries_only_left": Case(
        seeded(DONE, summaries=(0, 1)), ItemState.TRANSCRIPTION_ONLY, keys(S, 2),
        request={"image_settings": resolved("openai", MODEL, dpi=200), "model": MODEL},
    ),
    "changed_setting_on_complete_item": Case(
        seeded(DONE, summaries=ALL), ItemState.COMPLETE,
        request={"image_settings": resolved("openai", MODEL, dpi=200), "model": MODEL},
    ),
    "other_openai_model_resumes_with_note": Case(
        seeded(PENDING), ItemState.PARTIAL, keys(T, 0, 3) + keys(S, 0, 1, 2, 3),
        request={
            "image_settings": resolved("openai", OTHER_MODEL),
            "model": OTHER_MODEL,
        },
        note=(MODEL, OTHER_MODEL),
    ),
    "other_provider_resumes_with_note": Case(
        seeded(PENDING), ItemState.PARTIAL, keys(T, 0, 3) + keys(S, 0, 1, 2, 3),
        request={
            "image_settings": resolved("openrouter", "z-ai/glm-5.3-flash"),
            "model": "z-ai/glm-5.3-flash",
        },
        note=(MODEL, "z-ai/glm-5.3-flash"),
    ),
    "other_model_with_changed_setting_refused": Case(
        seeded(PENDING), ItemState.REFUSED,
        request={
            "image_settings": resolved("openrouter", "z-ai/glm-5.3-flash", dpi=200),
            "model": "z-ai/glm-5.3-flash",
        },
        reason="target_dpi",
    ),
    "other_model_on_complete_item": Case(
        seeded(DONE, summaries=ALL), ItemState.COMPLETE,
        request={
            "image_settings": resolved("openai", OTHER_MODEL),
            "model": OTHER_MODEL,
        },
        note=(MODEL, OTHER_MODEL),
    ),
    "other_summary_model_resumes_with_note": Case(
        seeded(DONE, summaries=(0, 1)), ItemState.TRANSCRIPTION_ONLY, keys(S, 2),
        request={"model": MODEL, "summary_model": OTHER_SUMMARY_MODEL},
        note=(SUMMARY_MODEL, OTHER_SUMMARY_MODEL),
    ),
    "other_model_without_images": Case(
        seeded(PENDING, summarize=False), ItemState.PARTIAL, keys(T, 0, 3),
        request={"model": OTHER_MODEL, "summarize": False},
        note=(MODEL, OTHER_MODEL),
    ),
}  # fmt: skip


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _request(root: Path, options: dict[str, Any]) -> ResumeRequest:
    return ResumeRequest(
        outputs=(transcription_file(root), summary_file(root)),
        legacy=(root / "doc_autoexcerpter" / "transcription.jsonl",),
        input_path=root / "doc.pdf",
        **options,
    )


@pytest.mark.parametrize("name", [*CASES, *D22])
def test_classify(tmp_path: Path, name: str) -> None:
    case = CASES.get(name) or D22[name]
    if case.setup is not None:
        case.setup(tmp_path)
    before = _fingerprint(tmp_path)

    result = classify(tmp_path, NAME, _request(tmp_path, case.request), holder=HOLDER)

    assert result.state is case.state, result.reason
    assert tuple((unit.kind, unit.idx) for unit in result.units) == case.units
    assert result.render is case.render
    assert result.reset is case.reset
    if case.reason is not None:
        assert case.reason in result.reason
    if case.note:
        assert result.note is not None
        assert all(model in result.note for model in case.note), result.note
    else:
        assert result.note is None
    assert _fingerprint(tmp_path) == before


def test_indices_split_the_units_by_kind(tmp_path: Path) -> None:
    seeded(PENDING, summaries=(1, 2))(tmp_path)

    result = classify(tmp_path, NAME, _request(tmp_path, {}))

    assert result.indices(KIND_TRANSCRIPTION) == (0, 3)
    assert result.indices(KIND_SUMMARY) == (0, 3)
    assert result.doc_id is not None


def test_a_missing_file_is_listed_for_rendering(tmp_path: Path) -> None:
    seeded(DONE, summaries=ALL)(tmp_path)
    summary_file(tmp_path).unlink()

    result = classify(tmp_path, NAME, _request(tmp_path, {}))

    assert result.state is ItemState.COMPLETE
    assert result.missing == (summary_file(tmp_path),)


def test_another_existing_input_under_the_same_name_raises(tmp_path: Path) -> None:
    seeded(DONE, summaries=ALL)(tmp_path)
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.7 first")
    other = tmp_path / "elsewhere" / "doc.pdf"
    other.parent.mkdir()
    other.write_bytes(b"%PDF-1.7 second")
    request = ResumeRequest(input_path=other)

    with pytest.raises(InputMismatchError, match="belongs to"):
        classify(tmp_path, NAME, request)
    same = classify(tmp_path, NAME, ResumeRequest(input_path=tmp_path / "doc.pdf"))
    assert same.state is ItemState.COMPLETE


def _versionless(root: Path) -> None:
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY, path TEXT)")
        connection.execute("CREATE TABLE pages (document TEXT, page_index INTEGER)")
        connection.execute("INSERT INTO documents VALUES (?, 'doc.pdf')", (NAME,))
        connection.commit()
    finally:
        connection.close()


def _newer(root: Path) -> None:
    seeded(DONE)(root)
    _execute(root, "UPDATE meta SET version = 99 WHERE component = 'autoexcerpter'")


def _unreadable(root: Path) -> None:
    (root / DATABASE_NAME).write_bytes(b"not a database, " * 256)


@pytest.mark.parametrize(
    ("setup", "error"),
    [
        (_versionless, VersionlessDatabaseError),
        (_newer, NewerSchemaError),
        (_unreadable, UnreadableDatabaseError),
    ],
)
def test_refused_databases_raise_and_stay_unchanged(
    tmp_path: Path, setup: Setup, error: type[StoreError]
) -> None:
    setup(tmp_path)
    before = _fingerprint(tmp_path)

    with pytest.raises(error) as raised:
        classify(tmp_path, NAME, _request(tmp_path, {}))

    assert raised.value.exit_code == 2
    assert _fingerprint(tmp_path) == before
