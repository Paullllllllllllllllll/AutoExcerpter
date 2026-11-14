"""The units a run writes and the schema check that refuses mixing two schemas."""

from __future__ import annotations

from autoexcerpter.common.selection import parse_page_range
from autoexcerpter.output.store import (
    KIND_SUMMARY,
    KIND_TRANSCRIPTION,
    ResumeRequest,
    changed_schema,
    run_writes,
    schema_record,
)

OLD, NEW = "a" * 64, "b" * 64
DONE = ("done", 0)


def _units(
    transcriptions: dict[int, tuple[str, int]],
    summaries: dict[int, tuple[str, int]] | None = None,
) -> dict[str, dict[int, tuple[str, int]]]:
    units = {KIND_TRANSCRIPTION: transcriptions}
    if summaries is not None:
        units[KIND_SUMMARY] = summaries
    return units


def test_a_resume_writes_the_pages_left_and_their_summaries() -> None:
    units = _units(
        {0: DONE, 1: ("failed", 0), 2: DONE},
        {0: DONE, 1: DONE, 2: ("done", 1)},
    )

    writes = run_writes(units, summarize=True)

    assert writes == {KIND_TRANSCRIPTION: {1}, KIND_SUMMARY: {1, 2}}
    assert run_writes(units, summarize=False)[KIND_SUMMARY] == frozenset()


def test_pages_and_retranscribe_narrow_and_widen_the_writes() -> None:
    units = _units({0: DONE, 1: DONE, 2: ("pending", 0)}, {0: DONE, 1: DONE})
    pages = ResumeRequest(pages=parse_page_range("1-2"), retranscribe=True)

    assert pages.writes(units) == {
        KIND_TRANSCRIPTION: {0, 1},
        KIND_SUMMARY: {0, 1},
    }
    every = ResumeRequest(retranscribe=True)
    assert every.writes(units)[KIND_TRANSCRIPTION] == {0, 1, 2}


def test_a_kept_done_unit_of_another_schema_mixes_its_kind() -> None:
    units = _units({0: DONE, 1: ("failed", 0)}, {0: DONE, 1: ("pending", 0)})
    known = {KIND_TRANSCRIPTION: {0: OLD}, KIND_SUMMARY: {0: OLD}}
    writes = run_writes(units, summarize=True)

    current = {KIND_TRANSCRIPTION: NEW, KIND_SUMMARY: OLD}
    assert changed_schema(units, known, current, writes) == KIND_TRANSCRIPTION
    current = {KIND_TRANSCRIPTION: OLD, KIND_SUMMARY: NEW}
    assert changed_schema(units, known, current, writes) == KIND_SUMMARY
    current = {KIND_TRANSCRIPTION: OLD, KIND_SUMMARY: OLD}
    assert changed_schema(units, known, current, writes) is None


def test_unknown_schemas_and_unused_schemas_never_mix() -> None:
    units = _units({0: DONE, 1: ("failed", 0)})
    writes = run_writes(units, summarize=False)

    assert changed_schema(units, {}, {KIND_TRANSCRIPTION: NEW}, writes) is None
    known = {KIND_TRANSCRIPTION: {0: OLD}}
    assert changed_schema(units, known, {KIND_TRANSCRIPTION: None}, writes) is None


def test_a_rewritten_unit_does_not_count_as_kept() -> None:
    units = _units({0: DONE, 1: DONE})
    known = {KIND_TRANSCRIPTION: {0: OLD, 1: OLD}}
    writes = {KIND_TRANSCRIPTION: frozenset({0, 1})}

    assert changed_schema(units, known, {KIND_TRANSCRIPTION: NEW}, writes) is None


def test_the_record_holds_an_entry_per_schema_used() -> None:
    assert schema_record({KIND_TRANSCRIPTION: OLD, KIND_SUMMARY: None}) == {
        "transcription_schema_sha256": {"value": OLD, "source": "computed"}
    }
