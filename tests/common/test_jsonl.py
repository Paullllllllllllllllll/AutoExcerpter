"""Versioned JSONL files, the append writer and input-change detection."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter.common.jsonl import (
    JsonlFormat,
    JsonlFormatError,
    JsonlWriter,
    file_identity,
    folder_identity,
    input_changed,
    parse_jsonl,
    read_header,
    read_jsonl,
    read_records,
    repair_trailing_newline,
    write_jsonl,
)

FMT = JsonlFormat("_format_version", 2)
IMAGE_EXTENSIONS = frozenset({".jpg", ".png"})

# A working-log header and page entry in the shape the pipeline writes.
HEADER: dict[str, Any] = {
    "log_type": "transcription",
    "input_item_name": "doc",
    "input_item_path": "C:/in/doc.pdf",
    "input_type": "pdf",
    "processing_start_time": "2026-10-05T12:00:00",
    "total_images": 2,
    "model_name": "gpt-test",
    "configuration": {"concurrent_requests": 4, "service_tier": "flex"},
    "file_provenance": {"source_file": "C:/in/doc.pdf", "size": 10},
}


def _entry(index: int, **extra: Any) -> dict[str, Any]:
    return {
        "original_input_order_index": index,
        "image_name": f"page_{index + 1:04d}",
        "transcription": f"Seite {index + 1} – Brot",
        **extra,
    }


def test_header_and_entries_round_trip_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    with JsonlWriter() as writer:
        assert writer.start(path, FMT, HEADER)
        assert writer.append(path, _entry(0))
        assert writer.append(path, _entry(1, error="timeout"))

    first = path.read_text(encoding="utf-8").split("\n")[0]
    assert json.loads(first) == {"_format_version": 2, **HEADER}
    assert first.startswith('{"_format_version": 2, "log_type"')
    content = read_jsonl(path, FMT)
    assert content is not None
    assert content.header == {"_format_version": 2, **HEADER}
    assert content.records == [_entry(0), _entry(1, error="timeout")]
    assert not content.dropped_tail
    assert b"\r\n" not in path.read_bytes()


def test_missing_and_blank_files_read_as_none(tmp_path: Path) -> None:
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n \n", encoding="utf-8")

    assert read_jsonl(tmp_path / "missing.jsonl", FMT) is None
    assert read_jsonl(blank, FMT) is None
    assert read_header(tmp_path / "missing.jsonl", FMT) is None


@pytest.mark.parametrize(
    "first_line",
    [
        '{"_format_version": 1, "log_type": "transcription"}',
        '{"log_type": "transcription"}',
        '[{"_format_version": 2}]',
        "not json",
    ],
)
def test_outdated_or_missing_header_is_refused(tmp_path: Path, first_line: str) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text(first_line + "\n" + json.dumps(_entry(0)) + "\n", encoding="utf-8")

    with pytest.raises(JsonlFormatError):
        read_jsonl(path, FMT)
    assert read_header(path, FMT) is None


def test_truncated_last_line_is_dropped_and_interior_line_skipped() -> None:
    text = "\n".join(
        [
            json.dumps({"_format_version": 2}),
            json.dumps(_entry(0)),
            '{"original_input_order_index": 1, "transcr',
            json.dumps(_entry(2)),
            '{"original_input_order_index": 3, "tra',
        ]
    )

    content = parse_jsonl(text, FMT)

    assert content is not None
    assert [r["original_input_order_index"] for r in content.records] == [0, 2]
    assert content.dropped_tail
    assert content.skipped_lines == (3,)


def test_bom_and_crlf_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    lines = [json.dumps({"_format_version": 2}), json.dumps(_entry(0))]
    path.write_bytes(("\ufeff" + "\r\n".join(lines) + "\r\n").encode("utf-8"))

    content = read_jsonl(path, FMT)

    assert content is not None and content.records == [_entry(0)]
    assert read_header(path, FMT) == {"_format_version": 2}
    assert read_records(path) == [{"_format_version": 2}, _entry(0)]


def test_append_repairs_a_crash_truncated_line(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    write_jsonl(path, FMT, HEADER, [_entry(0)])
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"original_input_order_index": 1, "transcr')

    with JsonlWriter() as writer:
        writer.append(path, _entry(2))

    content = read_jsonl(path, FMT)
    assert content is not None
    assert content.records == [_entry(0), _entry(2)]
    assert content.skipped_lines == (3,)


def test_repair_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    assert not repair_trailing_newline(path)
    path.write_bytes(b"")
    assert not repair_trailing_newline(path)
    path.write_bytes(b'{"a": 1}')
    assert repair_trailing_newline(path)
    assert not repair_trailing_newline(path)
    assert path.read_bytes() == b'{"a": 1}\n'


def test_concurrent_appends_keep_every_line_whole(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    writer = JsonlWriter()
    writer.start(path, FMT, HEADER)

    def work(worker: int) -> None:
        for index in range(50):
            writer.append(path, _entry(worker * 100 + index, text="x" * 500))

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    writer.close_all()

    content = read_jsonl(path, FMT)
    assert content is not None
    indices = sorted(r["original_input_order_index"] for r in content.records)
    assert indices == sorted(w * 100 + i for w in range(8) for i in range(50))
    assert not content.skipped_lines and not content.dropped_tail


def test_handle_is_reused_until_closed(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    writer = JsonlWriter()
    writer.start(path, FMT, HEADER)
    writer.append(path, _entry(0))
    handle = writer._handles[path][0]
    writer.append(path, _entry(1))
    assert writer._handles[path][0] is handle

    writer.close(path)
    assert handle.closed
    assert writer.append(path, _entry(2))
    assert path not in writer._handles

    content = read_jsonl(path, FMT)
    assert content is not None and len(content.records) == 3


def test_start_after_close_caches_again(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    writer = JsonlWriter()
    writer.start(path, FMT, HEADER)
    writer.close(path)
    writer.start(path, FMT, HEADER)
    writer.append(path, _entry(0))

    assert path in writer._handles
    writer.close_all()


def test_failed_start_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "logs"
    folder.mkdir()
    path = folder / "log.jsonl"
    write_jsonl(path, FMT, HEADER, [_entry(0)])
    before = path.read_bytes()

    def fail(*_args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)

    assert not JsonlWriter().start(path, FMT, {"log_type": "summary"})
    assert path.read_bytes() == before
    assert [p.name for p in folder.iterdir()] == ["log.jsonl"]


def test_unserializable_record_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    with JsonlWriter() as writer:
        writer.start(path, FMT, HEADER)
        assert not writer.append(path, {"bad": object()})


def test_update_header_merges_fields_and_keeps_records(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    writer = JsonlWriter()
    writer.start(path, FMT, HEADER)
    writer.append(path, _entry(0))

    assert writer.update_header(path, FMT, {"completed_at": "2026-10-05T13:00:00"})
    writer.append(path, _entry(1))
    writer.close_all()

    content = read_jsonl(path, FMT)
    assert content is not None
    assert content.header["completed_at"] == "2026-10-05T13:00:00"
    assert content.header["log_type"] == "transcription"
    assert content.records == [_entry(0), _entry(1)]
    assert not writer.update_header(tmp_path / "missing.jsonl", FMT, {"a": 1})


def test_format_header_rejects_a_conflicting_version() -> None:
    assert FMT.header({"_format_version": 2, "a": 1}) == {"_format_version": 2, "a": 1}
    with pytest.raises(ValueError):
        FMT.header({"_format_version": 1})


def test_file_input_change_is_detected_by_size(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"12345")
    identity = file_identity(pdf)

    assert identity == {"source_file": str(pdf), "size": 5}
    assert not input_changed(identity, extensions=IMAGE_EXTENSIONS)
    pdf.write_bytes(b"123456")
    assert input_changed(identity, extensions=IMAGE_EXTENSIONS)
    pdf.unlink()
    assert not input_changed(identity, extensions=IMAGE_EXTENSIONS)


def _images(folder: Path, names: dict[str, bytes]) -> list[Path]:
    folder.mkdir(exist_ok=True)
    paths = []
    for name, data in names.items():
        path = folder / name
        path.write_bytes(data)
        paths.append(path)
    return paths


def test_folder_input_change_by_count_names_and_bytes(tmp_path: Path) -> None:
    folder = tmp_path / "book"
    files = _images(folder, {"a.jpg": b"aa", "b.png": b"bbb"})
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")
    identity = folder_identity(folder, files)

    assert identity["image_count"] == 2
    assert identity["total_image_bytes"] == 5
    assert not input_changed(identity, extensions=IMAGE_EXTENSIONS)

    (folder / "a.jpg").rename(folder / "z.jpg")
    assert input_changed(identity, extensions=IMAGE_EXTENSIONS)
    (folder / "z.jpg").rename(folder / "a.jpg")

    (folder / "a.jpg").write_bytes(b"aaa")
    assert input_changed(identity, extensions=IMAGE_EXTENSIONS)
    (folder / "a.jpg").write_bytes(b"aa")

    _images(folder, {"c.jpg": b"c"})
    assert input_changed(identity, extensions=IMAGE_EXTENSIONS)


@pytest.mark.parametrize(
    "identity",
    [None, {}, {"source_file": 3}, {"source_file": "x"}, {"size": 4}],
)
def test_unprovable_change_is_not_reported(identity: dict[str, Any] | None) -> None:
    assert not input_changed(identity, extensions=IMAGE_EXTENSIONS)
