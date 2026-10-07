"""Tiered context resolution, prompt formatting and the context hash."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from autoexcerpter.common.context import (
    NO_CONTEXT_HASH,
    ContextError,
    ContextSource,
    compute_context_hash,
    format_context_for_prompt,
    resolve_context,
    resolve_context_image,
    sidecar_candidates,
)

SUFFIX = "_summary_context"
ARGUMENT_TEXT = "Focus on bread prices and urban wages."
FILE_LINES = "Track grain prices.\nNote every wage series.\n"
FILE_PROMPT = "Track grain prices., Note every wage series."
FOLDER_TEXT = "Pay attention to market regulation."


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _pdf(folder: Path, name: str = "doc.pdf") -> Path:
    return _write(folder / name, "%PDF-1.4\n")


def _image_folder(folder: Path) -> Path:
    folder.mkdir(parents=True)
    _write(folder / "1.png", "png")
    return folder


def test_pdf_context_from_argument(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    _write(tmp_path / "docs" / f"doc{SUFFIX}.txt", FILE_LINES)

    resolved = resolve_context(pdf, SUFFIX, explicit=ARGUMENT_TEXT)

    assert resolved is not None
    assert resolved.text == ARGUMENT_TEXT
    assert resolved.source is ContextSource.EXPLICIT
    assert resolved.path is None


def test_pdf_context_from_file_sidecar(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    sidecar = _write(tmp_path / "docs" / f"doc{SUFFIX}.txt", FILE_LINES)
    _write(tmp_path / f"docs{SUFFIX}.txt", FOLDER_TEXT)

    resolved = resolve_context(pdf, SUFFIX)

    assert resolved is not None
    assert resolved.source is ContextSource.FILE
    assert resolved.path == sidecar.resolve()
    assert format_context_for_prompt(resolved.text) == FILE_PROMPT


def test_image_folder_context_from_folder_sidecar(tmp_path: Path) -> None:
    collection = tmp_path / "in" / "collection"
    folder = _image_folder(collection / "book")
    sidecar = _write(tmp_path / "in" / f"collection{SUFFIX}.txt", FOLDER_TEXT + "\n")

    resolved = resolve_context(folder, SUFFIX)

    assert resolved is not None
    assert resolved.source is ContextSource.FOLDER
    assert resolved.path == sidecar.resolve()
    assert format_context_for_prompt(resolved.text) == FOLDER_TEXT


def test_image_folder_own_sidecar_uses_full_folder_name(tmp_path: Path) -> None:
    folder = _image_folder(tmp_path / "photos.2023")
    _write(tmp_path / f"photos{SUFFIX}.txt", "stem only")
    own = _write(tmp_path / f"photos.2023{SUFFIX}.txt", "full name")

    resolved = resolve_context(folder, SUFFIX)

    assert resolved is not None
    assert (resolved.text, resolved.path) == ("full name", own.resolve())


def test_directory_folder_sidecar_can_be_disabled(tmp_path: Path) -> None:
    folder = _image_folder(tmp_path / "in" / "collection" / "book")
    _write(tmp_path / "in" / f"collection{SUFFIX}.txt", FOLDER_TEXT)

    assert resolve_context(folder, SUFFIX, folder_sidecar_for_directories=False) is None


def test_empty_sidecar_falls_through_to_next_tier(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    _write(tmp_path / "docs" / f"doc{SUFFIX}.txt", "  \n")
    _write(tmp_path / f"docs{SUFFIX}.txt", FOLDER_TEXT)

    resolved = resolve_context(pdf, SUFFIX)

    assert resolved is not None
    assert resolved.source is ContextSource.FOLDER


def test_settings_default_is_the_last_tier(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    default_file = _write(tmp_path / "topics.txt", "Topic A\nTopic B\n")

    from_text = resolve_context(pdf, SUFFIX, default="General topics")
    from_file = resolve_context(pdf, SUFFIX, default=default_file)

    assert from_text is not None and from_text.source is ContextSource.DEFAULT
    assert from_text.text == "General topics"
    assert from_file is not None and from_file.path == default_file
    assert from_file.text == "Topic A\nTopic B"


def test_sidecar_beats_default_and_sidecars_can_be_skipped(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    _write(tmp_path / "docs" / f"doc{SUFFIX}.txt", FILE_LINES)

    with_sidecars = resolve_context(pdf, SUFFIX, default="fallback")
    without = resolve_context(pdf, SUFFIX, default="fallback", use_sidecars=False)

    assert with_sidecars is not None and with_sidecars.source is ContextSource.FILE
    assert without is not None and without.source is ContextSource.DEFAULT


def test_no_implicit_fallback(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    _write(tmp_path / "context" / "summary" / "general.txt", "general")

    assert resolve_context(pdf, SUFFIX) is None
    assert resolve_context(None, SUFFIX) is None


def test_explicit_path_is_read_and_missing_path_raises(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    context_file = _write(tmp_path / "ctx.txt", "\N{BYTE ORDER MARK}From a file\n")

    resolved = resolve_context(pdf, SUFFIX, explicit=context_file)

    assert resolved is not None
    assert resolved.text == "From a file"
    with pytest.raises(ContextError):
        resolve_context(pdf, SUFFIX, explicit=tmp_path / "missing.txt")
    with pytest.raises(ContextError):
        resolve_context(pdf, SUFFIX, explicit=_write(tmp_path / "blank.txt", " "))


def test_empty_explicit_text_counts_as_not_given(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    _write(tmp_path / "docs" / f"doc{SUFFIX}.txt", FILE_LINES)

    resolved = resolve_context(pdf, SUFFIX, explicit="  ")

    assert resolved is not None and resolved.source is ContextSource.FILE


def test_oversized_context_is_flagged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf = _pdf(tmp_path / "docs")
    _write(tmp_path / "docs" / f"doc{SUFFIX}.txt", "x" * 50)

    with caplog.at_level(logging.WARNING):
        resolved = resolve_context(pdf, SUFFIX, size_threshold=10)

    assert resolved is not None and resolved.oversized
    assert "is large (50 chars)" in caplog.text


def test_sidecar_candidates_order(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")

    candidates = sidecar_candidates(pdf, "_extract_context", (".txt", ".md"))

    docs = (tmp_path / "docs").resolve()
    assert candidates == [
        (ContextSource.FILE, docs / "doc_extract_context.txt"),
        (ContextSource.FILE, docs / "doc_extract_context.md"),
        (ContextSource.FOLDER, docs.parent / "docs_extract_context.txt"),
        (ContextSource.FOLDER, docs.parent / "docs_extract_context.md"),
    ]


def test_context_image_tiers(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "docs")
    folder_image = _write(tmp_path / "docs_page_context.jpg", "jpg")
    file_image = _write(tmp_path / "docs" / "doc_page_context.png", "png")

    found = resolve_context_image(pdf, "_page_context")
    assert found == (ContextSource.FILE, file_image.resolve())

    file_image.unlink()
    found = resolve_context_image(pdf, "_page_context")
    assert found == (ContextSource.FOLDER, folder_image.resolve())

    with pytest.raises(ContextError):
        resolve_context_image(pdf, "_page_context", explicit=tmp_path / "x.txt")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("one line", "one line"),
        ("  a  \n\n  b\n", "a, b"),
        ("a\r\nb", "a, b"),
        ("   ", ""),
    ],
)
def test_format_context_for_prompt(raw: str, expected: str) -> None:
    assert format_context_for_prompt(raw) == expected


def test_compute_context_hash() -> None:
    assert compute_context_hash(None) == NO_CONTEXT_HASH
    expected = hashlib.sha256(b"focus").hexdigest()
    assert compute_context_hash("focus") == expected
    assert compute_context_hash("focus") != compute_context_hash("focus ")
