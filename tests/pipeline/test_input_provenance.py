"""The identity of an input: its provenance and the hash the store keeps.

The store records each document's input hash; a changed hash resets the
document (see tests/output/test_ae_resume.py). A PDF's hash is the file's
SHA-256, an image folder's covers its image names and contents in page
order, so a renamed, added or edited image changes it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from autoexcerpter.imaging.payload import FolderPayloadSource
from autoexcerpter.pipeline.item import input_sha256
from autoexcerpter.pipeline.types import ItemSpec


def _make_folder(tmp_path: Path, count: int) -> Path:
    folder = tmp_path / "imgs"
    folder.mkdir(exist_ok=True)
    for i in range(count):
        Image.new("RGB", (20, 20), color=(i * 10, 0, 0)).save(
            folder / f"page_{i:03d}.jpg", "JPEG"
        )
    return folder


def _folder_hash(folder: Path) -> str | None:
    return input_sha256(ItemSpec("image_folder", folder))


@pytest.mark.usefixtures("mock_image_processing")
def test_the_folder_provenance_records_its_identity_fields(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path, 3)
    source = FolderPayloadSource(folder, provider="openai", model_name="gpt-5")
    prov = source.file_provenance()
    assert prov["image_count"] == 3
    expected = sum(p.stat().st_size for p in folder.glob("*.jpg"))
    assert prov["total_image_bytes"] == expected


def test_a_pdf_hash_is_the_file_hash(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 content")

    assert (
        input_sha256(ItemSpec("pdf", pdf))
        == hashlib.sha256(pdf.read_bytes()).hexdigest()
    )


def test_an_unreadable_input_has_no_hash(tmp_path: Path) -> None:
    assert input_sha256(ItemSpec("pdf", tmp_path / "missing.pdf")) is None


def test_the_folder_hash_is_stable(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path, 2)

    assert _folder_hash(folder) == _folder_hash(folder)


def test_an_added_image_changes_the_folder_hash(tmp_path: Path) -> None:
    folder = _make_folder(tmp_path, 2)
    before = _folder_hash(folder)

    Image.new("RGB", (20, 20), color=(9, 9, 9)).save(folder / "page_099.jpg", "JPEG")

    assert _folder_hash(folder) != before


def test_a_renamed_image_of_the_same_bytes_changes_the_folder_hash(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"aa")
    (folder / "b.jpg").write_bytes(b"bb")
    before = _folder_hash(folder)

    (folder / "a.jpg").rename(folder / "z.jpg")

    assert _folder_hash(folder) != before


def test_an_edited_image_changes_the_folder_hash(tmp_path: Path) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"aa")
    before = _folder_hash(folder)

    (folder / "a.jpg").write_bytes(b"ab")

    assert _folder_hash(folder) != before
