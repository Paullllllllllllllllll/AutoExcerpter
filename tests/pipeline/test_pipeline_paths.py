"""Output stem, output paths and working paths of one item."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoexcerpter.pipeline.paths import (
    HASH_LENGTH,
    WINDOWS_MAX_PATH,
    ItemPaths,
    item_base,
    name_hash,
)
from autoexcerpter.pipeline.types import ItemSpec
from tests.pipeline.helpers import make_processor


def _all_paths(paths: ItemPaths) -> list[Path]:
    return [
        paths.transcription,
        paths.summary_md,
        paths.summary_docx,
        paths.transcription_log,
        paths.summary_log,
    ]


def test_role_suffixed_names_beside_each_other(tmp_path: Path) -> None:
    paths = ItemPaths.for_item("Book 1791", tmp_path)

    assert paths.transcription == tmp_path / "Book 1791_transcription.md"
    assert paths.summary_md == tmp_path / "Book 1791_summary.md"
    assert paths.summary_docx == tmp_path / "Book 1791_summary.docx"
    assert paths.working_dir == tmp_path / "Book 1791_autoexcerpter"
    assert paths.transcription_log == paths.working_dir / "transcription.jsonl"
    assert paths.summary_log == paths.working_dir / "summary.jsonl"


def test_txt_format_changes_only_the_transcription_extension(tmp_path: Path) -> None:
    md = ItemPaths.for_item("doc", tmp_path)
    txt = ItemPaths.for_item("doc", tmp_path, "txt")

    assert txt.transcription == tmp_path / "doc_transcription.txt"
    assert (txt.summary_md, txt.summary_docx, txt.working_dir) == (
        md.summary_md,
        md.summary_docx,
        md.working_dir,
    )


def test_unknown_transcription_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="transcription format"):
        ItemPaths.for_item("doc", tmp_path, "docx")


def test_a_name_that_fits_is_kept_unchanged(tmp_path: Path) -> None:
    name = "Beukers et al 2025 Grape (Vitis vinifera) use in the early modern period"
    assert item_base(name, tmp_path) == name


def test_a_long_name_is_shortened_with_its_hash_for_every_file() -> None:
    output_dir = Path("C:/") / ("library" * 10)
    name = "A very long title " * 15

    paths = ItemPaths.for_item(name, output_dir)
    base = item_base(name, output_dir)

    assert base != name
    assert base.endswith("-" + name_hash(name))
    assert len(name_hash(name)) == HASH_LENGTH
    assert not base.removesuffix("-" + name_hash(name)).endswith(" ")
    for path in _all_paths(paths):
        assert len(str(path)) < WINDOWS_MAX_PATH
        assert base in str(path)


def test_shortened_names_stay_distinct_and_deterministic() -> None:
    output_dir = Path("C:/") / ("x" * 150)
    first = "Same long prefix " * 10 + "one"
    second = "Same long prefix " * 10 + "two"

    assert item_base(first, output_dir) != item_base(second, output_dir)
    assert item_base(first, output_dir) == item_base(first, output_dir)


def test_an_output_folder_too_long_for_any_name_leaves_the_hash() -> None:
    output_dir = Path("C:/") / ("y" * 240)
    assert item_base("doc", output_dir) == name_hash("doc")


class TestItemSpecOutputStem:
    """A PDF drops its extension; an image folder keeps its full name."""

    def test_pdf_drops_extension(self) -> None:
        spec = ItemSpec(kind="pdf", path=Path("C:/in/paper.v2.pdf"))
        assert spec.output_stem == "paper.v2"

    def test_image_folder_keeps_full_name(self) -> None:
        spec = ItemSpec(kind="image_folder", path=Path("C:/in/photos.2023"))
        assert spec.output_stem == "photos.2023"

    def test_dotted_sibling_folders_do_not_collide(self) -> None:
        a = ItemSpec(kind="image_folder", path=Path("C:/in/photos.2023"))
        b = ItemSpec(kind="image_folder", path=Path("C:/in/photos.2024"))
        assert a.output_stem != b.output_stem


class TestItemNameIdentity:
    """The item processor's name follows ``ItemSpec.output_stem``."""

    def test_dotted_image_folder_keeps_full_name(self, tmp_path: Path) -> None:
        folder = tmp_path / "photos.2023"
        folder.mkdir()
        processor = make_processor(folder, tmp_path / "out", kind="image_folder")
        assert processor.name == "photos.2023"
        assert processor.paths.transcription.name == "photos.2023_transcription.md"

    def test_pdf_name_drops_extension(self, tmp_path: Path) -> None:
        processor = make_processor(tmp_path / "paper.v2.pdf", tmp_path / "out")
        assert processor.name == "paper.v2"
