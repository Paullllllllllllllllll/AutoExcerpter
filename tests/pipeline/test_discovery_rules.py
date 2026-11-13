"""Discovery on fixture trees: the items found, their order, names and outputs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

import autoexcerpter.run as core
from autoexcerpter.output.paths import ItemPaths
from autoexcerpter.pipeline.types import ItemSpec
from tests.conftest import make_settings, make_spec

TREE = (
    "B.pdf",
    "a.pdf",
    "page_10.pdf",
    "page_2.pdf",
    "notes.txt",
    "a_summary_context.txt",
    "a_transcription.md",
    "a_summary.docx",
    "autoexcerpter.sqlite",
    "vol_10/x10.pdf",
    "vol_2/x2.pdf",
    "scans/p1.jpg",
    "scans/p2.png",
    "scans/thumbs/t1.jpg",
    "scans/inner/deep.pdf",
    "Scans_9/p1.jpg",
    "photos.2023/p1.jpg",
)

FOUND = [
    ("pdf", "a.pdf", None),
    ("pdf", "B.pdf", None),
    ("pdf", "page_2.pdf", None),
    ("pdf", "page_10.pdf", None),
    ("image_folder", "photos.2023", 1),
    ("image_folder", "scans", 2),
    ("pdf", "scans/inner/deep.pdf", None),
    ("image_folder", "Scans_9", 1),
    ("pdf", "vol_2/x2.pdf", None),
    ("pdf", "vol_10/x10.pdf", None),
]

IN_SUBFOLDERS = {"scans/inner/deep.pdf", "vol_2/x2.pdf", "vol_10/x10.pdf"}


def _tree(root: Path, *files: str) -> Path:
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (root / "empty").mkdir(exist_ok=True)
    return root


def _items(path: Path, *, recursive: bool = True) -> list[ItemSpec]:
    return core.discover(path, recursive=recursive).items


def _found(items: list[ItemSpec], root: Path) -> list[tuple[str, str, int | None]]:
    return [
        (item.kind, item.path.relative_to(root).as_posix(), item.image_count)
        for item in items
    ]


def _plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: object
) -> core.RunPlan:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "test-key")
    spec = make_spec(tmp_path / "in", output=tmp_path / "out", **flags)
    return core.plan(spec, make_settings())


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return _tree(tmp_path / "in", *TREE)


def test_a_tree_gives_its_items_in_natural_order_of_their_paths(tree: Path) -> None:
    assert _found(_items(tree), tree) == FOUND


def test_without_recursive_the_folder_gives_its_pdfs_and_image_folders(
    tree: Path,
) -> None:
    found = core.discover(tree)

    assert _found(found.items, tree) == [
        entry for entry in FOUND if entry[1] not in IN_SUBFOLDERS
    ]
    assert found.unentered == len(IN_SUBFOLDERS)


def test_the_plan_warns_of_the_items_in_unsearched_subfolders(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_plan = _plan(tmp_path, monkeypatch, all=True)

    assert len(run_plan.to_process) == len(FOUND) - len(IN_SUBFOLDERS)
    assert run_plan.warnings[0] == (
        "3 items lie in subfolders that were not searched; pass --recursive to "
        "include them."
    )
    recursive = _plan(tmp_path, monkeypatch, all=True, recursive=True)
    assert len(recursive.to_process) == len(FOUND)
    assert recursive.warnings == ()


def test_a_folder_with_items_only_below_names_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree(tmp_path / "in", "shelf/a.pdf")

    with pytest.raises(core.NoItemsError, match=r"1 item lies in subfolders .*"):
        _plan(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        ("3-4", ["page_2", "page_10"]),
        ("6", ["scans"]),
        ("vol_", ["x2", "x10"]),
        ("scans/", ["deep"]),
    ],
)
def test_select_numbers_and_searches_the_paths_below_the_input(
    tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    select: str,
    expected: list[str],
) -> None:
    run_plan = _plan(tmp_path, monkeypatch, select=select, recursive=True)

    assert [planned.name for planned in run_plan.to_process] == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (("vol_10/x10.pdf", "a.pdf"), ["a", "x10"]),
        (("vol_2\\x2.pdf", "scans/"), ["scans", "x2"]),
        (("deep.pdf", "Scans_9"), ["deep", "Scans_9"]),
    ],
)
def test_files_take_paths_below_the_input_at_every_level(
    tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: tuple[str, ...],
    expected: list[str],
) -> None:
    run_plan = _plan(tmp_path, monkeypatch, files=files)

    assert [planned.name for planned in run_plan.to_process] == expected


def test_files_refuse_a_shared_name_with_its_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree(tmp_path / "in", "a/book.pdf", "b/book.pdf")

    with pytest.raises(core.SelectionError) as raised:
        _plan(tmp_path, monkeypatch, files=("book.pdf",))

    assert str(raised.value) == (
        "--files name 'book.pdf' matches 2 items; name one by its path below the "
        "input folder: a/book.pdf, b/book.pdf"
    )
    run_plan = _plan(tmp_path, monkeypatch, files=("b/book.pdf",))
    assert [planned.item.path for planned in run_plan.to_process] == [
        tmp_path / "in" / "b" / "book.pdf"
    ]


def test_a_bare_name_shared_with_an_item_below_needs_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path / "in", "book.pdf", "archive/book.pdf")

    with pytest.raises(core.SelectionError) as raised:
        _plan(tmp_path, monkeypatch, files=("book.pdf",))

    assert str(raised.value).endswith("input folder: archive/book.pdf, ./book.pdf")
    for name, path in (
        ("./book.pdf", root / "book.pdf"),
        (".\\book.pdf", root / "book.pdf"),
        ("archive/book.pdf", root / "archive" / "book.pdf"),
    ):
        run_plan = _plan(tmp_path, monkeypatch, files=(name,))
        assert [planned.item.path for planned in run_plan.to_process] == [path]


def test_files_name_every_name_that_is_no_item(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(core.SelectionError) as raised:
        _plan(tmp_path, monkeypatch, files=("a.pdf", "notes.txt", "vol_2/x9.pdf"))

    assert "match no PDF or image folder" in str(raised.value)
    assert str(raised.value).endswith("'notes.txt', 'vol_2/x9.pdf'")


def test_a_pdf_and_an_image_folder_of_one_name_are_refused_with_both_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path / "in", "book.pdf", "book/p1.jpg", "other.pdf")

    with pytest.raises(core.DuplicateOutputError) as raised:
        _plan(tmp_path, monkeypatch, all=True)

    message = str(raised.value)
    assert f"'{root / 'book'}' and '{root / 'book.pdf'}'" in message
    assert raised.value.items_total == 3


def test_document_names_and_output_paths_follow_the_input(
    tree: Path, tmp_path: Path
) -> None:
    items = _items(tree)
    beside = {
        item.output_stem: ItemPaths.for_item(item.output_stem, item.path.parent)
        for item in items
    }
    into = {
        item.output_stem: ItemPaths.for_item(item.output_stem, tmp_path / "out")
        for item in items
    }

    assert {
        stem: paths.transcription.relative_to(tmp_path).as_posix()
        for stem, paths in beside.items()
    } == {
        "B": "in/B_transcription.md",
        "a": "in/a_transcription.md",
        "page_10": "in/page_10_transcription.md",
        "page_2": "in/page_2_transcription.md",
        "deep": "in/scans/inner/deep_transcription.md",
        "x10": "in/vol_10/x10_transcription.md",
        "x2": "in/vol_2/x2_transcription.md",
        "Scans_9": "in/Scans_9_transcription.md",
        "photos.2023": "in/photos.2023_transcription.md",
        "scans": "in/scans_transcription.md",
    }
    assert {paths.output_dir for paths in into.values()} == {tmp_path / "out"}
    assert into["photos.2023"].summary_md.name == "photos.2023_summary.md"


@pytest.mark.parametrize("recursive", [True, False])
def test_an_image_folder_below_another_is_left_out_with_a_warning(
    tree: Path, caplog: pytest.LogCaptureFixture, recursive: bool
) -> None:
    with caplog.at_level(logging.WARNING, logger="autoexcerpter.pipeline.scanner"):
        _items(tree, recursive=recursive)

    assert [record.getMessage() for record in caplog.records] == [
        f"Skipping image folder {tree / 'scans' / 'thumbs'} (1 image(s)): nested "
        f"under image folder {tree / 'scans'}. Point the input path at it "
        "directly to process it."
    ]


def test_an_input_folder_of_images_is_one_item(tmp_path: Path) -> None:
    root = _tree(tmp_path / "book", "p2.jpg", "p1.jpg", "sub/p1.jpg", "sub/x.pdf")

    assert _found(_items(root), root) == [
        ("image_folder", ".", 2),
        ("pdf", "sub/x.pdf", None),
    ]
    assert _found(_items(root, recursive=False), root) == [("image_folder", ".", 2)]


def test_a_file_input_is_one_pdf_or_nothing(tmp_path: Path) -> None:
    root = _tree(tmp_path / "in", "a.pdf", "notes.txt")

    assert [item.path for item in _items(root / "a.pdf")] == [root / "a.pdf"]
    assert _items(root / "notes.txt") == []
    assert _items(root / "missing") == []


def test_mabillon_working_folders_are_not_searched(tmp_path: Path) -> None:
    root = _tree(tmp_path / "in", "a.pdf", ".mabillon/work.pdf", "x/.mabillon/p1.png")

    assert _found(_items(root), root) == [("pdf", "a.pdf", None)]


def _link_folder(link: Path, target: Path, *, junction: bool) -> None:
    if junction:
        import _winapi  # noqa: PLC0415 - Windows only

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize(
    "junction",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="junctions exist on Windows only"
            ),
            id="junction",
        ),
        pytest.param(False, id="symlink"),
    ],
)
def test_a_linked_folder_is_not_entered(tmp_path: Path, junction: bool) -> None:
    _tree(tmp_path / "real", "linked.pdf")
    root = _tree(tmp_path / "in", "a.pdf")
    try:
        _link_folder(root / "link", tmp_path / "real", junction=junction)
    except OSError:
        pytest.skip("directory links need a privilege on this system")

    assert _found(_items(root), root) == [("pdf", "a.pdf", None)]
