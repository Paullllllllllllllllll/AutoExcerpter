"""Tests for modules/item_scanner.py - Item scanning utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.item_scanner import (
    is_pdf_file,
    is_supported_image,
    scan_input_path,
    _collect_items_from_directory,
    _build_pdf_item,
    _build_image_folder_items,
)
from modules.types import ItemSpec


# ============================================================================
# is_pdf_file
# ============================================================================
class TestIsPdfFile:
    """Tests for is_pdf_file()."""

    def test_lowercase_pdf(self):
        """Lowercase .pdf extension is recognized."""
        assert is_pdf_file(Path("document.pdf")) is True

    def test_uppercase_pdf(self):
        """Uppercase .PDF extension is recognized (case-insensitive)."""
        assert is_pdf_file(Path("DOCUMENT.PDF")) is True

    def test_mixed_case_pdf(self):
        """Mixed-case .Pdf extension is recognized."""
        assert is_pdf_file(Path("file.Pdf")) is True

    def test_non_pdf_txt(self):
        """Text file is not recognized as PDF."""
        assert is_pdf_file(Path("notes.txt")) is False

    def test_non_pdf_jpg(self):
        """JPEG file is not recognized as PDF."""
        assert is_pdf_file(Path("photo.jpg")) is False

    def test_no_extension(self):
        """File without extension is not recognized as PDF."""
        assert is_pdf_file(Path("document")) is False

    def test_pdf_in_name_but_different_extension(self):
        """File with 'pdf' in name but different extension is not PDF."""
        assert is_pdf_file(Path("pdf_backup.zip")) is False

    def test_dotfile(self):
        """Hidden dotfile is not recognized as PDF."""
        assert is_pdf_file(Path(".hidden")) is False


# ============================================================================
# is_supported_image
# ============================================================================
class TestIsSupportedImage:
    """Tests for is_supported_image()."""

    @pytest.mark.parametrize(
        "ext",
        [
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".tif",
            ".bmp",
            ".gif",
            ".webp",
        ],
    )
    def test_supported_extensions(self, ext: str):
        """All supported image extensions are recognized."""
        assert is_supported_image(Path(f"image{ext}")) is True

    @pytest.mark.parametrize(
        "ext",
        [
            ".JPG",
            ".JPEG",
            ".PNG",
            ".TIFF",
            ".TIF",
            ".BMP",
            ".GIF",
            ".WEBP",
        ],
    )
    def test_supported_uppercase(self, ext: str):
        """Uppercase image extensions are recognized (case-insensitive)."""
        assert is_supported_image(Path(f"image{ext}")) is True

    @pytest.mark.parametrize(
        "ext",
        [
            ".pdf",
            ".txt",
            ".doc",
            ".svg",
            ".ico",
            ".psd",
            ".raw",
        ],
    )
    def test_unsupported_extensions(self, ext: str):
        """Unsupported extensions are rejected."""
        assert is_supported_image(Path(f"file{ext}")) is False

    def test_no_extension(self):
        """File without extension is not a supported image."""
        assert is_supported_image(Path("image_file")) is False


# ============================================================================
# _build_pdf_item
# ============================================================================
class TestBuildPdfItem:
    """Tests for _build_pdf_item()."""

    def test_creates_correct_item_spec(self, tmp_path: Path):
        """_build_pdf_item returns an ItemSpec with kind='pdf'."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        item = _build_pdf_item(pdf_path)

        assert isinstance(item, ItemSpec)
        assert item.kind == "pdf"
        assert item.path == pdf_path
        assert item.image_count is None

    def test_preserves_full_path(self, tmp_path: Path):
        """The returned ItemSpec contains the full path."""
        deep_path = tmp_path / "a" / "b" / "c" / "doc.pdf"
        deep_path.parent.mkdir(parents=True)
        deep_path.touch()

        item = _build_pdf_item(deep_path)

        assert item.path == deep_path


# ============================================================================
# _build_image_folder_items
# ============================================================================
class TestBuildImageFolderItems:
    """Tests for _build_image_folder_items()."""

    def test_empty_images_list_is_skipped(self, tmp_path: Path):
        """Folder with an empty images list produces no items."""
        folder = tmp_path / "empty_folder"
        folder.mkdir()
        result = _build_image_folder_items({folder: []})

        assert result == []

    def test_normal_images(self, tmp_path: Path):
        """Folder with images produces a single ItemSpec."""
        folder = tmp_path / "images"
        folder.mkdir()
        img1 = folder / "page_002.jpg"
        img2 = folder / "page_001.jpg"
        img1.touch()
        img2.touch()

        result = _build_image_folder_items({folder: [img1, img2]})

        assert len(result) == 1
        item = result[0]
        assert item.kind == "image_folder"
        assert item.path == folder
        assert item.image_count == 2

    def test_sorting_does_not_affect_count(self, tmp_path: Path):
        """Image count is correct regardless of input order."""
        folder = tmp_path / "photos"
        folder.mkdir()
        images = [folder / f"img_{i:03d}.png" for i in range(5)]
        for img in images:
            img.touch()

        result = _build_image_folder_items({folder: list(reversed(images))})

        assert result[0].image_count == 5

    def test_multiple_folders(self, tmp_path: Path):
        """Multiple folders each produce their own ItemSpec."""
        folder_a = tmp_path / "folder_a"
        folder_b = tmp_path / "folder_b"
        folder_a.mkdir()
        folder_b.mkdir()

        img_a = folder_a / "a.jpg"
        img_b = folder_b / "b.png"
        img_a.touch()
        img_b.touch()

        result = _build_image_folder_items(
            {
                folder_a: [img_a],
                folder_b: [img_b],
            }
        )

        assert len(result) == 2
        paths = {item.path for item in result}
        assert folder_a in paths
        assert folder_b in paths


# ============================================================================
# scan_input_path
# ============================================================================
class TestScanInputPath:
    """Tests for scan_input_path()."""

    def test_single_pdf_file(self, tmp_path: Path):
        """A single PDF file returns one ItemSpec."""
        pdf = tmp_path / "document.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        items = scan_input_path(pdf)

        assert len(items) == 1
        assert items[0].kind == "pdf"
        assert items[0].path == pdf

    def test_non_pdf_file_returns_empty(self, tmp_path: Path):
        """A non-PDF file returns an empty list."""
        txt = tmp_path / "notes.txt"
        txt.write_text("hello")

        items = scan_input_path(txt)

        assert items == []

    def test_non_existent_path_returns_empty(self, tmp_path: Path):
        """A non-existent path returns an empty list."""
        missing = tmp_path / "does_not_exist.pdf"

        items = scan_input_path(missing)

        assert items == []

    def test_directory_with_pdfs(self, tmp_path: Path):
        """A directory containing PDF files returns ItemSpecs for each."""
        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            (tmp_path / name).write_bytes(b"%PDF")

        items = scan_input_path(tmp_path)

        pdf_items = [i for i in items if i.kind == "pdf"]
        assert len(pdf_items) == 3

    def test_directory_with_image_folders(self, tmp_path: Path):
        """A directory containing subfolders with images returns image_folder items."""
        img_dir = tmp_path / "scan_001"
        img_dir.mkdir()
        for name in ["page_01.jpg", "page_02.jpg"]:
            (img_dir / name).write_bytes(b"\xff\xd8\xff")

        items = scan_input_path(tmp_path)

        folder_items = [i for i in items if i.kind == "image_folder"]
        assert len(folder_items) == 1
        assert folder_items[0].image_count == 2

    def test_directory_mixed_pdfs_and_images(self, tmp_path: Path):
        """A directory with both PDFs and image folders returns both types."""
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "scan.png").write_bytes(b"\x89PNG")

        items = scan_input_path(tmp_path)

        kinds = {i.kind for i in items}
        assert "pdf" in kinds
        assert "image_folder" in kinds


# ============================================================================
# _collect_items_from_directory
# ============================================================================
class TestCollectItemsFromDirectory:
    """Tests for _collect_items_from_directory()."""

    def test_mixed_pdfs_and_image_folders(self, tmp_path: Path):
        """Collects both PDF files and image folders."""
        (tmp_path / "report.pdf").write_bytes(b"%PDF")

        img_dir = tmp_path / "scans"
        img_dir.mkdir()
        (img_dir / "p1.tiff").write_bytes(b"fake tiff")
        (img_dir / "p2.tiff").write_bytes(b"fake tiff")

        items = list(_collect_items_from_directory(tmp_path))

        pdf_items = [i for i in items if i.kind == "pdf"]
        folder_items = [i for i in items if i.kind == "image_folder"]
        assert len(pdf_items) == 1
        assert len(folder_items) == 1
        assert folder_items[0].image_count == 2

    def test_nested_directories(self, tmp_path: Path):
        """Recursively discovers items in nested directories."""
        sub1 = tmp_path / "sub1"
        sub2 = tmp_path / "sub1" / "sub2"
        sub2.mkdir(parents=True)

        (sub1 / "doc.pdf").write_bytes(b"%PDF")
        (sub2 / "deep.pdf").write_bytes(b"%PDF")

        items = list(_collect_items_from_directory(tmp_path))

        pdf_paths = {i.path.name for i in items if i.kind == "pdf"}
        assert "doc.pdf" in pdf_paths
        assert "deep.pdf" in pdf_paths

    def test_empty_directory(self, tmp_path: Path):
        """Empty directory returns no items."""
        empty = tmp_path / "empty"
        empty.mkdir()

        items = list(_collect_items_from_directory(empty))

        assert items == []

    def test_unsupported_files_ignored(self, tmp_path: Path):
        """Files that are neither PDF nor supported images are ignored."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b,c")

        items = list(_collect_items_from_directory(tmp_path))

        assert items == []

    def test_image_folder_with_nested_content(self, tmp_path: Path):
        """A directory with images is detected; its children are still walked.

        The ``dirs[:]`` filtering in ``_collect_items_from_directory``
        removes subdirectories from the walk only if those subdirectories
        are *themselves* already registered as image folders. Since
        ``os.walk`` is top-down, subdirectories are not yet in
        ``image_folders`` when the parent is evaluated, so the walk
        descends into them normally.
        """
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "photo.jpg").write_bytes(b"\xff\xd8")

        child = parent / "child"
        child.mkdir()
        (child / "nested.pdf").write_bytes(b"%PDF")

        items = list(_collect_items_from_directory(tmp_path))

        folder_items = [i for i in items if i.kind == "image_folder"]
        assert len(folder_items) == 1
        assert folder_items[0].path == parent

        # The nested PDF is still found because the child directory
        # is not itself an image folder, so it is not pruned.
        pdf_items = [i for i in items if i.kind == "pdf"]
        assert len(pdf_items) == 1
        assert pdf_items[0].path.name == "nested.pdf"
