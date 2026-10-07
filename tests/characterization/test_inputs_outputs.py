"""Every input type and output option, recorded through ``run_tool``.

Goldens live under ``golden/inputs/<case>/``. All runs use the default OpenAI
schema path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from autoexcerpter.pipeline.paths import ItemPaths
from autoexcerpter.rendering.sqlite import DATABASE_NAME
from tests.characterization.adapter import RunResult, item_paths, run_tool
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM
from tests.characterization.golden import Normalizer
from tests.characterization.inputs import (
    make_image_folder,
    make_named_images,
    make_pdf,
)

Golden = Callable[[str, RunResult], None]

MIXED_IMAGE_NAMES = (
    "1.jpg",
    "2.jpeg",
    "3.png",
    "4.tif",
    "5.tiff",
    "6.bmp",
    "7.gif",
    "8.webp",
    "9.jpg",
    "10.png",
    "11.jpeg",
)

CONTEXT_FROM_ARGUMENT = "Focus on bread prices and urban wages."
CONTEXT_FILE_LINES = "Track grain prices.\nNote every wage series.\n"
CONTEXT_FILE_PROMPT = "Track grain prices., Note every wage series."
CONTEXT_FOLDER_TEXT = "Pay attention to market regulation."


def _case(name: str) -> str:
    return f"inputs/{name}"


def _input_root(tmp_path: Path) -> Path:
    folder = tmp_path / "in"
    folder.mkdir()
    return folder


def _assert_controlled_exit(result: RunResult, code: int) -> None:
    assert result.exit_code == code, result.stderr
    assert result.summary is not None
    assert result.summary["items_complete"] == 0
    assert result.summary["outputs"] == []
    assert result.calls() == []
    assert result.created == []


def _assert_items_complete(result: RunResult, count: int) -> None:
    assert result.exit_code == 0, result.stderr
    assert result.summary is not None
    assert result.summary["items_total"] == count
    assert result.summary["items_complete"] == count


def _output_names(result: RunResult) -> list[str]:
    """Return the names of the run's outputs; the database comes last."""
    assert result.summary is not None
    return [Path(path).name for path in result.summary["outputs"]]


def _final_names(paths: ItemPaths) -> set[str]:
    """Return the names of an item's transcription and summary files."""
    return {
        paths.transcription.name,
        paths.summary_md.name,
        paths.summary_docx.name,
    }


# ============================================================================
# Image folder
# ============================================================================
def test_image_folder_mixed_extensions_natural_order(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    folder = make_named_images(_input_root(tmp_path) / "book", MIXED_IMAGE_NAMES)

    result = run_tool(folder, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out")

    _assert_items_complete(result, 1)
    pages = list(range(len(MIXED_IMAGE_NAMES)))
    assert [call["page"] for call in result.calls(TRANSCRIPTION)] == pages
    assert [call["page"] for call in result.calls(SUMMARY)] == pages
    text = result.paths("book").transcription.read_text(encoding="utf-8")
    keys = [fake_llm.page(index).key for index in pages]
    positions = [text.index(key) for key in keys]
    assert positions == sorted(positions)
    golden(_case("image_folder_mixed_extensions"), result)


# ============================================================================
# Directory tree and selection
# ============================================================================
def _make_tree(tmp_path: Path) -> Path:
    tree = _input_root(tmp_path) / "tree"
    make_pdf(tree / "alpha.pdf", pages=2)
    make_pdf(tree / "sub" / "beta.pdf", pages=2)
    make_image_folder(tree / "scans", pages=2)
    make_image_folder(tree / "scans" / "thumbs", pages=1)
    return tree


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        pytest.param("all", ["alpha", "beta", "scans"], id="directory_all"),
        pytest.param("2", ["beta"], id="directory_select_number"),
        pytest.param("2-3", ["beta", "scans"], id="directory_select_range"),
        pytest.param("alph", ["alpha"], id="directory_select_name"),
    ],
)
def test_directory_tree_selection(
    tmp_path: Path,
    fake_llm: FakeLLM,
    golden: Golden,
    request: pytest.FixtureRequest,
    selection: str,
    expected: list[str],
) -> None:
    tree = _make_tree(tmp_path)

    result = run_tool(
        tree,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        selection=selection,
    )

    _assert_items_complete(result, len(expected))
    names = _output_names(result)
    assert names[-1] == DATABASE_NAME
    assert set(names[:-1]) == set().union(
        *(_final_names(result.paths(name)) for name in expected)
    )
    assert "Skipping image folder" in result.stderr
    assert "nested under image folder" in result.stderr
    assert not any("thumbs" in str(path) for path in result.created)
    golden(_case(request.node.callspec.id), result)


# ============================================================================
# Controlled exits
# ============================================================================
def test_directory_several_items_without_selection_exits_2(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    tree = _input_root(tmp_path) / "tree"
    make_pdf(tree / "alpha.pdf", pages=1)
    make_pdf(tree / "beta.pdf", pages=1)

    result = run_tool(
        tree, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out", selection=None
    )

    _assert_controlled_exit(result, 2)
    assert "neither --all nor --select" in result.stderr
    golden(_case("exit_several_items_without_selection"), result)


def test_directory_select_matching_nothing_exits_1(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    tree = _input_root(tmp_path) / "tree"
    make_pdf(tree / "alpha.pdf", pages=1)
    make_pdf(tree / "beta.pdf", pages=1)

    result = run_tool(
        tree,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        selection="gamma",
    )

    _assert_controlled_exit(result, 1)
    assert "No items found matching 'gamma'" in result.stderr
    golden(_case("exit_select_matching_nothing"), result)


def test_directory_empty_exits_1(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    empty = _input_root(tmp_path) / "empty"
    empty.mkdir()

    result = run_tool(empty, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out")

    _assert_controlled_exit(result, 1)
    assert result.summary is not None
    assert result.summary["items_total"] == 0
    assert "No items found to process" in result.stderr
    golden(_case("exit_empty_directory"), result)


def test_directory_duplicate_output_stem_exits_2(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    tree = _input_root(tmp_path) / "tree"
    make_pdf(tree / "a" / "book.pdf", pages=1)
    make_pdf(tree / "b" / "book.pdf", pages=1)

    result = run_tool(tree, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out")

    _assert_controlled_exit(result, 2)
    assert result.summary is not None
    assert result.summary["items_total"] == 2
    assert "Duplicate output target" in result.stderr
    golden(_case("exit_duplicate_output_stem"), result)


# ============================================================================
# Output formats
# ============================================================================
def test_image_folder_summary_off(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    folder = make_image_folder(_input_root(tmp_path) / "book", pages=3)

    result = run_tool(
        folder,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        summarize=False,
    )

    _assert_items_complete(result, 1)
    assert result.calls(SUMMARY) == []
    assert _output_names(result) == [
        result.paths("book").transcription.name,
        DATABASE_NAME,
    ]
    golden(_case("image_folder_summary_off"), result)


@pytest.mark.parametrize(
    ("docx", "markdown", "expected"),
    [
        pytest.param(
            True, False, ["transcription", "summary_docx"], id="pdf_docx_only"
        ),
        pytest.param(
            False, True, ["transcription", "summary_md"], id="pdf_markdown_only"
        ),
    ],
)
def test_pdf_summary_format_choice(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    golden: Golden,
    request: pytest.FixtureRequest,
    docx: bool,
    markdown: bool,
    expected: list[str],
) -> None:
    """*expected* names the ``ItemPaths`` fields of the outputs, in order."""
    pdf = make_pdf(docs_dir / "doc.pdf", pages=3)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        docx=docx,
        markdown=markdown,
    )

    _assert_items_complete(result, 1)
    paths = result.paths("doc")
    assert _output_names(result) == [
        *(getattr(paths, field).name for field in expected),
        DATABASE_NAME,
    ]
    golden(_case(request.node.callspec.id), result)


def test_pdf_transcription_as_txt(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    """The txt format changes the transcription's extension, not its text."""
    pdf = make_pdf(docs_dir / "doc.pdf", pages=2)
    as_md = run_tool(pdf, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "md")
    fake_llm.reset_records()

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        transcription_file_format="txt",
    )

    _assert_items_complete(result, 1)
    txt = item_paths("doc", tmp_path / "out", "txt").transcription
    assert _output_names(result) == [
        txt.name,
        "doc_summary.docx",
        "doc_summary.md",
        DATABASE_NAME,
    ]
    norm = Normalizer(tmp_path)
    md_text = norm.text(as_md.paths("doc").transcription.read_text(encoding="utf-8"))
    assert norm.text(txt.read_text(encoding="utf-8")) == md_text
    golden(_case("pdf_transcription_txt"), result)


# ============================================================================
# Output beside the input
# ============================================================================
def test_pdf_beside_input(tmp_path: Path, fake_llm: FakeLLM, golden: Golden) -> None:
    pdf = make_pdf(_input_root(tmp_path) / "doc.pdf", pages=2)

    result = run_tool(
        pdf, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out", beside_input=True
    )

    _assert_items_complete(result, 1)
    assert result.summary is not None
    assert {path.parent for path in map(Path, result.summary["outputs"])} == {
        pdf.parent
    }
    assert not (tmp_path / "out").exists()
    golden(_case("pdf_beside_input"), result)


def test_image_folder_beside_input(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    folder = make_image_folder(_input_root(tmp_path) / "book", pages=2)

    result = run_tool(
        folder,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        beside_input=True,
    )

    _assert_items_complete(result, 1)
    assert result.summary is not None
    assert {path.parent for path in map(Path, result.summary["outputs"])} == {
        folder.parent
    }
    assert not (tmp_path / "out").exists()
    golden(_case("image_folder_beside_input"), result)


# ============================================================================
# Summary context
# ============================================================================
def _assert_context_in_summary_prompts(fake_llm: FakeLLM, expected: str) -> None:
    summary_prompts = fake_llm.system_prompts_for(SUMMARY)
    assert summary_prompts
    assert all(expected in prompt for prompt in summary_prompts)
    assert not any(expected in p for p in fake_llm.system_prompts_for(TRANSCRIPTION))


def test_pdf_context_from_argument(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    pdf = make_pdf(docs_dir / "doc.pdf", pages=2)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        context=CONTEXT_FROM_ARGUMENT,
    )

    _assert_items_complete(result, 1)
    _assert_context_in_summary_prompts(fake_llm, CONTEXT_FROM_ARGUMENT)
    golden(_case("pdf_context_from_argument"), result)


def test_pdf_context_from_file_sidecar(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    pdf = make_pdf(docs_dir / "doc.pdf", pages=2)
    (docs_dir / "doc_summary_context.txt").write_text(
        CONTEXT_FILE_LINES, encoding="utf-8", newline="\n"
    )

    result = run_tool(pdf, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out")

    _assert_items_complete(result, 1)
    _assert_context_in_summary_prompts(fake_llm, CONTEXT_FILE_PROMPT)
    golden(_case("pdf_context_from_file_sidecar"), result)


def test_image_folder_context_from_folder_sidecar(
    tmp_path: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    collection = _input_root(tmp_path) / "collection"
    folder = make_image_folder(collection / "book", pages=2)
    (collection.parent / "collection_summary_context.txt").write_text(
        CONTEXT_FOLDER_TEXT + "\n", encoding="utf-8", newline="\n"
    )

    result = run_tool(folder, llm=fake_llm, root=tmp_path, output_dir=tmp_path / "out")

    _assert_items_complete(result, 1)
    _assert_context_in_summary_prompts(fake_llm, CONTEXT_FOLDER_TEXT)
    golden(_case("image_folder_context_from_folder_sidecar"), result)


# ============================================================================
# OpenAlex enrichment
# ============================================================================
def test_pdf_openalex_enrichment(
    tmp_path: Path, docs_dir: Path, fake_llm: FakeLLM, golden: Golden
) -> None:
    pdf = make_pdf(docs_dir / "doc.pdf", pages=3)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        openalex=True,
    )

    _assert_items_complete(result, 1)
    markdown = result.paths("doc").summary_md.read_text(encoding="utf-8")
    assert "](https://doi.org/10.1006/exeh.2001.0775)" in markdown
    assert "](https://openalex.org/W1000000002)" in markdown
    montanari = next(line for line in markdown.splitlines() if "Montanari" in line)
    assert "](" not in montanari

    cache_file = tmp_path / "state" / "openalex_cache.json"
    assert cache_file.is_file()
    cache = json.loads(cache_file.read_text(encoding="utf-8"))
    assert sorted(entry["url"] for entry in cache.values()) == [
        "https://doi.org/10.1006/exeh.2001.0775",
        "https://openalex.org/W1000000002",
    ]
    assert not (tmp_path / "home" / ".autoexcerpter").exists()
    golden(_case("pdf_openalex_enrichment"), result)


# ============================================================================
# Working files
# ============================================================================
DOC = item_paths("doc", Path())
WORKING_LOGS = {DOC.transcription_log.name, DOC.summary_log.name}


@pytest.mark.parametrize(
    ("keep", "force", "expected_logs"),
    [
        pytest.param(True, False, WORKING_LOGS, id="pdf_keep_working_files"),
        pytest.param(True, True, WORKING_LOGS, id="pdf_keep_working_files_force"),
        pytest.param(False, False, WORKING_LOGS, id="pdf_cleanup_without_force"),
        pytest.param(False, True, set(), id="pdf_cleanup_force"),
    ],
)
def test_pdf_working_files(
    tmp_path: Path,
    docs_dir: Path,
    fake_llm: FakeLLM,
    golden: Golden,
    request: pytest.FixtureRequest,
    keep: bool,
    force: bool,
    expected_logs: set[str],
) -> None:
    pdf = make_pdf(docs_dir / "doc.pdf", pages=3)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        output_dir=tmp_path / "out",
        keep_working_files=keep,
        force=force,
    )

    _assert_items_complete(result, 1)
    working_dir = result.paths("doc").working_dir
    logs = {path.name for path in result.created if path.parent == working_dir}
    assert logs == expected_logs
    golden(_case(request.node.callspec.id), result)
