"""Read-back snapshots of the transcription, Markdown, DOCX and SQLite writers.

One fixed set of page results covers the page kinds (roman, arabic,
unnumbered, spread, blank, error, bibliography, figures), Markdown emphasis,
inline and display LaTeX with an unconvertible formula, currency dollars,
references with and without identifiers, and an XML-illegal heading
character. The real writers render it; the outputs are read back, normalized
and compared with ``golden/readback/``. ``AE_UPDATE_GOLDEN=1`` rewrites the
goldens.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import docx

from autoexcerpter.common.testing import WriteGuard
from autoexcerpter.common.usage import Usage, UsageTotals
from autoexcerpter.rendering.citations import CitationManager
from autoexcerpter.rendering.docx import create_docx_summary
from autoexcerpter.rendering.markdown import create_markdown_summary
from autoexcerpter.rendering.sqlite import (
    DATABASE_NAME,
    DocumentRecord,
    RunInfo,
    open_database,
    write_document,
)
from autoexcerpter.rendering.summary import build_render_context
from autoexcerpter.rendering.transcription import write_transcription_to_text
from tests.characterization.golden import (
    UPDATE_ENV,
    Normalizer,
    compare,
    dump_docx,
    dump_sqlite,
)

GOLDEN = Path(__file__).resolve().parent / "golden" / "readback"
NAME = "rich"

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
_PREFIXES = {_W: "w:", _M: "m:"}

SPEC: dict[str, dict[str, Any]] = {
    "input.path": {"value": "rich.pdf", "source": "flag"},
    "summary_model.model": {"value": "gpt-5-mini", "source": "settings"},
}

# A citation the test enriches by hand, as an OpenAlex match would.
ENRICHED = "Lancet, T. (2000). A controlled trial of rationing. The Lancet, 355."
ENRICHED_METADATA = {
    "doi": "10.1016/S0140-6736(00)12345-6",
    "url": "https://doi.org/10.1016/S0140-6736(00)12345-6",
    "publication_year": 2000,
}

INLINE_MATH = [
    r"Fraction $\frac{a}{b}$ inline.",
    r"Sum $\sum_{i=1}^{n} x_i$ inline.",
    r"Scripts $x_{t}^{2}$ inline.",
    r"Matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ inline.",
    r"Aligned $\begin{aligned} a &= b \\ c &= d \end{aligned}$ inline.",
    r"Unconvertible $\frac{a}{$ inline.",
]
DISPLAY_MATH = [
    r"Fraction $$\frac{p}{q}$$ display.",
    r"Sum $$\sum_{k=0}^{\infty} r^k$$ display.",
    r"Scripts $$y_{i,t}^{-1}$$ display.",
    r"Matrix $$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$ display.",
    r"Aligned $$\begin{aligned} w &= p \cdot q \\ v &= w - c \end{aligned}$$ "
    "display.",
    r"Unconvertible $$x^$$ display.",
]


# ============================================================================
# Fixture
# ============================================================================
def _info(
    number: Any,
    kind: str,
    types: list[str],
    *,
    end: int | None = None,
) -> dict[str, Any]:
    return {
        "page_number_integer": number,
        "page_number_type": kind,
        "page_types": types,
        "is_two_page_spread": end is not None,
        "page_number_integer_end": end,
    }


def _summary(
    index: int,
    info: dict[str, Any],
    bullets: list[str],
    references: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "original_input_order_index": index,
        "page": index + 1,
        "page_information": info,
        "bullet_points": bullets,
        "references": references,
        **extra,
    }


def _page(index: int, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "original_input_order_index": index,
        "image": f"page_{index + 1:04d}.jpg",
        "page_index": index,
        "transcription": text,
        **extra,
    }


def _ref(text: str, partial: bool = False) -> dict[str, Any]:
    return {"citation": text, "is_partial": partial}


ALLEN = (
    "Allen, R. C. (2001). The great divergence in European wages and prices. "
    "*Explorations in Economic History*, 38(4), 411-447. "
    "doi:10.1006/exeh.2001.0775"
)
CLARK = (
    "Clark, G. (2005). The condition of the working class in England. "
    "Retrieved from https://example.org/clark-2005"
)
BRAUDEL = (
    "Braudel, F. (1979). Civilisation matérielle, économie et capitalisme. "
    "Paris: Armand Colin."
)
MUELLER = "Müller, J., & Østergaard, K. (1998). Über die Preise in Zürich. Basel."
DVORAK = "Dvořák, A. (1890). Ceny obilí v Čechách. Praha: Šimáček."

TRANSCRIPTIONS = [
    _page(0, "<page_number>iii</page_number>\nPREFACE\nThe author thanks Ø."),
    _page(1, "<page_number>1</page_number>\nWages rose from $3 to $5; $x^2$."),
    _page(2, "<page_number>2</page_number>\n$$\\frac{p}{q}$$ as printed."),
    _page(3, "An unnumbered leaf without a folio."),
    _page(4, "<page_number>4</page_number> Left.\n<page_number>5</page_number>"),
    _page(5, "[empty page]"),
    _page(
        6,
        "[Transcription error: rate limit]",
        error="Rate limit exceeded",
        error_type="api",
    ),
    _page(7, "<page_number>10</page_number>\nBIBLIOGRAPHY"),
    _page(8, "<page_number>11</page_number>\nTable 3. Prices."),
]

SUMMARIES = [
    _summary(
        0,
        _info(3, "roman", ["preface"]),
        [
            "**Bold claim** with *italic aside* and ***both at once***.",
            "A product a * b * c stays literal.",
            "A bullet split\n   across two lines.",
        ],
    ),
    _summary(
        1,
        _info(1, "arabic", ["content"]),
        [
            *INLINE_MATH,
            "Wages rose from $3 to $5 per week, and bread cost $0.25.",
            r"A fee of \$10 and the ratio $\frac{w}{p}$ at a price of $5.",
        ],
        [_ref(ALLEN), _ref(ENRICHED)],
    ),
    _summary(
        2,
        _info(2, "arabic", ["content"]),
        [*DISPLAY_MATH, r"Inline $z^2$ beside display $$z_0$$ in one bullet."],
        [_ref("Allen (2001)", partial=True)],
    ),
    _summary(
        3,
        _info(None, "none", ["content"]),
        ["An unnumbered page cites a Czech source."],
        [_ref(DVORAK)],
    ),
    _summary(
        4,
        _info(4, "arabic", ["content"], end=5),
        ["A two-page spread on Zürich prices."],
        [_ref(MUELLER)],
    ),
    _summary(5, _info(None, "none", ["blank"]), ["[Empty page]"]),
    _summary(
        6,
        _info(None, "none", ["other"]),
        ["[Error generating summary: Rate limit exceeded]"],
        error="Rate limit exceeded",
    ),
    _summary(
        7,
        _info(10, "arabic", ["bibliography"]),
        [],
        [_ref(ALLEN), _ref(CLARK), _ref(BRAUDEL), _ref(MUELLER), _ref(DVORAK)],
    ),
    _summary(
        8,
        _info("11\x0b", "arabic", ["figures_tables_sources"]),
        ["Table 3 lists grain prices in $ and £."],
    ),
]


# ============================================================================
# Writers
# ============================================================================
def _citation_manager() -> tuple[CitationManager, Any]:
    manager, data = build_render_context(SUMMARIES, "pdf", openalex_enabled=False)
    (enriched,) = (c for c in manager.citations.values() if c.raw_text == ENRICHED)
    enriched.metadata = dict(ENRICHED_METADATA)
    enriched.doi = str(ENRICHED_METADATA["doi"])
    enriched.url = str(ENRICHED_METADATA["url"])
    return manager, data


def _usage(tokens: int) -> UsageTotals:
    totals = UsageTotals()
    totals.add("transcription", Usage(total_tokens=tokens))
    totals.add("summary", Usage(total_tokens=tokens // 10))
    return totals


def _write_sqlite(folder: Path, manager: CitationManager) -> Path:
    record = DocumentRecord(
        name=NAME,
        path=folder / "rich.pdf",
        kind="pdf",
        hash="0" * 64,
        transcription_model="gpt-5-mini",
        summary_model="gpt-5-mini",
        complete=True,
        report={
            "pages_total": len(TRANSCRIPTIONS),
            "pages_ok": len(TRANSCRIPTIONS) - 1,
            "pages_failed": 1,
            "outputs": [str(folder / f"{NAME}_summary.md")],
        },
        pages=TRANSCRIPTIONS,
        summaries=SUMMARIES,
        citations=manager.get_sorted_citations(),
        usage={i: _usage(100 * (i + 1)) for i in range(len(TRANSCRIPTIONS))},
    )

    async def main() -> None:
        async with open_database(folder) as writer:
            await write_document(writer, record, RunInfo(SPEC, "9.9.9"))

    asyncio.run(main())
    return folder / DATABASE_NAME


def _local(tag: str) -> str:
    for namespace, prefix in _PREFIXES.items():
        if tag.startswith(namespace):
            return prefix + tag[len(namespace) :]
    return tag


def _dump_math_element(element: Any, depth: int, lines: list[str]) -> None:
    attributes = " ".join(
        f'{_local(key)}="{value}"' for key, value in sorted(element.attrib.items())
    )
    line = "  " * depth + _local(element.tag)
    if attributes:
        line += f" [{attributes}]"
    if element.tag == _M + "t":
        line += f" {element.text or ''!r}"
    lines.append(line)
    for child in element:
        _dump_math_element(child, depth + 1, lines)


def dump_omml(path: Path) -> str:
    """Return the element tree of every paragraph-level OMML element."""
    body = docx.Document(str(path)).element.body
    lines: list[str] = []
    for number, paragraph in enumerate(body.iter(_W + "p")):
        for child in paragraph:
            if child.tag in (_M + "oMath", _M + "oMathPara"):
                lines.append(f"paragraph {number}:")
                _dump_math_element(child, 1, lines)
    return "\n".join(lines) + "\n"


def _render(folder: Path) -> dict[str, str]:
    norm = Normalizer(folder)
    manager, data = _citation_manager()
    transcription = folder / f"{NAME}.txt"
    markdown = folder / f"{NAME}_summary.md"
    document = folder / f"{NAME}_summary.docx"

    assert write_transcription_to_text(
        TRANSCRIPTIONS, transcription, NAME, "PDF", 125.0, folder / "rich.pdf"
    )
    create_markdown_summary(SUMMARIES, markdown, NAME, manager, data)
    create_docx_summary(SUMMARIES, document, NAME, manager, data)
    database = _write_sqlite(folder, manager)

    return {
        "transcription.txt": norm.text(transcription.read_text(encoding="utf-8")),
        "summary.md": norm.text(markdown.read_text(encoding="utf-8")),
        "summary.docx.dump.txt": norm.text(dump_docx(document)),
        "summary.docx.omml.txt": dump_omml(document),
        "autoexcerpter.sqlite.dump.txt": dump_sqlite(database, norm),
    }


# ============================================================================
# Golden comparison
# ============================================================================
def _read_golden() -> dict[str, str]:
    if not GOLDEN.is_dir():
        return {}
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(GOLDEN.iterdir())
        if path.is_file()
    }


def _write_golden(files: dict[str, str], write_guard: WriteGuard) -> None:
    if GOLDEN.exists():
        shutil.rmtree(GOLDEN)
    GOLDEN.mkdir(parents=True)
    for name, content in files.items():
        (GOLDEN / name).write_text(content, encoding="utf-8", newline="\n")
    write_guard.consume(GOLDEN.parent)


def test_writers_read_back_match_the_goldens(
    tmp_path: Path, write_guard: WriteGuard
) -> None:
    folder = tmp_path / "out"
    folder.mkdir()
    actual = _render(folder)

    if os.environ.get(UPDATE_ENV) == "1":
        _write_golden(actual, write_guard)
        return
    expected = _read_golden()
    assert expected, f"No goldens in {GOLDEN}; run with {UPDATE_ENV}=1 to create."
    problems = compare(expected, actual)
    assert not problems, "Read-back mismatch:\n" + "\n".join(problems)
