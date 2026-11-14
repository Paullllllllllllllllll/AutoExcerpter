"""Schema folders through the CLI: replacement, refusal and the recorded hash.

A run with ``--schemas-dir`` sends and stores the replacement schema and
records its hash with the run; a replacement that breaks a shape the code
reads stops ``run`` and ``batch check`` with exit code 2 before any request.
Under batch mode a document finalized under one schema is refused when a
later run or check would add units under another.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from autoexcerpter.llm.prompts import SCHEMAS_DIR
from autoexcerpter.llm.schemas import SUMMARY_SCHEMA_FILE, TRANSCRIPTION_SCHEMA_FILE
from autoexcerpter.output.store import DATABASE_NAME
from tests.characterization.adapter import run_batch_command, run_tool
from tests.characterization.fakes import SUMMARY, TRANSCRIPTION, FakeLLM, Injection
from tests.characterization.fakes_batch import FakeBatchService, permanent_error
from tests.characterization.inputs import make_pdf

FAILED_PAGE = 1
BATCH_OPTIONS: dict[str, Any] = {
    "transcription_provider": "openai",
    "transcription_model": "gpt-5.6-terra",
    "summary_provider": "openai",
    "summary_model": "gpt-5.6-terra",
    "batch": True,
}


def _folder(root: Path, file_name: str, **changes: Any) -> Path:
    """Write a copy of the packaged *file_name* with *changes* to its body."""
    folder = root / "schemas" / file_name.removesuffix(".json")
    folder.mkdir(parents=True, exist_ok=True)
    data = json.loads((SCHEMAS_DIR / file_name).read_text(encoding="utf-8"))
    data["schema"]["description"] = "A project's own instructions."
    data["schema"].update(changes)
    (folder / file_name).write_text(json.dumps(data), encoding="utf-8")
    return folder


def _rows(database: Path, sql: str) -> list[tuple[Any, ...]]:
    with contextlib.closing(sqlite3.connect(database)) as db:
        return db.execute(sql).fetchall()


def test_a_replacement_is_sent_stored_and_named_by_the_run_record(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=2)
    folder = _folder(tmp_path, TRANSCRIPTION_SCHEMA_FILE)

    result = run_tool(
        pdf, llm=fake_llm, root=tmp_path, extra=("--schemas-dir", str(folder))
    )

    assert result.exit_code == 0, result.stderr
    database = tmp_path / "out" / DATABASE_NAME
    artifacts = dict(_rows(database, "SELECT kind, sha256 FROM artifacts"))
    ((spec,),) = _rows(database, "SELECT spec FROM runs")
    record = json.loads(spec)
    for kind in ("transcription", "summary"):
        entry = record[f"{kind}_schema_sha256"]
        assert entry == {"value": artifacts[f"{kind}_schema"], "source": "computed"}
    ((text,),) = _rows(
        database, "SELECT text FROM artifacts WHERE kind = 'transcription_schema'"
    )
    assert "A project's own instructions." in text
    assert "--schemas-dir" not in json.dumps(record)


def test_a_replacement_that_breaks_a_read_shape_exits_2_before_any_request(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=1)
    folder = _folder(tmp_path, SUMMARY_SCHEMA_FILE, required=["bullet_points"])

    result = run_tool(
        pdf, llm=fake_llm, root=tmp_path, extra=("--schemas-dir", str(folder))
    )

    assert result.exit_code == 2
    assert str(folder / SUMMARY_SCHEMA_FILE) in result.stderr
    assert "page_information is not in the required list" in result.stderr
    assert fake_llm.calls == []
    assert not (tmp_path / "out" / DATABASE_NAME).exists()


def test_a_schema_folder_that_does_not_exist_is_a_usage_error(
    tmp_path: Path, fake_llm: FakeLLM
) -> None:
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=1)

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        extra=("--schemas-dir", str(tmp_path / "missing")),
    )

    assert result.exit_code == 2
    assert "not a folder" in result.stderr


def _finalized(tmp_path: Path, llm: FakeLLM) -> Path:
    """Submit a three-page PDF whose page 1 fails, and check it twice."""
    pdf = make_pdf(tmp_path / "docs" / "doc.pdf", pages=3)
    llm.inject(FAILED_PAGE, TRANSCRIPTION, 1, Injection(exception=permanent_error()))
    assert run_tool(pdf, llm=llm, root=tmp_path, **BATCH_OPTIONS).exit_code == 0
    for _ in range(2):
        run_batch_command("check", llm=llm, root=tmp_path)
    return pdf


def test_a_run_after_finalize_under_another_schema_is_refused(
    tmp_path: Path, fake_llm: FakeLLM, fake_batch: FakeBatchService
) -> None:
    pdf = _finalized(tmp_path, fake_llm)
    database = tmp_path / "out" / DATABASE_NAME
    before = _rows(database, "SELECT kind, idx, status FROM units ORDER BY kind, idx")
    folder = _folder(tmp_path, TRANSCRIPTION_SCHEMA_FILE)
    mark = fake_batch.mark()

    result = run_tool(
        pdf,
        llm=fake_llm,
        root=tmp_path,
        extra=("--schemas-dir", str(folder)),
        **BATCH_OPTIONS,
    )

    assert result.exit_code == 1
    assert result.summary is not None and result.summary["items_failed"] == 1
    assert "The transcription schema changed for 'doc'" in result.stderr
    assert fake_batch.request_records(mark)["requests"] == []
    assert _rows(
        database, "SELECT kind, idx, status FROM units ORDER BY kind, idx"
    ) == (before)


def test_a_check_under_another_summary_schema_submits_no_summary(
    tmp_path: Path, fake_llm: FakeLLM, fake_batch: FakeBatchService
) -> None:
    pdf = _finalized(tmp_path, fake_llm)
    assert run_tool(pdf, llm=fake_llm, root=tmp_path, **BATCH_OPTIONS).exit_code == 0
    folder = _folder(tmp_path, SUMMARY_SCHEMA_FILE)
    mark = fake_batch.mark()

    result = run_batch_command(
        "check", llm=fake_llm, root=tmp_path, extra=("--schemas-dir", str(folder))
    )

    assert result.exit_code == 1
    assert result.summary is not None and result.summary["finalized"] == 1
    roles = [r["role"] for r in fake_batch.request_records(mark)["requests"]]
    assert SUMMARY not in roles
    assert "The summary schema changed for 'doc'" in result.stderr

    later = run_batch_command("check", llm=fake_llm, root=tmp_path)

    assert later.exit_code == 0, later.stderr
    assert later.summary is not None and later.summary["submitted"] == 1


def test_a_check_with_a_broken_replacement_exits_2(
    tmp_path: Path, fake_llm: FakeLLM, fake_batch: FakeBatchService
) -> None:
    (tmp_path / "out").mkdir()
    folder = _folder(tmp_path, SUMMARY_SCHEMA_FILE, type="array")

    result = run_batch_command(
        "check", llm=fake_llm, root=tmp_path, extra=("--schemas-dir", str(folder))
    )

    assert result.exit_code == 2
    assert str(folder / SUMMARY_SCHEMA_FILE) in result.stderr
