"""The migrate subcommand: its JSON summary, report lines and exit codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoexcerpter import __main__ as entry
from autoexcerpter.output.store import DATABASE_NAME
from tests.migrate.legacy import (
    CITATION,
    Item,
    Page,
    make_item,
    write_files,
    write_logs,
)


@pytest.fixture
def item(tmp_path: Path, state_dir: Path) -> Item:
    """A complete version 1.0 item; the settings point at the OpenAlex cache."""
    legacy = make_item(
        tmp_path / "out",
        "doc",
        [Page("A.", references=(CITATION,)), Page("B.")],
        state_dir=state_dir,
    )
    write_logs(legacy)
    write_files(legacy)
    settings = tmp_path / "settings.yaml"
    settings.write_text(f"state_dir: '{state_dir.as_posix()}'\n", encoding="utf-8")
    return legacy


def _main(
    capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, dict[str, Any], str]:
    code = entry.main(["migrate", *argv, "--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out.strip().splitlines()[-1]), captured.err


def test_migrate_reports_what_it_imported(
    item: Item, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = str(tmp_path / "settings.yaml")
    code, summary, err = _main(
        capsys, "--input", str(item.root), "--settings", settings
    )

    assert code == 0
    assert summary["command"] == "migrate"
    assert (summary["databases"], summary["documents"]) == (1, 1)
    assert (summary["units"], summary["citations"], summary["outputs"]) == (4, 1, 3)
    assert summary["mismatches"] == 0
    assert "Resume state legacy -> complete: 1 document(s)" in err

    code, summary, _ = _main(capsys, "--input", str(item.root), "--settings", settings)

    assert code == 0
    assert (summary["units"], summary["citations"], summary["outputs"]) == (0, 0, 0)


def test_a_dry_run_reports_without_writing(
    item: Item, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, summary, err = _main(
        capsys,
        "--input",
        str(item.root),
        "--settings",
        str(tmp_path / "settings.yaml"),
        "--dry-run",
    )

    assert code == 0
    assert summary["dry_run"] is True
    assert summary["units"] == 4
    assert "DRY-RUN" in err
    assert not (item.root / DATABASE_NAME).exists()


def test_a_missing_input_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, summary, err = _main(capsys, "--input", str(tmp_path / "absent"))

    assert code == 2
    assert summary["units"] == 0
    assert "does not exist" in err


def test_a_parse_error_writes_the_migrate_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = entry.main(["migrate", "--bogus", "--json"])

    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out.strip())["command"] == "migrate"
