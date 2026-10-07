"""The README Command Reference tables agree with the option table."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from autoexcerpter.common.options import Kind, Option, OptionTable
from autoexcerpter.spec import OPTIONS, PHASES
from tests.conftest import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
SECTION = "## Command Reference"
FLAG_HEADER = ["Flag", "Description", "Default", "In `defaults:`"]
MATRIX_HEADER = ["Setting", "Values", "Both phases", "Transcription", "Summary"]
SETTABLE_TEXT = {True: "yes", False: "no"}


def _section(text: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index(SECTION) + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return lines[start:end]


def _cells(line: str) -> list[str]:
    inner = line.strip()[1:-1]
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", inner)]


def _tables(lines: list[str]) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*lines, ""]:
        if line.startswith("|"):
            current.append(_cells(line))
        elif current:
            tables.append([current[0], *current[2:]])
            current = []
    return tables


def default_text(option: Option) -> str:
    """Return the default that ``--help`` prints for *option*, or ''."""
    if option.default_help is not None:
        return option.default_help
    if option.default is None or option.kind is Kind.FLAG:
        return ""
    if option.kind is Kind.SWITCH:
        return "on" if option.default else "off"
    if option.kind is Kind.LIST:
        return ",".join(str(item) for item in option.default)
    return str(option.default)


def description(option: Option) -> str:
    """Return the ``--help`` line of *option* without its default."""
    text = option.help_text()
    default = default_text(option)
    return text.replace(f"; default: {default}", "") if default else text


def _argument(option: Option) -> str:
    if option.kind in (Kind.SWITCH, Kind.FLAG):
        return ""
    if option.metavar is not None:
        return option.metavar
    return "{" + ",".join(str(choice) for choice in option.choices or ()) + "}"


def _flag_spellings(option: Option) -> list[str]:
    argument = _argument(option)
    if option.kind is Kind.SWITCH:
        return [option.flag, option.negative_flag]
    return [f"{option.flag} {argument}" if argument else option.flag]


def _check_flag_row(
    row: list[str], table: OptionTable, seen: Counter[str], problems: list[str]
) -> None:
    spans = re.findall(r"`([^`]*)`", row[0])
    flags = [span.split(" ")[0] for span in spans]
    seen.update(flags)
    by_flag = {option.flag: option for option in table}
    if not flags or flags[0] not in by_flag:
        problems.append(f"unknown flag row: {row[0]}")
        return
    option = by_flag[flags[0]]
    spellings = _flag_spellings(option)
    if spans != spellings:
        problems.append(f"{option.flag}: flag cell {spans} != {spellings}")
    if row[1] != description(option):
        problems.append(f"{option.flag}: description {row[1]!r}")
    default = default_text(option)
    if row[2] != default:
        problems.append(f"{option.flag}: default {row[2]!r} != {default!r}")
    if row[3] != SETTABLE_TEXT[option.settable]:
        problems.append(f"{option.flag}: defaults column {row[3]!r}")


def _check_matrix_row(
    row: list[str], table: OptionTable, seen: Counter[str], problems: list[str]
) -> None:
    by_flag = {option.flag: option for option in table}
    cells = [re.fullmatch(r"`(--[\w-]+)`: (.*)", cell) for cell in row[2:]]
    seen.update(match.group(1) for match in cells if match)
    first = cells[0]
    if first is None or first.group(1) not in by_flag:
        problems.append(f"unknown model row: {row[2]}")
        return
    base = by_flag[first.group(1)]
    if row[0] != description(base):
        problems.append(f"{base.flag}: setting {row[0]!r}")
    if row[1] != f"`{_argument(base)}`":
        problems.append(f"{base.flag}: values {row[1]!r}")
    expected = [base.flag, *(f"--{phase}-{base.flag[2:]}" for phase in PHASES)]
    for flag, match in zip(expected, cells, strict=True):
        if match is None or match.group(1) != flag:
            problems.append(f"{base.flag}: expected {flag} in its row")
            continue
        option = by_flag[flag]
        if match.group(2) != default_text(option):
            problems.append(f"{flag}: default {match.group(2)!r}")
        if not option.settable:
            problems.append(f"{flag}: the README says every model option is settable")


def table_drift(text: str, table: OptionTable = OPTIONS) -> list[str]:
    """Return every disagreement between the README tables and *table*."""
    problems: list[str] = []
    seen: Counter[str] = Counter()
    for header, *rows in _tables(_section(text)):
        for row in rows:
            if header == FLAG_HEADER:
                _check_flag_row(row, table, seen, problems)
            elif header == MATRIX_HEADER:
                _check_matrix_row(row, table, seen, problems)
    expected = [
        flag
        for option in table
        for flag in (
            [option.flag, option.negative_flag]
            if option.kind is Kind.SWITCH
            else [option.flag]
        )
    ]
    for flag in expected:
        if seen[flag] != 1:
            problems.append(f"{flag}: {seen[flag]} README rows")
    problems.extend(f"{flag}: not an option" for flag in seen if flag not in expected)
    return problems


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_tables_match_the_option_table() -> None:
    assert table_drift(_readme()) == []


def test_default_text_is_what_help_prints() -> None:
    for option in OPTIONS:
        default = default_text(option)
        if default:
            assert f"default: {default}" in option.help_text(), option.flag
        else:
            assert "default: " not in option.help_text(), option.flag


DRIFT = [
    pytest.param(
        "| `--concurrency N` | parallel page requests | 16 | yes |",
        "| `--concurrency N` | parallel page requests | 32 | yes |",
        "--concurrency: default",
        id="changed-default",
    ),
    pytest.param(
        "| `--json` | print one JSON summary line on stdout | | no |\n",
        "",
        "--json: 0 README rows",
        id="missing-row",
    ),
    pytest.param(
        "| `--force` | reprocess items whose outputs exist (resume is the default)"
        " | | no |",
        "| `--force` | reprocess items whose outputs exist (resume is the default)"
        " | | yes |",
        "--force: defaults column",
        id="flipped-defaults-column",
    ),
    pytest.param(
        "`--summary-verbosity`: low",
        "`--summary-verbosity`: medium",
        "--summary-verbosity: default",
        id="changed-matrix-default",
    ),
    pytest.param(
        "| `--summarize`, `--no-summarize` |",
        "| `--summarize` |",
        "--no-summarize: 0 README rows",
        id="missing-no-form",
    ),
    pytest.param(
        "| `--all` | process every discovered item | | yes |",
        "| `--all` | process every discovered item | | yes |\n| `--bogus` | x | | no |",
        "unknown flag row",
        id="unknown-flag",
    ),
    pytest.param(
        "| `--dry-run` | plan without API calls or writes | | no |",
        "| `--dry-run` | plan without any calls or writes | | no |",
        "--dry-run: description",
        id="changed-description",
    ),
]


@pytest.mark.parametrize(("old", "new", "message"), DRIFT)
def test_table_drift_is_reported(old: str, new: str, message: str) -> None:
    text = _readme()
    assert text.count(old) == 1
    problems = table_drift(text.replace(old, new))
    assert any(message in problem for problem in problems), problems
