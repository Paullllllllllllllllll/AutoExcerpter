"""Tests for common/options.py: parser generation, round trip, resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoexcerpter.common.options import (
    Condition,
    Kind,
    Option,
    OptionError,
    OptionTable,
    Resolved,
    Source,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("must be > 0")
    return parsed


def _table() -> OptionTable:
    return OptionTable(
        [
            Option("path", "--path", "input path", type=Path, target="io.path"),
            Option(
                "out",
                "--out",
                "output folder",
                type=Path,
                exclusive="where",
                target="io.out",
            ),
            Option(
                "beside", "--beside", "beside input", kind=Kind.FLAG, exclusive="where"
            ),
            Option(
                "formats",
                "--formats",
                "formats",
                kind=Kind.LIST,
                choices=("a", "b", "c"),
                default=("a", "b", "c"),
                target="io.formats",
            ),
            Option(
                "summarize",
                "--summarize",
                "summarize",
                kind=Kind.SWITCH,
                default=True,
                target="io.summarize",
            ),
            Option("provider", "--provider", "provider", choices=("x", "y")),
            Option(
                "first_provider",
                "--first-provider",
                "provider of phase one",
                choices=("x", "y"),
                default="x",
                fallback="provider",
                target="first.provider",
            ),
            Option(
                "tier",
                "--tier",
                "tier",
                default="flex",
                condition=Condition("first_provider", frozenset({"x"})),
                target="first.tier",
            ),
            Option(
                "workers",
                "--workers",
                "workers",
                type=_positive,
                default=4,
                target="run.workers",
            ),
            Option("dry", "--dry", "dry run", kind=Kind.FLAG, default=False),
            Option("config", "--config", "config file", type=Path, settable=False),
        ]
    )


def test_absent_options_leave_no_value() -> None:
    assert _table().parse([]) == {}


def test_parse_converts_values() -> None:
    values = _table().parse(
        ["--path", "a.pdf", "--formats", "c, a", "--no-summarize", "--workers", "3"]
    )
    assert values == {
        "path": Path("a.pdf"),
        "formats": ("a", "c"),
        "summarize": False,
        "workers": 3,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["--workers", "0"],
        ["--workers", "x"],
        ["--formats", "a,z"],
        ["--provider", "z"],
        ["--out", "d", "--beside"],
    ],
)
def test_invalid_arguments_exit_with_status_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        _table().parse(argv)
    assert info.value.code == 2


@pytest.mark.parametrize(
    "values",
    [
        {"path": Path("in.pdf")},
        {"formats": ("b",), "summarize": True, "dry": True},
        {"formats": (), "summarize": False},
        {"first_provider": "y", "tier": "priority", "workers": 7},
        {"out": Path("out dir"), "config": Path("c.yaml")},
        {"beside": True, "provider": "y"},
    ],
)
def test_to_argv_round_trips(values: dict[str, object]) -> None:
    table = _table()
    assert table.parse(table.to_argv(values)) == values


def test_to_argv_rejects_unknown_names() -> None:
    with pytest.raises(OptionError):
        _table().to_argv({"nope": 1})


def test_false_flag_emits_nothing() -> None:
    assert _table().to_argv({"dry": False}) == []


def test_resolve_precedence_and_sources() -> None:
    resolved = _table().resolve(
        {"workers": 2}, {"workers": 9, "summarize": False, "tier": "default"}
    )
    assert resolved["workers"] == Resolved(2, Source.FLAG)
    assert resolved["summarize"] == Resolved(False, Source.SETTINGS)
    assert resolved["tier"] == Resolved("default", Source.SETTINGS)
    assert resolved["formats"] == Resolved(("a", "b", "c"), Source.DEFAULT)


def test_fallback_is_searched_per_level() -> None:
    table = _table()
    assert table.resolve({"provider": "y"})["first_provider"] == Resolved(
        "y", Source.FLAG
    )
    both = table.resolve({"provider": "y", "first_provider": "x"})
    assert both["first_provider"] == Resolved("x", Source.FLAG)
    flag_over_setting = table.resolve({"provider": "y"}, {"first_provider": "x"})
    assert flag_over_setting["first_provider"] == Resolved("y", Source.FLAG)
    setting = table.resolve({}, {"provider": "y"})
    assert setting["first_provider"] == Resolved("y", Source.SETTINGS)


def test_exclusive_options_keep_only_the_highest_level() -> None:
    resolved = _table().resolve({"beside": True}, {"out": Path("x")})
    assert resolved["out"] == Resolved(None, Source.FLAG)
    assert resolved["beside"] == Resolved(True, Source.FLAG)
    kept = _table().resolve({"out": Path("y")}, {})
    assert kept["out"] == Resolved(Path("y"), Source.FLAG)


def test_check_defaults_converts_native_values() -> None:
    checked = _table().check_defaults(
        {"workers": 5, "formats": ["b", "a"], "path": "in.pdf", "summarize": False}
    )
    assert checked == {
        "workers": 5,
        "formats": ("a", "b"),
        "path": Path("in.pdf"),
        "summarize": False,
    }


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": 1},
        {"config": "c.yaml"},
        {"workers": -1},
        {"workers": True},
        {"summarize": "yes"},
        {"formats": "a,z"},
        {"out": "d", "beside": True},
    ],
)
def test_check_defaults_rejects_invalid_blocks(raw: dict[str, object]) -> None:
    with pytest.raises(OptionError):
        _table().check_defaults(raw)


def test_by_targets_returns_values_and_sources() -> None:
    table = _table()
    values, sources = table.by_targets(table.resolve({"workers": 2}))
    assert values["run.workers"] == 2
    assert sources["run.workers"] is Source.FLAG
    assert sources["io.formats"] is Source.DEFAULT
    assert "dry" not in values


def test_condition_and_inapplicable() -> None:
    table = _table()
    assert table.inapplicable({"first_provider": "x", "tier": "flex"}) == []
    assert table.inapplicable({"first_provider": "y", "tier": "flex"}) == ["tier"]
    assert "only for first-provider x" in table["tier"].help_text()


def test_help_lists_defaults_and_list_choices() -> None:
    help_text = _table().build_parser().format_help()
    assert "default: 4" in help_text
    assert "from a, b, c" in help_text
    assert "--summarize, --no-summarize" in help_text


HYPHENATED = (
    "--summary-max-output-tokens",
    "summary-docx",
    "--transcription-reasoning-effort",
    "summary-md",
)


def _hyphen_table() -> OptionTable:
    return OptionTable(
        [
            Option(
                "summary_max_output_tokens",
                "--summary-max-output-tokens",
                "cap on the tokens of each summary reply; overrides "
                "--transcription-reasoning-effort budgets and applies to "
                "summary-docx and summary-md alike when --summary-max-output-tokens "
                "is given twice",
                type=int,
                group="model",
            ),
            Option(
                "formats",
                "--formats",
                "outputs to write next to the transcription",
                kind=Kind.LIST,
                choices=("summary-md", "summary-docx", "sqlite"),
                default=("summary-md", "summary-docx", "sqlite"),
                group="outputs",
            ),
        ],
        group_titles={"model": "Model", "outputs": "Outputs"},
    )


EPILOG = (
    "Examples:\n"
    "  tool run --input a.pdf --formats summary-md,summary-docx\n"
    "      --summary-max-output-tokens 4000\n"
    "\n"
    "A long closing paragraph that mentions --summary-max-output-tokens and "
    "summary-docx so that it needs to wrap at every tested width."
)


@pytest.mark.parametrize("width", [60, 80, 100])
def test_help_never_breaks_words_at_hyphens(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", str(width))
    help_text = _hyphen_table().build_parser(prog="tool", epilog=EPILOG).format_help()
    lines = help_text.splitlines()

    assert all(not line.rstrip().endswith("-") for line in lines)
    words = {word.strip(",;:()") for line in lines for word in line.split()}
    assert set(HYPHENATED) <= words
    assert "  tool run --input a.pdf --formats summary-md,summary-docx" in lines
    assert "      --summary-max-output-tokens 4000" in lines
    assert "Examples:" in lines
    closing = lines.index("Examples:") + 4
    assert lines[closing - 1] == ""
    assert lines[closing].startswith("A long closing paragraph")


def test_subset_keeps_order_and_group_titles() -> None:
    table = _hyphen_table()
    subset = table.subset(["formats"])

    assert subset.names == ("formats",)
    assert subset["formats"] is table["formats"]
    assert "Outputs:" in subset.build_parser().format_help()
    assert _table().subset(["workers", "path"]).names == ("path", "workers")


@pytest.mark.parametrize("names", [["nope"], ["first_provider"]])
def test_subset_rejects_unknown_or_dangling_names(names: list[str]) -> None:
    with pytest.raises(OptionError):
        _table().subset(names)


@pytest.mark.parametrize(
    "options",
    [
        [Option("a", "--a", "a"), Option("a", "--b", "b")],
        [Option("a", "--a", "a"), Option("b", "--a", "b")],
        [Option("a", "-a", "a")],
        [Option("a", "--a", "a", fallback="missing")],
    ],
)
def test_invalid_tables_are_rejected(options: list[Option]) -> None:
    with pytest.raises(OptionError):
        OptionTable(options)
