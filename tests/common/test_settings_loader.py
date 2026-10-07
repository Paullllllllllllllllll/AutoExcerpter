"""Tests for common/settings_loader.py: file resolution and the defaults block."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from autoexcerpter.common.options import Kind, Option, OptionTable
from autoexcerpter.common.settings_loader import (
    SettingsError,
    load_settings,
    locate_settings,
)

_OPTIONS = OptionTable(
    [
        Option("model", "--model", "model name", default="m"),
        Option("workers", "--workers", "workers", type=int, default=4),
        Option("fast", "--fast", "fast mode", kind=Kind.SWITCH, default=False),
    ]
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_real_file_is_read_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    real = _write(tmp_path / "settings.yaml", "a: 1\n")
    example = _write(tmp_path / "settings.example.yaml", "a: 2\n")
    with caplog.at_level(logging.DEBUG):
        loaded = load_settings(real, example=example)
    assert loaded.path == real
    assert not loaded.from_example
    assert loaded.data == {"a": 1}
    assert caplog.records == []


def test_example_is_read_with_one_info_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    real = tmp_path / "settings.yaml"
    example = _write(tmp_path / "settings.example.yaml", "a: 2\n")
    with caplog.at_level(logging.DEBUG):
        loaded = load_settings(real, example=example)
    assert loaded.from_example
    assert loaded.data == {"a": 2}
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "settings.yaml" in caplog.records[0].getMessage()


def test_missing_files_raise_an_error_naming_the_real_file(tmp_path: Path) -> None:
    real = tmp_path / "settings.yaml"
    with pytest.raises(SettingsError, match="settings.yaml"):
        load_settings(real, example=tmp_path / "settings.example.yaml")


def test_explicit_path_wins(tmp_path: Path) -> None:
    real = _write(tmp_path / "settings.yaml", "a: 1\n")
    explicit = _write(tmp_path / "other.yaml", "a: 3\n")
    assert locate_settings(real, None, explicit) == (explicit, False)
    assert load_settings(real, explicit=explicit).data == {"a": 3}


def test_missing_explicit_path_is_an_error(tmp_path: Path) -> None:
    real = _write(tmp_path / "settings.yaml", "a: 1\n")
    with pytest.raises(SettingsError, match="other.yaml"):
        load_settings(real, explicit=tmp_path / "other.yaml")


def test_defaults_block_is_split_and_converted(tmp_path: Path) -> None:
    real = _write(
        tmp_path / "settings.yaml",
        "a: 1\ndefaults:\n  model: big\n  workers: '8'\n  fast: true\n",
    )
    loaded = load_settings(real, options=_OPTIONS)
    assert loaded.data == {"a": 1}
    assert loaded.defaults == {"model": "big", "workers": 8, "fast": True}


def test_unknown_defaults_key_is_an_error(tmp_path: Path) -> None:
    real = _write(tmp_path / "settings.yaml", "defaults:\n  modle: big\n")
    with pytest.raises(SettingsError, match="modle"):
        load_settings(real, options=_OPTIONS)


def test_empty_file_and_null_defaults(tmp_path: Path) -> None:
    empty = load_settings(_write(tmp_path / "settings.yaml", ""), options=_OPTIONS)
    assert empty.data == {}
    assert empty.defaults == {}
    null = load_settings(_write(tmp_path / "s2.yaml", "defaults:\n"), options=_OPTIONS)
    assert null.defaults == {}


@pytest.mark.parametrize(
    "text",
    ["a: [1\n", "- 1\n- 2\n", "defaults: [model]\n"],
)
def test_invalid_files_raise_an_error_naming_the_file(
    tmp_path: Path, text: str
) -> None:
    real = _write(tmp_path / "settings.yaml", text)
    with pytest.raises(SettingsError, match="settings.yaml"):
        load_settings(real, options=_OPTIONS)
