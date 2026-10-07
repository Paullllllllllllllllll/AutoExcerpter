"""Logging stays unconfigured when the project modules are imported."""

from __future__ import annotations

import importlib
import logging
import pkgutil

PROJECT_PACKAGES = (
    "autoexcerpter.cli",
    "autoexcerpter.imaging",
    "autoexcerpter.llm",
    "autoexcerpter.pipeline",
    "autoexcerpter.rendering",
)


def _own_handlers() -> list[logging.Handler]:
    """Return the root handlers that ``configure_logging`` would install."""
    return [
        handler
        for handler in logging.getLogger().handlers
        if not type(handler).__module__.startswith("_pytest")
    ]


def _is_project_logger(name: str) -> bool:
    return name == "autoexcerpter" or name.startswith("autoexcerpter.")


def test_importing_project_modules_configures_no_logging() -> None:
    importlib.import_module("autoexcerpter.__main__")
    for package_name in PROJECT_PACKAGES:
        package = importlib.import_module(package_name)
        for info in pkgutil.walk_packages(package.__path__, package_name + "."):
            importlib.import_module(info.name)

    assert _own_handlers() == []
    for name, entry in logging.Logger.manager.loggerDict.items():
        if not _is_project_logger(name) or not isinstance(entry, logging.Logger):
            continue
        assert entry.handlers == [], name
        assert entry.propagate is True, name
