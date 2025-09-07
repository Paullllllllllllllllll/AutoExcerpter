from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Resolve project root as the parent of this modules directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULES_DIR = Path(__file__).resolve().parent
CONFIG_DIR = MODULES_DIR / "config"
PROMPTS_DIR = MODULES_DIR / "prompts"
SCHEMAS_DIR = MODULES_DIR / "schemas"


class ConfigLoader:
    """Lightweight loader for YAML/JSON configs residing under modules/."""

    def __init__(self) -> None:
        self._image_processing: Dict[str, Any] | None = None
        self._concurrency: Dict[str, Any] | None = None
        self._model: Dict[str, Any] | None = None

    def load_configs(self) -> None:
        # Load image processing config
        img_cfg_path = CONFIG_DIR / "image_processing.yaml"
        if img_cfg_path.exists():
            try:
                with img_cfg_path.open("r", encoding="utf-8") as f:
                    self._image_processing = yaml.safe_load(f) or {}
            except Exception:
                self._image_processing = {}
        else:
            self._image_processing = {}

        # Load concurrency config
        conc_path = CONFIG_DIR / "concurrency.yaml"
        if conc_path.exists():
            try:
                with conc_path.open("r", encoding="utf-8") as f:
                    self._concurrency = yaml.safe_load(f) or {}
            except Exception:
                self._concurrency = {}
        else:
            self._concurrency = {}

        # Load model config if present (optional)
        model_cfg_path = CONFIG_DIR / "model.yaml"
        if model_cfg_path.exists():
            try:
                with model_cfg_path.open("r", encoding="utf-8") as f:
                    self._model = yaml.safe_load(f) or {}
            except Exception:
                self._model = {}
        else:
            self._model = {}

    def get_image_processing_config(self) -> Dict[str, Any]:
        return dict(self._image_processing or {})

    def get_concurrency_config(self) -> Dict[str, Any]:
        return dict(self._concurrency or {})

    def get_model_config(self) -> Dict[str, Any]:
        return dict(self._model or {})
