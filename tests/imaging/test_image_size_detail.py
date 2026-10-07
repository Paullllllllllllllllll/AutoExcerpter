"""Tests for the OpenAI per-image ``detail`` run option (``--image-detail``).

Covers the local-preprocessing interaction (``imaging.payload``) that skips
resizing when full-resolution ("original") detail is requested.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

import autoexcerpter.imaging.settings as image_settings
from autoexcerpter.imaging.payload import FolderPayloadSource
from autoexcerpter.spec import ImageSpec

_SECTIONS = {
    "api_image_processing": {
        "grayscale_conversion": True,
        "handle_transparency": True,
        "llm_detail": "high",
        "resize_profile": "high",
        "high_target_box": [768, 1536],
        "jpeg_quality": 90,
    }
}


class TestPayloadOriginalResizeSkip:
    """Local preprocessing skips resize when 'original' detail is active."""

    @pytest.fixture
    def image_folder(self, tmp_path: Path) -> Path:
        folder = tmp_path / "scans"
        folder.mkdir()
        Image.new("RGB", (1000, 1500), color=(120, 120, 120)).save(
            folder / "page_0001.jpg"
        )
        return folder

    def _patched(self) -> Any:
        return patch.object(image_settings, "IMAGE_PROCESSING", _SECTIONS)

    def test_resize_skipped_keeps_native_size(self, image_folder: Path) -> None:
        """'original' on GPT-5.6 keeps native dimensions (under the caps)."""
        with self._patched():
            source = FolderPayloadSource(
                image_folder,
                provider="openai",
                model_name="gpt-5.6-sol",
                images=ImageSpec(detail="original"),
            )
            # Routed through the capped 'original' path, not resize_profile none.
            assert source.img_cfg["llm_detail"] == "original"
            assert source.img_cfg["resize_profile"] == "high"
            payload = source.build_payload(0)
        # 1000x1500 is well under the caps, so native size is preserved
        # (not padded to the 768x1536 box).
        assert payload.provenance["width"] == 1000
        assert payload.provenance["height"] == 1500

    def test_oversized_original_is_capped(self, tmp_path: Path) -> None:
        """An oversized 'original' scan is capped to max side and pixel budget."""
        folder = tmp_path / "big"
        folder.mkdir()
        Image.new("RGB", (7000, 9000), color=(120, 120, 120)).save(
            folder / "page_0001.jpg"
        )
        with self._patched():
            source = FolderPayloadSource(
                folder,
                provider="openai",
                model_name="gpt-5.6-sol",
                images=ImageSpec(detail="original"),
            )
            payload = source.build_payload(0)
        width = payload.provenance["width"]
        height = payload.provenance["height"]
        # GPT-5.6 uses a 30,000-patch budget with a 65,535 px edge.
        assert max(width, height) <= 65535
        assert math.ceil(width / 32) * math.ceil(height / 32) <= 30000
        # Aspect ratio preserved (7000:9000 == 7:9), within rounding.
        assert abs((width / height) - (7000 / 9000)) < 0.01

    def test_resize_applied_when_not_original(self, image_folder: Path) -> None:
        """Without 'original', the usual box-fit resize still applies."""
        with self._patched():
            source = FolderPayloadSource(
                image_folder,
                provider="openai",
                model_name="gpt-5.6-sol",
                images=ImageSpec(detail="high"),
            )
            assert source.img_cfg.get("resize_profile") == "high"
            payload = source.build_payload(0)
        # Box-fit pads to the configured 768x1536 target.
        assert payload.provenance["width"] == 768
        assert payload.provenance["height"] == 1536
