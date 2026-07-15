"""Tests for the OpenAI per-image ``detail`` (image_size) knob.

Covers the transcription-path wiring (``TranscriptionManager``) and the
local-preprocessing interaction (``imaging.payload``) that skips resizing when
full-resolution ("original") detail is requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imaging.payload import FolderPayloadSource, _openai_detail_is_original
from llm.transcription import TranscriptionManager


def _make_manager(
    model_name: str,
    model_config: dict[str, Any],
    provider: str = "openai",
) -> TranscriptionManager:
    """Build a minimal TranscriptionManager without running __init__.

    Sets only the attributes ``_build_model_inputs`` touches so the image
    block can be inspected without any real API key or network access.
    """
    mgr = TranscriptionManager.__new__(TranscriptionManager)
    mgr.provider = provider  # type: ignore[assignment]
    mgr.model_name = model_name
    mgr.model_config = model_config
    mgr.service_tier = "auto"
    mgr._output_schema = None
    mgr.custom_capabilities = None
    mgr.system_prompt = ""
    return mgr


def _image_block(mgr: TranscriptionManager) -> dict[str, Any]:
    """Return the OpenAI image_url dict from a built user message."""
    messages, _ = mgr._build_model_inputs("QUJD")  # base64 for "ABC"
    user_msg = messages[1]
    content = user_msg.content
    assert isinstance(content, list)
    block = content[0]
    assert block["type"] == "image_url"
    image_url: dict[str, Any] = block["image_url"]
    return image_url


class TestResolveImageDetail:
    """Transcription-path detail resolution and image-block wiring."""

    def test_detail_present_when_set_and_supported(self) -> None:
        """image_size set on a supporting OpenAI model adds a detail key."""
        mgr = _make_manager("gpt-5.4-mini", {"image_size": "high"})
        image_url = _image_block(mgr)
        assert image_url["detail"] == "high"
        assert image_url["url"].startswith("data:image/jpeg;base64,")

    def test_original_on_supported_model(self) -> None:
        """'original' is forwarded verbatim on a GPT-5.6 model."""
        mgr = _make_manager("gpt-5.6-sol", {"image_size": "original"})
        assert _image_block(mgr)["detail"] == "original"

    def test_no_detail_when_unset(self) -> None:
        """No detail key is sent when image_size is unset (byte-identical)."""
        mgr = _make_manager("gpt-5.4-mini", {})
        assert "detail" not in _image_block(mgr)

    def test_original_fallback_to_high_with_warning(self) -> None:
        """'original' on a non-5.6 model falls back to 'high' and warns."""
        mgr = _make_manager("gpt-5.4-mini", {"image_size": "original"})
        with patch("llm.transcription.logger.warning") as mock_warn:
            image_url = _image_block(mgr)
        assert image_url["detail"] == "high"
        assert mock_warn.called

    def test_invalid_value_ignored(self) -> None:
        """An unrecognized image_size sends no detail key."""
        mgr = _make_manager("gpt-5.4-mini", {"image_size": "gigantic"})
        assert "detail" not in _image_block(mgr)

    def test_non_openai_untouched(self) -> None:
        """OpenRouter shape gets no detail even with image_size set."""
        mgr = _make_manager(
            "openai/gpt-5.4-mini", {"image_size": "high"}, provider="openrouter"
        )
        assert "detail" not in _image_block(mgr)


class TestPayloadOriginalResizeSkip:
    """Local preprocessing skips resize when 'original' detail is active."""

    @staticmethod
    def _loader(model_name: str, image_size: str | None) -> MagicMock:
        """Config loader mock with an OpenAI image section and model config."""
        model_cfg: dict[str, Any] = {"transcription_model": {"name": model_name}}
        if image_size is not None:
            model_cfg["transcription_model"]["image_size"] = image_size
        loader = MagicMock()
        loader.get_image_processing_config.return_value = {
            "api_image_processing": {
                "grayscale_conversion": True,
                "handle_transparency": True,
                "llm_detail": "high",
                "resize_profile": "high",
                "high_target_box": [768, 1536],
                "jpeg_quality": 90,
            }
        }
        loader.get_model_config.return_value = model_cfg
        return loader

    @pytest.fixture
    def image_folder(self, tmp_path: Path) -> Path:
        folder = tmp_path / "scans"
        folder.mkdir()
        Image.new("RGB", (1000, 1500), color=(120, 120, 120)).save(
            folder / "page_0001.jpg"
        )
        return folder

    def _patched(self, loader: MagicMock) -> Any:
        return patch("imaging.payload.get_config_loader", return_value=loader)

    def test_helper_true_for_original_on_5_6(self) -> None:
        """_openai_detail_is_original: True for 'original' on GPT-5.6."""
        loader = self._loader("gpt-5.6-sol", "original")
        with self._patched(loader):
            assert _openai_detail_is_original("gpt-5.6-sol") is True

    def test_helper_false_for_original_on_non_5_6(self) -> None:
        """Non-5.6 model does not skip resize even with 'original'."""
        loader = self._loader("gpt-5.4-mini", "original")
        with self._patched(loader):
            assert _openai_detail_is_original("gpt-5.4-mini") is False

    def test_helper_false_when_unset(self) -> None:
        """Unset image_size never skips resize."""
        loader = self._loader("gpt-5.6-sol", None)
        with self._patched(loader):
            assert _openai_detail_is_original("gpt-5.6-sol") is False

    def test_resize_skipped_keeps_native_size(self, image_folder: Path) -> None:
        """'original' on GPT-5.6 keeps native dimensions (under the caps)."""
        loader = self._loader("gpt-5.6-sol", "original")
        with self._patched(loader):
            source = FolderPayloadSource(
                image_folder, provider="openai", model_name="gpt-5.6-sol"
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
        loader = self._loader("gpt-5.6-sol", "original")
        with self._patched(loader):
            source = FolderPayloadSource(
                folder, provider="openai", model_name="gpt-5.6-sol"
            )
            payload = source.build_payload(0)
        width = payload.provenance["width"]
        height = payload.provenance["height"]
        # Longest side capped at 6000 and total pixels within 10,240,000.
        assert max(width, height) <= 6000
        assert width * height <= 10_240_000
        # Aspect ratio preserved (7000:9000 == 7:9), within rounding.
        assert abs((width / height) - (7000 / 9000)) < 0.01

    def test_resize_applied_when_not_original(self, image_folder: Path) -> None:
        """Without 'original', the usual box-fit resize still applies."""
        loader = self._loader("gpt-5.6-sol", "high")
        with self._patched(loader):
            source = FolderPayloadSource(
                image_folder, provider="openai", model_name="gpt-5.6-sol"
            )
            assert source.img_cfg.get("resize_profile") == "high"
            payload = source.build_payload(0)
        # Box-fit pads to the configured 768x1536 target.
        assert payload.provenance["width"] == 768
        assert payload.provenance["height"] == 1536
