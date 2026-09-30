"""PNG MIME types survive request serialization and schema retries."""

import base64
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from tests.test_native_images import load_image_payload, settings
from tests.test_transcribe_api_extended import _make_manager


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_png_sync_and_retry(tmp_path: Path, provider: str) -> None:
    path = tmp_path / "image.png"
    Image.new("L", (32, 48), 128).save(path)
    model = "claude-opus-5" if provider == "anthropic" else "gpt-6-astra"
    payload = load_image_payload(
        path,
        0,
        img_cfg=settings(model, provider, payload_format="png"),
        model_type=provider,
    )
    manager = _make_manager(
        provider=provider, model_name=model, model_config={"image_size": "original"}
    )
    captured: list[str] = []

    def invoke(_model: Any, messages: Any, _kwargs: Any, _label: str) -> AIMessage:
        captured.append(json.dumps([message.model_dump() for message in messages]))
        return AIMessage(
            content=""
            if len(captured) == 1
            else json.dumps(
                {
                    "transcription": "Synthetic transcription.",
                    "image_analysis": "Clear text.",
                    "no_transcribable_text": False,
                    "transcription_not_possible": False,
                }
            )
        )

    with (
        patch.object(manager, "_build_invoke_kwargs", return_value={}),
        patch.object(manager, "_apply_structured_output_kwargs"),
        patch.object(manager, "_get_structured_chat_model"),
        patch.object(manager, "_invoke_with_retry", side_effect=invoke),
        patch.object(manager, "_report_token_usage"),
        patch.object(manager, "_report_success"),
        patch("llm.transcription.time.sleep"),
    ):
        result = manager.transcribe_payload(payload)
        jpeg_messages, _ = manager._build_model_inputs("fixture")
    assert "error" not in result
    assert len(captured) == 2 and captured[0] == captured[1]
    assert "image/png" in captured[0] and payload.base64 in captured[0]
    raw = base64.b64decode(payload.base64)
    assert raw.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(raw)) as image:
        assert image.mode == "L"
    assert "image/jpeg" in json.dumps([m.model_dump() for m in jpeg_messages])
