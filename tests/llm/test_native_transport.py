"""PNG MIME types survive request serialization and in-run retries."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from autoexcerpter.llm.transcription import TranscriptionManager
from tests.imaging.test_native_images import load_image_payload, settings
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import TRANSCRIPTION_ANSWER


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_png_payload_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps, provider: str
) -> None:
    path = tmp_path / "image.png"
    Image.new("L", (32, 48), 128).save(path)
    model = "claude-opus-5" if provider == "anthropic" else "gpt-6-astra"
    payload = load_image_payload(
        path,
        0,
        img_cfg=settings(model, provider, payload_format="png"),
        model_type=provider,
    )
    script = helpers.install(monkeypatch, ["", json.dumps(TRANSCRIPTION_ANSWER)])
    manager = TranscriptionManager(
        helpers.phase(model, provider, options={"image_size": "original"}),
        helpers.env(),
    )

    result = asyncio.run(manager.transcribe_payload(payload))

    assert "error" not in result
    sent = [
        json.dumps([message.model_dump() for message in request.messages])
        for request in script.requests
    ]
    assert len(sent) == 2 and sent[0] == sent[1]
    assert "image/png" in sent[0] and payload.base64 in sent[0]
    raw = base64.b64decode(payload.base64)
    assert raw.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(raw)) as image:
        assert image.mode == "L"
