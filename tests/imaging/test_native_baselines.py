"""Payload hashes of numeric-DPI rendering per provider and render strategy."""

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image

from autoexcerpter.imaging.payload import FolderPayloadSource, PdfPayloadSource
from tests.conftest import image_config


def numeric_config(provider: str, strategy: str) -> dict[str, Any]:
    return {
        "resampling_algorithm": "bilinear",
        "render_strategy": strategy,
        {
            "openai": "api_image_processing",
            "anthropic": "anthropic_image_processing",
            "google": "google_image_processing",
            "custom": "custom_image_processing",
        }[provider]: {
            "target_dpi": 150 if provider == "custom" else 300,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "jpeg_quality": 85 if provider == "custom" else 100,
            "llm_detail": "original" if provider == "openai" else "high",
            "media_resolution": "high",
            "resize_profile": (
                "low"
                if provider == "custom"
                else "auto"
                if provider == "anthropic"
                else "high"
            ),
            "low_max_side_px": 768 if provider == "custom" else 512,
            "high_target_box": [512, 1024] if provider == "custom" else [768, 1536],
            "original_max_side_px": 6000,
            "original_max_pixels": 10240000,
            "high_max_side_px": 2576,
        },
    }


def fixture_files(folder: Path) -> tuple[Path, Path]:
    image = Image.new("RGB", (240, 360), "white")
    image.putdata(
        [
            ((x * 7) % 256, (y * 3) % 256, (x + y) % 256)
            for y in range(360)
            for x in range(240)
        ]
    )
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    images = folder / "images"
    images.mkdir()
    image_path = images / "fixture.png"
    image_path.write_bytes(raw.getvalue())
    pdf_path = folder / "fixture.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=144, height=216)
        page.insert_image(page.rect, stream=raw.getvalue())
        page.insert_text((10, 20), "Synthetic fixture")
        doc.save(pdf_path)
    return pdf_path, images


def payload_hashes(folder: Path, provider: str, strategy: str) -> tuple[str, str]:
    pdf, images = fixture_files(folder)
    model = "claude-opus-5" if provider == "anthropic" else "gpt-6-astra"
    image_size = "original" if provider == "openai" else None
    with image_config(numeric_config(provider, strategy), image_size):
        with PdfPayloadSource(pdf, provider, model) as source:
            pdf_payload = source.build_payload(0)
        with FolderPayloadSource(images, provider, model) as source:
            image_payload = source.build_payload(0)
    return tuple(  # type: ignore[return-value]
        hashlib.sha256(base64.b64decode(payload.base64)).hexdigest()
        for payload in (pdf_payload, image_payload)
    )


_COMMON = (
    "8d4794012ec2ae98a4e33e5fa236335be4a8bc25b45a7713a39936f1a56a0c96",
    "e9982b2e71d89dd1aea443bbafa3256cb45b2e3230ca556f0f5643b15b1cea29",
)
_GOOGLE = (
    "a74271094c8b708344e77a5a1ce569ea2029d305d34719065a141ffa8072db83",
    "88d13c850c13e84a5c2d95e2b21bbfe807904d70f56e5d68598d8bd552d83da7",
)
_CUSTOM = (
    "b5e6d67f8a0e3493539e04082d6ad3c057f766bf6b238f0a7361706850eed5af",
    "7f935de8bd86cc48d352424aa805eee4308eeebeb89dc7a08bdb39f29b45d301",
)
BASELINES = {
    (provider, strategy): hashes
    for provider, hashes in (
        ("openai", _COMMON),
        ("anthropic", _COMMON),
        ("google", _GOOGLE),
        ("custom", _CUSTOM),
    )
    for strategy in ("direct", "supersample")
}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "custom"])
@pytest.mark.parametrize("strategy", ["direct", "supersample"])
def test_numeric_payload_baseline(tmp_path: Path, provider: str, strategy: str) -> None:
    assert payload_hashes(tmp_path, provider, strategy) == BASELINES[provider, strategy]
