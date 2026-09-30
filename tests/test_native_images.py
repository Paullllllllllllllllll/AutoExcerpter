"""Synthetic geometry, payload, sizing, settings and resume contracts."""

import io
import json
import math
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import fitz
import pytest
from PIL import Image

from config.constants import LOG_FORMAT_VERSION
from imaging.native import (
    encode_payload,
    guarded_payload,
    image_settings_fingerprint,
    model_image_cap,
    native_page_dpi,
    resized_size,
    resolve_target_size,
    validate_image_settings,
)
from imaging.payload import FolderPayloadSource, PagePayload, PdfPayloadSource
from imaging.settings import resolve_image_settings
from pipeline.paths import create_safe_directory_name, create_safe_log_filename
from pipeline.resume import ProcessingState, ResumeChecker, verify_image_settings


def loader_for(
    cfg: dict[str, Any],
    provider: str,
    model: str,
    max_pixels: int = 24000000,
    strategy: str = "direct",
) -> MagicMock:
    loader = MagicMock()
    section = "api" if provider in ("openai", "openrouter") else provider
    loader.get_image_processing_config.return_value = {
        "render_strategy": strategy,
        "max_pixels_per_page": max_pixels,
        f"{section}_image_processing": cfg,
    }
    loader.get_model_config.return_value = {
        "transcription_model": {
            "name": model,
            "provider": provider,
            "image_size": cfg.get("request_detail", cfg.get("llm_detail", "original")),
        },
    }
    return loader


def settings(
    model: str = "gpt-6-astra", provider: str = "openai", **overrides: Any
) -> dict[str, Any]:
    loader = loader_for(
        {"target_dpi": "native", "llm_detail": "original", **overrides}, provider, model
    )
    return resolve_image_settings(loader, provider, model)[0]


def render_single_pdf_page_payload(
    pdf: Path,
    index: int,
    *,
    target_dpi: int | str,
    img_cfg: dict[str, Any],
    model_type: str,
    max_pixels: int = 24000000,
    render_strategy: str = "direct",
) -> PagePayload:
    loader = loader_for(
        {**img_cfg, "target_dpi": target_dpi, "render_strategy": render_strategy},
        model_type,
        img_cfg["model_name"],
        max_pixels,
        render_strategy,
    )
    with (
        patch("imaging.payload.get_config_loader", return_value=loader),
        PdfPayloadSource(pdf, model_type, img_cfg["model_name"]) as source,
    ):
        return source.build_payload(index)


def load_image_payload(
    path: Path, index: int, *, img_cfg: dict[str, Any], model_type: str
) -> PagePayload:
    loader = loader_for(img_cfg, model_type, img_cfg["model_name"])
    with (
        patch("imaging.payload.get_config_loader", return_value=loader),
        FolderPayloadSource(path.parent, model_type, img_cfg["model_name"]) as source,
    ):
        return source.build_payload(index)


def scan_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    image.paste((40, 80, 120), (width // 4, height // 4, width // 2, height // 2))
    return encode_payload(image, "png", 95)[0]


def scan_page(doc: Any, dpi: int = 300, width: int = 144, height: int = 216) -> Any:
    page = doc.new_page(width=width, height=height)
    page.insert_image(
        page.rect,
        stream=scan_bytes(
            round(width * dpi / 72),
            round(height * dpi / 72),
        ),
    )
    return page


@pytest.mark.parametrize("dpi", [150, 360, 600])
@pytest.mark.parametrize("strategy", ["direct", "supersample"])
def test_native_full_page(tmp_path: Path, dpi: int, strategy: str) -> None:
    pdf = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        page = scan_page(doc, dpi)
        density = native_page_dpi(page, 300)
        assert density.dpi_x == pytest.approx(dpi, abs=0.5)
        assert density.dpi_y == pytest.approx(dpi, abs=0.5)
        doc.save(pdf)
    payload = render_single_pdf_page_payload(
        pdf,
        0,
        target_dpi="native",
        img_cfg=settings(),
        model_type="openai",
        max_pixels=100,
        render_strategy=strategy,
    )
    assert abs(payload.provenance["width"] - dpi * 2) <= 1
    assert abs(payload.provenance["height"] - dpi * 3) <= 1
    assert payload.provenance["render_dpi"] == pytest.approx(dpi)
    assert payload.provenance["downscale_reason"] == "none"


def test_mrc_high_layer_and_small_stamp() -> None:
    with fitz.open() as doc:
        page = scan_page(doc, 150)
        page.insert_image(fitz.Rect(1, 1, 11, 11), stream=scan_bytes(600, 600))
        assert native_page_dpi(page, 300).dpi == pytest.approx(150)
        layer = Image.new("RGBA", (720, 1080), (0, 0, 0, 128))
        data = io.BytesIO()
        layer.save(data, format="PNG")
        page.insert_image(page.rect, stream=data.getvalue())
        page = doc.reload_page(page)
        assert native_page_dpi(page, 300).dpi == pytest.approx(360)


@pytest.mark.parametrize("geometry", ["rotate", "crop", "anisotropic", "diagonal"])
def test_native_geometry(geometry: str) -> None:
    with fitz.open() as doc:
        if geometry == "diagonal":
            with fitz.open() as original:
                scan_page(original, 300, 144, 144)
                page = doc.new_page(width=144, height=144)
                page.show_pdf_page(page.rect, original, 0, rotate=45)
                # Rotating into the same square shrinks the placement by sqrt(2).
                expected = 300 * math.sqrt(2)
        else:
            page = scan_page(doc, 300)
            expected = 300
            if geometry == "rotate":
                page.set_rotation(90)
            elif geometry == "crop":
                page.set_cropbox(fitz.Rect(72, 0, 144, 216))
            else:
                page = doc.new_page(width=144, height=216)
                page.insert_image(
                    page.rect, stream=scan_bytes(600, 450), keep_proportion=False
                )
        density = native_page_dpi(page, 72)
        assert density.dpi_x == pytest.approx(expected, abs=0.5)
        assert density.dpi_y == pytest.approx(
            150 if geometry == "anisotropic" else expected,
            abs=0.5,
        )


@pytest.mark.parametrize("kind", ["text", "pixel", "figure"])
def test_native_fallback(kind: str) -> None:
    with fitz.open() as doc:
        page = doc.new_page(width=144, height=216)
        if kind == "text":
            page.insert_text((10, 20), "Born digital")
        elif kind == "pixel":
            page.insert_image(page.rect, stream=scan_bytes(1, 1))
        else:
            page.insert_image(fitz.Rect(0, 0, 60, 60), stream=scan_bytes(600, 600))
        result = native_page_dpi(page, 240)
        assert result.source == "fallback"
        assert result.dpi == 240


def test_anthropic_published_resize() -> None:
    assert resized_size(1075, 1520, 1568, 1568) == (924, 1307)


@pytest.mark.parametrize(
    "model,provider,side,patches,patch_px",
    [
        ("claude-opus-5", "anthropic", 2576, 4784, 28),
        ("claude-haiku-4-5", "anthropic", 1568, 1568, 28),
        ("gpt-5.4", "openai", 6000, 10000, 32),
        ("gpt-6-astra", "openai", 65535, 30000, 32),
    ],
)
def test_registry_caps(
    model: str, provider: str, side: int, patches: int, patch_px: int
) -> None:
    cfg = settings(model, provider)
    cap = model_image_cap(provider, model, cfg["resolved_detail"], cfg)
    assert cap is not None
    assert (cap.max_side, cap.patches, cap.patch_px) == (side, patches, patch_px)
    for width, height in [(5000, 8000), (10000, 100), (3199, 3201), (3200, 3200)]:
        (w, h), reason = resolve_target_size(
            width,
            height,
            provider,
            model,
            cfg["resolved_detail"],
            cfg,
        )
        assert max(w, h) <= side
        assert math.ceil(w / patch_px) * math.ceil(h / patch_px) <= patches
        assert reason in ("none", "model_cap")


def test_unknown_original_is_conservative() -> None:
    size, reason = resolve_target_size(5000, 8000, "openai", "unknown", "original", {})
    assert math.ceil(size[0] / 32) * math.ceil(size[1] / 32) <= 10000
    assert reason == "model_cap"


def test_anthropic_low_profile_keeps_low_cap() -> None:
    cfg = settings("claude-opus-5", "anthropic", resize_profile="low")
    assert model_image_cap("anthropic", "claude-opus-5", "low", cfg) is None
    size, reason = resolve_target_size(
        3000, 4000, "anthropic", "claude-opus-5", "low", cfg
    )
    assert max(size) == 512
    assert reason == "profile"


def test_large_original_and_tighter_config() -> None:
    cfg = settings()
    assert resolve_target_size(
        4000, 7000, "openai", "gpt-6-astra", "original", cfg
    ) == (
        (4000, 7000),
        "none",
    )
    w, h = resolve_target_size(5000, 8000, "openai", "gpt-6-astra", "original", cfg)[0]
    assert math.ceil(w / 32) * math.ceil(h / 32) <= 30000
    strict = {**cfg, "original_max_side_px": 1200, "original_max_pixels": 1000000}
    (w, h), reason = resolve_target_size(
        4000,
        7000,
        "openai",
        "gpt-6-astra",
        "original",
        strict,
    )
    assert max(w, h) <= 1200 and w * h <= 1000000
    assert reason == "model_cap"


@pytest.mark.parametrize("strategy", ["direct", "supersample"])
def test_numeric_model_caps_and_native_memory(tmp_path: Path, strategy: str) -> None:
    pdf = tmp_path / "large.pdf"
    with fitz.open() as doc:
        scan_page(doc, 300, 720, 960)
        doc.save(pdf)
    for model in ("gpt-6-astra", "gpt-5.4"):
        payload = render_single_pdf_page_payload(
            pdf,
            0,
            target_dpi=300,
            img_cfg=settings(model),
            model_type="openai",
            max_pixels=24000000,
            render_strategy=strategy,
        )
        if model == "gpt-6-astra":
            assert (payload.provenance["width"], payload.provenance["height"]) == (
                3000,
                4000,
            )
        else:
            assert (
                math.ceil(payload.provenance["width"] / 32)
                * math.ceil(payload.provenance["height"] / 32)
                <= 10000
            )
            assert payload.provenance["downscale_reason"] == "model_cap"
    numeric = render_single_pdf_page_payload(
        pdf,
        0,
        target_dpi=300,
        img_cfg=settings(),
        model_type="openai",
        max_pixels=1000000,
        render_strategy=strategy,
    )
    assert numeric.provenance["width"] * numeric.provenance["height"] <= 1003000
    assert numeric.provenance["downscale_reason"] == "memory_guard"


def test_png_roundtrip_fallback_and_error(tmp_path: Path) -> None:
    image = Image.frombytes("RGB", (200, 200), random.Random(42).randbytes(120000))
    png, mime = encode_payload(image, "png", 95)
    assert mime == "image/png"
    assert Image.open(io.BytesIO(png)).tobytes() == image.tobytes()
    jpg, _, _ = guarded_payload(image, "jpeg", 70, 0)
    limit = 4 * ((len(jpg) + 2) // 3)
    with pytest.raises(ValueError, match="max_image_bytes"):
        guarded_payload(image, "png", 70, limit - 1)
    data, mime, fallback = guarded_payload(image, "png", 70, limit)
    assert mime == "image/jpeg" and fallback and data == jpg
    path = tmp_path / "source.png"
    path.write_bytes(png)
    cfg = settings(
        payload_format="png",
        grayscale_conversion=False,
        jpeg_quality=70,
        max_image_bytes=limit,
    )
    payload = load_image_payload(path, 0, img_cfg=cfg, model_type="openai")
    assert payload.provenance["format_fallback"]
    assert payload.mime_type == "image/jpeg"
    assert payload.image_name == path.name


def test_downscale_one_log_and_provenance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "source.png"
    Image.new("L", (800, 1200), "white").save(path, dpi=(150, 150))
    cfg = settings(original_max_side_px=600)
    with caplog.at_level("INFO"):
        payload = load_image_payload(path, 0, img_cfg=cfg, model_type="openai")
    lines = [rec.message for rec in caplog.records if "reason model_cap" in rec.message]
    assert len(lines) == 1
    assert "800x1200" in lines[0] and "400x600" in lines[0]
    assert payload.provenance["file_dpi_metadata"] == pytest.approx((150, 150), abs=0.1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("target_dpi", 0),
        ("target_dpi", "300"),
        ("target_dpi", True),
        ("native_fallback_dpi", None),
        ("native_fallback_dpi", 1.5),
        ("payload_format", "gif"),
        ("max_image_bytes", -1),
    ],
)
def test_invalid_settings(key: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"{key}.*api_image_processing"):
        validate_image_settings({key: value}, "api_image_processing")


@pytest.mark.parametrize("provider", ["google", "custom"])
def test_unbounded_native_rejected(provider: str) -> None:
    with pytest.raises(
        ValueError, match=f"resize_profile.*{provider}_image_processing"
    ):
        settings(provider=provider, resize_profile="none")


@pytest.mark.parametrize(
    "model,side,patches",
    [
        ("claude-opus-5", 2576, 4784),
        ("claude-haiku-4-5", 1568, 1568),
    ],
)
def test_native_anthropic_render_cap(
    tmp_path: Path, model: str, side: int, patches: int
) -> None:
    pdf = tmp_path / "scan.pdf"
    with fitz.open() as doc:
        scan_page(doc, 600, 288, 432)
        doc.save(pdf)
    payload = render_single_pdf_page_payload(
        pdf,
        0,
        target_dpi="native",
        img_cfg=settings(model, "anthropic"),
        model_type="anthropic",
        max_pixels=100,
    )
    assert max(payload.provenance["width"], payload.provenance["height"]) <= side
    assert (
        math.ceil(payload.provenance["width"] / 28)
        * math.ceil(payload.provenance["height"] / 28)
        <= patches
    )
    assert payload.provenance["downscale_reason"] == "model_cap"
    assert payload.provenance["source_dpi_x"] == pytest.approx(600)


def test_large_native_ignores_numeric_memory_guard(tmp_path: Path) -> None:
    pdf = tmp_path / "large.pdf"
    with fitz.open() as doc:
        scan_page(doc, 300, 1440, 1200)
        doc.save(pdf)
    native = render_single_pdf_page_payload(
        pdf,
        0,
        target_dpi="native",
        img_cfg=settings(),
        model_type="openai",
        max_pixels=24000000,
    )
    assert (native.provenance["width"], native.provenance["height"]) == (6000, 5000)
    assert native.provenance["downscale_reason"] == "none"
    numeric = render_single_pdf_page_payload(
        pdf,
        0,
        target_dpi=300,
        img_cfg=settings(),
        model_type="openai",
        max_pixels=24000000,
    )
    assert numeric.provenance["width"] * numeric.provenance["height"] == pytest.approx(
        24000000, rel=0.001
    )
    assert numeric.provenance["downscale_reason"] == "memory_guard"


def test_fingerprint_resume_and_legacy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "source.png"
    path.write_bytes(scan_bytes(100, 150))
    cfg = settings()
    loader = loader_for(cfg, "openai", cfg["model_name"])
    with (
        patch("imaging.payload.get_config_loader", return_value=loader),
        FolderPayloadSource(tmp_path, "openai", cfg["model_name"]) as source,
    ):
        provenance = source.file_provenance()
    current = provenance["image_config"]
    assert provenance["detail"] == "original"
    assert provenance["image_settings_fingerprint"] == image_settings_fingerprint(
        current
    )
    assert image_settings_fingerprint(current) == image_settings_fingerprint(
        dict(reversed(list(current.items())))
    )
    header = {
        "_format_version": LOG_FORMAT_VERSION,
        "file_provenance": provenance,
        "model_name": cfg["model_name"],
        "total_images": 2,
    }
    verify_image_settings(header, current)
    with pytest.raises(ValueError, match="payload_format.*--overwrite"):
        verify_image_settings(header, {**current, "payload_format": "png"})
    work = tmp_path / create_safe_directory_name("document", "_working_files")
    work.mkdir()
    log = work / create_safe_log_filename("document", "transcription")
    entry = {"original_input_order_index": 0, "transcription": "Synthetic text."}
    log.write_text(json.dumps(header) + "\n" + json.dumps(entry), encoding="utf-8")
    changed_loader = loader_for(
        {**cfg, "payload_format": "png"}, "openai", cfg["model_name"]
    )
    checker = ResumeChecker("skip", summarize=False)
    with patch("pipeline.resume.get_config_loader", return_value=changed_loader):
        # The checker never classifies a settings mismatch as complete: the
        # item stays partial, and the transcriber's check fails it before its
        # working log is rewritten, so the run exits non-zero.
        result = checker.should_skip("document", tmp_path)
        assert result.state == ProcessingState.PARTIAL
        assert result.completed_page_indices == {0}
        assert ResumeChecker("overwrite").should_skip("document", tmp_path).state == (
            ProcessingState.NONE
        )
    assert any(record.levelname == "ERROR" for record in caplog.records)
    header.pop("file_provenance")
    verify_image_settings(header, {**current, "payload_format": "png"})
    assert "may mix settings" in caplog.text


@pytest.mark.parametrize("provider", ["openrouter", "custom", "google", "anthropic"])
def test_non_openai_never_sizes_for_original(provider: str) -> None:
    model = {
        "openrouter": "openai/gpt-6-astra",
        "custom": "gpt-6-astra",
        "google": "gemini-3-flash",
        "anthropic": "claude-opus-5",
    }[provider]
    cfg = settings(model, provider)
    assert cfg["request_detail"] is None
    if provider == "openrouter":
        # No detail is sent; original sizing keeps the conservative cap.
        assert cfg["resolved_detail"] == "original"
        assert cfg["cap_policy"] == "openai-original-10k-v1"
        size, reason = resolve_target_size(
            4000, 7000, cfg["model_type"], model, cfg["resolved_detail"], cfg
        )
        assert math.ceil(size[0] / 32) * math.ceil(size[1] / 32) <= 10000
        return
    assert cfg["resolved_detail"] != "original"
    if provider == "custom":
        assert cfg["cap_policy"] == "profile-v1"
        size, reason = resolve_target_size(
            4000, 7000, cfg["model_type"], model, cfg["resolved_detail"], cfg
        )
        assert size[0] <= 768 and size[1] <= 1536
        assert reason == "profile"


@pytest.mark.parametrize("provider", ["openrouter", "custom"])
def test_non_openai_keeps_local_low_profile(provider: str) -> None:
    model = "openai/gpt-6-astra" if provider == "openrouter" else "gpt-6-astra"
    cfg = settings(model, provider, llm_detail="low", low_max_side_px=512)
    assert cfg["resolved_detail"] == "low"
    size, reason = resolve_target_size(
        600, 900, cfg["model_type"], model, cfg["resolved_detail"], cfg
    )
    assert max(size) == 512 and reason == "profile"


def test_profile_warning_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imaging.payload import _PayloadSourceBase

    monkeypatch.setattr(_PayloadSourceBase, "_native_profile_warned", False)
    loader = loader_for({"target_dpi": "native"}, "google", "gemini-3-flash")
    with patch("imaging.payload.get_config_loader", return_value=loader):
        for _ in range(2):
            FolderPayloadSource(tmp_path, "google", "gemini-3-flash").close()
    assert sum("discards native resolution" in r.message for r in caplog.records) == 1
