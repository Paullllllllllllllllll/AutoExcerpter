"""Pytest fixtures and configuration for AutoExcerpter tests.

This module provides shared fixtures, mock configurations, and test utilities
used across the test suite.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ============================================================================
# Path Fixtures
# ============================================================================
@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory, creating it if necessary."""
    test_data = project_root / "tests" / "test_data"
    test_data.mkdir(parents=True, exist_ok=True)
    return test_data


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Image Fixtures
# ============================================================================
@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    return Image.new("RGB", (1000, 1500), color=(128, 128, 128))


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Create a sample RGBA image with transparency for testing."""
    img = Image.new("RGBA", (800, 600), color=(128, 128, 128, 128))
    return img


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Create a sample grayscale image for testing."""
    return Image.new("L", (500, 500), color=128)


@pytest.fixture
def sample_image_file(temp_dir: Path, sample_rgb_image: Image.Image) -> Path:
    """Create a sample image file for testing."""
    image_path = temp_dir / "test_image.jpg"
    sample_rgb_image.save(image_path, "JPEG", quality=95)
    return image_path


@pytest.fixture
def sample_png_with_transparency(temp_dir: Path, sample_rgba_image: Image.Image) -> Path:
    """Create a PNG file with transparency for testing."""
    image_path = temp_dir / "test_transparent.png"
    sample_rgba_image.save(image_path, "PNG")
    return image_path


# ============================================================================
# Configuration Fixtures
# ============================================================================
@pytest.fixture
def mock_image_processing_config() -> Dict[str, Any]:
    """Return a mock image processing configuration."""
    return {
        "api_image_processing": {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "llm_detail": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 1536],
        },
        "google_image_processing": {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "media_resolution": "high",
            "low_max_side_px": 512,
            "high_target_box": [768, 768],
        },
        "anthropic_image_processing": {
            "target_dpi": 300,
            "jpeg_quality": 95,
            "grayscale_conversion": True,
            "handle_transparency": True,
            "resize_profile": "auto",
            "low_max_side_px": 512,
            "high_max_side_px": 1568,
        },
        "text_cleaning": {
            "enabled": True,
            "unicode_normalization": True,
            "latex_fixing": {
                "enabled": True,
                "balance_dollar_signs": True,
                "close_unclosed_braces": True,
                "fix_common_commands": True,
            },
            "merge_hyphenation": False,
            "whitespace_normalization": {
                "enabled": True,
                "collapse_internal_spaces": True,
                "max_blank_lines": 2,
                "tab_size": 4,
            },
            "line_wrapping": {
                "enabled": False,
                "auto_width": False,
                "fixed_width": 80,
            },
        },
    }


@pytest.fixture
def mock_concurrency_config() -> Dict[str, Any]:
    """Return a mock concurrency configuration."""
    return {
        "image_processing": {
            "concurrency_limit": 24,
        },
        "api_requests": {
            "transcription": {
                "concurrency_limit": 150,
                "delay_between_tasks": 0.05,
                "service_tier": "flex",
            },
            "summary": {
                "concurrency_limit": 150,
                "delay_between_tasks": 0.05,
                "service_tier": "flex",
            },
        },
        "rate_limits": [[120, 1], [15000, 60]],
        "api_timeout": 320,
    }


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Return a mock model configuration."""
    return {
        "transcription_model": {
            "name": "gpt-5-mini",
            "provider": "openai",
        },
        "summary_model": {
            "name": "gpt-5-mini",
            "provider": "openai",
        },
    }


@pytest.fixture
def mock_config_loader(
    mock_image_processing_config: Dict[str, Any],
    mock_concurrency_config: Dict[str, Any],
    mock_model_config: Dict[str, Any],
) -> MagicMock:
    """Create a mock ConfigLoader with predefined configurations."""
    mock_loader = MagicMock()
    mock_loader.get_image_processing_config.return_value = mock_image_processing_config
    mock_loader.get_concurrency_config.return_value = mock_concurrency_config
    mock_loader.get_model_config.return_value = mock_model_config
    mock_loader.is_loaded.return_value = True
    return mock_loader


# ============================================================================
# API Response Fixtures
# ============================================================================
@pytest.fixture
def mock_transcription_response() -> Dict[str, Any]:
    """Return a mock transcription API response."""
    return {
        "page": 1,
        "transcription": "This is a sample transcription text.",
        "no_transcribable_text": False,
        "transcription_not_possible": False,
    }


@pytest.fixture
def mock_summary_response() -> Dict[str, Any]:
    """Return a mock summary API response."""
    return {
        "page_number": {
            "page_number_integer": 1,
            "page_number_type": "arabic",
        },
        "bullet_points": [
            "First key point from the page",
            "Second key point with important information",
            "Third point summarizing conclusions",
        ],
        "references": [
            "Smith, J. (2020). Sample Reference. Journal of Testing, 10(2), 123-145.",
        ],
        "contains_no_semantic_content": False,
    }


# ============================================================================
# Citation Fixtures
# ============================================================================
@pytest.fixture
def sample_citations() -> list[str]:
    """Return sample citation strings for testing."""
    return [
        "Smith, J. (2020). Introduction to Testing. New York: Test Press.",
        "Johnson, A., & Williams, B. (2019). Advanced Python Testing. London: Code Publishers.",
        "Brown, C. (2021). Modern Software Development. Cambridge University Press.",
        "Smith, J. (2020). Introduction to Testing. New York: Test Press.",  # Duplicate
    ]


@pytest.fixture
def mock_openalex_response() -> Dict[str, Any]:
    """Return a mock OpenAlex API response."""
    return {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/test.2020.001",
        "title": "Introduction to Testing",
        "publication_year": 2020,
        "authorships": [
            {"author": {"display_name": "John Smith"}},
        ],
        "primary_location": {
            "source": {"display_name": "Test Press"},
        },
    }


# ============================================================================
# Environment Fixtures
# ============================================================================
@pytest.fixture
def mock_api_keys():
    """Set up mock API keys for testing."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key",
        "OPENROUTER_API_KEY": "test-openrouter-key",
    }):
        yield


# ============================================================================
# Text Fixtures
# ============================================================================
@pytest.fixture
def sample_text_with_latex() -> str:
    """Return sample text containing LaTeX formulas."""
    return r"""
This is a sample text with inline math $E = mc^2$ and more text.

Here's a display equation:
$$\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$

And another inline formula: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$.
"""


@pytest.fixture
def sample_text_with_unicode_issues() -> str:
    """Return sample text with Unicode normalization issues."""
    return (
        "Café\u0301 "  # Combining acute accent (should normalize)
        "test\u00AD"   # Soft hyphen (should be removed)
        "\uFEFFstart"  # BOM (should be removed)
        " \u200Bword"  # Zero-width space (should be removed)
    )


@pytest.fixture
def sample_hyphenated_text() -> str:
    """Return sample text with hyphenated line breaks."""
    return """The quick brown fox jumped over the lazy dog. This is a demon-
stration of hyphen-
ation at line breaks that should be merged back together."""


# ============================================================================
# Summary Result Fixtures
# ============================================================================
@pytest.fixture
def sample_summary_results() -> list[Dict[str, Any]]:
    """Return sample summary results for DOCX generation testing."""
    return [
        {
            "original_input_order_index": 0,
            "model_page_number": 1,
            "page": 1,
            "image_filename": "page_0001.jpg",
            "summary": {
                "page_number": {
                    "page_number_integer": 1,
                    "page_number_type": "arabic",
                },
                "bullet_points": [
                    "Introduction to the document topic",
                    "Key definitions and terminology",
                ],
                "references": [
                    "Smith (2020). Test Reference.",
                ],
                "contains_no_semantic_content": False,
            },
        },
        {
            "original_input_order_index": 1,
            "model_page_number": 2,
            "page": 2,
            "image_filename": "page_0002.jpg",
            "summary": {
                "page_number": {
                    "page_number_integer": 2,
                    "page_number_type": "arabic",
                },
                "bullet_points": [
                    "Methodology section begins",
                    "Data collection procedures described",
                ],
                "references": [],
                "contains_no_semantic_content": False,
            },
        },
    ]
