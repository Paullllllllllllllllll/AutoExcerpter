"""Tests for modules/types.py - Type definitions and data structures."""

from __future__ import annotations

from pathlib import Path

import pytest

from modules.types import (
    TranscriptionResult,
    PageInformation,
    SummaryContent,
    SummaryResult,
    ConcurrencyConfig,
    ItemSpec,
)


class TestTranscriptionResult:
    """Tests for TranscriptionResult TypedDict."""

    def test_basic_structure(self):
        """TranscriptionResult can hold basic fields."""
        result: TranscriptionResult = {
            "page": 1,
            "image": "page_0001.jpg",
            "transcription": "Sample text",
            "processing_time": 1.5,
        }

        assert result["page"] == 1
        assert result["transcription"] == "Sample text"

    def test_optional_fields(self):
        """TranscriptionResult allows optional fields."""
        result: TranscriptionResult = {
            "page": 1,
            "image": "test.jpg",
            "transcription": "Text",
        }

        # Optional fields should not be required
        assert "error" not in result

    def test_error_field(self):
        """TranscriptionResult can hold error information."""
        result: TranscriptionResult = {
            "page": 1,
            "image": "test.jpg",
            "transcription": "",
            "error": "API timeout",
        }

        assert result["error"] == "API timeout"


class TestPageInformation:
    """Tests for PageInformation TypedDict."""

    def test_basic_structure(self):
        """PageInformation holds page metadata including page_types."""
        info: PageInformation = {
            "page_number_integer": 5,
            "page_number_type": "arabic",
            "page_types": ["content"],
        }

        assert info["page_number_integer"] == 5
        assert info["page_number_type"] == "arabic"
        assert info["page_types"] == ["content"]

    def test_unnumbered_page(self):
        """PageInformation can represent unnumbered pages."""
        info: PageInformation = {
            "page_number_integer": None,
            "page_number_type": "none",
            "page_types": ["blank"],
        }

        assert info["page_number_type"] == "none"
        assert info["page_types"] == ["blank"]


class TestSummaryContent:
    """Tests for SummaryContent TypedDict."""

    def test_full_structure(self):
        """SummaryContent holds all summary fields."""
        content: SummaryContent = {
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Point 1", "Point 2"],
            "references": ["Ref 1"],
        }

        assert len(content["bullet_points"]) == 2
        assert len(content["references"]) == 1

    def test_empty_content(self):
        """SummaryContent can represent blank pages."""
        content: SummaryContent = {
            "page_information": {
                "page_number_integer": None,
                "page_number_type": "none",
                "page_types": ["blank"],
            },
            "bullet_points": None,
            "references": None,
        }

        assert content["page_information"]["page_types"] == ["blank"]


class TestSummaryResult:
    """Tests for SummaryResult TypedDict."""

    def test_full_structure(self):
        """SummaryResult holds complete summary result (flat structure)."""
        result: SummaryResult = {
            "page": 1,
            "page_information": {
                "page_number_integer": 1,
                "page_number_type": "arabic",
                "page_types": ["content"],
            },
            "bullet_points": ["Point"],
            "references": [],
            "image_filename": "page_0001.jpg",
            "original_input_order_index": 0,
            "processing_time": 2.5,
            "provider": "openai",
            "api_response": {},
            "schema_retries": {},
            "error": None,
        }

        assert result["page"] == 1
        assert result["page_information"]["page_number_integer"] == 1


class TestConcurrencyConfig:
    """Tests for ConcurrencyConfig dataclass."""

    def test_default_values(self):
        """ConcurrencyConfig has sensible defaults."""
        config = ConcurrencyConfig()

        assert config.image_processing_limit == 24
        assert config.transcription_limit == 150
        assert config.summary_limit == 150
        assert config.transcription_delay == 0.05
        assert config.summary_delay == 0.05
        assert config.transcription_service_tier == "flex"
        assert config.summary_service_tier == "flex"

    def test_custom_values(self):
        """ConcurrencyConfig accepts custom values."""
        config = ConcurrencyConfig(
            image_processing_limit=16,
            transcription_limit=100,
            summary_limit=50,
            transcription_delay=0.1,
            summary_delay=0.2,
            transcription_service_tier="default",
            summary_service_tier="default",
        )

        assert config.image_processing_limit == 16
        assert config.transcription_limit == 100

    def test_from_dict(self):
        """ConcurrencyConfig can be created from dictionary."""
        config_dict = {
            "image_processing": {
                "concurrency_limit": 32,
            },
            "api_requests": {
                "transcription": {
                    "concurrency_limit": 200,
                    "delay_between_tasks": 0.1,
                    "service_tier": "default",
                },
                "summary": {
                    "concurrency_limit": 100,
                    "delay_between_tasks": 0.05,
                    "service_tier": "flex",
                },
            },
        }

        config = ConcurrencyConfig.from_dict(config_dict)

        assert config.image_processing_limit == 32
        assert config.transcription_limit == 200
        assert config.transcription_delay == 0.1
        assert config.transcription_service_tier == "default"

    def test_from_dict_with_missing_keys(self):
        """ConcurrencyConfig.from_dict handles missing keys."""
        config = ConcurrencyConfig.from_dict({})

        # Should use defaults
        assert config.image_processing_limit == 24
        assert config.transcription_limit == 150

    def test_frozen(self):
        """ConcurrencyConfig is immutable."""
        config = ConcurrencyConfig()

        with pytest.raises(Exception):
            config.image_processing_limit = 100  # type: ignore


class TestItemSpec:
    """Tests for ItemSpec dataclass."""

    def test_pdf_item(self, temp_dir: Path):
        """ItemSpec represents PDF items correctly."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.touch()

        item = ItemSpec(kind="pdf", path=pdf_path)

        assert item.path == pdf_path
        assert item.kind == "pdf"

    def test_image_folder_item(self, temp_dir: Path):
        """ItemSpec represents image folder items correctly."""
        folder_path = temp_dir / "images"
        folder_path.mkdir()

        item = ItemSpec(kind="image_folder", path=folder_path, image_count=10)

        assert item.path == folder_path
        assert item.kind == "image_folder"
        assert item.image_count == 10

    def test_display_label(self, temp_dir: Path):
        """ItemSpec.display_label returns readable string."""
        pdf_path = temp_dir / "my_document.pdf"
        pdf_path.touch()

        item = ItemSpec(kind="pdf", path=pdf_path)
        label = item.display_label()

        assert "my_document" in label or str(pdf_path) in label
        assert "PDF" in label

    def test_output_stem_property(self, temp_dir: Path):
        """ItemSpec provides output_stem property."""
        pdf_path = temp_dir / "test_file.pdf"
        pdf_path.touch()

        item = ItemSpec(kind="pdf", path=pdf_path)

        assert item.output_stem == "test_file"

    def test_display_label_with_image_count(self, temp_dir: Path):
        """Display label includes image count for image folders."""
        folder_path = temp_dir / "images"
        folder_path.mkdir()

        item = ItemSpec(kind="image_folder", path=folder_path, image_count=25)
        label = item.display_label()

        assert "25 images" in label
        assert "Image Folder" in label
