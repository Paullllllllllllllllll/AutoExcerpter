"""Type definitions and data structures for AutoExcerpter.

This module provides type-safe data structures using dataclasses and TypedDicts
to replace loose dictionary payloads throughout the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


# ============================================================================
# Result Type Definitions
# ============================================================================
class TranscriptionResult(TypedDict, total=False):
    """Type definition for transcription API results."""
    page: int
    image: str
    transcription: str
    processing_time: float
    retries: int
    api_retries: int
    schema_retries: Dict[str, int]
    error: Optional[str]
    original_input_order_index: int


class PageNumberInfo(TypedDict):
    """Type definition for page number metadata."""
    page_number_integer: int
    contains_no_page_number: bool


class SummaryContent(TypedDict, total=False):
    """Type definition for summary content structure."""
    page_number: PageNumberInfo
    bullet_points: List[str]
    references: List[str]
    contains_no_semantic_content: bool


class SummaryResult(TypedDict, total=False):
    """Type definition for summary API results."""
    page: int
    summary: SummaryContent
    model_page_number: Optional[int]
    image_filename: str
    original_input_order_index: int
    processing_time: float
    retries: int
    api_retries: int
    schema_retries: Dict[str, int]
    error: Optional[str]


# ============================================================================
# Configuration Data Classes
# ============================================================================
@dataclass(frozen=True)
class ConcurrencyConfig:
    """Configuration for concurrent processing."""
    image_processing_limit: int = 24
    transcription_limit: int = 150
    summary_limit: int = 150
    transcription_delay: float = 0.05
    summary_delay: float = 0.05
    transcription_service_tier: str = "flex"
    summary_service_tier: str = "flex"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> ConcurrencyConfig:
        """Create ConcurrencyConfig from configuration dictionary."""
        img_proc = config.get("image_processing", {})
        api_req = config.get("api_requests", {})
        trans_cfg = api_req.get("transcription", {})
        summ_cfg = api_req.get("summary", {})
        
        return cls(
            image_processing_limit=img_proc.get("concurrency_limit", 24),
            transcription_limit=trans_cfg.get("concurrency_limit", 150),
            summary_limit=summ_cfg.get("concurrency_limit", 150),
            transcription_delay=trans_cfg.get("delay_between_tasks", 0.05),
            summary_delay=summ_cfg.get("delay_between_tasks", 0.05),
            transcription_service_tier=trans_cfg.get("service_tier", "flex"),
            summary_service_tier=summ_cfg.get("service_tier", "flex"),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model parameters."""
    name: str
    max_output_tokens: int
    reasoning_effort: Optional[str] = None
    text_verbosity: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from configuration dictionary."""
        reasoning = config.get("reasoning", {})
        text = config.get("text", {})
        
        return cls(
            name=config.get("name", "gpt-5-mini"),
            max_output_tokens=config.get("max_output_tokens", 16384),
            reasoning_effort=reasoning.get("effort") if isinstance(reasoning, dict) else None,
            text_verbosity=text.get("verbosity") if isinstance(text, dict) else None,
        )


# ============================================================================
# Processing Data Classes
# ============================================================================
@dataclass(frozen=True)
class ItemSpec:
    """Descriptor for a PDF file or image folder to process."""
    kind: str  # "pdf" or "image_folder"
    path: Path
    image_count: Optional[int] = None
    
    @property
    def output_stem(self) -> str:
        """Get the output filename stem."""
        return self.path.stem
    
    def display_label(self) -> str:
        """Generate a human-readable display label."""
        item_type_label = "PDF" if self.kind == "pdf" else "Image Folder"
        count_str = ""
        if self.kind == "image_folder" and self.image_count is not None:
            count_str = f" ({self.image_count} images)"
        return f"{item_type_label}: {self.path.name}{count_str} (from: {self.path.parent})"


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    total_time: float = 0.0
    average_time_per_item: float = 0.0
    processing_times: List[float] = field(default_factory=list)
    
    def add_success(self, processing_time: float) -> None:
        """Record a successful processing operation."""
        self.successful_items += 1
        self.processing_times.append(processing_time)
        if self.processing_times:
            self.average_time_per_item = sum(self.processing_times) / len(self.processing_times)
    
    def add_failure(self) -> None:
        """Record a failed processing operation."""
        self.failed_items += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100.0


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    "TranscriptionResult",
    "PageNumberInfo",
    "SummaryContent",
    "SummaryResult",
    "ConcurrencyConfig",
    "ModelConfig",
    "ItemSpec",
    "ProcessingStats",
]
