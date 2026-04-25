"""Type definitions for pipeline inputs and processing statistics.

- ``ItemSpec`` — descriptor for a PDF or image folder produced by
  :mod:`pipeline.scanner` and consumed by :class:`pipeline.transcriber.ItemTranscriber`.
- ``ProcessingStats`` — aggregated counters over a batch of item runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ItemSpec:
    """Descriptor for a PDF file or image folder to process."""

    kind: str  # "pdf" or "image_folder"
    path: Path
    image_count: int | None = None

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
        return (
            f"{item_type_label}: {self.path.name}{count_str} (from: {self.path.parent})"
        )


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""

    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    total_time: float = 0.0
    average_time_per_item: float = 0.0
    processing_times: list[float] = field(default_factory=list)

    def add_success(self, processing_time: float) -> None:
        """Record a successful processing operation."""
        self.successful_items += 1
        self.processing_times.append(processing_time)
        if self.processing_times:
            self.average_time_per_item = sum(self.processing_times) / len(
                self.processing_times
            )

    def add_failure(self) -> None:
        """Record a failed processing operation."""
        self.failed_items += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100.0


__all__ = ["ItemSpec", "ProcessingStats"]
