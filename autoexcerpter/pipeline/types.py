"""The descriptor of one input item, as found by the input scanner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ItemSpec:
    """Descriptor for a PDF file or image folder to process."""

    kind: str  # "pdf" or "image_folder"
    path: Path
    image_count: int | None = None

    @property
    def output_stem(self) -> str:
        """Get the output filename stem.

        For a PDF the extension is dropped; for an image folder the full
        directory name is kept ("photos.2023" must not collapse onto a
        sibling "photos.2024" via ``Path.stem``).
        """
        return self.path.stem if self.kind == "pdf" else self.path.name


__all__ = ["ItemSpec"]
