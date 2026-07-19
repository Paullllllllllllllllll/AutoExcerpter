"""Transcription text file output for AutoExcerpter."""

from __future__ import annotations

import contextlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.logger import setup_logger

logger = setup_logger(__name__)


def write_transcription_to_text(
    transcription_results: list[dict[str, Any]],
    output_path: Path,
    document_name: str,
    item_type: str,
    total_elapsed_time: float,
    source_path: Path,
    metadata_notes: list[str] | None = None,
) -> bool:
    """Persist transcription output as a text file alongside basic metadata.

    Args:
        metadata_notes: Optional extra provenance lines recorded in the header
            (e.g. a summary-only-resume model mismatch).
    """
    elapsed_str = str(timedelta(seconds=int(total_elapsed_time)))
    successes = sum(1 for result in transcription_results if "error" not in result)
    failures = len(transcription_results) - successes

    # Write to a sibling temp file and atomically replace the target, so a crash
    # mid-write never leaves a truncated .txt that resume trusts as COMPLETE
    # (classification keys on exists() and st_size > 0).
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write(f"# Transcription of: {document_name}\n")
            file_handle.write(f"# Source Path: {source_path}\n")
            file_handle.write(f"# Type: {item_type}\n")
            file_handle.write(
                f"# Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            file_handle.write(
                f"# Total images processed: {len(transcription_results)}\n"
            )
            file_handle.write(f"# Successfully transcribed: {successes}\n")
            file_handle.write(f"# Failed items: {failures}\n")
            for note in metadata_notes or []:
                file_handle.write(f"# Note: {note}\n")
            file_handle.write(
                f"# Total processing time for this item: {elapsed_str}\n\n"
            )

            for index, result in enumerate(transcription_results):
                transcription = result.get(
                    "transcription", "[ERROR] Transcription data missing"
                )
                # A reused or corrupt log entry can carry a None / non-string
                # transcription; coerce it so file.write() cannot raise TypeError
                # (which would escape past the tmp-file cleanup below).
                if not isinstance(transcription, str):
                    transcription = str(transcription)
                file_handle.write(transcription)
                if index < len(transcription_results) - 1:
                    file_handle.write("\n")

        os.replace(tmp_path, output_path)
        logger.info("Transcription text file saved: %s", output_path)
        return True
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        logger.error(
            "Error writing transcription to text file %s: %s", output_path, exc
        )
        return False
    except Exception:
        # Any non-OSError failure (e.g. a corrupt log entry) must not orphan the
        # sibling .txt.tmp file; clean it up before re-raising.
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
