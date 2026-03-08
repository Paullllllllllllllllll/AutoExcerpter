# AutoExcerpter v3.0 Release Notes

**Release Date:** March 2026

AutoExcerpter v3.0 removes legacy backward-compatibility code from v1.x, overhauls the README, and introduces image processing performance improvements. Core functionality is unchanged.

## Breaking Changes

### Removed Legacy Transcription Flags

The `contains_no_text` and `cannot_transcribe` JSON response flags are no longer recognized by the transcription parser. These were superseded in v2.0 by `no_transcribable_text` and `transcription_not_possible`, which remain fully supported.

**Impact:** Only affects users who built external tooling that emits the old flag names into transcription JSON. The LLM schemas have used the new flags since v2.0.

### Removed Legacy `page_type` Field

The singular `page_type` (string) field is no longer accepted in summary JSON. Use `page_types` (array) instead. When `page_types` is absent, the system defaults to `["content"]`.

**Affected modules:** `api/summary_api.py`, `core/page_numbering.py`, `processors/file_manager.py`

### Removed Legacy Nested Summary Structure

The doubly-nested `summary.summary.page_information` structure is no longer navigated. All summary data must use the flat structure with `page_information`, `bullet_points`, and `references` at the top level (or directly under a single `summary` key).

**Affected modules:** `core/page_numbering.py`, `processors/file_manager.py`

### Removed Legacy Prompt Token

The `{{TRANSCRIPTION_SCHEMA}}` prompt placeholder is no longer recognized. Use `{{SCHEMA}}` instead (supported since v2.0).

**Affected module:** `modules/prompt_utils.py`

## Performance Improvements

### BILINEAR Resampling (Default)

Image downscaling now defaults to `BILINEAR` resampling instead of `LANCZOS`. For document scans, BILINEAR is approximately 2x faster with negligible quality difference for text content.

The algorithm is configurable via `image_processing.yaml`:

```yaml
resampling_algorithm: bilinear  # or 'lanczos' to restore previous behavior
```

### Consolidated Image Preprocessing

The duplicate preprocessing logic between `processors/pdf_processor.py` and `modules/image_utils.py` has been consolidated into a single `ImageProcessor.preprocess_pil_image()` static method. This eliminates code duplication and ensures consistent preprocessing behavior across both the PDF extraction and direct image processing paths.

## Documentation

### README Overhaul

The README has been reduced from 1,586 to 1,043 lines (34% reduction):

- Fixed malformed markdown tables in the Supported Models section (added proper `|` pipe separators)
- Removed references to non-existent documentation files (`docs/USER_PROMPTS_AND_LOGGING.md`, `docs/CLI_MODE.md`)
- Updated Project Structure to reflect the actual codebase (removed deleted `api/providers/`, added `cli/`, `core/page_numbering.py`, `core/resume.py`, and all `modules/` and `processors/` utility files)
- Merged "Quick Start", "CLI Mode and Interactive Mode", and "Usage" into a single Usage section
- Trimmed duplicate configuration examples and verbose Output Files section
- Fixed "How It Works" step 5 to reference the configured LLM provider instead of hardcoded "OpenAI API"
- Updated copyright year to 2026 and version references to v3.0
- Added GPT-5.4 Pro, GPT-5.4, and GPT-5.3 to the Supported Models tables

## Deleted Files

- `.windsurf/` — Unused IDE workflow files
- `RELEASE_NOTES_v2.0.md` — Superseded by this document

## Test Suite

All 1,053 tests pass. Six legacy test cases were removed alongside the code they tested. Overall coverage is 90%.

## Upgrade Guide

1. **If you use custom prompt templates:** Replace any `{{TRANSCRIPTION_SCHEMA}}` tokens with `{{SCHEMA}}`.
2. **If you parse summary JSON externally:** Ensure your code uses `page_types` (array) instead of `page_type` (string), and reads `page_information` from the top level rather than navigating `summary.summary`.
3. **If you want LANCZOS resampling:** Add `resampling_algorithm: lanczos` to your `image_processing.yaml`.
4. No changes to configuration file formats, CLI arguments, or output file structures.
