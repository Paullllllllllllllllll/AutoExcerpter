# AutoExcerpter v4.0 Release Notes

**Release Date:** April 2026

AutoExcerpter v4.0 is a deep-module restructure. The top-level layout now
reflects cohesion clusters rather than incremental accretion: seven
packages (`config/`, `llm/`, `imaging/`, `rendering/`, `pipeline/`,
`cli/`, plus root-level leaf utilities) replace the five-package layout
with its flat 18-file `modules/` grab bag. Every public module has a
narrow, curated interface; implementation details are package-private.

Core functionality and user-facing behaviour are unchanged. All 1,204
tests pass. No CLI arguments, output file formats, or YAML schema field
names have changed. Only file locations and import paths have moved.

## Breaking Changes

### Directory Restructure

| Old path | New path |
|---|---|
| `api/` | `llm/` (with files renamed — see below) |
| `core/` | `pipeline/` |
| `processors/docx_writer.py`, `markdown_writer.py`, `file_manager.py`, `citation_manager.py` | `rendering/` (split: `text.py`, `summary.py`, `docx.py`, `markdown.py`, `citations.py`) |
| `processors/pdf_processor.py` | `imaging/pdf.py` |
| `processors/log_manager.py` | `pipeline/log.py` |
| `modules/app_config.py`, `config_loader.py`, `concurrency_helper.py`, `constants.py` | `config/app.py`, `config/loader.py`, `config/accessors.py`, `config/constants.py` |
| `modules/config/*.yaml` | `config/defaults/*.yaml` |
| `modules/image_utils.py`, `model_utils.py` | `imaging/preprocessing.py`, `imaging/_provider.py` |
| `modules/prompts/`, `modules/schemas/` | `llm/resources/prompts/`, `llm/resources/schemas/` |
| `modules/prompt_utils.py` | `llm/prompts.py` |
| `modules/item_scanner.py`, `context_resolver.py`, `path_utils.py` | `pipeline/scanner.py`, `pipeline/context.py`, `pipeline/paths.py` |
| `modules/user_prompts.py` | `cli/interaction.py` |
| `modules/error_handler.py` | `cli/errors.py` |
| `cli/argument_parser.py`, `cli/processing.py` | `cli/args.py`, `cli/loop.py` |
| `modules/logger.py` | `logger.py` (root) |
| `modules/token_tracker.py` | `token_tracker.py` (root) |
| `modules/types.py` | `app_types.py` (root; renamed to avoid shadowing stdlib `types`) |
| `modules/text_cleaner.py` | `text_cleaner.py` (root) |
| `modules/roman_numerals.py` | inlined into `rendering/summary.py` (file removed) |

The `modules/` package and `api/`, `core/`, `processors/` top-level packages
are gone. The `api/` → `llm/` rename also renamed every sub-file inside:

| Old `api/` path | New `llm/` path |
|---|---|
| `api/llm_client.py` | `llm/client.py` |
| `api/base_llm_client.py` | `llm/base.py` |
| `api/model_capabilities.py` | `llm/capabilities.py` |
| `api/rate_limiter.py` | `llm/rate_limit.py` |
| `api/transcribe_api.py` | `llm/transcription.py` |
| `api/summary_api.py` | `llm/summary.py` |

### Public-Surface Reductions

- `LLMClientBase` is no longer exported from `llm.base.__all__`. It is a
  package-private base class; prefer the concrete `TranscriptionManager`
  and `SummaryManager` subclasses exposed via `llm/__init__.py`. Tests
  that need to exercise the base can still import from `llm.base`.
- `RateLimiter` is no longer exported from the `llm` package facade. Both
  managers now construct their own limiters from
  `config.accessors.get_rate_limits()`. External orchestration code (e.g.,
  `pipeline.transcriber.ItemTranscriber`) no longer needs to know about
  `RateLimiter`.

### Lazy API-Key Validation

`config.app` (formerly `modules/app_config.py`) no longer raises
`EnvironmentError` at import time if no LLM provider API key is set.
The check has been replaced with two lazy helpers:

- `config.app.get_available_providers() -> list[str]` — list providers
  whose API key is currently set.
- `config.app.require_api_key(provider) -> str` — return the API key for
  the given provider or raise `EnvironmentError` if unset.

External callers that relied on the import-time raise for fail-fast
configuration validation should invoke `require_api_key(provider)` just
before the first LLM call.

### YAML Defaults Location

User-edited YAML files must be moved from `modules/config/` to
`config/defaults/`:

```
modules/config/app.yaml              → config/defaults/app.yaml
modules/config/model.yaml            → config/defaults/model.yaml
modules/config/concurrency.yaml      → config/defaults/concurrency.yaml
modules/config/image_processing.yaml → config/defaults/image_processing.yaml
```

The `.gitignore` is updated to reflect the new location.

## Architectural Improvements

- **`api/__init__.py` dead facade** replaced by an active
  `llm/__init__.py` curated interface. External callers now import
  through the facade: `from llm import TranscriptionManager, SummaryManager`.
- **DOCX and Markdown writers now share** the OpenAlex enrichment call
  through a single `rendering.citations.enrich_if_enabled` helper,
  eliminating one of the two copy-pasted blocks between the two writers.
  The references-rendering loops remain format-specific (genuine
  divergence: DOCX paragraph construction vs. Markdown string output).
- **`token_tracker.__all__` staleness fixed**: the non-existent
  `reset_token_tracker` entry was removed; the leaking `config` re-export
  was removed.
- **`roman_numerals.int_to_roman`** (37-LOC single-file module) inlined
  into `rendering/summary.py` where it is used.
- **`modules/model_utils.py`** (tiny 2-function helper) moved into
  `imaging/_provider.py` as a package-private helper.
- **`app_types.py` name change** avoids shadowing the stdlib `types`
  module at the package root.

## Known Tech Debt (Not Addressed in v4.0)

Three shallow-module symptoms identified during the refactor planning
remain for follow-up PRs because fixing them requires substantial test
rewrites:

1. Three independent provider-detection implementations still coexist
   (`llm.client._infer_provider`, `llm.capabilities.detect_provider`,
   `imaging._provider.detect_model_type`). The capability registry in
   `llm.capabilities` remains the canonical one. Unifying the others
   requires rewriting `tests/test_llm_client.py`, which asserts the
   current `_infer_provider` behaviour directly.
2. The schema-retry loop is structurally identical in
   `llm/transcription.py` and `llm/summary.py` (~150 lines each). Pushing
   the common loop down into `LLMClientBase._invoke_with_schema_retries`
   requires coordinated changes to both managers and their test files.
3. `ItemTranscriber` in `pipeline/transcriber.py` remains a 700-LOC
   orchestrator that owns working-directory setup, concurrent page
   dispatch, ETA computation, and output-file writing. Splitting it into
   a slim orchestrator plus `PageBatchProcessor` and `WorkingContext`
   helpers is the largest single deferred refactor.

## Deleted Files

- `api/__init__.py`, `core/__init__.py`, `processors/__init__.py`,
  `modules/__init__.py`, `modules/config/__init__.py` — empty package
  markers for packages that no longer exist.
- `modules/roman_numerals.py` — 37 LOC, single function inlined into
  `rendering/summary.py`.
- `processors/file_manager.py` — 400 LOC, split into `rendering/text.py`
  (transcription output) and `rendering/summary.py` (summary preparation).
- `RELEASE_NOTES_v3.0.md` — superseded by this document, following the
  precedent set by v3.0 (which deleted v2.0's notes).

## Test Suite

1,204 tests pass (27 s). One obsolete test was removed:
`test_config_loader.py::TestPathConstants::test_modules_dir_exists`
(asserted the existence of `modules/`, which no longer exists). The
`MODULES_DIR` constant has been removed from `config/loader.py`.

Test files renamed for clarity:

| Old test path | New test path |
|---|---|
| `tests/test_user_prompts.py` | `tests/test_cli_interaction.py` |
| `tests/test_error_handler.py` | `tests/test_cli_errors.py` |
| `tests/test_argument_parser.py` | `tests/test_cli_args.py` |
| `tests/test_transcriber.py` (was misnamed) | `tests/test_pipeline_page_numbering.py` |
| `tests/test_item_scanner.py` | `tests/test_pipeline_scanner.py` |
| `tests/test_context_resolver.py` | `tests/test_pipeline_context.py` |
| `tests/test_path_utils_extended.py` | `tests/test_pipeline_paths.py` |
| `tests/test_pdf_processor.py` | `tests/test_imaging_pdf.py` |
| `tests/test_pdf_processor_extended.py` | `tests/test_imaging_pdf_extended.py` |
| `tests/test_image_utils.py` | `tests/test_imaging_preprocessing.py` |
| `tests/test_model_utils.py` | `tests/test_imaging_provider.py` |
| `tests/test_file_manager.py` | `tests/test_rendering_summary.py` |
| `tests/test_file_manager_extended.py` | `tests/test_rendering_extended.py` |
| `tests/test_citation_manager.py` | `tests/test_rendering_citations.py` |
| `tests/test_prompt_utils_extended.py` | `tests/test_llm_prompts.py` |

Change statistics: 90 files changed, 1,059 insertions, 899 deletions.

## Upgrade Guide

For downstream tooling or scripts that import from AutoExcerpter
internals:

1. **Move user YAML configs**:

   ```
   modules/config/app.yaml              → config/defaults/app.yaml
   modules/config/model.yaml            → config/defaults/model.yaml
   modules/config/concurrency.yaml      → config/defaults/concurrency.yaml
   modules/config/image_processing.yaml → config/defaults/image_processing.yaml
   ```

2. **Update import statements** via the mapping table below:

   | Old import | New import |
   |---|---|
   | `from api.transcribe_api import TranscriptionManager` | `from llm import TranscriptionManager` |
   | `from api.summary_api import SummaryManager` | `from llm import SummaryManager` |
   | `from api.llm_client import LLMConfig, get_chat_model` | `from llm import LLMConfig, get_chat_model` |
   | `from api.model_capabilities import detect_capabilities` | `from llm import detect_capabilities` |
   | `from api.rate_limiter import RateLimiter` | (no longer needed; managers own their limiter) |
   | `from core.transcriber import ItemTranscriber` | `from pipeline import ItemTranscriber` |
   | `from core.resume import ResumeChecker` | `from pipeline import ResumeChecker` |
   | `from modules.item_scanner import scan_input_path` | `from pipeline import scan_input_path` |
   | `from processors.pdf_processor import extract_pdf_pages_to_images` | `from imaging import extract_pdf_pages_to_images` |
   | `from processors.docx_writer import create_docx_summary` | `from rendering import create_docx_summary` |
   | `from processors.markdown_writer import create_markdown_summary` | `from rendering import create_markdown_summary` |
   | `from processors.file_manager import write_transcription_to_text` | `from rendering import write_transcription_to_text` |
   | `from processors.citation_manager import CitationManager` | `from rendering.citations import CitationManager` |
   | `from modules import app_config as config` | `from config import app as config` |
   | `from modules.config_loader import get_config_loader` | `from config import get_config_loader` |
   | `from modules.concurrency_helper import get_rate_limits, ...` | `from config.accessors import get_rate_limits, ...` |
   | `from modules.constants import DEFAULT_MODEL, ...` | `from config.constants import DEFAULT_MODEL, ...` |
   | `from modules.logger import setup_logger` | `from logger import setup_logger` |
   | `from modules.token_tracker import get_token_tracker` | `from token_tracker import get_token_tracker` |
   | `from modules.text_cleaner import clean_transcription` | `from text_cleaner import clean_transcription` |
   | `from modules.types import CustomEndpointCapabilities, ItemSpec` | `from app_types import CustomEndpointCapabilities, ItemSpec` |
   | `from modules.user_prompts import print_error, prompt_yes_no` | `from cli.interaction import print_error, prompt_yes_no` |
   | `from modules.error_handler import handle_critical_error` | `from cli.errors import handle_critical_error` |
   | `from modules.image_utils import ImageProcessor` | `from imaging import ImageProcessor` |
   | `from modules.prompt_utils import render_prompt_with_schema` | `from llm.prompts import render_prompt_with_schema` |

3. **Replace import-time API-key checks** with lazy validation:

   ```python
   # v3.0:
   import modules.app_config  # raises EnvironmentError if no key set

   # v4.0:
   from config import app as config
   config.require_api_key("openai")  # raises only if/when needed
   ```

4. **If you used `RateLimiter` directly**, remove the import and the
   construction — `TranscriptionManager` and `SummaryManager` now own
   their rate limiters.

5. **Custom user scripts that imported** `strip_markdown_code_block`
   from `text_cleaner` must update to `from llm.prompts import
   strip_markdown_code_block` — the helper moved because it is a
   response-parsing concern, not a transcription-cleaning concern.

6. **No changes** to CLI arguments, YAML schema field names, output file
   formats, or user-visible behaviour.
