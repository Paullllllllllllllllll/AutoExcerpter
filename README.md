# AutoExcerpter v1.0.1

AutoExcerpter is a document processing pipeline that transcribes
and summarizes PDFs and image collections using vision-enabled
LLMs. Built on LangChain, it supports OpenAI, Anthropic, Google,
OpenRouter, and custom OpenAI-compatible endpoints. Scanned pages
are converted to structured text with optional summaries and
citation-enriched bibliographies.

> **Work in Progress** -- AutoExcerpter is under active
> development. Bugs may be present. If you encounter any issues,
> please [report them on GitHub](https://github.com/Paullllllllllllllllll/AutoExcerpter/issues).

## Table of Contents

-   [Overview](#overview)
-   [Key Features](#key-features)
-   [Supported Models](#supported-models)
-   [How It Works](#how-it-works)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Usage](#usage)
    -   [Interactive Mode (Default)](#interactive-mode-default)
    -   [CLI Mode](#cli-mode)
    -   [Processing Modes](#processing-modes)
    -   [Batch Processing](#batch-processing)
-   [Configuration](#configuration)
    -   [Basic Configuration (app.yaml)](#basic-configuration-appyaml)
    -   [Model Configuration (model.yaml)](#model-configuration-modelyaml)
    -   [Concurrency Configuration (concurrency.yaml)](#concurrency-configuration-concurrencyyaml)
    -   [Image Processing Configuration (image_processing.yaml)](#image-processing-configuration-image_processingyaml)
    -   [Summary Context](#summary-context)
    -   [Citation Management](#citation-management)
    -   [Daily Token Limit](#daily-token-limit)
-   [Output Files](#output-files)
-   [Project Structure](#project-structure)
-   [Advanced Topics](#advanced-topics)
-   [Troubleshooting](#troubleshooting)
-   [Versioning](#versioning)
-   [Contributing](#contributing)
-   [License](#license)

## Overview

AutoExcerpter processes documents in two stages. In the first
stage, each page is sent to a vision-enabled LLM and transcribed
into structured JSON conforming to a strict schema. In the
optional second stage, transcribed text is summarized with
bullet-point extraction, and citations are deduplicated and
enriched via the OpenAlex academic database.

**Primary use cases:**
digitizing scanned academic papers and archival documents;
generating structured literature summaries with consolidated
bibliographies; extracting content from multi-column layouts,
mathematical notation (LaTeX), tables, and figures.

## Key Features

**Transcription:**
PDF and image folder input (PNG, JPG, TIFF, BMP, GIF, WEBP);
structure preservation (headers, footers, footnotes, tables,
multi-column layouts); LaTeX math conversion; visual content
descriptions; schema-driven JSON output; in-memory image
preprocessing; configurable text cleaning (Unicode normalization,
LaTeX repair, hyphenation merging, whitespace cleanup, line
wrapping).

**Summarization:**
bullet-point summaries per page; automatic exclusion of
non-semantic pages (title pages, blanks, reference lists);
page-number tracking from document headers/footers; dual output
(DOCX + Markdown); hierarchical context system for topic-focused
summarization.

**Citation management:**
normalized-hash deduplication; page-range consolidation
(e.g., "pp. 3, 7-9, 15"); OpenAlex metadata enrichment (DOI,
authors, year, venue); clickable hyperlinks in DOCX; consolidated
bibliography section.

**Performance and reliability:**
concurrent page transcription via ThreadPoolExecutor; sliding-
window rate limiter; exponential backoff with jitter; schema-
specific retries (validation failures, content flags); daily
token budgeting with midnight reset and persistent state; OpenAI
Flex tier support; progress bars; JSONL audit logs.

**Multi-provider architecture:**
unified LangChain interface; provider auto-detection from model
name; centralized capability registry that guards unsupported
parameters per model; custom endpoint support with three usage
patterns (full structured, plain text, prompt-guided JSON).

## Supported Models

**OpenAI:**

| Family | Models | Notes |
|---|---|---|
| GPT-5.4 Pro/5.4 | gpt-5.4-pro, gpt-5.4 | Reasoning, verbosity, 1.05M ctx |
| GPT-5.3/5.2 | gpt-5.3, gpt-5.2, gpt-5.2-pro | Reasoning, verbosity, 400k ctx |
| GPT-5.1 | gpt-5.1, gpt-5.1-instant, gpt-5.1-thinking | Reasoning, verbosity |
| GPT-5 | gpt-5, gpt-5-mini, gpt-5-nano | Reasoning, verbosity |
| O-series | o4, o4-mini, o3, o3-mini, o1, o1-mini | Reasoning (no temperature) |
| GPT-4.1 | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | Multimodal |
| GPT-4o | gpt-4o, gpt-4o-mini | Multimodal |

**Anthropic:**

| Family | Models | Notes |
|---|---|---|
| Claude 4.6 | claude-opus-4-6, claude-sonnet-4-6 | Extended thinking, 200k ctx |
| Claude 4.5 | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 | Extended thinking (opus/sonnet) |
| Claude 4 | claude-opus-4, claude-sonnet-4 | Extended thinking |
| Claude 3.x | claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku | Multimodal |

**Google:**

| Family | Models | Notes |
|---|---|---|
| Gemini 3 Preview | gemini-3-pro-preview, gemini-3-flash-preview | Thinking, 1M ctx |
| Gemini 3 | gemini-3-pro, gemini-3-flash | Thinking |
| Gemini 2.5 | gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite | Thinking |
| Gemini 2.0 | gemini-2.0-flash, gemini-2.0-flash-lite | Multimodal |
| Gemini 1.5 | gemini-1.5-pro, gemini-1.5-flash | Multimodal |

**OpenRouter:** any supported model via unified API.

**Custom endpoint:** any OpenAI-compatible server with vision
support. Set `provider: custom` in `model.yaml` and configure
the `custom_endpoint` block:

```yaml
transcription_model:
  provider: custom
  name: "org/model-name"
  custom_endpoint:
    base_url: "https://your-endpoint.example.com/v1"
    api_key_env_var: "YOUR_CUSTOM_API_KEY"
    capabilities:
      supports_vision: true
      supports_structured_output: false
      use_plain_text_prompt: false
```

Three capability patterns are available:

| Pattern | structured_output | plain_text_prompt | Behavior |
|---|---|---|---|
| A: Full structured | true | false | API-level JSON schema enforcement |
| B: Plain text | false | true | Simplified prompt, raw text response |
| C: Prompt-guided JSON | false | false | Schema in prompt, validation retries |

## How It Works

1. **Scan** the input directory for PDFs and image folders.
2. **Extract** PDF pages as high-resolution images (PyMuPDF).
3. **Preprocess** images in memory (grayscale, transparency
   handling, resize, JPEG encode).
4. **Transcribe** each page via LangChain with structured JSON
   schema and capability-guarded parameters.
5. **Summarize** (optional) transcribed text into bullet-point
   extracts with citation extraction.
6. **Process citations**: deduplicate, consolidate page ranges,
   enrich via OpenAlex.
7. **Render** outputs: `.txt` transcription, `.docx` and `.md`
   summaries, JSONL logs.
8. **Clean up** temporary working directories (if configured).

## Prerequisites

-   Python 3.13+
-   At least one API key: `OPENAI_API_KEY`,
    `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or
    `OPENROUTER_API_KEY`

## Installation

```bash
git clone https://github.com/yourusername/AutoExcerpter.git
cd AutoExcerpter
uv sync
```

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"     # optional
export GOOGLE_API_KEY="your-key"        # optional
export OPENROUTER_API_KEY="your-key"    # optional
```

On Windows, use `set` instead of `export`, or configure via
System Properties.

Configure paths in `config/defaults/app.yaml` and models in
`config/defaults/model.yaml` (see [Configuration](#configuration)).

For development:

```bash
uv sync --extra dev
uv run pytest
```

## Usage

### Interactive Mode (Default)

Set `cli_mode: false` in `app.yaml` (default) and run:

```bash
python main.py
```

The application scans the input directory and presents a
selection menu. Supported selection syntax:

```
42                  # single item
1,5,10              # comma-separated
1-10                # range
all                 # everything
Mennell             # filename search
food history        # text search
```

Exit at any prompt with `exit`, `quit`, or `q`.

### CLI Mode

Set `cli_mode: true` in `app.yaml`.

```bash
python main.py <input> <output> [options]
```

**Core arguments:**

| Argument | Description |
|---|---|
| `<input>` | PDF file, image folder, or directory of items |
| `<output>` | Destination directory |
| `--input-path` / `--output-path` | Named path overrides |
| `--all` | Process all discovered items |
| `--select PATTERN` | Filter by number, range, or filename |
| `--context TOPICS` | Summarization focus topics |
| `--summarize` / `--no-summarize` | Override app.yaml at runtime |
| `--cleanup` / `--no-cleanup` | Override temp dir deletion |
| `--resume` / `--force` | Resume or force-reprocess |

**Model overrides** (global or per-phase with
`--transcription-` / `--summary-` prefix):

`--model`, `--reasoning-effort {minimal,low,medium,high}`,
`--verbosity {low,medium,high}` (GPT-5 only),
`--max-output-tokens N`, `--temperature F`,
`--provider {openai,anthropic,google,openrouter,custom}`

**Examples:**

```bash
# Single PDF
python main.py "./docs/paper.pdf" "./output"

# All items in a directory
python main.py "./docs" "./output" --all

# Filtered selection with context
python main.py "./docs" "./output" --select "1-10" \
  --context "Food History, Wages"

# Per-phase model overrides
python main.py "./docs" "./output" --all \
  --transcription-model "gpt-5.2" \
  --summary-model "gpt-5-mini" \
  --transcription-max-output-tokens 128000

# Shell loop
for pdf in ./papers/*.pdf; do
    python main.py "$pdf" "./output"
done
```

### Processing Modes

**Transcription only** (`summarize: false`): produces `.txt`
files. Faster, cheaper, suitable for building text archives.

**Transcription + summarization** (`summarize: true`): adds
`.docx` and/or `.md` summaries with extracted citations.

### Batch Processing

Place all PDFs and image folders in the input directory, then
select "all" interactively or pass `--all` in CLI mode. Already-
processed items are skipped automatically (override with
`--force`).

## Configuration

Four YAML files in `config/defaults/` (all gitignored):

| File | Purpose |
|---|---|
| `app.yaml` | Paths, feature toggles, citations, daily token limit |
| `model.yaml` | LLM provider/model for transcription and summary |
| `concurrency.yaml` | Rate limits, retries, parallelism, service tier |
| `image_processing.yaml` | DPI, grayscale, resize, JPEG quality, text cleaning |

### Basic Configuration (app.yaml)

```yaml
cli_mode: false
summarize: true
summary_output:
  docx: true
  markdown: true

input_folder_path: 'C:\path\to\PDFs'
output_folder_path: ''
input_paths_is_output_path: true
delete_temp_working_dir: true

citation:
  openalex_email: 'you@example.com'
  max_api_requests: 300
  enable_openalex_enrichment: true

daily_token_limit:
  enabled: true
  daily_tokens: 9000000
```

### Model Configuration (model.yaml)

Separate sections for `transcription_model` and `summary_model`,
each with:

```yaml
transcription_model:
  name: "gpt-5.2"
  provider: "openai"        # auto-detected if unambiguous
  max_output_tokens: 12000
  temperature: 1.0
  reasoning:
    effort: medium           # maps to native reasoning param
  text:
    verbosity: medium        # GPT-5 family only
```

**Reasoning effort mapping:**
OpenAI uses `reasoning_effort`; Anthropic maps to
`budget_tokens` (low=2048, medium=4096, high=8192); Google maps
to `thinking_budget`.

**Provider examples:**

```yaml
# Anthropic
transcription_model:
  name: "claude-sonnet-4-5-20250929"
  provider: "anthropic"
  max_output_tokens: 16384
  temperature: 1.0

# Google
transcription_model:
  name: "gemini-2.5-pro"
  provider: "google"
  max_output_tokens: 32768
  reasoning:
    effort: high
```

### Concurrency Configuration (concurrency.yaml)

```yaml
image_processing:
  concurrency_limit: 24

api_requests:
  api_timeout: 900
  rate_limits:
    - [120, 1]
    - [15000, 60]
    - [15000, 3600]
  transcription:
    concurrency_limit: 5
    delay_between_tasks: 0.1
    service_tier: default    # default | flex | priority
  summary:
    concurrency_limit: 5
    delay_between_tasks: 0.1
    service_tier: flex

retry:
  max_attempts: 5
  backoff_base: 1.0
  backoff_multipliers:
    rate_limit: 2.0
    timeout: 1.5
    server_error: 2.0
    other: 2.0
  jitter: { min: 0.5, max: 1.0 }

  schema_retries:
    transcription:
      validation_failure: { enabled: true, max_attempts: 3 }
      no_transcribable_text: { enabled: true, max_attempts: 0 }
      transcription_not_possible: { enabled: true, max_attempts: 3 }
    summary:
      validation_failure: { enabled: true, max_attempts: 3 }
      page_type_null_bullets: { enabled: false, max_attempts: 0 }
```

**Three retry layers** operate in sequence: (1) API errors
(rate limits, timeouts, server errors) with exponential backoff
and jitter; (2) validation retries for malformed JSON or missing
schema keys; (3) per-flag retries for specific model-returned
content conditions.

**Service tiers:** `default` (standard), `flex` (slower,
~50% cheaper), `priority` (faster, higher cost).

### Image Processing Configuration (image_processing.yaml)

Provider-specific sections (`api_image_processing`,
`google_image_processing`, `anthropic_image_processing`,
`custom_image_processing`) each control:

```yaml
api_image_processing:
  target_dpi: 300
  grayscale_conversion: true
  handle_transparency: true
  jpeg_quality: 100
  resize_profile: high       # high | low | auto | none
  llm_detail: high           # high | low | auto
  high_target_box: [768, 1536]
  low_max_side_px: 512
```

Post-transcription text cleaning:

```yaml
text_cleaning:
  enabled: true
  unicode_normalization: true
  latex_fixing:
    enabled: true
    balance_dollar_signs: true
    close_unclosed_braces: true
    fix_common_commands: true
  merge_hyphenation: false
  whitespace_normalization:
    enabled: true
    collapse_internal_spaces: true
    max_blank_lines: 2
  line_wrapping:
    enabled: true
    auto_width: true
    fixed_width: 80
```

### Summary Context

A hierarchical context system guides the summarization model to
focus on specific topics. Resolution order (highest priority
first):

1. CLI `--context` flag or interactive prompt
2. `<filename>_summary_context.txt` beside the input file
3. `<foldername>_summary_context.txt` in the parent directory
4. `context/summary/general.txt` in the project root

Each context file is plain text with one topic per line.

### Citation Management

Configured in `app.yaml` under `citation`. The citation manager
deduplicates via normalized text hashing, consolidates page
ranges, and enriches metadata through the OpenAlex API (free, no
key required). Set `openalex_email` for faster polite-pool
responses.

### Daily Token Limit

Configured in `app.yaml` under `daily_token_limit`. Tracks
`total_tokens` from every API call, persists state to
`.autoexcerpter_token_state.json`, and resets at local midnight.
When the limit is reached, processing pauses with a countdown;
type `q` + Enter to cancel.

## Output Files

For each processed document:

1. **`<name>.txt`** -- verbatim transcription with metadata
   header, LaTeX math, XML-style page tags, and preserved
   structural elements.
2. **`<name>_summary.docx`** -- formatted Word summary with
   bullet-point extracts, LaTeX converted to native Word
   equations (MathML/OMML), and a consolidated bibliography with
   clickable DOI hyperlinks.
3. **`<name>_summary.md`** -- Markdown summary with LaTeX
   preserved as-is for MathJax/KaTeX compatibility.
4. **`<name>_working_files/`** -- JSONL logs for transcription
   and summarization (per-page metadata, timing, retries).
   Auto-deleted when `delete_temp_working_dir: true`.

## Project Structure

```
AutoExcerpter/
├── main.py                          # Entry point
├── config/                          # Configuration package
│   ├── app.py                       # App settings (paths, toggles, API keys)
│   ├── loader.py                    # YAML loader singleton
│   ├── accessors.py                 # Typed config accessors
│   ├── constants.py                 # Hardcoded defaults
│   └── defaults/                    # User-editable YAML configs
│       ├── app.yaml
│       ├── model.yaml
│       ├── concurrency.yaml
│       └── image_processing.yaml
├── llm/                             # LLM client layer
│   ├── client.py                    # Model factory (LLMConfig, get_chat_model)
│   ├── base.py                      # Shared retry, token tracking, capability guard
│   ├── capabilities.py              # Provider/model capability registry
│   ├── rate_limit.py                # Sliding-window rate limiter
│   ├── transcription.py             # TranscriptionManager
│   ├── summary.py                   # SummaryManager
│   ├── prompts.py                   # Prompt rendering + response parsing
│   └── resources/                   # Prompt templates and JSON schemas
├── imaging/                         # PDF rendering + image preprocessing
│   ├── pdf.py                       # Page extraction via PyMuPDF
│   └── preprocessing.py             # In-memory image pipeline
├── rendering/                       # Output writers
│   ├── text.py                      # .txt transcription writer
│   ├── docx.py                      # .docx summary writer
│   ├── markdown.py                  # .md summary writer
│   ├── summary.py                   # Shared summary data preparation
│   └── citations.py                 # Deduplication, OpenAlex enrichment
├── pipeline/                        # Orchestration
│   ├── transcriber.py               # ItemTranscriber (per-item orchestrator)
│   ├── scanner.py                   # Input directory scanning
│   ├── resume.py                    # Resume/checkpoint logic
│   ├── context.py                   # Hierarchical summary context resolution
│   ├── page_numbering.py            # Page-number correction
│   ├── text_cleaner.py              # Post-transcription cleanup
│   ├── paths.py                     # Windows-safe path helpers
│   └── log.py                       # JSONL log lifecycle
├── cli/                             # Command-line interface
│   ├── args.py                      # Argparse + execution-mode resolution
│   ├── display.py                   # Interactive menus + progress display
│   ├── interaction.py               # Terminal I/O primitives
│   ├── loop.py                      # Per-item processing loop
│   └── errors.py                    # Domain exceptions
├── context/summary/general.txt      # Default summarization topics
├── tests/                           # Test suite (1,204 tests)
├── requirements.txt                 # Runtime dependencies
└── requirements-dev.txt             # Dev dependencies (pytest, mypy)
```

## Advanced Topics

**Maximizing throughput:**
increase `concurrency_limit` based on provider tier (OpenAI:
50-150, Anthropic: 5-10); use `service_tier: flex` for batch
work; raise `image_processing.concurrency_limit` to 24-48 on
SSDs; use `llm_detail: auto` or `low` for clean documents.

**Reducing cost:**
use Flex tier (~50% savings); lower `target_dpi` to 200-250 for
clean scans; disable summarization when not needed; use
`llm_detail: low`.

**Best practices:**
start with defaults and adjust after reviewing results; test on
a small document before large batches; review JSONL logs for
failed pages; spot-check transcriptions against sources; verify
citation deduplication and page-number alignment.

## Troubleshooting

**"No API key found"** -- set the environment variable for your
provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) and verify
it is active in the current shell session.

**"Unsupported parameter" errors** -- the capability registry
should filter these automatically. Ensure you are on the latest
version.

**Rate limit errors (429)** -- reduce `concurrency_limit` (try
2-3); adjust `rate_limits` to match your provider tier; consider
`service_tier: flex`.

**Timeout errors** -- increase `api_timeout` in
`concurrency.yaml`; reduce `target_dpi` or use `llm_detail: low`.

**Poor transcription quality** -- increase `target_dpi` (try 400
or 600); set `llm_detail: high`; ensure source scans are legible.

**Memory errors** -- reduce `concurrency_limit` for both API
requests and image processing; process in smaller batches.

**Missing page numbers in summaries** -- verify page numbers are
visible in the source; check the transcription `.txt` for
`<page_number>` tags.

**Further help:** check JSONL logs in `_working_files/`, review
your provider's status page, or open an issue on the repository.

## Versioning

This project uses semantic versioning. The commit history was
squashed to a single baseline commit at v1.0.0 on 25 April
2026. All prior development history was consolidated; version
numbers before v1.0.0 do not exist. v1.0.1 migrates the
project to `pyproject.toml` as the single source of truth for
dependencies and tooling configuration.

## Contributing

Contributions are welcome. Fork the repository, create a feature
branch, and submit a pull request with a clear description.

**Guidelines:** PEP 8 with type annotations; unit tests for new
functionality; clear separation of concerns; use the logger
module for debug output.

**Areas of interest:** additional LLM providers, batch API
integration, enhanced image preprocessing, extended model
capability profiles for new releases.

## License

MIT License

Copyright (c) 2026 Paul Goetz

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
