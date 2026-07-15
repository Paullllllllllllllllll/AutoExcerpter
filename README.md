# AutoExcerpter v1.26.0

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

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Interactive Mode (Default)](#interactive-mode-default)
  - [CLI Mode](#cli-mode)
  - [Processing Modes](#processing-modes)
  - [Batch Processing](#batch-processing)
- [Configuration](#configuration)
  - [Basic Configuration (app.yaml)](#basic-configuration-appyaml)
  - [Model Configuration (model.yaml)](#model-configuration-modelyaml)
  - [Concurrency Configuration (concurrency.yaml)](#concurrency-configuration-concurrencyyaml)
  - [Image Processing Configuration (image_processing.yaml)](#image-processing-configuration-image_processingyaml)
  - [Summary Context](#summary-context)
  - [Citation Management](#citation-management)
  - [Daily Token Limit](#daily-token-limit)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)
- [Versioning](#versioning)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

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
page-number tracking from document headers/footers, including
two-page-spread scans rendered as page ranges ("Pages 13-14");
section-grouped page ordering in the final summary; dual output
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
token budgeting that resets at 00:01 UTC (one minute after OpenAI's
00:00 UTC free-tier reset) with persistent state; OpenAI
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
2. **Stream** each page fully in memory: render the PDF page
   (PyMuPDF) or load the source image on demand in the
   transcription worker, preprocess (grayscale, transparency
   handling, resize), and JPEG/base64-encode -- no temporary
   image files are written to disk. Already-transcribed pages
   are skipped before rendering when resuming.
3. **Record provenance**: each log entry carries the SHA-256,
   dimensions, byte size, and effective DPI of the exact image
   sent to the API; the log header records the source-file
   SHA-256, library versions, and the image-config snapshot.
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

- Python 3.13+
- At least one API key: `OPENAI_API_KEY`,
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

On a fresh clone the application runs immediately using the scrubbed
`*.example.yaml` templates that ship with the repository. Each missing real
config triggers one informational log line pointing you to the example file.
To customize, copy the relevant example and edit it:

```bash
cp config/defaults/app.example.yaml config/defaults/app.yaml
cp config/defaults/model.example.yaml config/defaults/model.yaml
# repeat for concurrency, image_processing, api_keys as needed
```

See [Configuration](#configuration) for all available settings.

For development, install all dependencies including dev extras:

```bash
uv sync --extra dev
```

Run the test suite, linter, and type checker:

```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run mypy .
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
| `--retranscribe` | Re-transcribe resumable items instead of reusing logged transcriptions |
| `--cli` / `--interactive` | Force execution mode, overriding `cli_mode` in `app.yaml` |
| `--json` | Emit one machine-readable JSON run-summary line on stdout at exit |
| `--dry-run` | Discover inputs and classify resume state without any API calls or writes |

**Agent contract.** The primary entry point aggregates per-item status and
sets the process exit code: `0` = all requested items succeeded; `1` = one or
more items failed or were partial; `2` = usage/config error (including a
non-TTY invocation of interactive mode, or multiple discovered items with
neither `--all` nor `--select`); `130` = user interrupt. `--cli` /
`--interactive` let an agent drive the tool without editing the gitignored
YAML. `--dry-run` reports the planned actions (and, with `--json`, a JSON
plan) and exits without side effects.

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

Each config file ships in two forms:

- **`<name>.example.yaml`** (tracked) -- scrubbed template with conservative
  defaults. Loaded automatically when the real file is absent; a single INFO
  log line tells you to copy and customize it.
- **`<name>.yaml`** (gitignored) -- your private, machine-specific settings.
  Takes precedence over the example when present.

To start from a template, copy it and remove the `.example` infix:

```bash
cp config/defaults/app.example.yaml config/defaults/app.yaml
```

| Real file | Example template | Purpose |
|---|---|---|
| `app.yaml` | `app.example.yaml` | Paths, feature toggles, citations, daily token limit |
| `model.yaml` | `model.example.yaml` | LLM provider/model for transcription and summary |
| `concurrency.yaml` | `concurrency.example.yaml` | Rate limits, retries, parallelism, service tier |
| `image_processing.yaml` | `image_processing.example.yaml` | DPI, grayscale, resize, JPEG quality, text cleaning |
| `api_keys.yaml` | `api_keys.example.yaml` | Provider to env-var-name mapping (optional) |

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
  daily_tokens: 9000000   # combined cap across tools (secondary guard)
  scope: pooled           # pooled = cap only calls in a defined pool; all = legacy
  per_key_pool_caps:      # per-(API key, pool) daily caps (primary gate)
    enabled: true
    openai:
      small: 9750000      # bare int: cap; model list from built-in defaults
      large:
        cap: 975000       # mapping form: custom cap and/or model prefixes
        # models: ["gpt-5", "o3"]
    # any provider can define its own named pools:
    # myhost:
    #   standard:
    #     cap: 5000000
    #     models: ["my-model"]
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

Configured in `app.yaml` under `citation`. Extracted citations are folded
(accents, case, curly quotes, `&`) and keyed structurally by
`(author, year, volume, title)`, so different editions never collapse
together. A conservative fuzzy pass then merges near-identical variants only
within a `(first-author surname, year)` block, gated by `merge_ratio`
(SequenceMatcher) or `merge_jaccard` (token-set); differing years or volumes
never merge, and every merge is logged. Page ranges are consolidated (including
`unnumbered` pages). The summary schema (`Summary_2_3_0`) additionally flags
in-text-only citations (`is_partial`): the model records bare author-year
pointers as-is instead of fabricating placeholder titles, and after the fuzzy
pass each partial stub whose tokens are contained in exactly one full
reference within its author-year block merges into it (pages union); stubs
with no full match, or with several ambiguous same-author same-year
candidates, are dropped rather than guessed, with every merge and drop
logged. OpenAlex enrichment (free, no key required; set
`openalex_email` for the faster polite pool) runs exactly once per document and
is shared by both writers. A candidate links only when title-word overlap
clears `match_title_overlap` AND a corroborating signal matches (publication
year within +/-1 of a cited year, or the candidate author surname appears in
the citation) -- preferring no link over a wrong one. Every OpenAlex request
counts against `max_api_requests`, and results are cached across runs in the
state directory. When OpenAlex signals daily-quota exhaustion (a 429 with a
long `retryAfter`), enrichment latches off process-wide and across runs
(`openalex_budget.json` in the state directory) until the quota window
passes; remaining citations are still served from the persistent cache, and
short rate-limit waits are slept through and retried.

### State Directory

Daily token-budget state and the persistent OpenAlex cache live under a
user-level directory (`~/.autoexcerpter` by default), overridable via
`paths.state_dir` in `app.yaml`. A legacy `.autoexcerpter_token_state.json` in
the working directory is adopted once when the user-level file is absent.

### Daily Token Limit

Configured in `app.yaml` under `daily_token_limit`. Tracks `total_tokens` from
every API call, persists debounced state to `token_state.json` in the state
directory, and resets at 00:01 UTC (one minute after OpenAI's 00:00 UTC
free-tier reset). The budget is enforced per page
(transcription plus optional summary reserved together); when it cannot fit
another page, in-flight pages drain, the run waits for the reset, then resumes
the still-pending pages from the log. An item is never marked complete while
its log shows pages missing.

#### Shared Cross-Tool Token Budget (optional)

AutoExcerpter can share ONE combined daily budget with its sibling tools
(ChronoMiner, ChronoTranscriber) instead of enforcing its cap in isolation.
Off by default; single-tool installations need not care. Enable it in
`app.yaml`:

```yaml
shared_token_budget:
  enabled: true
  ledger_dir: ''   # empty = ~/.chronopipeline; or an absolute path
```

When enabled, every participating tool merges its usage into one shared
ledger (`token_ledger.json`) guarded by an OS file lock, and
`daily_token_limit.daily_tokens` is enforced against the COMBINED total, so
several tools running concurrently cannot collectively overshoot the budget.
Usage is merged as deltas under the lock (concurrent processes lose
nothing); the hot path stays in memory with a debounced sync, plus forced
refreshes near the cap and while waiting at the limit. The `--json` summary
adds `combined_tokens_today` alongside the tool's own `tokens_used_today`,
and token summaries show the per-tool breakdown. If the ledger is ever
unavailable, the tool degrades to its private counter with a single warning
and never crashes. Keep `daily_tokens` identical across participating
tools; the strictest value simply stops its tool first. Editing
`daily_tokens` while a run waits at the limit lifts the cap within a poll
cycle, no restart needed.

#### Per-Key-Pool Accounting and Caps

A "pool" is a named set of models that share one daily token allowance per
API key. Pools are defined per provider in `per_key_pool_caps` — each entry
gives a cap and, optionally, a model prefix list — and built-in defaults
mirroring OpenAI's complimentary daily token program apply when a provider
has no configured model lists, so zero-config installs keep working. Every
API call's usage is stamped with its provider, the NAME of the environment
variable that served it (key values are never stored or logged), and the
pool derived from the model name. The shared ledger records a per-(tool,
provider, key env, pool) breakdown alongside the per-tool totals, so you
can always tell how much of a daily allowance remains on any key you use.
Enforcement is two-tier: `per_key_pool_caps` gates each key's own pool (set
your own caps and pools, or disable the gate, to match your account's
terms), and `daily_tokens` remains a combined secondary guard. Under the
default `scope: pooled`, calls whose model belongs to no pool — local or
self-hosted endpoints, providers without an allowance program — are counted
but never blocked. The transcription and summary roles stamp their buckets
independently, so an exhausted pooled summary key never blocks a pool-less
transcription endpoint running in the same process (with `scope: all` the
combined cap applies to every call, the legacy behavior). When a key's pool
cap is reached, the wait message names the exhausted key and reports the
remaining pool of any other keys visible in the ledger; remapping a
provider to a different key env var in `api_keys.yaml` is picked up at the
next item without a restart. Usage that predates the upgrade (or arrives
from un-stamped paths) is kept under an "unattributed" row and counts
toward the combined total only; a v1 ledger is adopted in place without
losing the day's count.

## Output Files

For each processed document (`<name>` is the input file/folder stem):

1. **`<name>.txt`** -- verbatim transcription with metadata
   header, LaTeX math, XML-style page tags, and preserved
   structural elements.
2. **`<name>.docx`** -- formatted Word summary with
   bullet-point extracts, LaTeX converted to native Word
   equations (MathML/OMML), and a consolidated bibliography with
   clickable DOI hyperlinks.
3. **`<name>.md`** -- Markdown summary with LaTeX
   preserved as-is for MathJax/KaTeX compatibility.
4. **`<name>_working_files/`** -- versioned JSONL logs for
   transcription and summarization (one JSON object per line;
   per-page metadata, timing, retries, image provenance;
   file-level provenance in the header). Resume refuses logs
   lacking the current format-version marker. Auto-deleted when
   `delete_temp_working_dir: true`.

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
│   ├── payload.py                   # Streaming page payload sources (PyMuPDF)
│   ├── pdf.py                       # Image folder scanning
│   └── preprocessing.py             # In-memory preprocessing core
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
├── tests/                           # Test suite (1,232 tests)
├── pyproject.toml                   # Project metadata and dependencies
└── uv.lock                         # Pinned dependency lockfile
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

This project follows semantic versioning (`MAJOR.MINOR.PATCH`). The version in
`pyproject.toml` is the single source of truth; it is mirrored in the title
heading above and tagged in git as `vX.Y.Z`. The commit history was squashed to
a single baseline commit at v1.0.0 on 25 April 2026; version numbers before
v1.0.0 do not exist.

## Changelog

- **v1.26.0** (15 July 2026) -- Stop fabricated citation stubs from
    duplicating full references. The summary schema (`Summary_2_3_0`) turns
    `references` items into `{citation, is_partial}` objects and both summary
    prompts instruct the model to record in-text author-year pointers as-is
    (never inventing bracketed placeholder titles) and to flag them
    `is_partial`. After the fuzzy consolidation pass, a partial stub whose
    token set is contained in exactly one full reference within its
    author-year block merges into it (union of pages, full text stays
    canonical), while stubs with no or several ambiguous full matches are
    dropped rather than guessed; every merge and drop is logged. Legacy
    cached/resumed responses with plain-string references still parse.
    Verified end-to-end on a live document whose reference list shrank from
    nine entries (four bracketed stubs) to five clean entries with correct
    merged page ranges.
- **v1.25.0** (15 July 2026) -- Restyle both summary writers into dense
    academic reference documents. The DOCX writer now applies explicit named
    styles (Times New Roman throughout, with theme-font and theme-color
    attributes stripped so Word cannot substitute Calibri), A4 geometry with
    2 cm margins, a bottom-right page-number footer, a thin rule under
    section headings, keep-with-next page headings in a restrained navy,
    compact hanging-indent bullets, and a tighter references layout;
    Markdown emphasis inside citation text is rendered as italic/bold runs
    instead of literal asterisks. The Markdown writer drops the "Summary of"
    title prefix, adds a compact metadata line, groups pages as H3 sections
    under "Page Summaries", and fences references behind a single rule.
    Verified end-to-end with a live gpt-5.6-luna run and visual inspection
    of the rasterized output.
- **v1.24.0** (15 July 2026) -- Page-spread support, section-grouped summary
    ordering, and OpenAlex quota hardening. The summary schema
    (`Summary_2_2_0`) gains `is_two_page_spread` and
    `page_number_integer_end`; two facing pages scanned as one image are now
    detected, transcribed left page first with both page numbers tagged, and
    rendered as a page range ("Pages 13-14" / "Pages xii-xiii"), with
    citations and Document Structure entries attributed to both pages. Page
    numbering is spread-aware end to end: anchors, offsets, and gap inference
    operate on document-wide virtual page positions (a spread occupies two
    slots). The final summary now groups each section's pages together
    (sections ordered by median document position, scan order preserved
    within a section), and the Document Structure section renders
    Roman-numbered front matter as Roman numerals ("pp. iii-xii, 100-105")
    instead of colliding with Arabic page numbers. OpenAlex daily-quota
    exhaustion (429 with a long `retryAfter`) now latches enrichment off
    process-wide and across runs via `openalex_budget.json` in the state
    directory, remaining citations are still served from the persistent
    cache instead of being skipped, short rate-limit waits (<= 30 s) are
    slept through and retried, and the standard `Retry-After` header is
    honored as a fallback.
- **v1.23.0** (12 July 2026) -- Bug-fix release closing five defects found in
    an automated audit. The per-key pool cap is now enforced for the summary
    role on the fresh-page path: when the summary stamp resolves to a
    different accounting bucket than the transcription stamp, the page must
    be admitted by both buckets, and the budget wait polls both before
    resuming. Working directories of incomplete items (budget-deferred or
    failed pages) are retained for resume even under `--force`/`--overwrite`,
    as the retention notice promises. Image folders are ordered by natural
    sort, so unpadded filenames (`page_2` before `page_10`) no longer
    scramble transcription and summary order. Shared-ledger seeding after a
    degraded start pushes session usage through the additive sync path
    instead of max-merging it away. `retry.max_attempts` is now honored as
    the total attempt count, as documented. Regression tests cover all five
    fixes.
- **v1.22.0** (12 July 2026) -- Per-key token accounting and definable daily
    pools. The shared cross-tool ledger moves to schema v2: every API call
    is stamped with its provider, the NAME of the env var that served it
    (key values are never stored), and a pool label, recorded per
    (tool, provider, key env, pool) alongside the per-tool totals. Budget
    enforcement becomes two-tier: per-(key, pool) daily caps as the primary
    gate -- pools definable per provider in
    `daily_token_limit.per_key_pool_caps` (bare-int cap or `{cap, models}`
    mapping), with built-in defaults mirroring OpenAI's complimentary daily
    token program -- and the combined `daily_tokens` cap as a secondary
    guard, scoped by the new `scope` knob (`pooled` default: calls whose
    model belongs to no pool are counted but never blocked). The
    transcription and summary roles stamp independent buckets, so an
    exhausted pooled summary key never blocks a pool-less transcription
    endpoint in the same run. The wait loop names the exhausted key,
    reports other keys' remaining pools, live-reloads pool settings, and
    picks up an `api_keys.yaml` remap at the next item. v1 ledgers are
    adopted in place with un-attributable usage kept under an
    "unattributed" row. Vendored `shared_ledger.py` v2.1.0.

- **v1.21.0** (9 July 2026) -- Register current LLM models in the capability
    registry (`llm/capabilities.py`). OpenAI: the GPT-5.6 family
    (`gpt-5.6-sol`/`-terra`/`-luna` plus the bare `gpt-5.6` alias resolving to
    the `sol` flagship) and GPT-5.5 (`gpt-5.5`, `gpt-5.5-pro`) -- vision,
    1.05M context, 128k output, reasoning effort, and text verbosity. Anthropic:
    the Claude 5 generation (`claude-fable-5`, `claude-sonnet-5`,
    `claude-opus-4-8`) -- adaptive thinking, 1M context, 128k output, high-res
    vision, with `top_p` gated off to mirror the existing opus-4.5/sonnet-4.5
    handling. Google: `gemini-3.5-flash` (GA) -- vision/PDF, media resolution,
    1,048,576 context, 65,536 output, thinking levels. Matching uses
    longest-prefix-first ordering; deprecated-model entries are retained. All
    1,352 tests pass.

- **v1.20.0** (7 July 2026) -- Adopt shared token ledger 1.2.0 and fix the
    rate limiter's adaptive backoff. The vendored `llm/shared_ledger.py` is
    re-copied from ChronoMiner: `_merge` now coerces a non-numeric stored
    tool value to 0 and catches `ValueError`/`TypeError` alongside `OSError`,
    so a hand-edited or corrupt ledger degrades to standalone mode instead of
    crashing the call path (never-crash contract). The rate limiter's error
    backoff now imposes a real, bounded admission delay after 429s: the
    multiplier previously scaled a zero wait when no window was saturated (a
    silent no-op), and the new penalty is a deadline from wait start rather
    than a perpetual floor, so admission always resumes. All 1,352 tests
    pass.

- **v1.19.0** (6 July 2026) -- Multi-agent bug-hunt sweep fixing twenty
    verified defects across all packages. Highest impact: on the Anthropic and
    OpenRouter structured-output paths, invoke kwargs (`max_tokens`,
    `temperature`, extended-thinking config) were silently dropped by the
    LangChain runnable chain and are now bound onto the model before
    `with_structured_output`; the resume map is keyed by item path instead of
    output stem, so same-named items in different directories no longer
    clobber each other's resume state; runs that logged failed transcription
    or summary pages are no longer classified COMPLETE on rerun, making
    reported failures repairable; and `FolderPayloadSource` no longer encodes
    a closed PIL image when preprocessing returns the input unchanged. Also
    fixed: the directory scanner no longer silently drops PDFs and image
    folders nested under an image-bearing directory (suppressed folders now
    warn); the summary-only resume path is gated by the daily token budget;
    a no-match `--select` exits 1 and the ambiguous multiple-items guard
    emits the `--json` summary before exiting; items never attempted after a
    cancelled budget wait are reported as skipped, not failed; markdown-fenced
    JSON no longer bypasses the transcription content-flag retries;
    schema-validation exhaustion is labeled `schema_validation` instead of
    `api_failure` and no longer double-counts stats; the schema-retry backoff
    uses the documented additive jitter; request counters are lock-guarded;
    the token-ledger seed no longer discards usage committed during ledger
    I/O; `PdfPayloadSource.close()` takes the render lock; P-mode transparent
    images flatten to white; DOI URLs with parentheses render as valid
    markdown links; inline math straddling a display block no longer
    duplicates it in DOCX output; and `citation.max_api_requests: 0` means
    zero requests instead of unlimited.

- **v1.18.0** (6 July 2026) -- Fail items with budget-deferred pages truthfully
    and withhold masking outputs. When the daily token budget defers pages
    mid-item (a stalled reset, a cancelled budget wait, or shared-ledger
    exhaustion), `process_item` now compares the source page count against the
    completed results and fails the item on any shortfall, so the run exits 1
    and the `--json` summary reports an honest `items_failed` instead of
    exit 0 with a silently truncated document. On such a shortfall the partial
    `.txt` and summary `.docx`/`.md` outputs are withheld entirely -- previously
    a truncated final text was written whose header self-reported as complete
    -- while the completed pages stay finalized in the working-log JSONL, so a
    later resume run reconstructs them and finishes only the missing pages.
    Regression tests cover the withheld-output partial run, the main-level
    exit-code and JSON contract, and the full partial-then-resume round trip.

- **v1.17.0** (5 July 2026) -- Fix the daily token budget's reset boundary.
    Both the private per-tool tracker (`llm/token_tracker.py`) and the vendored
    shared cross-tool ledger (`llm/shared_ledger.py`, bumped to 1.1.0) now roll
    the budget day over at 00:01 UTC -- one minute after OpenAI's 00:00 UTC
    free-tier reset -- instead of local midnight, so the tool never frees its
    budget before OpenAI's own counter has actually reset. The one-minute
    buffer is a deliberate safety margin against clock skew. `get_reset_time()`
    now returns a timezone-aware UTC datetime; user-facing wait messages show
    the local wall-clock time alongside an explicit "(00:01 UTC)" anchor for
    clarity. Updated docs and config comments accordingly. All tests pass.

- **v1.16.0** (3 July 2026) -- Honest run status and CLI-contract fixes
    from a live cross-provider bug hunt. Propagate page-level failures to
    the item status, the `--json` summary, and the exit code: an item with
    any failed transcription or summary page (or zero extracted pages) now
    counts as failed and the run exits 1 instead of reporting success.
    Emit the `--json` run summary on every exit path, including all-skipped
    resume runs and summary-only resume (previously such runs printed
    nothing and exited 0 silently). Populate the `outputs` field of the
    JSON summary with the absolute paths of all files actually written
    (txt, docx, md). Fix OpenRouter structured output by injecting the
    required top-level `title` into the JSON schema (previously every
    OpenRouter page failed with "Unsupported function"). Extend the CLI
    `--reasoning-effort` choices to the full supported set (`none`,
    `minimal`, `low`, `medium`, `high`, `xhigh`).

- **v1.15.0** (3 July 2026) -- Optional shared cross-tool token budget.
    Add the vendored `llm/shared_ledger.py` (locked delta merges into
    per-tool fields, atomic per-process temp writes, local-midnight
    rollover, degrade-to-standalone) and wire it into the daily token
    tracker behind the new opt-in `shared_token_budget` config block: when
    enabled, `daily_token_limit.daily_tokens` is enforced against the
    COMBINED usage of AutoExcerpter, ChronoMiner, and ChronoTranscriber via
    one ledger at `~/.chronopipeline/token_ledger.json`, with seed-once
    adoption of legacy same-day counts, delta syncs riding the debounced
    save, forced refreshes near the cap and in both wait loops, per-tool
    breakdown in token summaries, and `combined_tokens_today` in the
    `--json` run summary. Default behavior (feature off) is unchanged.
    Verified live: concurrent tools share one limit with zero lost updates.

- **v1.14.1** (3 July 2026) -- Await async SDK client closers during manager
    teardown: `_close_client_obj` now runs coroutine-returning `close()`
    methods on a fresh event loop (or discards them cleanly when a loop is
    active) instead of leaving an un-awaited coroutine warning behind.
    Surfaced by the live post-release validation run.

- **v1.14.0** (3 July 2026) -- Concurrency and token-budget hardening.
    Replace the eager page fan-out with one executor per item, bounded
    submission, completion-order processing, and prompt cancellation of
    queued pages on interrupt (`cancel_futures`); fix the lost-update race
    on the progress counter; make the budget reservation EWMA per page
    (transcription + summary observed together) so the admission gate
    matches the documented contract; capture prompt-cache tokens
    defensively and commit them at full weight across raw-Anthropic,
    LangChain-normalized, and OpenAI shapes; honor HTTP `Retry-After` with
    a 120 s backoff cap and lower the default retry budget from 15 to 8
    attempts; re-read `daily_token_limit.daily_tokens` during both
    wait-at-limit paths so a mid-wait config edit lifts the cap without
    restart; unify atomic state writes on a per-process-unique temp-file
    helper shared with the OpenAlex cache; close manager clients
    deterministically per item; remove the dead `delay_between_tasks`,
    `image_processing`, and summary-concurrency config surface.

- **v1.13.0** (2 July 2026) -- Hardening release closing the resume and
    citation defects found in a full production audit. Page-level resume now
    snapshots both working logs and reuses or regenerates each completed
    page's summary, so a crashed run no longer yields outputs covering only
    post-crash pages; summary-only resume reuses logged transcriptions with
    zero transcription API calls, warns on model mismatch (recorded in the
    output metadata), and `--retranscribe` forces a fresh pass. The
    token-budget path now waits when the remaining budget cannot fit a page,
    and an item is never marked COMPLETE while the log covers fewer pages
    than the document; working logs move to true JSONL with a format-version
    marker, crash-truncated tail lines are dropped on parse, and unversioned
    logs are refused for resume. Citations are overhauled: Unicode folding
    and structured keys (author, year, volume, title) end diacritic and
    author-form duplicates while different volumes and years never merge; a
    conservative consolidation pass (thresholds configurable under
    `citation:`) fuzzy-merges within author-year blocks and logs every
    merge; enrichment runs once per item through a single CitationManager
    shared by both writers, links require title overlap plus a year-or-author
    cross-check, every OpenAlex request counts against `max_api_requests`, a
    persistent cache in the state dir is reused across items and runs,
    citations on unnumbered pages are kept, and the bibliography sorts
    stably by folded author, year, and title. Temperature is wired
    (capability-guarded), Anthropic output tokens default from the
    capability registry, one shared rate limiter per provider spans
    transcription and summary, and the dead summary-concurrency keys are
    removed. The CLI adopts the agent contract: exit codes 0/1/2/130, a
    `--json` run summary, `--dry-run`, `--cli`/`--interactive` overrides, a
    non-TTY guard, an explicit error for multiple items without
    `--all`/`--select`, and reporting of unmatched `--select` parts. Token
    state and the citation cache live under `~/.autoexcerpter` (configurable
    via `paths.state_dir`) with one-time legacy adoption; blank-page
    sentinels match the transcription layer's actual markers; the final
    `.txt` keeps document order for error pages; the scanner no longer emits
    nested duplicates of image folders; README output names are corrected.

- **v1.12.0** (28 June 2026) -- Ship scrubbed `*.example.yaml` config templates
    with conservative defaults so a fresh clone runs with clear guidance instead
    of silently using a stale personal path. Each real config (`app.yaml`,
    `model.yaml`, `concurrency.yaml`, `image_processing.yaml`, `api_keys.yaml`)
    now has a tracked sibling with conservative defaults (gpt-5.4-mini at medium
    effort, low concurrency limits, daily token limit disabled). The loaders in
    `config/loader.py` and `config/app.py` fall back to the example when the real
    file is absent and log one INFO line naming the example file. The gitignore
    rule is consolidated from five per-file lines to a single pattern
    (`config/defaults/*.yaml` with `!config/defaults/*.example.yaml`). The
    hardcoded personal fallback path for `INPUT_FOLDER_PATH`/`OUTPUT_FOLDER_PATH`
    is scrubbed to `''`, and the "no items found" message now names the config
    file and suggests the copy command.

- **v1.11.0** (28 June 2026) -- Added an optional `api_keys.yaml` config that maps
    each LLM provider to the NAME of the environment variable holding its API key,
    so a key can be swapped between runs (for example `openai: OPENAI_API_KEY_2`) by
    editing one file rather than the environment. The file is fully backward
    compatible: a missing file, a missing provider entry, or an empty value falls
    back to the default env-var name, so existing setups are unaffected. The loader
    exposes the mapping via `get_api_keys_config()`, and both key resolution and the
    provider-availability probes honor it. All 1246 tests pass.

- **v1.10.0** (24 June 2026) -- The daily token limit is now enforced at the
    page level, not just between files. When the limit is enabled, each page
    reserves a combined transcription + summary estimate (a self-calibrating
    rolling average) before any API call, so concurrent worker threads cannot
    collectively overshoot; once the budget is exhausted mid-file the run drains
    in-flight pages, waits for the daily reset, and re-runs the still-pending
    pages from the log. A page is reserved and skipped atomically, so resume
    never sees a transcribed-but-unsummarized page. Configured concurrency is
    unchanged when budget is plentiful. Two optional `daily_token_limit`
    settings tune the estimate (`chunk_estimate_seed`, `estimate_smoothing`).
    All 1241 tests pass.

- **v1.9.0** (21 June 2026) -- Refreshed the transitive google-genai dependency
    to the 2.x major (`google-genai` 1.73.1 -> 2.9.0), pulled via
    `langchain-google-genai` (unchanged). AutoExcerpter uses the Google path only
    through the LangChain wrapper; the wrapper imports clean under 2.x and all
    1,232 tests pass.

- **v1.8.0** (21 June 2026) -- Adopted mypy 2.x for static type checking.
    Relaxed the dev pin from `mypy>=1.20,<2` to `mypy>=2.1` and refreshed the
    lockfile. The full source tree type-checks clean under mypy 2.1.0 and all
    1,232 tests pass; no runtime code changed.

- **v1.7.0** (20 June 2026) -- Removed two pieces of confirmed dead code: the
    unused format_string parameter of setup_logger in `config/logger.py` and the
    unused model_page_number parameter of the private `_build_summary_result`
    method in `pipeline/transcriber.py`, along with its sole call-site argument.
    Consolidated two within-module duplications: the repeated OMML
    namespace-ensuring logic in `rendering/docx.py` was extracted into a private
    `_ensure_omml_namespace` helper called from both branches of
    add_math_to_paragraph, and the byte-identical `_ANTHROPIC_BASE` and
    `_OPENROUTER_BASE` capability dicts in `llm/capabilities.py` now build through
    a private `_non_openai_base` factory. All resulting values are byte-identical
    to the originals; runtime behavior is unchanged.

- **v1.6.0** (20 June 2026) -- Refreshed dependencies under the conservative,
    majors-gated policy. Upgraded the LangChain stack within its current majors
    (`langchain-core` 1.4.0 -> 1.4.8, `langchain-openai` 1.2.1 -> 1.3.2,
    `langchain-anthropic` 1.4.1 -> 1.4.6, `langchain-google-genai` 4.2.2 ->
    4.2.5) along with `requests` (2.33.1 -> 2.34.2), `tqdm` (4.67.3 -> 4.68.3),
    and the dev tools `pytest` (9.0.3 -> 9.1.1), `ruff` (0.15.12 -> 0.15.18),
    `types-PyYAML`, and `types-requests`. Raised the >= floors for `tqdm`,
    `langchain-core`, `langchain-openai`, `requests`, and `pytest` accordingly.
    Held the mypy major at 1.20.2 (2.1.0 would be a major jump) and pinned it to
    <2 so the resolver cannot advance it; the transitive `google-genai` likewise
    stays on its 1.x line. No dependencies were removed (the confirmed-unused set
    was empty) and no missing runtime dependency was added (the lone guarded
    colorama import has a graceful fallback and is not a hard requirement).
    Regenerated uv.lock and synced the environment; all runtime imports and the
    dev toolchain verified clean.

- **v1.5.0** (10 June 2026) -- Streaming in-memory image pipeline: PDF pages are
    rendered, preprocessed, and base64-encoded on demand inside the transcription
    workers (lock-guarded PyMuPDF access) instead of being extracted up front into
    `_working_files/images/`; no temporary image files are written anymore.
    Page-level resume now filters BEFORE rendering, so resuming a mostly-done PDF
    no longer re-renders every page. New `imaging/payload.py` with `PagePayload`,
    `PdfPayloadSource`, and `FolderPayloadSource`;
    `TranscriptionManager.transcribe_image(path)` replaced by
    `transcribe_payload(payload)`. Full reproducibility records: per-page
    `image_provenance` (SHA-256/dimensions/byte size/effective DPI of the exact
    JPEG sent) and a `file_provenance` log-header block (source SHA-256,
    PyMuPDF/Pillow versions, image-config snapshot). Behavior change: pages that
    fail to render are now recorded as `[preprocessing error: ...]` pages in the
    outputs instead of being silently dropped. Removed dead disk-image machinery
    (`extract_pdf_pages_to_images`, file-based `ImageProcessor` instance API,
    `MAX_EXTRACTION_WORKERS`). Bug fix: on page-level resume, previously completed
    pages were lost from the final `.txt` because the transcription log was
    truncated before prior results were re-read; prior results are now snapshotted
    before log reinitialization and re-appended to the fresh log.

- **v1.4.0** (31 May 2026) -- Low-risk code-review cleanups (behavior-preserving):
    route `config.accessors` fallbacks through `constants` (`DEFAULT_RATE_LIMITS`,
    `DEFAULT_TARGET_DPI`, `DEFAULT_OPENAI_TIMEOUT`); add a `MAX_ERROR_MULTIPLIER`
    constant and reference it in `RateLimiter`; reuse `OPENAI_MODEL_PREFIXES` in
    `imaging._provider` model detection; collapse the duplicate per-directory file
    scan in `pipeline.scanner` into a single pass; log instead of silently
    swallowing the `output_text` extractor fallback in `llm.base`; document the
    `RateLimiter.get_stats` per-poll reset side effect.

- **v1.3.0** (29 May 2026) -- Fix spurious mid-paragraph line breaks: disable
    `text_cleaning.line_wrapping` by default (it re-wrapped already laid-out LLM
    output into short orphan lines) and enable a conservative `merge_hyphenation`
    with a compound-preserving guard (`should_keep_hyphen`); warn when wrapping is
    enabled. Add `scripts/repair_layout`, a deterministic line-break repairer with
    a content-preservation verifier for fixing affected transcription files in
    place.

- **v1.2.1** (19 May 2026) -- Dependency refresh from environment-wide CVE audit:
    bump `langchain-core` 1.3.2 -> 1.4.0 (RCE on deserialization); `langsmith`
    0.7.36 -> 0.8.5 (unsafe deserialization; full fix to 1.0.x deferred pending
    upstream constraint relaxation); `urllib3` 2.6.3 -> 2.7.0.

- **v1.2.0** (10 May 2026) -- Apply ruff linting and formatting across all source
    and test files; add `ruff>=0.15` to dev dependencies; resolve 404 lint
    violations.

- **v1.1.2** (10 May 2026) -- Add ruff linter and formatter configuration;
    gitignore `CLAUDE.md`.

- **v1.1.1** (10 May 2026) -- Resolve all pre-existing mypy strict-mode errors
    across the codebase.

- **v1.1.0** (10 May 2026) -- Migrate build system to `pyproject.toml` and `uv`;
    remove `requirements.txt`, `mypy.ini`, and `pytest.ini`.

- **v1.0.0** (25 April 2026) -- Initial public release; squashed baseline.

## Contributing

Contributions are welcome. Fork the repository, create a feature
branch, and submit a pull request with a clear description.

**Guidelines:** PEP 8 enforced via `ruff check` and
`ruff format`; run both before submitting. Type annotations
required; unit tests for new functionality; clear separation
of concerns; use the logger module for debug output.

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
