# AutoExcerpter

AutoExcerpter is an intelligent document processing pipeline that automatically transcribes and summarizes PDF documents and image collections. Built on LangChain for multi-provider LLM support, it converts scanned documents into searchable, structured text with optional summaries and enriched bibliographic citations. The tool supports OpenAI (GPT-5.1, GPT-5, GPT-4, o-series), Anthropic (Claude 4.5, Claude 4), Google (Gemini 3, Gemini 2.5), and OpenRouter for accessing additional models. It is designed for researchers, academics, and professionals who need to digitize and analyze large volumes of documents efficiently.

## Table of Contents

-   [Overview](#overview)
-   [Key Features](#key-features)
-   [Supported Models](#supported-models)
-   [How It Works](#how-it-works)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Quick Start](#quick-start)
-   [CLI Mode and Interactive Mode](#cli-mode-and-interactive-mode)
-   [Configuration](#configuration)
    -   [Basic Configuration](#basic-configuration)
    -   [Provider Configuration](#provider-configuration)
    -   [Model Configuration](#model-configuration)
    -   [Concurrency Configuration](#concurrency-configuration)
    -   [Image Processing Configuration](#image-processing-configuration)
    -   [Text Cleaning Configuration](#text-cleaning-configuration)
    -   [Summary Context Configuration](#summary-context-configuration)
    -   [Citation Management](#citation-management)
    -   [Daily Token Limit Tracking](#daily-token-limit-tracking)
-   [Usage](#usage)
-   [Output Files](#output-files)
-   [Project Structure](#project-structure)
-   [Advanced Topics](#advanced-topics)
-   [Troubleshooting](#troubleshooting)
-   [Contributing](#contributing)
-   [License](#license)

## Overview

AutoExcerpter processes documents through a sophisticated two-stage pipeline that leverages vision-enabled LLMs with optical character recognition (OCR) capabilities. Built on LangChain, the system supports multiple AI providers including OpenAI, Anthropic, Google, and OpenRouter, allowing you to choose the best model for your needs. In the first stage, each page is transcribed using structured JSON schemas to ensure consistent output. In the second optional stage, transcribed text is analyzed to generate structured summaries with automatically deduplicated citations enriched with metadata from the OpenAlex academic database.

**Primary Use Cases:**

-   **Academic Research**: Digitize scanned academic papers, extract citations, and generate structured literature reviews
-   **Document Digitization**: Convert image-based archival documents into searchable, machine-readable text
-   **Literature Analysis**: Automatically summarize research papers with consolidated bibliographies
-   **Knowledge Management**: Build searchable databases from historical documents and scanned materials
-   **Content Extraction**: Process multi-column layouts, preserve mathematical equations, and capture visual content descriptions

## Key Features

**Advanced Transcription:**

-   **Multi-Format Support**: Processes PDFs (with automatic page extraction) and direct image folders
-   **Format Compatibility**: Supports PNG, JPG, JPEG, TIFF, BMP, GIF, and WEBP image formats
-   **Structure Preservation**: Maintains document structure including headers, footers, page numbers, footnotes, tables, and multi-column layouts
-   **Mathematical Notation**: Converts mathematical equations to LaTeX format for accurate representation
-   **Visual Content**: Provides detailed descriptions of images, diagrams, charts, and figures
-   **Schema-Driven Output**: Uses strict JSON schemas to ensure consistent, parseable transcription results
-   **In-Memory Processing**: Optimized image preprocessing pipeline that eliminates disk I/O bottlenecks and prevents race conditions
-   **Intelligent Formatting**: Preserves markdown formatting (headings, bold, italic) and line breaks
-   **Configurable Text Cleaning**: Runs Unicode normalization, LaTeX repairs, optional hyphenation merging, whitespace cleanup, and line wrapping before summarization or export

**Intelligent Summarization:**

-   **Concise Extraction**: Generates bullet-point summaries highlighting main ideas and key findings for each page
-   **Smart Filtering**: Automatically identifies and excludes pages without semantic content (title pages, blank pages, reference lists)
-   **Page Tracking**: Accurately tracks page numbers from document headers and footers
-   **Dual Output Formats**: Exports summaries as both formatted DOCX documents and Markdown files for maximum flexibility
-   **Markdown Output**: Version-control friendly `.md` files compatible with Writage and other markdown-to-Word tools
-   **Focused Summarization**: Hierarchical context system allows file-specific, folder-specific, or general context to guide the model to pay special attention to specific topics during summarization

**Enhanced Citation Management:**

-   **Automatic Deduplication**: Uses normalized text hashing to identify and merge duplicate citations across the entire document
-   **Page Range Tracking**: Displays comprehensive page numbers where each citation appears (e.g., "pp. 5, 12-15, 23")
-   **Metadata Enrichment**: Integrates with OpenAlex API to enrich citations with DOI, publication year, authors, and venue information
-   **Clickable Hyperlinks**: Adds direct hyperlinks to citations for instant access to extended metadata
-   **Consolidated Bibliography**: Presents all citations in a dedicated section at the end of summary documents
-   **Smart Matching**: Uses both text similarity and DOI extraction for accurate citation identification

**Performance and Reliability:**

-   **Concurrent Processing**: Configurable parallelism for both image preprocessing and API requests
-   **Adaptive Rate Limiting**: Sliding window rate limiter prevents API quota violations
-   **LangChain Built-in Retry**: Automatic exponential backoff with jitter handled by LangChain for API errors
-   **Schema-Specific Retries**: Optional retries based on model-returned content flags (no_transcribable_text, etc.)
-   **Daily Token Budgeting**: Built-in token tracker enforces configurable daily limits with automatic midnight resets and per-request accounting
-   **Service Tier Support**: Full support for OpenAI Flex tier to reduce processing costs by up to 50%
-   **Progress Tracking**: Real-time progress bars with estimated time of completion
-   **Comprehensive Logging**: Detailed JSON logs for debugging, quality assurance, and audit trails
-   **Automatic Cleanup**: Optional deletion of temporary working directories after successful processing

**Multi-Provider Architecture:**

-   **LangChain Integration**: Unified interface for multiple LLM providers via LangChain
-   **Provider Flexibility**: Switch between OpenAI, Anthropic, Google, or OpenRouter with configuration changes
-   **Capability Guarding**: Automatic parameter filtering based on model capabilities (prevents unsupported parameter errors)
-   **Model Auto-Detection**: Automatic provider inference from model names (gpt-5 to OpenAI, claude to Anthropic, etc.)

**Architecture Excellence:**

-   **Modular Design**: Clear separation of concerns with well-defined component responsibilities
-   **YAML-Based Configuration**: Human-readable configuration with validation and sensible defaults
-   **Base Classes**: Shared API logic through inheritance to eliminate code duplication
-   **Testable Components**: Well-defined interfaces that facilitate unit testing
-   **Type Safety**: Comprehensive type hints throughout the codebase
-   **Public API**: Clear module exports via `__all__` declarations

## Supported Models

AutoExcerpter supports a wide range of models from multiple providers. The system automatically detects model capabilities and filters parameters accordingly.

**OpenAI Models:**

Model Family

Models

Capabilities

GPT-5.1 (Nov 2025)

gpt-5.1, gpt-5.1-instant, gpt-5.1-thinking

Reasoning, text verbosity, multimodal

GPT-5 (Aug 2025)

gpt-5, gpt-5-mini, gpt-5-nano

Reasoning, text verbosity, multimodal

O-series

o4, o4-mini, o3, o3-mini, o1, o1-mini

Reasoning (no temperature control)

GPT-4.1

gpt-4.1, gpt-4.1-mini, gpt-4.1-nano

Multimodal

GPT-4o

gpt-4o, gpt-4o-mini

Multimodal

**Anthropic Claude Models:**

Model Family

Models

Capabilities

Claude 4.5 (Oct-Nov 2025)

claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5

Multimodal, extended thinking (opus/sonnet)

Claude 4

claude-opus-4, claude-sonnet-4

Multimodal, extended thinking

Claude 3.x

claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku

Multimodal

**Google Gemini Models:**

Model Family

Models

Capabilities

Gemini 3 (Nov 2025)

gemini-3-pro

Thinking, multimodal

Gemini 2.5 (Mar 2025)

gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite

Thinking, multimodal

Gemini 2.0

gemini-2.0-flash, gemini-2.0-flash-lite

Multimodal

Gemini 1.5

gemini-1.5-pro, gemini-1.5-flash

Multimodal

**OpenRouter:**

OpenRouter provides access to models from all providers through a unified API. Use the `openrouter` provider with any supported model name.

## How It Works

AutoExcerpter follows a systematic workflow to transform documents into structured, searchable content:

**1. Input Selection and Scanning**

The application scans your configured input directory for processable items (PDF files and image folders). It presents an interactive menu where you can select specific items or process all available documents in batch.

**2. Page Extraction (PDF Only)**

For PDF inputs, each page is extracted as a high-resolution image using configurable DPI settings (default: 300 DPI). The extraction process applies optimizations including grayscale conversion, transparency handling, and format normalization. For image folder inputs, existing images are processed directly.

**3. Image Preprocessing**

Images undergo in-memory preprocessing to optimize OCR accuracy:

-   Grayscale conversion to reduce noise and improve text recognition
-   Transparency handling to flatten alpha channels onto white backgrounds
-   Intelligent resizing based on detail level (maintains aspect ratio)
-   JPEG compression optimization for API transmission
-   All processing happens in memory to prevent race conditions and improve performance

**4. Transcription via LangChain**

Each preprocessed image is sent to the configured LLM provider through LangChain with:

-   A detailed system prompt instructing the model to perform verbatim transcription
-   Structured JSON schema defining the expected output format
-   Configuration for reasoning effort and text verbosity (from model.yaml)
-   Model parameters including max_output_tokens and service tier settings
-   Automatic capability guarding to filter unsupported parameters based on model

The API returns structured JSON containing:

-   Full verbatim transcription with markdown formatting
-   Preserved structural elements (headers, footers, tables, footnotes)
-   Mathematical equations in LaTeX notation
-   Page numbers marked with special XML-style tags
-   Detailed descriptions of visual elements (images, diagrams, charts)

LangChain handles API retries with exponential backoff automatically. Schema-specific retries are available for content flags.

**5. Summarization (Optional)**

When enabled, transcribed text is processed by the OpenAI API again with a summarization prompt that extracts:

-   Concise bullet-point summaries of main ideas
-   Full bibliographic citations in APA format
-   Page number metadata for accurate referencing
-   Flags indicating pages without semantic content

**6. Citation Processing**

The citation manager processes extracted citations through:

-   Normalization of citation text for duplicate detection
-   Deduplication using text similarity hashing
-   Page range consolidation (merges consecutive and discontinuous page references)
-   OpenAlex API enrichment for DOI, authors, publication year, and venue
-   Hyperlink generation for citations with sufficient metadata

**7. Output Generation**

The pipeline produces multiple output files:

-   Plain text file with complete transcriptions including metadata headers
-   Formatted DOCX file with structured summaries and consolidated citations
-   JSON log files with detailed processing metadata, timing, and error information

**8. Cleanup**

If configured, the system automatically deletes temporary working directories including extracted images and intermediate processing files, keeping only the final outputs.

## Prerequisites

Before installing AutoExcerpter, ensure you have:

-   **Python 3.10 or higher**: Required for LangChain v1.0 compatibility
-   **API Key(s)**: At least one API key from a supported provider:
    -   OpenAI API key (for GPT-5, GPT-4, o-series models)
    -   Anthropic API key (for Claude models)
    -   Google API key (for Gemini models)
    -   OpenRouter API key (for accessing multiple providers)

## Installation

1.  **Clone the repository:**

```bash
git clone https://github.com/yourusername/AutoExcerpter.git
cd AutoExcerpter
```

2.  **Create and activate a virtual environment:**

```bash
python -m venv .venv

# On Windows:
.venvScriptsactivate

# On macOS/Linux:
source .venv/bin/activate
```

3.  **Install dependencies:**

```bash
pip install -r requirements.txt
```

For development and testing, install the dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
python -m pytest
```

4.  **Set up API keys as environment variables:**

```bash
# OpenAI (required for default configuration)
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic (optional, for Claude models)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google (optional, for Gemini models)
export GOOGLE_API_KEY="your-google-api-key"

# OpenRouter (optional, for multi-provider access)
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

On Windows, use `set` instead of `export`, or configure environment variables through System Properties.

5.  **Configure the application:**

Edit `modules/config/app.yaml` to set your input/output paths:

```yaml
input_folder_path: 'C:UsersyournameDocumentsPDFs'
output_folder_path: 'C:UsersyournameDocumentsOutput'
input_paths_is_output_path: true  # Write outputs next to each input file
```

Edit `modules/config/model.yaml` to set your preferred models:

```yaml
transcription_model:
  name: "gpt-5-mini"  # or any supported model
  provider: "openai"

summary_model:
  name: "gpt-5-mini"
  provider: "openai"
```

## Quick Start

1.  **Run AutoExcerpter:**

```bash
python main.py
```

2.  **Select documents to process** from the interactive menu
    
3.  **Wait for processing** to complete (progress bars show estimated time)
    
4.  **Find outputs** in your configured output directory:
    
    -   `.txt` files contain full transcriptions
    -   `.docx` files contain formatted summaries (if enabled)
    -   Enable `input_paths_is_output_path` to drop each item's outputs next to the original input when running in interactive mode

## CLI Mode and Interactive Mode

AutoExcerpter supports two execution modes: **Interactive Mode** (default) for user-friendly document processing, and **CLI Mode** for automation and scripting.

### Execution Modes

#### Interactive Mode (Default)

Interactive mode provides a guided experience with structured prompts and visual feedback.

**Enable Interactive Mode** in `modules/config/app.yaml`:

```yaml
cli_mode: false
```

**Run the application:**

```bash
python main.py
```

**Highlights:**

-   Styled console output with headers, sections, and status indicators
-   Guided item selection for PDFs and image folders discovered under the input path
-   Exit options available at every prompt (`exit`, `quit`, or `q`)
-   Daily token-limit wait screen can be cancelled instantly by typing `q` and pressing Enter
-   Flexible selection syntax supporting single indices (`1`), multiple selections (`1;3;5`), ranges (`1-5`), `all`, or filename search
-   **Filename search**: Enter a filename or partial text to find matching items (case-insensitive)
-   Immediate confirmation and error feedback for each action
-   Inline progress updates for every document processed

**Selection Examples:**

```
Select items to process: 42                    # Select item #42
Select items to process: 1,5,10               # Select items #1, #5, and #10
Select items to process: 1-10                 # Select items #1 through #10
Select items to process: all                  # Select all items
Select items to process: Mennell              # Find items containing "Mennell"
Select items to process: food history         # Find items containing "food history"
```

The full interactive interface is documented in `docs/USER_PROMPTS_AND_LOGGING.md`.

#### CLI Mode (Automation-Friendly)

CLI mode is optimized for batch processing, automation, and integration into pipelines without interactive prompts.

**Enable CLI Mode** in `modules/config/app.yaml`:

```yaml
cli_mode: true
```

**Command syntax:**

```bash
python main.py <input> <output> [--all] [--select PATTERN] [--context TOPICS]
```

**Arguments:**

-   `input` (required): Path to a PDF file, an image folder, or a directory containing multiple items
-   `output` (required): Destination directory for generated transcriptions and summaries
-   `--all` (optional): Process every item discovered under the input directory
-   `--select PATTERN` (optional): Select items by number, range, or filename pattern. Supports:
    -   Single numbers: `--select 5`
    -   Comma-separated: `--select "1,3,5"`
    -   Ranges: `--select "1-10"`
    -   Filename search: `--select "Mennell"` (case-insensitive partial match)
-   `--context TOPICS` (optional): Specify topics for focused summarization. The model will pay special attention to content related to these topics and summarize them in greater detail. Example: `--context "Food History, Wages, Early Modern"`

When neither `--all` nor `--select` is specified, only the first item is processed if multiple items exist.

**Usage examples:**

```bash
# Process a single PDF using relative paths
python main.py "./documents/paper.pdf" "./output"

# Process every item found in a directory
python main.py "./documents" "./output" --all

# Process specific items by number
python main.py "./documents" "./output" --select "1,5,10"

# Process a range of items
python main.py "./documents" "./output" --select "1-20"

# Process items matching a filename pattern
python main.py "./documents" "./output" --select "Mennell"

# Process items containing specific text (case-insensitive)
python main.py "./documents" "./output" --select "food history"

# Process with focused summarization on specific topics
python main.py "./documents" "./output" --all --context "Food History, Wages, Early Modern"

# Process with absolute paths on Windows
python main.py "C:Documentspaper.pdf" "C:Output"

# Automate multiple PDFs in a shell script
for pdf in ./papers/*.pdf; do
    python main.py "$pdf" "./output"
done
```

**Highlights:**

-   No interactive prompts; suitable for cron jobs and CI/CD workflows
-   Console output limited to structured logging
-   Respects all configuration values defined in YAML files (models, concurrency, retries, cleanup)
-   Allows absolute or relative input/output paths and resolves them before processing

Additional CLI guidance, including scripting patterns and CI/CD samples, is available in `docs/CLI_MODE.md`.

## Configuration

AutoExcerpter uses a multi-file YAML configuration system that provides fine-grained control over every aspect of processing. All configuration files are located in `modules/config/` and include cross-references to related files for easy navigation.

File

Purpose

`app.yaml`

Application settings, file paths, feature toggles, daily limits

`model.yaml`

LLM provider and model settings (transcription and summary)

`concurrency.yaml`

API rate limits, retries, and parallelism settings

`image_processing.yaml`

Image preprocessing and text cleaning options

### Basic Configuration

**File**: `modules/config/app.yaml`

This is the primary configuration file for application-level settings. Model configuration is in `model.yaml`, and API concurrency/rate limits are in `concurrency.yaml`.

```yaml
# Execution Mode
cli_mode: false  # true = CLI with arguments, false = interactive prompts

# Feature Toggles
summarize: true  # false = transcription only (no summarization phase)

# Summary Output Formats (only applies when summarize=true)
summary_output:
  docx: true      # Generate .docx summary with formatted Word document
  markdown: true  # Generate .md summary for version control and markdown viewers

# File Paths
input_folder_path: 'C:UsersyournameDocumentsPDFs'
output_folder_path: ''  # Empty = use input_folder_path as output location
input_paths_is_output_path: true  # true = write outputs next to each input file

# Cleanup Settings
delete_temp_working_dir: true  # Delete extracted images and temp files after completion

# Citation Management (OpenAlex API integration)
citation:
  openalex_email: 'your-email@example.com'  # Required for polite pool (faster responses)
  max_api_requests: 300  # Cap API calls per document to avoid rate limits

# Daily Token Limit (budget control)
daily_token_limit:
  enabled: true  # Enforce the daily token budget
  daily_tokens: 9000000  # Max tokens per day (resets at midnight)
```

**Key Settings Explained:**

-   **cli_mode**: Set to `true` for automation and scripting without interactive prompts
-   **summarize**: Set to `false` if you only need transcription without summaries (faster and cheaper)
-   **summary_output.docx**: Generate formatted Word documents (default: true)
-   **summary_output.markdown**: Generate Markdown files for version control and markdown viewers (default: true)
-   **input_paths_is_output_path**: When `true`, outputs are saved beside the source file instead of the global `output_folder_path`
-   **delete_temp_working_dir**: Automatically cleans up extracted images and temporary files after processing
-   **daily_token_limit**: Enforces a configurable daily token budget to stay within API allowances

### Provider Configuration

AutoExcerpter supports multiple LLM providers through LangChain. Configure your preferred provider in `modules/config/model.yaml`:

**Using OpenAI (default):**

```yaml
transcription_model:
  name: "gpt-5-mini"
  provider: "openai"  # Optional, auto-detected from model name
  max_output_tokens: 12000

summary_model:
  name: "gpt-5-mini"
  provider: "openai"
  max_output_tokens: 16384
```

**Using Anthropic Claude:**

```yaml
transcription_model:
  name: "claude-sonnet-4-5-20250929"
  provider: "anthropic"
  max_output_tokens: 16384
  temperature: 1.0  # Required for extended thinking
```

**Using Google Gemini:**

```yaml
transcription_model:
  name: "gemini-2.5-pro"
  provider: "google"
  max_output_tokens: 32768
  reasoning:
    effort: high  # Maps to thinking_level: "high"
```

**Using OpenRouter (access to multiple providers):**

```yaml
transcription_model:
  name: "anthropic/claude-sonnet-4-5"  # Use provider/model format
  provider: "openrouter"
  max_output_tokens: 8192
```

**Provider Auto-Detection:**

If you do not specify a provider, AutoExcerpter will infer it from the model name:

-   Models starting with `gpt-`, `o1`, `o3`, `o4` use OpenAI
-   Models starting with `claude-` use Anthropic
-   Models starting with `gemini-` use Google

**API Keys:**

Each provider requires its own API key set as an environment variable:

-   `OPENAI_API_KEY` for OpenAI
-   `ANTHROPIC_API_KEY` for Anthropic
-   `GOOGLE_API_KEY` for Google
-   `OPENROUTER_API_KEY` for OpenRouter

### Model Configuration

**File**: `modules/config/model.yaml`

LLM settings for transcription and summarization phases. Loaded by `api/llm_client.py`, `api/transcribe_api.py`, and `api/summary_api.py`. The file includes cross-references to related configuration files.

```yaml
# Transcription Model (OCR/image-to-text conversion)
transcription_model:
  name: "gpt-5.2"      # Model identifier
  provider: "openai"   # Optional if model prefix is unambiguous
  max_output_tokens: 12000
  
  # Cross-provider reasoning (maps to native implementations):
  # OpenAI: reasoning_effort | Anthropic: extended_thinking | Google: thinking_level
  reasoning:
    effort: medium  # low | medium | high
  
  # OpenAI GPT-5 only: controls output verbosity
  text:
    verbosity: medium  # low | medium | high
  
  temperature: 1.0  # 0.0-2.0 (null = provider default)

# Summary Model (structured summaries from transcriptions)
summary_model:
  name: "gpt-5-mini"
  provider: "openai"
  max_output_tokens: 16384
  
  reasoning:
    effort: medium
  
  text:
    verbosity: low  # Lower verbosity for concise summaries
  
  temperature: 1.0
```

**Parameter Details:**

-   **name**: Model identifier (e.g., "gpt-5-mini", "claude-sonnet-4-5", "gemini-2.5-pro")
-   **provider**: LLM provider (openai, anthropic, google, openrouter); can be auto-detected from model name
-   **max_output_tokens**: Controls the maximum length of model responses; increase for longer documents
-   **temperature**: Controls randomness in generation (0.0 for deterministic, up to 2.0 for creative output)
-   **reasoning.effort**: Higher effort improves accuracy but increases processing time and cost (OpenAI GPT-5/o-series only)
    -   `minimal`: Fastest, lowest cost, basic reasoning
    -   `low`: Balanced for simple documents (recommended)
    -   `medium`: Good balance for complex documents
    -   `high`: Maximum accuracy for complex technical content
-   **text.verbosity**: Controls output detail level (OpenAI GPT-5 family only)
    -   `low`: More concise, fewer details (good for summaries)
    -   `medium`: Balanced detail level (good for transcriptions)
    -   `high`: Maximum detail, comprehensive output

**Usage Recommendations:**

-   For **technical papers** with equations: Use `medium` or `high` reasoning effort
-   For **general documents**: Use `low` reasoning effort and `medium` verbosity
-   For **summaries**: Use `low` verbosity to keep them concise
-   For **transcriptions**: Use `medium` verbosity to capture all content
-   For **Anthropic/Google models**: The reasoning and text parameters are ignored; temperature and max_output_tokens apply

**Example configurations for other providers:**

```yaml
# Anthropic Claude
transcription_model:
  name: "claude-sonnet-4-5-20250929"
  provider: "anthropic"
  max_output_tokens: 8192
  temperature: 0.0

# Google Gemini
transcription_model:
  name: "gemini-2.5-pro"
  provider: "google"
  max_output_tokens: 8192
  temperature: 0.0

# OpenRouter (access any model)
transcription_model:
  name: "anthropic/claude-3-opus"
  provider: "openrouter"
  max_output_tokens: 4096
```

### Concurrency Configuration

**File**: `modules/config/concurrency.yaml`

Controls parallel processing behavior for both local operations and API requests. Each config file includes cross-references to related configuration files for easy navigation.

```yaml
# Local Image Processing (CPU/Disk bound)
image_processing:
  concurrency_limit: 24  # Parallel tasks (8-24 for SSD, lower for HDD)
  delay_between_tasks: 0  # No delay needed for local operations

# API Request Concurrency
api_requests:
  api_timeout: 900  # Request timeout (seconds). 900s for flex tier queuing
  
  # Rate limiting: [max_requests, window_seconds]
  rate_limits:
    - [120, 1]       # Per-second burst limit
    - [15000, 60]    # Per-minute sustained limit
    - [15000, 3600]  # Per-hour aggregate limit
  
  transcription:
    concurrency_limit: 5  # Parallel API requests (OpenAI: 50-150, Anthropic: 5-10)
    delay_between_tasks: 0.1
    service_tier: default  # 'default' | 'flex' | 'priority'
  
  summary:
    concurrency_limit: 5
    delay_between_tasks: 0.1
    service_tier: flex  # Cost optimization for batch summarization

# Retry Configuration (exponential backoff with jitter)
retry:
  max_attempts: 5  # Total attempts before giving up
  backoff_base: 1.0  # Initial wait time in seconds
  backoff_multipliers:
    rate_limit: 2.0   # 429 Too Many Requests
    timeout: 1.5      # Connection/read timeouts
    server_error: 2.0 # 500-series errors
    other: 2.0
  jitter:
    min: 0.5
    max: 1.0
  
  # Schema-specific retries (based on model output flags)
  schema_retries:
    transcription:
      no_transcribable_text:  # Image contains no text
        enabled: true
        max_attempts: 0  # 0 = disabled
        backoff_base: 0.5
        backoff_multiplier: 1.5
      transcription_not_possible:  # Illegible/corrupted image
        enabled: true
        max_attempts: 3
        backoff_base: 0.5
        backoff_multiplier: 1.5
    summary:
      contains_no_semantic_content:  # Blank page, TOC, etc.
        enabled: true
        max_attempts: 0
        backoff_base: 0.5
        backoff_multiplier: 1.5
      contains_no_page_number:
        enabled: true
        max_attempts: 0
        backoff_base: 0.5
        backoff_multiplier: 1.5
```

**Service Tier Options:**

-   **`auto`**: Let OpenAI choose the best tier automatically
-   **`default`**: Standard processing speed and cost
-   **`flex`**: Lower cost, longer processing time (recommended for batch processing)
-   **`priority`**: Faster processing, higher cost (for time-sensitive work)

**Tuning Guidelines:**

-   **Lower OpenAI Tiers (1-2)**: Set `concurrency_limit: 10-30`
-   **Mid-Tier (3-4)**: Set `concurrency_limit: 50-100`
-   **High Tier (4-5)**: Set `concurrency_limit: 100-200`
-   **Image Processing**: 8-24 for HDD, 24-48 for SSD systems

#### Schema-Aware Retry Controls

AutoExcerpter implements two complementary retry layers:

-   **API errors**: Controlled by `max_attempts`, `backoff_base`, and `backoff_multipliers`. Applies to rate limits, timeouts, and server-side errors using exponential backoff with jitter, based on OpenAI cookbook guidance.
-   **Schema flags**: Configure per-flag policies under `schema_retries`. Each flag exposes `enabled`, `max_attempts`, `backoff_base`, and `backoff_multiplier`. Most flags default to `max_attempts: 0` to avoid unnecessary reprocessing, except `transcription_not_possible` which defaults to 3 attempts for handling temporarily illegible images. Increase attempts when working with noisy scans or documents that frequently trigger these flags.

**Supported Flags:**

-   **Transcription**: `no_transcribable_text`, `transcription_not_possible`
-   **Summary**: `contains_no_semantic_content`, `contains_no_page_number`

When a flag is enabled and the model returns `true`, AutoExcerpter automatically re-issues the same request after waiting `backoff_base * backoff_multiplier^attempt + jitter`. Statistics for both API-level and schema-level retries are logged per page for auditing.

### Image Processing Configuration

**File**: `modules/config/image_processing.yaml`

Controls image preprocessing before LLM API calls. The file includes cross-references to related configuration files and is loaded by `modules/image_utils.py`.

```yaml
api_image_processing:
  # PDF rendering resolution (higher = better quality, more tokens)
  target_dpi: 300  # 150-300 recommended for OCR
  
  # Preprocessing steps
  grayscale_conversion: true  # Convert to grayscale (improves OCR, reduces noise)
  handle_transparency: true   # Flatten alpha channel onto white background
  
  # Image fidelity for OpenAI Vision API
  # high = better OCR, more tokens | low = faster, cheaper | auto = model decides
  llm_detail: high
  
  # JPEG compression quality (1-100). Higher = better quality, larger files
  jpeg_quality: 100
  
  # Resize strategy: 'high' | 'low' | 'auto' | 'none'
  resize_profile: high
  
  # Resize parameters
  low_max_side_px: 512         # Max dimension for 'low' profile
  high_target_box: [768, 1536] # [width, height] target for 'high' profile
```

**Parameter Effects:**

-   **target_dpi**: Higher DPI improves OCR accuracy but increases file size and processing time
    -   150-250: Basic documents, low quality scans
    -   300: Standard (recommended for most documents)
    -   400-600: High quality, small text, or complex layouts
-   **llm_detail**: Controls OpenAI Vision API processing fidelity
    -   `low`: Faster, cheaper, suitable for clean text
    -   `high`: Better accuracy for complex layouts (recommended)
    -   `auto`: Let the model decide based on image characteristics
-   **jpeg_quality**: Higher quality preserves text clarity
    -   85-95: Good balance of quality and file size
    -   95-100: Maximum quality for difficult documents

### Text Cleaning Configuration

**File**: `modules/config/image_processing.yaml` (`text_cleaning` section)

Post-processing pipeline that polishes transcription text before summarization and TXT export. All stages are individually configurable. Loaded by `modules/text_cleaner.py`.

```yaml
text_cleaning:
  enabled: true  # Master switch for all text cleaning
  
  # NFC normalization, removes control chars, soft hyphens, zero-width spaces
  unicode_normalization: true
  
  # Repairs common OCR errors in mathematical notation
  latex_fixing:
    enabled: true
    balance_dollar_signs: true   # Fix unbalanced $ and $$ delimiters
    close_unclosed_braces: true  # Close orphan { in LaTeX commands
    fix_common_commands: true    # Fix typos like "frac {" -> "frac{"
  
  # Rejoins words split at line breaks (disabled by default)
  # WARNING: Can damage genuine compounds like "Jean-Baptiste"
  merge_hyphenation: false
  
  whitespace_normalization:
    enabled: true
    collapse_internal_spaces: true  # 3+ spaces -> 2 spaces
    max_blank_lines: 2              # Remove excess blank lines
    tab_size: 4                     # Spaces per tab when expanding
  
  # Wrap excessively long lines (usually not needed for LLM output)
  line_wrapping:
    enabled: true
    auto_width: true   # Compute width from text statistics
    fixed_width: 80    # Used if auto_width is false
```

**Capabilities:**

-   **Unicode normalization** removes soft hyphens, zero-width characters, BOMs, and other control characters while preserving semantic content.
-   **LaTeX fixing** balances `$`/`$$` delimiters, closes unpaired braces, and corrects common OCR spacing errors in math commands.
-   **Hyphenation merging** (disabled by default) rejoins words split across line breaks. Enable only for heavily hyphenated scans, as it can damage genuine hyphenated compounds.
-   **Whitespace normalization** trims trailing spaces, collapses long internal gaps, expands tabs, and limits consecutive blank lines for clean TXT output.
-   **Line wrapping** optionally reflows especially long lines using indentation-aware wrapping or automatically computed widths.

These cleanups run immediately after each successful transcription so both the TXT output and downstream summarization benefit from the sanitized text.

### Summary Context Configuration

AutoExcerpter supports a hierarchical context system that allows you to guide the summarization model to pay special attention to specific topics. This is useful when you want more detailed summaries for content related to particular subjects.

**Context Resolution Hierarchy (highest to lowest priority):**

1. **CLI/Interactive Context**: Provided via `--context` flag or interactive prompt
2. **File-Specific Context**: `<filename>_summary_context.txt` next to the input file
3. **Folder-Specific Context**: `<foldername>_summary_context.txt` in the parent directory
4. **General Context**: `context/summary/general.txt` in the project root

**Creating Context Files:**

Create a plain text file with topics or keywords that you want the model to focus on during summarization:

```text
Food History
Wages and Labor Economics
Early Modern History
Agricultural Trade
```

**File Naming Examples:**

```
# For a single PDF: my_document.pdf
my_document_summary_context.txt

# For all files in a folder named "research_papers"
research_papers_summary_context.txt

# For all documents (global fallback)
context/summary/general.txt
```

**Usage:**

```bash
# CLI mode: specify context directly
python main.py "./documents" "./output" --all --context "Food History, Wages, Early Modern"

# Interactive mode: prompted after item selection (if summarization enabled)
# Or use context files for automatic resolution
```

**How It Works:**

When context is provided (via any method), the summarization prompt includes an instruction like:
> "Pay special attention to the following topics during summarization. Content related to these topics should be summarized in greater detail: Food History, Wages, Early Modern History."

If no context is found through any of these methods, summarization proceeds normally without topic-specific focus.

### Citation Management

**File**: `modules/config/app.yaml` (citation section)

Configure the enhanced citation system with OpenAlex integration.

```yaml
citation:
  openalex_email: 'your-email@example.com'  # Email for OpenAlex API polite pool
  max_api_requests: 300  # Maximum metadata enrichment calls per document
```

**Citation Features:**

The citation manager automatically:

1.  **Deduplicates** identical citations using normalized text comparison
2.  **Tracks pages** where each citation appears and displays as ranges (e.g., "pp. 3, 7-9, 15")
3.  **Enriches metadata** via OpenAlex API: DOI, authors, publication year, venue
4.  **Adds hyperlinks** to citations with DOIs for direct access to extended metadata
5.  **Consolidates** all citations in a dedicated bibliography section
6.  **Smart Matching**: Uses both text similarity and DOI extraction for accurate citation identification

**Best Practices:**

-   Replace `your-email@example.com` with your real email for faster OpenAlex response times (polite pool)
-   Set `max_api_requests` based on expected citation count (300 handles large documents with many references)
-   OpenAlex API is free and requires no API key
-   Citation matching uses both text similarity and DOI extraction for accuracy

### Daily Token Limit Tracking

**File**: `modules/config/app.yaml` (`daily_token_limit` section)

AutoExcerpter enforces a configurable daily token budget to keep usage aligned with your OpenAI allowance. Tokens are counted after every OpenAI API response (including retried attempts) and persisted to `.autoexcerpter_token_state.json` so limits survive restarts.

```yaml
daily_token_limit:
  enabled: true           # Toggle token tracking and enforcement
  daily_tokens: 9000000   # Maximum tokens allowed per calendar day (9 million)
```

**Behavior:**

-   **Accurate Accounting**: Tracks `usage.total_tokens` from each OpenAI Responses API call (transcription + summary).
-   **Daily Reset**: Counter resets automatically at local midnight; tokens persist across restarts until reset.
-   **Wait Handling**: When the limit is reached, processing pauses and shows the reset ETA.
-   **Interactive Cancellation**: During the wait, type `q` + Enter to cancel immediately (no need for Ctrl+C).
-   **Manual Overrides**: Delete or edit `.autoexcerpter_token_state.json` to synchronize with official dashboard totals if required.

## Usage

### Basic Usage

Run AutoExcerpter from the command line:

```bash
python main.py
```

The application will scan the configured input directory and present an interactive menu.

### Command-Line Arguments (CLI Mode)

When `cli_mode: true` is set in `app.yaml`, use positional arguments:

```bash
# Process a single PDF file
python main.py "./documents/paper.pdf" "./output"

# Process a specific folder of images
python main.py "./ScannedBook" "./output"

# Process all items in a directory
python main.py "./documents" "./output" --all
```

See the [CLI Mode and Interactive Mode](#cli-mode-and-interactive-mode) section for detailed usage.

### Interactive Selection

When prompted, use flexible selection syntax:

```
Enter your choice(s) (e.g., 1; 3-5; all): 

# Single item
1

# Range of items
3-5

# Multiple individual items
1; 5; 8

# Combined ranges and items
2; 4-7; 10

# Process all items
all
```

### Processing Modes

**Transcription-Only Mode:**

For faster processing without summaries, set `summarize: false` in `modules/config/app.yaml`:

```yaml
summarize: false
```

This mode is ideal for:

-   Creating searchable text archives
-   Quick digitization projects
-   Cost-sensitive workflows
-   Building document databases for later analysis

**Transcription + Summarization Mode:**

Enable full processing with `summarize: true`:

```yaml
summarize: true
```

This mode provides:

-   Complete transcriptions
-   Structured summaries
-   Extracted and enriched citations
-   Professional DOCX output

### Batch Processing

Process multiple documents in sequence:

1.  Place all PDFs and image folders in your input directory
2.  Run `python main.py`
3.  Select "all" when prompted
4.  The system processes each item sequentially with progress tracking

The application automatically skips items that have already been processed (output files exist).

## Output Files

AutoExcerpter generates organized outputs for each processed document.

### Main Output Files

Located in the configured output directory:

**1. `<document_name>.txt` - Complete Transcription**

Plain text file containing:

-   Metadata header with processing information (source file, total time, page count, model used)
-   Full verbatim transcription with markdown formatting
-   Mathematical equations in LaTeX notation (e.g., `$E = mc^2$`)
-   Page numbers marked with XML-style tags: `<page_number>15</page_number>`
-   Footnotes in markdown format: `[^1]: Reference text here`
-   Preserved tables, headers, footers, and multi-column layouts
-   Descriptions of visual elements (images, diagrams, charts)

Example snippet:

```
# Metadata
Source: research_paper.pdf
Total Processing Time: 00:05:32
Total Pages: 42
Model: gpt-5-mini

# Transcription

<page_number>1</page_number>

# Quantum Computing Fundamentals

**Abstract**

This paper explores the mathematical foundations of quantum computing...

The Schrödinger equation is represented as: $ihbarfrac{partial}{partial t}Psi = hat{H}Psi$
```

**2. `<document_name>_summary.docx` - Formatted Summary** (if summarization and DOCX output enabled)

Professional DOCX document containing:

-   Document title and metadata
-   Page-by-page structured summaries with clear headings (e.g., "Page 5 Summary")
-   Bullet-point extraction of main ideas and key findings
-   Automatic exclusion of non-semantic pages (title pages, blank pages, reference lists)
-   Consolidated bibliography section at the end
-   Deduplicated citations with page ranges (e.g., "Smith et al. (2023) - pp. 3, 7-9, 15")
-   Clickable hyperlinks to citations with DOI information
-   Professional formatting suitable for reports and documentation
-   LaTeX formulas converted to native Word equations (via MathML/OMML)

**3. `<document_name>_summary.md` - Markdown Summary** (if summarization and Markdown output enabled)

Version-control friendly Markdown file containing:

-   Same content structure as the DOCX summary
-   Page-by-page summaries with Markdown headings (`## Page X`)
-   Bullet points in standard Markdown format
-   LaTeX formulas preserved as-is (`$...$` and `$$...$$`) for compatibility with MathJax/KaTeX
-   Consolidated references section with hyperlinks in Markdown format
-   Compatible with Writage and other markdown-to-Word conversion tools
-   Ideal for Git repositories, documentation systems, and static site generators

### Working Files

Located in `<document_name>_working_files/` subdirectory (automatically deleted if `delete_temp_working_dir: true`):

**4. `images/` - Temporary Images** (PDF processing only)

Directory containing extracted page images:

-   Named sequentially: `page_0001.jpg`, `page_0002.jpg`, etc.
-   Preprocessed format: grayscale JPEG at configured DPI
-   Used for API submission during transcription

**5. `<document_name>_transcription_log.json` - Transcription Log**

Detailed JSON log containing:

```json
{
  "metadata": {
    "input_path": "C:Users...research_paper.pdf",
    "model": "gpt-5-mini",
    "extraction_dpi": 300,
    "total_pages": 42,
    "start_time": "2025-09-30T18:30:00",
    "end_time": "2025-09-30T18:35:32",
    "total_duration_seconds": 332
  },
  "pages": [
    {
      "page_number": 1,
      "status": "success",
      "transcription": "...",
      "processing_time_seconds": 3.2,
      "timestamp": "2025-09-30T18:30:03"
    }
  ],
  "errors": []
}
```

Useful for:

-   Debugging transcription issues
-   Quality assurance and verification
-   Performance analysis and optimization
-   Audit trails and documentation

**6. `<document_name>_summary_log.json` - Summary Log** (if summarization enabled)

Detailed JSON log for summarization:

```json
{
  "metadata": {
    "transcription_log_path": "..._transcription_log.json",
    "model": "gpt-5-mini",
    "total_pages_summarized": 40,
    "pages_excluded": 2
  },
  "summaries": [
    {
      "page_number": 5,
      "summary_text": "...",
      "citations": ["Smith et al. (2023)"],
      "has_semantic_content": true,
      "processing_time_seconds": 2.1
    }
  ]
}
```

## Project Structure

AutoExcerpter follows a modular architecture with clear separation of concerns:

```
AutoExcerpter/
├── api/                                # LangChain Multi-Provider API Layer
│   ├── providers/                      # Provider-specific implementations
│   │   ├── base.py                     # BaseProvider abstract class
│   │   ├── openai_provider.py          # OpenAI GPT-5/4/o-series
│   │   ├── anthropic_provider.py       # Anthropic Claude models
│   │   ├── google_provider.py          # Google Gemini models
│   │   └── openrouter_provider.py      # OpenRouter multi-provider proxy
│   ├── llm_client.py                   # Multi-provider LLM client with capability guarding
│   ├── base_llm_client.py              # Provider-agnostic base class with retry logic
│   ├── transcribe_api.py               # TranscriptionManager for image-to-text
│   ├── summary_api.py                  # SummaryManager for structured summaries
│   └── rate_limiter.py                 # Sliding window rate limiter for API quotas
│
├── core/                               # Core Processing Logic
│   └── transcriber.py                  # Main ItemTranscriber orchestration class
│
├── processors/                         # File I/O and Processing Utilities
│   ├── citation_manager.py             # Citation deduplication and OpenAlex enrichment
│   ├── file_manager.py                 # Output file management (TXT, DOCX, JSON)
│   └── pdf_processor.py                # PDF page extraction and preprocessing
│
├── modules/                            # Configuration and Utilities
│   ├── config/                         # YAML Configuration Files
│   │   ├── app.yaml                    # Application settings and paths
│   │   ├── model.yaml                  # LLM provider and model settings
│   │   ├── concurrency.yaml            # API concurrency, rate limits, retries
│   │   └── image_processing.yaml       # Image preprocessing and text cleaning
│   ├── prompts/                        # System Prompts for AI Models
│   │   ├── transcription_system_prompt.txt
│   │   └── summary_system_prompt.txt
│   ├── schemas/                        # JSON Schemas for Structured Outputs
│   │   ├── transcription_schema.json
│   │   └── summary_schema.json
│   ├── app_config.py                   # Configuration loader with validation
│   ├── config_loader.py                # YAML/JSON parsing utilities
│   ├── concurrency_helper.py           # Concurrency settings access
│   ├── image_utils.py                  # In-memory image preprocessing
│   ├── text_cleaner.py                 # Post-transcription text cleaning
│   ├── token_tracker.py                # Daily token budget tracking
│   └── logger.py                       # Logging configuration
│
├── main.py                             # Entry point and CLI interface
├── requirements.txt                    # Python runtime dependencies
├── requirements-dev.txt                # Python development and testing dependencies
└── README.md                           # This file
```

**Module Responsibilities:**

**`api/` - LangChain Multi-Provider API Layer**

-   Provides a unified interface for multiple LLM providers (OpenAI, Anthropic, Google, OpenRouter)
-   `providers/`: Provider-specific implementations with capability detection
-   `llm_client.py`: Multi-provider LLM client with model capability profiles and auto-detection
-   `base_llm_client.py`: Provider-agnostic base class with rate limiting integration and schema retries
-   `transcribe_api.py`: TranscriptionManager for image-to-text processing with structured output
-   `summary_api.py`: SummaryManager for generating structured summaries with citation extraction
-   `rate_limiter.py`: Implements sliding window rate limiting to prevent quota violations

**`core/` - Pipeline Orchestration**

-   Contains the main `ItemTranscriber` class that orchestrates the entire processing pipeline
-   Manages workflow from input selection through transcription, summarization, and output generation
-   Handles progress tracking, ETA calculation, and error recovery
-   Coordinates between API clients, file processors, and configuration modules

**`processors/` - File Operations**

-   Handles all file I/O operations including reading, writing, and format conversion
-   `citation_manager.py`: Manages citation deduplication, page tracking, and OpenAlex API enrichment
-   `file_manager.py`: Creates output files (TXT transcriptions, DOCX summaries, JSON logs)
-   `pdf_processor.py`: Extracts PDF pages as images with preprocessing (grayscale, transparency, resizing)

**`modules/` - Configuration and Utilities**

-   Provides configuration management, utility functions, and resources
-   `config/`: YAML configuration files for all aspects of the application
-   `prompts/`: System prompts that instruct the AI models on how to process content
-   `schemas/`: JSON schemas that enforce structured output from API responses
-   `app_config.py`: Loads and validates configuration with sensible defaults
-   `image_utils.py`: In-memory image preprocessing to eliminate disk I/O bottlenecks
-   `text_cleaner.py`: Post-transcription text cleaning (Unicode, LaTeX, whitespace)
-   `token_tracker.py`: Daily token budget tracking with midnight reset
-   `concurrency_helper.py`: Concurrency and rate limit settings access

**`main.py` - Application Entry Point**

-   CLI interface with input scanning and user selection
-   High-level workflow coordination
-   Error handling and cleanup logic
-   Manages temporary working directories

## Advanced Topics

### Performance Optimization

**Maximizing Throughput:**

To achieve optimal processing speed:

1.  Set `concurrency_limit` in `concurrency.yaml` based on your provider tier (OpenAI: 50-150, Anthropic: 5-10)
2.  Use `service_tier: flex` for batch processing (slower but 50% cheaper)
3.  Increase `image_processing.concurrency_limit` to 24-48 on SSD systems
4.  Use `llm_detail: auto` or `low` for straightforward documents
5.  Set appropriate `target_dpi` (300 is usually optimal)

**Memory Management:**

For large batch processing:

-   Enable `delete_temp_working_dir: true` to clean up temporary files
-   Process documents in batches rather than all at once
-   Reduce `concurrency_limit` in `concurrency.yaml` if experiencing memory pressure
-   Monitor system resources during processing

### Cost Management

**Reducing Processing Costs:**

1.  **Use Flex Tier**: Set `service_tier: flex` for 40-50% cost savings
2.  **Lower Image Quality**: Reduce `target_dpi` to 200-250 for clean documents
3.  **Transcription-Only Mode**: Set `summarize: false` if summaries aren't needed
4.  **Optimize Detail Level**: Use `llm_detail: low` for straightforward text
5.  **Batch Processing**: Process multiple documents in one session to amortize startup costs

**Cost Tracking:**

Monitor your OpenAI API usage at [https://platform.openai.com/usage](https://platform.openai.com/usage) to understand:

-   Cost per page for different document types
-   Impact of various configuration settings
-   Optimal settings for your use cases

### Best Practices

**Document Preparation:**

-   Ensure scanned documents are clear and high-contrast
-   Remove unnecessary blank pages before processing
-   Verify PDFs are not already text-based (use native PDF text extraction if possible)
-   Organize documents in logical folders for batch processing

**Configuration Management:**

-   Start with default settings and adjust based on results
-   Test with a small document before processing large batches
-   Keep separate configuration files for different document types
-   Document your configuration choices for reproducibility

**Quality Assurance:**

-   Review transcription logs for errors and failed pages
-   Spot-check transcriptions against source documents
-   Verify citations are correctly extracted and deduplicated
-   Check page number alignment in summaries

**Security and Privacy:**

-   Never commit API keys to version control (use environment variables)
-   Be mindful of sensitive information in documents
-   Review data usage policies for your chosen AI provider
-   Consider using environment variables or secret management tools
-   Keep temporary working directories secure during processing

## Troubleshooting

**Common Issues and Solutions:**

**Issue: "No API key found in environment"**

-   Ensure you have set the appropriate API key environment variable for your provider:
    -   `OPENAI_API_KEY` for OpenAI models
    -   `ANTHROPIC_API_KEY` for Claude models
    -   `GOOGLE_API_KEY` for Gemini models
    -   `OPENROUTER_API_KEY` for OpenRouter
-   Verify the key is correct and has not been revoked
-   Check that the environment variable is set in your current shell session

**Issue: "Unsupported parameter" errors**

-   The model may not support certain parameters (e.g., reasoning_effort on gpt-4o)
-   AutoExcerpter should automatically filter unsupported parameters via capability guarding
-   If you see this error, ensure you are using the latest version of the application

**Issue: Rate limit errors (429 responses)**

-   Reduce `concurrency_limit` in `concurrency.yaml` (try 2 or 3)
-   Adjust `rate_limits` in `concurrency.yaml` to match your provider tier limits
-   Set `service_tier: flex` for less aggressive rate limiting (OpenAI only)
-   Check your API account for quota limits

**Issue: Timeout errors**

-   Increase `api_timeout` in `concurrency.yaml` (e.g., 600 for larger images)
-   Reduce `target_dpi` in `image_processing.yaml` to decrease image size
-   Use `llm_detail: low` for faster processing

**Issue: Poor transcription quality**

-   Increase `target_dpi` in `image_processing.yaml` (try 400 or 600)
-   Set `llm_detail: high` for better accuracy
-   Ensure source images/PDFs are high quality and legible
-   Check that the correct language and formatting are used in source documents

**Issue: Memory errors during processing**

-   Reduce `concurrency_limit` in `concurrency.yaml` for both transcription and summary tasks
-   Reduce `image_processing.concurrency_limit` to limit parallel image processing
-   Process documents in smaller batches

**Issue: "Permission denied" errors when deleting working files**

-   Set `delete_temp_working_dir: false` to preserve working files
-   Close any programs that might have files open (e.g., image viewers)
-   Run the application with appropriate file system permissions

**Issue: Missing or incorrect page numbers in summaries**

-   Verify that page numbers are visible in the source document
-   Check transcription output for `<page_number>X</page_number>` tags
-   Ensure pages actually contain page numbers in headers/footers

**Getting Help:**

If you encounter issues not covered here:

1.  Check the log files in the `_working_files/` directory for detailed error messages
2.  Review the OpenAI API status page for service disruptions
3.  Enable debug logging by modifying `modules/logger.py`
4.  Open an issue on the project repository with relevant log excerpts

## Contributing

Contributions to AutoExcerpter are welcome! Whether you are fixing bugs, adding features, improving documentation, or suggesting enhancements, your input is valuable.

**Development Best Practices:**

-   **Code Style**: Follow PEP 8 guidelines for Python code
-   **Type Hints**: Use type annotations for function signatures
-   **Documentation**: Add docstrings to all public functions and classes
-   **Error Handling**: Include robust error handling with informative messages
-   **Testing**: Write unit tests for new functionality
-   **Modularity**: Maintain clear separation of concerns with well-defined interfaces
-   **Logging**: Use the logger module for debugging information

**Contribution Workflow:**

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/your-feature-name`)
3.  Make your changes with clear, descriptive commit messages
4.  Test your changes thoroughly
5.  Update documentation as needed
6.  Submit a pull request with a detailed description of your changes

**Areas for Contribution:**

-   Support for additional output formats (Markdown, HTML, etc.)
-   Support for additional LLM providers (AWS Bedrock, Cohere, Mistral, etc.)
-   Enhanced image preprocessing algorithms
-   Batch API integration for cost-optimized processing
-   Additional unit and integration tests
-   Performance optimizations
-   User interface improvements
-   Extended model capability profiles for new model releases

## License

MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.