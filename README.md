# AutoExcerpter

AutoExcerpter is an intelligent document processing pipeline that automatically transcribes and summarizes PDF documents and image collections. Powered by OpenAI's advanced language models (GPT-5-mini by default), the tool converts scanned documents into searchable, structured text with optional summaries and enriched bibliographic citations. It is designed for researchers, academics, and professionals who need to digitize and analyze large volumes of documents efficiently.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Basic Configuration](#basic-configuration)
  - [Model Configuration](#model-configuration)
  - [Concurrency Configuration](#concurrency-configuration)
  - [Image Processing Configuration](#image-processing-configuration)
  - [Citation Management](#citation-management)
- [Usage](#usage)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

AutoExcerpter processes documents through a sophisticated two-stage pipeline that leverages OpenAI's vision-enabled models with optical character recognition (OCR) capabilities. In the first stage, each page is transcribed using the OpenAI Responses API with structured JSON schemas to ensure consistent output. In the second optional stage, transcribed text is analyzed to generate structured summaries with automatically deduplicated citations enriched with metadata from the OpenAlex academic database.

**Primary Use Cases:**

- **Academic Research**: Digitize scanned academic papers, extract citations, and generate structured literature reviews
- **Document Digitization**: Convert image-based archival documents into searchable, machine-readable text
- **Literature Analysis**: Automatically summarize research papers with consolidated bibliographies
- **Knowledge Management**: Build searchable databases from historical documents and scanned materials
- **Content Extraction**: Process multi-column layouts, preserve mathematical equations, and capture visual content descriptions

## Key Features

**Advanced Transcription:**

- **Multi-Format Support**: Processes PDFs (with automatic page extraction) and direct image folders
- **Format Compatibility**: Supports PNG, JPG, JPEG, TIFF, BMP, GIF, and WEBP image formats
- **Structure Preservation**: Maintains document structure including headers, footers, page numbers, footnotes, tables, and multi-column layouts
- **Mathematical Notation**: Converts mathematical equations to LaTeX format for accurate representation
- **Visual Content**: Provides detailed descriptions of images, diagrams, charts, and figures
- **Schema-Driven Output**: Uses strict JSON schemas to ensure consistent, parseable transcription results
- **In-Memory Processing**: Optimized image preprocessing pipeline that eliminates disk I/O bottlenecks and prevents race conditions
- **Intelligent Formatting**: Preserves markdown formatting (headings, bold, italic) and line breaks

**Intelligent Summarization:**

- **Concise Extraction**: Generates bullet-point summaries highlighting main ideas and key findings for each page
- **Smart Filtering**: Automatically identifies and excludes pages without semantic content (title pages, blank pages, reference lists)
- **Page Tracking**: Accurately tracks page numbers from document headers and footers
- **Professional Output**: Exports summaries as formatted DOCX documents suitable for reports and documentation

**Enhanced Citation Management:**

- **Automatic Deduplication**: Uses normalized text hashing to identify and merge duplicate citations across the entire document
- **Page Range Tracking**: Displays comprehensive page numbers where each citation appears (e.g., "pp. 5, 12-15, 23")
- **Metadata Enrichment**: Integrates with OpenAlex API to enrich citations with DOI, publication year, authors, and venue information
- **Clickable Hyperlinks**: Adds direct hyperlinks to citations for instant access to extended metadata
- **Consolidated Bibliography**: Presents all citations in a dedicated section at the end of summary documents
- **Smart Matching**: Uses both text similarity and DOI extraction for accurate citation identification

**Performance and Reliability:**

- **Concurrent Processing**: Configurable parallelism for both image preprocessing and API requests
- **Adaptive Rate Limiting**: Sliding window rate limiter prevents API quota violations
- **Intelligent Retry Logic**: Configurable dual-layer retries combining exponential backoff with schema-aware content retries
- **Service Tier Support**: Full support for OpenAI Flex tier to reduce processing costs by up to 50%
- **Progress Tracking**: Real-time progress bars with estimated time of completion
- **Comprehensive Logging**: Detailed JSON logs for debugging, quality assurance, and audit trails
- **Automatic Cleanup**: Optional deletion of temporary working directories after successful processing

**Architecture Excellence:**

- **Modular Design**: Clear separation of concerns with well-defined component responsibilities
- **YAML-Based Configuration**: Human-readable configuration with validation and sensible defaults
- **Base Classes**: Shared API logic through inheritance to eliminate code duplication
- **Testable Components**: Well-defined interfaces that facilitate unit testing
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Public API**: Clear module exports via `__all__` declarations

## How It Works

AutoExcerpter follows a systematic workflow to transform documents into structured, searchable content:

**1. Input Selection and Scanning**

The application scans your configured input directory for processable items (PDF files and image folders). It presents an interactive menu where you can select specific items or process all available documents in batch.

**2. Page Extraction (PDF Only)**

For PDF inputs, each page is extracted as a high-resolution image using configurable DPI settings (default: 300 DPI). The extraction process applies optimizations including grayscale conversion, transparency handling, and format normalization. For image folder inputs, existing images are processed directly.

**3. Image Preprocessing**

Images undergo in-memory preprocessing to optimize OCR accuracy:
- Grayscale conversion to reduce noise and improve text recognition
- Transparency handling to flatten alpha channels onto white backgrounds
- Intelligent resizing based on detail level (maintains aspect ratio)
- JPEG compression optimization for API transmission
- All processing happens in memory to prevent race conditions and improve performance

**4. Transcription via OpenAI Responses API**

Each preprocessed image is sent to OpenAI's Responses API with:
- A detailed system prompt instructing the model to perform verbatim transcription
- Structured JSON schema defining the expected output format
- Configuration for reasoning effort and text verbosity (from model.yaml)
- Model parameters including max_output_tokens and service tier settings

The API returns structured JSON containing:
- Full verbatim transcription with markdown formatting
- Preserved structural elements (headers, footers, tables, footnotes)
- Mathematical equations in LaTeX notation
- Page numbers marked with special XML-style tags
- Detailed descriptions of visual elements (images, diagrams, charts)
 - Built-in exponential backoff with jitter and schema-specific retries to gracefully recover from transient errors or ambiguous content flags

**5. Summarization (Optional)**

When enabled, transcribed text is processed by the OpenAI API again with a summarization prompt that extracts:
- Concise bullet-point summaries of main ideas
- Full bibliographic citations in APA format
- Page number metadata for accurate referencing
- Flags indicating pages without semantic content

**6. Citation Processing**

The citation manager processes extracted citations through:
- Normalization of citation text for duplicate detection
- Deduplication using text similarity hashing
- Page range consolidation (merges consecutive and discontinuous page references)
- OpenAlex API enrichment for DOI, authors, publication year, and venue
- Hyperlink generation for citations with sufficient metadata

**7. Output Generation**

The pipeline produces multiple output files:
- Plain text file with complete transcriptions including metadata headers
- Formatted DOCX file with structured summaries and consolidated citations
- JSON log files with detailed processing metadata, timing, and error information

**8. Cleanup**

If configured, the system automatically deletes temporary working directories including extracted images and intermediate processing files, keeping only the final outputs.

## Configuration

AutoExcerpter uses a multi-file YAML configuration system that provides fine-grained control over every aspect of processing. All configuration files are located in `modules/config/`.

### Basic Configuration

**File**: `modules/config/app.yaml`

This is the primary configuration file for application-level settings.

```yaml
# Feature Toggle
summarize: true  # Enable/disable summarization; false = transcription only

# Folder Paths
input_folder_path: 'C:\Users\yourname\Documents\PDFs'
output_folder_path: 'C:\Users\yourname\Documents\Output'

# Cleanup Settings
delete_temp_working_dir: true  # Delete temporary files after processing

# Performance Settings
concurrent_requests: 250  # Maximum concurrent API requests
api_timeout: 600  # Timeout per API request in seconds

# Citation Management
citation:
  openalex_email: 'your-email@example.com'  # Email for OpenAlex API polite pool
  max_api_requests: 50  # Maximum metadata enrichment calls per document

# OpenAI Configuration
openai:
  model: 'gpt-5-mini'  # Model for summarization
  transcription_model: 'gpt-5-mini'  # Model for transcription
  api_timeout: 900  # Extended timeout for Flex processing (seconds)
  use_flex: true  # Enable Flex tier for cost savings
  
  # Rate Limiting (adjust based on your OpenAI tier)
  rate_limits:
    - [120, 1]      # Max 120 requests per second
    - [15000, 60]   # Max 15,000 requests per minute
    - [15000, 3600] # Max 15,000 requests per hour
```

**Key Settings Explained:**

- **summarize**: Set to `false` if you only need transcription without summaries (faster and cheaper)
- **concurrent_requests**: Higher values increase speed but may hit rate limits; adjust based on your OpenAI tier
- **use_flex**: Flex tier offers 50% cost savings with slightly longer processing times (recommended for batch jobs)
- **rate_limits**: Must match your OpenAI account tier limits; check your account dashboard for specific limits

### Model Configuration

**File**: `modules/config/model.yaml`

This file controls advanced model-specific parameters for GPT-5 and o-series models. These settings allow fine-tuning of model behavior for optimal transcription and summarization quality.

```yaml
# Transcription Model Configuration
transcription_model:
  name: "gpt-5-mini"  # Model identifier
  max_output_tokens: 28000  # Maximum tokens for model output
  reasoning:
    effort: medium  # Options: minimal, low, medium, high
  text:
    verbosity: medium  # Options: low, medium, high

# Summary Model Configuration
summary_model:
  name: "gpt-5-mini"  # Model identifier
  max_output_tokens: 12000  # Maximum tokens for model output
  reasoning:
    effort: medium  # Options: minimal, low, medium, high
  text:
    verbosity: low  # Options: low, medium, high (low for concise summaries)
```

**Parameter Details:**

- **max_output_tokens**: Controls the maximum length of model responses; increase for longer documents
- **reasoning.effort**: Higher effort improves accuracy but increases processing time and cost
  - `minimal`: Fastest, lowest cost, basic reasoning
  - `low`: Balanced for simple documents
  - `medium`: Good balance for most documents (recommended)
  - `high`: Maximum accuracy for complex technical content
- **text.verbosity**: Controls output detail level
  - `low`: More concise, fewer details (good for summaries)
  - `medium`: Balanced detail level (good for transcriptions)
  - `high`: Maximum detail, comprehensive output

**Usage Recommendations:**

- For **technical papers** with equations: Use `medium` or `high` reasoning effort
- For **general documents**: Use `medium` reasoning effort and verbosity
- For **summaries**: Use `low` verbosity to keep them concise
- For **transcriptions**: Use `medium` verbosity to capture all content

### Concurrency Configuration

**File**: `modules/config/concurrency.yaml`

Controls parallel processing behavior for both local operations and API requests.

```yaml
# Local Image Processing (CPU/Disk bound)
image_processing:
  concurrency_limit: 24
  delay_between_tasks: 0

# API Request Concurrency
api_requests:
  transcription:
    concurrency_limit: 150
    delay_between_tasks: 0.05
    service_tier: flex
    batch_chunk_size: 50
  summary:
    concurrency_limit: 150
    delay_between_tasks: 0.05
    service_tier: flex
    batch_chunk_size: 50

# Retry Configuration
retry:
  max_attempts: 5
  backoff_base: 1.0
  backoff_multipliers:
    rate_limit: 2.0
    timeout: 1.5
    server_error: 2.0
    other: 2.0
  jitter:
    min: 0.5
    max: 1.0
  schema_retries:
    transcription:
      no_transcribable_text:
        enabled: true
        max_attempts: 0
        backoff_base: 2.0
        backoff_multiplier: 1.5
      transcription_not_possible:
        enabled: true
        max_attempts: 0
        backoff_base: 2.0
        backoff_multiplier: 1.5
    summary:
      contains_no_semantic_content:
        enabled: true
        max_attempts: 0
        backoff_base: 2.0
        backoff_multiplier: 1.5
      contains_no_page_number:
        enabled: true
        max_attempts: 0
        backoff_base: 2.0
        backoff_multiplier: 1.5
```

**Service Tier Options:**

- **`auto`**: Let OpenAI choose the best tier automatically
- **`default`**: Standard processing speed and cost
- **`flex`**: Lower cost, longer processing time (recommended for batch processing)
- **`priority`**: Faster processing, higher cost (for time-sensitive work)

**Tuning Guidelines:**

- **Lower OpenAI Tiers (1-2)**: Set `concurrency_limit: 10-30`
- **Mid-Tier (3-4)**: Set `concurrency_limit: 50-100`
- **High Tier (4-5)**: Set `concurrency_limit: 100-200`
- **Image Processing**: 8-24 for HDD, 24-48 for SSD systems

#### Schema-Aware Retry Controls

AutoExcerpter implements two complementary retry layers:

* __API errors__: Controlled by `max_attempts`, `backoff_base`, and `backoff_multipliers`. Applies to rate limits, timeouts, and server-side errors using exponential backoff with jitter, based on OpenAI cookbook guidance.
* __Schema flags__: Configure per-flag policies under `schema_retries`. Each flag exposes `enabled`, `max_attempts`, `backoff_base`, and `backoff_multiplier`. Defaults keep these retries disabled (`max_attempts: 0`) to avoid unnecessary reprocessing, but you can increase attempts when working with noisy scans or documents that frequently trigger these flags.

**Supported Flags:**

* __Transcription__: `no_transcribable_text`, `transcription_not_possible`
* __Summary__: `contains_no_semantic_content`, `contains_no_page_number`

When a flag is enabled and the model returns `true`, AutoExcerpter automatically re-issues the same request after waiting `backoff_base * backoff_multiplier^attempt + jitter`. Statistics for both API-level and schema-level retries are logged per page for auditing.

### Image Processing Configuration

**File**: `modules/config/image_processing.yaml`

Controls image preprocessing and optimization for API submission.

```yaml
api_image_processing:
  target_dpi: 300  # DPI for PDF page extraction
  grayscale_conversion: true  # Convert to grayscale to reduce noise
  handle_transparency: true  # Flatten alpha channels onto white
  llm_detail: high  # Options: low, high, auto
  jpeg_quality: 100  # JPEG compression quality (1-100)
  resize_profile: high  # Options: auto, none, low, high
  
  # Resize Parameters
  low_max_side_px: 512  # Max dimension for low detail
  high_target_box: [768, 1536]  # Target dimensions for high detail
```

**Parameter Effects:**

- **target_dpi**: Higher DPI improves OCR accuracy but increases file size and processing time
  - 200-250: Basic documents, low quality scans
  - 300: Standard (recommended for most documents)
  - 400-600: High quality, small text, or complex layouts
- **llm_detail**: Controls OpenAI's processing fidelity
  - `low`: Faster, cheaper, suitable for clean text
  - `high`: Better accuracy for complex layouts (recommended)
  - `auto`: Let the model decide based on image characteristics
- **jpeg_quality**: Higher quality preserves text clarity
  - 85-95: Good balance of quality and file size
  - 95-100: Maximum quality for difficult documents

### Citation Management

**File**: `modules/config/app.yaml` (citation section)

Configure the enhanced citation system with OpenAlex integration.

```yaml
citation:
  openalex_email: 'your-email@example.com'  # Email for OpenAlex API polite pool
  max_api_requests: 50  # Maximum metadata enrichment calls per document
```

**Citation Features:**

The citation manager automatically:
1. **Deduplicates** identical citations using normalized text comparison
2. **Tracks pages** where each citation appears and displays as ranges (e.g., "pp. 3, 7-9, 15")
3. **Enriches metadata** via OpenAlex API: DOI, authors, publication year, venue
4. **Adds hyperlinks** to citations with DOIs for direct access to extended metadata
5. **Consolidates** all citations in a dedicated bibliography section
6. **Smart Matching**: Uses both text similarity and DOI extraction for accurate citation identification

**Best Practices:**

- Replace `your-email@example.com` with your real email for faster OpenAlex response times (polite pool)
- Set `max_api_requests` based on expected citation count (50 handles most papers)
- OpenAlex API is free and requires no API key
- Citation matching uses both text similarity and DOI extraction for accuracy

## Usage

### Basic Usage

Run AutoExcerpter from the command line:

```bash
python main.py
```

The application will scan the configured input directory and present an interactive menu.

### Command-Line Options

**Process Specific File or Directory:**

```bash
# Process a single PDF file
python main.py --input "C:\Users\yourname\Documents\paper.pdf"

# Process a specific folder of images
python main.py --input "C:\Users\yourname\Documents\ScannedBook"

# Process all items in a directory
python main.py --input "C:\Users\yourname\Documents\Research"
```

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
- Creating searchable text archives
- Quick digitization projects
- Cost-sensitive workflows
- Building document databases for later analysis

**Transcription + Summarization Mode:**

Enable full processing with `summarize: true`:

```yaml
summarize: true
```

This mode provides:
- Complete transcriptions
- Structured summaries
- Extracted and enriched citations
- Professional DOCX output

### Batch Processing

Process multiple documents in sequence:

1. Place all PDFs and image folders in your input directory
2. Run `python main.py`
3. Select "all" when prompted
4. The system processes each item sequentially with progress tracking

The application automatically skips items that have already been processed (output files exist).

## Output Files

AutoExcerpter generates organized outputs for each processed document.

### Main Output Files

Located in the configured output directory:

**1. `<document_name>.txt` - Complete Transcription**

Plain text file containing:
- Metadata header with processing information (source file, total time, page count, model used)
- Full verbatim transcription with markdown formatting
- Mathematical equations in LaTeX notation (e.g., `$E = mc^2$`)
- Page numbers marked with XML-style tags: `<page_number>15</page_number>`
- Footnotes in markdown format: `[^1]: Reference text here`
- Preserved tables, headers, footers, and multi-column layouts
- Descriptions of visual elements (images, diagrams, charts)

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

The Schrödinger equation is represented as: $i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi$
```

**2. `<document_name>_summary.docx` - Formatted Summary** (if summarization enabled)

Professional DOCX document containing:
- Document title and metadata
- Page-by-page structured summaries with clear headings (e.g., "Page 5 Summary")
- Bullet-point extraction of main ideas and key findings
- Automatic exclusion of non-semantic pages (title pages, blank pages, reference lists)
- Consolidated bibliography section at the end
- Deduplicated citations with page ranges (e.g., "Smith et al. (2023) - pp. 3, 7-9, 15")
- Clickable hyperlinks to citations with DOI information
- Professional formatting suitable for reports and documentation

### Working Files

Located in `<document_name>_working_files/` subdirectory (automatically deleted if `delete_temp_working_dir: true`):

**3. `images/` - Temporary Images** (PDF processing only)

Directory containing extracted page images:
- Named sequentially: `page_0001.jpg`, `page_0002.jpg`, etc.
- Preprocessed format: grayscale JPEG at configured DPI
- Used for API submission during transcription

**4. `<document_name>_transcription_log.json` - Transcription Log**

Detailed JSON log containing:

```json
{
  "metadata": {
    "input_path": "C:\\Users\\...\\research_paper.pdf",
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
- Debugging transcription issues
- Quality assurance and verification
- Performance analysis and optimization
- Audit trails and documentation

**5. `<document_name>_summary_log.json` - Summary Log** (if summarization enabled)

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
├── api/                                # OpenAI API Integration Layer
│   ├── __init__.py                     # Package initialization
│   ├── base_openai_client.py          # Base class with shared retry/error handling logic
│   ├── openai_api.py                   # Summary generation API client
│   ├── openai_transcribe_api.py       # Transcription API client with vision support and model.yaml parameter loading
│   └── rate_limiter.py                 # Sliding window rate limiter for API quotas
│
├── core/                               # Core Processing Logic
│   ├── __init__.py                     # Package initialization
│   └── transcriber.py                  # Main ItemTranscriber orchestration class
│
├── processors/                         # File I/O and Processing Utilities
│   ├── __init__.py                     # Package initialization
│   ├── citation_manager.py            # Citation deduplication and enrichment
│   ├── file_manager.py                 # Output file management (TXT, DOCX, JSON)
│   └── pdf_processor.py                # PDF page extraction and conversion
│
├── modules/                            # Configuration and Utilities
│   ├── config/                         # Configuration Files
│   │   ├── __init__.py                 # Package initialization
│   │   ├── app.yaml                    # Main application settings
│   │   ├── concurrency.yaml            # Concurrency and retry configuration
│   │   ├── image_processing.yaml       # Image preprocessing parameters
│   │   └── model.yaml                  # Model-specific parameters (GPT-5, o-series)
│   │
│   ├── prompts/                        # System Prompts for AI Models
│   │   ├── transcription_system_prompt.txt  # Detailed transcription instructions
│   │   └── summary_system_prompt.txt        # Summarization and citation extraction
│   │
│   ├── schemas/                        # JSON Schemas for Structured Outputs
│   │   ├── transcription_schema.json   # Schema for transcription API responses
│   │   └── summary_schema.json         # Schema for summarization API responses
│   │
│   ├── __init__.py                     # Package initialization
│   ├── app_config.py                   # Configuration loader with validation
│   ├── config_loader.py                # YAML parsing utilities
│   ├── image_utils.py                  # Image preprocessing and format handling
│   ├── logger.py                       # Logging configuration
│   └── prompt_utils.py                 # Prompt rendering utilities
│
├── tests/                              # Unit Tests
│   └── test_prompt_utils.py           # Tests for prompt utilities
│
├── main.py                             # Entry point and CLI interface
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

**Module Responsibilities:**

**`api/` - OpenAI API Integration**
- Encapsulates all OpenAI API interactions with retry logic, rate limiting, and error handling
- `base_openai_client.py`: Provides shared functionality for API clients including exponential backoff, jitter, and error recovery
- `openai_api.py`: Handles summarization API calls with the standard OpenAI API
- `openai_transcribe_api.py`: Manages transcription using the Responses API with vision support and model.yaml parameter loading
- `rate_limiter.py`: Implements sliding window rate limiting to prevent quota violations

**`core/` - Pipeline Orchestration**
- Contains the main `ItemTranscriber` class that orchestrates the entire processing pipeline
- Manages workflow from input selection through transcription, summarization, and output generation
- Handles progress tracking, ETA calculation, and error recovery
- Coordinates between API clients, file processors, and configuration modules

**`processors/` - File Operations**
- Handles all file I/O operations including reading, writing, and format conversion
- `citation_manager.py`: Manages citation deduplication, page tracking, and OpenAlex API enrichment
- `file_manager.py`: Creates output files (TXT transcriptions, DOCX summaries, JSON logs)
- `pdf_processor.py`: Extracts PDF pages as images with preprocessing (grayscale, transparency, resizing)

**`modules/` - Configuration and Utilities**
- Provides configuration management, utility functions, and resources
- `config/`: YAML configuration files for all aspects of the application
- `prompts/`: System prompts that instruct the AI models on how to process content
- `schemas/`: JSON schemas that enforce structured output from API responses
- `app_config.py`: Loads and validates configuration with sensible defaults
- `image_utils.py`: In-memory image preprocessing to eliminate disk I/O bottlenecks
- `logger.py`: Centralized logging configuration for debugging and audit trails
- `prompt_utils.py`: Utilities for loading and formatting prompts

**`main.py` - Application Entry Point**
- CLI interface with input scanning and user selection
- High-level workflow coordination
- Error handling and cleanup logic
- Manages temporary working directories

## Advanced Topics

### Performance Optimization

**Maximizing Throughput:**

To achieve optimal processing speed:
1. Set `concurrent_requests` based on your OpenAI tier (50-200 for Tier 4+)
2. Use `service_tier: flex` for batch processing (slower but 50% cheaper)
3. Increase `image_processing.concurrency_limit` to 24-48 on SSD systems
4. Use `llm_detail: auto` or `low` for straightforward documents
5. Set appropriate `target_dpi` (300 is usually optimal)

**Memory Management:**

For large batch processing:
- Enable `delete_temp_working_dir: true` to clean up temporary files
- Process documents in batches rather than all at once
- Reduce `concurrent_requests` if experiencing memory pressure
- Monitor system resources during processing

### Cost Management

**Reducing Processing Costs:**

1. **Use Flex Tier**: Set `service_tier: flex` for 40-50% cost savings
2. **Lower Image Quality**: Reduce `target_dpi` to 200-250 for clean documents
3. **Transcription-Only Mode**: Set `summarize: false` if summaries aren't needed
4. **Optimize Detail Level**: Use `llm_detail: low` for straightforward text
5. **Batch Processing**: Process multiple documents in one session to amortize startup costs

**Cost Tracking:**

Monitor your OpenAI API usage at https://platform.openai.com/usage to understand:
- Cost per page for different document types
- Impact of various configuration settings
- Optimal settings for your use cases

### Best Practices

**Document Preparation:**

- Ensure scanned documents are clear and high-contrast
- Remove unnecessary blank pages before processing
- Verify PDFs are not already text-based (use native PDF text extraction if possible)
- Organize documents in logical folders for batch processing

**Configuration Management:**

- Start with default settings and adjust based on results
- Test with a small document before processing large batches
- Keep separate configuration files for different document types
- Document your configuration choices for reproducibility

**Quality Assurance:**

- Review transcription logs for errors and failed pages
- Spot-check transcriptions against source documents
- Verify citations are correctly extracted and deduplicated
- Check page number alignment in summaries

**Security and Privacy:**

- Never commit your OpenAI API key to version control
- Be mindful of sensitive information in documents
- Review OpenAI's data usage policies for your organization
- Consider using environment variables or secret management tools
- Keep temporary working directories secure during processing

## Troubleshooting

**Common Issues and Solutions:**

**Issue: "No OPENAI_API_KEY found in environment"**
- Ensure you have set the `OPENAI_API_KEY` environment variable
- Verify the key is correct and has not been revoked
- Check that the environment variable is set in your current shell session

**Issue: Rate limit errors (429 responses)**
- Reduce `concurrent_requests` in `app.yaml` (try 2 or 3)
- Adjust `rate_limits` to match your OpenAI tier
- Enable `use_flex: true` for less aggressive rate limiting
- Check your OpenAI account for quota limits

**Issue: Timeout errors**
- Increase `api_timeout` in `app.yaml` (e.g., 600 for larger images)
- Reduce `target_dpi` in `image_processing.yaml` to decrease image size
- Use `llm_detail: low` for faster processing

**Issue: Poor transcription quality**
- Increase `target_dpi` in `image_processing.yaml` (try 400 or 600)
- Set `llm_detail: high` for better accuracy
- Ensure source images/PDFs are high quality and legible
- Check that the correct language and formatting are used in source documents

**Issue: Memory errors during processing**
- Reduce `concurrent_requests` to limit parallel processing
- Reduce `concurrency_limit` for transcription tasks
- Process documents in smaller batches

**Issue: "Permission denied" errors when deleting working files**
- Set `delete_temp_working_dir: false` to preserve working files
- Close any programs that might have files open (e.g., image viewers)
- Run the application with appropriate file system permissions

**Issue: Missing or incorrect page numbers in summaries**
- Verify that page numbers are visible in the source document
- Check transcription output for `<page_number>X</page_number>` tags
- Ensure pages actually contain page numbers in headers/footers

**Getting Help:**

If you encounter issues not covered here:
1. Check the log files in the `_working_files/` directory for detailed error messages
2. Review the OpenAI API status page for service disruptions
3. Enable debug logging by modifying `modules/logger.py`
4. Open an issue on the project repository with relevant log excerpts

## Contributing

Contributions to AutoExcerpter are welcome! Whether you are fixing bugs, adding features, improving documentation, or suggesting enhancements, your input is valuable.

**Development Best Practices:**

- **Code Style**: Follow PEP 8 guidelines for Python code
- **Type Hints**: Use type annotations for function signatures
- **Documentation**: Add docstrings to all public functions and classes
- **Error Handling**: Include robust error handling with informative messages
- **Testing**: Write unit tests for new functionality
- **Modularity**: Maintain clear separation of concerns with well-defined interfaces
- **Logging**: Use the logger module for debugging information

**Contribution Workflow:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes with clear, descriptive commit messages
4. Test your changes thoroughly
5. Update documentation as needed
6. Submit a pull request with a detailed description of your changes

**Areas for Contribution:**

- Support for additional output formats (Markdown, HTML, etc.)
- Integration with other AI providers (Anthropic, Google, etc.)
- Enhanced image preprocessing algorithms
- Batch API integration for cost-optimized processing
- Additional unit and integration tests
- Performance optimizations
- User interface improvements

## License

MIT License

Copyright (c) 2025 Paul Goetz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
