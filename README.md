# AutoExcerpter

AutoExcerpter is an automated transcription and summarization pipeline for PDF documents and image collections. The tool leverages OpenAI's language models (default: gpt-5-mini) to extract text from scanned documents and optionally generate structured summaries with citations, making it ideal for academic research, document digitization, and literature review workflows.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

AutoExcerpter processes PDF files or folders of images through a two-stage pipeline. First, it transcribes each page using OpenAI's vision-enabled models with optical character recognition (OCR) capabilities. Second, it optionally generates structured summaries with bullet points and bibliographic citations. The pipeline is designed for high-volume processing with concurrent request handling, adaptive rate limiting, and robust error recovery.

**Use Cases:**

- Digitizing and transcribing scanned academic papers or books
- Extracting structured summaries from research literature
- Converting image-based documents into searchable text
- Building searchable document databases from archival materials
- Automating literature review processes

## Key Features

**Transcription Capabilities:**

- Processes PDF files (automatically extracts pages as images) or existing image folders
- Supports multiple image formats: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP
- Preserves document structure: headers, footers, page numbers, footnotes, tables
- Handles mathematical equations (converts to LaTeX format)
- Maintains multi-column layouts and line breaks
- Provides detailed image/diagram descriptions for visual elements
- Uses strict JSON schemas to ensure consistent output format

**Summarization Capabilities:**

- Generates concise bullet-point summaries for each page
- Extracts full bibliographic citations in APA format
- Identifies pages without semantic content (reference lists, title pages, etc.)
- Accurately tracks page numbers from document headers/footers
- Outputs summaries as formatted DOCX documents

**Performance and Reliability:**

- Concurrent processing with configurable parallelism levels
- Adaptive rate limiting to stay within OpenAI API quotas
- Automatic retry logic with exponential backoff for transient errors
- Support for OpenAI Flex tier to reduce processing costs
- Progress tracking with estimated time of completion
- Comprehensive logging for debugging and audit trails

**Architecture Highlights:**

- Modular design with clear separation of concerns
- YAML-based configuration with validation and sensible defaults
- Base classes for shared API logic to eliminate code duplication
- Testable components with well-defined interfaces
- Public API definitions via `__all__` exports

## How It Works

1. **Input Selection**: The tool scans your input directory for PDF files and image folders, presenting an interactive menu to select which items to process.

2. **Page Extraction**: For PDFs, each page is extracted as a high-resolution image (configurable DPI). Image folders are processed directly.

3. **Transcription**: Each image is sent to OpenAI's vision API with a detailed system prompt and JSON schema that instructs the model to:
   - Perform verbatim transcription with markdown formatting
   - Preserve all structural elements (headers, footers, tables)
   - Convert equations to LaTeX
   - Mark page numbers with special tags
   - Describe images and diagrams

4. **Summarization** (Optional): Transcribed text is sent to OpenAI's API again with a summarization prompt that extracts:
   - Concise bullet points of main ideas
   - Full bibliographic citations
   - Page number metadata
   - Flags for pages without semantic content

5. **Output Generation**: The pipeline produces:
   - Plain text file with all transcriptions
   - DOCX file with formatted summaries (if enabled)
   - JSON log files with detailed processing metadata

## Prerequisites

**System Requirements:**

- Python 3.10 or higher
- Operating System: Windows, macOS, or Linux
- Sufficient disk space for temporary image files (varies by document size)

**API Access:**

- OpenAI API key with access to vision-enabled models (gpt-5-mini or compatible)
- API credits sufficient for your processing volume (see cost estimation below)

**Cost Estimation:**

Processing costs depend on your document length and OpenAI pricing. As a rough estimate:
- Transcription: ~$0.50-2.00 per 100 pages (varies by image complexity and resolution)
- Summarization: Additional ~$0.20-0.80 per 100 pages
- Using Flex tier can reduce costs significantly (lower priority, longer processing time)

## Installation

**Step 1: Clone or Download the Repository**

```bash
git clone https://github.com/yourusername/AutoExcerpter.git
cd AutoExcerpter
```

**Step 2: Create a Virtual Environment (Recommended)**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- `openai>=1.44.0` - OpenAI API client
- `pillow>=10.3.0` - Image processing
- `PyMuPDF>=1.24.0` - PDF manipulation
- `tqdm>=4.66.0` - Progress bars
- `python-docx>=1.1.0` - DOCX generation
- `PyYAML>=6.0.1` - Configuration parsing

**Step 4: Set Up Your OpenAI API Key**

The application requires your OpenAI API key as an environment variable.

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

**macOS/Linux (Bash/Zsh):**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

For persistent configuration, add the export command to your shell profile file (`.bashrc`, `.zshrc`, or PowerShell profile).

## Configuration

AutoExcerpter uses YAML-based configuration files located in `modules/config/`. The system validates all configuration values and provides sensible defaults for missing or invalid entries.

**Quick Configuration:**

Edit `modules/config/app.yaml` to customize basic settings:

```yaml
# Feature toggle
summarize: true  # Set to false for transcription only

# Input/Output Paths
input_folder_path: 'C:\Users\yourname\Documents\PDFs'
output_folder_path: 'C:\Users\yourname\Documents\Output'

# Performance Settings
concurrent_requests: 4  # Number of parallel API requests
api_timeout: 320  # Timeout per API request (seconds)

# OpenAI Configuration
openai:
  model: 'gpt-5-mini'  # Model for summarization
  transcription_model: 'gpt-5-mini'  # Model for transcription
  use_flex: true  # Use Flex tier for cost savings
```

**Configuration Files:**

- **`app.yaml`**: Main application settings (I/O paths, API configuration, feature toggles)
- **`concurrency.yaml`**: Concurrency limits and service tier selection
- **`image_processing.yaml`**: Image preprocessing settings (DPI, JPEG quality, detail level)

For detailed configuration options, see the [Advanced Configuration](#advanced-configuration) section below.

## Usage

**Basic Usage:**

Run the application from the command line:

```bash
python main.py
```

The application will:
1. Scan the default input folder (configured in `app.yaml`)
2. Display an interactive menu of available PDFs and image folders
3. Prompt you to select which items to process
4. Process selected items with progress tracking
5. Save outputs to the configured output folder

**Custom Input Path:**

Process a specific file or directory:

```bash
# Process a single PDF
python main.py --input "C:\Users\yourname\Documents\paper.pdf"

# Process a specific folder
python main.py --input "C:\Users\yourname\Documents\ScannedImages"
```

**Interactive Selection Examples:**

When prompted, you can select items using various formats:

```
Enter your choice(s) (e.g., 1; 3-5; all): 

# Process item 1 only
1

# Process items 3, 4, and 5
3-5

# Process items 1 and 5
1; 5

# Process items 2, 3, and 7 through 10
2; 3; 7-10

# Process all items
all
```

**Transcription-Only Mode:**

To skip summarization and only transcribe documents:

1. Set `summarize: false` in `modules/config/app.yaml`
2. Run the application as normal

This mode is faster and less expensive, ideal for creating searchable text archives.

## Output Files

AutoExcerpter generates several output files for each processed item:

**Main Outputs:**

1. **`<name>.txt`**: Plain text transcription
   - Contains verbatim transcription of all pages
   - Includes markdown formatting (headings, bold, italic)
   - Mathematical equations in LaTeX format
   - Page numbers marked with `<page_number>X</page_number>` tags
   - Footnotes formatted as `[^1]: Reference text`
   - Processing metadata header (total time, page count, source file)

2. **`<name>_summary.docx`**: Formatted summary document (if summarization enabled)
   - Structured by page with clear headings
   - Bullet points of main ideas and findings
   - Bibliographic citations in APA format
   - Excludes pages without semantic content (reference lists, title pages)
   - Professional formatting suitable for documentation

**Working Files:**

Located in `<name>_working_files/` subdirectory (deleted after processing if `delete_temp_working_dir: true`):

3. **`images/`**: Temporary folder with extracted page images (for PDFs)

4. **`<name>_transcription_log.json`**: Detailed transcription log
   - Metadata: input path, model name, extraction DPI, total pages
   - Per-page entries with original text, processing time, status
   - Error messages for failed pages
   - Useful for debugging and quality assurance

5. **`<name>_summary_log.json`**: Detailed summarization log (if enabled)
   - Per-page summary data with timestamps
   - Error tracking for failed summaries
   - Page number verification information

## Project Structure

The project is organized into clear, modular components:

```
AutoExcerpter/
├── api/                          # OpenAI API clients
│   ├── base_openai_client.py    # Base class with shared retry logic and error handling
│   ├── openai_api.py             # Summary generation client
│   ├── openai_transcribe_api.py # Transcription client
│   └── rate_limiter.py           # Adaptive rate limiting to stay within API quotas
├── core/
│   └── transcriber.py            # Main orchestration logic for the processing pipeline
├── processors/
│   ├── pdf_processor.py          # PDF extraction utilities (page-to-image conversion)
│   └── file_manager.py           # Output file management (TXT, DOCX, JSON logging)
├── modules/
│   ├── config/                   # YAML configuration files
│   │   ├── app.yaml              # Main application settings
│   │   ├── concurrency.yaml      # Concurrency and service tier settings
│   │   └── image_processing.yaml # Image preprocessing configuration
│   ├── prompts/                  # System prompts for OpenAI models
│   │   ├── transcription_system_prompt.txt
│   │   └── summary_system_prompt.txt
│   ├── schemas/                  # JSON schemas for structured outputs
│   │   ├── transcription_schema.json
│   │   └── summary_schema.json
│   ├── app_config.py             # Configuration loader with validation
│   ├── config_loader.py          # YAML config utilities
│   ├── image_utils.py            # Image preprocessing and format handling
│   ├── prompt_utils.py           # Prompt rendering utilities
│   └── logger.py                 # Logging configuration
├── tests/
│   └── test_prompt_utils.py     # Unit tests
├── main.py                       # Entry point and CLI interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

**Module Responsibilities:**

- **`api/`**: Encapsulates all OpenAI API interactions with retry logic, rate limiting, and error handling
- **`core/`**: Contains the main `ItemTranscriber` class that orchestrates the entire processing pipeline
- **`processors/`**: Handles file I/O operations (PDF extraction, DOCX creation, text writing)
- **`modules/`**: Provides configuration management, utilities, and resources (prompts, schemas)
- **`main.py`**: CLI interface with input scanning, user selection, and high-level workflow coordination

## Advanced Configuration

**Concurrency Settings** (`modules/config/concurrency.yaml`):

```yaml
concurrency:
  transcription:
    concurrency_limit: 150  # Max concurrent transcription requests
    delay_between_tasks: 0.05  # Delay between starting tasks (seconds)
    service_tier: flex  # Options: auto, default, flex, priority
  image_processing:
    concurrency_limit: 24  # Max concurrent image preprocessing tasks
```

**Service Tiers:**
- **`auto`**: Let OpenAI choose the best tier automatically
- **`default`**: Standard processing speed and cost
- **`flex`**: Lower cost, longer processing time (recommended for large batches)
- **`priority`**: Faster processing, higher cost (for time-sensitive work)

**Image Processing Settings** (`modules/config/image_processing.yaml`):

```yaml
api_image_processing:
  target_dpi: 300  # PDF rasterization DPI (higher = better quality, larger files)
  jpeg_quality: 95  # JPEG compression quality (1-100)
  llm_detail: auto  # Options: low, high, auto
```

**LLM Detail Levels:**
- **`low`**: Faster processing, lower cost, reduced accuracy
- **`high`**: Slower processing, higher cost, improved accuracy
- **`auto`**: Automatically choose based on image characteristics

**Rate Limiting** (`modules/config/app.yaml`):

The rate limiter uses a sliding window approach to prevent API quota violations:

```yaml
openai:
  rate_limits:
    - [120, 1]      # Max 120 requests per 1 second
    - [15000, 60]   # Max 15,000 requests per minute
    - [15000, 3600] # Max 15,000 requests per hour
```

Adjust these values based on your OpenAI tier limits.

**Customizing Prompts and Schemas:**

Advanced users can modify the system prompts and JSON schemas to customize transcription and summarization behavior:

- Edit `modules/prompts/transcription_system_prompt.txt` to change transcription instructions
- Edit `modules/prompts/summary_system_prompt.txt` to change summarization instructions
- Modify `modules/schemas/transcription_schema.json` to change transcription output structure
- Modify `modules/schemas/summary_schema.json` to change summary output structure

**Note**: Schema modifications require understanding of OpenAI's structured output format. Ensure schemas remain valid and compatible with the `strict: true` mode.

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

This project is provided as-is for educational and research purposes. Please ensure compliance with OpenAI's Terms of Service when using this tool, particularly regarding API usage and data handling.

**Important Notes:**

- This tool sends document images to OpenAI's API for processing
- Ensure you have the right to process and transmit your documents
- Be mindful of sensitive or confidential information in your documents
- Review OpenAI's data usage policies for your use case

For questions about licensing or usage rights, please contact the repository maintainer.
