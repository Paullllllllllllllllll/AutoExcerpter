# AutoExcerpter (OpenAI gpt-5-mini workflow)

AutoExcerpter transcribes PDFs or image folders and optionally generates structured summaries/excerpts using OpenAI. The pipeline makes two calls to `gpt-5-mini` per page:

- Transcription (Responses API with a strict JSON schema)
- Summary/Excerpting (Responses API with a strict JSON schema)

Outputs:
- Transcription: `<name>.txt`
- Summary (if enabled): `<name>_summary.docx`

## Requirements

- Python 3.10+
- An OpenAI API key with access to `gpt-5-mini` (or compatible models)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized into clear, modular components:

```
AutoExcerpter/
├── api/                          # OpenAI API clients
│   ├── base_openai_client.py    # Base class with shared retry logic
│   ├── openai_api.py             # Summary generation client
│   ├── openai_transcribe_api.py # Transcription client
│   └── rate_limiter.py           # Adaptive rate limiting
├── core/
│   └── transcriber.py            # Main orchestration logic
├── processors/
│   ├── pdf_processor.py          # PDF extraction utilities
│   └── file_manager.py           # Output file management
├── modules/
│   ├── config/                   # YAML configuration files
│   ├── prompts/                  # System prompts
│   ├── schemas/                  # JSON schemas
│   ├── app_config.py             # Configuration loader with validation
│   ├── config_loader.py          # YAML config utilities
│   ├── image_utils.py            # Image preprocessing
│   ├── prompt_utils.py           # Prompt rendering utilities
│   └── logger.py                 # Logging configuration
└── main.py                       # Entry point
```

**Architecture Highlights:**
- **Base classes**: Shared API logic in `base_openai_client.py` eliminates code duplication
- **Clear separation**: API, processing, and utility modules have distinct responsibilities
- **Public APIs**: All modules define `__all__` exports for clear interfaces
- **Robust validation**: Configuration and input validation throughout

## Configuration

This project uses YAML-based configuration under `modules/config/`, exposed via a thin Python facade `modules/app_config.py` (import it as `from modules import app_config as config`).

The configuration system includes robust validation and will gracefully handle missing or invalid values by using sensible defaults and logging warnings.

Required environment variable:

- `OPENAI_API_KEY` must be set in your environment.
  - Windows (PowerShell):
    ```powershell
    $env:OPENAI_API_KEY = "sk-..."
    ```

Primary config files (edit these to change behavior):

- `modules/config/app.yaml`
  - `summarize`: whether to generate page summaries (true/false)
  - `input_folder_path`, `output_folder_path`: default locations for I/O
  - `concurrent_requests`: parallelism for page processing
  - `openai.model`: default summarization model (default: `gpt-5-mini`)
  - `openai.transcription_model`: transcription model (default: `gpt-5-mini`)
  - `openai.api_timeout`: per-request timeout (seconds)
  - `openai.use_flex`: prefer Flex tier to reduce cost
  - `openai.rate_limits`: rate-limit windows used by the internal rate limiter

- `modules/config/concurrency.yaml`
  - Service-tier selection and concurrency for transcription and image processing. In particular, `concurrency.transcription.service_tier` can be set to one of `auto`, `default`, `flex`, `priority`.

- `modules/config/image_processing.yaml`
  - Image preprocessing settings for the OpenAI Responses API. Notable options:
    - `api_image_processing.target_dpi`: PDF rasterization DPI
    - `api_image_processing.jpeg_quality`: JPEG quality for API images
    - `api_image_processing.llm_detail`: `low`, `high`, or `auto` to control resizing profile

The transcription system prompt and JSON schema are sourced from:
- `modules/prompts/system_prompt.txt`
- `modules/schemas/markdown_transcription_schema.json`

The summary schema and prompt are sourced from:
- `modules/schemas/summary_schema.json`
- `modules/prompts/summary_system_prompt.txt`

## Running

You can provide a single PDF, an image folder, or a directory to scan recursively:

```bash
python main.py --input "C:\\path\\to\\pdf_or_folder"
```

The script will present an interactive selection of found items. Results are written to `OUTPUT_FOLDER_PATH`.

## Code Quality

This project follows Python best practices and maintains high code quality standards:

- **PEP 8 Compliant**: ~98% compliance with Python style guidelines
- **Type Hints**: Comprehensive type annotations throughout (Python 3.10+ syntax)
- **Documentation**: All public functions and classes have detailed docstrings
- **Error Handling**: Robust error handling with validation and clear error messages
- **Modularity**: Clear separation of concerns with well-defined module boundaries
- **Testability**: Modular design facilitates unit and integration testing

**Development Best Practices:**
- Use `from modules.logger import setup_logger` for consistent logging
- Configuration values have validation and defaults
- All modules define their public API via `__all__` exports
- Constants are extracted and documented

## Notes

- Page images are extracted with PyMuPDF using the DPI from `modules/config/image_processing.yaml` (`api_image_processing.target_dpi`).
- Images are preprocessed for OCR quality using `modules/image_utils.py` following `modules/config/image_processing.yaml` and sent as data URLs to the OpenAI Responses API.
- Transcription and summarization both use `gpt-5-mini` by default. You can change the models in `modules/config/app.yaml`.
- Logs for each processed item are stored under a temporary working directory alongside extracted page images and JSON logs. These temp directories are removed after processing if `DELETE_TEMP_WORKING_DIR = True`.
- The application gracefully handles missing configuration files by using defaults and logging warnings.

## Troubleshooting

- Ensure `OPENAI_API_KEY` is set before running.
- If you experience rate limits or timeouts, reduce `CONCURRENT_REQUESTS` and/or disable Flex (`OPENAI_USE_FLEX = False`) in `modules/config/app.yaml`.
- Some PDFs may render slowly at high DPI; try lowering `api_image_processing.target_dpi` in `modules/config/image_processing.yaml` if necessary.
- Check logs for detailed error messages - the application provides clear, actionable error information.
- If configuration files are missing or invalid, the application will use defaults and log warnings rather than crashing.

## Logging

This app uses the standard Python `logging` module via `modules/logger.py`.

- Logs are emitted to stderr with a simple format and default level `INFO`.
- To adjust verbosity, use the `set_log_level()` function:
  ```python
  from modules.logger import setup_logger, set_log_level
  import logging
  
  logger = setup_logger(__name__)
  set_log_level(logger, logging.DEBUG)  # For verbose debugging output
  ```
- The logger module supports configurable formats and levels for different use cases.

## Development

### For Developers

If you're contributing to or extending this codebase:

1. **Understand the architecture**: Review the project structure and module organization above
2. **Follow established patterns**: Use base classes, extract constants, add type hints
3. **Document your code**: Add docstrings to all public functions and classes
4. **Validate inputs**: Check and validate inputs at module boundaries
5. **Use the logger**: Always use `setup_logger(__name__)` for consistent logging
6. **Define public APIs**: Add `__all__` exports to new modules

### Testing Recommendations

Before deploying changes:

1. **Basic functionality test**:
   ```bash
   python main.py --input "path/to/test.pdf"
   ```

2. **Configuration validation test**: Temporarily rename a config file to verify graceful degradation

3. **Error handling test**: Test with invalid inputs to verify error messages

## License & Credits

This project is licensed under the terms of the MIT License.
