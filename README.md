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

## Configuration

This project uses YAML-based configuration under `modules/config/`, exposed via a thin Python facade `modules/app_config.py` (import it as `from modules import app_config as config`).

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

## Notes

- Page images are extracted with PyMuPDF using the DPI from `modules/config/image_processing.yaml` (`api_image_processing.target_dpi`).
- Images are preprocessed for OCR quality using `modules/image_utils.py` following `modules/config/image_processing.yaml` and sent as data URLs to the OpenAI Responses API.
- Transcription and summarization both use `gpt-5-mini` by default. You can change the models in `modules/config/app.yaml`.
- Logs for each processed item are stored under a temporary working directory alongside extracted page images and JSON logs. These temp directories are removed after processing if `DELETE_TEMP_WORKING_DIR = True`.

## Troubleshooting

- Ensure `OPENAI_API_KEY` is set before running.
- If you experience rate limits or timeouts, reduce `CONCURRENT_REQUESTS` and/or disable Flex (`OPENAI_USE_FLEX = False`).
- Some PDFs may render slowly at high DPI; try lowering `api_image_processing.target_dpi` in `modules/config/image_processing.yaml` if necessary.

## Logging

This app uses the standard Python `logging` module via `modules/logger.py`.

- Logs are emitted to stdout with a simple format and default level `INFO`.
- To adjust verbosity programmatically, change the level in `modules/logger.py` or configure handlers as needed.
- Runtime examples:
  - For more verbose output during debugging, temporarily set `logger.setLevel(logging.DEBUG)` in `modules/logger.py`.
