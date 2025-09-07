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

All settings live in `config.py`.

- `OPENAI_API_KEY` must be set in your environment.
  - Windows (PowerShell):
    ```powershell
    $env:OPENAI_API_KEY = "sk-..."
    ```
- `OPENAI_TRANSCRIPTION_MODEL`: defaults to `gpt-5-mini`.
- `OPENAI_MODEL`: defaults to `gpt-5-mini` for summarization.
- `SUMMARIZE`: set to `True` to generate page summaries.
- `INPUT_FOLDER_PATH`, `OUTPUT_FOLDER_PATH`: default locations for I/O.
- Performance knobs:
  - `CONCURRENT_REQUESTS`: parallelism for page processing
  - `OPENAI_USE_FLEX`: use the Flex tier to reduce cost (longer latency tolerated)
  - `modules/config/image_processing.yaml`: image processing settings
    - `api_image_processing.target_dpi`: PDF rasterization DPI
    - `api_image_processing.jpeg_quality`: JPEG quality for API images

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
- Transcription and summarization both use `gpt-5-mini` by default. You can change the models in `config.py`.
- Logs for each processed item are stored under a temporary working directory alongside extracted page images and JSON logs. These temp directories are removed after processing if `DELETE_TEMP_WORKING_DIR = True`.

## Troubleshooting

- Ensure `OPENAI_API_KEY` is set before running.
- If you experience rate limits or timeouts, reduce `CONCURRENT_REQUESTS` and/or disable Flex (`OPENAI_USE_FLEX = False`).
- Some PDFs may render slowly at high DPI; try lowering `api_image_processing.target_dpi` in `modules/config/image_processing.yaml` if necessary.
