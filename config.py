import os

# Feature toggle
SUMMARIZE = True  # When False, only transcription will be performed

# Folder Paths
INPUT_FOLDER_PATH = r"C:\Users\paulg\OneDrive\Desktop\New Literature"
OUTPUT_FOLDER_PATH = r"C:\Users\paulg\OneDrive\Desktop\New Literature"

# Cleanup Settings:
DELETE_TEMP_WORKING_DIR = True  # Delete the item's temporary working directory (including all temp files/images)

# Performance Settings
CONCURRENT_REQUESTS = 4
EXTRACTION_DPI = 250
MAX_WORKERS_PDF_EXTRACTION = 8  # For parallel PDF page extraction
API_TIMEOUT = 320  # Timeout for each API call in seconds

# Rate Limiting Configuration - New separated rate limits
# Format: List of (max_requests, time_period_seconds) tuples
SAIA_RATE_LIMITS = [(2, 1), (55, 60), (2900, 3600)]  # Default SAIA rate limits
OPENAI_RATE_LIMITS = [(120, 1), (15000, 60), (15000, 3600)] # Higher rate limits for OpenAI

# Image Pixel Count Bounds
MIN_TOTAL_PIXELS_DEFAULT = 256 * 28 * 28
MAX_TOTAL_PIXELS_DEFAULT = 1280 * 28 * 28
JPEG_QUALITY = 95

# SAIA API Configuration
SAIA_API_KEY = os.environ.get("SAIA_API_KEY")
if not SAIA_API_KEY:
    raise EnvironmentError("SAIA_API_KEY environment variable is not set")
SAIA_API_BASE_URL = "https://chat-ai.academiccloud.de/v1"
SAIA_MODEL_NAME = "qwen2.5-vl-72b-instruct"
SAIA_API_TIMEOUT = API_TIMEOUT  # Default to general API timeout

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
OPENAI_MODEL = "o4-mini"
OPENAI_API_TIMEOUT = 900  # 15 minutes for flex processing
OPENAI_USE_FLEX = True  # Enable flex processing for cost savings

IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp",
    ".jfif", ".heic", ".heif", ".svg", ".raw", ".jp2",
]

# Prompt Templates
SYSTEM_PROMPT = """You are an OCR and Image Analysis Expert.

# Task
Analyze the provided image and then transcribe all visible text while:

- Preserving original formatting, spelling, and layout
- Converting tables to markdown table format using | and - characters
- Moving column headers to before the respective text blocks; collecting them at the beginning of the page if more than one is present
- Rendering equations in LaTeX between $$ delimiters
- Transcribing handwritten text
- Noting images/diagrams with brief descriptions: [image: description]
- Indicating empty pages: [empty page]
- Acknowledging poor quality: [no transcription possible]
- Including all headers, footers, page numbers, and formatting elements
- Transcribing multi-column text column by column
- Preserving text that continues from/to other pages exactly as shown

Provide only the transcription without explanations or meta-commentary.
"""

SUMMARY_SYSTEM_PROMPT = """Summarize the provided text concisely and factually according to the JSON schema below.
The JSON schema:
{
  "name": "article_page_summary",
  "strict": true,
  "schema": {
    "$schema": "Summary_1_1_0",
    "title": "Article Page Summary",
    "type": "object",
    "properties": {
      "page_number": {
        "type": "object",
        "description": "Contains information about the page number. Includes the page number as an integer and a boolean indicating whether the page contains no visible or assigned page number. Reason hard to properly identify and format the page number.",
        "properties": {
          "page_number_integer": {
            "type": "integer",
            "description": "Page number of the text being summarized. Page numbers normally appear in running headers or footers and may be flanked by the article, chapter, monograph title or author names (e.g., "4 T. Albers et al.", "716 QUARTERLY JOURNAL OF ECONOMICS", or "22 CAPITALISM IN AMSTERDAM IN THE SEVENTEENTH CENTURY"). Academic pagination may begin at a higher number (e.g., 215 – 240) rather than 1."
          },
          "contains_no_page_number": {
            "type": "boolean",
            "description": "Set to true if the page contains no visible or assigned page number; otherwise, false."
          }
        },
        "required": [
          "page_number_integer",
          "contains_no_page_number"
        ],
        "additionalProperties": false
      },
      "contains_no_semantic_content": {
        "type": "boolean",
        "description": "Set to true if the page contains no substantial content, but only lists of references, sources, title or introductory pages, tables of contents, or similar non-semantic material."
      },
      "bullet_points": {
        "type": "array",
        "description": "Concise bullet-point statements that capture the main ideas or findings on the page. Return an empty string if contains_no_semantic_content is true",
        "items": {
          "type": "string"
        }
      },
      "references": {
        "type": "array",
        "description": "Full bibliographic citations appearing on the page, formatted in American Psychological Association (APA) style.",
        "items": {
          "type": "string"
        }
      }
    },
    "required": [
      "page_number",
      "contains_no_semantic_content",
      "bullet_points",
      "references"
    ],
    "additionalProperties": false
  }
}
"""

# JSON schema for structured summarization
SUMMARY_SCHEMA = {
  "name": "article_page_summary",
  "strict": True,
  "schema": {
    "$schema": "Summary_1_1_0",
    "title": "Article Page Summary",
    "type": "object",
    "properties": {
      "page_number": {
        "type": "object",
        "description": "Contains information about the page number. Includes the page number as an integer and a boolean indicating whether the page contains no visible or assigned page number. Reason hard to properly identify and format the page number.",
        "properties": {
          "page_number_integer": {
            "type": "integer",
            "description": "Page number of the text being summarized. Page numbers normally appear in running headers or footers and may be flanked by the article, chapter, or monograph title or author names (e.g., ""4 T. Albers et al""., ""716 QUARTERLY JOURNAL OF ECONOMICS"", or ""22 CAPITALISM IN AMSTERDAM IN THE SEVENTEENTH CENTURY""). Academic pagination may begin at a higher number (e.g., 215 – 240) rather than 1."
          },
          "contains_no_page_number": {
            "type": "boolean",
            "description": "Set to true if the page contains no visible or assigned page number; otherwise, false."
          }
        },
        "required": [
          "page_number_integer",
          "contains_no_page_number"
        ],
        "additionalProperties": False
      },
      "contains_no_semantic_content": {
        "type": "boolean",
        "description": "Set to true if the page contains no substantial content, but only lists of references, sources, title or introductory pages, tables of contents or similar non-semantic material."
      },
      "bullet_points": {
        "type": "array",
        "description": "Concise bullet-point statements that capture the main ideas or findings on the page. Return an empty string if contains_no_semantic_content is true.",
        "items": {
          "type": "string"
        }
      },
      "references": {
        "type": "array",
        "description": "Full bibliographic citations appearing on the page, formatted in American Psychological Association (APA) style.",
        "items": {
          "type": "string"
        }
      }
    },
    "required": [
      "page_number",
      "contains_no_semantic_content",
      "bullet_points",
      "references"
    ],
    "additionalProperties": False
  }
}
