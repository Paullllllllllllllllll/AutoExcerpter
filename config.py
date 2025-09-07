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
API_TIMEOUT = 320  # Timeout for each API call in seconds

# Rate Limiting Configuration (OpenAI)
# Format: List of (max_requests, time_period_seconds) tuples
OPENAI_RATE_LIMITS = [(120, 1), (15000, 60), (15000, 3600)]

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
OPENAI_MODEL = "gpt-5-mini"  # Summary/excerpting and default OpenAI model
OPENAI_TRANSCRIPTION_MODEL = "gpt-5-mini"  # Transcription model
OPENAI_API_TIMEOUT = 900  # 15 minutes for flex processing
OPENAI_USE_FLEX = True  # Enable flex processing for cost savings
