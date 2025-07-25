import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Load from environment or use defaults
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './clips'))
TEMP_DIR = Path(os.getenv('TEMP_DIR', './temp'))
# Whisper model size (options: tiny, base, small, medium, large-v2)
# Using medium model for better performance while maintaining good accuracy
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'medium')
# YOUTUBE_COOKIES_CONTENT is no longer used as we've switched to pytube
YOUTUBE_USER_AGENT = os.getenv('YOUTUBE_USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

if "YOUR_API_KEY_HERE" in GEMINI_API_KEY:
    print("⚠️ WARNING: Please replace 'YOUR_API_KEY_HERE' with your actual Google AI Studio API key in your .env file.")
