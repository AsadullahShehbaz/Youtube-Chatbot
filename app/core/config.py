# app/core/config.py

from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Get API keys from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# Optional safety check
if not OPENROUTER_API_KEY or not OPENROUTER_BASE_URL:
    raise ValueError("⚠️ Missing OpenRouter API credentials in .env file")
print("✅ OpenRouter API credentials loaded successfully")