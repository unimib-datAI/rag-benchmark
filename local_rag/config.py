"""Settings read from local_rag/.env."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def require(*names):
    """Exit with a clear message if some of these settings are missing."""
    if missing := [name for name in names if not globals()[name]]:
        sys.exit(f"Set {', '.join(missing)} in local_rag/.env (see local_rag/.env.example)")
