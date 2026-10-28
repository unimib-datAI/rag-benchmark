"""config.json persistence (shared with the CLI) and DAVE connection/generation settings."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
# config.json and the DAVE state files live in STATE_DIR (a mounted folder in Docker)
STATE_DIR = Path(os.environ.get("STATE_DIR") or ROOT_DIR)
CONFIG_FILE = STATE_DIR / "config.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# retrieval type of DAVE search.mostSimilarDocuments (hybrid = dense + full-text + entities)
DAVE_RETRIEVAL_METHODS = {
    "full": "Hybrid (dense + sparse)",
    "dense": "Dense (vector)",
    "full-text": "Sparse (full-text)",
}

# deterministic defaults so runs are comparable
DAVE_GENERATION_DEFAULTS = {
    "retrieval_method": "full",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "top_k": 50,
    "repetition_penalty": 1.0,
}


def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config):
    """Atomic write (temp file + rename); chmod 600 because it holds the DAVE password."""
    fd, tmp = tempfile.mkstemp(dir=CONFIG_FILE.parent, prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        os.chmod(tmp, 0o600)
        os.replace(tmp, CONFIG_FILE)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def dave_settings(config=None):
    """Saved `dave` values, else .env / defaults from dave_client."""
    import dave_client
    saved = (config if config is not None else load_config()).get("dave", {})
    base = {
        "base_url": dave_client.NEXTJS_BASE_URL,
        "base_path": dave_client.NEXTJS_BASE_PATH,
        "username": os.environ.get("DAVE_USERNAME", ""),
        "password": os.environ.get("DAVE_PASSWORD", ""),
        **DAVE_GENERATION_DEFAULTS,
    }
    return {k: saved.get(k, v) for k, v in base.items()}
