import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# API configuration


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set.")


def get_api_key() -> str:

    return GROQ_API_KEY


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"

MAX_TOKENS = 4096
REQUESTS_PER_MINUTE = 30


# Data configuration


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "twin_data.json"


def load_twin_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Twin data file not found: {DATA_FILE}")

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {DATA_FILE}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Twin data must contain a JSON object.")

    required_keys = {
        "user_profile",
        "activity_history",
        "temporal_patterns",
    }

    missing_keys = required_keys - data.keys()

    if missing_keys:
        raise ValueError(
            f"Twin data is missing required fields: "
            f"{', '.join(sorted(missing_keys))}"
        )

    if not isinstance(data["user_profile"], dict):
        raise ValueError("'user_profile' must be a JSON object.")

    if not isinstance(data["activity_history"], list):
        raise ValueError("'activity_history' must be a JSON array.")

    if not isinstance(data["temporal_patterns"], dict):
        raise ValueError("'temporal_patterns' must be a JSON object.")

    return data


TWIN_DATA = load_twin_data()

USER_PROFILE = TWIN_DATA["user_profile"]
ACTIVITY_HISTORY = TWIN_DATA["activity_history"]
TEMPORAL_PATTERNS = TWIN_DATA["temporal_patterns"]
