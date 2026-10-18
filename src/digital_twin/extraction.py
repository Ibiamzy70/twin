import re
from typing import List

# --------------------------------------------------
# Pattern library
# --------------------------------------------------

PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "date": r"\d{4}-\d{2}-\d{2}",
    "time": r"\b\d{1,2}:\d{2}(?:am|pm)?\b",
    "url": r"https?://[^\s]+",
    "hashtag": r"#[a-zA-Z0-9_]+",
    "mention": r"(?<![\w@])@[a-zA-Z0-9_]+\b",
    "number": r"\d+(?:\.\d+)?",
    "duration": r"\b\d+(?:\.\d+)?\s*(?:km|min|mins|minutes?|hours?|hrs?|h)\b",
}


# --------------------------------------------------
# Generic pattern extraction
# --------------------------------------------------


def extract_patterns(text: str, pattern_type: str) -> list[str]:

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if pattern_type not in PATTERNS:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    return re.findall(PATTERNS[pattern_type], text)


# --------------------------------------------------
# Profile data extraction
# --------------------------------------------------


def extract_profile_data(text: str) -> dict:

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return {
            "identifiers": {"emails": [], "phones": [], "handles": []},
            "temporal": {"dates": [], "times": [], "durations": []},
            "preferences": {"interests": []},
        }

    durations_found = extract_patterns(text, "duration")

    filtered_durations = [
        d
        for d in durations_found
        if not re.search(r"\b" + re.escape(d) + r"\s+in\b", text)
    ]

    hashtags = extract_patterns(text, "hashtag")
    interests = [tag.lstrip("#") for tag in hashtags]

    return {
        "identifiers": {
            "emails": extract_patterns(text, "email"),
            "phones": extract_patterns(text, "phone"),
            "handles": extract_patterns(text, "mention"),
        },
        "temporal": {
            "dates": extract_patterns(text, "date"),
            "times": extract_patterns(text, "time"),
            "durations": filtered_durations,
        },
        "preferences": {
            "interests": interests,
        },
    }
