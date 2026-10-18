import argparse
import json
import sys

import requests

from digital_twin.agent import safe_chat
from digital_twin.config import load_twin_data

DEFAULT_QUESTION = "What should I do next?"
FOLLOW_UP_QUESTION = "Give me the top 3 priorities only."
REQUIRED_KEYS = (
    "user",
    "current_context",
    "predicted_next_action",
    "missing_information",
    "response_quality",
)




def leaf_strings(obj):
    """Every string value inside the twin's data (profile values, activities, dates, patterns)."""
    if isinstance(obj, str):
        yield obj.strip().lower()
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from leaf_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from leaf_strings(value)


def report(answer: dict, data: dict) -> bool:
    """Print a short check of the answer. Returns False if the structure is wrong."""
    missing = [k for k in REQUIRED_KEYS if k not in answer]
    print("\n--- checks ---")
    print("[ok] valid JSON")
    if missing:
        print(f"[!!] missing keys: {missing}")
        return False
    print("[ok] all required keys present")

    quality = answer.get("response_quality", {})
    print(
        f"model's own flags: grounded_in_data={quality.get('grounded_in_data')}, "
        f"contains_guessing={quality.get('contains_guessing')}"
    )

    # Rough heuristic, not proof: does each supporting_data entry contain (or sit inside) a real data string?
    leaves = [s for s in leaf_strings(data) if len(s) >= 6]
    blob = " ".join(leaves)
    entries = [
        str(e)
        for a in answer.get("predicted_next_action", [])
        for e in a.get("supporting_data", [])
    ]
    traceable = sum(
        1
        for e in entries
        if e.lower() in blob or any(leaf in e.lower() for leaf in leaves)
    )
    print(
        f"supporting_data traceable to your data: {traceable}/{len(entries)} (rough check)"
    )
    return True


def ask(question, history, data):
    print(f"\nYou: {question}\n")
    result = safe_chat(question, history, identifier="demo")
    if not result.ok:
        print(f"[{result.status}] {result.message}")
        if result.retry_after:
            print(f"retry in {result.retry_after:.0f}s")
        return None
    print(json.dumps(result.reply, indent=2, ensure_ascii=False))
    for n in result.notices:
        print(f"notice: {n}")
    if not report(result.reply, data):
        sys.exit(1)
    return result.reply_text


def main():
    parser = argparse.ArgumentParser(description="Ask the digital twin a question.")
    parser.add_argument("question", nargs="*", help=f"default: {DEFAULT_QUESTION!r}")
    parser.add_argument(
        "--follow-up", action="store_true", help="send a second turn to test history"
    )
    args = parser.parse_args()
    question = " ".join(args.question) or DEFAULT_QUESTION

    try:
        data = load_twin_data()
        history = []
        reply = ask(question, history, data)
        if args.follow_up:
            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
            ]
            ask(FOLLOW_UP_QUESTION, history, data)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"\nSetup problem: {e}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nGroq returned an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
