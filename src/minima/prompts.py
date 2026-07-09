"""Short prompt suffixes for Fireworks calls."""

from __future__ import annotations


CATEGORY_CONFIG = {
    "sentiment": {
        "suffix": "Return only one label: positive, negative, neutral, or mixed. Use mixed when both positive and negative sentiment are substantial.",
        "max_tokens": 120,
    },
    "ner": {
        "suffix": "List entities only as TYPE: value, one per line. TYPES: PERSON, ORG, LOCATION, DATE. No prose.",
        "max_tokens": 260,
    },
    "summarization": {
        "suffix": "Follow any requested length, count, and format exactly. Otherwise give a concise summary only.",
        "max_tokens": 220,
    },
    "factual": {
        "suffix": "Give the shortest correct answer. No explanation unless requested.",
        "max_tokens": 300,
    },
    "math": {
        "suffix": "Give the final answer only unless steps are requested.",
        "max_tokens": 400,
    },
    "logic": {
        "suffix": "Give the final answer only unless explanation is requested.",
        "max_tokens": 420,
    },
    "code_debugging": {
        "suffix": "Return corrected code or the minimal fix only. Explain only if requested.",
        "max_tokens": 520,
    },
    "code_generation": {
        "suffix": "Return code only unless explanation is requested.",
        "max_tokens": 520,
    },
    "unknown": {
        "suffix": "Give a concise direct answer and preserve any requested format.",
        "max_tokens": 300,
    },
}


def suffix_for_category(category: str) -> str:
    config = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["unknown"])
    return str(config["suffix"])


def max_tokens_for_category(category: str) -> int:
    config = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["unknown"])
    return int(config["max_tokens"])


def build_user_prompt(category: str, prompt: str) -> str:
    return f"{prompt}\n\n{suffix_for_category(category)}"
