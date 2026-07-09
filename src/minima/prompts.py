"""Short prompt suffixes for Fireworks calls."""

from __future__ import annotations


CATEGORY_CONFIG = {
    "sentiment": {
        "suffix": "Pick exactly one sentiment label: positive, negative, neutral, or mixed. Use mixed when both positive and negative sentiment are substantial. Then give one brief justification.",
        "max_tokens": 120,
    },
    "ner": {
        "suffix": 'List each entity as "text - TYPE" where TYPE is PERSON, ORG, LOCATION, or DATE. One entity per line. No extra text.',
        "max_tokens": 260,
    },
    "summarization": {
        "suffix": "Follow the requested summary length and format exactly.",
        "max_tokens": 220,
    },
    "factual": {
        "suffix": "Answer accurately and concisely in at most three sentences.",
        "max_tokens": 300,
    },
    "math": {
        "suffix": 'Show minimal working, then end with "Answer: <result>".',
        "max_tokens": 400,
    },
    "logic": {
        "suffix": "Reason briefly, check the constraints, then clearly state the final answer.",
        "max_tokens": 420,
    },
    "code_debugging": {
        "suffix": "State the bug in one sentence, then provide only the corrected code that fulfills the stated purpose.",
        "max_tokens": 520,
    },
    "code_generation": {
        "suffix": "Return only the code, with a brief docstring if appropriate. No explanation after the code.",
        "max_tokens": 520,
    },
    "unknown": {
        "suffix": "Answer accurately and preserve any requested format.",
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
