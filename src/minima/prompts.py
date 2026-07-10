"""Prompt helpers for Fireworks calls."""

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


SYSTEM_PROMPTS = {
    "factual": "English only. Be concise and accurate. No preamble. Follow any requested format.",
    "math": 'English only. Solve accurately with brief necessary work. End with "Answer: <value>".',
    "sentiment": "English only. Use the requested sentiment label and give one brief reason. No preamble.",
    "summarization": "English only. Summarize faithfully. Obey any requested length, structure, and format. Do not add facts.",
    "ner": "English only. Extract every named entity. Use the requested entity types and output format exactly. No preamble.",
    "code_debugging": "English only. State the bug briefly, then provide corrected code. Follow the requested format.",
    "logic": 'English only. Reason in short numbered steps. Check every constraint. Answer every part. End with "Answer: <value>". Do not hedge.',
    "code_generation": "English only. Return correct code that follows the requested signature and constraints. Avoid extra explanation unless requested.",
    "unknown": "English only. Answer accurately and concisely. Follow any requested format.",
}


def suffix_for_category(category: str) -> str:
    config = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["unknown"])
    return str(config["suffix"])


def max_tokens_for_category(category: str) -> int:
    config = CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["unknown"])
    return int(config["max_tokens"])


def build_user_prompt(category: str, prompt: str) -> str:
    return f"{prompt}\n\n{suffix_for_category(category)}"


def system_prompt_for_category(category: str) -> str:
    return SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["unknown"])


def build_chat_messages(category: str, prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt_for_category(category)},
        {"role": "user", "content": prompt},
    ]
