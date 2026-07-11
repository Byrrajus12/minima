"""Prompt helpers for Fireworks calls."""

from __future__ import annotations

import os


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


COMPACT_CATEGORY_CONFIG = {
    "sentiment": {
        "suffix": "Label first: positive, negative, neutral, or mixed.",
        "max_tokens": 24,
    },
    "ner": {
        "suffix": "Format: text - TYPE. Types PERSON ORG LOCATION DATE.",
        "max_tokens": 120,
    },
    "summarization": {
        "suffix": "Follow exactly. No extra facts.",
        "max_tokens": 120,
    },
    "factual": {
        "suffix": "Answer only.",
        "max_tokens": 48,
    },
    "math": {
        "suffix": "Solve. End with Answer: <value>.",
        "max_tokens": 96,
    },
    "logic": {
        "suffix": "Answer only. Brief reasoning only if needed.",
        "max_tokens": 120,
    },
    "code_debugging": {
        "suffix": "Return fixed code only.",
        "max_tokens": 220,
    },
    "code_generation": {
        "suffix": "Return code only.",
        "max_tokens": 220,
    },
    "unknown": {
        "suffix": "Answer only.",
        "max_tokens": 120,
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


COMPACT_SYSTEM_PROMPTS = {
    "factual": "Answer only.",
    "math": "Solve. End with Answer: <value>.",
    "sentiment": "Label first: positive, negative, neutral, or mixed.",
    "summarization": "Follow exactly. No extra facts.",
    "ner": "Format: text - TYPE. Types PERSON ORG LOCATION DATE.",
    "code_debugging": "Return fixed code only.",
    "logic": "Answer only. Brief reasoning only if needed.",
    "code_generation": "Return code only.",
    "unknown": "Answer only.",
}


LOCAL_CATEGORY_CONFIG = {
    "factual": {
        "system": "English only. Give a direct concise answer. No preamble. Follow the user's requested format. If the request needs current or live information, say it requires up-to-date lookup.",
        "max_tokens": 96,
        "format": "direct",
    },
    "math": {
        "system": "English only. Calculate before responding. Return the complete requested final values first. No long derivation, no examples, no JSON. Output exactly this style: FINAL: complete requested values with labels and units.",
        "max_tokens": 128,
        "format": "FINAL: ...",
    },
    "sentiment": {
        "system": "English only. Output exactly one sentiment label and one concise reason. Use mixed when substantial positive and negative evaluation are both present. Format:\nLABEL: positive|negative|neutral|mixed\nREASON: one concise sentence",
        "max_tokens": 48,
        "format": "LABEL/REASON",
    },
    "summarization": {
        "system": "English only. Return only the requested summary. No wrapper, no label, no JSON, no preamble. Obey requested length, structure, chronology, and format. Do not add facts.",
        "max_tokens": 120,
        "format": "summary only",
    },
    "ner": {
        "system": "English only. Extract named entities using exact source spans, one entity per line. Format: exact source span | PERSON/ORG/LOCATION/DATE. Include relative dates and times. Do not expand PERSON spans with titles unless the title is part of the requested span. Separate locations from institutions or venues. Do not merge adjacent entities. Do not invent spans. Use only allowed types.",
        "max_tokens": 128,
        "format": "span | TYPE",
    },
    "code_debugging": {
        "system": "English only. Provide the corrected Python code. Include concise bug identification only if the user requested it. No unnecessary explanation.",
        "max_tokens": 256,
        "format": "corrected code",
    },
    "logic": {
        "system": "English only. Check all constraints before answering. Output the complete concise conclusion first. No long derivation. Answer every requested part. Format: ANSWER: complete concise conclusion",
        "max_tokens": 72,
        "format": "ANSWER: ...",
    },
    "code_generation": {
        "system": "English only. Return Python code only. No JSON, no markdown fences, no prose after code. Preserve the requested name, signature, return type, and constraints.",
        "max_tokens": 256,
        "format": "code only",
    },
    "unknown": {
        "system": "English only. Answer concisely and follow the user's requested format.",
        "max_tokens": 120,
        "format": "direct",
    },
}


def _remote_compact_enabled() -> bool:
    return os.getenv("MINIMA_REMOTE_COMPACT") == "1"


def _category_config(category: str) -> dict[str, object]:
    config = COMPACT_CATEGORY_CONFIG if _remote_compact_enabled() else CATEGORY_CONFIG
    return config.get(category, config["unknown"])


def _system_prompts() -> dict[str, str]:
    return COMPACT_SYSTEM_PROMPTS if _remote_compact_enabled() else SYSTEM_PROMPTS


def suffix_for_category(category: str) -> str:
    config = _category_config(category)
    return str(config["suffix"])


def max_tokens_for_category(category: str) -> int:
    config = _category_config(category)
    return int(config["max_tokens"])


def build_user_prompt(category: str, prompt: str) -> str:
    return f"{prompt}\n\n{suffix_for_category(category)}"


def system_prompt_for_category(category: str) -> str:
    prompts = _system_prompts()
    return prompts.get(category, prompts["unknown"])


def build_chat_messages(category: str, prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt_for_category(category)},
        {"role": "user", "content": prompt},
    ]


def local_system_prompt_for_category(category: str) -> str:
    config = LOCAL_CATEGORY_CONFIG.get(category, LOCAL_CATEGORY_CONFIG["unknown"])
    return str(config["system"])


def local_max_tokens_for_category(category: str) -> int:
    config = LOCAL_CATEGORY_CONFIG.get(category, LOCAL_CATEGORY_CONFIG["unknown"])
    return int(config["max_tokens"])


def local_output_format_for_category(category: str) -> str:
    config = LOCAL_CATEGORY_CONFIG.get(category, LOCAL_CATEGORY_CONFIG["unknown"])
    return str(config["format"])
