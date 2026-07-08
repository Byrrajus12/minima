"""Compact prompts for Fireworks calls."""

from __future__ import annotations


CATEGORY_HINTS = {
    "factual": "Answer directly with the key fact. Avoid extra rambling.",
    "math": "Compute carefully. Return the final answer clearly; include minimal calculation if helpful.",
    "sentiment": "Label as positive, negative, neutral, or mixed, with one short reason.",
    "summarization": "Summarize faithfully and obey any requested sentence or length constraint.",
    "ner": "Extract all named entities and label their types clearly.",
    "code_debugging": "Identify the bug and provide corrected code.",
    "logic": "Satisfy all constraints and answer clearly.",
    "code_generation": "Provide complete correct code only, with brief explanation only if requested.",
    "unknown": "Answer accurately and concisely. Preserve any requested format.",
}


def build_user_prompt(category: str, prompt: str) -> str:
    hint = CATEGORY_HINTS.get(category, CATEGORY_HINTS["unknown"])
    return f"{hint}\n\nTask:\n{prompt}\n\nAnswer only; no JSON wrapper."
