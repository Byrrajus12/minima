"""Compact prompts for Fireworks calls."""

from __future__ import annotations


CATEGORY_HINTS = {
    "factual": "Answer directly in 1-3 concise sentences.",
    "math": "Give the final answer with minimal necessary explanation. Include units when implied.",
    "sentiment": "Return one label (positive, negative, neutral, or mixed) plus a brief justification.",
    "summarization": "Summarize faithfully. Honor any requested length or format.",
    "ner": "Extract named entities with concise labels such as person, organization, location, date.",
    "code_debugging": "Provide corrected code first. Add a short bug note only if helpful.",
    "logic": "Give the conclusion with minimal reasoning. Satisfy all constraints.",
    "code_generation": "Provide correct, well-structured code first. Avoid extra prose unless requested.",
    "unknown": "Answer accurately and concisely. Preserve any requested format.",
}


def build_user_prompt(category: str, prompt: str) -> str:
    hint = CATEGORY_HINTS.get(category, CATEGORY_HINTS["unknown"])
    return f"{hint}\n\nTask:\n{prompt}\n\nAnswer only; no JSON wrapper."
