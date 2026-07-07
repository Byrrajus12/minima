"""Compact prompts for Fireworks calls."""

from __future__ import annotations


CATEGORY_HINTS = {
    "factual": "Answer only.",
    "math": "Final answer only; add units if needed.",
    "sentiment": "Return one label (positive, negative, neutral, or mixed) plus a brief justification.",
    "summarization": "Summarize as requested; no intro.",
    "ner": "List entities compactly as Type: value.",
    "code_debugging": "Provide corrected code first. Add a short bug note only if helpful.",
    "logic": "Answer first; minimal reason only if needed.",
    "code_generation": "Provide correct, well-structured code first. Avoid extra prose unless requested.",
    "unknown": "Answer concisely; keep requested format.",
}


def build_user_prompt(category: str, prompt: str) -> str:
    hint = CATEGORY_HINTS.get(category, CATEGORY_HINTS["unknown"])
    return f"{hint}\n\nTask:\n{prompt}\n\nAnswer only; no JSON wrapper."
