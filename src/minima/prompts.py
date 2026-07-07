"""Compact prompts for baseline Fireworks calls."""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are minima, a concise benchmark agent. Answer in English only. "
    "Return only the answer, with no logs or JSON wrapper."
)


CATEGORY_HINTS = {
    "factual": "Answer the factual question directly.",
    "math": "Solve carefully and give the final answer.",
    "sentiment": "Classify sentiment as positive, negative, neutral, or mixed, with a short reason if useful.",
    "summarization": "Summarize the text briefly and faithfully.",
    "ner": "Extract named entities and group them by type.",
    "code_debugging": "Identify the bug and provide the minimal fix.",
    "logic": "Reason step by step internally, then give the conclusion.",
    "code_generation": "Provide correct, minimal code and any essential note.",
    "unknown": "Answer the user task as accurately and concisely as possible.",
}


def build_user_prompt(category: str, prompt: str) -> str:
    hint = CATEGORY_HINTS.get(category, CATEGORY_HINTS["unknown"])
    return f"Task type: {category}\nInstruction: {hint}\n\nTask:\n{prompt}"
