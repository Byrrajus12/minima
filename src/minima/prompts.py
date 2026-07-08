"""Prompts for Fireworks calls."""

from __future__ import annotations


CATEGORY_HINTS = {
    "factual": "Answer the question directly and include the key fact or definition.",
    "math": "Solve carefully and provide the final numeric answer with units if needed.",
    "sentiment": "Classify as positive, negative, neutral, or mixed. Add a short reason only if the requested format allows it.",
    "summarization": "Preserve the central facts and obey exact format constraints.",
    "ner": "List all named entities with their types.",
    "code_debugging": "Provide corrected code as plain code and explain the bug briefly only if the requested format allows it.",
    "logic": "Reason through the constraints and answer clearly.",
    "code_generation": "Provide complete runnable code that satisfies the spec. Use plain code only; do not add Markdown fences or explanation unless requested.",
    "unknown": "Answer accurately and preserve any requested format.",
}

UNIVERSAL_INSTRUCTIONS = (
    "Follow the requested format exactly. Do not omit required details. "
    "For math and logic, compute and check carefully before finalizing. "
    "For summarization, obey sentence count and length constraints. "
    "For NER, extract all requested entities and label them clearly. "
    "For code, provide complete correct code as plain text without Markdown code fences unless requested. "
    "Output the final answer directly; avoid unnecessary preamble."
)


def build_user_prompt(category: str, prompt: str) -> str:
    hint = CATEGORY_HINTS.get(category, CATEGORY_HINTS["unknown"])
    return f"{UNIVERSAL_INSTRUCTIONS}\n{hint}\n\nTask:\n{prompt}\n\nAnswer only; no JSON wrapper."
