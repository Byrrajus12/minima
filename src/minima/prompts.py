"""Prompts for Fireworks calls."""

from __future__ import annotations


CATEGORY_HINTS = {
    "factual": "Answer the question directly and include the key fact or definition.",
    "math": "Solve carefully and provide the final numeric answer with units if needed.",
    "sentiment": "Classify as positive, negative, neutral, or mixed. Add a short reason only if the requested format allows it.",
    "summarization": "Preserve the central facts and obey exact format constraints.",
    "ner": "List all named entities with their types.",
    "code_debugging": "Return only the corrected code unless the task explicitly asks for an explanation. Make the smallest possible fix, preserve the original structure and variable names, and use idiomatic in-place updates such as += for accumulator bugs.",
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


def build_user_prompt(
    category: str,
    prompt: str,
    retry_instruction: str | None = None,
) -> str:
    hint = CATEGORY_HINTS.get(category, CATEGORY_HINTS["unknown"])
    retry_text = f"\n{retry_instruction}" if retry_instruction else ""
    return (
        f"{UNIVERSAL_INSTRUCTIONS}{retry_text}\n{hint}\n\n"
        f"Task:\n{prompt}\n\nAnswer only; no JSON wrapper."
    )
