"""Simple task classifier used for compact prompt routing."""

from __future__ import annotations

import re


CATEGORIES = (
    "factual",
    "math",
    "sentiment",
    "summarization",
    "ner",
    "code_debugging",
    "logic",
    "code_generation",
    "unknown",
)


def classify_task(prompt: str) -> str:
    text = prompt.lower()

    if re.search(r"\b(sentiment|positive|negative|neutral)\b", text):
        return "sentiment"
    if re.search(r"\b(summarize|summary|tl;dr|condense)\b", text):
        return "summarization"
    if re.search(r"\b(named entities|entities|extract names|person|organization|location)\b", text):
        return "ner"
    if re.search(r"\b(debug|bug|traceback|exception|fix this code|why does this fail)\b", text):
        return "code_debugging"
    if re.search(r"\b(write code|generate code|implement|function|class|script)\b", text):
        return "code_generation"
    if re.search(
        r"\b(sum|calculate|solve|evaluate|product|difference|quotient|"
        r"multiplied|divided|plus|minus|times|costs?|how much)\b",
        text,
    ):
        return "math"
    if re.search(
        r"\b(if all|if no|if the|therefore|deduce|logic|true or false|"
        r"which statement|taller than|shorter than|contains only|can it be)\b",
        text,
    ):
        return "logic"
    if re.search(r"\b(who|what|when|where|why|which|capital|define)\b", text):
        return "factual"

    return "unknown"
