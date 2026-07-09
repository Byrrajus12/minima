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


def _looks_like_code(text: str) -> bool:
    return bool(
        re.search(r"```|def\s+\w+\s*\(|class\s+\w+|for\s+\w+\s+in|return\s+", text)
        or re.search(r"\b(function|script|program|method|python|javascript)\b", text)
    )


def classify_task(prompt: str) -> str:
    text = prompt.lower()

    if re.search(r"\b(sentiment|positive|negative|neutral)\b", text):
        return "sentiment"
    if re.search(
        r"\b(summarize|summary|tl;dr|condense|one sentence|bullet points|key points)\b",
        text,
    ):
        return "summarization"
    if re.search(
        r"\b(named entities|entities|extract names|people|persons?|orgs?|"
        r"organizations?|locations?|dates?)\b",
        text,
    ):
        return "ner"
    if re.search(r"\b(debug|bug|traceback|exception|fix this code|why does this fail)\b", text):
        return "code_debugging"
    if _looks_like_code(text) and re.search(r"\b(fix|bug|wrong|error|debug|correct)\b", text):
        return "code_debugging"
    if re.search(
        r"\b(write|implement|create|generate)\b.*\b(function|script|class|code|program)\b",
        text,
    ):
        return "code_generation"
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
