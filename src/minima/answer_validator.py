"""Conservative answer validation before accepting a model response."""

from __future__ import annotations

import re


GLOBAL_INVALID_EXACT = {
    "unable to determine",
    "unable to determine.",
    "unknown",
    "i don't know",
    "i cannot answer",
}

GLOBAL_INVALID_SUBSTRINGS = (
    "cannot determine",
    "local test placeholder",
    "placeholder",
    "lorem ipsum",
    "todo",
)

CODE_TOKENS = (
    r"\bdef\s+\w+\s*\(",
    r"\bfor\s+.+\s+in\s+.+:",
    r"\bwhile\s+.+:",
    r"\bif\s+.+:",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\+=",
    r"\w+\s*=\s*",
    r"\bconsole\.log\s*\(",
    r"\bfunction\s+\w+\s*\(",
    r"\bpublic\b",
    r"\bprint\s*\(",
    r"[{};]",
)


def _normalized(answer: str) -> str:
    return re.sub(r"\s+", " ", answer.casefold()).strip()


def _is_global_invalid(answer: str) -> bool:
    normalized = _normalized(answer)
    if not normalized:
        return True
    if normalized in GLOBAL_INVALID_EXACT:
        return True
    return any(value in normalized for value in GLOBAL_INVALID_SUBSTRINGS)


def _has_code_like_content(answer: str) -> bool:
    return any(re.search(pattern, answer, re.IGNORECASE) for pattern in CODE_TOKENS)


def _has_entity_like_content(answer: str) -> bool:
    if re.search(r"[:,;\n]", answer):
        return True
    capitalized_spans = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
        answer,
    )
    return len(capitalized_spans) >= 2


def is_valid_answer(category: str, prompt: str, answer: str) -> bool:
    if _is_global_invalid(answer):
        return False

    stripped = answer.strip()
    if category == "sentiment":
        return re.search(r"\b(positive|negative|neutral|mixed)\b", stripped, re.I) is not None

    if category == "math":
        return re.search(r"\d+(/\d+)?|\d*\.\d+|\d+%", stripped) is not None

    if category in {"code_debugging", "code_generation"}:
        return _has_code_like_content(stripped)

    if category == "summarization":
        return len(re.findall(r"\b\w+\b", stripped)) >= 2

    if category == "ner":
        return _has_entity_like_content(stripped)

    if category == "logic":
        return not _is_global_invalid(stripped)

    return True
