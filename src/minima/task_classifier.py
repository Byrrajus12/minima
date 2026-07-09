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

    # Code/debug checks come first so prompts like
    # "write a function to extract dates" do not get misrouted to NER,
    # and "handle negative numbers" does not get misrouted to sentiment.
    if re.search(
        r"\b(debug|bug|traceback|exception|fix this code|fix the following|"
        r"correct this|returns wrong|expected output|actual output|wrong result|"
        r"why does this fail|why is this not working)\b",
        text,
    ):
        return "code_debugging"
    if _looks_like_code(text) and re.search(r"\b(fix|bug|wrong|error|debug|correct)\b", text):
        return "code_debugging"

    if re.search(
        r"\b(write|implement|create|generate|build)\b.*\b(function|script|class|code|program)\b",
        text,
    ):
        return "code_generation"
    if re.search(
        r"\b(write a function|create a function|implement a function|write a script|"
        r"build a class|write code|generate code)\b",
        text,
    ):
        return "code_generation"

    if re.search(r"\b(sentiment|classify.*positive|classify.*negative|classify.*neutral)\b", text):
        return "sentiment"
    if re.search(r"\b(is|was|are|were|seems|sounds)\b.*\b(positive|negative|neutral|mixed)\b", text):
        return "sentiment"

    if re.search(
        r"\b(summarize|briefly summarize|summary|tl;dr|condense|one sentence|"
        r"bullet points|key points|key takeaways|main takeaway)\b",
        text,
    ):
        return "summarization"

    if re.search(
        r"\b(named entities|entities|extract names|extract all|list the names)\b",
        text,
    ):
        return "ner"
    if re.search(
        r"\b(extract|list|identify|find|pull out)\b.*\b("
        r"people|persons?|orgs?|organizations?|locations?|dates?)\b",
        text,
    ):
        return "ner"
    if re.search(
        r"\b(people|persons?|orgs?|organizations?|locations?|dates?) mentioned\b",
        text,
    ):
        return "ner"

    if re.search(
        r"\b(sum|calculate|compute|solve|solve for|evaluate|percentage|average|"
        r"total cost|product|difference|quotient|multiplied|divided|plus|minus|"
        r"times|costs?|how much|how many)\b",
        text,
    ):
        return "math"

    if re.search(
        r"\b(if all|if no|if the|therefore|deduce|logic|true or false|"
        r"which statement|taller than|shorter than|contains only|can it be|"
        r"constraints|exactly one|necessarily follows)\b",
        text,
    ):
        return "logic"

    return "factual"
