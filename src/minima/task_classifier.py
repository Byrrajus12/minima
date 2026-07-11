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
        or re.search(
            r"\b(code|function|snippet|implementation|script|program|method|python|javascript)\b",
            text,
        )
    )


def classify_task(prompt: str) -> str:
    text = prompt.lower()

    # Code routes come first to avoid misrouting generation/debugging prompts
    # that mention entities, dates, or negative numbers.
    if re.search(
        r"\b(traceback|exception|expected output|actual output|returns wrong|"
        r"wrong result|why does this fail|why is this not working)\b",
        text,
    ):
        return "code_debugging"
    if re.search(
        r"\b(bug|debug|fix|error|incorrect|broken|wrong|correct this)\b.*"
        r"\b(code|function|snippet|implementation)\b",
        text,
    ):
        return "code_debugging"
    if re.search(
        r"\b(code|function|snippet|implementation)\b.*"
        r"\b(bug|debug|fix|error|incorrect|broken|wrong|correct)\b",
        text,
    ):
        return "code_debugging"
    # An explicit construction verb plus a function-like contract is
    # generation even when the prompt also specifies error behavior.
    if re.search(r"\b(write|implement|create|generate|build)\b", text) and (
        re.search(r"\b[A-Za-z_]\w*\s*\([^)]*\)", prompt)
        or re.search(r"\b(?:function|class|program|script|method|code)\b", text)
        or re.search(r"\b(?:returning|returns?|raise|input|output)\b", text)
    ):
        return "code_generation"
    if _looks_like_code(text) and re.search(
        r"\b(fix|bug|wrong|error|debug|incorrect|broken|correct)\b",
        text,
    ):
        return "code_debugging"

    if re.search(
        r"\b(write|implement|create|generate|build)\b.*"
        r"\b(function|class|program|script|method|code)\b",
        text,
    ):
        return "code_generation"
    if re.search(
        r"\b(write a function|create a function|implement a function|write a script|"
        r"build a class|write code|generate code)\b",
        text,
    ):
        return "code_generation"
    if re.search(r"\bdef\s+\w+\s*\(|\bfunction that\b", text):
        return "code_generation"

    if re.search(r"\b(sentiment|classify tone|classify emotion)\b", text):
        return "sentiment"
    if re.search(r"\bpositive\s+or\s+negative\b", text):
        return "sentiment"
    if re.search(r"\bpositive\b.*\bnegative\b.*\bneutral\b.*\bclassification\b", text):
        return "sentiment"
    if re.search(r"\bclassify\b.*\b(positive|negative|neutral|mixed)\b", text):
        return "sentiment"
    if re.search(
        r"\b(?:label|classify)\b.*\b(?:sentiment|review|tone|overall|positive|negative|neutral|mixed)\b",
        text,
    ):
        return "sentiment"
    if re.search(r"\bsentiment\s+(?:label|with)\b|\blabel\s+and\s+(?:justify|explain)\b", text):
        return "sentiment"
    if re.search(r"\blabel\s+and\s+(?:one\s+)?reason\b", text):
        return "sentiment"

    if re.search(
        r"\b(summarize|summarise|briefly summarize|summary|tl;dr|condense|"
        r"key points|key takeaways|main takeaway)\b",
        text,
    ):
        return "summarization"
    if re.search(r"\bin\s+(one|two|three|\d+)\s+(sentence|sentences|words)\b", text):
        return "summarization"
    if re.search(r"\bexactly\s+(one|two|three|\d+)\s+(sentence|sentences|words|bullets)\b", text):
        return "summarization"

    if re.search(
        r"\b(named entities|entities|ner)\b",
        text,
    ):
        return "ner"
    if re.search(
        r"\b(extract|list|identify|find|label)\b.*\b("
        r"people|persons?|orgs?|organizations?|locations?|dates?|names)\b",
        text,
    ):
        return "ner"
    if re.search(
        r"\b(people|persons?|orgs?|organizations?|locations?|dates?) mentioned\b",
        text,
    ):
        return "ner"
    if re.search(r"\b(?:extract|identify|find|list)\b.*\b(?:complete\s+)?spans?\b", text):
        return "ner"

    if re.search(r"\bdo not calculate\b|\bno calculation\b", text):
        return "factual"
    if re.search(r"\b(?:difference between|compare)\b.*\b(?:authentication|authorization|definition|concept|system|method)\b", text):
        return "factual"

    if re.search(
        r"\b(sum|calculate|compute|solve|solve for|evaluate|percentage|average|"
        r"total cost|product|difference|quotient|multiplied|divided|plus|minus|"
        r"times|costs?|how much|how many|percent|profit|discount|compound|interest|"
        r"sale price|elapsed time|duration|split equally|ratio|proportion)\b",
        text,
    ):
        return "math"
    if re.search(r"\$\d+(?:\.\d+)?\s+for\s+\d+\s+\w+.*\b(unit price|per)\b", text):
        return "math"
    if re.search(r"\brectangle\b.*\b(length|width)\b.*\b(area|perimeter)\b", text):
        return "math"
    if re.search(r"\bfrom\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+to\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", text):
        return "math"
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", text):
        return "math"

    if re.search(
        r"\b(who sits|who is|who owns|who lives|seated|puzzle|deduce|constraint|"
        r"constraints|exactly one|necessarily follows|knights and knaves|"
        r"truth-teller|liar)\b",
        text,
    ):
        return "logic"
    if re.search(
        r"\b(one each|each.*different|all-different|unique (?:order|solution)|"
        r"seats? (?:are )?numbered|immediately (?:left|right)|not adjacent|"
        r"complete order|uniquely determined|which (?:key|shape|box)\b)",
        text,
    ):
        return "logic"
    if re.search(r"\bif\b.+\bthen\b.+\bis\b.+\?", text):
        return "logic"
    if re.search(r"\bif\s+(?:all|no)\b.+\b(?:is|are|can)\b.+\?", text):
        return "logic"
    if re.search(r"\bif\b.+\b(?:requires?|implies?|then)\b.+\b(?:does|is|can)\b.+\?", text):
        return "logic"
    if re.search(r"\bneed not differ\b|\bwhat does\s+[A-Z]?\w+\s+choose\b", prompt, re.I):
        return "logic"
    if re.search(r"\b(all|every|no|some)\b.*\b(therefore|which|who|what|can|is|are)\b", text):
        return "logic"

    if _looks_like_code(text):
        return "code_debugging"

    return "factual"
