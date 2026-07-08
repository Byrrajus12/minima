"""Deterministic high-confidence local answers for simple tasks."""

from __future__ import annotations

import re


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

ORG_SUFFIXES = (
    "Labs",
    "Bank",
    "School",
    "Museum",
    "Robotics",
    "University",
    "Corp",
    "Inc",
    "LLC",
)

POSITIVE_TERMS = {
    "excellent",
    "fixed",
    "great",
    "improved",
    "love",
    "loved",
    "quick",
    "resolved",
    "smooth",
    "working",
}

NEGATIVE_TERMS = {
    "awful",
    "bad",
    "broken",
    "confusing",
    "crash",
    "crashed",
    "delay",
    "delayed",
    "failed",
    "frustrating",
    "issue",
    "issues",
    "poor",
    "problem",
    "problems",
    "scratch",
    "scratched",
    "scratches",
    "slow",
    "slowly",
    "terrible",
    "unreliable",
    "wasted",
    "worse",
}

CONTRAST_TERMS = {"although", "but", "however", "though", "yet"}
NEGATION_TERMS = {"barely", "hardly", "never", "not"}


def solve_local(category: str, prompt: str) -> str | None:
    if category == "math":
        return _solve_math(prompt)
    if category == "sentiment":
        return _solve_sentiment(prompt)
    if category == "ner":
        return _solve_ner(prompt)
    if category == "logic":
        return _solve_logic(prompt)
    return None


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


def _solve_math(prompt: str) -> str | None:
    text = prompt.lower()

    arithmetic = re.search(
        r"(-?\d+(?:\.\d+)?)\s*"
        r"(\+|-|\*|/|plus|minus|times|multiplied by|divided by)\s*"
        r"(-?\d+(?:\.\d+)?)",
        text,
    )
    if arithmetic:
        left = float(arithmetic.group(1))
        operator = arithmetic.group(2)
        right = float(arithmetic.group(3))
        if operator in {"+", "plus"}:
            return _format_number(left + right)
        if operator in {"-", "minus"}:
            return _format_number(left - right)
        if operator in {"*", "times", "multiplied by"}:
            return _format_number(left * right)
        if operator in {"/", "divided by"} and right != 0:
            return _format_number(left / right)

    each = re.search(
        r"\b(?:has|contains)\s+(\d+)\s+\w+\s+with\s+(\d+)\s+\w+\s+each\b",
        text,
    )
    if each:
        return _format_number(float(each.group(1)) * float(each.group(2)))

    cost = re.search(
        r"\bcosts?\s+(\d+(?:\.\d+)?)\s+\w+,\s+how much do\s+(\d+(?:\.\d+)?)\b",
        text,
    )
    if cost:
        return _format_number(float(cost.group(1)) * float(cost.group(2)))

    average = re.search(r"\baverage of ([\d\s,.-]+)\??", text)
    if average:
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", average.group(1))]
        if len(numbers) >= 2:
            return _format_number(sum(numbers) / len(numbers))

    return None


def _solve_sentiment(prompt: str) -> str | None:
    text = prompt.lower()
    if "sentiment" not in text and "classify" not in text:
        return None

    positive = _has_sentiment_evidence(text, POSITIVE_TERMS)
    negative = _has_sentiment_evidence(text, NEGATIVE_TERMS)
    contrast = any(re.search(rf"\b{re.escape(term)}\b", text) for term in CONTRAST_TERMS)
    if contrast and positive != negative:
        return None
    if positive and negative:
        return "mixed - positive and negative signals."
    if positive:
        return "positive - positive wording."
    if negative:
        return "negative - negative wording."
    if re.search(r"\b(arrived|scheduled|located|package arrived|at noon)\b", text):
        return "neutral - factual wording."
    return None


def _has_sentiment_evidence(text: str, terms: set[str]) -> bool:
    for term in terms:
        match = re.search(rf"\b{re.escape(term)}\b", text)
        if match and not _near_negation(text, match.start()):
            return True
    return False


def _near_negation(text: str, position: int) -> bool:
    prefix = text[:position]
    words = re.findall(r"\b\w+\b", prefix)[-3:]
    return any(word in NEGATION_TERMS for word in words)


def _solve_ner(prompt: str) -> str | None:
    if not re.search(r"\b(named entities|extract named entities|entities)\b", prompt, re.I):
        return None

    text = prompt.split(":", 1)[-1].strip()
    date = _find_date(text)
    org = _find_org(text)
    person = _find_person(text, org)
    location = _find_location(text, org)

    if not all([person, org, location]):
        return None

    parts = [f"Person: {person}", f"Organization: {org}", f"Location: {location}"]
    if date:
        parts.append(f"Date: {date}")
    return "; ".join(parts)


def _find_date(text: str) -> str | None:
    month_pattern = "|".join(MONTHS)
    match = re.search(rf"\b({month_pattern})\s+\d{{1,2}}\b", text)
    return match.group(0) if match else None


def _find_org(text: str) -> str | None:
    suffix_pattern = "|".join(ORG_SUFFIXES)
    match = re.search(rf"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:{suffix_pattern}))\b", text)
    return match.group(1) if match else None


def _find_person(text: str, org: str | None) -> str | None:
    match = re.match(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
    if not match:
        return None
    person = match.group(1)
    if org and person in org:
        return None
    return person


def _find_location(text: str, org: str | None) -> str | None:
    matches = re.findall(r"\b(?:in|from|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    for match in matches:
        if org and match in org:
            continue
        if match in MONTHS:
            continue
        return match
    return None


def _solve_logic(prompt: str) -> str | None:
    text = prompt.lower().rstrip(" ?.")

    if re.search(r"\bif all\b", text) and re.search(r"\bis (?:a |an )?", text):
        match = re.search(
            r"if all (?P<class>.+?) are (?P<prop>.+?) and (?P<item>.+?) is "
            r"(?:a |an )?(?P<member>.+?), is (?P=item) (?P=prop)",
            text,
        )
        if match and _same_noun(match.group("class"), match.group("member")):
            return "yes"

    match = re.search(
        r"if no (?P<class>.+?) are (?P<excluded>.+?) and (?P<item>.+?) is "
        r"(?:a |an )?(?P<member>.+?), can (?P=item) be (?:a |an )?(?P<target>.+)",
        text,
    )
    if (
        match
        and _same_noun(match.group("class"), match.group("member"))
        and _same_noun(match.group("excluded"), match.group("target"))
    ):
        return "no"

    return None


def _same_noun(left: str, right: str) -> bool:
    clean_left = left.strip()
    clean_right = right.strip()
    return clean_left == clean_right or clean_left.rstrip("s") == clean_right.rstrip("s")
