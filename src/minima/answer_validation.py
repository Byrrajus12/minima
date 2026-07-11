"""Category-aware answer normalization and objective validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
import re
import copy
from typing import Any


ALLOWED_SENTIMENT = {"positive", "negative", "neutral", "mixed"}
ALLOWED_ENTITY_TYPES = {"PERSON", "ORG", "LOCATION", "DATE"}
CORRECTION_LANGUAGE = re.compile(r"\b(wait|correction|recalculate|recheck|let me recalculate)\b", re.I)
_HONORIFIC_RE = re.compile(r"^(Dr\.|Doctor|Prof\.|Professor|Mr\.|Mrs\.|Ms\.|Miss|Sir|Dame)\s+(.+)$", re.I)


@dataclass(frozen=True)
class NormalizedAnswer:
    category: str
    raw_text: str
    answer: str
    final_section: str
    parse_status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    failure_code: str = ""
    failure_detail: str = ""
    safe_to_repair: bool = False
    requires_remote: bool = False
    verified_local: bool = True


@dataclass(frozen=True)
class TruncationResult:
    possibly_truncated: bool
    incomplete: bool
    reasons: tuple[str, ...] = ()


def normalize_answer(category: str, text: str | None, prompt: str = "") -> NormalizedAnswer:
    raw = _clean(text or "")
    if category in {"code_generation", "code_debugging"}:
        code, stripped = strip_code_fences(raw)
        return NormalizedAnswer(category, raw, code, code, "code", {"fence_stripped": stripped})
    if category == "math":
        final = _extract_block(raw, "FINAL", ())
        parse_status = "final" if final else "raw"
        if not final:
            final = raw
        return NormalizedAnswer(category, raw, final.strip(), final.strip(), parse_status)
    if category == "logic":
        answer = _extract_block(raw, "ANSWER", ())
        parse_status = "answer" if answer else "raw"
        if not answer:
            answer = raw
        return NormalizedAnswer(category, raw, answer.strip(), answer.strip(), parse_status)
    if category == "sentiment":
        label = _extract_label(raw, "LABEL")
        reason = _extract_block(raw, "REASON", ())
        answer = f"LABEL: {label.strip()}" if label else raw
        if reason:
            answer = f"{answer}\nREASON: {reason.strip()}"
        return NormalizedAnswer(category, raw, answer.strip(), label.strip(), "label" if label else "raw", {"label": label.strip(), "reason": reason.strip()})
    if category == "ner":
        ner_answer = normalize_ner_lines(raw, prompt)
        return NormalizedAnswer(
            category,
            raw,
            ner_answer,
            ner_answer,
            "ner_lines",
            {"honorific_stripped": raw.strip() != ner_answer.strip() and bool(re.search(r"^\s*(?:Dr\.|Doctor|Prof\.|Professor|Mr\.|Mrs\.|Ms\.|Miss|Sir|Dame)\s+", raw, re.I | re.M))},
        )
    return NormalizedAnswer(category, raw, raw.strip(), raw.strip(), "raw")


def validate_answer(
    category: str,
    prompt: str,
    normalized: NormalizedAnswer,
    *,
    deterministic_answer: str | None = None,
    truncated: bool = False,
    truncation: TruncationResult | None = None,
) -> ValidationResult:
    if truncation is not None:
        truncated = truncation.incomplete
    if not normalized.answer.strip():
        return ValidationResult(False, "empty", "answer is empty", False, True)
    if truncated and category in {"factual", "code_generation", "code_debugging", "ner"}:
        return ValidationResult(False, "truncated_output", "answer appears incomplete or cut off", category in {"code_debugging"}, True)
    if truncated and category in {"math", "logic"}:
        # A complete final section can still be valid, but correction language/trailing incomplete work is unsafe.
        if CORRECTION_LANGUAGE.search(normalized.raw_text):
            return ValidationResult(False, "truncated_correction", "truncated answer contains correction language", True, True)
    if category == "factual":
        if is_freshness_sensitive(prompt):
            return ValidationResult(False, "freshness_required", "factual prompt requires current information", False, True)
        return _validate_factual(prompt, normalized)
    if category == "math":
        return _validate_math(prompt, normalized, deterministic_answer=deterministic_answer, truncated=truncated)
    if category == "logic":
        return _validate_logic(prompt, normalized, deterministic_answer=deterministic_answer, truncated=truncated)
    if category == "sentiment":
        return _validate_sentiment(prompt, normalized)
    if category == "summarization":
        return _validate_summary(prompt, normalized)
    if category == "ner":
        return _validate_ner(prompt, normalized)
    if category in {"code_generation", "code_debugging"}:
        return _validate_code(prompt, normalized)
    return ValidationResult(True)


def is_freshness_sensitive(prompt: str) -> bool:
    text = prompt.casefold()
    return bool(
        re.search(
            r"\b(current|latest|today|recent|presently|currently|this year|live|price|prices|score|scores|schedule|law|laws|version|versions|office|role|president|ceo)\b",
            text,
        )
    )


def strip_code_fences(text: str) -> tuple[str, bool]:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip(), True
    return text.strip(), False


def normalize_ner_lines(text: str, prompt: str = "") -> str:
    lines: list[str] = []
    source = _ner_source(prompt) if prompt else ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            span, entity_type = line.rsplit("|", 1)
        elif " - " in line:
            span, entity_type = line.rsplit(" - ", 1)
        else:
            lines.append(line)
            continue
        entity_type = entity_type.strip().upper()
        span = _normalize_person_honorific(span.strip(), source, entity_type)
        lines.append(f"{span.strip()} | {entity_type}")
    return "\n".join(lines).strip()


def parse_ner_lines(answer: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for line in [item.strip() for item in answer.splitlines() if item.strip()]:
        if "|" not in line:
            continue
        span, entity_type = line.rsplit("|", 1)
        parsed.append((span.strip(), entity_type.strip().upper()))
    return parsed


def summary_requirements(prompt: str) -> dict[str, Any]:
    text = prompt.casefold()
    req: dict[str, Any] = {}
    bullet = re.search(r"\b(?:exactly\s+)?(one|two|three|four|five|\d+)\s+bullets?\b", text)
    if bullet:
        req["bullets"] = _number_word(bullet.group(1))
    sentence = re.search(r"\b(?:exactly\s+|in\s+)?(one|two|three|four|five|\d+)\s+sentences?\b", text)
    if sentence:
        req["sentences"] = _number_word(sentence.group(1))
    words = re.search(r"\b(?:no more than|at most|max(?:imum)? of?)\s+(one|two|three|four|five|\d+)\s+words?\b", text)
    if words:
        req["max_words"] = _number_word(words.group(1))
    if re.search(r"\bchronolog|timeline|sequence|in order\b", text):
        req["chronology"] = True
    if "decision" in text and "rationale" in text:
        req["decision_rationale"] = True
    return req


def summary_requirement_instructions(prompt: str) -> str:
    req = summary_requirements(prompt)
    instructions: list[str] = []
    if "bullets" in req:
        instructions.append(f"Return exactly {req['bullets']} bullets and no introductory text.")
    if "sentences" in req:
        instructions.append(f"Return exactly {req['sentences']} sentence{'s' if req['sentences'] != 1 else ''}.")
    if "max_words" in req:
        instructions.append(f"Use no more than {req['max_words']} words.")
    if req.get("chronology"):
        instructions.append("Keep events in chronological order.")
    if req.get("decision_rationale"):
        instructions.append("Separate the decision from the rationale.")
    return "\n".join(f"- {item}" for item in instructions)


def requested_function_name(prompt: str) -> str | None:
    patterns = (
        r"\b(?:function|method)\s+(?:named|called)\s+([A-Za-z_]\w*)\b",
        r"\b(?:write|implement|create)\s+([A-Za-z_]\w*)\s*\(",
        r"\b([A-Za-z_]\w*)\s*\([^)]*\)\s+(?:returning|that|which|to)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.I)
        if match:
            return match.group(1)
    return None


def numbers(text: str) -> list[float]:
    out: list[float] = []
    for raw in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def _validate_factual(prompt: str, normalized: NormalizedAnswer) -> ValidationResult:
    if re.search(r"\b(?:not sure|unknown|i cannot|can't determine)\b", normalized.answer, re.I):
        return ValidationResult(False, "uncertain_factual", "answer hedges or refuses", True, True)
    return ValidationResult(True)


def _validate_math(prompt: str, normalized: NormalizedAnswer, *, deterministic_answer: str | None, truncated: bool) -> ValidationResult:
    final = normalized.final_section
    if not re.search(r"-?\d|cannot|determin", final, re.I):
        return ValidationResult(False, "missing_final_value", "math final answer has no value", True, True)
    if CORRECTION_LANGUAGE.search(normalized.raw_text):
        return ValidationResult(False, "correction_language", "answer contains correction language", True, True)
    if _approximate_prompt(prompt) and re.search(r"\bexact(?:ly)?\b", final, re.I) and not re.search(r"cannot|not possible|undetermined|about|approx", final, re.I):
        return ValidationResult(False, "fabricated_exact", "asserts exact value from approximate prompt", False, True)
    if deterministic_answer and _numeric_sets_conflict(final, deterministic_answer):
        return ValidationResult(False, "deterministic_mismatch", f"final answer disagrees with deterministic result: {deterministic_answer}", True, True)
    inferred = _infer_math_values(prompt)
    if inferred and _numeric_sets_conflict(final, " ".join(str(value) for value in inferred)):
        return ValidationResult(False, "objective_math_mismatch", f"final answer disagrees with prompt-derived values: {inferred}", True, True)
    if truncated and not final.strip():
        return ValidationResult(False, "truncated_final", "truncated before final answer", True, True)
    return ValidationResult(True)


def _validate_logic(prompt: str, normalized: NormalizedAnswer, *, deterministic_answer: str | None, truncated: bool) -> ValidationResult:
    answer = normalized.final_section
    if "?" in answer:
        return ValidationResult(False, "placeholder", "logic answer contains placeholder", True, True)
    if truncated and normalized.parse_status == "raw":
        return ValidationResult(False, "truncated_logic", "logic answer is a cut-off raw reasoning trace without required ANSWER structure", True, True)
    if truncated and len(answer.split()) < 2 and "," in prompt:
        return ValidationResult(False, "truncated_logic", "logic answer appears incomplete", True, True)
    if deterministic_answer and not _logic_answers_compatible(answer, deterministic_answer):
        return ValidationResult(False, "constraint_mismatch", f"answer disagrees with deterministic result: {deterministic_answer}", True, True)
    ordering = _infer_first_last(prompt)
    if ordering:
        lowered = answer.casefold()
        first = ordering.get("first")
        last = ordering.get("last")
        if first and "first" in prompt.casefold() and first.casefold() not in lowered:
            return ValidationResult(False, "ordering_first_mismatch", f"first should be {first}", True, True)
        if last and "last" in prompt.casefold() and last.casefold() not in lowered:
            return ValidationResult(False, "ordering_last_mismatch", f"last should be {last}", True, True)
    if _ordered_assignment_prompt(prompt):
        violation = _validate_ordered_assignment(prompt, answer)
        if violation:
            return ValidationResult(False, "ordered_assignment_violation", violation, True, True)
    return ValidationResult(True)


def _validate_sentiment(prompt: str, normalized: NormalizedAnswer) -> ValidationResult:
    label_lines = re.findall(r"^\s*LABEL\s*:\s*(positive|negative|neutral|mixed)\b", normalized.raw_text, flags=re.I | re.M)
    label = (normalized.diagnostics.get("label") or "").casefold()
    if label not in ALLOWED_SENTIMENT:
        return ValidationResult(False, "invalid_label", "missing or invalid sentiment label", True, True)
    if len(label_lines) != 1:
        return ValidationResult(False, "conflicting_label", "answer must contain exactly one LABEL value", True, True)
    if re.search(r"\b(reason|justify|why)\b", prompt, re.I) and not normalized.diagnostics.get("reason"):
        return ValidationResult(False, "missing_reason", "prompt requested a reason", True, False)
    return ValidationResult(True)


def _validate_summary(prompt: str, normalized: NormalizedAnswer) -> ValidationResult:
    answer = normalized.answer
    req = summary_requirements(prompt)
    if "bullets" in req:
        bullets = len([line for line in answer.splitlines() if line.strip().startswith(("-", "*"))])
        if bullets != req["bullets"]:
            return ValidationResult(False, "bullet_count", f"expected {req['bullets']} bullets, got {bullets}", True, True)
    if "sentences" in req:
        sentences = _sentence_count(answer)
        if sentences != req["sentences"]:
            return ValidationResult(False, "sentence_count", f"expected {req['sentences']} sentences, got {sentences}", True, True)
    if "max_words" in req:
        words = len(re.findall(r"\b\w+\b", answer))
        if words > req["max_words"]:
            return ValidationResult(False, "word_count", f"expected at most {req['max_words']} words, got {words}", True, True)
    return ValidationResult(True)


def _validate_ner(prompt: str, normalized: NormalizedAnswer) -> ValidationResult:
    entities = parse_ner_lines(normalized.answer)
    if not entities:
        return ValidationResult(False, "empty_entities", "no parseable entity lines", False, True)
    seen: set[tuple[str, str]] = set()
    source = _ner_source(prompt)
    lower_source = source.casefold()
    for span, entity_type in entities:
        if entity_type not in ALLOWED_ENTITY_TYPES:
            return ValidationResult(False, "invalid_entity_type", f"{entity_type} is not allowed", False, True)
        key = (span, entity_type)
        if key in seen:
            return ValidationResult(False, "duplicate_entity", f"duplicate entity line: {span} | {entity_type}", False, False)
        seen.add(key)
        if span.casefold() not in lower_source:
            return ValidationResult(False, "span_not_in_source", f"span is not in source text: {span}", False, True)
        if entity_type == "PERSON" and _title_expanded_person(span, source):
            return ValidationResult(False, "title_expanded_person", f"title-expanded person span: {span}", False, True)
        if _looks_merged_entity(span, entity_type, source):
            return ValidationResult(False, "merged_entity", f"span appears to merge adjacent entities: {span}", False, True)
        if entity_type == "LOCATION" and _looks_org_like_venue(span):
            return ValidationResult(False, "wrong_entity_type", f"venue or institution emitted as LOCATION: {span}", False, True)
        if span.casefold() in {"person", "org", "organization", "location", "date", "entities", "named entities"}:
            return ValidationResult(False, "instruction_span", f"span appears copied from instruction: {span}", False, True)
    missing = _missing_required_ner_entities(source, entities)
    if missing:
        detail = "; ".join(f"{span} | {entity_type}" for span, entity_type in missing)
        return ValidationResult(False, "missing_required_entity", f"missing conservative required entity: {detail}", False, True)
    return ValidationResult(True)


def _validate_code(prompt: str, normalized: NormalizedAnswer) -> ValidationResult:
    code = normalized.answer
    if re.search(r"^\s*(here|sure|this code|explanation)\b", code, re.I):
        return ValidationResult(False, "prose_around_code", "answer contains prose before code", True, True)
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return ValidationResult(False, "syntax_error", f"{exc.msg} at line {exc.lineno}", True, True)
    name = requested_function_name(prompt)
    if name and not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name for node in ast.walk(tree)):
        return ValidationResult(False, "missing_requested_name", f"missing requested function/class name {name}", True, True)
    blocked = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    if any(isinstance(node, blocked) for node in ast.walk(tree)):
        return ValidationResult(False, "unsafe_construct", "imports/global/nonlocal are not accepted locally", False, True)
    if normalized.category == "code_debugging" and _prompt_has_mutable_default_bug(prompt):
        if has_mutable_default_defect(tree):
            return ValidationResult(False, "mutable_default_still_present", "corrected code still has a mutable default argument", True, True)
    if normalized.category == "code_generation":
        risk = classify_code_generation_risk(prompt)
        examples = derive_prompt_code_examples(prompt, name)
        if risk["risk"] == "high":
            return ValidationResult(
                False,
                "high_risk_code_generation",
                ", ".join(risk["reasons"]) or "code task needs behavioral verification",
                False,
                True,
                False,
            )
        if examples:
            failure = _run_prompt_examples(tree, name, examples)
            if failure:
                return ValidationResult(False, "prompt_example_failed", failure, False, True)
            return ValidationResult(True, verified_local=True)
        return ValidationResult(True, verified_local=False)
    return ValidationResult(True)


def _clean(text: str) -> str:
    for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        text = text.replace(token, "")
    return re.sub(r"^\s*assistant\s*:\s*", "", text, flags=re.I).strip()


def _extract_label(text: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.*)$", text, flags=re.I | re.M)
    return match.group(1).strip() if match else ""


def _extract_block(text: str, label: str, following_labels: tuple[str, ...]) -> str:
    if following_labels:
        labels = "|".join(re.escape(item) for item in following_labels)
        pattern = rf"^\s*{re.escape(label)}\s*:\s*(.*?)(?=^\s*(?:{labels})\s*:|\Z)"
    else:
        pattern = rf"^\s*{re.escape(label)}\s*:\s*(.*?)(?=^\s*[A-Z][A-Z _-]{{1,24}}\s*:|\Z)"
    match = re.search(pattern, text, flags=re.I | re.M | re.S)
    return match.group(1).strip() if match else ""


def _number_word(raw: str) -> int:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    return words.get(raw.casefold(), int(raw) if raw.isdigit() else 0)


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(re.findall(r"[.!?](?:\s|$)", stripped)))


def _approximate_prompt(prompt: str) -> bool:
    return bool(re.search(r"\b(about|approximately|roughly|around|near|almost)\b", prompt, re.I))


def _numeric_sets_conflict(answer: str, deterministic: str) -> bool:
    a = numbers(answer)
    d = numbers(deterministic)
    if not a or not d:
        return False
    # Require every deterministic value to appear in final answer; extra prompt numbers in answer are suspicious.
    return not all(any(abs(x - y) <= max(1e-6, abs(y) * 1e-4) for x in a) for y in d)


def _infer_math_values(prompt: str) -> list[float]:
    text = prompt.casefold()
    if re.search(r"\btravels?\b", text) and re.search(r"\bkm/h\b", text) and re.search(r"\baverage speed\b", text):
        legs = [
            (float(distance), float(speed))
            for distance, speed in re.findall(
                r"(\d+(?:\.\d+)?)\s*km\s+at\s+(\d+(?:\.\d+)?)\s*km/h",
                text,
            )
        ]
        if len(legs) >= 2 and all(speed > 0 for _, speed in legs):
            total_distance = sum(distance for distance, _ in legs)
            total_time = sum(distance / speed for distance, speed in legs)
            if total_time > 0:
                return [_round_number(total_time), _round_number(total_distance / total_time)]
    return []


def _round_number(value: float) -> float:
    rounded = round(value, 6)
    return int(rounded) if abs(rounded - int(rounded)) < 1e-9 else rounded


def _logic_answers_compatible(answer: str, deterministic: str) -> bool:
    a = re.sub(r"\s+", " ", answer.casefold())
    d = re.sub(r"\s+", " ", deterministic.casefold())
    if d in a or a in d:
        return True
    names = re.findall(r"\b[A-Z][a-z]+\b", deterministic)
    return all(name.casefold() in a for name in names) if names else False


def _ordered_assignment_prompt(prompt: str) -> bool:
    return bool(re.search(r"\b(seat|order|position|first|last|left|right|adjacent|immediately)\b", prompt, re.I))


def _validate_ordered_assignment(prompt: str, answer: str) -> str | None:
    assignments = _parse_numbered_assignments(answer)
    if not assignments:
        return None
    reverse = {name.casefold(): seat for seat, name in assignments.items()}
    prompt_lower = prompt.casefold()
    for match in re.finditer(r"([A-Z][a-z]+)\s+is\s+in\s+seat\s+(\d+)", prompt):
        name, seat = match.group(1), int(match.group(2))
        if reverse.get(name.casefold()) != seat:
            return f"{name} must be in seat {seat}"
    for match in re.finditer(r"([A-Z][a-z]+)\s+sits\s+immediately\s+left\s+of\s+([A-Z][a-z]+)", prompt):
        left, right = match.group(1), match.group(2)
        if reverse.get(right.casefold()) != reverse.get(left.casefold(), -100) + 1:
            return f"{left} must be immediately left of {right}"
    for match in re.finditer(r"([A-Z][a-z]+)\s+is\s+not\s+adjacent\s+to\s+([A-Z][a-z]+)", prompt):
        a, b = match.group(1), match.group(2)
        if a.casefold() in reverse and b.casefold() in reverse and abs(reverse[a.casefold()] - reverse[b.casefold()]) == 1:
            return f"{a} must not be adjacent to {b}"
    if "not adjacent to" in prompt_lower and len(assignments) != len(set(assignments.values())):
        return "assignment is incomplete or duplicated"
    return None


def _infer_first_last(prompt: str) -> dict[str, str]:
    edges: list[tuple[str, str]] = []
    for a, b in re.findall(r"\b([A-Z][a-z]+)\s+finishes\s+before\s+([A-Z][a-z]+)\b", prompt):
        edges.append((a, b))
    for a, b in re.findall(r"\b([A-Z][a-z]+)\s+finishes\s+after\s+([A-Z][a-z]+)\b", prompt):
        edges.append((b, a))
    if not edges:
        return {}
    names = {name for edge in edges for name in edge}
    before = {name: set() for name in names}
    after = {name: set() for name in names}
    for a, b in edges:
        before.setdefault(a, set()).add(b)
        after.setdefault(b, set()).add(a)
    changed = True
    while changed:
        changed = False
        for name in list(before):
            expanded = set(before[name])
            for child in list(before[name]):
                expanded.update(before.get(child, set()))
            if expanded != before[name]:
                before[name] = expanded
                changed = True
        for name in list(after):
            expanded = set(after[name])
            for parent in list(after[name]):
                expanded.update(after.get(parent, set()))
            if expanded != after[name]:
                after[name] = expanded
                changed = True
    result: dict[str, str] = {}
    first = [name for name in names if len(before.get(name, set())) == len(names) - 1]
    last = [name for name in names if len(after.get(name, set())) == len(names) - 1]
    if len(first) == 1:
        result["first"] = first[0]
    if len(last) == 1:
        result["last"] = last[0]
    return result


def _parse_numbered_assignments(answer: str) -> dict[int, str]:
    assignments: dict[int, str] = {}
    for seat, name in re.findall(r"\b(\d+)\s*[-:]\s*([A-Z][A-Za-z'’-]+)", answer):
        assignments[int(seat)] = name
    return assignments


def _ner_source(prompt: str) -> str:
    if ":" in prompt:
        return prompt.split(":", 1)[1].strip()
    return prompt


def _title_expanded_person(span: str, source: str) -> bool:
    match = _HONORIFIC_RE.match(span)
    if not match:
        return False
    bare = match.group(2)
    return bare in source


def _looks_merged_entity(span: str, entity_type: str, source: str) -> bool:
    if entity_type != "DATE" and "'s " in span and re.search(r"\b[A-Z][\wÀ-ÖØ-öø-ÿ'’-]+(?:\s+[A-Z][\wÀ-ÖØ-öø-ÿ'’-]+)+", span):
        return True
    # Venue/institution after a location is commonly a merged LOCATION + ORG error.
    if re.search(r"\b(Casa|Museum|University|Institute|Project|Labs|Hall|Center|Centre)\b", span) and re.search(r"'s|,\s*", span):
        return True
    return False


def _looks_org_like_venue(span: str) -> bool:
    return bool(re.search(r"\b(Casa|Museum|University|Institute|Project|Labs|Hall|Center|Centre|Network|Company|Collective)\b", span))


def _normalize_person_honorific(span: str, source: str, entity_type: str) -> str:
    if entity_type != "PERSON" or not source:
        return span
    match = _HONORIFIC_RE.match(span.strip())
    if not match:
        return span
    bare = match.group(2).strip()
    if not bare:
        return span
    if span in source and bare in source:
        return bare
    return span


def _missing_required_ner_entities(source: str, entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
    present = {(span.casefold(), entity_type) for span, entity_type in entities}
    required: set[tuple[str, str]] = set()
    name = r"[A-Z][\wÀ-ÖØ-öø-ÿ'’.-]+(?:\s+(?:de|da|del|la|le|van|von|al|bin|&|and|[A-Z][\wÀ-ÖØ-öø-ÿ'’.-]+))*"
    patterns = [
        (rf"\b(?P<person>{name})\s+joined\s+(?P<org>{name})\s+as\b", ("person", "org")),
        (rf"\b(?P<person>{name})\s+works\s+at\s+(?P<org>{name})\b", ("person", "org")),
        (rf"\b(?P<person>{name})\s+was\s+hired\s+by\s+(?P<org>{name})\b", ("person", "org")),
        (rf"\b(?P<org>{name})\s+hired\s+(?P<person>{name})\b", ("org", "person")),
        (rf"\b(?P<org>{name})\s+appointed\s+(?P<person>{name})\b", ("org", "person")),
        (rf"\b(?P<person>{name})\s+spoke\s+for\s+(?P<org>{name})\b", ("person", "org")),
        (rf"\b(?P<person>{name})\s+of\s+(?P<org>{name})\b", ("person", "org")),
    ]
    for pattern, groups in patterns:
        for match in re.finditer(pattern, source):
            for group in groups:
                span = match.group(group).strip(" ,.;:")
                if not span:
                    continue
                entity_type = "PERSON" if group == "person" else "ORG"
                if entity_type == "PERSON":
                    span = _normalize_person_honorific(span, source, entity_type)
                if entity_type == "ORG" and match.start(group) >= 2 and source[match.start(group) - 2 : match.start(group)] == "& ":
                    continue
                required.add((span, entity_type))
    for match in re.finditer(
        r"\b(?:last\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+(?:morning|afternoon|evening|night))?\b"
        r"|\b(?:tomorrow|yesterday|today)\s+(?:morning|afternoon|evening|night)\b"
        r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
        r"|(?<!\d\s)\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        source,
    ):
        required.add((match.group(0), "DATE"))
    missing = [(span, entity_type) for span, entity_type in sorted(required) if (span.casefold(), entity_type) not in present]
    return missing


def detect_truncation(
    category: str,
    prompt: str,
    normalized: NormalizedAnswer,
    *,
    finish_reason: str | None = None,
    completion_tokens: int | None = None,
    max_completion_tokens: int | None = None,
) -> TruncationResult:
    reasons: list[str] = []
    finish = (finish_reason or "").casefold()
    if finish == "length":
        reasons.append("finish_reason_length")
    if max_completion_tokens and completion_tokens is not None and completion_tokens >= max_completion_tokens:
        reasons.append("reached_token_cap")
    if _looks_incomplete(category, normalized.answer, normalized.raw_text):
        reasons.append("incomplete_output")
    if any(reason in reasons for reason in ("finish_reason_length", "reached_token_cap")) and _cap_hit_prose_unfinished(category, normalized.answer, normalized.raw_text):
        reasons.append("cap_hit_unfinished_prose")
    if _category_required_structure_incomplete(category, normalized):
        reasons.append("required_structure_incomplete")
    incomplete = any(reason in reasons for reason in ("incomplete_output", "cap_hit_unfinished_prose", "required_structure_incomplete"))
    return TruncationResult(bool(reasons), incomplete, tuple(reasons))


def _looks_incomplete(category: str, answer: str, raw_text: str) -> bool:
    text = (raw_text or answer).rstrip()
    if not text:
        return False
    if category in {"code_generation", "code_debugging"}:
        if text.endswith("```"):
            return False
        if text.count("```") % 2 == 1:
            return True
        try:
            ast.parse(strip_code_fences(text)[0], mode="exec")
            return False
        except SyntaxError:
            pass
        return bool(re.search(r"(\breturn\s+\w*$|[+\-*/=([{:,.]$|:\s*$)", text))
    if category == "factual":
        if re.search(r"\b(or|and|the|a|an|to|of|with|that|which|because|if|when|while|for)$", text, re.I):
            return True
        if re.search(r"\w{4,}$", text) and not re.search(r"[.!?)]$|\b(?:UTC|API|CPU|GPU|RAM|O₂|CO₂)$", text):
            last = re.findall(r"[A-Za-z]+$", text)
            if last and last[0].casefold() in {"diff", "alter", "consist", "transmit", "calculat", "receiv"}:
                return True
        return False
    if category == "logic":
        return bool(re.search(r"\b(and|or|so|because|therefore|then|with)$", text, re.I))
    if category == "ner":
        last = answer.splitlines()[-1] if answer.splitlines() else ""
        return bool(last and "|" not in last)
    return False


def _cap_hit_prose_unfinished(category: str, answer: str, raw_text: str) -> bool:
    if category not in {"factual", "summarization"}:
        return False
    text = (answer or raw_text).strip()
    if not text:
        return False
    if re.search(r"[.!?)]\s*(?:[\"'”’])?$", text):
        return False
    if re.search(r"[,;:—-]\s*$", text):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not words:
        return False
    if words[-1].casefold() in {"and", "or", "but", "because", "although", "while", "when", "if", "that", "which", "who", "whose", "with", "without", "to", "of", "for", "from", "by", "as", "the", "a", "an"}:
        return True
    return len(words) >= 8


def _category_required_structure_incomplete(category: str, normalized: NormalizedAnswer) -> bool:
    if category == "sentiment":
        return "LABEL:" not in normalized.answer or "REASON:" not in normalized.answer
    if category == "math":
        return not normalized.final_section.strip()
    if category == "logic":
        return not normalized.final_section.strip()
    if category == "ner":
        return any("|" not in line for line in normalized.answer.splitlines() if line.strip())
    return False


def _prompt_has_mutable_default_bug(prompt: str) -> bool:
    lowered = prompt.casefold()
    if "mutable-default" in lowered or "mutable default" in lowered:
        return True
    for code in _extract_prompt_code_blocks(prompt):
        try:
            if has_mutable_default_defect(ast.parse(code, mode="exec")):
                return True
        except SyntaxError:
            continue
    return False


def _extract_prompt_code_blocks(prompt: str) -> list[str]:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", prompt, flags=re.S | re.I)
    if blocks:
        return [block.strip() for block in blocks]
    return []


def has_mutable_default_defect(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defaults = list(node.args.defaults) + [default for default in node.args.kw_defaults if default is not None]
        if any(_is_mutable_default(default) for default in defaults):
            return True
    return False


def _is_mutable_default(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"list", "dict", "set"}


def classify_code_generation_risk(prompt: str) -> dict[str, Any]:
    text = prompt.casefold()
    high_patterns = {
        "mutation": r"\b(mutate|mutation|in[- ]place|unchanged|copy|preserve)\b",
        "directional_indexing": r"\b(rotate|rotation|left|right|oversized|negative step|index)\b",
        "intervals": r"\b(interval|merge overlapping|touching intervals)\b",
        "nested_state": r"\b(nested|tree|graph|stateful|class|cache|memo|recursive|recursion)\b",
        "external_io": r"\b(file|network|http|database|sql|api|thread|concurrency|socket|path)\b",
        "edge_cases": r"\b(edge cases?|none|missing key|complexity|custom class)\b",
    }
    reasons = [name for name, pattern in high_patterns.items() if re.search(pattern, text)]
    if reasons:
        return {"risk": "high", "reasons": reasons}
    low = bool(
        re.search(r"\b(count|filter|format|sum|average|uppercase|lowercase|reverse|string|arithmetic|mapping)\b", text)
        or requested_function_name(prompt)
    )
    return {"risk": "low" if low else "high", "reasons": [] if low else ["unclear_behavior"]}


def derive_prompt_code_examples(prompt: str, function_name: str | None = None) -> list[tuple[str, Any]]:
    name = function_name or requested_function_name(prompt)
    if not name:
        return []
    examples: list[tuple[str, Any]] = []
    pattern = rf"\b({re.escape(name)}\s*\([^)]*\))\s+(?:should\s+)?(?:return|returns|=>|->)\s+([^.;\n]+)"
    for call_src, expected_src in re.findall(pattern, prompt, flags=re.I):
        try:
            call = ast.parse(call_src, mode="eval").body
            if not _safe_literal_call(call, name):
                continue
            expected = ast.literal_eval(expected_src.strip())
        except Exception:
            try:
                expected = ast.literal_eval(expected_src.strip().rstrip("."))
            except Exception:
                continue
        examples.append((call_src, expected))
    return examples[:3]


def _safe_literal_call(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != name:
        return False
    try:
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            ast.literal_eval(arg)
    except Exception:
        return False
    return True


def _run_prompt_examples(tree: ast.AST, function_name: str | None, examples: list[tuple[str, Any]]) -> str:
    if not function_name:
        return "cannot identify function name for prompt examples"
    if any(isinstance(node, (ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.Lambda)) for node in ast.walk(tree)):
        return "prompt examples skipped because code contains complex control flow"
    namespace: dict[str, Any] = {
        "__builtins__": {
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "sorted": sorted,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
        }
    }
    try:
        exec(compile(tree, "<local-answer>", "exec"), namespace, namespace)
        fn = namespace.get(function_name)
        if not callable(fn):
            return f"{function_name} is not callable"
        for call_src, expected in examples:
            call_node = ast.parse(call_src, mode="eval").body
            assert isinstance(call_node, ast.Call)
            args = [copy.deepcopy(ast.literal_eval(arg)) for arg in call_node.args]
            kwargs = {kw.arg: copy.deepcopy(ast.literal_eval(kw.value)) for kw in call_node.keywords if kw.arg}
            actual = fn(*args, **kwargs)
            if actual != expected:
                return f"{call_src} returned {actual!r}, expected {expected!r}"
    except Exception as exc:
        return f"prompt example raised {type(exc).__name__}: {exc}"
    return ""
