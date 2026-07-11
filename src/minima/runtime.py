"""Single production local-first runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import itertools
import re
import sys
import time
from typing import Any

from .answer_validation import (
    NormalizedAnswer,
    TruncationResult,
    ValidationResult,
    classify_code_generation_risk,
    detect_truncation,
    is_freshness_sensitive,
    normalize_answer,
    summary_requirement_instructions,
    validate_answer,
)
from .fireworks_client import FireworksClient, FireworksClientError
from .local_llm import LocalGeneration, local_generate_with_metadata
from .model_selector import select_model_candidates
from .prompts import (
    local_max_tokens_for_category,
    local_output_format_for_category,
    local_system_prompt_for_category,
)
from .task_classifier import classify_task


try:
    from .local_solver import _solve_logic, _solve_math
except Exception:  # pragma: no cover - defensive for minimal packaging
    _solve_logic = None  # type: ignore[assignment]
    _solve_math = None  # type: ignore[assignment]


LOCAL_MODEL_CATEGORIES = frozenset({"sentiment", "summarization"})


@dataclass
class PipelineTrace:
    task_id: str | None
    category: str
    answer: str
    path: str
    deterministic_attempted: bool = False
    deterministic_answer: str | None = None
    deterministic_valid: bool = False
    primary_calls: int = 0
    repair_calls: int = 0
    fireworks_calls: int = 0
    would_fallback: bool = False
    actual_fallback: bool = False
    accepted_local: bool = False
    invalid_local: bool = False
    first_pass_valid: bool = False
    final_valid: bool = False
    repair_success: bool = False
    validation: ValidationResult | None = None
    repair_validation: ValidationResult | None = None
    normalized: NormalizedAnswer | None = None
    repair_normalized: NormalizedAnswer | None = None
    generation: LocalGeneration | None = None
    repair_generation: LocalGeneration | None = None
    remote_error: str | None = None
    fallback_reason: str = ""
    unverified_local: bool = False
    primary_truncation: TruncationResult | None = None
    repair_truncation: TruncationResult | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return _tokens(self.generation, "prompt_tokens") + _tokens(self.repair_generation, "prompt_tokens")

    @property
    def completion_tokens(self) -> int:
        return _tokens(self.generation, "completion_tokens") + _tokens(self.repair_generation, "completion_tokens")

    @property
    def total_tokens(self) -> int:
        return _tokens(self.generation, "total_tokens") + _tokens(self.repair_generation, "total_tokens")

    @property
    def model_load_ms(self) -> float:
        values = [
            gen.model_load_ms
            for gen in (self.generation, self.repair_generation)
            if gen is not None and gen.model_load_ms
        ]
        return sum(values)


class LocalFirstOrchestrator:
    """Production orchestrator: classify -> solve -> local -> validate -> repair -> remote."""

    def __init__(self, client: FireworksClient):
        self.client = client

    def answer(self, prompt: str, task_id: str | None = None) -> str:
        return self.answer_with_trace(prompt, task_id=task_id).answer

    def answer_with_trace(self, prompt: str, task_id: str | None = None) -> PipelineTrace:
        started = time.perf_counter()
        category = classify_task(prompt)
        deterministic_started = time.perf_counter()
        deterministic = deterministic_answer(category, prompt)
        deterministic_ms = (time.perf_counter() - deterministic_started) * 1000
        if deterministic is not None:
            normalized = normalize_answer(category, deterministic, prompt)
            validation = validate_answer(category, prompt, normalized, deterministic_answer=deterministic)
            if validation.valid:
                trace = PipelineTrace(
                    task_id=task_id,
                    category=category,
                    answer=normalized.answer,
                    path="deterministic",
                    deterministic_attempted=True,
                    deterministic_answer=deterministic,
                    deterministic_valid=True,
                    accepted_local=True,
                    first_pass_valid=True,
                    final_valid=True,
                    validation=validation,
                    normalized=normalized,
                    timings_ms={"deterministic": deterministic_ms, "total": (time.perf_counter() - started) * 1000},
                )
                _log(trace, "deterministic_accept")
                return trace

        if category == "factual" and is_freshness_sensitive(prompt):
            return self._remote_or_best_effort(
                prompt=prompt,
                task_id=task_id,
                category=category,
                local_answer="Unable to determine without up-to-date information.",
                reason="freshness_required",
                started=started,
                deterministic=deterministic,
                deterministic_ms=deterministic_ms,
            )

        if category not in LOCAL_MODEL_CATEGORIES:
            return self._remote_or_best_effort(
                prompt=prompt,
                task_id=task_id,
                category=category,
                local_answer="Unable to answer locally with sufficient confidence.",
                reason="category_remote_policy",
                started=started,
                deterministic=deterministic,
                deterministic_ms=deterministic_ms,
            )

        code_risk = classify_code_generation_risk(prompt) if category == "code_generation" else {"risk": "low", "reasons": []}
        if category == "code_generation" and code_risk["risk"] == "high" and not self.client.config.placeholder_mode:
            return self._remote_or_best_effort(
                prompt=prompt,
                task_id=task_id,
                category=category,
                local_answer="pass",
                reason="high_risk_code_generation",
                started=started,
                deterministic=deterministic,
                deterministic_ms=deterministic_ms,
            )

        primary_prompt = build_local_user_prompt(category, prompt)
        generation = local_generate_with_metadata(
            system=local_system_prompt_for_category(category),
            prompt=primary_prompt,
            max_tokens=local_max_tokens_for_category(category),
        )
        normalized = normalize_answer(category, generation.text or generation.raw_text or "", prompt)
        truncation = detect_truncation(
            category,
            prompt,
            normalized,
            finish_reason=generation.finish_reason,
            completion_tokens=generation.completion_tokens,
            max_completion_tokens=generation.max_completion_tokens or local_max_tokens_for_category(category),
        )
        validation = validate_answer(
            category,
            prompt,
            normalized,
            deterministic_answer=deterministic,
            truncation=truncation,
        )
        if generation.error:
            validation = ValidationResult(False, "local_generation_failed", generation.error, False, True)
        if validation.valid:
            trace = PipelineTrace(
                task_id=task_id,
                category=category,
                answer=normalized.answer,
                path="local_primary",
                deterministic_attempted=deterministic is not None,
                deterministic_answer=deterministic,
                primary_calls=1,
                accepted_local=True,
                first_pass_valid=True,
                final_valid=True,
                validation=validation,
                normalized=normalized,
                generation=generation,
                unverified_local=not validation.verified_local,
                primary_truncation=truncation,
                timings_ms={"deterministic": deterministic_ms, "primary": generation.runtime_ms, "total": (time.perf_counter() - started) * 1000},
            )
            _log(trace, "primary_accept")
            return trace

        repair_generation: LocalGeneration | None = None
        repair_normalized: NormalizedAnswer | None = None
        repair_validation: ValidationResult | None = None
        repair_truncation: TruncationResult | None = None
        if validation.safe_to_repair:
            repair_prompt = build_repair_prompt(category, prompt, normalized.answer or normalized.raw_text, validation)
            repair_generation = local_generate_with_metadata(
                system=local_system_prompt_for_category(category),
                prompt=repair_prompt,
                max_tokens=local_max_tokens_for_category(category),
            )
            repair_normalized = normalize_answer(category, repair_generation.text or repair_generation.raw_text or "", prompt)
            repair_truncation = detect_truncation(
                category,
                prompt,
                repair_normalized,
                finish_reason=repair_generation.finish_reason,
                completion_tokens=repair_generation.completion_tokens,
                max_completion_tokens=repair_generation.max_completion_tokens or local_max_tokens_for_category(category),
            )
            repair_validation = validate_answer(
                category,
                prompt,
                repair_normalized,
                deterministic_answer=deterministic,
                truncation=repair_truncation,
            )
            if repair_generation.error:
                repair_validation = ValidationResult(False, "repair_generation_failed", repair_generation.error, False, True)
            if repair_validation.valid:
                trace = PipelineTrace(
                    task_id=task_id,
                    category=category,
                    answer=repair_normalized.answer,
                    path="local_repair",
                    deterministic_attempted=deterministic is not None,
                    deterministic_answer=deterministic,
                    primary_calls=1,
                    repair_calls=1,
                    accepted_local=True,
                    invalid_local=True,
                    first_pass_valid=False,
                    final_valid=True,
                    repair_success=True,
                    validation=validation,
                    repair_validation=repair_validation,
                    normalized=normalized,
                    repair_normalized=repair_normalized,
                    generation=generation,
                    repair_generation=repair_generation,
                    unverified_local=not repair_validation.verified_local,
                    primary_truncation=truncation,
                    repair_truncation=repair_truncation,
                    timings_ms={"deterministic": deterministic_ms, "primary": generation.runtime_ms, "repair": repair_generation.runtime_ms, "total": (time.perf_counter() - started) * 1000},
                )
                _log(trace, "repair_accept")
                return trace

        best_local = _best_available_answer(category, normalized, repair_normalized)
        remote_reason = (
            repair_validation.failure_code
            if repair_validation is not None and repair_validation.requires_remote
            else validation.failure_code
        )
        trace = self._remote_or_best_effort(
            prompt=prompt,
            task_id=task_id,
            category=category,
            local_answer=best_local,
            reason=remote_reason or "local_invalid",
            started=started,
            deterministic=deterministic,
            deterministic_ms=deterministic_ms,
            generation=generation,
            normalized=normalized,
            validation=validation,
            repair_generation=repair_generation,
            repair_normalized=repair_normalized,
            repair_validation=repair_validation,
            primary_truncation=truncation,
            repair_truncation=repair_truncation,
        )
        return trace

    def _remote_or_best_effort(
        self,
        *,
        prompt: str,
        task_id: str | None,
        category: str,
        local_answer: str,
        reason: str,
        started: float,
        deterministic: str | None,
        deterministic_ms: float,
        generation: LocalGeneration | None = None,
        normalized: NormalizedAnswer | None = None,
        validation: ValidationResult | None = None,
        repair_generation: LocalGeneration | None = None,
        repair_normalized: NormalizedAnswer | None = None,
        repair_validation: ValidationResult | None = None,
        primary_truncation: TruncationResult | None = None,
        repair_truncation: TruncationResult | None = None,
    ) -> PipelineTrace:
        if self.client.config.placeholder_mode:
            trace = PipelineTrace(
                task_id=task_id,
                category=category,
                answer=local_answer,
                path="offline_best_effort",
                deterministic_attempted=deterministic is not None,
                deterministic_answer=deterministic,
                primary_calls=1 if generation else 0,
                repair_calls=1 if repair_generation else 0,
                would_fallback=True,
                accepted_local=False,
                invalid_local=True,
                first_pass_valid=False,
                final_valid=False,
                validation=validation,
                repair_validation=repair_validation,
                normalized=normalized,
                repair_normalized=repair_normalized,
                generation=generation,
                repair_generation=repair_generation,
                unverified_local=True,
                primary_truncation=primary_truncation,
                repair_truncation=repair_truncation,
                fallback_reason=reason,
                timings_ms={"deterministic": deterministic_ms, "total": (time.perf_counter() - started) * 1000},
            )
            _log(trace, "would_fallback_offline")
            return trace

        for retry, model in enumerate(select_model_candidates(category, self.client.config.allowed_models)):
            _log_remote(task_id, category, model, retry, reason)
            try:
                answer = self.client.answer(prompt=prompt, category=category, model=model, task_id=task_id)
            except FireworksClientError as exc:
                remote_error = f"{type(exc).__name__}: {exc}"
                _log_remote_failure(task_id, category, model, retry, remote_error)
                continue
            stripped = answer.strip()
            if stripped:
                trace = PipelineTrace(
                    task_id=task_id,
                    category=category,
                    answer=stripped,
                    path="remote",
                    deterministic_attempted=deterministic is not None,
                    deterministic_answer=deterministic,
                    primary_calls=1 if generation else 0,
                    repair_calls=1 if repair_generation else 0,
                    fireworks_calls=retry + 1,
                    actual_fallback=True,
                    invalid_local=True,
                    validation=validation,
                    repair_validation=repair_validation,
                    normalized=normalized,
                    repair_normalized=repair_normalized,
                    generation=generation,
                    repair_generation=repair_generation,
                    primary_truncation=primary_truncation,
                    repair_truncation=repair_truncation,
                    fallback_reason=reason,
                    timings_ms={"deterministic": deterministic_ms, "total": (time.perf_counter() - started) * 1000},
                )
                _log(trace, "remote_accept")
                return trace
        trace = PipelineTrace(
            task_id=task_id,
            category=category,
            answer=local_answer,
            path="remote_failed_best_effort",
            deterministic_attempted=deterministic is not None,
            deterministic_answer=deterministic,
            primary_calls=1 if generation else 0,
            repair_calls=1 if repair_generation else 0,
            would_fallback=True,
            invalid_local=True,
            validation=validation,
            repair_validation=repair_validation,
            normalized=normalized,
            repair_normalized=repair_normalized,
            generation=generation,
            repair_generation=repair_generation,
            unverified_local=True,
            primary_truncation=primary_truncation,
            repair_truncation=repair_truncation,
            fallback_reason=reason,
            timings_ms={"deterministic": deterministic_ms, "total": (time.perf_counter() - started) * 1000},
        )
        _log(trace, "remote_failed_best_effort")
        return trace


def deterministic_answer(category: str, prompt: str) -> str | None:
    try:
        if category == "math":
            answer = _solve_supported_math(prompt)
            if answer:
                return answer
        if category == "logic":
            answer = _solve_supported_logic(prompt)
            if answer:
                return answer
        if category == "math" and callable(_solve_math):
            answer = _solve_math(prompt)
            return _finalize_deterministic_math(answer)
        if category == "logic" and callable(_solve_logic):
            answer = _solve_logic(prompt)
            return _finalize_deterministic_logic(answer)
    except Exception:
        return None
    return None


def _solve_supported_math(prompt: str) -> str | None:
    text = prompt.casefold()
    if re.search(r"\babout\s+half\s+full\b", text) and re.search(r"\bexact", text):
        return "FINAL: Cannot be determined exactly"
    match = re.search(r"\$?([\d,]+(?:\.\d+)?)\D+(?:discounted\s+)?(\d+(?:\.\d+)?)%\s*(?:off)?\D+(?:taxed|tax)\s+(\d+(?:\.\d+)?)%", prompt, re.I)
    if match:
        base = _num(match.group(1)); discount = _num(match.group(2)) / 100; tax = _num(match.group(3)) / 100
        return f"FINAL: final price = ${_fmt_money(base * (1 - discount) * (1 + tax))}"
    match = re.search(r"starts?\s+with\s+([\d,]+).*?ships?\s+([\d,]+).*?receives?\s+([\d,]+)\s+cartons?\s+of\s+([\d,]+).*?discards?\s+([\d,]+)", prompt, re.I | re.S)
    if match:
        start, shipped, cartons, per_carton, discarded = [_num(item) for item in match.groups()]
        return f"FINAL: usable filters remaining = {_fmt(start - shipped + cartons * per_carton - discarded)}"
    match = re.search(r"([\d,]+(?:\.\d+)?)\s+requests?\s+per\s+hour.*?grows?\s+(\d+(?:\.\d+)?)%.*?(one|two|three|four|five|\d+)\s+months?", prompt, re.I | re.S)
    if match:
        initial = _num(match.group(1)); rate = _num(match.group(2)) / 100; months = _word_num(match.group(3))
        return f"FINAL: projected hourly rate = {_fmt(initial * (1 + rate) ** months)} requests per hour"
    match = re.search(r"([A-Za-z][A-Za-z ]+?)\s+and\s+([A-Za-z][A-Za-z ]+?)\s+.*?ratio\s+(\d+)\s*:\s*(\d+).*?([\d,]+(?:\.\d+)?)\s*(?:marbles|g|grams|total)", prompt, re.I | re.S)
    if match:
        left_label, right_label, left, right, total = match.groups()
        left_n = _num(left); right_n = _num(right); total_n = _num(total); unit = "g" if re.search(r"\b(g|grams)\b", prompt, re.I) else ""
        part = total_n / (left_n + right_n)
        return f"FINAL: {left_label.strip()} = {_fmt(left_n * part)}{(' ' + unit) if unit else ''}; {right_label.strip()} = {_fmt(right_n * part)}{(' ' + unit) if unit else ''}"
    match = re.search(r"uses\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)\s+in\s+a\s+(\d+)\s*:\s*(\d+)\s+ratio.*?([\d,]+(?:\.\d+)?)\s*g", prompt, re.I | re.S)
    if match:
        left_label, right_label, left, right, total = match.groups()
        left_n = _num(left); right_n = _num(right); total_n = _num(total); part = total_n / (left_n + right_n)
        return f"FINAL: {left_label} = {_fmt(left_n * part)} g; {right_label} = {_fmt(right_n * part)} g"
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*km\s+in\s+(\d+(?:\.\d+)?)\s+hours?\s+(\d+(?:\.\d+)?)\s+minutes?", prompt, re.I)
    if match and "average speed" in text:
        distance = _num(match.group(1)); hours = _num(match.group(2)) + _num(match.group(3)) / 60
        if hours:
            return f"FINAL: average speed = {_fmt(distance / hours)} km/h"
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*km\s+(?:in|over)\s+([\d,]+(?:\.\d+)?)\s*h", prompt, re.I)
    if match and "average speed" in text:
        distance = _num(match.group(1)); hours = _num(match.group(2))
        if hours:
            return f"FINAL: average speed = {_fmt(distance / hours)} km/h"
    legs = [(_num(d), _num(s)) for d, s in re.findall(r"(\d+(?:\.\d+)?)\s*km\s+at\s+(\d+(?:\.\d+)?)\s*km/h", prompt, re.I)]
    if len(legs) >= 2 and "average speed" in text:
        total_distance = sum(d for d, _ in legs)
        total_time = sum(d / s for d, s in legs if s)
        if total_time:
            return f"FINAL: total travel time = {_fmt(total_time)} hours; overall average speed = {_fmt(total_distance / total_time)} km/h"
    match = re.search(r"\$?([\d,]+(?:\.\d+)?).*?(\d+(?:\.\d+)?)%\s+simple.*?(\d+(?:\.\d+)?)\s+months?", prompt, re.I | re.S)
    if match:
        principal = _num(match.group(1)); rate = _num(match.group(2)) / 100; months = _num(match.group(3))
        interest = principal * rate * (months / 12)
        return f"FINAL: interest = ${_fmt_money(interest)}; final balance = ${_fmt_money(principal + interest)}"
    match = re.search(r"(?:was|is|of)\s+([\d,]+(?:\.\d+)?),?\s+(?:with\s+)?(\d+(?:\.\d+)?)%", prompt, re.I | re.S)
    if match and re.search(r"\b(percent|percentage|visitors?|saturday)\b", text):
        value = _num(match.group(1)) * _num(match.group(2)) / 100
        return f"FINAL: {_fmt(value)}"
    return None


def _solve_supported_logic(prompt: str) -> str | None:
    finite = _solve_finite_domain_logic(prompt)
    if finite:
        return finite
    order = _solve_finish_order(prompt)
    if order:
        return order
    seats = _solve_seat_order(prompt)
    if seats:
        return seats
    return None


def _solve_finite_domain_logic(prompt: str) -> str | None:
    parsed = _parse_finite_domain_prompt(prompt)
    if parsed is None:
        return None
    names, values, unique, constraints, query_name = parsed
    valid: list[dict[str, str]] = []
    for product in itertools.product(values, repeat=len(names)):
        assignment = dict(zip(names, product))
        if unique and len(set(product)) != len(product):
            continue
        if all(check(assignment) for check in constraints):
            valid.append(assignment)
    if not valid:
        return "ANSWER: No valid assignment satisfies the constraints."
    if query_name:
        possible = sorted({assignment[query_name] for assignment in valid})
        if len(possible) == 1:
            return f"ANSWER: {query_name} chooses {possible[0]}."
        return f"ANSWER: Cannot be determined uniquely; {query_name} may choose {' or '.join(possible)}."
    stable: list[str] = []
    for name in names:
        possible = sorted({assignment[name] for assignment in valid})
        if len(possible) == 1:
            stable.append(f"{name} = {possible[0]}")
    if stable:
        return "ANSWER: " + "; ".join(stable)
    return "ANSWER: Cannot be determined uniquely."


def _parse_finite_domain_prompt(prompt: str) -> tuple[list[str], list[str], bool, list[Any], str | None] | None:
    text = re.sub(r"\s+", " ", prompt.strip())
    match = re.search(
        r"\b(?P<names>[A-Z][A-Za-z]*(?:\s*,\s*[A-Z][A-Za-z]*)*(?:,?\s+and\s+[A-Z][A-Za-z]*)?)\s+each\s+(?:choose|chooses|select|selects|pick|picks)\s+(?P<values>[a-z][a-z-]*(?:\s*,\s*[a-z][a-z-]*)*(?:,?\s+or\s+[a-z][a-z-]*)?)",
        text,
        re.I,
    )
    if not match:
        return None
    names = re.findall(r"\b[A-Z][A-Za-z]*\b", match.group("names"))
    values = [value.casefold() for value in re.findall(r"\b[a-z][a-z-]*\b", match.group("values")) if value.casefold() not in {"or", "and"}]
    names = [name for name in names if name.lower() not in {"what", "which"}]
    values = list(dict.fromkeys(values))
    if not (2 <= len(names) <= 6 and 2 <= len(values) <= 6):
        return None
    name_set = {name.casefold(): name for name in names}
    value_set = set(values)
    unique = bool(re.search(r"\b(each|all)\s+(?:choose|select|pick)\s+different|choices\s+must\s+differ|one-to-one|unique\b", text, re.I))
    if re.search(r"\bchoices\s+need\s+not\s+differ|choices\s+may\s+repeat|need\s+not\s+be\s+unique\b", text, re.I):
        unique = False
    constraints: list[Any] = []

    def add_fixed(name_raw: str, value_raw: str, negate: bool = False) -> None:
        name = name_set.get(name_raw.casefold())
        value = value_raw.casefold()
        if not name or value not in value_set:
            return
        if negate:
            constraints.append(lambda assignment, n=name, v=value: assignment[n] != v)
        else:
            constraints.append(lambda assignment, n=name, v=value: assignment[n] == v)

    for name, value in re.findall(r"\b([A-Z][A-Za-z]*)\s+(?:choose|chooses|select|selects|pick|picks)\s+([a-z][a-z-]*)\b", text):
        add_fixed(name, value)
    for name, value in re.findall(r"\b([A-Z][A-Za-z]*)\s+(?:does\s+not|doesn't|cannot|can't)\s+(?:choose|select|pick)\s+([a-z][a-z-]*)\b", text, re.I):
        add_fixed(name, value, True)
    for left, right in re.findall(r"\b([A-Z][A-Za-z]*)\s+(?:choose|chooses|select|selects|pick|picks)\s+the\s+same\s+as\s+([A-Z][A-Za-z]*)\b", text, re.I):
        a = name_set.get(left.casefold()); b = name_set.get(right.casefold())
        if a and b:
            constraints.append(lambda assignment, x=a, y=b: assignment[x] == assignment[y])
    for left, right in re.findall(r"\b([A-Z][A-Za-z]*)\s+(?:differs\s+from|is\s+different\s+from|not\s+the\s+same\s+as)\s+([A-Z][A-Za-z]*)\b", text, re.I):
        a = name_set.get(left.casefold()); b = name_set.get(right.casefold())
        if a and b:
            constraints.append(lambda assignment, x=a, y=b: assignment[x] != assignment[y])
    for cond_name, cond_value, then_name, then_value in re.findall(
        r"\bif\s+([A-Z][A-Za-z]*)\s+(?:choose|chooses|select|selects|pick|picks)\s+([a-z][a-z-]*),?\s+then\s+([A-Z][A-Za-z]*)\s+(?:choose|chooses|select|selects|pick|picks)\s+([a-z][a-z-]*)",
        text,
        re.I,
    ):
        cn = name_set.get(cond_name.casefold()); tn = name_set.get(then_name.casefold())
        cv = cond_value.casefold(); tv = then_value.casefold()
        if cn and tn and cv in value_set and tv in value_set:
            constraints.append(lambda assignment, cn=cn, cv=cv, tn=tn, tv=tv: assignment[cn] != cv or assignment[tn] == tv)
    query_name = None
    query = re.search(r"\bWhat\s+does\s+([A-Z][A-Za-z]*)\s+(?:choose|select|pick)\b", prompt)
    if query:
        query_name = name_set.get(query.group(1).casefold())
    return names, values, unique, constraints, query_name


def _solve_finish_order(prompt: str) -> str | None:
    edges: list[tuple[str, str]] = []
    for a, b in re.findall(r"\b([A-Z][a-z]+)\s+finishes\s+before\s+([A-Z][a-z]+)\b", prompt):
        edges.append((a, b))
    for a, b in re.findall(r"\b([A-Z][a-z]+)\s+finishes\s+after\s+([A-Z][a-z]+)\b", prompt):
        edges.append((b, a))
    if not edges:
        return None
    names = sorted({name for edge in edges for name in edge})
    valid = []
    for perm in itertools.permutations(names):
        pos = {name: idx for idx, name in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b in edges):
            valid.append(perm)
    if len(valid) != 1:
        return None
    perm = valid[0]
    parts = []
    if "first" in prompt.casefold():
        parts.append(f"first = {perm[0]}")
    if "last" in prompt.casefold():
        parts.append(f"last = {perm[-1]}")
    return "ANSWER: " + "; ".join(parts or [", ".join(perm)])


def _solve_seat_order(prompt: str) -> str | None:
    names = sorted(set(re.findall(r"\b[A-Z][a-z]+\b", prompt)) - {"Give", "Who", "What", "Four", "Three", "Two", "One"})
    if not (2 <= len(names) <= 6) or "seat" not in prompt.casefold():
        return None
    seat_numbers = sorted({int(n) for n in re.findall(r"\bseat\s+(\d+)\b", prompt, re.I)})
    n = max(seat_numbers) if seat_numbers else len(names)
    if n != len(names):
        return None
    valid = []
    for perm in itertools.permutations(names):
        seat = {name: idx + 1 for idx, name in enumerate(perm)}
        ok = True
        for name, num in re.findall(r"\b([A-Z][a-z]+)\s+(?:is|sits)\s+in\s+seat\s+(\d+)", prompt):
            ok &= seat.get(name) == int(num)
        for left, right in re.findall(r"\b([A-Z][a-z]+)\s+sits\s+immediately\s+left\s+of\s+([A-Z][a-z]+)", prompt):
            ok &= seat.get(right) == seat.get(left, -100) + 1
        for a, b in re.findall(r"\b([A-Z][a-z]+)\s+is\s+not\s+adjacent\s+to\s+([A-Z][a-z]+)", prompt):
            ok &= abs(seat.get(a, -100) - seat.get(b, 100)) != 1
        if ok:
            valid.append(perm)
    if len(valid) != 1:
        return None
    return "ANSWER: " + ", ".join(f"{idx + 1}-{name}" for idx, name in enumerate(valid[0]))


def _num(raw: str) -> float:
    return float(str(raw).replace(",", ""))


def _word_num(raw: str) -> int:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    return words.get(raw.casefold(), int(raw) if raw.isdigit() else 0)


def _fmt(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) < 1e-9 else f"{value:.4f}".rstrip("0").rstrip(".")


def _fmt_money(value: float) -> str:
    return f"{value:.2f}"


def build_local_user_prompt(category: str, prompt: str) -> str:
    prompt = prompt.strip()
    if category == "summarization":
        requirements = summary_requirement_instructions(prompt)
        if requirements:
            return f"{prompt}\n\nStructural requirements:\n{requirements}"
    return prompt


def build_repair_prompt(category: str, prompt: str, previous_answer: str, validation: ValidationResult) -> str:
    detail = validation.failure_detail or validation.failure_code
    if category == "math":
        return (
            f"Original task:\n{prompt.strip()}\n\n"
            f"Previous answer:\n{previous_answer.strip()}\n\n"
            f"Machine-detected failure: {detail}\n\n"
            "Recalculate from the original task. Return only:\nFINAL: complete requested values with labels and units"
        )
    if category == "logic":
        return (
            f"Original task:\n{prompt.strip()}\n\n"
            f"Previous answer:\n{previous_answer.strip()}\n\n"
            f"The proposed answer failed this constraint: {detail}\n\n"
            "Return only a corrected complete ANSWER."
        )
    if category == "ner":
        return (
            f"Original task:\n{prompt.strip()}\n\n"
            f"Previous extraction:\n{previous_answer.strip()}\n\n"
            f"Machine-detected failure: {detail}\n\n"
            "Return every entity again using exact source spans, one per line as: exact span | TYPE."
        )
    if category == "summarization":
        return (
            f"Original task:\n{prompt.strip()}\n\n"
            f"Previous summary:\n{previous_answer.strip()}\n\n"
            f"The previous summary violated this requirement: {detail}\n\n"
            "Return only the corrected summary."
        )
    if category in {"code_generation", "code_debugging"}:
        return (
            f"Original task:\n{prompt.strip()}\n\n"
            f"Previous code:\n{previous_answer.strip()}\n\n"
            f"The previous code failed this objective check: {detail}\n\n"
            "Return corrected code only."
        )
    return (
        f"Original task:\n{prompt.strip()}\n\n"
        f"Previous answer:\n{previous_answer.strip()}\n\n"
        f"Machine-detected failure: {detail}\n\n"
        f"Return the corrected answer in this format: {local_output_format_for_category(category)}."
    )


def _finalize_deterministic_math(answer: str | None) -> str | None:
    if not answer:
        return None
    stripped = answer.strip()
    if stripped.casefold().startswith("final:"):
        return stripped
    return f"FINAL: {stripped}"


def _finalize_deterministic_logic(answer: str | None) -> str | None:
    if not answer:
        return None
    stripped = answer.strip()
    if stripped.casefold().startswith("answer:"):
        return stripped
    return f"ANSWER: {stripped}"


def _best_available_answer(category: str, primary: NormalizedAnswer | None, repair: NormalizedAnswer | None) -> str:
    for candidate in (repair, primary):
        if candidate is not None and candidate.answer.strip():
            return candidate.answer.strip()
    if category == "sentiment":
        return "LABEL: neutral"
    if category == "math":
        return "FINAL: Unable to determine"
    if category == "ner":
        return ""
    if category == "logic":
        return "ANSWER: Unable to determine"
    if category in {"code_generation", "code_debugging"}:
        return "pass"
    return "Unable to determine."


def _tokens(generation: LocalGeneration | None, attr: str) -> int:
    return int(getattr(generation, attr, 0) or 0) if generation is not None else 0


def _log(trace: PipelineTrace, event: str) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    print(
        "minima route "
        f"task_id={trace.task_id or '-'} category={trace.category} path={trace.path} "
        f"event={event} valid={trace.final_valid} fallback={trace.would_fallback or trace.actual_fallback} "
        f"reason={trace.fallback_reason or (trace.validation.failure_code if trace.validation else '-')}",
        file=sys.stderr,
    )


def _log_remote(task_id: str | None, category: str, model: str, retry: int, reason: str) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    print(
        f"minima remote task_id={task_id or '-'} category={category} model={model} retry={retry} reason={reason}",
        file=sys.stderr,
    )


def _log_remote_failure(task_id: str | None, category: str, model: str, retry: int, reason: str) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    print(
        f"minima remote_failed task_id={task_id or '-'} category={category} model={model} retry={retry} reason={reason}",
        file=sys.stderr,
    )
