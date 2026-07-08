"""Task routing for the minima agent."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys

from .fireworks_client import FireworksClient, FireworksClientError
from .local_solver import solve_local
from .model_selector import select_model_candidates
from .task_classifier import classify_task


PLACEHOLDER_PATTERNS = (
    "local test placeholder",
    "placeholder",
    "lorem ipsum",
    "todo",
)


def _looks_placeholder(answer: str) -> bool:
    lowered = answer.casefold()
    return any(pattern in lowered for pattern in PLACEHOLDER_PATTERNS)


def _looks_malformed(category: str, answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    if _looks_placeholder(stripped):
        return True
    if category == "sentiment":
        return re.search(r"\b(positive|negative|neutral|mixed)\b", stripped, re.I) is None
    if category == "ner":
        return len(stripped) > 5000
    if category in {"code_generation", "code_debugging"}:
        lowered = stripped.casefold()
        if "cannot" in lowered and "provide" in lowered:
            return True
    return False


def _log_routing(task_id: str | None, category: str, model: str, retry: int) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    label = task_id or "-"
    print(
        f"minima route task_id={label} category={category} model={model} retry={retry}",
        file=sys.stderr,
    )


def _log_model_failure(
    task_id: str | None,
    category: str,
    model: str,
    retry: int,
    reason: str,
) -> None:
    if os.getenv("MINIMA_LOG_ROUTING") != "1":
        return
    label = task_id or "-"
    print(
        f"minima route_failed task_id={label} category={category} "
        f"model={model} retry={retry} reason={reason}",
        file=sys.stderr,
    )


def _safe_fallback_answer(category: str) -> str:
    if category == "sentiment":
        return "neutral"
    if category == "logic":
        return "unknown"
    if category == "math":
        return "0"
    return "Unable to determine."


@dataclass(frozen=True)
class Router:
    client: FireworksClient

    def answer(self, prompt: str, task_id: str | None = None) -> str:
        category = classify_task(prompt)
        local_answer = solve_local(category, prompt)
        if local_answer is not None:
            _log_routing(task_id, category, f"local:{category}", retry=0)
            return local_answer

        if self.client.config.placeholder_mode:
            _log_routing(task_id, category, "placeholder", retry=0)
            return self.client.answer(prompt=prompt, category=category)

        last_answer: str | None = None
        for retry, model in enumerate(
            select_model_candidates(category, self.client.config.allowed_models)
        ):
            _log_routing(task_id, category, model, retry=retry)
            try:
                answer = self.client.answer(prompt=prompt, category=category, model=model)
            except FireworksClientError:
                _log_model_failure(task_id, category, model, retry, "client_error")
                continue
            if not _looks_malformed(category, answer):
                return answer
            last_answer = answer
            _log_model_failure(task_id, category, model, retry, "malformed_answer")

        if last_answer:
            return last_answer
        _log_model_failure(task_id, category, "all", retry=0, reason="no_model_succeeded")
        return _safe_fallback_answer(category)
