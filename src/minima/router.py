"""Task routing for the minima agent."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys

from .fireworks_client import FireworksClient
from .model_selector import select_fallback_model, select_model
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


@dataclass(frozen=True)
class Router:
    client: FireworksClient

    def answer(self, prompt: str, task_id: str | None = None) -> str:
        category = classify_task(prompt)
        if self.client.config.placeholder_mode:
            _log_routing(task_id, category, "placeholder", retry=0)
            return self.client.answer(prompt=prompt, category=category)

        model = select_model(category, self.client.config.allowed_models)
        _log_routing(task_id, category, model, retry=0)
        answer = self.client.answer(prompt=prompt, category=category, model=model)
        if not _looks_malformed(category, answer):
            return answer

        fallback = select_fallback_model(
            category,
            self.client.config.allowed_models,
            current_model=model,
        )
        if fallback is None:
            return answer
        _log_routing(task_id, category, fallback, retry=1)
        return self.client.answer(prompt=prompt, category=category, model=fallback)
